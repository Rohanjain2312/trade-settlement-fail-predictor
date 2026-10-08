"""Notebook 01 pipeline: restore, generate (reference data, calibration, every month),
validate, test sample, publish dataset.

Run in Colab with `python -m src.pipeline.run_generate`. The mode comes from config/run.yaml.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from src.config import config_hash
from src.data import validate as V
from src.data.coverage import report_markdown, run_coverage
from src.data.generator import (
    calibrate_intercept,
    effect_weights,
    generate_months,
    month_plan,
    write_json,
    write_parquet,
)
from src.data.reference_data import build_reference, reference_tables
from src.features.definitions import FEATURES
from src.hub.publish import publish
from src.models.preprocess import split_frames
from src.pipeline.checkpoint import StageSpec
from src.pipeline.runner import Context, Stage, run_notebook

log = logging.getLogger(__name__)

NOTEBOOK = "01_generate_data"
GEN_SOURCES = ("src/data/reference_data.py", "src/data/generator.py", "src/data/probability.py",
               "src/data/scenarios.py", "src/features/definitions.py")
GEN_KEYS = ("seed", "data", "reference", "regimes", "effects", "interactions", "hidden", "scenarios")


class ValidationFailed(RuntimeError):
    pass


def load_generated(workdir: Path) -> pd.DataFrame:
    files = sorted((workdir / "gen" / "trades").glob("trades_*.parquet"))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


# ---- stages --------------------------------------------------------------------------------

def stage_restore(ctx: Context) -> list[str]:
    ctx.ckpt.restore(["gen/*"])
    return []


def stage_generate(ctx: Context) -> list[str]:
    cfg, wd = ctx.cfg, ctx.workdir
    ref = build_reference(cfg)
    tables = reference_tables(ref)
    ref_files = [f"gen/{name}.parquet" for name in tables]
    if not ctx.ckpt.unit_done("generate", "reference"):
        for name, df in tables.items():
            write_parquet(df, wd / f"gen/{name}.parquet")
        ctx.ckpt.save_unit("generate", "reference", ref_files)

    cal_files = ["gen/calibration.json", "gen/truth/effect_weights.json"]
    if ctx.ckpt.unit_done("generate", "calibration"):
        calibration = json.loads((wd / "gen/calibration.json").read_text())
    else:
        calibration = calibrate_intercept(ref, cfg)
        write_json(calibration, wd / "gen/calibration.json")
        write_json(effect_weights(cfg, calibration["intercept"]), wd / "gen/truth/effect_weights.json")
        ctx.ckpt.save_unit("generate", "calibration", cal_files)
    log.info("intercept %.4f", calibration["intercept"])

    def progress(**info):
        ctx.report.progress("generate", **info)

    months = generate_months(ref, cfg, calibration["intercept"], wd, ctx.ckpt, "generate", progress)
    ctx.state["info:generate"] = {"months": len(month_plan(ref)), "intercept": calibration["intercept"]}
    return ref_files + cal_files + months


def _run_check(ctx: Context, name: str, fn, results: dict, files: list) -> dict:
    """One check is one checkpoint unit with its own result file."""
    path = f"gen/validation/{name}.json"
    files.append(path)
    if ctx.ckpt.unit_done("validate", name):
        results[name] = json.loads((ctx.workdir / path).read_text())
    else:
        results[name] = fn()
        write_json(results[name], ctx.workdir / path)
        ctx.ckpt.save_unit("validate", name, [path])
    log.info("check %s: %s", name, "PASS" if results[name]["passed"] else "FAIL")
    return results[name]


def stage_validate(ctx: Context) -> list[str]:
    cfg, wd = ctx.cfg, ctx.workdir
    df = load_generated(wd)
    results, files = {}, []
    imp = _run_check(ctx, "true_importance",
                     lambda: {"passed": True, "importance": V.true_importance(df, cfg)}, results, files)["importance"]
    _run_check(ctx, "fail_rates", lambda: V.check_rates(df, cfg), results, files)
    _run_check(ctx, "feature_distinctness", lambda: V.check_distinctness(df, cfg), results, files)
    _run_check(ctx, "rolling_features_no_future", lambda: V.recompute_rolling(df, cfg), results, files)
    _run_check(ctx, "strength_tiers", lambda: V.check_strength_tiers(imp), results, files)
    _run_check(ctx, "lr_recovers_planted_effects", lambda: V.check_lr_recovery(df, cfg, imp), results, files)

    summary = {
        "passed": all(r["passed"] for r in results.values()),
        "checks": {k: bool(r["passed"]) for k, r in results.items()},
        "fail_rate": results["fail_rates"]["overall"],
        "n_trades": results["fail_rates"]["n_trades"],
        "reason_mix": V.reason_mix(df),
        "scenarios": V.scenario_summary(df),
        "true_importance": imp,
    }
    write_json(summary, wd / "gen/validation/summary.json")
    write_json(imp, wd / "gen/truth/true_importance.json")
    files += ["gen/validation/summary.json", "gen/truth/true_importance.json"]
    ctx.report.add_metrics({"data": {k: summary[k] for k in ("passed", "checks", "fail_rate", "n_trades", "reason_mix")}})
    failed = [k for k, ok in summary["checks"].items() if not ok]
    if failed:
        raise ValidationFailed(f"data checks failed: {failed}. Details in gen/validation/*.json")
    ctx.state["info:validate"] = summary["checks"]
    return files


class CoverageFailed(RuntimeError):
    pass


def stage_coverage(ctx: Context) -> list[str]:
    cfg, wd = ctx.cfg, ctx.workdir
    df = load_generated(wd)
    result = run_coverage(df, split_frames(df, cfg), cfg)
    write_json(result, wd / "gen/validation/coverage.json")
    (wd / "gen/reports").mkdir(parents=True, exist_ok=True)
    (wd / "gen/reports/coverage_report.md").write_text(report_markdown(result, cfg))
    ctx.report.add_metrics({"coverage": {
        "passed": result["passed"],
        "failures": result["failures"][:25],
        "reason_mix": result["realism"]["reason_mix"],
        "cold_start_test_share": result["time_coverage"]["cold_start_test_share"],
        "smallest_pairs": result["pairwise_coverage"]["smallest_pairs"][:5],
    }})
    if not result["passed"]:
        raise CoverageFailed(f"{len(result['failures'])} coverage failures, first: {result['failures'][:3]}")
    ctx.state["info:coverage"] = {"rows": len(result["matrix"]), "passed": True}
    return ["gen/validation/coverage.json", "gen/reports/coverage_report.md"]


def stage_sample(ctx: Context) -> list[str]:
    cfg, wd = ctx.cfg, ctx.workdir
    test = split_frames(load_generated(wd), cfg)["test"]
    n = min(cfg["data"]["test_sample_rows"], len(test))
    sample = test.sample(n=n, random_state=cfg["seed"]).sort_values("trade_id").reset_index(drop=True)
    write_parquet(sample, wd / "gen/samples/test_sample.parquet")
    ctx.state["info:sample"] = {"rows": n, "failed": int(sample["failed"].sum())}
    return ["gen/samples/test_sample.parquet"]


def dataset_card(cfg: dict, wd: Path) -> str:
    summary = json.loads((wd / "gen/validation/summary.json").read_text())
    gh = cfg["project"]["github_repo"]
    lines = [
        "---",
        "license: mit",
        "pretty_name: Synthetic Trade Settlement Fails",
        "tags: [synthetic, finance, tabular-classification, imbalanced-classification]",
        "size_categories: [1M<n<10M]" if summary["n_trades"] > 1_000_000 else "size_categories: [10K<n<100K]",
        "configs:",
        "- config_name: trades",
        "  data_files: data/trades/*.parquet",
        "- config_name: test_sample",
        "  data_files: samples/test_sample.parquet",
        "---",
        "",
        "# Synthetic Trade Settlement Fails",
        "",
        "> **All data in this dataset is synthetic.** It was generated by a seeded numpy generator in "
        f"[github.com/{gh}](https://github.com/{gh}). No real trades, counterparties, or securities are "
        "included. Counterparty and security names are fictional.",
        "",
        f"- Trades: {summary['n_trades']:,} over {cfg['data']['n_days']} business days, "
        f"{summary['fail_rate']:.2%} failed",
        f"- Seed: {cfg['seed']}, generator config hash: `{config_hash(cfg, GEN_KEYS)}`",
        "- Splits by trade month: "
        f"train {cfg['split']['train_months']}, validation {cfg['split']['val_months']}, "
        f"test {cfg['split']['test_months']}, with a gap of {cfg['split']['gap_days']} business days",
        f"- All data checks passed: {summary['passed']}",
        "",
        "## Files",
        "",
        "| Path | Contents |",
        "|---|---|",
        "| `data/trades/trades_YYYY-MM.parquet` | One file per month of trades |",
        "| `data/reference/*.parquet` | Fictional counterparties, securities, and the trading calendar |",
        "| `truth/*` | Hidden truth used only for validation (latent risk, planted effects). Never model inputs |",
        "| `samples/test_sample.parquet` | A sample of test-period trades used by the demo app |",
        "| `reports/` | Validation and coverage reports |",
        "",
        "## Columns",
        "",
        "IDs and dates: `trade_id`, `trade_date`, `settle_date`, `desk_id`, `cpty_id`, `sec_id`, `notional`.",
        "Targets: `failed` (label), `fail_reason`, `scenario_mask` (bitmask of scenarios, for reports only).",
        "None of these are model inputs. The 20 model features:",
        "",
        "| # | Feature | Type | What it is |",
        "|---|---|---|---|",
    ]
    for i, f in enumerate(FEATURES, 1):
        lines.append(f"| {i} | `{f.name}` | {f.kind} | {f.description} |")
    lines += [
        "",
        "## Fail reasons (share of failed trades)",
        "",
        *[f"- {k}: {v:.1%}" for k, v in summary["reason_mix"].items()],
        "",
        "## Limits",
        "",
        "The generator covers a defined catalog of fail scenarios. Out of scope: partial settlements, "
        "buy-in regimes, penalty mechanics under specific regulations, and market-wide outages. Real "
        "production data always contains cases nobody planned for.",
        "",
        "Query with DuckDB:",
        "",
        "```sql",
        "SELECT fail_reason, count(*) FROM "
        f"'hf://datasets/{cfg['project']['dataset_repo']}/data/trades/*.parquet' WHERE failed = 1 GROUP BY 1;",
        "```",
        "",
    ]
    return "\n".join(lines)


def stage_publish(ctx: Context) -> list[str]:
    cfg, wd = ctx.cfg, ctx.workdir
    (wd / "gen/dataset_card.md").write_text(dataset_card(cfg, wd))
    mapping = {"README.md": "gen/dataset_card.md",
               "reports/validation_summary.json": "gen/validation/summary.json",
               "samples/test_sample.parquet": "gen/samples/test_sample.parquet"}
    for p in sorted((wd / "gen/reference").glob("*.parquet")):
        mapping[f"data/reference/{p.name}"] = f"gen/reference/{p.name}"
    for p in sorted((wd / "gen/truth").glob("*")):
        mapping[f"truth/{p.name}"] = f"gen/truth/{p.name}"
    for p in sorted((wd / "gen/trades").glob("*.parquet")):
        mapping[f"data/trades/{p.name}"] = f"gen/trades/{p.name}"
    for p in sorted((wd / "gen/reports").glob("*")) if (wd / "gen/reports").exists() else []:
        mapping[f"reports/{p.name}"] = f"gen/reports/{p.name}"
    store = ctx.stores["dataset"]
    publish(ctx, "publish_dataset", store, mapping, "publish/dataset", message="publish synthetic dataset")
    store.tag(cfg["project"]["dataset_tag"], "synthetic dataset")
    revision = store.revision()
    write_json({"revision": revision, "tag": cfg["project"]["dataset_tag"], "files": len(mapping)},
               wd / "gen/published.json")
    ctx.report.set("dataset_revision", revision)
    ctx.state["info:publish_dataset"] = {"files": len(mapping), "revision": revision}
    return ["gen/published.json"]


def build_stages() -> list[Stage]:
    return [
        Stage(StageSpec("restore", (), ()), stage_restore, always_run=True),
        Stage(StageSpec("generate", GEN_SOURCES, GEN_KEYS), stage_generate),
        Stage(StageSpec("validate", ("src/data/validate.py", "src/pipeline/run_generate.py"),
                        ("seed", "data", "effects", "interactions", "split")), stage_validate),
        Stage(StageSpec("coverage", ("src/data/coverage.py", "src/data/scenarios.py"),
                        ("seed", "split", "coverage", "scenarios")), stage_coverage),
        Stage(StageSpec("sample", ("src/models/preprocess.py",), ("seed", "data", "split")), stage_sample),
        Stage(StageSpec("publish_dataset", ("src/hub/publish.py", "src/pipeline/run_generate.py"),
                        ("project", "data", "split")), stage_publish),
    ]


def main() -> None:
    run_notebook(NOTEBOOK, build_stages)


if __name__ == "__main__":
    main()
