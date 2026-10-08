"""Scenario coverage, measured instead of assumed.

1. Coverage matrix: rows are scenarios (and root-cause combinations) and feature bins
   (levels for categoricals, quantile bins from the training split for numerics, plus a
   missing bin). Columns are train, validation, and test, each split by label. Every cell
   must reach its minimum from config.
2. Pairwise coverage: every pair of levels across the key categorical features appears at
   least a minimum number of times, unless config lists the pair as allowed to be empty.
3. Time coverage: each split contains its expected regimes, the test period contains a
   stress regime, and some test trades come from counterparties never seen in training.
4. Realism: the fail rate and the root-cause mix sit inside rough public ranges.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from src.data.generator import FAIL_REASONS
from src.data.scenarios import REGIMES, SCENARIO_IDS, combination_members, decode, scenario_names
from src.features.definitions import BY_NAME, FEATURES

SPLITS = ("train", "val", "test")
COMBO_NAMES = {
    "S1+S6": "SSI mismatch on a cross-border trade with a holiday or cutoff issue",
    "S3+S5": "Shortfall inside a settlement chain",
    "S4+overnight": "Late confirmation on an overnight instruction",
}


def _minimum(cov: dict, row_id: str, split: str, label: str) -> int:
    base = cov["min_train"] if split == "train" else cov["min_eval"]
    return int(cov.get("overrides", {}).get(row_id, {}).get(label, base[label]))


def _members(df: pd.DataFrame) -> dict[str, np.ndarray]:
    m = decode(df["scenario_mask"].to_numpy())
    bucket = pd.Categorical(df["instruction_hour_bucket"], categories=list(BY_NAME["instruction_hour_bucket"].levels)).codes
    m.update(combination_members(df["scenario_mask"].to_numpy(), bucket))
    return m


def feature_bins(train: pd.DataFrame, cov: dict) -> dict[str, list[tuple[str, object]]]:
    """For each feature: (bin label, selector) pairs. A selector maps a frame to a mask."""
    out = {}
    for f in FEATURES:
        bins = []
        if f.kind == "categorical":
            for lv in f.levels:
                bins.append((lv, lambda d, n=f.name, lv=lv: (d[n] == lv).to_numpy()))
        elif f.kind == "binary":
            for v in (0, 1):
                bins.append((str(v), lambda d, n=f.name, v=v: (d[n] == v).to_numpy()))
        else:
            q = np.linspace(0, 1, cov["numeric_bins"] + 1)
            edges = np.unique(np.nanquantile(train[f.name].to_numpy(dtype=float), q))
            edges[0], edges[-1] = -np.inf, np.inf
            for lo, hi in zip(edges[:-1], edges[1:]):
                label = f"({lo:.4g}, {hi:.4g}]".replace("(-inf", "(min").replace("inf]", "max]")
                bins.append((label, lambda d, n=f.name, lo=lo, hi=hi: ((d[n] > lo) & (d[n] <= hi)).to_numpy()))
            if f.may_be_missing:
                bins.append(("missing", lambda d, n=f.name: d[n].isna().to_numpy()))
        out[f.name] = bins
    return out


def coverage_matrix(splits: dict[str, pd.DataFrame], cfg: dict) -> list[dict]:
    cov = cfg["coverage"]
    names = {**scenario_names(cfg), **COMBO_NAMES}
    members = {s: _members(splits[s]) for s in SPLITS}
    labels = {s: splits[s]["failed"].to_numpy() == 1 for s in SPLITS}
    rows = []

    def add(kind, row_id, name, masks, eval_required=True):
        row = {"kind": kind, "id": row_id, "name": name, "cells": {}, "passed": True}
        for s in SPLITS:
            m, y = masks[s], labels[s]
            required = not (row_id in REGIMES and row_id not in cov["expected_regimes"][s])
            required &= eval_required or s == "train"
            for label, count in (("failed", int((m & y).sum())), ("settled", int((m & ~y).sum()))):
                need = _minimum(cov, row_id, s, label) if required else 0
                ok = count >= need
                row["cells"][f"{s}_{label}"] = {"count": count, "min": need, "ok": ok}
                row["passed"] &= ok
        rows.append(row)

    for sid in (*SCENARIO_IDS, *COMBO_NAMES):
        add("scenario", sid, names[sid], {s: members[s][sid] for s in SPLITS})
    exempt = cov.get("eval_exempt_features", {})
    for feature, bins in feature_bins(splits["train"], cov).items():
        for label, select in bins:
            add("feature_bin", f"{feature}={label}", BY_NAME[feature].label,
                {s: select(splits[s]) for s in SPLITS}, eval_required=feature not in exempt)
    return rows


def pairwise_coverage(df: pd.DataFrame, cfg: dict) -> dict:
    pw = cfg["coverage"]["pairwise"]
    allowed = {a["pair"]: a["reason"] for a in pw.get("allowed_empty", [])}
    failures, smallest = [], []
    for a, b in itertools.combinations(pw["features"], 2):
        la = BY_NAME[a].levels or (0, 1)
        lb = BY_NAME[b].levels or (0, 1)
        table = pd.crosstab(df[a], df[b])
        for x in la:
            for y in lb:
                count = int(table.loc[x, y]) if x in table.index and y in table.columns else 0
                key = f"{a}={x}|{b}={y}"
                smallest.append((count, key))
                if count < pw["min_count"] and key not in allowed:
                    failures.append(f"{key}: {count} trades (min {pw['min_count']})")
    smallest.sort()
    return {
        "name": "pairwise_coverage",
        "passed": not failures,
        "failures": failures,
        "min_count": pw["min_count"],
        "allowed_empty": allowed,
        "smallest_pairs": [{"pair": k, "count": c} for c, k in smallest[:10]],
        "n_pairs": len(smallest),
    }


def time_coverage(splits: dict[str, pd.DataFrame], cfg: dict) -> dict:
    cov = cfg["coverage"]
    failures, regimes = [], {}
    for s in SPLITS:
        m = decode(splits[s]["scenario_mask"].to_numpy())
        regimes[s] = {r: int(m[r].sum()) for r in REGIMES}
        for r in cov["expected_regimes"][s]:
            if regimes[s][r] < cov["min_regime_trades"]:
                failures.append(f"{s} has {regimes[s][r]} trades in regime {r} (min {cov['min_regime_trades']})")
    stress = {r: regimes["test"][r] for r in cov["stress_regimes"]}
    if not any(n >= cov["min_regime_trades"] for n in stress.values()):
        failures.append(f"test period has no stress regime {cov['stress_regimes']}")
    seen = set(splits["train"]["cpty_id"].unique())
    cold = float((~splits["test"]["cpty_id"].isin(seen)).mean())
    if cold < cov["min_cold_start_test_share"]:
        failures.append(f"only {cold:.2%} of test trades come from counterparties unseen in training")
    months = {s: sorted(pd.to_datetime(splits[s]["trade_date"]).dt.strftime("%Y-%m").unique()) for s in SPLITS}
    return {
        "name": "time_coverage",
        "passed": not failures,
        "failures": failures,
        "regime_trades": regimes,
        "test_stress_trades": stress,
        "cold_start_test_share": round(cold, 5),
        "months": {s: [v[0], v[-1]] if v else [] for s, v in months.items()},
    }


def realism(df: pd.DataFrame, cfg: dict) -> dict:
    rc = cfg["coverage"]["realism"]
    failures = []
    rate = float(df["failed"].mean())
    if not rc["fail_rate"][0] <= rate <= rc["fail_rate"][1]:
        failures.append(f"fail rate {rate:.2%} outside {rc['fail_rate']}")
    share = df.loc[df["failed"] == 1, "fail_reason"].value_counts(normalize=True)
    mix = {r: round(float(share.get(r, 0.0)), 4) for r in FAIL_REASONS}
    for r, (lo, hi) in rc["reason_share"].items():
        if not lo <= mix[r] <= hi:
            failures.append(f"{r} is {mix[r]:.1%} of fails, target range {lo:.0%} to {hi:.0%}")
    return {"name": "realism", "passed": not failures, "failures": failures, "fail_rate": round(rate, 5),
            "reason_mix": mix, "targets": rc}


def run_coverage(df: pd.DataFrame, splits: dict[str, pd.DataFrame], cfg: dict) -> dict:
    rows = coverage_matrix(splits, cfg)
    parts = [pairwise_coverage(df, cfg), time_coverage(splits, cfg), realism(df, cfg)]
    failed_rows = [r for r in rows if not r["passed"]]
    failures = [f"coverage {r['id']}: " + ", ".join(
        f"{k} {c['count']} < {c['min']}" for k, c in r["cells"].items() if not c["ok"]) for r in failed_rows]
    for p in parts:
        failures += [f"{p['name']}: {f}" for f in p["failures"]]
    return {
        "name": "scenario_coverage",
        "passed": not failures,
        "failures": failures,
        "matrix": rows,
        **{p["name"]: p for p in parts},
    }


def report_markdown(result: dict, cfg: dict) -> str:
    cov = cfg["coverage"]
    status = "PASS" if result["passed"] else "FAIL"
    lines = [
        "# Scenario coverage report",
        "",
        "> All data is synthetic. Coverage means coverage of the scenarios defined in "
        "`config/scenarios.yaml`; real production data always contains cases nobody planned for.",
        "",
        f"**Result: {status}** ({cfg['mode']} run, seed {cfg['seed']})",
        "",
        f"Minimum trades per cell: train {cov['min_train']}, validation and test {cov['min_eval']}. "
        "Regimes not expected in a split have no minimum there (validation sits between the two "
        "volatility spikes).",
        "",
    ]
    if result["failures"]:
        lines += ["## Failures", "", *[f"- {f}" for f in result["failures"]], ""]

    def table(kind):
        out = ["| Row | Description | " + " | ".join(f"{s} failed | {s} settled" for s in SPLITS) + " |",
               "|---|---|" + "---|" * (2 * len(SPLITS))]
        for r in result["matrix"]:
            if r["kind"] != kind:
                continue
            cells = []
            for s in SPLITS:
                for label in ("failed", "settled"):
                    c = r["cells"][f"{s}_{label}"]
                    cells.append(f"{c['count']:,}" + ("" if c["ok"] else " ❌"))
            out.append(f"| `{r['id']}` | {r['name']} | " + " | ".join(cells) + " |")
        return out

    t, p, rl = result["time_coverage"], result["pairwise_coverage"], result["realism"]
    lines += ["## Scenarios", "", *table("scenario"), "",
              "## Feature bins", "", *table("feature_bin"), "",
              "## Time coverage", "",
              "- Months: " + ", ".join(f"{s} {m[0]} to {m[1]}" for s, m in t["months"].items() if m),
              f"- Test-period stress regime trades: {t['test_stress_trades']}",
              f"- Test trades from counterparties never seen in training: {t['cold_start_test_share']:.2%}",
              "",
              "## Pairwise coverage", "",
              f"- {p['n_pairs']} level pairs across {', '.join(cfg['coverage']['pairwise']['features'])}; "
              f"minimum {p['min_count']} trades each",
              "- Smallest pairs: " + ", ".join(f"`{x['pair']}` {x['count']}" for x in p["smallest_pairs"][:5]),
              "",
              "## Realism calibration", "",
              f"- Fail rate {rl['fail_rate']:.2%} (target {rl['targets']['fail_rate'][0]:.0%} to "
              f"{rl['targets']['fail_rate'][1]:.1%}; public figures put euro-area fails near 2 to 3%)",
              *[f"- {r}: {v:.1%} of fails (target {rl['targets']['reason_share'][r][0]:.0%} to "
                f"{rl['targets']['reason_share'][r][1]:.0%})" for r, v in rl["reason_mix"].items()],
              ""]
    return "\n".join(lines)
