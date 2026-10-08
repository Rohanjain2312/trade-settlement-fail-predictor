"""Notebook 02 pipeline: restore, pull data, env check, train LR, SVM, XGBoost, evaluate,
publish models, deploy the Space.

Run in Colab with `python -m src.pipeline.run_train`. The mode comes from config/run.yaml.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data.generator import write_json
from src.features.definitions import feature_schema
from src.hub import deploy_space
from src.hub.publish import publish
from src.models import evaluate as E
from src.models import train_logreg, train_svm, train_xgb
from src.models.preprocess import load_trades, split_frames, xy
from src.pipeline import env_check
from src.pipeline.checkpoint import StageSpec
from src.pipeline.runner import Context, Stage, run_notebook

log = logging.getLogger(__name__)

NOTEBOOK = "02_train_explain"
TRAIN_KEYS = ("seed", "split", "train")
FAMILIES = ("logreg", "svm_linear", "xgb")


def model_names(cfg: dict, family: str) -> list[str]:
    return [f"{family}__{v}" for v in cfg["train"]["models"][family]]


def model_path(name: str) -> str:
    return f"train/models/{name}.json" if name.startswith("xgb") else f"train/models/{name}.joblib"


def load_model(wd: Path, name: str):
    if name.startswith("xgb"):
        return train_xgb.load_booster(wd / model_path(name))[0]
    return joblib.load(wd / model_path(name))


def score(model, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return train_xgb.predict(model, X)


# ---- preparation (always run) ----------------------------------------------------------------

def stage_restore(ctx: Context) -> list[str]:
    ctx.ckpt.restore(["train/*"])
    return []


def stage_pull_data(ctx: Context) -> list[str]:
    store = ctx.stores["dataset"]
    data_dir = ctx.workdir / "dataset"
    store.download_all(data_dir, ["data/*", "truth/*", "samples/*", "reports/*"])
    ctx.ckpt.dataset_revision = store.revision()
    splits = split_frames(load_trades(data_dir), ctx.cfg)
    ctx.state["splits"] = splits
    ctx.state["test_sample"] = pd.read_parquet(data_dir / "samples" / "test_sample.parquet")
    ctx.report.set("dataset_revision", ctx.ckpt.dataset_revision)
    ctx.state["info:pull_data"] = {
        k: {"rows": len(v), "failed": int(v["failed"].sum())} for k, v in splits.items()
    }
    return []


def stage_env_check(ctx: Context) -> list[str]:
    info = env_check.check(require_gpu=ctx.cfg["train"]["require_gpu"])
    ctx.device = info["device"]
    ctx.state["cuml"] = info["cuml"]
    ctx.report.set("versions", info["versions"])
    ctx.state["info:env_check"] = {"gpu": info["gpu"], "device": info["device"], "cuml": info["cuml"]}
    return []


# ---- training --------------------------------------------------------------------------------

def stage_train_logreg(ctx: Context) -> list[str]:
    cfg, wd = ctx.cfg, ctx.workdir
    X, y = xy(ctx.state["splits"]["train"])
    out = []
    for name in model_names(cfg, "logreg"):
        path = model_path(name)
        out.append(path)
        if ctx.ckpt.unit_done("train_logreg", name):
            continue
        variant = name.split("__")[1]
        model = train_logreg.fit(X, y, cfg["train"]["logreg"]["C"], variant, cfg["seed"])
        (wd / path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, wd / path)
        ctx.ckpt.save_unit("train_logreg", name, [path])
        log.info("trained %s", name)
    return out


def stage_train_svm(ctx: Context) -> list[str]:
    cfg, wd = ctx.cfg, ctx.workdir
    X, y = xy(ctx.state["splits"]["train"])
    Xv, yv = xy(ctx.state["splits"]["val"])
    out = []
    for name in model_names(cfg, "svm_linear"):
        path = model_path(name)
        out.append(path)
        if ctx.ckpt.unit_done("train_svm", name):
            continue
        variant = name.split("__")[1]
        svm = train_svm.make_linear(cfg["train"]["svm_linear"]["C"], variant, cfg["seed"]).fit(X, y)
        model = train_svm.calibrate(svm, Xv, yv)
        (wd / path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, wd / path)
        ctx.ckpt.save_unit("train_svm", name, [path])
        log.info("trained %s", name)
    return out


def stage_train_xgb(ctx: Context) -> list[str]:
    cfg, wd = ctx.cfg, ctx.workdir
    xcfg = cfg["train"]["xgb"]
    X, y = xy(ctx.state["splits"]["train"])
    Xv, yv = xy(ctx.state["splits"]["val"])
    dval = train_xgb.dmatrix(Xv, yv)
    out = []
    for name in model_names(cfg, "xgb"):
        path = model_path(name)
        out += [path, path + ".state.json"]
        if ctx.ckpt.unit_done("train_xgb", name):
            continue
        variant = name.split("__")[1]
        Xt, yt = (train_xgb.smote_resample(X, y, cfg["seed"]) if variant == "smote" else (X, y))
        spw = float((yt == 0).sum() / max(1, (yt == 1).sum())) if variant == "scale_pos_weight" else 1.0
        p = train_xgb.params(xcfg, ctx.device, cfg["seed"], spw)
        dtrain = train_xgb.dmatrix(Xt, yt)
        progress = f"train/xgb/{name}.progress.json"
        progress_files = [progress, progress + ".state.json"]

        def save_progress(bst, state, progress=progress, progress_files=progress_files, name=name):
            train_xgb.save_booster(bst, wd / progress, state)
            ctx.ckpt.save_unit("train_xgb", f"{name}:progress", progress_files)

        resume = None
        if ctx.ckpt.unit_done("train_xgb", f"{name}:progress"):
            resume = train_xgb.load_booster(wd / progress, p, [dtrain, dval])
        bst, state = train_xgb.train(p, dtrain, dval, xcfg["max_rounds"], xcfg["patience"],
                                     xcfg["checkpoint_every"], save_progress, resume)
        train_xgb.save_booster(bst, wd / path, state)
        ctx.ckpt.save_unit("train_xgb", name, [path, path + ".state.json"], delete=progress_files)
        log.info("trained %s: best iteration %d, val aucpr %.4f", name, state["best_iteration"], state["best_score"])
    return out


# ---- evaluation ------------------------------------------------------------------------------

def all_models(cfg: dict) -> list[str]:
    return [n for fam in FAMILIES for n in model_names(cfg, fam)]


def stage_evaluate(ctx: Context) -> list[str]:
    cfg, wd = ctx.cfg, ctx.workdir
    splits = ctx.state["splits"]
    Xv, yv = xy(splits["val"])
    Xt, yt = xy(splits["test"])
    Xs, _ = xy(ctx.state["test_sample"])
    capacity = cfg["train"]["ops_capacity_share"]
    out, metrics, scores = [], {}, {"trade_id": ctx.state["test_sample"]["trade_id"].to_numpy()}
    for name in all_models(cfg):
        path = f"train/metrics/{name}.json"
        out.append(path)
        model = load_model(wd, name)
        scores[name] = score(model, Xs).astype(np.float32)
        if ctx.ckpt.unit_done("evaluate", name):
            metrics[name] = json.loads((wd / path).read_text())
            continue
        pv = score(model, Xv)
        thr = E.threshold_for_capacity(pv, capacity)
        family, resampling = name.split("__")
        metrics[name] = {
            "model": name, "family": family, "resampling": resampling,
            "val": E.metrics(yv, pv, thr),
            "test": E.metrics(yt, score(model, Xt), thr, with_curves=True),
        }
        write_json(metrics[name], wd / path)
        ctx.ckpt.save_unit("evaluate", name, [path])
        log.info("%s: test PR-AUC %.4f, recall in top 2%% %.3f", name, metrics[name]["test"]["pr_auc"],
                 metrics[name]["test"]["recall_top_2pct"])
    summary = {
        "synthetic_data_note": "All metrics are on synthetic data generated by this project.",
        "ops_capacity_share": capacity,
        "dataset_revision": ctx.ckpt.dataset_revision,
        "models": metrics,
    }
    write_json(summary, wd / "train/metrics.json")
    (wd / "train/scores").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(scores).to_parquet(wd / "train/scores/test_sample_scores.parquet", index=False)
    ctx.report.add_metrics({n: {k: m["test"].get(k) for k in ("pr_auc", "recall_top_2pct", "brier")}
                            for n, m in metrics.items()})
    return out + ["train/metrics.json", "train/scores/test_sample_scores.parquet"]


# ---- publishing ------------------------------------------------------------------------------

def model_card(cfg: dict, wd: Path) -> str:
    m = json.loads((wd / "train/metrics.json").read_text())["models"]
    gh, ds = cfg["project"]["github_repo"], cfg["project"]["dataset_repo"]
    rows = ["| Model | Resampling | PR-AUC | Recall in top 2% | Recall at precision 0.5 | Brier |",
            "|---|---|---|---|---|---|"]
    for name, r in m.items():
        t = r["test"]
        rows.append(f"| {r['family']} | {r['resampling']} | {t['pr_auc']:.3f} | {t['recall_top_2pct']:.3f} | "
                    f"{t['recall_at_precision_50']:.3f} | {t['brier']:.4f} |")
    return "\n".join([
        "---",
        "license: mit",
        "tags: [tabular-classification, synthetic-data, xgboost, svm, shap, imbalanced-learn]",
        f"datasets: [{ds}]",
        "---",
        "",
        "# Trade Settlement Fail Predictor",
        "",
        "> **Trained on synthetic data only.** The data comes from a seeded generator in "
        f"[github.com/{gh}](https://github.com/{gh}) (dataset: [{ds}](https://huggingface.co/datasets/{ds})). "
        "These models and metrics demonstrate the modeling approach. They are not validated on real "
        "trades and should not be used for real settlement decisions.",
        "",
        "## Models",
        "",
        "- `logreg.joblib`: L2 logistic regression baseline (scikit-learn pipeline with preprocessing)",
        "- `svm_linear_calibrated.joblib`: linear SVM with SMOTENC inside the training pipeline, "
        "calibrated on validation data",
        "- `xgb_model.json`: XGBoost with native categorical and missing-value handling",
        "- `models/`: every trained variant; `feature_schema.json`: the 20 input features in order",
        "",
        "## Test metrics (synthetic test period)",
        "",
        *rows,
        "",
        "Accuracy is not reported as a headline metric: at a 3% fail rate, always predicting "
        "'settles' scores about 97%.",
        "",
        "## Reload",
        "",
        "```python",
        "import joblib, xgboost as xgb",
        "from huggingface_hub import hf_hub_download",
        f"repo = '{cfg['project']['model_repo']}'",
        "logreg = joblib.load(hf_hub_download(repo, 'logreg.joblib'))",
        "bst = xgb.Booster(); bst.load_model(hf_hub_download(repo, 'xgb_model.json'))",
        "```",
        "",
        "Inputs must be the 20 columns in `feature_schema.json`, with categoricals as pandas "
        "categories using the listed levels.",
        "",
    ])


def stage_publish_models(ctx: Context) -> list[str]:
    cfg, wd = ctx.cfg, ctx.workdir
    write_json(feature_schema(), wd / "train/feature_schema.json")
    (wd / "train/model_card.md").write_text(model_card(cfg, wd))
    lr = load_model(wd, model_names(cfg, "logreg")[0])
    from imblearn.pipeline import Pipeline

    pre = Pipeline([("encode", lr.named_steps["encode"]), ("expand", lr.named_steps["expand"])])
    joblib.dump(pre, wd / "train/preprocessor.joblib")
    mapping = {
        "README.md": "train/model_card.md",
        "feature_schema.json": "train/feature_schema.json",
        "metrics.json": "train/metrics.json",
        "preprocessor.joblib": "train/preprocessor.joblib",
        "logreg.joblib": model_path(model_names(cfg, "logreg")[0]),
        "svm_linear_calibrated.joblib": model_path(model_names(cfg, "svm_linear")[0]),
        "xgb_model.json": model_path(model_names(cfg, "xgb")[0]),
    }
    for name in all_models(cfg):
        mapping[f"models/{Path(model_path(name)).name}"] = model_path(name)
    store = ctx.stores["model"]
    publish(ctx, "publish_models", store, mapping, "publish/model", message="publish models")
    store.tag(cfg["project"]["model_tag"], "models trained on synthetic data")
    ctx.report.set("model_revision", store.revision())
    # Kept in the work repo so a later stage can run after a restart.
    return ["train/feature_schema.json", "train/model_card.md", "train/preprocessor.joblib"]


def stage_deploy_space(ctx: Context) -> list[str]:
    cfg, wd = ctx.cfg, ctx.workdir
    assets = {
        "metrics.json": "train/metrics.json",
        "feature_schema.json": "train/feature_schema.json",
    }
    files = deploy_space.build_space_dir(cfg, wd, assets)
    from src.config import repo_ids

    space_id = repo_ids(cfg)["space"]
    result = deploy_space.deploy(ctx.stores["space"], wd, files, space_id, wait=ctx.state.get("on_hub", True))
    ctx.report.set("space_url", result["url"])
    ctx.state["info:deploy_space"] = result
    return []


def build_stages() -> list[Stage]:
    model_src = ("src/models/preprocess.py", "src/features/definitions.py")
    return [
        Stage(StageSpec("restore", (), ()), stage_restore, always_run=True),
        Stage(StageSpec("pull_data", (), ()), stage_pull_data, always_run=True),
        Stage(StageSpec("env_check", (), ()), stage_env_check, always_run=True),
        Stage(StageSpec("train_logreg", (*model_src, "src/models/train_logreg.py"), TRAIN_KEYS, True),
              stage_train_logreg),
        Stage(StageSpec("train_svm", (*model_src, "src/models/train_svm.py"), TRAIN_KEYS, True), stage_train_svm),
        Stage(StageSpec("train_xgb", (*model_src, "src/models/train_xgb.py"), TRAIN_KEYS, True), stage_train_xgb),
        Stage(StageSpec("evaluate", ("src/models/evaluate.py",), TRAIN_KEYS, True), stage_evaluate),
        Stage(StageSpec("publish_models", ("src/hub/publish.py", "src/pipeline/run_train.py"),
                        ("project", *TRAIN_KEYS), True), stage_publish_models),
        Stage(StageSpec("deploy_space", ("app/*", "src/hub/deploy_space.py"), ("project",), True),
              stage_deploy_space),
    ]


def main() -> None:
    run_notebook(NOTEBOOK, build_stages)


if __name__ == "__main__":
    main()
