"""Notebook 02 pipeline: restore, pull data, env check, tune, train (LR, linear SVM, RBF SVM,
XGBoost, each three ways), calibrate, evaluate (overall and per scenario), held-out scenario,
SHAP, SHAP versus truth, publish models, deploy the Space.

Run in Colab with `python -m src.pipeline.run_train`. The mode comes from config/run.yaml.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss

from src.data.coverage import COMBO_NAMES
from src.data.generator import write_json
from src.data.scenarios import SCENARIO_IDS, combination_members, decode, scenario_names
from src.explain import shap_views as SV
from src.features.definitions import BY_NAME, FEATURE_NAMES, feature_schema
from src.hub import deploy_space
from src.hub.publish import publish
from src.models import calibrate as C
from src.models import evaluate as E
from src.models import train_logreg, train_svm, train_xgb
from src.models import tune as T
from src.models.preprocess import load_trades, split_frames, stratified_index, stratified_subsample, xy
from src.pipeline import env_check
from src.pipeline.checkpoint import StageSpec
from src.pipeline.runner import Context, Stage, run_notebook

log = logging.getLogger(__name__)

NOTEBOOK = "02_train_explain"
TRAIN_KEYS = ("seed", "split", "train")
FAMILIES = ("logreg", "svm_linear", "svm_rbf", "xgb")


def model_names(cfg: dict, family: str) -> list[str]:
    return [f"{family}__{v}" for v in cfg["train"]["models"][family]]


def all_models(cfg: dict) -> list[str]:
    return [n for fam in FAMILIES for n in model_names(cfg, fam)]


def primary(cfg: dict, family: str) -> str:
    return model_names(cfg, family)[0]


def model_path(name: str) -> str:
    return f"train/models/{name}.json" if name.startswith("xgb") else f"train/models/{name}.joblib"


def calibrated_path(name: str) -> str:
    return f"train/calibrated/{name}.joblib"


def load_raw(wd: Path, name: str):
    if name.startswith("xgb"):
        return C.xgb_classifier(wd / model_path(name))
    return joblib.load(wd / model_path(name))


def tuned(wd: Path, name: str) -> dict:
    return json.loads((wd / f"train/tune/{name}.json").read_text())["params"]


def _xy(ctx: Context, split: str):
    key = f"xy:{split}"
    if key not in ctx.state:
        ctx.state[key] = xy(ctx.state["splits"][split])
    return ctx.state[key]


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
    ctx.state["info:pull_data"] = {k: {"rows": len(v), "failed": int(v["failed"].sum())} for k, v in splits.items()}
    return []


def stage_env_check(ctx: Context) -> list[str]:
    info = env_check.check(require_gpu=ctx.cfg["train"]["require_gpu"])
    ctx.device = info["device"]
    ctx.state["cuml"] = info["cuml"]
    ctx.report.set("versions", info["versions"])
    ctx.state["info:env_check"] = {"gpu": info["gpu"], "device": info["device"], "cuml": info["cuml"]}
    return []


# ---- tuning ----------------------------------------------------------------------------------

def stage_tune(ctx: Context) -> list[str]:
    cfg, wd = ctx.cfg, ctx.workdir
    tcfg = cfg["train"]["tune"]
    X, y = _xy(ctx, "train")
    dates_all = ctx.state["splits"]["train"]["trade_date"].to_numpy()
    out, info = [], {}
    for family in FAMILIES:
        idx = stratified_index(y, tcfg["rbf_rows"] if family == "svm_rbf" else tcfg["rows"], cfg["seed"])
        Xs, ys, dates = X.iloc[idx].reset_index(drop=True), y[idx], dates_all[idx]
        for name in model_names(cfg, family):
            db, res = f"train/tune/{name}.db", f"train/tune/{name}.json"
            out += [db, res]
            if ctx.ckpt.unit_done("tune", name):
                info[name] = json.loads((wd / res).read_text())["cv_pr_auc"]
                continue

            def save(db=db, name=name):
                ctx.ckpt.save_unit("tune", f"{name}:progress", [db])
                ctx.report.progress("tune", study=name)

            best = T.tune(family, name.split("__")[1], Xs, ys, dates, cfg, wd / db, tcfg["trials"][family],
                          ctx.device, ctx.state.get("cuml", False), on_save=save)
            write_json(best, wd / res)
            ctx.ckpt.save_unit("tune", name, [db, res])
            info[name] = best["cv_pr_auc"]
            log.info("tuned %s: cv PR-AUC %.4f with %s", name, best["cv_pr_auc"], best["params"])
    ctx.state["info:tune"] = info
    return out


# ---- training --------------------------------------------------------------------------------

def _save_sklearn(ctx: Context, stage: str, name: str, model) -> None:
    path = model_path(name)
    (ctx.workdir / path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, ctx.workdir / path)
    ctx.ckpt.save_unit(stage, name, [path])
    log.info("trained %s", name)


def stage_train_logreg(ctx: Context) -> list[str]:
    cfg, wd = ctx.cfg, ctx.workdir
    X, y = _xy(ctx, "train")
    for name in model_names(cfg, "logreg"):
        if not ctx.ckpt.unit_done("train_logreg", name):
            model = train_logreg.fit(X, y, tuned(wd, name)["C"], name.split("__")[1], cfg["seed"])
            _save_sklearn(ctx, "train_logreg", name, model)
    return [model_path(n) for n in model_names(cfg, "logreg")]


def stage_train_svm(ctx: Context) -> list[str]:
    cfg, wd = ctx.cfg, ctx.workdir
    X, y = _xy(ctx, "train")
    for name in model_names(cfg, "svm_linear"):
        if not ctx.ckpt.unit_done("train_svm", name):
            model = train_svm.make_linear(tuned(wd, name)["C"], name.split("__")[1], cfg["seed"]).fit(X, y)
            _save_sklearn(ctx, "train_svm", name, model)
    return [model_path(n) for n in model_names(cfg, "svm_linear")]


def stage_train_rbf(ctx: Context) -> list[str]:
    cfg, wd = ctx.cfg, ctx.workdir
    use_cuml = ctx.state.get("cuml", False)
    rows = cfg["train"]["svm_rbf"]["subsample_rows" if use_cuml else "cpu_subsample_rows"]
    X, y = stratified_subsample(*_xy(ctx, "train"), rows, cfg["seed"])
    for name in model_names(cfg, "svm_rbf"):
        if not ctx.ckpt.unit_done("train_rbf", name):
            p = tuned(wd, name)
            model = train_svm.make_rbf(p["C"], p["gamma"], name.split("__")[1], cfg["seed"], use_cuml).fit(X, y)
            _save_sklearn(ctx, "train_rbf", name, model)
    ctx.state["info:train_rbf"] = {"rows": len(y), "cuml": use_cuml}
    return [model_path(n) for n in model_names(cfg, "svm_rbf")]


def stage_train_xgb(ctx: Context) -> list[str]:
    cfg, wd = ctx.cfg, ctx.workdir
    X, y = _xy(ctx, "train")
    Xv, yv = _xy(ctx, "val")
    dval = train_xgb.dmatrix(Xv, yv)
    out = []
    for name in model_names(cfg, "xgb"):
        path = model_path(name)
        out += [path, path + ".state.json"]
        if ctx.ckpt.unit_done("train_xgb", name):
            continue
        xcfg = {**cfg["train"]["xgb"], **tuned(wd, name)}
        variant = name.split("__")[1]
        Xt, yt = train_xgb.smote_resample(X, y, cfg["seed"]) if variant == "smote" else (X, y)
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


# ---- calibration and evaluation --------------------------------------------------------------

def stage_calibrate(ctx: Context) -> list[str]:
    cfg, wd = ctx.cfg, ctx.workdir
    Xv, yv = _xy(ctx, "val")
    for name in all_models(cfg):
        if ctx.ckpt.unit_done("calibrate", name):
            continue
        cal = C.calibrate(load_raw(wd, name), Xv, yv, name.split("__")[0])
        (wd / calibrated_path(name)).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(cal, wd / calibrated_path(name))
        ctx.ckpt.save_unit("calibrate", name, [calibrated_path(name)])
    return [calibrated_path(n) for n in all_models(cfg)]


def scenario_masks(df: pd.DataFrame) -> dict[str, np.ndarray]:
    mask = df["scenario_mask"].to_numpy()
    m = decode(mask)
    levels = list(BY_NAME["instruction_hour_bucket"].levels)
    m.update(combination_members(mask, pd.Categorical(df["instruction_hour_bucket"], categories=levels).codes))
    return m


def stage_evaluate(ctx: Context) -> list[str]:
    cfg, wd = ctx.cfg, ctx.workdir
    Xv, yv = _xy(ctx, "val")
    Xt, yt = _xy(ctx, "test")
    sample = ctx.state["test_sample"]
    Xs, _ = xy(sample)
    masks = scenario_masks(ctx.state["splits"]["test"])
    names = {**scenario_names(cfg), **COMBO_NAMES}
    capacity = cfg["train"]["ops_capacity_share"]
    out, metrics = [], {}
    scores = {"trade_id": sample["trade_id"].to_numpy()}
    for name in all_models(cfg):
        path = f"train/metrics/{name}.json"
        out.append(path)
        cal = joblib.load(wd / calibrated_path(name))
        scores[name] = cal.predict_proba(Xs)[:, 1].astype(np.float32)
        if ctx.ckpt.unit_done("evaluate", name):
            metrics[name] = json.loads((wd / path).read_text())
            continue
        raw = load_raw(wd, name)
        pv, pt = cal.predict_proba(Xv)[:, 1], cal.predict_proba(Xt)[:, 1]
        thr = E.threshold_for_capacity(pv, capacity)
        raw_t = np.asarray(C.raw_score(raw, Xt), dtype=float)
        family, resampling = name.split("__")
        m = {
            "model": name, "family": family, "resampling": resampling,
            "val": E.metrics(yv, pv, thr),
            "test": E.metrics(yt, pt, thr, with_curves=True),
            "raw_test": {"pr_auc": float(average_precision_score(yt, raw_t))},
            "per_scenario": {sid: {"name": names[sid], **E.metrics(yt[mk], pt[mk], thr)}
                             for sid, mk in masks.items() if sid in names},
        }
        if family in ("logreg", "xgb"):  # their raw score is a probability, so its calibration can be compared
            m["raw_test"]["brier"] = float(brier_score_loss(yt, raw_t))
            m["raw_test"]["calibration_curve"] = E.calibration_curve(yt, raw_t)
        else:
            m["raw_test"]["score_quantiles"] = [float(q) for q in np.quantile(raw_t, [0.01, 0.5, 0.99])]
        if family == "svm_rbf":
            m["support_vectors"] = train_svm.support_vector_count(raw)
        metrics[name] = m
        write_json(m, wd / path)
        ctx.ckpt.save_unit("evaluate", name, [path])
        log.info("%s: test PR-AUC %.4f, recall in top 2%% %.3f, Brier %.4f", name, m["test"]["pr_auc"],
                 m["test"]["recall_top_2pct"], m["test"]["brier"])
    summary = {
        "synthetic_data_note": "All metrics are on synthetic data generated by this project.",
        "ops_capacity_share": capacity,
        "dataset_revision": ctx.ckpt.dataset_revision,
        "scenario_ids": list(SCENARIO_IDS) + list(COMBO_NAMES),
        "models": metrics,
    }
    write_json(summary, wd / "train/metrics.json")
    (wd / "train/scores").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(scores).to_parquet(wd / "train/scores/test_sample_scores.parquet", index=False)
    ctx.report.add_metrics({n: {k: m["test"].get(k) for k in ("pr_auc", "recall_top_2pct", "brier")}
                            for n, m in metrics.items()})
    return out + ["train/metrics.json", "train/scores/test_sample_scores.parquet"]


# ---- held-out scenario -----------------------------------------------------------------------

def stage_heldout(ctx: Context) -> list[str]:
    """Train XGBoost without one scenario, test on that scenario, report the drop."""
    cfg, wd = ctx.cfg, ctx.workdir
    name = primary(cfg, "xgb")
    variant = name.split("__")[1]
    rounds = json.loads((wd / (model_path(name) + ".state.json")).read_text())["best_iteration"] + 1
    xcfg = {**cfg["train"]["xgb"], **tuned(wd, name)}
    X, y = _xy(ctx, "train")
    Xv, yv = _xy(ctx, "val")
    Xt, yt = _xy(ctx, "test")
    full, _ = train_xgb.load_booster(wd / model_path(name))
    p_full = train_xgb.predict(full, Xt)
    train_masks = decode(ctx.state["splits"]["train"]["scenario_mask"].to_numpy())
    test_masks = decode(ctx.state["splits"]["test"]["scenario_mask"].to_numpy())
    capacity = cfg["train"]["ops_capacity_share"]
    out = []
    for sid in cfg["train"]["heldout_scenarios"]:
        path = f"train/heldout/{sid}.json"
        out.append(path)
        if ctx.ckpt.unit_done("heldout", sid):
            continue
        keep = ~train_masks[sid]
        Xk, yk = X[keep].reset_index(drop=True), y[keep]
        if variant == "smote":
            Xk, yk = train_xgb.smote_resample(Xk, yk, cfg["seed"])
        spw = float((yk == 0).sum() / max(1, yk.sum())) if variant == "scale_pos_weight" else 1.0
        p = train_xgb.params(xcfg, ctx.device, cfg["seed"], spw)
        bst, _ = train_xgb.train(p, train_xgb.dmatrix(Xk, yk), train_xgb.dmatrix(Xv, yv), rounds,
                                 patience=10**9, every=10**9)
        p_held = train_xgb.predict(bst, Xt)
        m = test_masks[sid]

        def summarize(pred, m=m):
            thr = float(np.quantile(pred, 1 - capacity))
            fails = m & (yt == 1)
            res = {"recall_at_capacity": float((pred[fails] >= thr).mean()) if fails.any() else None}
            if 0 < yt[m].sum() < m.sum():
                res["pr_auc"] = float(average_precision_score(yt[m], pred[m]))
            return res

        r = {"scenario": sid, "name": scenario_names(cfg)[sid], "train_rows_removed": int((~keep).sum()),
             "test_rows": int(m.sum()), "test_failed": int(yt[m].sum()),
             "full_model": summarize(p_full), "without_scenario": summarize(p_held)}
        for k in ("pr_auc", "recall_at_capacity"):
            a, b = r["full_model"].get(k), r["without_scenario"].get(k)
            r[f"drop_{k}"] = None if a is None or b is None else a - b
        write_json(r, wd / path)
        ctx.ckpt.save_unit("heldout", sid, [path])
        log.info("held-out %s: recall at capacity %s -> %s", sid, r["full_model"]["recall_at_capacity"],
                 r["without_scenario"]["recall_at_capacity"])
    return out


# ---- SHAP ------------------------------------------------------------------------------------

def stage_shap(ctx: Context) -> list[str]:
    cfg, wd = ctx.cfg, ctx.workdir
    scfg = cfg["train"]["shap"]
    bst, _ = train_xgb.load_booster(wd / model_path(primary(cfg, "xgb")))
    bst.set_param({"device": ctx.device})
    Xt, _ = _xy(ctx, "test")
    phi, files = SV.contributions_chunked(bst, Xt, scfg["chunk_rows"], wd, ctx.ckpt, "shap", "test")
    write_json({"model": primary(cfg, "xgb"), "rows": len(Xt), "importance": SV.mean_abs(phi)},
               wd / "train/shap/global_importance.json")
    Xs, _ = xy(ctx.state["test_sample"])
    np.save(wd / "train/shap/test_sample_shap.npy", train_xgb.contributions(bst, Xs).astype(np.float32))
    k = min(scfg["interaction_rows"], len(Xs))
    inter = train_xgb.contributions(bst, Xs.iloc[:k], interactions=True)
    idx = {n: i for i, n in enumerate(Xs.columns)}
    planted = {name: (inter[:, idx[a], idx[b]] + inter[:, idx[b], idx[a]]).astype(np.float32)
               for name, (a, b) in SV.PLANTED_INTERACTIONS.items()}
    np.savez(wd / "train/shap/planted_interactions.npz", **planted)
    write_json(SV.interaction_summary(inter), wd / "train/shap/interaction_summary.json")
    return files + ["train/shap/global_importance.json", "train/shap/test_sample_shap.npy",
                    "train/shap/planted_interactions.npz", "train/shap/interaction_summary.json"]


def stage_shap_truth(ctx: Context) -> list[str]:
    cfg, wd = ctx.cfg, ctx.workdir
    sample = ctx.state["test_sample"]
    true_imp, phi_true = SV.true_importance_on(sample, cfg)
    result = SV.shap_vs_truth(SV.mean_abs(np.load(wd / "train/shap/test_sample_shap.npy")), true_imp)
    result["note"] = ("True importance is the mean absolute Shapley value of each feature in the planted "
                      "log-odds; model importance is the mean absolute XGBoost SHAP value. Both on the test sample.")
    write_json(result, wd / "train/shap/shap_vs_truth.json")
    np.save(wd / "train/shap/test_sample_true_shap.npy",
            np.stack([phi_true[n] for n in FEATURE_NAMES], axis=1).astype(np.float32))
    ctx.report.add_metrics({"shap_vs_truth": {"spearman": result["spearman"],
                                              "strong_above_weak": result["strong_above_weak"]}})
    if not result["strong_above_weak"]:
        log.warning("SHAP does not rank every Strong feature above every Weak one: investigate before M6")
    return ["train/shap/shap_vs_truth.json", "train/shap/test_sample_true_shap.npy"]


# ---- publishing ------------------------------------------------------------------------------

def model_card(cfg: dict, wd: Path) -> str:
    m = json.loads((wd / "train/metrics.json").read_text())["models"]
    truth = json.loads((wd / "train/shap/shap_vs_truth.json").read_text())
    gh, ds = cfg["project"]["github_repo"], cfg["project"]["dataset_repo"]
    rows = ["| Model | Resampling | PR-AUC | Recall in top 2% | Recall at precision 0.5 | Brier |",
            "|---|---|---|---|---|---|"]
    for r in m.values():
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
        "- `logreg.joblib`: L2 logistic regression baseline, calibrated on validation data",
        "- `svm_linear_calibrated.joblib`: linear SVM with SMOTENC inside the training pipeline, Platt-calibrated",
        "- `svm_rbf_calibrated.joblib`: RBF SVM on a stratified subsample, Platt-calibrated "
        "(fitted with cuML on the GPU when available; reloading it then needs cuML)",
        "- `xgb_model.json`: XGBoost (native format); `xgb_calibrated.joblib`: the same with isotonic calibration",
        "- `models/` and `calibrated/`: every variant (no resampling, class weights, SMOTENC)",
        "- `metrics.json`: every model overall and per scenario; `shap/`: SHAP values and the truth check; "
        "`heldout/`: the held-out scenario experiment",
        "",
        "## Test metrics (synthetic test period, calibrated scores)",
        "",
        *rows,
        "",
        f"SHAP versus the planted truth: rank correlation {truth['spearman']:.2f}; every Strong feature "
        f"ranks above every Weak one: {truth['strong_above_weak']}.",
        "",
        "Accuracy is not a headline metric: at a 3% fail rate, always predicting 'settles' scores about 97%.",
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
        "Inputs are the 20 columns in `feature_schema.json`, with categoricals as pandas categories "
        "using the listed levels.",
        "",
    ])


def stage_publish_models(ctx: Context) -> list[str]:
    cfg, wd = ctx.cfg, ctx.workdir
    write_json(feature_schema(), wd / "train/feature_schema.json")
    (wd / "train/model_card.md").write_text(model_card(cfg, wd))
    lr = joblib.load(wd / model_path(primary(cfg, "logreg")))
    from imblearn.pipeline import Pipeline

    joblib.dump(Pipeline([("encode", lr.named_steps["encode"]), ("expand", lr.named_steps["expand"])]),
                wd / "train/preprocessor.joblib")
    mapping = {
        "README.md": "train/model_card.md",
        "feature_schema.json": "train/feature_schema.json",
        "metrics.json": "train/metrics.json",
        "preprocessor.joblib": "train/preprocessor.joblib",
        "logreg.joblib": calibrated_path(primary(cfg, "logreg")),
        "svm_linear_calibrated.joblib": calibrated_path(primary(cfg, "svm_linear")),
        "svm_rbf_calibrated.joblib": calibrated_path(primary(cfg, "svm_rbf")),
        "xgb_model.json": model_path(primary(cfg, "xgb")),
        "xgb_calibrated.joblib": calibrated_path(primary(cfg, "xgb")),
    }
    for name in all_models(cfg):
        mapping[f"models/{Path(model_path(name)).name}"] = model_path(name)
        mapping[f"calibrated/{name}.joblib"] = calibrated_path(name)
    for p in sorted((wd / "train/shap").glob("*")):
        if p.is_file():
            mapping[f"shap/{p.name}"] = f"train/shap/{p.name}"
    for p in sorted((wd / "train/heldout").glob("*.json")):
        mapping[f"heldout/{p.name}"] = f"train/heldout/{p.name}"
    store = ctx.stores["model"]
    publish(ctx, "publish_models", store, mapping, "publish/model", message="publish models")
    store.tag(cfg["project"]["model_tag"], "models trained on synthetic data")
    ctx.report.set("model_revision", store.revision())
    return ["train/feature_schema.json", "train/model_card.md", "train/preprocessor.joblib"]


def stage_deploy_space(ctx: Context) -> list[str]:
    cfg, wd = ctx.cfg, ctx.workdir
    assets = {"metrics.json": "train/metrics.json", "feature_schema.json": "train/feature_schema.json"}
    files = deploy_space.build_space_dir(cfg, wd, assets)
    from src.config import repo_ids

    space_id = repo_ids(cfg)["space"]
    result = deploy_space.deploy(ctx.stores["space"], wd, files, space_id, wait=ctx.state.get("on_hub", True))
    ctx.report.set("space_url", result["url"])
    ctx.state["info:deploy_space"] = result
    return []


def build_stages() -> list[Stage]:
    base = ("src/models/preprocess.py", "src/features/definitions.py")

    def spec(name, *sources, keys=TRAIN_KEYS):
        return StageSpec(name, (*base, *sources), keys, True)

    return [
        Stage(StageSpec("restore", (), ()), stage_restore, always_run=True),
        Stage(StageSpec("pull_data", (), ()), stage_pull_data, always_run=True),
        Stage(StageSpec("env_check", (), ()), stage_env_check, always_run=True),
        Stage(spec("tune", "src/models/tune.py", "src/models/train_*.py"), stage_tune),
        Stage(spec("train_logreg", "src/models/train_logreg.py"), stage_train_logreg),
        Stage(spec("train_svm", "src/models/train_svm.py"), stage_train_svm),
        Stage(spec("train_rbf", "src/models/train_svm.py"), stage_train_rbf),
        Stage(spec("train_xgb", "src/models/train_xgb.py"), stage_train_xgb),
        Stage(spec("calibrate", "src/models/calibrate.py"), stage_calibrate),
        Stage(spec("evaluate", "src/models/evaluate.py", "src/data/scenarios.py"), stage_evaluate),
        Stage(spec("heldout", "src/models/train_xgb.py"), stage_heldout),
        Stage(spec("shap", "src/explain/shap_views.py"), stage_shap),
        Stage(spec("shap_truth", "src/explain/shap_views.py", "src/data/probability.py",
                   keys=(*TRAIN_KEYS, "effects", "interactions")), stage_shap_truth),
        Stage(spec("publish_models", "src/hub/publish.py", "src/pipeline/run_train.py",
                   keys=("project", *TRAIN_KEYS)), stage_publish_models),
        Stage(StageSpec("deploy_space", ("app/*", "src/hub/deploy_space.py"), ("project",), True),
              stage_deploy_space),
    ]


def main() -> None:
    run_notebook(NOTEBOOK, build_stages)


if __name__ == "__main__":
    main()
