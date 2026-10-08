"""SHAP values for the XGBoost model, and the check against the planted truth.

SHAP values come from XGBoost itself (pred_contribs, and pred_interactions on a sample),
on the GPU when the booster runs on cuda. They are on the log-odds scale and add up to the
model's margin. Because the data is synthetic, the true contribution of each feature to the
planted log-odds is known (src/data/probability.true_shap), so the model's explanations can
be compared with the truth.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from src.data import probability as prob
from src.data.generator import frame_to_arrays
from src.features.definitions import BY_NAME, FEATURE_NAMES, FEATURES, STRENGTH_ORDER
from src.models import train_xgb

PLANTED_INTERACTIONS = prob.INTERACTIONS  # name -> (feature, feature)


def contributions_chunked(bst, X: pd.DataFrame, chunk_rows: int, workdir, ckpt, stage: str, prefix: str):
    """SHAP values for every row, computed and checkpointed one chunk at a time."""
    parts, files = [], []
    for i, start in enumerate(range(0, len(X), chunk_rows)):
        rel = f"train/shap/chunks/{prefix}_{i:04d}.npy"
        files.append(rel)
        path = workdir / rel
        if not ckpt.unit_done(stage, f"{prefix}_{i:04d}"):
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, train_xgb.contributions(bst, X.iloc[start:start + chunk_rows]).astype(np.float32))
            ckpt.save_unit(stage, f"{prefix}_{i:04d}", [rel])
        parts.append(np.load(path))
    return np.concatenate(parts), files


def mean_abs(phi: np.ndarray) -> dict[str, float]:
    """Global importance: mean absolute SHAP value per feature (bias column dropped)."""
    return {name: float(v) for name, v in zip(FEATURE_NAMES, np.abs(phi[:, : len(FEATURE_NAMES)]).mean(axis=0))}


def interaction_summary(inter: np.ndarray) -> dict:
    """Mean absolute interaction value per feature pair (off-diagonal, both halves added)."""
    n = len(FEATURE_NAMES)
    m = np.abs(inter[:, :n, :n]).mean(axis=0)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append({"a": FEATURE_NAMES[i], "b": FEATURE_NAMES[j], "mean_abs": float(m[i, j] + m[j, i])})
    pairs.sort(key=lambda p: -p["mean_abs"])
    planted = {frozenset(v) for v in PLANTED_INTERACTIONS.values()}
    for rank, p in enumerate(pairs, 1):
        p["rank"] = rank
        p["planted"] = frozenset((p["a"], p["b"])) in planted
    return {"top_pairs": pairs[:15], "planted": [p for p in pairs if p["planted"]], "n_pairs": len(pairs)}


def true_importance_on(df: pd.DataFrame, cfg: dict) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    phi = prob.true_shap(frame_to_arrays(df), cfg)
    return {name: float(np.mean(np.abs(phi[name]))) for name in FEATURE_NAMES}, phi


def shap_vs_truth(model_imp: dict[str, float], true_imp: dict[str, float]) -> dict:
    """Rank agreement between the model's SHAP importance and the planted truth, and whether
    every Strong feature ranks above every Weak feature."""
    names = list(FEATURE_NAMES)
    rho = float(stats.spearmanr([model_imp[n] for n in names], [true_imp[n] for n in names])[0])
    model_rank = {n: r for r, n in enumerate(sorted(names, key=lambda n: -model_imp[n]), 1)}
    true_rank = {n: r for r, n in enumerate(sorted(names, key=lambda n: -true_imp[n]), 1)}
    strong = [f.name for f in FEATURES if f.strength == "strong"]
    weak = [f.name for f in FEATURES if f.strength == "weak"]
    strong_above_weak = min(model_imp[n] for n in strong) > max(model_imp[n] for n in weak)
    rows = [{
        "feature": n,
        "label": BY_NAME[n].label,
        "strength": BY_NAME[n].strength,
        "shap_importance": model_imp[n],
        "true_importance": true_imp[n],
        "shap_rank": model_rank[n],
        "true_rank": true_rank[n],
    } for n in sorted(names, key=lambda n: model_rank[n])]
    tier_mean = {t: float(np.mean([model_imp[f.name] for f in FEATURES if f.strength == t])) for t in STRENGTH_ORDER}
    return {
        "spearman": rho,
        "strong_above_weak": bool(strong_above_weak),
        "shap_tier_means": tier_mean,
        "rows": rows,
    }
