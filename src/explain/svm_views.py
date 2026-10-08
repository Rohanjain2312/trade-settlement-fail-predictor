"""Precomputed views for the SVM tab: a 2-feature decision-boundary illustration across C and
gamma, a SMOTE illustration, the leakage comparison, and why SVM scores need calibrating.

The boundary view uses only two features so it can be drawn. It is an illustration of how an
SVM works, not the production model, and the app labels it that way.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTENC
from sklearn.metrics import average_precision_score
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import SVC

from src.features.definitions import BY_NAME, CATEGORICAL, NUMERIC

C_VALUES = (0.1, 1.0, 10.0, 100.0)
GAMMA_VALUES = (0.1, 1.0, 10.0)
GRID = 60


def _balanced(df: pd.DataFrame, n_per_class: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    parts = []
    for cls in (0, 1):
        rows = np.flatnonzero(df["failed"].to_numpy() == cls)
        parts.append(df.iloc[rng.choice(rows, size=min(n_per_class, len(rows)), replace=False)])
    return pd.concat(parts).reset_index(drop=True)


def boundary_views(train: pd.DataFrame, features: tuple[str, str], seed: int, n_per_class: int = 400) -> dict:
    """Fit linear and RBF SVMs on two standardized features for a grid of C and gamma."""
    a, b = features
    fit_df = _balanced(train.dropna(subset=[a, b]), n_per_class, seed)
    hold_df = _balanced(train.dropna(subset=[a, b]), n_per_class, seed + 1)
    mu = fit_df[[a, b]].mean().to_numpy()
    sd = fit_df[[a, b]].std().to_numpy() + 1e-9
    Z = (fit_df[[a, b]].to_numpy() - mu) / sd
    Zh = (hold_df[[a, b]].to_numpy() - mu) / sd
    y, yh = fit_df["failed"].to_numpy(), hold_df["failed"].to_numpy()
    lo, hi = np.quantile(Z, 0.005, axis=0) - 0.3, np.quantile(Z, 0.995, axis=0) + 0.3
    gx, gy = np.linspace(lo[0], hi[0], GRID), np.linspace(lo[1], hi[1], GRID)
    mesh = np.column_stack([np.repeat(gx, GRID), np.tile(gy, GRID)])
    settings = []
    for kernel in ("linear", "rbf"):
        for C in C_VALUES:
            for gamma in (GAMMA_VALUES if kernel == "rbf" else (None,)):
                model = SVC(kernel=kernel, C=C, gamma=gamma if gamma else "scale").fit(Z, y)
                settings.append({
                    "kernel": kernel, "C": C, "gamma": gamma,
                    "decision": np.round(model.decision_function(mesh).reshape(GRID, GRID).T, 3).tolist(),
                    "support": model.support_.tolist(),
                    "n_support": int(len(model.support_)),
                    "train_accuracy": float((model.predict(Z) == y).mean()),
                    "holdout_accuracy": float((model.predict(Zh) == yh).mean()),
                    "holdout_pr_auc": float(average_precision_score(yh, model.decision_function(Zh))),
                })
    return {
        "features": [a, b],
        "labels": [BY_NAME[a].label, BY_NAME[b].label],
        "note": ("Illustration on two standardized features and a balanced sample of "
                 f"{len(y)} training trades, so the boundary can be drawn. The real models use all 20 features."),
        "points": {"x": np.round(Z[:, 0], 3).tolist(), "y": np.round(Z[:, 1], 3).tolist(), "failed": y.tolist()},
        "grid": {"x": np.round(gx, 3).tolist(), "y": np.round(gy, 3).tolist()},
        "settings": settings,
    }


def smote_view(train: pd.DataFrame, features: tuple[str, str], seed: int, n_trades: int = 4000,
               n_synthetic: int = 300, k: int = 5) -> dict:
    """How SMOTE makes a synthetic fail: pick a fail, pick one of its k nearest fails, and
    place a new point at a random spot on the line between them."""
    a, b = features
    rng = np.random.default_rng(seed)
    df = train.dropna(subset=[a, b])
    df = df.iloc[rng.choice(len(df), size=min(n_trades, len(df)), replace=False)]
    Z = df[[a, b]].to_numpy()
    mu, sd = Z.mean(axis=0), Z.std(axis=0) + 1e-9
    Z = (Z - mu) / sd
    y = df["failed"].to_numpy()
    fails = Z[y == 1]
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(fails))).fit(fails)
    neighbors = nn.kneighbors(fails, return_distance=False)[:, 1:]
    parent = rng.integers(0, len(fails), n_synthetic)
    partner = neighbors[parent, rng.integers(0, neighbors.shape[1], n_synthetic)]
    t = rng.random(n_synthetic)
    synth = fails[parent] + t[:, None] * (fails[partner] - fails[parent])
    return {
        "features": [a, b],
        "labels": [BY_NAME[a].label, BY_NAME[b].label],
        "settled": {"x": np.round(Z[y == 0, 0], 3).tolist(), "y": np.round(Z[y == 0, 1], 3).tolist()},
        "failed": {"x": np.round(fails[:, 0], 3).tolist(), "y": np.round(fails[:, 1], 3).tolist()},
        "synthetic": {"x": np.round(synth[:, 0], 3).tolist(), "y": np.round(synth[:, 1], 3).tolist(),
                      "parent": parent.tolist(), "partner": partner.tolist(), "t": np.round(t, 3).tolist()},
        "k": k,
        "n_trades": int(len(y)),
    }


def leakage_check(train: pd.DataFrame, test: pd.DataFrame, encoder, seed: int, rows: int = 60000) -> dict:
    """Median distance from each test fail to its nearest synthetic fail, when SMOTE runs on
    the training rows only (right) versus on train and test together before the split (wrong)."""
    rng = np.random.default_rng(seed)
    tr = train.iloc[rng.choice(len(train), size=min(rows, len(train)), replace=False)]
    te = test.iloc[rng.choice(len(test), size=min(rows // 3, len(test)), replace=False)]
    from src.models.preprocess import xy

    Xtr, ytr = xy(tr)
    Xte, yte = xy(te)
    enc = encoder.fit(Xtr)
    A, B = enc.transform(Xtr), enc.transform(Xte)
    num = slice(len(CATEGORICAL), len(CATEGORICAL) + len(NUMERIC))
    scale = A[:, num].std(axis=0) + 1e-9
    sm = SMOTENC(categorical_features=list(range(len(CATEGORICAL))), random_state=seed)
    right, _ = sm.fit_resample(A, ytr)
    both = np.vstack([A, B])
    wrong, _ = sm.fit_resample(both, np.concatenate([ytr, yte]))
    test_fails = B[yte == 1][:, num] / scale

    def dist(synth):
        nn = NearestNeighbors(n_neighbors=1).fit(synth[:, num] / scale)
        return float(np.median(nn.kneighbors(test_fails)[0]))

    d_right, d_wrong = dist(right[len(A):]), dist(wrong[len(both):])
    return {"median_distance_right": d_right, "median_distance_wrong": d_wrong,
            "ratio": d_wrong / d_right if d_right else None, "train_rows": len(tr), "test_rows": len(te)}


def default_threshold_effect(models: dict, X: pd.DataFrame, y: np.ndarray) -> list[dict]:
    """What each raw SVM flags with its own decision rule (score > 0), before calibration."""
    out = []
    for name, model in models.items():
        flagged = np.asarray(model.decision_function(X)) > 0
        tp = int((flagged & (y == 1)).sum())
        out.append({"model": name, "flag_rate": float(flagged.mean()),
                    "recall": tp / max(1, int(y.sum())), "precision": tp / max(1, int(flagged.sum()))})
    return out


def platt_params(calibrated) -> dict | None:
    """Slope and intercept of the Platt map p = 1 / (1 + exp(a * score + b))."""
    try:
        c = calibrated.calibrated_classifiers_[0].calibrators[0]
        return {"a": float(c.a_), "b": float(c.b_)}
    except (AttributeError, IndexError):
        return None
