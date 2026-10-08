"""Model 3: XGBoost with native categorical and missing-value handling.

Training runs its own boosting loop instead of xgb.train, so a run that resumes from a saved
booster passes the global iteration number to every update. With seed_per_iteration the
random row and column sampling of iteration i then depends only on (seed, i), and a resumed
run grows the same trees as an uninterrupted one. Early stopping watches validation aucpr.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from imblearn.over_sampling import SMOTENC

from src.features.definitions import BY_NAME, CATEGORICAL, FEATURE_NAMES, NUMERIC

log = logging.getLogger(__name__)


def dmatrix(X: pd.DataFrame, y=None) -> xgb.DMatrix:
    return xgb.DMatrix(X[list(FEATURE_NAMES)], label=y, enable_categorical=True, missing=np.nan)


def params(xcfg: dict, device: str, seed: int, scale_pos_weight: float = 1.0) -> dict:
    return {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "tree_method": "hist",
        "device": device,
        "max_depth": xcfg["max_depth"],
        "eta": xcfg["eta"],
        "subsample": xcfg["subsample"],
        "colsample_bytree": xcfg["colsample_bytree"],
        "min_child_weight": xcfg["min_child_weight"],
        "lambda": xcfg["reg_lambda"],
        "max_cat_to_onehot": 1,
        "scale_pos_weight": scale_pos_weight,
        "seed": seed,
        "seed_per_iteration": True,
    }


def smote_resample(X: pd.DataFrame, y: np.ndarray, seed: int, k_neighbors: int = 5):
    """SMOTENC on the training rows only. Numerics are median-imputed first because SMOTENC
    needs complete rows; categoricals go in as codes and come back as categories."""
    codes = pd.DataFrame({c: X[c].cat.codes.astype(float) for c in CATEGORICAL})
    num = X[list(NUMERIC)].astype(float)
    num = num.fillna(num.median())
    Z = pd.concat([codes, num], axis=1)
    sm = SMOTENC(categorical_features=list(range(len(CATEGORICAL))), k_neighbors=k_neighbors, random_state=seed)
    Zr, yr = sm.fit_resample(Z.to_numpy(), y)
    out = pd.DataFrame(Zr, columns=list(CATEGORICAL) + list(NUMERIC))
    for c in CATEGORICAL:
        out[c] = pd.Categorical.from_codes(out[c].round().astype(int), categories=list(BY_NAME[c].levels))
    return out[list(FEATURE_NAMES)], yr


def _score(bst: xgb.Booster, dval: xgb.DMatrix) -> float:
    # Booster.eval returns e.g. "[0]\tval-aucpr:0.123456"
    return float(bst.eval(dval, "val").split(":")[-1])


def train(
    p: dict,
    dtrain: xgb.DMatrix,
    dval: xgb.DMatrix,
    max_rounds: int,
    patience: int,
    every: int,
    save_progress: Callable[[xgb.Booster, dict], None] | None = None,
    resume: tuple[xgb.Booster, dict] | None = None,
) -> tuple[xgb.Booster, dict]:
    """Boost until validation aucpr has not improved for `patience` rounds. Calls
    save_progress(booster, state) every `every` rounds. Returns the booster cut at the best
    iteration and the training record."""
    if resume is None:
        bst = xgb.Booster(p, [dtrain, dval])
        state = {"next_iteration": 0, "best_score": -1.0, "best_iteration": -1, "history": []}
    else:
        bst, state = resume
        bst.set_param(p)
        log.info("resuming boosting at iteration %d", state["next_iteration"])
    i = state["next_iteration"]
    while i < max_rounds:
        bst.update(dtrain, iteration=i)
        score = _score(bst, dval)
        state["history"].append(round(score, 6))
        if score > state["best_score"] + 1e-12:
            state["best_score"], state["best_iteration"] = score, i
        i += 1
        state["next_iteration"] = i
        done = i - 1 - state["best_iteration"] >= patience or i >= max_rounds
        if save_progress is not None and (i % every == 0 or done):
            save_progress(bst, state)
        if done:
            break
    best = bst[: state["best_iteration"] + 1]
    return best, state


def save_booster(bst: xgb.Booster, path: Path, state: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bst.save_model(str(path))
    if state is not None:
        Path(str(path) + ".state.json").write_text(json.dumps(state))


def load_booster(path: Path, p: dict | None = None, cache: list | None = None) -> tuple[xgb.Booster, dict | None]:
    bst = xgb.Booster(p or {}, cache or [], model_file=str(path))
    state_path = Path(str(path) + ".state.json")
    state = json.loads(state_path.read_text()) if state_path.exists() else None
    return bst, state


def predict(bst: xgb.Booster, X: pd.DataFrame) -> np.ndarray:
    return bst.predict(dmatrix(X))


def contributions(bst: xgb.Booster, X: pd.DataFrame, interactions: bool = False) -> np.ndarray:
    """SHAP values on the log-odds scale computed by XGBoost itself (GPU when the booster
    runs on cuda). Last column (or last row and column) is the bias."""
    d = dmatrix(X)
    if interactions:
        return bst.predict(d, pred_interactions=True)
    return bst.predict(d, pred_contribs=True)
