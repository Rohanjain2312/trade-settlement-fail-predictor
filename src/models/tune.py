"""Hyperparameter tuning with Optuna.

Each (model family, resampling variant) gets its own study. Trials score PR-AUC with
time-series cross-validation inside the training period: folds are consecutive blocks of
trade dates, each fold trains on earlier dates only, and the last gap_days business days
before each validation block are dropped. SMOTE runs inside the pipeline, so it only ever
sees a fold's training rows. The study lives in a SQLite file that is saved to the work repo
every few trials and reloaded with load_if_exists=True after a restart.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import average_precision_score

from src.models import train_logreg, train_svm, train_xgb

log = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


def time_folds(dates: np.ndarray, n_folds: int, gap_days: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Expanding-window folds over sorted unique dates: fold k trains on blocks 0..k and
    validates on block k+1, minus a gap of gap_days dates before the validation block."""
    days = np.unique(dates)
    blocks = np.array_split(days, n_folds + 1)
    folds = []
    for k in range(n_folds):
        val_days = blocks[k + 1]
        train_days = np.concatenate(blocks[: k + 1])
        if gap_days > 0:
            train_days = train_days[:-gap_days] if len(train_days) > gap_days else train_days[:0]
        tr = np.flatnonzero(np.isin(dates, train_days))
        va = np.flatnonzero(np.isin(dates, val_days))
        if len(tr) and len(va):
            folds.append((tr, va))
    return folds


def suggest(family: str, trial: optuna.Trial) -> dict:
    if family == "logreg":
        return {"C": trial.suggest_float("C", 1e-3, 10.0, log=True)}
    if family == "svm_linear":
        return {"C": trial.suggest_float("C", 1e-4, 0.5, log=True)}
    if family == "svm_rbf":
        return {"C": trial.suggest_float("C", 0.1, 100.0, log=True),
                "gamma": trial.suggest_float("gamma", 1e-3, 1.0, log=True)}
    if family == "xgb":
        return {
            "max_depth": trial.suggest_int("max_depth", 3, 9),
            "eta": trial.suggest_float("eta", 0.02, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 20.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
        }
    raise ValueError(family)


def fit_score(family: str, variant: str, params: dict, X: pd.DataFrame, y: np.ndarray,
              tr: np.ndarray, va: np.ndarray, cfg: dict, device: str, use_cuml: bool) -> float:
    seed = cfg["seed"]
    Xtr, ytr, Xva, yva = X.iloc[tr], y[tr], X.iloc[va], y[va]
    if yva.sum() == 0 or ytr.sum() < 6:
        return float("nan")
    if family == "xgb":
        xcfg = {**cfg["train"]["xgb"], **params}
        if variant == "smote":
            Xtr, ytr = train_xgb.smote_resample(Xtr, ytr, seed)
        spw = float((ytr == 0).sum() / max(1, ytr.sum())) if variant == "scale_pos_weight" else 1.0
        p = train_xgb.params(xcfg, device, seed, spw)
        bst, _ = train_xgb.train(p, train_xgb.dmatrix(Xtr, ytr), train_xgb.dmatrix(Xva, yva),
                                 xcfg["max_rounds"], xcfg["patience"], every=10**9)
        scores = train_xgb.predict(bst, Xva)
    else:
        if family == "logreg":
            model = train_logreg.make(params["C"], variant, seed)
        elif family == "svm_linear":
            model = train_svm.make_linear(params["C"], variant, seed)
        else:
            model = train_svm.make_rbf(params["C"], params["gamma"], variant, seed, use_cuml)
        model.fit(Xtr, ytr)
        scores = model.decision_function(Xva) if hasattr(model, "decision_function") else model.predict_proba(Xva)[:, 1]
    return float(average_precision_score(yva, np.asarray(scores, dtype=float)))


def tune(family: str, variant: str, X: pd.DataFrame, y: np.ndarray, dates: np.ndarray, cfg: dict,
         storage: Path, n_trials: int, device: str, use_cuml: bool,
         on_save: Callable[[], None] | None = None) -> dict:
    """Run (or resume) the study until it has n_trials finished trials. Returns the best."""
    tcfg = cfg["train"]["tune"]
    folds = time_folds(dates, tcfg["folds"], cfg["split"]["gap_days"])
    storage.parent.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        study_name=f"{family}__{variant}",
        storage=f"sqlite:///{storage}",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=cfg["seed"]),
        load_if_exists=True,
    )
    done = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])

    def objective(trial):
        params = suggest(family, trial)
        scores = [fit_score(family, variant, params, X, y, tr, va, cfg, device, use_cuml) for tr, va in folds]
        scores = [s for s in scores if np.isfinite(s)]
        return float(np.mean(scores)) if scores else 0.0

    def checkpoint(study, trial):
        finished = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        if on_save is not None and finished % tcfg["save_every"] == 0:
            on_save()

    if n_trials - done > 0:
        log.info("tuning %s__%s: %d trials to go (%d done)", family, variant, n_trials - done, done)
        study.optimize(objective, n_trials=n_trials - done, callbacks=[checkpoint])
    best = study.best_trial
    return {"params": best.params, "cv_pr_auc": float(best.value), "trials": len(study.trials),
            "folds": len(folds)}
