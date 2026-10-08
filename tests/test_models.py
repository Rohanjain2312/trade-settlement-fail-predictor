"""Unit tests for tuning folds, calibration, and SHAP."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.explain import shap_views as SV
from src.features.definitions import FEATURE_NAMES, FEATURES
from src.models import calibrate as C
from src.models import train_logreg, train_xgb
from src.models.preprocess import split_frames, xy
from src.models.tune import time_folds


def test_time_folds_train_only_on_earlier_dates_with_a_gap():
    dates = np.repeat(np.arange(100), 5)
    folds = time_folds(dates, n_folds=3, gap_days=5)
    assert len(folds) == 3
    for tr, va in folds:
        assert dates[tr].max() + 5 < dates[va].min()  # earlier, and the gap is kept
        assert not set(tr) & set(va)


def test_calibration_fixes_inflated_probabilities(smoke_data):
    splits = split_frames(smoke_data.df, smoke_data.cfg)
    X, y = xy(splits["train"])
    Xv, yv = xy(splits["val"])
    weighted = train_logreg.fit(X, y, 1.0, "class_weight", 0)
    raw = weighted.predict_proba(Xv)[:, 1]
    cal = C.calibrate(weighted, Xv, yv, "logreg").predict_proba(Xv)[:, 1]
    # Class weights push the average predicted risk far above the real fail rate.
    assert raw.mean() > 3 * yv.mean()
    assert abs(cal.mean() - yv.mean()) < 0.5 * yv.mean()


def _toy_xgb():
    rng = np.random.default_rng(1)
    n = 2000
    X = pd.DataFrame({f.name: (pd.Categorical(rng.choice(f.levels, n), categories=list(f.levels)) if f.levels
                               else rng.normal(size=n)) for f in FEATURES})
    y = ((X["hours_to_confirmation"] + (X["ssi_match_status"] == "mismatch") * 2 + rng.normal(size=n)) > 1).astype(int)
    p = train_xgb.params({"max_depth": 3, "eta": 0.2, "subsample": 1.0, "colsample_bytree": 1.0,
                          "min_child_weight": 1.0, "reg_lambda": 1.0}, "cpu", 0)
    bst, _ = train_xgb.train(p, train_xgb.dmatrix(X, y.to_numpy()), train_xgb.dmatrix(X, y.to_numpy()), 30, 100, 10**9)
    return bst, X


def test_shap_values_add_up_to_the_margin():
    bst, X = _toy_xgb()
    phi = train_xgb.contributions(bst, X)
    margin = bst.predict(train_xgb.dmatrix(X), output_margin=True)
    np.testing.assert_allclose(phi.sum(axis=1), margin, atol=1e-4)
    imp = SV.mean_abs(phi)
    assert max(imp, key=imp.get) in ("hours_to_confirmation", "ssi_match_status")


def test_shap_vs_truth_flags_a_wrong_ranking():
    true = {f.name: {"strong": 3.0, "medium": 2.0, "weak": 1.0}[f.strength] for f in FEATURES}
    good = SV.shap_vs_truth(true, true)
    assert good["strong_above_weak"] and good["spearman"] > 0.99
    flipped = {n: 4.0 - v for n, v in true.items()}
    bad = SV.shap_vs_truth(flipped, true)
    assert not bad["strong_above_weak"] and bad["spearman"] < 0
    assert [r["feature"] for r in good["rows"]][:1] and len(good["rows"]) == len(FEATURE_NAMES)
