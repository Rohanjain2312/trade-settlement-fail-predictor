from __future__ import annotations

import copy

import numpy as np

from src.data.generator import generate_day, init_state
from src.data.validate import (
    check_lr_recovery,
    check_rates,
    check_strength_tiers,
    reason_mix,
    true_importance,
)


def test_intercept_calibration_converges(smoke_data):
    hist = smoke_data.calibration["history"]
    target = smoke_data.cfg["data"]["target_fail_rate"]
    tol = smoke_data.cfg["data"]["intercept_calibration"]["tolerance"]
    assert abs(hist[-1]["fail_rate"] - target) <= tol


def test_overall_and_monthly_fail_rates_are_in_band(smoke_data):
    result = check_rates(smoke_data.df, smoke_data.cfg)
    assert result["passed"], result


def test_every_fail_reason_occurs(smoke_data):
    mix = reason_mix(smoke_data.df)
    assert all(0.05 <= share <= 0.6 for share in mix.values()), mix
    assert smoke_data.df.loc[smoke_data.df["failed"] == 0, "fail_reason"].isna().all()


def test_planted_strengths_show_up_in_the_data(smoke_data):
    result = check_strength_tiers(true_importance(smoke_data.df, smoke_data.cfg))
    assert result["passed"], result


def test_plain_logistic_regression_recovers_the_planted_signs(smoke_data):
    importance = true_importance(smoke_data.df, smoke_data.cfg)
    result = check_lr_recovery(smoke_data.df, smoke_data.cfg, importance)
    assert result["passed"], {k: result[k] for k in ("wrong_sign", "rank_correlation_with_truth")}


def test_generation_is_deterministic(smoke_data):
    ref, cfg = smoke_data.ref, smoke_data.cfg
    intercept = smoke_data.calibration["intercept"]
    a, b = init_state(ref, cfg), init_state(ref, cfg)
    for t in range(5):
        da = generate_day(t, a, ref, cfg, intercept)
        db = generate_day(t, b, ref, cfg, intercept)
        for k in da:
            np.testing.assert_array_equal(da[k], db[k])
    # A copy of the state continues identically, which is what a restart relies on.
    c = copy.deepcopy(a)
    da, dc = generate_day(5, a, ref, cfg, intercept), generate_day(5, c, ref, cfg, intercept)
    for k in da:
        np.testing.assert_array_equal(da[k], dc[k])
