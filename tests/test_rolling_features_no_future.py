"""Rolling features use only outcomes known on the trade date: outcomes of trades whose
settle date is before the trade date. Same-day and future information never leaks in."""

from __future__ import annotations

import numpy as np

from src.data.generator import (
    GenState,
    group_median,
    rolling_cpty_median,
    rolling_fail_rates,
    rolling_pair_counts,
)
from src.data.validate import recompute_rolling


def test_stored_rolling_features_match_a_recompute_from_raw_history(smoke_data):
    result = recompute_rolling(smoke_data.df, smoke_data.cfg, n_sample=500)
    assert result["n_checked"] >= 300
    assert result["passed"], result["mismatches"][:5]


def _state(n_cpty=3, n_sec=2):
    return GenState(day=0, next_trade_id=0, ssi_verified_ord=np.zeros((n_cpty, 5), dtype=np.int64))


def _outcome(cpty_fails, cpty_n, n_sec=2):
    cc = np.array([cpty_fails, cpty_n], dtype=np.int64)
    return cc, np.zeros((2, n_sec), dtype=np.int64)


def test_fail_rate_window_is_prior_30_days_by_settle_date(smoke_data):
    cfg = smoke_data.cfg
    ref = type("R", (), {"n_cpty": 3, "n_sec": 2})()
    t = 1000
    s = _state()
    s.outcomes = {
        t: _outcome([50, 0, 0], [50, 0, 0]),       # settles today: not known yet
        t + 1: _outcome([50, 0, 0], [50, 0, 0]),   # settles in the future
        t - 1: _outcome([1, 0, 0], [10, 0, 0]),    # known
        t - 30: _outcome([1, 0, 0], [10, 0, 0]),   # known, oldest day in the window
        t - 31: _outcome([50, 0, 0], [50, 0, 0]),  # too old
    }
    cpty_rate, _ = rolling_fail_rates(s, t, ref, cfg)
    a, prior = cfg["data"]["rate_prior_trades"]["cpty"], cfg["data"]["target_fail_rate"]
    assert np.isclose(cpty_rate[0], (2 + a * prior) / (20 + a))
    assert np.isclose(cpty_rate[1], prior)  # no history: the prior


def test_pair_and_size_windows_exclude_the_current_day():
    t = 1000
    s = _state()
    s.pair_days = [(t, np.array([9, 9])), (t - 1, np.array([1, 0])), (t - 90, np.array([1, 1])),
                   (t - 91, np.array([5, 5]))]
    np.testing.assert_array_equal(rolling_pair_counts(s, t, 2), [2, 1])
    s.notional_days = [
        (t, np.array([0], dtype=np.int16), np.array([1e9], dtype=np.float32)),
        (t - 1, np.array([0, 0, 1], dtype=np.int16), np.array([1.0, 3.0, 7.0], dtype=np.float32)),
        (t - 91, np.array([1], dtype=np.int16), np.array([1e9], dtype=np.float32)),
    ]
    med = rolling_cpty_median(s, t, 3)
    assert med[0] == 2.0 and med[1] == 7.0 and np.isnan(med[2])


def test_group_median_matches_numpy():
    rng = np.random.default_rng(0)
    g = rng.integers(0, 7, 500)
    v = rng.lognormal(size=500).astype(np.float32)
    med = group_median(g, v, 8)
    for k in range(7):
        assert med[k] == np.median(v[g == k].astype(np.float64))
    assert np.isnan(med[7])
