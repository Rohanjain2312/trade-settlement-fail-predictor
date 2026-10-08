"""The ops queue simulator: ranking within days, capacity limits, and model versus no model."""

from __future__ import annotations

import numpy as np

from src.explain.app_assets import OPS_CHECK
from src.features.definitions import FEATURE_NAMES
from src.sim.simulate_queue import compare, day_ranks, simulate


def _data(n_days=40, per_day=500, seed=0):
    rng = np.random.default_rng(seed)
    day = np.repeat(np.arange(n_days), per_day)
    risk = rng.random(len(day))
    failed = rng.random(len(day)) < 0.06 * risk * 2
    return day, failed, risk


def test_day_ranks_rank_within_each_day():
    day = np.array([0, 0, 0, 1, 1])
    score = np.array([0.1, 0.9, 0.5, 0.3, 0.8])
    np.testing.assert_array_equal(day_ranks(day, score), [2, 0, 1, 1, 0])


def test_no_capacity_fixes_nothing_and_full_capacity_fixes_everything():
    day, failed, risk = _data()
    none = simulate(day, failed, risk, 0.0, 1.0)
    assert none["fixed"].sum() == 0 and none["pending"].sum() == failed.sum()
    full = simulate(day, failed, risk, 1.0, 1.0)
    assert full["pending"].sum() == 0


def test_an_informative_score_beats_no_model_and_a_random_score_does_not():
    day, failed, risk = _data()
    good = compare(day, failed, risk, 0.05, 0.6, 100, 200)
    assert good["pending_with"] < good["pending_without"]
    assert 0 < good["pending_reduction"] < 0.6
    assert good["quarterly_exposure_with"] < good["quarterly_exposure_without"]
    noise = compare(day, failed, np.random.default_rng(9).random(len(day)), 0.05, 0.6, 100, 200)
    assert abs(noise["pending_reduction"]) < 0.02


def test_every_feature_has_an_ops_action():
    assert set(OPS_CHECK) == set(FEATURE_NAMES)
