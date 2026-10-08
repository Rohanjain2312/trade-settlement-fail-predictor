"""Scenario coverage is a measured property: the mask matches each trade's situation,
injectors raise prevalence, hard negatives mostly settle, and the coverage report passes."""

from __future__ import annotations

import copy

import numpy as np

from src.data.coverage import report_markdown, run_coverage
from src.data.scenarios import ROOT_CAUSES, SCENARIO_IDS, decode, quota, to_mask
from src.models.preprocess import split_frames


def _coverage(smoke_data, cfg=None):
    cfg = cfg or smoke_data.cfg
    return run_coverage(smoke_data.df, split_frames(smoke_data.df, cfg), cfg)


def test_coverage_report_passes_on_smoke_data(smoke_data):
    result = _coverage(smoke_data)
    assert result["passed"], result["failures"][:15]


def test_report_lists_every_scenario_and_says_synthetic(smoke_data):
    md = report_markdown(_coverage(smoke_data), smoke_data.cfg)
    for sid in SCENARIO_IDS:
        assert f"`{sid}`" in md
    assert "synthetic" in md.lower() and "Realism calibration" in md


def test_unmet_minimum_fails_the_report(smoke_data):
    cfg = copy.deepcopy(smoke_data.cfg)
    cfg["coverage"]["min_train"]["failed"] = 10**6
    result = _coverage(smoke_data, cfg)
    assert not result["passed"]
    assert any(f.startswith("coverage S1:") for f in result["failures"])


def test_mask_round_trip():
    rng = np.random.default_rng(0)
    members = {sid: rng.random(500) < 0.3 for sid in SCENARIO_IDS}
    back = decode(to_mask(members))
    for sid in SCENARIO_IDS:
        np.testing.assert_array_equal(back[sid], members[sid])


def test_mask_matches_each_trades_situation(smoke_data):
    df = smoke_data.df
    m = decode(df["scenario_mask"].to_numpy())
    np.testing.assert_array_equal(m["S1"], (df["ssi_match_status"] == "mismatch").to_numpy())
    np.testing.assert_array_equal(m["S3"], (df["obligation_coverage_ratio"] < 1).to_numpy())
    np.testing.assert_array_equal(m["S7"], (df["corporate_action_in_window"] == 1).to_numpy())
    np.testing.assert_array_equal(m["R1"], ~(m["R2"] | m["R3"] | m["R4"]))
    assert not (m["H1"] & (df["ssi_match_status"] == "matched").to_numpy()).any()
    roots = np.zeros(len(df), dtype=bool)
    for sid in ROOT_CAUSES:
        roots |= m[sid]
    assert not (m["H4"] & roots).any(), "unexplained fails must look clean"


def test_injectors_raise_prevalence_above_their_quota(smoke_data):
    m = decode(smoke_data.df["scenario_mask"].to_numpy())
    for sid in ("S1", "S3", "S4", "S5", "S7", "S8"):
        assert m[sid].mean() >= quota(smoke_data.cfg, sid), sid


def test_hard_negatives_mostly_settle_and_unexplained_fails_fail(smoke_data):
    m = decode(smoke_data.df["scenario_mask"].to_numpy())
    y = smoke_data.df["failed"].to_numpy()
    for sid in ("H1", "H2", "H3"):
        assert m[sid].sum() > 50 and y[m[sid]].mean() < 0.25, sid
    assert y[m["H4"]].mean() > 0.5
