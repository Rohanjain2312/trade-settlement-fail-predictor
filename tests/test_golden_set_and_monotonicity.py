"""The planted probability function responds in the expected direction.

Golden set: about 50 hand-written trades, at least one per scenario (tests/golden_set.yaml).
Property tests: raising a risk feature with everything else fixed never lowers the log-odds.
"""

from __future__ import annotations

import zlib
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.config import load_config
from src.data import probability as prob
from src.data.scenarios import MATCHED, ROOT_CAUSES, SCENARIO_IDS
from src.features.definitions import BY_NAME, CATEGORICAL, FEATURE_NAMES, FEATURES

GOLDEN = yaml.safe_load((Path(__file__).parent / "golden_set.yaml").read_text())
CFG = load_config("full")


def _arrays(trades: list[dict]) -> dict:
    out = {}
    for name in FEATURE_NAMES:
        vals = [t[name] for t in trades]
        if name in CATEGORICAL:
            out[name] = np.array([BY_NAME[name].levels.index(v) for v in vals])
        else:
            out[name] = np.array(vals, dtype=float)
    return out


def _logit(trade: dict, hidden: dict | None = None) -> float:
    hidden = hidden or {}
    F = _arrays([trade])
    repaired = np.array([hidden.get("auto_repair", False)]) & (F["ssi_match_status"] != MATCHED)
    borrowed = np.array([hidden.get("late_borrow", False)]) & (F["obligation_coverage_ratio"] < 1)
    logit, _ = prob.planted_logit(
        F, CFG, 0.0, repaired=repaired, borrowed=borrowed,
        unexplained=np.array([hidden.get("unexplained", False)]),
        cpty_latent=np.zeros(1), sec_latent=np.zeros(1), noise=np.zeros(1),
    )
    return float(logit[0])


def _case_logits():
    base = GOLDEN["baseline"]
    out = {"baseline": _logit(base)}
    for case in GOLDEN["cases"]:
        out[case["id"]] = _logit({**base, **case["change"]}, case.get("hidden"))
    return out


LOGITS = _case_logits()


def test_golden_set_size_and_scenario_coverage():
    cases = GOLDEN["cases"]
    assert len(cases) >= 50
    assert len({c["id"] for c in cases}) == len(cases)
    named = {part for c in cases for part in str(c["scenario"]).split("+")}
    missing = [sid for sid in SCENARIO_IDS if sid not in named]
    assert not missing, f"golden set has no case for {missing}"


@pytest.mark.parametrize("case", GOLDEN["cases"], ids=lambda c: c["id"])
def test_golden_case(case):
    got, ref = LOGITS[case["id"]], LOGITS[case.get("vs", "baseline")]
    if case["expect"] == "higher":
        assert got > ref + 1e-9
    elif case["expect"] == "lower":
        assert got < ref - 1e-9
    else:
        assert abs(got - ref) < 1e-9


def _random_trades(rng: np.random.Generator, n: int) -> dict:
    F = {}
    for f in FEATURES:
        if f.kind == "categorical":
            F[f.name] = rng.integers(0, len(f.levels), n)
        elif f.kind == "binary":
            F[f.name] = rng.integers(0, 2, n).astype(float)
        elif f.kind == "integer":
            F[f.name] = rng.integers(0, 12, n).astype(float)
        else:
            F[f.name] = rng.uniform(0, 1, n)
    F["ssi_age_days"] *= 1500
    F["hours_to_confirmation"] *= 96
    F["allocation_delay_hrs"] *= 48
    F["obligation_coverage_ratio"] *= 1.6
    F["cpty_fail_rate_30d"] *= 0.4
    F["security_fail_rate_30d"] *= 0.4
    F["notional_vs_cpty_median"] *= 50
    F["abs_price_deviation_bps"] *= 500
    F["market_volatility_level"] = 9 + 40 * F["market_volatility_level"]
    F["pair_history_trades_90d"] *= 20
    return F


@pytest.mark.parametrize("feature", [f.name for f in FEATURES if f.direction != 0])
def test_raising_a_risk_feature_never_lowers_the_log_odds(feature):
    rng = np.random.default_rng(zlib.crc32(feature.encode()))
    F = _random_trades(rng, 5000)
    before = prob.deterministic_logit(F, CFG)
    f = BY_NAME[feature]
    G = dict(F)
    step = rng.uniform(0, 1, 5000) * (1 if f.kind == "binary" else np.maximum(np.abs(F[feature]), 1.0))
    G[feature] = np.clip(F[feature] + f.direction * step, 0, 1 if f.kind == "binary" else None)
    after = prob.deterministic_logit(G, CFG)
    assert (after >= before - 1e-12).all()
    assert (after > before + 1e-9).any(), "the feature should matter somewhere"


@pytest.mark.parametrize("feature", ["ssi_match_status", "instruction_hour_bucket", "cpty_type", "asset_class"])
def test_leaving_the_reference_level_never_lowers_the_log_odds(feature):
    rng = np.random.default_rng(7)
    F = _random_trades(rng, 5000)
    f = BY_NAME[feature]
    F[feature] = np.full(5000, f.levels.index(f.reference_level))
    before = prob.deterministic_logit(F, CFG)
    for i, level in enumerate(f.levels):
        G = dict(F)
        G[feature] = np.full(5000, i)
        assert (prob.deterministic_logit(G, CFG) >= before - 1e-12).all(), level


def test_true_shap_is_exact_for_the_planted_logit():
    """Shapley values of the planted log-odds add up to logit minus the average logit
    when features are independent, which holds for the random trades used here."""
    rng = np.random.default_rng(3)
    F = _random_trades(rng, 20000)
    phi = prob.true_shap(F, CFG)
    total = sum(phi[name] for name in FEATURE_NAMES)
    logit = prob.deterministic_logit(F, CFG)
    # Exact for additive terms; product terms are exact under independence up to sampling noise.
    assert np.corrcoef(total, logit - logit.mean())[0, 1] > 0.999
    assert np.abs(total.mean()) < 0.02


def test_hidden_effects_have_the_right_direction():
    assert set(ROOT_CAUSES) <= set(SCENARIO_IDS)
    assert LOGITS["h1_auto_repaired"] < LOGITS["s1_mismatch"]
    assert LOGITS["h2_late_borrow"] < LOGITS["s3_shortfall_half"]
    boost = LOGITS["h4_unexplained"] - LOGITS["baseline"]
    assert boost == pytest.approx(CFG["hidden"]["unexplained_boost"])
