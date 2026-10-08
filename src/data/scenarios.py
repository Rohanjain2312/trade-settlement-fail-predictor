"""Scenario catalog, injectors, and the scenario bitmask.

The mask is used only for coverage reports and per-scenario evaluation, never as a model
input. Membership is decided from each trade's true situation (feature values before any
are blanked as missing, plus hidden events), not from whether an injector touched it.
"""

from __future__ import annotations

import numpy as np

from src.features.definitions import BY_NAME

ROOT_CAUSES = ("S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8")
REGIMES = ("R1", "R2", "R3", "R4", "R5")
HARD_NEGATIVES = ("H1", "H2", "H3", "H4")
SCENARIO_IDS = ROOT_CAUSES + REGIMES + HARD_NEGATIVES
BIT = {sid: i for i, sid in enumerate(SCENARIO_IDS)}

LEVEL = {
    name: {lv: i for i, lv in enumerate(BY_NAME[name].levels)}
    for name in ("ssi_match_status", "instruction_hour_bucket", "cpty_type", "asset_class")
}
MATCHED, MISMATCH, MISSING = (LEVEL["ssi_match_status"][k] for k in ("matched", "mismatch", "missing"))
OVERNIGHT = LEVEL["instruction_hour_bucket"]["overnight"]
LATE = LEVEL["instruction_hour_bucket"]["late"]

STALE_SSI_DAYS = 365
SLOW_CONFIRMATION_HOURS = 12
LONG_ALLOCATION_HOURS = 6
LARGE_SIZE = 3.0
WEAK_CPTY_LATENT = 1.0
COLD_START_DAYS = 60


def scenario_names(cfg: dict) -> dict[str, str]:
    sc = cfg["scenarios"]
    out = {}
    for group in ("root_causes", "regimes", "hard_negatives"):
        out.update({sid: v["name"] for sid, v in sc[group].items()})
    return out


def quota(cfg: dict, sid: str) -> float:
    return float(cfg["scenarios"]["root_causes"][sid]["quota"])


def _pick(rng: np.random.Generator, n: int, share: float) -> np.ndarray:
    return rng.random(n) < share


def inject_routing(cfg: dict, rng: np.random.Generator, market: np.ndarray, bucket: np.ndarray,
                   n_markets: int) -> np.ndarray:
    """S6: route a share of trades cross-border with a late or overnight instruction.
    Runs before settlement dates are computed, so the holiday flag follows the new market."""
    hit = _pick(rng, len(market), quota(cfg, "S6"))
    new_market = rng.integers(1, n_markets, len(market))
    new_bucket = np.where(rng.random(len(market)) < 0.5, LATE, OVERNIGHT)
    market[hit & (market == 0)] = new_market[hit & (market == 0)]
    bucket[hit] = new_bucket[hit]
    return hit


def inject_features(cfg: dict, rng: np.random.Generator, F: dict) -> np.ndarray:
    """S1 to S5, S7, S8 (modifying F in place). Returns the trade-size multiplier for the
    extreme-size edge case."""
    n = len(F["ssi_match_status"])
    hit = _pick(rng, n, quota(cfg, "S1"))
    F["ssi_match_status"][hit] = MISMATCH

    hit = _pick(rng, n, quota(cfg, "S2"))
    missing = rng.random(n) < 0.5
    stale_age = rng.uniform(STALE_SSI_DAYS + 30, 1000, n)
    F["ssi_match_status"][hit & missing] = MISSING
    F["ssi_age_days"][hit & ~missing] = stale_age[hit & ~missing]

    hit = _pick(rng, n, quota(cfg, "S3"))
    F["obligation_coverage_ratio"][hit] = rng.beta(2.0, 2.5, n)[hit]

    hit = _pick(rng, n, quota(cfg, "S4"))
    F["hours_to_confirmation"][hit] = rng.uniform(SLOW_CONFIRMATION_HOURS + 2, 72, n)[hit]

    hit = _pick(rng, n, quota(cfg, "S5"))
    F["chain_depth"][hit] = np.minimum(2 + rng.poisson(1.2, n), 8)[hit]

    hit = _pick(rng, n, quota(cfg, "S7"))
    F["corporate_action_in_window"][hit] = 1

    hit = _pick(rng, n, quota(cfg, "S8"))
    amend = np.minimum(2 + rng.poisson(2.0, n), 10)
    delay = rng.uniform(LONG_ALLOCATION_HOURS + 1, 48, n)
    with_delay = rng.random(n) < 0.5
    F["amendment_count"][hit] = amend[hit]
    F["allocation_delay_hrs"][hit & with_delay] = delay[hit & with_delay]

    hit = _pick(rng, n, cfg["scenarios"]["edge_cases"]["extreme_size_quota"])
    return np.where(hit, rng.uniform(20, 80, n), 1.0)


def membership(F: dict, ctx: dict) -> dict[str, np.ndarray]:
    """Boolean membership per scenario id. ctx carries calendar flags and hidden events."""
    st = F["ssi_match_status"]
    m = {
        "S1": st == MISMATCH,
        "S2": (st == MISSING) | (F["ssi_age_days"] > STALE_SSI_DAYS),
        "S3": F["obligation_coverage_ratio"] < 1.0,
        "S4": F["hours_to_confirmation"] > SLOW_CONFIRMATION_HOURS,
        "S5": F["chain_depth"] >= 2,
        "S6": (F["is_cross_border"] > 0)
        & ((F["holiday_in_settlement_window"] > 0) | np.isin(F["instruction_hour_bucket"], [LATE, OVERNIGHT])),
        "S7": F["corporate_action_in_window"] > 0,
        "S8": (F["amendment_count"] >= 2) | (F["allocation_delay_hrs"] > LONG_ALLOCATION_HOURS),
        "R2": ctx["vol_spike"],
        "R3": (F["is_period_end"] > 0) | ctx["trade_month_end"],
        "R4": ctx["holiday_week"],
        "R5": ctx["drift_active"] | ctx["cold_start"],
        "H1": (st != MATCHED) & ctx["auto_repair"],
        "H2": (F["obligation_coverage_ratio"] < 1.0) & ctx["late_borrow"],
        "H3": (np.nan_to_num(F["notional_vs_cpty_median"]) > LARGE_SIZE) & (ctx["cpty_latent"] > WEAK_CPTY_LATENT),
    }
    m["R1"] = ~(m["R2"] | m["R3"] | m["R4"])
    return m


def any_root_cause(m: dict) -> np.ndarray:
    out = np.zeros_like(m["S1"])
    for sid in ROOT_CAUSES:
        out |= m[sid]
    return out


def to_mask(m: dict[str, np.ndarray]) -> np.ndarray:
    n = len(next(iter(m.values())))
    mask = np.zeros(n, dtype=np.int32)
    for sid, member in m.items():
        mask |= member.astype(np.int32) << BIT[sid]
    return mask


def decode(mask) -> dict[str, np.ndarray]:
    mask = np.asarray(mask, dtype=np.int64)
    return {sid: ((mask >> BIT[sid]) & 1).astype(bool) for sid in SCENARIO_IDS}


def combination_members(mask, bucket_codes) -> dict[str, np.ndarray]:
    d = decode(mask)
    return {
        "S1+S6": d["S1"] & d["S6"],
        "S3+S5": d["S3"] & d["S5"],
        "S4+overnight": d["S4"] & (np.asarray(bucket_codes) == OVERNIGHT),
    }
