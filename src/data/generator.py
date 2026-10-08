"""Vectorized synthetic trade generator, one batch per business day.

Per day: draw trades, compute the 20 features, apply scenario injectors, compute the planted
fail probability, draw the label and fail reason, then update the rolling counters.

Rolling features use only information known on the trade date:
  - fail rates count trades whose settle date is in [trade date - 30 days, trade date),
    so a trade that has not settled yet has no known outcome;
  - pair history and the size median use trades with trade date in [trade date - 90 days,
    trade date), never the current day.
Randomness comes from a generator seeded by (seed, day index), so a restart from a saved
month-end state continues with exactly the data an uninterrupted run would produce.
"""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import probability as prob
from src.data import scenarios as sc
from src.data.reference_data import (
    ASSET_CLASSES,
    CPTY_TYPES,
    HOME,
    MARKETS,
    Reference,
)
from src.features.definitions import BY_NAME, FEATURE_NAMES, FEATURES, TRADE_COLUMNS

log = logging.getLogger(__name__)

FAIL_REASONS = ("ssi_problem", "shortfall", "unmatched", "other")
REASON_GROUPS = {
    "ssi_problem": ("ssi_match_status", "ssi_age_days", "ssi_mismatch_x_cross_border"),
    "shortfall": ("obligation_coverage_ratio", "shortfall_x_security_fail_rate",
                  "security_fail_rate_30d", "chain_depth", "corporate_action_in_window"),
    "unmatched": ("hours_to_confirmation", "amendment_count", "allocation_delay_hrs",
                  "instruction_hour_bucket", "abs_price_deviation_bps", "confirmation_x_overnight"),
}
REASON_BASE = {"ssi_problem": 0.05, "shortfall": 0.05, "unmatched": 0.05, "other": 0.25}

# Base-population distribution parameters (the scenario injectors add to these).
P_SSI_MISMATCH = 0.022
P_SSI_MISSING = 0.007
SSI_REVERIFY_HAZARD = 1 / 150  # per SSI record per business day
HOUR_BUCKET_P = {False: [0.08, 0.14, 0.64, 0.14], True: [0.16, 0.20, 0.48, 0.16]}  # by cross-border
BLOCK_P = {"custodian": 0.08, "broker_dealer": 0.10, "asset_manager": 0.55, "hedge_fund": 0.30,
           "corporate_treasury": 0.05}
P_CHAINED = 0.18
P_SHORTFALL = 0.10
PRICE_SIGMA_BPS = {"equity": 12.0, "corporate_bond": 16.0, "government_bond": 8.0, "etf": 10.0, "repo": 6.0}
P_OFF_MARKET = 0.01
P_CROSS_VIA_CPTY = 0.35
P_RARE_DESK = 0.03
# Trade size does not depend on asset class, so size relative to the counterparty median stays
# distinct from asset_class.
BASE_NOTIONAL = 2.0e6

_L = sc.LEVEL
REPO = _L["asset_class"]["repo"]
_BLOCK_BY_TYPE = np.array([BLOCK_P[t] for t in CPTY_TYPES])
_PRICE_SIGMA = np.array([PRICE_SIGMA_BPS[a] for a in ASSET_CLASSES])
_EPOCH = date(1970, 1, 1).toordinal()


@dataclass
class GenState:
    day: int  # next business-day index to generate
    next_trade_id: int
    ssi_verified_ord: np.ndarray  # [cpty, asset class] date ordinal of last SSI verification
    # settle-date ordinal -> (cpty [fails, settled], security [fails, settled])
    outcomes: dict = field(default_factory=dict)
    pair_days: list = field(default_factory=list)  # (trade ordinal, counts per desk * cpty)
    notional_days: list = field(default_factory=list)  # (trade ordinal, cpty ids, notional)


def init_state(ref: Reference, cfg: dict) -> GenState:
    rng = np.random.default_rng([cfg["seed"], 2])
    start = ref.cal.ords[0]
    age = rng.exponential(210.0, (ref.n_cpty, len(ASSET_CLASSES)))
    return GenState(day=0, next_trade_id=0, ssi_verified_ord=(start - age).astype(np.int64))


# ---- rolling windows ----------------------------------------------------------------------

def rolling_fail_rates(state: GenState, ord_t: int, ref: Reference, cfg: dict):
    c = np.zeros((2, ref.n_cpty), dtype=np.int64)
    s = np.zeros((2, ref.n_sec), dtype=np.int64)
    for settle_ord, (cc, ss) in state.outcomes.items():
        if ord_t - 30 <= settle_ord < ord_t:
            c += cc
            s += ss
    prior = cfg["data"]["target_fail_rate"]
    a_c, a_s = cfg["data"]["rate_prior_trades"]["cpty"], cfg["data"]["rate_prior_trades"]["security"]
    return (c[0] + a_c * prior) / (c[1] + a_c), (s[0] + a_s * prior) / (s[1] + a_s)


def rolling_pair_counts(state: GenState, ord_t: int, size: int) -> np.ndarray:
    out = np.zeros(size, dtype=np.int64)
    for trade_ord, counts in state.pair_days:
        if ord_t - 90 <= trade_ord < ord_t:
            out += counts
    return out


def group_median(groups: np.ndarray, values: np.ndarray, n_groups: int) -> np.ndarray:
    """Median of values per group id (NaN for empty groups), same as np.median per group."""
    out = np.full(n_groups, np.nan)
    if len(values) == 0:
        return out
    order = np.lexsort((values, groups))
    g, v = groups[order], values[order].astype(np.float64)
    counts = np.bincount(g, minlength=n_groups)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    has = counts > 0
    lo = starts + (counts - 1) // 2
    hi = starts + counts // 2
    out[has] = 0.5 * (v[lo[has]] + v[hi[has]])
    return out


def rolling_cpty_median(state: GenState, ord_t: int, n_cpty: int) -> np.ndarray:
    keep = [(c, v) for o, c, v in state.notional_days if ord_t - 90 <= o < ord_t]
    if not keep:
        return np.full(n_cpty, np.nan)
    cps = np.concatenate([c for c, _ in keep]).astype(np.int64)
    vals = np.concatenate([v for _, v in keep])
    return group_median(cps, vals, n_cpty)


# ---- one business day ---------------------------------------------------------------------

def generate_day(t: int, state: GenState, ref: Reference, cfg: dict, intercept: float,
                 trades_per_day: float | None = None) -> dict:
    """Generate business day t and update the state. Returns the day's columns."""
    if t != state.day:
        raise ValueError(f"state is at day {state.day}, asked for day {t}")
    cal, data, hidden = ref.cal, cfg["data"], cfg["hidden"]
    rng = np.random.default_rng([cfg["seed"], 1, t])
    ord_t = int(cal.ords[t])
    tpd = data["trades_per_day"] if trades_per_day is None else trades_per_day

    cpty_rate, sec_rate = rolling_fail_rates(state, ord_t, ref, cfg)
    pair_counts = rolling_pair_counts(state, ord_t, ref.n_desks * ref.n_cpty)
    cpty_median = rolling_cpty_median(state, ord_t, ref.n_cpty)

    # Who trades what.
    n = max(1, int(rng.poisson(tpd * cal.volume_mult[t])))
    active = ref.cpty_onboard_day <= t
    w = ref.cpty_activity * active
    cpty = rng.choice(ref.n_cpty, size=n, p=w / w.sum())
    cum = np.cumsum(ref.desk_affinity[cpty], axis=1)
    desk = np.minimum((rng.random(n)[:, None] > cum).sum(axis=1), ref.n_desks - 1)
    rare = rng.random(n) < P_RARE_DESK
    desk = np.where(rare, rng.integers(0, ref.n_desks, n), desk)
    ctype = ref.cpty_type[cpty]
    cls_cum = np.cumsum(ref.class_probs_by_type, axis=1)[ctype]
    cls = np.minimum((rng.random(n)[:, None] > cls_cum).sum(axis=1), len(ASSET_CLASSES) - 1)
    sec = np.zeros(n, dtype=np.int64)
    for a in range(len(ASSET_CLASSES)):
        idx = np.flatnonzero(cls == a)
        if len(idx):
            sec[idx] = rng.choice(ref.sec_by_class[a], size=len(idx), p=ref.sec_probs_by_class[a])

    # Routing, instruction time, and settlement date.
    sec_mkt, cp_mkt = ref.sec_market[sec], ref.cpty_market[cpty]
    via_cpty = (sec_mkt == HOME) & (cp_mkt != HOME) & (rng.random(n) < P_CROSS_VIA_CPTY)
    market = np.where(sec_mkt != HOME, sec_mkt, np.where(via_cpty, cp_mkt, HOME)).astype(np.int64)
    u = rng.random(n)
    bucket = np.where(
        market != HOME,
        np.searchsorted(np.cumsum(HOUR_BUCKET_P[True]), u, side="right"),
        np.searchsorted(np.cumsum(HOUR_BUCKET_P[False]), u, side="right"),
    ).clip(0, 3)
    sc.inject_routing(cfg, rng, market, bucket, len(MARKETS))
    cross = (market != HOME).astype(np.int8)
    lag = np.where(cross > 0, 2, np.where((cls == REPO) & (rng.random(n) < 0.6), 0, 1))
    settle_ord = cal.settle_ord[market, lag, t]
    holiday = cal.holiday_in_window[market, lag, t]
    period_end = cal.settle_is_period_end[market, lag, t]

    # Base feature values.
    z = ref.cpty_latent[cpty]
    zs = ref.sec_latent[sec]
    cmult = np.exp(0.35 * z)
    drift_start = date.fromisoformat(cfg["regimes"]["ssi_drift_start"]).toordinal()
    ramp = np.clip((ord_t - drift_start) / max(1, cal.ords[-1] - drift_start), 0.0, 1.0)
    drift_on = ref.cpty_ssi_drift[cpty] & (ord_t >= drift_start)
    drift_mult = np.where(drift_on, 1.0 + (cfg["regimes"]["ssi_drift_max_multiplier"] - 1.0) * ramp, 1.0)
    p_mis = P_SSI_MISMATCH * cmult * drift_mult
    p_miss = P_SSI_MISSING * cmult
    u = rng.random(n)
    status = np.where(u < p_mis, sc.MISMATCH, np.where(u < p_mis + p_miss, sc.MISSING, sc.MATCHED))
    F = {
        "ssi_match_status": status.astype(np.int64),
        "ssi_age_days": (ord_t - state.ssi_verified_ord[cpty, cls]).astype(np.float64),
        "instruction_hour_bucket": bucket.astype(np.int64),
        "hours_to_confirmation": np.clip(rng.lognormal(np.log(1.8) + 0.25 * z, 0.95), 0.05, 96.0),
        "amendment_count": rng.poisson(0.22 * np.exp(0.2 * z)).astype(np.int64),
        "allocation_delay_hrs": np.where(
            rng.random(n) < _BLOCK_BY_TYPE[ctype], np.clip(rng.lognormal(np.log(1.5), 0.9, n), 0.05, 96.0), 0.0
        ),
        "chain_depth": np.where(rng.random(n) < P_CHAINED, 1 + rng.poisson(0.4, n), 0).astype(np.int64),
        "cpty_fail_rate_30d": cpty_rate[cpty],
        "cpty_type": ctype.astype(np.int64),
        "pair_history_trades_90d": pair_counts[desk * ref.n_cpty + cpty],
        "security_fail_rate_30d": sec_rate[sec],
        "asset_class": cls.astype(np.int64),
        "is_cross_border": cross,
        "market_volatility_level": np.full(n, float(cal.vol[t])),
        "holiday_in_settlement_window": holiday.astype(np.int8),
        "is_period_end": period_end.astype(np.int8),
    }
    p_sf = P_SHORTFALL * np.exp(0.3 * zs) * (1.5 if cal.vol_spike[t] else 1.0)
    short = rng.random(n) < p_sf
    at_one = rng.random(n) < 0.55
    F["obligation_coverage_ratio"] = np.where(
        short, rng.beta(5.0, 1.6, n), np.where(at_one, 1.0, 1.0 + rng.exponential(0.15, n))
    ).clip(0.0, 2.0)
    vol_boost = 1.0 + 0.03 * max(0.0, float(cal.vol[t]) - 16.0)
    dev = np.abs(rng.standard_normal(n)) * _PRICE_SIGMA[cls] * vol_boost
    F["abs_price_deviation_bps"] = np.where(rng.random(n) < P_OFF_MARKET, rng.uniform(80, 400, n), dev)
    period = ref.sec_ca_period[sec].astype(np.int64)
    safe = np.where(period > 0, period, 1)
    next_event = t + (ref.sec_ca_phase[sec] - t) % safe
    F["corporate_action_in_window"] = ((period > 0) & (next_event <= t + lag)).astype(np.int8)

    size_mult = sc.inject_features(cfg, rng, F)
    notional = (
        BASE_NOTIONAL * ref.cpty_median_notional[cpty] * rng.lognormal(0.0, 1.0, n) * size_mult
    ).astype(np.float32)
    with np.errstate(invalid="ignore"):
        F["notional_vs_cpty_median"] = notional.astype(np.float64) / cpty_median[cpty]

    # Hidden events and the planted probability.
    auto_repair = rng.random(n) < hidden["auto_repair_prob"]
    late_borrow = rng.random(n) < hidden["late_borrow_prob"]
    onboard = ref.cpty_onboard_day[cpty]
    ctx = {
        "vol_spike": np.full(n, bool(cal.vol_spike[t])),
        "trade_month_end": np.full(n, bool(cal.trade_month_end[t])),
        "holiday_week": cal.holiday_week[market, t],
        "drift_active": drift_on,
        "cold_start": (onboard > 0) & (t - onboard < sc.COLD_START_DAYS),
        "auto_repair": auto_repair,
        "late_borrow": late_borrow,
        "cpty_latent": z,
    }
    members = sc.membership(F, ctx)
    unexplained = ~sc.any_root_cause(members) & (rng.random(n) < hidden["unexplained_rate"])
    members["H4"] = unexplained
    logit, terms = prob.planted_logit(
        F, cfg, intercept, repaired=members["H1"], borrowed=members["H2"], unexplained=unexplained,
        cpty_latent=z, sec_latent=zs, noise=rng.standard_normal(n),
    )
    failed = rng.random(n) < prob.sigmoid(logit)

    # Fail reason, in proportion to which groups of terms pushed this trade up the most.
    weights = []
    grouped = set()
    for reason in FAIL_REASONS[:3]:
        cols = REASON_GROUPS[reason]
        grouped |= set(cols)
        excess = sum(np.maximum(terms[c] - terms[c].mean(), 0.0) for c in cols)
        weights.append(excess + REASON_BASE[reason])
    other = sum(np.maximum(terms[c] - terms[c].mean(), 0.0) for c in prob.TERMS if c not in grouped)
    other = other + np.maximum(hidden["cpty_latent_weight"] * z, 0.0) + hidden["unexplained_boost"] * unexplained
    weights.append(other + REASON_BASE["other"])
    W = np.stack(weights, axis=1)
    pick = (rng.random(n)[:, None] > np.cumsum(W / W.sum(axis=1, keepdims=True), axis=1)).sum(axis=1)
    reason = np.where(failed, np.minimum(pick, 3), -1)

    # Data-quality gaps: the label above used the true values.
    observed = dict(F)
    for name, share in data["missing_share"].items():
        gap = rng.random(n) < share
        observed[name] = np.where(gap, np.nan, np.asarray(F[name], dtype=np.float64))

    # Update rolling state with what becomes known later.
    for s_ord in np.unique(settle_ord):
        m = settle_ord == s_ord
        cc = np.stack([np.bincount(cpty[m], weights=failed[m].astype(np.float64), minlength=ref.n_cpty),
                       np.bincount(cpty[m], minlength=ref.n_cpty)]).astype(np.int64)
        ss = np.stack([np.bincount(sec[m], weights=failed[m].astype(np.float64), minlength=ref.n_sec),
                       np.bincount(sec[m], minlength=ref.n_sec)]).astype(np.int64)
        prev = state.outcomes.get(int(s_ord))
        state.outcomes[int(s_ord)] = (cc, ss) if prev is None else (prev[0] + cc, prev[1] + ss)
    state.outcomes = {o: v for o, v in state.outcomes.items() if o >= ord_t - 31}
    state.pair_days.append(
        (ord_t, np.bincount(desk * ref.n_cpty + cpty, minlength=ref.n_desks * ref.n_cpty).astype(np.int32))
    )
    state.pair_days = [x for x in state.pair_days if x[0] >= ord_t - 90]
    state.notional_days.append((ord_t, cpty.astype(np.int16), notional))
    state.notional_days = [x for x in state.notional_days if x[0] >= ord_t - 90]
    reverify = rng.random(state.ssi_verified_ord.shape) < SSI_REVERIFY_HAZARD
    state.ssi_verified_ord = np.where(reverify, ord_t, state.ssi_verified_ord)
    trade_id = state.next_trade_id + np.arange(n, dtype=np.int64)
    state.next_trade_id += n
    state.day = t + 1

    cols = {
        "trade_id": trade_id,
        "trade_date": np.full(n, ord_t - _EPOCH),
        "settle_date": settle_ord - _EPOCH,
        "desk_id": desk.astype(np.int16),
        "cpty_id": cpty.astype(np.int16),
        "sec_id": sec.astype(np.int16),
        "notional": notional,
        "failed": failed.astype(np.int8),
        "fail_reason": reason.astype(np.int8),
        "scenario_mask": sc.to_mask(members),
    }
    cols.update({name: observed[name] for name in FEATURE_NAMES})
    return cols


# ---- frames -------------------------------------------------------------------------------

def to_frame(days: list[dict]) -> pd.DataFrame:
    cols = {k: np.concatenate([d[k] for d in days]) for k in days[0]}
    out = {}
    for name in TRADE_COLUMNS:
        v = cols[name]
        if name in ("trade_date", "settle_date"):
            out[name] = v.astype("datetime64[D]")
        elif name == "fail_reason":
            out[name] = pd.Categorical.from_codes(v.astype(np.int64), categories=list(FAIL_REASONS))
        elif name in BY_NAME:
            f = BY_NAME[name]
            if f.kind == "categorical":
                out[name] = pd.Categorical.from_codes(v.astype(np.int64), categories=list(f.levels))
            else:
                out[name] = v.astype(f.dtype)
        else:
            out[name] = v
    return pd.DataFrame(out)


def frame_to_arrays(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Feature arrays in the generator's representation (categorical codes)."""
    out = {}
    for f in FEATURES:
        if f.kind == "categorical":
            out[f.name] = pd.Categorical(df[f.name], categories=list(f.levels)).codes.astype(np.int64)
        else:
            out[f.name] = df[f.name].to_numpy(dtype=np.float64)
    return out


# ---- intercept calibration ------------------------------------------------------------------

def pilot_fail_rate(ref: Reference, cfg: dict, intercept: float, trades_per_day: float) -> float:
    state = init_state(ref, cfg)
    fails = total = 0
    for t in range(len(ref.cal.bdays)):
        cols = generate_day(t, state, ref, cfg, intercept, trades_per_day)
        fails += int(cols["failed"].sum())
        total += len(cols["failed"])
    return fails / total


def calibrate_intercept(ref: Reference, cfg: dict) -> dict:
    """Bisection on the intercept so a pilot run over the same days hits the target rate.
    The fail rate rises monotonically with the intercept, because every random draw is
    fixed by the day seed and does not depend on outcomes."""
    c = cfg["data"]["intercept_calibration"]
    target = cfg["data"]["target_fail_rate"]
    tpd = min(c["trades_per_day"], cfg["data"]["trades_per_day"])
    lo, hi = c["bracket"]
    history = []
    mid = 0.5 * (lo + hi)
    for _ in range(c["max_iter"]):
        mid = 0.5 * (lo + hi)
        rate = pilot_fail_rate(ref, cfg, mid, tpd)
        history.append({"intercept": mid, "fail_rate": rate})
        log.info("calibration: intercept %.4f -> pilot fail rate %.4f", mid, rate)
        if abs(rate - target) <= c["tolerance"]:
            break
        if rate > target:
            hi = mid
        else:
            lo = mid
    return {"intercept": mid, "target": target, "pilot_trades_per_day": tpd, "history": history}


# ---- months and checkpoints -----------------------------------------------------------------

def month_plan(ref: Reference) -> list[tuple[str, list[int]]]:
    plan: dict[str, list[int]] = {}
    for t, label in enumerate(ref.cal.month_label):
        plan.setdefault(label, []).append(t)
    return list(plan.items())


def trades_path(label: str) -> str:
    return f"gen/trades/trades_{label}.parquet"


def state_path(label: str) -> str:
    return f"gen/state/state_{label}.pkl"


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, engine="pyarrow", compression="zstd", index=False)


def save_state(state: GenState, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(state, fh, protocol=5)


def load_state(path: Path) -> GenState:
    with open(path, "rb") as fh:
        return pickle.load(fh)


def generate_months(ref: Reference, cfg: dict, intercept: float, workdir: Path, ckpt, stage: str,
                    progress=None) -> list[str]:
    """Generate every month, checkpointing the month's Parquet file and the generator state
    together. Months already saved are skipped and generation resumes from the last state."""
    plan = month_plan(ref)
    outputs, last_done, state = [], None, None
    for label, days in plan:
        outputs.append(trades_path(label))
        if ckpt.unit_done(stage, label):
            last_done = label
            continue
        if state is None:
            state = load_state(workdir / state_path(last_done)) if last_done else init_state(ref, cfg)
        frame = to_frame([generate_day(t, state, ref, cfg, intercept) for t in days])
        write_parquet(frame, workdir / trades_path(label))
        save_state(state, workdir / state_path(label))
        delete = [state_path(last_done)] if last_done else []
        ckpt.save_unit(stage, label, [trades_path(label), state_path(label)], delete=delete)
        log.info("month %s: %d trades, fail rate %.4f", label, len(frame), frame["failed"].mean())
        if progress:
            progress(month=label, trades=len(frame), fail_rate=round(float(frame["failed"].mean()), 5))
        last_done = label
    outputs.append(state_path(plan[-1][0]))
    return outputs


def effect_weights(cfg: dict, intercept: float) -> dict:
    return {
        "note": "True planted effects of the synthetic generator. For validation only, never model inputs.",
        "intercept": intercept,
        "effects": cfg["effects"],
        "interactions": cfg["interactions"],
        "hidden": cfg["hidden"],
        "strength": {f.name: f.strength for f in FEATURES},
        "interaction_features": prob.INTERACTIONS,
    }


def write_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=str))
