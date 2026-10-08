"""Fictional reference data: counterparties, securities, desks, market calendars, and the
volatility path. Everything is drawn from the seed, so it is identical on every run.

Hidden truth (counterparty latent risk, security fail propensity) lives here too, but it is
published only to truth/ and is never a model input.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd

from src.features.definitions import BY_NAME

MARKETS = ("HOME", "MKT_A", "MKT_B", "MKT_C")
HOME = 0
CPTY_TYPES = BY_NAME["cpty_type"].levels
ASSET_CLASSES = BY_NAME["asset_class"].levels

# Fictional fixed-date holidays for the three foreign markets. Clusters create holiday weeks.
FOREIGN_HOLIDAYS = {
    "MKT_A": [(1, 1), (4, 18), (4, 21), (5, 1), (8, 15), (12, 24), (12, 25), (12, 26), (12, 31)],
    "MKT_B": [(1, 1), (1, 2), (2, 10), (2, 11), (2, 12), (5, 1), (10, 1), (10, 2), (10, 3), (12, 25)],
    "MKT_C": [(1, 1), (1, 2), (1, 3), (3, 20), (4, 29), (5, 3), (5, 5), (7, 15), (9, 23), (11, 3),
              (11, 24), (12, 31)],
}

# Counterparty mix. Corporate treasuries are the rare category (about 0.5% of trades).
CPTY_TYPE_SHARE = {"custodian": 0.14, "broker_dealer": 0.36, "asset_manager": 0.26,
                   "hedge_fund": 0.20, "corporate_treasury": 0.04}
CPTY_TYPE_ACTIVITY = {"custodian": 1.0, "broker_dealer": 1.0, "asset_manager": 1.0,
                      "hedge_fund": 1.0, "corporate_treasury": 0.12}
CPTY_TYPE_LATENT_MEAN = {"custodian": -0.3, "broker_dealer": 0.0, "asset_manager": 0.0,
                         "hedge_fund": 0.3, "corporate_treasury": 0.3}
ASSET_CLASS_SHARE = {"equity": 0.45, "corporate_bond": 0.20, "government_bond": 0.12,
                     "etf": 0.15, "repo": 0.08}
ASSET_CLASS_PROPENSITY_MEAN = {"equity": 0.0, "corporate_bond": 0.2, "government_bond": -0.3,
                               "etf": -0.1, "repo": 0.1}
# Mild preference of each counterparty type for asset classes (rows: CPTY_TYPES order).
CLASS_TILT = {
    "custodian": {},
    "broker_dealer": {},
    "asset_manager": {"etf": 1.3, "corporate_bond": 1.2},
    "hedge_fund": {"equity": 1.4, "etf": 1.2, "government_bond": 0.7},
    "corporate_treasury": {"government_bond": 2.5, "repo": 2.5, "equity": 0.4},
}


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    d = date(year, month, 1)
    d += timedelta(days=(weekday - d.weekday()) % 7)
    return d + timedelta(weeks=n - 1)


def _last_weekday(year: int, month: int, weekday: int) -> date:
    d = date(year + month // 12, month % 12 + 1, 1) - timedelta(days=1)
    return d - timedelta(days=(d.weekday() - weekday) % 7)


def _observed(d: date) -> date:
    if d.weekday() == 5:
        return d - timedelta(days=1)
    if d.weekday() == 6:
        return d + timedelta(days=1)
    return d


def home_holidays(year: int) -> set[date]:
    """Rule-based home-market holidays (fictional calendar)."""
    return {
        _observed(date(year, 1, 1)),
        _nth_weekday(year, 1, 0, 3),
        _nth_weekday(year, 2, 0, 3),
        _nth_weekday(year, 4, 4, 1),
        _last_weekday(year, 5, 0),
        _observed(date(year, 6, 19)),
        _observed(date(year, 7, 4)),
        _nth_weekday(year, 9, 0, 1),
        _nth_weekday(year, 11, 3, 4),
        _observed(date(year, 12, 25)),
    }


def market_holidays(market: str, years: range) -> set[date]:
    if market == "HOME":
        return set().union(*(home_holidays(y) for y in years))
    return {date(y, m, d) for y in years for m, d in FOREIGN_HOLIDAYS[market]}


def home_business_days(start: date, n_days: int) -> list[date]:
    """The first n_days home-market business days on or after start."""
    years = range(start.year, start.year + n_days // 200 + 3)
    hol = market_holidays("HOME", years)
    out, d = [], start
    while len(out) < n_days:
        if d.weekday() < 5 and d not in hol:
            out.append(d)
        d += timedelta(days=1)
    return out


@dataclass
class Calendar:
    bdays: list[date]  # trade dates (home business days)
    ords: np.ndarray  # date ordinals of bdays
    month_index: np.ndarray  # 1-based month number of each trade date
    month_label: list[str]  # "YYYY-MM" of each trade date
    settle_ord: np.ndarray  # [market, lag, t] settlement date ordinal
    holiday_in_window: np.ndarray  # [market, lag, t] holiday in [trade date, settle date)
    settle_is_period_end: np.ndarray  # [market, lag, t]
    holiday_week: np.ndarray  # [market, t] trade-date week has a holiday in HOME or market
    trade_month_end: np.ndarray  # [t] trade date is one of the last 2 business days of a month
    vol: np.ndarray  # [t] volatility index level
    vol_spike: np.ndarray  # [t] inside a volatility spike window (regime R2)
    volume_mult: np.ndarray  # [t] relative trading volume


def build_calendar(cfg: dict) -> Calendar:
    data, reg = cfg["data"], cfg["regimes"]
    start = date.fromisoformat(data["start_date"])
    n = data["n_days"]
    bdays = home_business_days(start, n)
    years = range(start.year, bdays[-1].year + 2)
    hol = {m: market_holidays(m, years) for m in MARKETS}
    home_hol = hol["HOME"]

    # All home business days through 45 days past the horizon, for settlement and month ends.
    horizon_end = bdays[-1] + timedelta(days=45)
    ext, d = [], start
    while d <= horizon_end:
        if d.weekday() < 5 and d not in home_hol:
            ext.append(d)
        d += timedelta(days=1)
    last_of_month = {}
    for d in ext:
        last_of_month[(d.year, d.month)] = d  # ext is sorted, so the last write wins
    period_end_days = set(last_of_month.values())

    settle_ord = np.zeros((len(MARKETS), 3, n), dtype=np.int64)
    holiday_flag = np.zeros((len(MARKETS), 3, n), dtype=np.int8)
    period_end = np.zeros((len(MARKETS), 3, n), dtype=np.int8)
    holiday_week = np.zeros((len(MARKETS), n), dtype=bool)
    for mi, m in enumerate(MARKETS):
        closed = home_hol | hol[m]

        def is_open(day: date, closed=closed) -> bool:
            return day.weekday() < 5 and day not in closed

        for t, td in enumerate(bdays):
            monday = td - timedelta(days=td.weekday())
            holiday_week[mi, t] = any(
                (monday + timedelta(days=k)) in closed for k in range(5)
            )
            for lag in range(3):
                s, steps = td, 0
                while not is_open(s) or steps < lag:
                    s += timedelta(days=1)
                    if is_open(s):
                        steps += 1
                    if steps >= lag and is_open(s):
                        break
                window = (td + timedelta(days=k) for k in range((s - td).days))
                holiday_flag[mi, lag, t] = any(w.weekday() < 5 and w in closed for w in window)
                settle_ord[mi, lag, t] = s.toordinal()
                period_end[mi, lag, t] = s in period_end_days

    months = [f"{d.year:04d}-{d.month:02d}" for d in bdays]
    uniq = list(dict.fromkeys(months))
    month_index = np.array([uniq.index(m) + 1 for m in months], dtype=np.int16)
    last_two = set()
    for last in last_of_month.values():
        i = ext.index(last)
        last_two |= {ext[i], ext[i - 1]} if i > 0 else {ext[i]}
    trade_month_end = np.array([d in last_two for d in bdays])

    vol, spike = _volatility_path(cfg, bdays)
    volume = 1.0 + reg["volume_surge"] * spike + reg["period_end_volume"] * trade_month_end
    return Calendar(
        bdays=bdays,
        ords=np.array([d.toordinal() for d in bdays], dtype=np.int64),
        month_index=month_index,
        month_label=months,
        settle_ord=settle_ord,
        holiday_in_window=holiday_flag,
        settle_is_period_end=period_end,
        holiday_week=holiday_week,
        trade_month_end=trade_month_end,
        vol=vol.astype(np.float32),
        vol_spike=spike > 0.15,
        volume_mult=volume,
    )


def _volatility_path(cfg: dict, bdays: list[date]) -> tuple[np.ndarray, np.ndarray]:
    """Mean-reverting volatility index with configured spikes. Returns level and spike shape."""
    reg = cfg["regimes"]
    rng = np.random.default_rng([cfg["seed"], 7])
    n = len(bdays)
    base = reg["vol_baseline"]
    level = np.empty(n)
    x = base
    for t in range(n):
        x = x + 0.08 * (base - x) + 0.9 * rng.standard_normal()
        level[t] = x
    shape = np.zeros(n)
    index = {d: t for t, d in enumerate(bdays)}
    for sp in reg["vol_spikes"]:
        d0 = date.fromisoformat(sp["start"])
        t0 = next((index[d] for d in bdays if d >= d0), None)
        if t0 is None:
            continue
        length = sp["business_days"]
        k = np.arange(length)
        bump = np.sin(np.pi * (k + 0.5) / length) ** 1.5
        end = min(n, t0 + length)
        shape[t0:end] = np.maximum(shape[t0:end], bump[: end - t0])
        level[t0:end] += (sp["peak"] - base) * bump[: end - t0]
    return np.clip(level, 9.0, None), shape


@dataclass
class Reference:
    cal: Calendar
    n_cpty: int
    n_sec: int
    n_desks: int
    cpty_type: np.ndarray  # int codes into CPTY_TYPES
    cpty_activity: np.ndarray  # sampling weights
    cpty_market: np.ndarray  # 0 for home counterparties
    cpty_onboard_day: np.ndarray  # first business day index the counterparty trades
    cpty_ssi_drift: np.ndarray  # bool, SSI quality degrades over time (regime R5)
    cpty_median_notional: np.ndarray
    cpty_latent: np.ndarray  # hidden risk, truth only
    desk_affinity: np.ndarray  # [cpty, desk] probabilities
    sec_class: np.ndarray  # int codes into ASSET_CLASSES
    sec_popularity: np.ndarray
    sec_market: np.ndarray
    sec_ca_period: np.ndarray  # business days between corporate actions (0 = none)
    sec_ca_phase: np.ndarray
    sec_latent: np.ndarray  # hidden fail propensity, truth only
    class_probs_by_type: np.ndarray  # [cpty type, asset class]
    sec_by_class: list[np.ndarray]  # security ids of each class
    sec_probs_by_class: list[np.ndarray]


def build_reference(cfg: dict) -> Reference:
    rng = np.random.default_rng([cfg["seed"], 0])
    rc = cfg["reference"]
    n_cpty, n_sec, n_desks = rc["n_counterparties"], rc["n_securities"], cfg["data"]["n_desks"]
    cal = build_calendar(cfg)
    n_days = len(cal.bdays)

    type_p = np.array([CPTY_TYPE_SHARE[t] for t in CPTY_TYPES])
    cpty_type = rng.choice(len(CPTY_TYPES), size=n_cpty, p=type_p / type_p.sum())
    activity = rng.lognormal(0.0, 1.1, n_cpty) * np.array(
        [CPTY_TYPE_ACTIVITY[CPTY_TYPES[c]] for c in cpty_type]
    )
    latent = rng.normal(0.0, 1.0, n_cpty) + np.array(
        [CPTY_TYPE_LATENT_MEAN[CPTY_TYPES[c]] for c in cpty_type]
    )
    is_foreign = rng.random(n_cpty) < 0.3
    cpty_market = np.where(is_foreign, rng.integers(1, len(MARKETS), n_cpty), HOME)
    late = rng.random(n_cpty) < rc["late_onboard_share"]
    onboard = np.where(late, rng.integers(20, max(21, n_days - 10), n_cpty), 0)
    drift = rng.random(n_cpty) < rc["ssi_drift_share"]
    median_notional = rng.lognormal(0.0, 0.7, n_cpty)

    desk_weight = rng.dirichlet(np.ones(n_desks) * 2.0)
    affinity = np.zeros((n_cpty, n_desks))
    for c in range(n_cpty):
        k = rng.integers(1, 7)
        desks = rng.choice(n_desks, size=k, replace=False, p=desk_weight)
        affinity[c, desks] = rng.dirichlet(np.ones(k))

    class_p = np.array([ASSET_CLASS_SHARE[a] for a in ASSET_CLASSES])
    sec_class = rng.choice(len(ASSET_CLASSES), size=n_sec, p=class_p / class_p.sum())
    popularity = rng.lognormal(0.0, 1.2, n_sec)
    sec_market = np.where(rng.random(n_sec) < 0.15, rng.integers(1, len(MARKETS), n_sec), HOME)
    sec_latent = rng.normal(0.0, 1.0, n_sec) + np.array(
        [ASSET_CLASS_PROPENSITY_MEAN[ASSET_CLASSES[a]] for a in sec_class]
    )
    period = np.zeros(n_sec, dtype=np.int16)
    for i, a in enumerate(sec_class):
        name = ASSET_CLASSES[a]
        if name == "equity" and rng.random() < 0.7:
            period[i] = 63
        elif name in ("corporate_bond", "government_bond"):
            period[i] = 126
        elif name == "etf":
            period[i] = 21 if rng.random() < 0.5 else 63
    phase = np.array([rng.integers(0, p) if p else 0 for p in period], dtype=np.int16)

    # Class choice per counterparty type, then a security within the class by popularity.
    class_mass = np.array([popularity[sec_class == a].sum() for a in range(len(ASSET_CLASSES))])
    probs_by_type = np.zeros((len(CPTY_TYPES), len(ASSET_CLASSES)))
    for ti, tname in enumerate(CPTY_TYPES):
        tilt = np.array([CLASS_TILT[tname].get(a, 1.0) for a in ASSET_CLASSES])
        w = class_mass * tilt
        probs_by_type[ti] = w / w.sum()
    sec_by_class = [np.flatnonzero(sec_class == a) for a in range(len(ASSET_CLASSES))]
    sec_probs = [popularity[ids] / popularity[ids].sum() for ids in sec_by_class]

    return Reference(
        cal=cal, n_cpty=n_cpty, n_sec=n_sec, n_desks=n_desks,
        cpty_type=cpty_type.astype(np.int8), cpty_activity=activity,
        cpty_market=cpty_market.astype(np.int8), cpty_onboard_day=onboard.astype(np.int32),
        cpty_ssi_drift=drift, cpty_median_notional=median_notional, cpty_latent=latent,
        desk_affinity=affinity, sec_class=sec_class.astype(np.int8), sec_popularity=popularity,
        sec_market=sec_market.astype(np.int8), sec_ca_period=period, sec_ca_phase=phase,
        sec_latent=sec_latent, class_probs_by_type=probs_by_type, sec_by_class=sec_by_class,
        sec_probs_by_class=sec_probs,
    )


def reference_tables(ref: Reference) -> dict[str, pd.DataFrame]:
    """Tables published to the dataset repo. Truth tables go to truth/, never to features."""
    cal = ref.cal
    return {
        "reference/counterparties": pd.DataFrame({
            "cpty_id": np.arange(ref.n_cpty, dtype=np.int16),
            "cpty_name": [f"CP{i:04d}" for i in range(ref.n_cpty)],
            "cpty_type": pd.Categorical([CPTY_TYPES[c] for c in ref.cpty_type], categories=CPTY_TYPES),
            "market": pd.Categorical([MARKETS[m] for m in ref.cpty_market], categories=MARKETS),
            "first_trade_date": [cal.bdays[min(d, len(cal.bdays) - 1)] for d in ref.cpty_onboard_day],
        }),
        "reference/securities": pd.DataFrame({
            "sec_id": np.arange(ref.n_sec, dtype=np.int16),
            "sec_name": [f"SEC{i:05d}" for i in range(ref.n_sec)],
            "asset_class": pd.Categorical([ASSET_CLASSES[a] for a in ref.sec_class], categories=ASSET_CLASSES),
            "market": pd.Categorical([MARKETS[m] for m in ref.sec_market], categories=MARKETS),
            "corporate_action_every_bdays": ref.sec_ca_period,
        }),
        "reference/calendar": pd.DataFrame({
            "trade_date": cal.bdays,
            "month": cal.month_label,
            "market_volatility_level": cal.vol,
            "vol_spike": cal.vol_spike,
            "volume_multiplier": cal.volume_mult.astype(np.float32),
            "home_holiday_week": cal.holiday_week[HOME],
        }),
        "truth/counterparty_latent": pd.DataFrame({
            "cpty_id": np.arange(ref.n_cpty, dtype=np.int16),
            "latent_risk": ref.cpty_latent.astype(np.float32),
            "ssi_drift": ref.cpty_ssi_drift,
        }),
        "truth/security_latent": pd.DataFrame({
            "sec_id": np.arange(ref.n_sec, dtype=np.int16),
            "fail_propensity": ref.sec_latent.astype(np.float32),
        }),
    }
