"""The planted fail-probability function.

log-odds = intercept + one term per feature + three interactions + hidden terms + noise

Every feature term is zero at the clean reference value and never decreases as the feature
moves in its risk direction, so contributions can be read directly and the golden set and
monotonicity tests can check them. Inputs are the true feature values (before values are
blanked as missing) with categoricals as integer codes into the levels in definitions.py.
"""

from __future__ import annotations

import numpy as np

from src.features.definitions import BY_NAME, FEATURE_NAMES

INTERACTIONS = {
    "ssi_mismatch_x_cross_border": ("ssi_match_status", "is_cross_border"),
    "shortfall_x_security_fail_rate": ("obligation_coverage_ratio", "security_fail_rate_30d"),
    "confirmation_x_overnight": ("hours_to_confirmation", "instruction_hour_bucket"),
}
TERMS = FEATURE_NAMES + tuple(INTERACTIONS)


def _level(feature: str, level: str) -> int:
    return BY_NAME[feature].levels.index(level)


def _table(feature: str, weights: dict) -> np.ndarray:
    levels = BY_NAME[feature].levels
    return np.array([weights.get(lv, 0.0) for lv in levels], dtype=np.float64)


def _f(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


# Building blocks shared by the main terms and the interactions.
def confirmation_shape(hours, scale):
    return np.log1p(np.maximum(_f(hours), 0.0) / scale)


def shortfall(ratio):
    return np.clip(1.0 - _f(ratio), 0.0, 1.0)


def security_rate_excess(rate, it):
    return np.clip((_f(rate) - it["rate_floor"]) / it["rate_scale"], 0.0, it["cap"])


def feature_terms(F: dict, cfg: dict) -> dict[str, np.ndarray]:
    """Per-trade contribution of each feature and interaction to the log-odds."""
    e, it = cfg["effects"], cfg["interactions"]
    out: dict[str, np.ndarray] = {}
    cat = {name: np.asarray(F[name], dtype=np.int64) for name in
           ("ssi_match_status", "instruction_hour_bucket", "cpty_type", "asset_class")}
    for name, codes in cat.items():
        out[name] = _table(name, e[name])[codes]

    out["ssi_age_days"] = e["ssi_age_days"]["per_year"] * np.minimum(
        np.maximum(_f(F["ssi_age_days"]), 0.0), e["ssi_age_days"]["cap_days"]
    ) / 365.0
    h = e["hours_to_confirmation"]
    out["hours_to_confirmation"] = h["weight"] * confirmation_shape(F["hours_to_confirmation"], h["scale_hours"])
    out["amendment_count"] = e["amendment_count"]["weight"] * np.log1p(np.maximum(_f(F["amendment_count"]), 0))
    a = e["allocation_delay_hrs"]
    out["allocation_delay_hrs"] = a["weight"] * np.log1p(np.maximum(_f(F["allocation_delay_hrs"]), 0) / a["scale_hours"])
    out["chain_depth"] = e["chain_depth"]["weight"] * np.log1p(np.maximum(_f(F["chain_depth"]), 0))
    out["obligation_coverage_ratio"] = e["obligation_coverage_ratio"]["weight"] * shortfall(
        F["obligation_coverage_ratio"]
    )
    c = e["cpty_fail_rate_30d"]
    out["cpty_fail_rate_30d"] = c["weight"] * np.log1p(np.maximum(_f(F["cpty_fail_rate_30d"]), 0) / c["scale"])
    p = e["pair_history_trades_90d"]
    out["pair_history_trades_90d"] = p["weight"] * np.exp(
        -np.maximum(_f(F["pair_history_trades_90d"]), 0) / p["scale_trades"]
    )
    s = e["security_fail_rate_30d"]
    out["security_fail_rate_30d"] = s["weight"] * np.log1p(
        np.maximum(_f(F["security_fail_rate_30d"]), 0) / s["scale"]
    )
    x = _f(F["notional_vs_cpty_median"])
    # A counterparty with no trade history has no median: that trade gets no size effect.
    size = np.where(np.isfinite(x) & (x > 0), np.log(np.where(x > 0, x, 1.0)), 0.0)
    out["notional_vs_cpty_median"] = e["notional_vs_cpty_median"]["weight"] * np.maximum(size, 0.0)
    b = e["abs_price_deviation_bps"]
    out["abs_price_deviation_bps"] = b["weight"] * np.log1p(np.maximum(_f(F["abs_price_deviation_bps"]), 0) / b["scale_bps"])
    v = e["market_volatility_level"]
    out["market_volatility_level"] = v["weight"] * np.maximum(_f(F["market_volatility_level"]) - v["floor"], 0) / v["scale"]
    for flag in ("corporate_action_in_window", "is_cross_border", "holiday_in_settlement_window", "is_period_end"):
        out[flag] = e[flag]["weight"] * (_f(F[flag]) > 0)

    mismatch = cat["ssi_match_status"] == _level("ssi_match_status", "mismatch")
    out["ssi_mismatch_x_cross_border"] = it["ssi_mismatch_x_cross_border"]["weight"] * (
        mismatch & (_f(F["is_cross_border"]) > 0)
    )
    i2 = it["shortfall_x_security_fail_rate"]
    out["shortfall_x_security_fail_rate"] = i2["weight"] * shortfall(F["obligation_coverage_ratio"]) * (
        security_rate_excess(F["security_fail_rate_30d"], i2)
    )
    overnight = cat["instruction_hour_bucket"] == _level("instruction_hour_bucket", "overnight")
    out["confirmation_x_overnight"] = it["confirmation_x_overnight"]["weight"] * overnight * confirmation_shape(
        F["hours_to_confirmation"], h["scale_hours"]
    )
    return out


def deterministic_logit(F: dict, cfg: dict, intercept: float = 0.0) -> np.ndarray:
    """intercept + feature terms + interactions, without hidden terms or noise."""
    terms = feature_terms(F, cfg)
    return intercept + sum(terms[name] for name in TERMS)


def true_shap(F: dict, cfg: dict, reference: dict | None = None) -> dict[str, np.ndarray]:
    """Exact interventional Shapley values of the planted log-odds, per feature.

    Additive terms give t(x) - E[t]. A product term a(x) * b(y) splits as
    0.5 * (a(x) - E[a]) * (b(y) + E[b]) to x and the mirror image to y. Expectations are
    taken over `reference` (default: the same trades), treating features as independent.
    Hidden terms and noise are not attributable to features and are left out.
    """
    ref = F if reference is None else reference
    t, tr = feature_terms(F, cfg), feature_terms(ref, cfg)
    # A value blanked as missing gets no attribution: its term is set to the reference mean.
    phi = {}
    for name in FEATURE_NAMES:
        m = np.nanmean(tr[name])
        phi[name] = np.where(np.isnan(t[name]), m, t[name]) - m
    e, it = cfg["effects"], cfg["interactions"]
    h = e["hours_to_confirmation"]["scale_hours"]
    i2 = it["shortfall_x_security_fail_rate"]

    def parts(D):
        st = np.asarray(D["ssi_match_status"], dtype=np.int64)
        hb = np.asarray(D["instruction_hour_bucket"], dtype=np.int64)
        return {
            "ssi_mismatch_x_cross_border": (
                (st == _level("ssi_match_status", "mismatch")).astype(float),
                (_f(D["is_cross_border"]) > 0).astype(float),
            ),
            "shortfall_x_security_fail_rate": (
                shortfall(D["obligation_coverage_ratio"]),
                security_rate_excess(D["security_fail_rate_30d"], i2),
            ),
            "confirmation_x_overnight": (
                confirmation_shape(D["hours_to_confirmation"], h),
                (hb == _level("instruction_hour_bucket", "overnight")).astype(float),
            ),
        }

    pf, pr = parts(F), parts(ref)
    for name, (fx, fy) in INTERACTIONS.items():
        w = it[name]["weight"]
        ea, eb = np.nanmean(pr[name][0]), np.nanmean(pr[name][1])
        a = np.where(np.isnan(pf[name][0]), ea, pf[name][0])
        b = np.where(np.isnan(pf[name][1]), eb, pf[name][1])
        phi[fx] = phi[fx] + 0.5 * w * (a - ea) * (b + eb)
        phi[fy] = phi[fy] + 0.5 * w * (b - eb) * (a + ea)
    return phi


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=np.float64)))


def planted_logit(F: dict, cfg: dict, intercept: float, *, repaired, borrowed, unexplained,
                  cpty_latent, sec_latent, noise) -> tuple[np.ndarray, dict]:
    """The full planted log-odds, including hidden effects that no feature shows:
    an SSI break that gets repaired in time (H1) or a shortfall covered by a late borrow (H2)
    keeps only a small share of its term, and an unexplained shock (H4) adds a large boost.
    Returns the log-odds and the per-term contributions after those adjustments."""
    hidden = cfg["hidden"]
    terms = feature_terms(F, cfg)
    k = hidden["mitigation_factor"]
    for name in ("ssi_match_status", "ssi_mismatch_x_cross_border"):
        terms[name] = np.where(repaired, k * terms[name], terms[name])
    for name in ("obligation_coverage_ratio", "shortfall_x_security_fail_rate"):
        terms[name] = np.where(borrowed, k * terms[name], terms[name])
    logit = (
        intercept
        + sum(terms[name] for name in TERMS)
        + hidden["cpty_latent_weight"] * np.asarray(cpty_latent)
        + hidden["security_latent_weight"] * np.asarray(sec_latent)
        + hidden["noise_sd"] * np.asarray(noise)
        + hidden["unexplained_boost"] * np.asarray(unexplained)
    )
    return logit, terms
