"""Quality checks on generated data.

Each check returns {"name", "passed", ...details}. The validate stage writes one result file
per check, so a restart skips the checks that already ran.
"""

from __future__ import annotations

import itertools
import logging

import numpy as np
import pandas as pd
from scipy import stats

from src.data import probability as prob
from src.data.generator import FAIL_REASONS, frame_to_arrays
from src.features.definitions import (
    BINARY,
    BY_NAME,
    CATEGORICAL,
    CONTINUOUS,
    FEATURE_NAMES,
    FEATURES,
    NUMERIC,
    STRENGTH_ORDER,
)

log = logging.getLogger(__name__)

SPEARMAN_MAX = 0.6
CRAMERS_V_MAX = 0.5
CORR_RATIO_MAX = 0.5
VIF_MAX = 5.0
SAMPLE_ROWS = 200_000


def _sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    return df if len(df) <= n else df.sample(n=n, random_state=seed)


# ---- fail rates -----------------------------------------------------------------------------

def check_rates(df: pd.DataFrame, cfg: dict) -> dict:
    d = cfg["data"]
    target, tol = d["target_fail_rate"], d["overall_rate_tolerance"]
    lo, hi = d["monthly_rate_band"]
    overall = float(df["failed"].mean())
    month = pd.to_datetime(df["trade_date"]).dt.strftime("%Y-%m")
    monthly = df.groupby(month)["failed"].mean()
    bad = {m: round(float(r), 5) for m, r in monthly.items() if not lo <= r <= hi}
    passed = abs(overall - target) <= tol * target and not bad
    return {
        "name": "fail_rates",
        "passed": bool(passed),
        "overall": round(overall, 5),
        "target": target,
        "allowed_overall": [round(target * (1 - tol), 5), round(target * (1 + tol), 5)],
        "monthly_band": [lo, hi],
        "monthly": {m: round(float(r), 5) for m, r in monthly.items()},
        "months_out_of_band": bad,
        "n_trades": int(len(df)),
        "n_failed": int(df["failed"].sum()),
    }


def reason_mix(df: pd.DataFrame) -> dict[str, float]:
    failed = df[df["failed"] == 1]
    share = failed["fail_reason"].value_counts(normalize=True)
    return {r: round(float(share.get(r, 0.0)), 4) for r in FAIL_REASONS}


# ---- distinctness ---------------------------------------------------------------------------

def cramers_v(x: pd.Series, y: pd.Series) -> float:
    table = pd.crosstab(x, y)
    if min(table.shape) < 2:
        return 0.0
    chi2 = stats.chi2_contingency(table, correction=False)[0]
    n = table.to_numpy().sum()
    return float(np.sqrt(chi2 / (n * (min(table.shape) - 1))))


def correlation_ratio(categories: pd.Series, values: pd.Series) -> float:
    ok = values.notna() & categories.notna()
    c, v = categories[ok], values[ok].astype(float)
    if v.var() == 0 or len(v) == 0:
        return 0.0
    means = v.groupby(c, observed=True).agg(["mean", "count"])
    between = float((means["count"] * (means["mean"] - v.mean()) ** 2).sum())
    total = float(((v - v.mean()) ** 2).sum())
    return float(np.sqrt(between / total))


def vif(df: pd.DataFrame, cols: list[str]) -> dict[str, float]:
    X = df[cols].astype(float)
    X = X.fillna(X.median())
    corr = np.corrcoef(X.to_numpy(), rowvar=False)
    inv = np.linalg.pinv(corr)
    return {c: float(inv[i, i]) for i, c in enumerate(cols)}


def check_distinctness(df: pd.DataFrame, cfg: dict) -> dict:
    s = _sample(df, SAMPLE_ROWS, cfg["seed"])
    failures = []
    spearman = s[list(CONTINUOUS)].astype(float).corr(method="spearman")
    pairs = {}
    for a, b in itertools.combinations(CONTINUOUS, 2):
        r = float(abs(spearman.loc[a, b]))
        pairs[f"{a}|{b}"] = round(r, 4)
        if r >= SPEARMAN_MAX:
            failures.append(f"spearman {a} vs {b} = {r:.3f}")
    cats = list(CATEGORICAL) + list(BINARY)
    cv = {}
    for a, b in itertools.combinations(cats, 2):
        v = cramers_v(s[a], s[b])
        cv[f"{a}|{b}"] = round(v, 4)
        if v >= CRAMERS_V_MAX:
            failures.append(f"cramers_v {a} vs {b} = {v:.3f}")
    eta = {}
    for a in cats:
        for b in CONTINUOUS:
            r = correlation_ratio(s[a], s[b])
            eta[f"{a}|{b}"] = round(r, 4)
            if r >= CORR_RATIO_MAX:
                failures.append(f"correlation_ratio {a} vs {b} = {r:.3f}")
    vifs = vif(s, list(NUMERIC))
    failures += [f"vif {c} = {v:.2f}" for c, v in vifs.items() if v >= VIF_MAX]
    missing = {c: round(float(df[c].isna().mean()), 4) for c in FEATURE_NAMES}
    max_missing = cfg["data"]["max_missing_share"]
    failures += [f"missing share {c} = {m:.3f}" for c, m in missing.items() if m > max_missing]
    constant = [c for c in FEATURE_NAMES if df[c].nunique(dropna=True) < 2]
    failures += [f"constant feature {c}" for c in constant]

    def top(d, k=5):
        return dict(sorted(d.items(), key=lambda kv: -kv[1])[:k])

    return {
        "name": "feature_distinctness",
        "passed": not failures,
        "failures": failures,
        "thresholds": {"spearman": SPEARMAN_MAX, "cramers_v": CRAMERS_V_MAX,
                       "correlation_ratio": CORR_RATIO_MAX, "vif": VIF_MAX, "max_missing": max_missing},
        "top_spearman": top(pairs),
        "top_cramers_v": top(cv),
        "top_correlation_ratio": top(eta),
        "vif": {c: round(v, 3) for c, v in vifs.items()},
        "missing_share": missing,
    }


# ---- planted effects ------------------------------------------------------------------------

def true_importance(df: pd.DataFrame, cfg: dict) -> dict[str, float]:
    """Mean absolute true Shapley value of each feature in the planted log-odds."""
    s = _sample(df, SAMPLE_ROWS, cfg["seed"])
    phi = prob.true_shap(frame_to_arrays(s), cfg)
    return {name: float(np.mean(np.abs(phi[name]))) for name in FEATURE_NAMES}


def check_strength_tiers(importance: dict[str, float]) -> dict:
    """The planted strengths have to show up in the data: every strong feature must matter
    more than every weak one, and the tier averages must be ordered."""
    tiers = {t: [importance[f.name] for f in FEATURES if f.strength == t] for t in STRENGTH_ORDER}
    means = {t: float(np.mean(v)) for t, v in tiers.items()}
    passed = min(tiers["strong"]) > max(tiers["weak"]) and means["strong"] > means["medium"] > means["weak"]
    return {
        "name": "strength_tiers",
        "passed": bool(passed),
        "tier_means": {t: round(v, 4) for t, v in means.items()},
        "importance": {k: round(v, 4) for k, v in sorted(importance.items(), key=lambda kv: -kv[1])},
    }


def check_lr_recovery(df: pd.DataFrame, cfg: dict, importance: dict[str, float]) -> dict:
    """A plain logistic regression on the generated data should recover the sign of every
    directional strong and medium effect and roughly their ranking."""
    from sklearn.linear_model import LogisticRegression

    s = _sample(df, 300_000, cfg["seed"])
    num = s[list(NUMERIC)].astype(float)
    num = num.fillna(num.median())
    mu, sd = num.mean(), num.std().replace(0, 1.0)
    Z = (num - mu) / sd
    dummies = []
    for name in CATEGORICAL:
        f = BY_NAME[name]
        for lv in f.levels:
            if lv != f.reference_level:
                dummies.append(((s[name] == lv).astype(float)).rename(f"{name}={lv}"))
    X = pd.concat([Z] + dummies, axis=1)
    model = LogisticRegression(C=1.0, max_iter=2000)
    model.fit(X.to_numpy(), s["failed"].to_numpy())
    coef = dict(zip(X.columns, model.coef_[0]))

    wrong_sign = []
    lr_importance = {}
    for f in FEATURES:
        if f.kind == "categorical":
            cols = [c for c in X.columns if c.startswith(f.name + "=")]
            contrib = X[cols].to_numpy() @ np.array([coef[c] for c in cols])
            if f.strength != "weak" and f.name in ("ssi_match_status", "instruction_hour_bucket"):
                if np.mean([coef[c] for c in cols]) <= 0:
                    wrong_sign.append(f.name)
        else:
            contrib = coef[f.name] * X[f.name].to_numpy()
            if f.strength != "weak" and np.sign(coef[f.name]) != f.direction:
                wrong_sign.append(f.name)
        lr_importance[f.name] = float(np.mean(np.abs(contrib - contrib.mean())))
    rho = stats.spearmanr([lr_importance[n] for n in FEATURE_NAMES], [importance[n] for n in FEATURE_NAMES])[0]
    return {
        "name": "lr_recovers_planted_effects",
        "passed": not wrong_sign and rho >= 0.5,
        "wrong_sign": wrong_sign,
        "rank_correlation_with_truth": round(float(rho), 3),
        "coefficients": {k: round(float(v), 4) for k, v in coef.items()},
        "lr_importance": {k: round(v, 4) for k, v in sorted(lr_importance.items(), key=lambda kv: -kv[1])},
    }


# ---- rolling features use no future information ---------------------------------------------

def recompute_rolling(df: pd.DataFrame, cfg: dict, n_sample: int = 400) -> dict:
    """Recompute the rolling features of sampled trades from raw trade history and compare
    with the stored values. Any use of future outcomes would show up as a mismatch."""
    prior = cfg["data"]["target_fail_rate"]
    a_c, a_s = cfg["data"]["rate_prior_trades"]["cpty"], cfg["data"]["rate_prior_trades"]["security"]
    td = df["trade_date"].to_numpy().astype("datetime64[D]").astype(np.int64)
    sd = df["settle_date"].to_numpy().astype("datetime64[D]").astype(np.int64)
    failed = df["failed"].to_numpy().astype(np.int64)
    cpty = df["cpty_id"].to_numpy().astype(np.int64)
    sec = df["sec_id"].to_numpy().astype(np.int64)
    desk = df["desk_id"].to_numpy().astype(np.int64)
    notional = df["notional"].to_numpy()
    first = td.min()
    eligible = np.flatnonzero(td >= first + 35)
    rng = np.random.default_rng([cfg["seed"], 99])
    picks = rng.choice(eligible, size=min(n_sample, len(eligible)), replace=False)

    by_cpty = {c: np.flatnonzero(cpty == c) for c in np.unique(cpty[picks])}
    by_sec = {s: np.flatnonzero(sec == s) for s in np.unique(sec[picks])}
    worst = {"cpty_fail_rate_30d": 0.0, "security_fail_rate_30d": 0.0,
             "pair_history_trades_90d": 0.0, "notional_vs_cpty_median": 0.0}
    mismatches = []
    for i in picks:
        d = td[i]
        rows = by_cpty[cpty[i]]
        w = rows[(sd[rows] >= d - 30) & (sd[rows] < d)]
        exp_c = (failed[w].sum() + a_c * prior) / (len(w) + a_c)
        rows_s = by_sec[sec[i]]
        ws = rows_s[(sd[rows_s] >= d - 30) & (sd[rows_s] < d)]
        exp_s = (failed[ws].sum() + a_s * prior) / (len(ws) + a_s)
        wp = rows[(td[rows] >= d - 90) & (td[rows] < d) & (desk[rows] == desk[i])]
        wn = rows[(td[rows] >= d - 90) & (td[rows] < d)]
        exp_x = notional[i] / np.median(notional[wn].astype(np.float64)) if len(wn) else np.nan
        expected = {
            "cpty_fail_rate_30d": exp_c,
            "security_fail_rate_30d": exp_s,
            "pair_history_trades_90d": float(len(wp)),
            "notional_vs_cpty_median": exp_x,
        }
        for name, exp in expected.items():
            got = df[name].iat[i]
            if pd.isna(got):
                continue  # blanked as missing, or no history for the size median
            err = abs(float(got) - float(exp)) / max(1.0, abs(float(exp)))
            worst[name] = max(worst[name], err)
            if err > 1e-5:
                mismatches.append({"trade_id": int(df["trade_id"].iat[i]), "feature": name,
                                   "stored": float(got), "recomputed": float(exp)})
    return {
        "name": "rolling_features_no_future",
        "passed": not mismatches,
        "n_checked": int(len(picks)),
        "worst_relative_error": worst,
        "mismatches": mismatches[:20],
    }


def run_checks(df: pd.DataFrame, cfg: dict) -> list[dict]:
    importance = true_importance(df, cfg)
    results = [
        check_rates(df, cfg),
        check_distinctness(df, cfg),
        check_strength_tiers(importance),
        check_lr_recovery(df, cfg, importance),
        recompute_rolling(df, cfg),
    ]
    for r in results:
        log.info("check %s: %s", r["name"], "PASS" if r["passed"] else "FAIL")
    return results


def scenario_summary(df: pd.DataFrame) -> dict:
    from src.data.scenarios import SCENARIO_IDS, decode

    members = decode(df["scenario_mask"].to_numpy())
    y = df["failed"].to_numpy()
    return {
        sid: {"share": round(float(m.mean()), 4), "fail_rate": round(float(y[m].mean()), 4) if m.any() else None,
              "failed": int(y[m].sum())}
        for sid, m in members.items() if sid in SCENARIO_IDS
    }


def main() -> None:
    """CI diagnostics: generate the smoke dataset in memory and print every check."""
    import argparse
    import json
    import sys

    from src.config import load_config
    from src.data.generator import calibrate_intercept, generate_day, init_state, to_frame
    from src.data.reference_data import build_reference

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="smoke")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, stream=sys.stdout)
    cfg = load_config(args.mode)
    ref = build_reference(cfg)
    cal = calibrate_intercept(ref, cfg)
    state = init_state(ref, cfg)
    df = to_frame([generate_day(t, state, ref, cfg, cal["intercept"]) for t in range(len(ref.cal.bdays))])
    out = {
        "calibration": {k: cal[k] for k in ("intercept", "pilot_trades_per_day")},
        "calibration_steps": len(cal["history"]),
        "reason_mix": reason_mix(df),
        "scenarios": scenario_summary(df),
        "checks": run_checks(df, cfg),
    }
    for r in out["checks"]:
        r.pop("monthly", None)
        r.pop("coefficients", None)
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
