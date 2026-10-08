"""Assets the demo app reads, precomputed by notebook 02 so the Space stays fast on free CPU.

Everything here is derived from the published data and models with a fixed seed.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import xgboost as xgb

from src.data.coverage import run_coverage
from src.data.generator import FAIL_REASONS
from src.features.definitions import BINARY, BY_NAME, CATEGORICAL, FEATURE_NAMES, FEATURES
from src.models import train_xgb

# What an ops analyst should check first when a feature pushes a trade's risk up.
OPS_CHECK = {
    "ssi_match_status": "Repair the settlement instructions: confirm account, custodian, and BIC with the counterparty.",
    "ssi_age_days": "Re-verify the standing settlement instructions on file before settlement date.",
    "instruction_hour_bucket": "Check the instruction reached the depository before its cutoff; resend in business hours.",
    "hours_to_confirmation": "Chase the counterparty confirmation; an unconfirmed trade is the classic 'don't know' fail.",
    "amendment_count": "Re-match the latest version of the trade with the counterparty after the amendments.",
    "allocation_delay_hrs": "Push the block allocation through so the settlement instruction can go out.",
    "chain_depth": "Check the upstream deliveries this trade depends on and whether any are failing.",
    "obligation_coverage_ratio": "Source the shortfall: arrange a borrow or recall for securities, or fund the cash.",
    "cpty_fail_rate_30d": "Contact the counterparty's operations team early; they have failed often this month.",
    "cpty_type": "Apply the playbook for this counterparty type (manual checks for less automated firms).",
    "pair_history_trades_90d": "New or rare relationship: double-check instructions and contacts set up for this pair.",
    "security_fail_rate_30d": "The security is hard to source right now: line up a borrow or partial delivery early.",
    "corporate_action_in_window": "Check entitlements and holds around the corporate action dates.",
    "notional_vs_cpty_median": "Unusually large for this counterparty: confirm funding and inventory before settlement.",
    "abs_price_deviation_bps": "Off-market price: confirm the economics with the counterparty before matching fails.",
    "asset_class": "Follow the market practice and cutoffs for this instrument type.",
    "is_cross_border": "Check the foreign market's cutoffs, holidays, and the intermediary chain.",
    "market_volatility_level": "Stressed market: expect volume surges and prioritize this trade early.",
    "holiday_in_settlement_window": "A holiday shortens the window: send instructions and funding a day earlier.",
    "is_period_end": "Period end: expect funding pressure and higher volume, act before the cutoff.",
}


def features_json() -> list[dict]:
    return [{
        "index": i + 1, "name": f.name, "label": f.label, "group": f.group, "kind": f.kind,
        "description": f.description, "mechanism": f.mechanism, "strength": f.strength,
        "direction": f.direction, "levels": list(f.levels), "ops_check": OPS_CHECK[f.name],
    } for i, f in enumerate(FEATURES)]


def data_summary(df: pd.DataFrame, splits: dict, cfg: dict) -> dict:
    fails = df[df["failed"] == 1]
    return {
        "n_trades": int(len(df)),
        "fail_rate": float(df["failed"].mean()),
        "first_date": str(pd.to_datetime(df["trade_date"]).min().date()),
        "last_date": str(pd.to_datetime(df["trade_date"]).max().date()),
        "splits": {k: {"rows": int(len(v)), "fail_rate": float(v["failed"].mean()),
                       "first": str(pd.to_datetime(v["trade_date"]).min().date()),
                       "last": str(pd.to_datetime(v["trade_date"]).max().date())} for k, v in splits.items()},
        "gap_days": cfg["split"]["gap_days"],
        "reason_mix": {r: float((fails["fail_reason"] == r).mean()) for r in FAIL_REASONS},
        "n_counterparties": int(df["cpty_id"].nunique()),
        "n_securities": int(df["sec_id"].nunique()),
        "seed": cfg["seed"],
        "dataset_repo": cfg["project"]["dataset_repo"],
        "model_repo": cfg["project"]["model_repo"],
        "github_repo": cfg["project"]["github_repo"],
    }


def feature_bins(df: pd.DataFrame, n_bins: int = 10) -> dict:
    """Trades and fail rate per level or per quantile bin, over all generated trades."""
    out = {}
    y = df["failed"].to_numpy()
    for f in FEATURES:
        col = df[f.name]
        if f.name in CATEGORICAL or f.name in BINARY:
            keys = list(f.levels) if f.levels else [0, 1]
            rows = [{"bin": str(k), "trades": int((col == k).sum()),
                     "fail_rate": float(y[(col == k).to_numpy()].mean()) if (col == k).any() else None} for k in keys]
        else:
            vals = col.to_numpy(dtype=float)
            edges = np.unique(np.nanquantile(vals, np.linspace(0, 1, n_bins + 1)))
            rows = []
            for lo, hi in zip(edges[:-1], edges[1:]):
                m = (vals >= lo) & ((vals <= hi) if hi == edges[-1] else (vals < hi))
                rows.append({"bin": f"{lo:.3g} to {hi:.3g}", "lo": float(lo), "hi": float(hi), "trades": int(m.sum()),
                             "fail_rate": float(y[m].mean()) if m.any() else None})
            miss = np.isnan(vals)
            if miss.any():
                rows.append({"bin": "missing", "trades": int(miss.sum()), "fail_rate": float(y[miss].mean())})
        out[f.name] = rows
    return out


def coverage_summary(df: pd.DataFrame, splits: dict, cfg: dict) -> dict:
    res = run_coverage(df, splits, cfg)
    rows = [{"id": r["id"], "kind": r["kind"], "name": r["name"], "passed": r["passed"],
             **{k: c["count"] for k, c in r["cells"].items()}} for r in res["matrix"]]
    return {"passed": res["passed"], "failures": res["failures"], "rows": rows,
            "realism": res["realism"], "time": res["time_coverage"], "pairwise": res["pairwise_coverage"]}


def xgb_views(bst: xgb.Booster, state: dict, X: pd.DataFrame, picks: dict[str, int], max_depth: int = 3) -> dict:
    """The first tree (top levels), the validation curve, and how three trades' log-odds move
    as trees are added."""
    tree = json.loads(bst.get_dump(with_stats=True, dump_format="json")[0])
    nodes = []

    def walk(node, depth, parent):
        leaf = "leaf" in node
        if leaf:
            label = f"leaf value {node['leaf']:+.3f}"
        else:
            feat = node["split"]
            cats = node.get("categories")
            cond = node.get("split_condition")
            if cats is None and isinstance(cond, list):
                cats = cond
            if cats is not None:
                levels = BY_NAME[feat].levels if feat in BY_NAME else ()
                names = [levels[int(c)] if int(c) < len(levels) else str(c) for c in cats]
                label = f"{feat} in {names}"
            else:
                label = f"{feat} < {float(cond):.4g}"
        nodes.append({"id": node["nodeid"], "parent": parent, "depth": depth, "leaf": leaf, "label": label,
                      "cover": float(node.get("cover", 0.0)), "yes": node.get("yes"), "missing": node.get("missing")})
        if not leaf and depth < max_depth:
            for child in node["children"]:
                walk(child, depth + 1, node["nodeid"])

    walk(tree, 0, None)
    n_trees = bst.num_boosted_rounds()
    steps = sorted({int(k) for k in np.unique(np.geomspace(1, n_trees, 30).astype(int))} | {n_trees})
    d = train_xgb.dmatrix(X.iloc[list(picks.values())])
    staged = {name: [] for name in picks}
    for k in steps:
        margin = bst.predict(d, iteration_range=(0, k), output_margin=True)
        for i, name in enumerate(picks):
            staged[name].append(float(margin[i]))
    return {"tree0": nodes, "n_trees": n_trees, "best_iteration": state["best_iteration"],
            "val_aucpr": state["history"], "staged_steps": steps, "staged_margin": staged}


def app_sample(sample: pd.DataFrame, scores: dict[str, np.ndarray], phi: np.ndarray, phi_true: np.ndarray,
               inter: dict[str, np.ndarray], rows: np.ndarray, pool: np.ndarray) -> pd.DataFrame:
    """Rows shipped to the app: features, labels, scores, SHAP and true SHAP per feature."""
    out = sample.iloc[rows].reset_index(drop=True)[["trade_id", "trade_date", *FEATURE_NAMES, "failed",
                                                    "fail_reason", "scenario_mask"]].copy()
    for name, s in scores.items():
        out[f"score__{name}"] = s.astype(np.float32)
    for j, name in enumerate(FEATURE_NAMES):
        out[f"shap__{name}"] = phi[:, j].astype(np.float32)
        out[f"true__{name}"] = phi_true[:, j].astype(np.float32)
    out["shap__bias"] = phi[:, len(FEATURE_NAMES)].astype(np.float32)
    for name, v in inter.items():
        out[f"inter__{name}"] = v.astype(np.float32)
    out["pool"] = pool
    return out


def sim_trades(test: pd.DataFrame, scores: dict[str, np.ndarray]) -> pd.DataFrame:
    days = pd.to_datetime(test["trade_date"])
    out = pd.DataFrame({"day": (days - days.min()).dt.days.astype(np.int16), "failed": test["failed"].astype(np.int8)})
    for name, s in scores.items():
        out[name] = s.astype(np.float32)
    return out
