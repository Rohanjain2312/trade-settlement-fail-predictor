"""Evaluation metrics for imbalanced fail prediction.

Accuracy is not a headline metric: at 3% positives, predicting "settles" for every trade
scores 97%. The headline metrics are PR-AUC, recall at fixed precision, recall within the
top-scored share of trades, and the Brier score for probability quality.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
)


def recall_at_precision(y, p, target: float) -> float:
    precision, recall, _ = precision_recall_curve(y, p)
    ok = precision >= target
    return float(recall[ok].max()) if ok.any() else 0.0


def recall_in_top(y, p, share: float) -> float:
    y = np.asarray(y)
    k = max(1, int(np.ceil(share * len(y))))
    top = np.argsort(-np.asarray(p), kind="stable")[:k]
    return float(y[top].sum() / max(1, y.sum()))


def threshold_for_capacity(p_val, capacity_share: float) -> float:
    """Score threshold that flags `capacity_share` of trades, chosen on validation data."""
    return float(np.quantile(np.asarray(p_val), 1.0 - capacity_share))


def calibration_curve(y, p, bins: int = 10) -> dict:
    """Reliability curve with equal-count bins (most scores are tiny at a 3% base rate)."""
    y, p = np.asarray(y), np.asarray(p)
    order = np.argsort(p, kind="stable")
    chunks = np.array_split(order, bins)
    return {
        "mean_predicted": [float(p[c].mean()) for c in chunks if len(c)],
        "fraction_failed": [float(y[c].mean()) for c in chunks if len(c)],
        "count": [int(len(c)) for c in chunks if len(c)],
    }


def pr_curve(y, p, points: int = 101) -> dict:
    precision, recall, _ = precision_recall_curve(y, p)
    grid = np.linspace(0, 1, points)
    # Interpolated precision: best precision at recall >= r.
    prec = [float(precision[recall >= r].max()) if (recall >= r).any() else 0.0 for r in grid]
    return {"recall": grid.round(4).tolist(), "precision": [round(v, 5) for v in prec]}


def metrics(y, p, threshold: float, with_curves: bool = False) -> dict:
    y, p = np.asarray(y).astype(int), np.asarray(p, dtype=float)
    out = {"n": int(len(y)), "positives": int(y.sum()), "base_rate": float(y.mean()) if len(y) else 0.0}
    if len(y) == 0 or y.min() == y.max():
        out["note"] = "single class in this slice"
        flagged = p >= threshold
        out.update(flag_rate=float(flagged.mean()) if len(y) else 0.0)
        return out
    flagged = p >= threshold
    tp = int((flagged & (y == 1)).sum())
    fp = int((flagged & (y == 0)).sum())
    fn = int((~flagged & (y == 1)).sum())
    tn = int((~flagged & (y == 0)).sum())
    out.update(
        pr_auc=float(average_precision_score(y, p)),
        roc_auc=float(roc_auc_score(y, p)),
        brier=float(brier_score_loss(y, p)),
        recall_at_precision_50=recall_at_precision(y, p, 0.5),
        recall_at_precision_70=recall_at_precision(y, p, 0.7),
        recall_top_1pct=recall_in_top(y, p, 0.01),
        recall_top_2pct=recall_in_top(y, p, 0.02),
        threshold=float(threshold),
        confusion={"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        precision_at_threshold=tp / max(1, tp + fp),
        recall_at_threshold=tp / max(1, tp + fn),
        flag_rate=float(flagged.mean()),
        accuracy=float((tp + tn) / len(y)),
        accuracy_if_always_settles=float(1 - y.mean()),
    )
    if with_curves:
        out["calibration_curve"] = calibration_curve(y, p)
        out["pr_curve"] = pr_curve(y, p)
    return out
