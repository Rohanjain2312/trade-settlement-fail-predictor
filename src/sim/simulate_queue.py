"""Ops queue what-if simulator.

Each business day, ops can work `capacity_share` of that day's trades before settlement.
A worked trade that would have failed gets fixed with probability `fix_rate` (for example
an SSI repaired, a borrow arranged, a confirmation chased). Trades nobody works settle or
fail as they would have.

  Without the model, ops pick trades with no risk ranking, so on average they work a
  random share of the day's would-be fails.
  With the model, ops work the highest-scored trades first.

Pending trades are the would-be fails that nobody fixed. Every number depends on the
assumptions chosen, and this is a what-if on synthetic data, not a measured result.
This file has no dependencies beyond numpy so it ships inside the demo Space unchanged.
"""

from __future__ import annotations

import numpy as np


def day_ranks(day: np.ndarray, score: np.ndarray) -> np.ndarray:
    """0-based rank of each trade within its day, highest score first."""
    order = np.lexsort((-score, day))
    ranks = np.empty(len(day), dtype=np.int64)
    d = day[order]
    starts = np.r_[0, np.flatnonzero(np.diff(d)) + 1]
    counts = np.diff(np.r_[starts, len(d)])
    ranks[order] = np.arange(len(d)) - np.repeat(starts, counts)
    return ranks


def simulate(day: np.ndarray, failed: np.ndarray, score: np.ndarray | None, capacity_share: float,
             fix_rate: float, ranks: np.ndarray | None = None) -> dict:
    """Per-day would-be fails, fixes, and pending fails. score=None means no model."""
    day = np.asarray(day)
    failed = np.asarray(failed).astype(bool)
    days, inverse = np.unique(day, return_inverse=True)
    trades = np.bincount(inverse)
    fails = np.bincount(inverse, weights=failed).astype(float)
    capacity = np.floor(capacity_share * trades)
    if score is None:
        worked_fails = fails * capacity / np.maximum(trades, 1)  # expected value of picking at random
    else:
        r = day_ranks(day, np.asarray(score, dtype=float)) if ranks is None else ranks
        worked = r < capacity[inverse]
        worked_fails = np.bincount(inverse, weights=worked & failed).astype(float)
    fixed = fix_rate * worked_fails
    return {
        "days": days,
        "trades": trades,
        "would_fail": fails,
        "worked_fails": worked_fails,
        "fixed": fixed,
        "pending": fails - fixed,
        "capacity": capacity,
    }


def compare(day, failed, score, capacity_share: float, fix_rate: float, cost_per_fail: float,
            penalty_per_fail: float, days_per_quarter: int = 63, ranks=None) -> dict:
    """With and without the model, with pending trades and quarterly exposure per business day
    scaled to a quarter."""
    base = simulate(day, failed, None, capacity_share, fix_rate)
    model = simulate(day, failed, score, capacity_share, fix_rate, ranks)
    n_days = max(1, len(base["days"]))
    per_fail = cost_per_fail + penalty_per_fail

    def quarterly(pending):
        return float(pending.sum() / n_days * days_per_quarter * per_fail)

    pend_base, pend_model = base["pending"].sum(), model["pending"].sum()
    return {
        "without_model": base,
        "with_model": model,
        "pending_without": float(pend_base),
        "pending_with": float(pend_model),
        "pending_reduction": float(1 - pend_model / pend_base) if pend_base else 0.0,
        "quarterly_exposure_without": quarterly(base["pending"]),
        "quarterly_exposure_with": quarterly(model["pending"]),
        "share_of_fails_worked_with_model": float(model["worked_fails"].sum() / max(1, base["would_fail"].sum())),
        "n_days": n_days,
    }
