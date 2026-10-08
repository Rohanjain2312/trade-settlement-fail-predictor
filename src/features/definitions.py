"""Single source of truth for the 20 model features.

Every other module (generator, preprocessing, models, app, docs) reads feature names,
types, order, and explanations from here. Columns that are not listed in FEATURES are
never model inputs.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Feature:
    name: str
    kind: str  # "categorical", "numeric", "integer", or "binary"
    group: str
    label: str
    description: str  # what it is
    mechanism: str  # how it affects settlement
    strength: str  # planted strength in the generator: "strong", "medium", or "weak"
    direction: int  # +1: risk rises with the value, -1: risk falls, 0: categorical
    dtype: str  # storage dtype in Parquet
    levels: tuple[str, ...] = ()  # categorical levels, in display order
    reference_level: str | None = None  # lowest-risk categorical level (no added risk)
    may_be_missing: bool = False  # the generator blanks a small share of values


FEATURES: tuple[Feature, ...] = (
    Feature(
        name="ssi_match_status",
        kind="categorical",
        group="SSI and instructions",
        label="SSI match status",
        description=(
            "Whether the account, custodian, and BIC details on the instruction agree with "
            "the standing settlement instructions (SSI) on file."
        ),
        mechanism=(
            "A mismatch or missing instruction means cash and securities cannot be routed, so "
            "the instruction is rejected or sits unmatched until someone repairs it. SSI "
            "problems are the largest single cause of fails."
        ),
        strength="strong",
        direction=0,
        dtype="category",
        levels=("matched", "mismatch", "missing"),
        reference_level="matched",
    ),
    Feature(
        name="ssi_age_days",
        kind="numeric",
        group="SSI and instructions",
        label="SSI record age (days)",
        description="Days since the counterparty's SSI record was last verified.",
        mechanism=(
            "Old records are more likely to point at closed or changed accounts, so risk rises "
            "with age even while the record still shows as matched."
        ),
        strength="medium",
        direction=1,
        dtype="float32",
        may_be_missing=True,
    ),
    Feature(
        name="instruction_hour_bucket",
        kind="categorical",
        group="SSI and instructions",
        label="Instruction time bucket",
        description="Time of day the settlement instruction was submitted.",
        mechanism=(
            "Overnight and early-morning instructions go in with less market visibility and no "
            "one on hand to fix exceptions, and late ones miss depository cutoffs."
        ),
        strength="medium",
        direction=0,
        dtype="category",
        levels=("overnight", "early_morning", "business_hours", "late"),
        reference_level="business_hours",
    ),
    Feature(
        name="hours_to_confirmation",
        kind="numeric",
        group="Matching and timing",
        label="Hours to confirmation",
        description="Hours from execution to counterparty confirmation.",
        mechanism=(
            "Slow confirmation leaves less time to catch breaks before settlement date, and an "
            "unconfirmed trade is the classic 'counterparty does not know the trade' fail."
        ),
        strength="strong",
        direction=1,
        dtype="float32",
        may_be_missing=True,
    ),
    Feature(
        name="amendment_count",
        kind="integer",
        group="Matching and timing",
        label="Amendment count",
        description="Number of amends or cancel-and-correct events after execution.",
        mechanism=(
            "Every amendment restarts matching and can reintroduce a break in price, quantity, "
            "or date."
        ),
        strength="medium",
        direction=1,
        dtype="int8",
    ),
    Feature(
        name="allocation_delay_hrs",
        kind="numeric",
        group="Matching and timing",
        label="Allocation delay (hours)",
        description=(
            "Hours from block execution to allocation across the underlying accounts "
            "(0 for trades that are not blocks)."
        ),
        mechanism=(
            "Late allocation delays the settlement instruction and squeezes the time left to "
            "fix problems."
        ),
        strength="medium",
        direction=1,
        dtype="float32",
        may_be_missing=True,
    ),
    Feature(
        name="chain_depth",
        kind="integer",
        group="Matching and timing",
        label="Settlement chain depth",
        description="Number of upstream trades whose delivery this trade depends on (0 means none).",
        mechanism=(
            "A fail upstream cascades down the chain, so the deeper the chain, the more places "
            "a fail can start."
        ),
        strength="medium",
        direction=1,
        dtype="int8",
    ),
    Feature(
        name="obligation_coverage_ratio",
        kind="numeric",
        group="Obligation",
        label="Obligation coverage ratio",
        description=(
            "Share of what is owed on settlement date (securities for sells, cash for buys) "
            "that is already available or confirmed incoming."
        ),
        mechanism=(
            "A ratio below 1 is a shortfall. Not having the securities or cash to deliver is "
            "the second biggest cause of fails."
        ),
        strength="strong",
        direction=-1,
        dtype="float32",
    ),
    Feature(
        name="cpty_fail_rate_30d",
        kind="numeric",
        group="Counterparty",
        label="Counterparty fail rate (30d)",
        description=(
            "Share of this counterparty's settled trades that failed in the prior 30 days "
            "(smoothed toward the overall rate when there are few trades)."
        ),
        mechanism=(
            "Counterparties with recent fails tend to keep failing because their operational "
            "problems persist."
        ),
        strength="strong",
        direction=1,
        dtype="float32",
    ),
    Feature(
        name="cpty_type",
        kind="categorical",
        group="Counterparty",
        label="Counterparty type",
        description="Kind of counterparty.",
        mechanism=(
            "Different types have different levels of automation and operational maturity, "
            "which shifts the baseline fail rate regardless of recent history."
        ),
        strength="medium",
        direction=0,
        dtype="category",
        levels=("custodian", "broker_dealer", "asset_manager", "hedge_fund", "corporate_treasury"),
        reference_level="custodian",
    ),
    Feature(
        name="pair_history_trades_90d",
        kind="integer",
        group="Counterparty",
        label="Pair history (trades, 90d)",
        description="Trades between this desk and this counterparty in the prior 90 days.",
        mechanism=(
            "Pairs that trade often have tested instructions and routines. New or rare pairs go "
            "through manual setup and have more breaks."
        ),
        strength="weak",
        direction=-1,
        dtype="int32",
    ),
    Feature(
        name="security_fail_rate_30d",
        kind="numeric",
        group="Security",
        label="Security fail rate (30d)",
        description=(
            "Fail rate of this security over its settled trades in the prior 30 days "
            "(smoothed toward the overall rate when there are few trades)."
        ),
        mechanism=(
            "Scarce or hard-to-source securities fail repeatedly, for example when they trade "
            "on special."
        ),
        strength="medium",
        direction=1,
        dtype="float32",
    ),
    Feature(
        name="corporate_action_in_window",
        kind="binary",
        group="Security",
        label="Corporate action in window",
        description="A dividend, coupon, or record date falls inside the settlement window.",
        mechanism=(
            "Entitlement and position adjustments around events create holds and claims that "
            "delay settlement."
        ),
        strength="medium",
        direction=1,
        dtype="int8",
    ),
    Feature(
        name="notional_vs_cpty_median",
        kind="numeric",
        group="Trade economics",
        label="Size vs counterparty median",
        description=(
            "Trade size divided by the counterparty's median trade size over the prior 90 days."
        ),
        mechanism=(
            "Unusually large trades are harder to source or fund and more often get manual checks."
        ),
        strength="medium",
        direction=1,
        dtype="float32",
        may_be_missing=True,
    ),
    Feature(
        name="abs_price_deviation_bps",
        kind="numeric",
        group="Trade economics",
        label="Price deviation (bps)",
        description=(
            "Absolute distance of the trade price from the market reference at execution, in "
            "basis points."
        ),
        mechanism="Off-market prices are more likely to be disputed at the matching step.",
        strength="weak",
        direction=1,
        dtype="float32",
        may_be_missing=True,
    ),
    Feature(
        name="asset_class",
        kind="categorical",
        group="Trade economics",
        label="Asset class",
        description="Type of instrument.",
        mechanism=(
            "Each class settles through different infrastructure and rules, with different "
            "baseline fail rates."
        ),
        strength="medium",
        direction=0,
        dtype="category",
        levels=("equity", "corporate_bond", "government_bond", "etf", "repo"),
        reference_level="government_bond",
    ),
    Feature(
        name="is_cross_border",
        kind="binary",
        group="Market and calendar",
        label="Cross-border",
        description="Settlement involves a market or depository outside the home market.",
        mechanism="Extra intermediaries, time zones, and local rules add failure points.",
        strength="medium",
        direction=1,
        dtype="int8",
    ),
    Feature(
        name="market_volatility_level",
        kind="numeric",
        group="Market and calendar",
        label="Market volatility level",
        description="Volatility index level on the trade date.",
        mechanism=(
            "Volume spikes and strained operations in volatile periods push fails up."
        ),
        strength="medium",
        direction=1,
        dtype="float32",
    ),
    Feature(
        name="holiday_in_settlement_window",
        kind="binary",
        group="Market and calendar",
        label="Holiday in settlement window",
        description=(
            "A market holiday falls between trade date and settlement date in either market."
        ),
        mechanism="Holidays shrink the working window and shift cutoffs, especially across markets.",
        strength="weak",
        direction=1,
        dtype="int8",
    ),
    Feature(
        name="is_period_end",
        kind="binary",
        group="Market and calendar",
        label="Period end",
        description="Settlement date is a month-end or quarter-end.",
        mechanism=(
            "Balance-sheet and funding pressure plus higher volumes at period end raise fails."
        ),
        strength="weak",
        direction=1,
        dtype="int8",
    ),
)

FEATURE_NAMES: tuple[str, ...] = tuple(f.name for f in FEATURES)
BY_NAME: dict[str, Feature] = {f.name: f for f in FEATURES}
CATEGORICAL: tuple[str, ...] = tuple(f.name for f in FEATURES if f.kind == "categorical")
BINARY: tuple[str, ...] = tuple(f.name for f in FEATURES if f.kind == "binary")
# Numeric in the modeling sense: continuous and integer features plus 0/1 flags.
NUMERIC: tuple[str, ...] = tuple(f.name for f in FEATURES if f.kind != "categorical")
CONTINUOUS: tuple[str, ...] = tuple(f.name for f in FEATURES if f.kind in ("numeric", "integer"))

# Columns stored with each trade that are never model inputs.
ID_COLUMNS: tuple[str, ...] = (
    "trade_id",
    "trade_date",
    "settle_date",
    "desk_id",
    "cpty_id",
    "sec_id",
    "notional",
)
LABEL = "failed"
TARGET_COLUMNS: tuple[str, ...] = ("failed", "fail_reason", "scenario_mask")
NON_FEATURE_COLUMNS: tuple[str, ...] = ID_COLUMNS + TARGET_COLUMNS
TRADE_COLUMNS: tuple[str, ...] = ID_COLUMNS + FEATURE_NAMES + TARGET_COLUMNS

STRENGTH_ORDER = {"strong": 3, "medium": 2, "weak": 1}
GROUPS: tuple[str, ...] = tuple(dict.fromkeys(f.group for f in FEATURES))

assert len(FEATURES) == 20, "the project is defined around exactly 20 features"
assert len(set(FEATURE_NAMES)) == 20
assert not set(FEATURE_NAMES) & set(NON_FEATURE_COLUMNS)


def model_inputs(df: pd.DataFrame) -> pd.DataFrame:
    """Return exactly the 20 model features, in the canonical order, with canonical dtypes."""
    missing = [c for c in FEATURE_NAMES if c not in df.columns]
    if missing:
        raise KeyError(f"missing feature columns: {missing}")
    X = df.loc[:, list(FEATURE_NAMES)].copy()
    for f in FEATURES:
        if f.kind == "categorical":
            X[f.name] = pd.Categorical(X[f.name].astype("object"), categories=list(f.levels))
        else:
            X[f.name] = X[f.name].astype("float32")
    return X


def feature_schema() -> dict:
    """JSON-serializable schema published with the models."""
    return {
        "features": [
            {
                "index": i,
                "name": f.name,
                "kind": f.kind,
                "group": f.group,
                "dtype": f.dtype,
                "levels": list(f.levels),
                "reference_level": f.reference_level,
                "direction": f.direction,
                "strength": f.strength,
            }
            for i, f in enumerate(FEATURES)
        ],
        "label": LABEL,
        "never_inputs": list(NON_FEATURE_COLUMNS),
    }
