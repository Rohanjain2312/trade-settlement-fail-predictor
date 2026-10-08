"""Time-based splits and the preprocessing pipelines shared by the models.

LR and SVM pipelines (imblearn Pipeline, so a sampler runs only when fitting):
    encode:  ordinal-encode the 4 categoricals, median-impute the 16 numeric features
    smote:   SMOTENC on those columns (optional), with the categorical column indices
    expand:  one-hot encode the categoricals, scale the numerics
    model:   the estimator
Because SMOTENC sits inside the pipeline, it only ever sees the rows the pipeline is fitted
on: the training set, or a training fold inside cross-validation. Validation and test rows
are never resampled and never leak into the synthetic points.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from src.features.definitions import BY_NAME, CATEGORICAL, LABEL, NUMERIC, model_inputs

RESAMPLING = ("none", "class_weight", "smote")
N_CAT = len(CATEGORICAL)


def load_trades(data_dir: Path) -> pd.DataFrame:
    files = sorted((Path(data_dir) / "data" / "trades").glob("trades_*.parquet"))
    if not files:
        raise FileNotFoundError(f"no trade files under {data_dir}/data/trades")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def split_frames(df: pd.DataFrame, cfg: dict) -> dict[str, pd.DataFrame]:
    """Split by trade month, then drop the last gap_days business days of train and
    validation so no trade that is still settling straddles a boundary."""
    sp = cfg["split"]
    month = pd.to_datetime(df["trade_date"]).dt.to_period("M")
    months = sorted(month.unique())
    index = month.map({m: i + 1 for i, m in enumerate(months)})
    out = {}
    for name, (lo, hi) in (("train", sp["train_months"]), ("val", sp["val_months"]), ("test", sp["test_months"])):
        part = df[(index >= lo) & (index <= hi)]
        if name != "test" and sp["gap_days"] > 0:
            days = np.sort(part["trade_date"].unique())
            part = part[part["trade_date"] < days[-sp["gap_days"]]]
        out[name] = part.reset_index(drop=True)
    for a, b in (("train", "val"), ("val", "test")):
        if len(out[a]) and len(out[b]) and out[a]["settle_date"].max() >= out[b]["trade_date"].min():
            raise ValueError(f"{a} trades still settle after {b} starts; increase split.gap_days")
    return out


def xy(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    return model_inputs(df), df[LABEL].to_numpy().astype(int)


def _encode() -> ColumnTransformer:
    cats = [list(BY_NAME[c].levels) for c in CATEGORICAL]
    return ColumnTransformer(
        [
            ("cat", OrdinalEncoder(categories=cats), list(CATEGORICAL)),
            ("num", SimpleImputer(strategy="median"), list(NUMERIC)),
        ],
        verbose_feature_names_out=False,
    )


def _expand() -> ColumnTransformer:
    cats = [np.arange(len(BY_NAME[c].levels), dtype=float) for c in CATEGORICAL]
    return ColumnTransformer(
        [
            ("onehot", OneHotEncoder(categories=cats, handle_unknown="ignore"), list(range(N_CAT))),
            ("scale", StandardScaler(), list(range(N_CAT, N_CAT + len(NUMERIC)))),
        ]
    )


def build_pipeline(estimator, resampling: str, seed: int, k_neighbors: int = 5) -> Pipeline:
    if resampling not in RESAMPLING:
        raise ValueError(f"resampling must be one of {RESAMPLING}")
    steps = [("encode", _encode())]
    if resampling == "smote":
        steps.append(
            ("smote", SMOTENC(categorical_features=list(range(N_CAT)), k_neighbors=k_neighbors, random_state=seed))
        )
    steps += [("expand", _expand()), ("model", estimator)]
    return Pipeline(steps)


def expanded_feature_names() -> list[str]:
    """Column names after one-hot encoding and scaling, in pipeline output order."""
    names = [f"{c}={lv}" for c in CATEGORICAL for lv in BY_NAME[c].levels]
    return names + list(NUMERIC)


def stratified_subsample(X: pd.DataFrame, y: np.ndarray, n: int, seed: int):
    """Keep the class ratio while taking n rows (all rows if there are fewer)."""
    if len(y) <= n:
        return X, y
    rng = np.random.default_rng(seed)
    idx = []
    for cls in (0, 1):
        rows = np.flatnonzero(y == cls)
        take = int(round(n * len(rows) / len(y)))
        idx.append(rng.choice(rows, size=min(take, len(rows)), replace=False))
    idx = np.sort(np.concatenate(idx))
    return X.iloc[idx].reset_index(drop=True), y[idx]
