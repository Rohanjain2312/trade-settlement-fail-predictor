"""The label, the fail reason, the scenario mask, IDs, dates, and raw notional must never be
model inputs. A leak would make every result meaningless."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from src.features import definitions as d


def test_definitions_keep_targets_and_ids_out_of_features():
    assert len(d.FEATURE_NAMES) == 20
    for col in ("failed", "fail_reason", "scenario_mask", "trade_id", "trade_date", "settle_date",
                "cpty_id", "sec_id", "desk_id", "notional"):
        assert col not in d.FEATURE_NAMES
        assert col in d.NON_FEATURE_COLUMNS


def test_generated_table_has_the_documented_columns(smoke_data):
    assert list(smoke_data.df.columns) == list(d.TRADE_COLUMNS)


def test_model_inputs_select_exactly_the_20_features(smoke_data):
    X = d.model_inputs(smoke_data.df)
    assert list(X.columns) == list(d.FEATURE_NAMES)
    assert not set(X.columns) & set(d.NON_FEATURE_COLUMNS)
    for name in d.CATEGORICAL:
        assert list(X[name].cat.categories) == list(d.BY_NAME[name].levels)


def test_no_single_feature_reveals_the_label(smoke_data):
    df = smoke_data.df
    y = df["failed"].to_numpy()
    X = d.model_inputs(df)
    for name in d.FEATURE_NAMES:
        col = X[name]
        x = col.cat.codes.to_numpy().astype(float) if name in d.CATEGORICAL else col.to_numpy(dtype=float)
        x = np.where(np.isnan(x), np.nanmedian(x), x)
        auc = roc_auc_score(y, x)
        assert max(auc, 1 - auc) < 0.85, f"{name} alone separates the label (AUC {auc:.3f})"
