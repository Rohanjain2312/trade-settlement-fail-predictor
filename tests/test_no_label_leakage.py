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


def test_splits_follow_time_with_a_gap_and_nothing_straddles_a_boundary(smoke_data):
    import numpy as np

    from src.models.preprocess import split_frames

    df, cfg = smoke_data.df, smoke_data.cfg
    s = split_frames(df, cfg)
    all_days = np.sort(df["trade_date"].unique())
    for a, b in (("train", "val"), ("val", "test")):
        assert s[a]["trade_date"].max() < s[b]["trade_date"].min()
        # Business days between the two splits that belong to neither: at least the configured gap.
        skipped = ((all_days > s[a]["trade_date"].max()) & (all_days < s[b]["trade_date"].min())).sum()
        assert skipped >= cfg["split"]["gap_days"]
        # Every earlier-split trade has settled before the later split starts.
        assert s[a]["settle_date"].max() < s[b]["trade_date"].min()
    assert not set(s["train"]["trade_id"]) & set(s["test"]["trade_id"])
