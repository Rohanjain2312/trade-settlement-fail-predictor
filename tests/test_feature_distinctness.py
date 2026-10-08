from __future__ import annotations

from src.data.validate import check_distinctness


def test_features_are_distinct(smoke_data):
    result = check_distinctness(smoke_data.df, smoke_data.cfg)
    assert result["passed"], result["failures"]
