"""docs/FEATURES.md stays in sync with the single source of truth."""

from __future__ import annotations

import re
from pathlib import Path

from src.features.definitions import FEATURES

DOC = (Path(__file__).resolve().parents[1] / "docs" / "FEATURES.md").read_text()


def test_features_doc_lists_every_feature_in_order_with_its_strength():
    rows = re.findall(r"^\| (\d+) \| `([a-z0-9_]+)` \|.*\| (strong|medium|weak) \|$", DOC, flags=re.M)
    assert [(int(i), n, s) for i, n, s in rows] == [(i + 1, f.name, f.strength) for i, f in enumerate(FEATURES)]
