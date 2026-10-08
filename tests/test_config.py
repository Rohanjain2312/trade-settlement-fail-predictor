from __future__ import annotations

import pytest

from src.config import config_hash, deep_merge, load_config, load_run_config, repo_ids
from src.features import definitions as d


@pytest.mark.parametrize("mode", ["smoke", "full"])
def test_configs_load_and_validate(mode):
    cfg = load_config(mode)
    assert cfg["mode"] == mode
    assert 0.02 <= cfg["data"]["target_fail_rate"] <= 0.05
    ids = repo_ids(cfg)
    assert ids["reports"] == cfg["project"]["reports_repo"]
    if mode == "smoke":
        assert all(ids[k].endswith("-smoke") for k in ("dataset", "model", "work", "space"))
    else:
        assert not any(ids[k].endswith("-smoke") for k in ids)


def test_run_config_is_valid():
    run = load_run_config()
    assert run["mode"] in ("smoke", "full", "auto")


def test_config_hash_only_depends_on_listed_sections():
    cfg = load_config("full")
    other = deep_merge(cfg, {"train": {"ops_capacity_share": 0.5}})
    assert config_hash(cfg, ["data", "effects"]) == config_hash(other, ["data", "effects"])
    assert config_hash(cfg, ["train"]) != config_hash(other, ["train"])


def test_every_feature_has_a_planted_effect_with_the_right_sign():
    cfg = load_config("full")
    effects = cfg["effects"]
    assert set(effects) == set(d.FEATURE_NAMES)
    for f in d.FEATURES:
        values = effects[f.name]
        if f.kind == "categorical":
            # Every non-reference level adds risk relative to the lowest-risk reference level.
            assert set(values) == set(f.levels) - {f.reference_level}
            assert all(v > 0 for v in values.values())
        else:
            assert values["weight" if "weight" in values else "per_year"] > 0


def test_feature_definitions_are_complete():
    assert len(d.FEATURES) == 20
    for f in d.FEATURES:
        assert f.description and f.mechanism and f.label
        assert f.strength in d.STRENGTH_ORDER
        assert (f.kind == "categorical") == bool(f.levels)
        if f.kind == "categorical":
            assert f.reference_level in f.levels
    assert {f.strength for f in d.FEATURES} == {"strong", "medium", "weak"}
