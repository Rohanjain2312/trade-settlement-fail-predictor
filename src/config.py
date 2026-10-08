"""Load and validate the YAML configuration, and hash config sections for checkpointing."""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "config"
MODES = ("smoke", "full", "auto")


def _load_yaml(path: Path) -> dict:
    with open(path) as fh:
        data = yaml.safe_load(fh)
    return data or {}


def deep_merge(base: dict, override: dict) -> dict:
    """Return base with override merged in. Nested dicts merge, everything else replaces."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_run_config(config_dir: Path = CONFIG_DIR) -> dict:
    run = _load_yaml(config_dir / "run.yaml")
    mode = run.get("mode")
    if mode not in MODES:
        raise ValueError(f"config/run.yaml: mode must be one of {MODES}, got {mode!r}")
    from_stage = run.get("from_stage")
    if from_stage is not None and not isinstance(from_stage, str):
        raise ValueError("config/run.yaml: from_stage must be a stage name or null")
    force = run.get("force") or []
    if not isinstance(force, list) or not all(isinstance(s, str) for s in force):
        raise ValueError("config/run.yaml: force must be a list of stage names")
    return {"mode": mode, "from_stage": from_stage, "force": force}


def load_config(mode: str, config_dir: Path = CONFIG_DIR, overrides: dict | None = None) -> dict:
    """Resolved config for a concrete mode ("smoke" or "full")."""
    if mode not in ("smoke", "full"):
        raise ValueError(f"load_config needs a concrete mode, got {mode!r}")
    cfg = _load_yaml(config_dir / "params.yaml")
    scenarios_path = config_dir / "scenarios.yaml"
    if scenarios_path.exists():
        cfg["scenarios"] = _load_yaml(scenarios_path)
    if mode == "smoke":
        cfg = deep_merge(cfg, _load_yaml(config_dir / "smoke.yaml"))
    if overrides:
        cfg = deep_merge(cfg, overrides)
    cfg["mode"] = mode
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict) -> None:
    data = cfg["data"]
    if not 0.02 <= data["target_fail_rate"] <= 0.05:
        raise ValueError("data.target_fail_rate must be between 0.02 and 0.05")
    if data["n_days"] < 20 or data["trades_per_day"] < 10:
        raise ValueError("data.n_days must be >= 20 and data.trades_per_day >= 10")
    split = cfg["split"]
    tr, va, te = split["train_months"], split["val_months"], split["test_months"]
    if not (1 <= tr[0] <= tr[1] < va[0] <= va[1] < te[0] <= te[1]):
        raise ValueError("split months must be ordered: train < validation < test")
    if split["gap_days"] < 0:
        raise ValueError("split.gap_days must be >= 0")
    for name, share in data["missing_share"].items():
        if not 0 <= share <= data["max_missing_share"]:
            raise ValueError(f"data.missing_share.{name} exceeds data.max_missing_share")


def config_hash(cfg: dict, keys: tuple[str, ...] | list[str]) -> str:
    """Stable hash of the listed top-level config sections."""
    payload = {k: cfg.get(k) for k in sorted(keys)}
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def repo_ids(cfg: dict) -> dict[str, str]:
    """Hub repo ids for the current mode. Smoke runs use the -smoke repos."""
    p = cfg["project"]
    suffix = p["smoke_suffix"] if cfg["mode"] == "smoke" else ""
    return {
        "dataset": p["dataset_repo"] + suffix,
        "model": p["model_repo"] + suffix,
        "work": p["work_repo"] + suffix,
        "space": p["space_repo"] + suffix,
        "reports": p["reports_repo"],
    }


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
