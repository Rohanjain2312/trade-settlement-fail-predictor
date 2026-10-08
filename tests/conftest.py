"""Shared fixtures, and GitHub Actions ::error:: annotations for failing tests, so a failure
can be diagnosed from the check-run annotations alone."""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest


def _escape(text: str) -> str:
    return text.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    if not os.environ.get("GITHUB_ACTIONS"):
        return
    for report in terminalreporter.stats.get("failed", []) + terminalreporter.stats.get("error", []):
        path, line, _ = report.location
        lines = [ln for ln in str(report.longrepr).splitlines() if ln.strip()]
        # Keep the lines that say what went wrong: assertion and error lines, then the tail.
        key = [ln for ln in lines if ln.startswith("E ") or "Error" in ln][:12] or lines[-12:]
        msg = _escape("\n".join(key)[:1800])
        terminalreporter.write_line(
            f"::error file={path},line={(line or 0) + 1},title={report.nodeid}::{msg}"
        )


def generate_in_memory(cfg: dict, intercept: float | None = None):
    """Generate a full dataset for cfg without checkpointing. Returns (ref, calibration, df)."""
    from src.data.generator import calibrate_intercept, generate_day, init_state, to_frame
    from src.data.reference_data import build_reference

    ref = build_reference(cfg)
    calibration = calibrate_intercept(ref, cfg) if intercept is None else {"intercept": intercept}
    state = init_state(ref, cfg)
    days = [generate_day(t, state, ref, cfg, calibration["intercept"]) for t in range(len(ref.cal.bdays))]
    return ref, calibration, to_frame(days)


@pytest.fixture(scope="session")
def smoke_data():
    """The smoke-config dataset, generated once per test session."""
    from src.config import load_config

    cfg = load_config("smoke")
    ref, calibration, df = generate_in_memory(cfg)
    return SimpleNamespace(cfg=cfg, ref=ref, calibration=calibration, df=df)
