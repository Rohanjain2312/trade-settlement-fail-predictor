"""Emit GitHub Actions ::error:: annotations for failing tests, so a failure can be
diagnosed from the check-run annotations alone."""

from __future__ import annotations

import os


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
