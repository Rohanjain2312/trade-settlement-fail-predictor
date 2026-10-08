"""Run reports pushed to the public model repo's runs/ folder, readable without a token.

Files per run:
  runs/<run_id>/events.jsonl   one line per stage start and end
  runs/<run_id>/summary.json   stages done, metrics so far, timings, package versions
  runs/<run_id>/log_tail.txt   last 300 log lines, with anything that looks like a token redacted
  runs/latest.json             pointer to the newest run
Reports are pushed at the end of each stage, immediately on failure, and at most every
few minutes during long stages, so the pushes stay well under the Hub's rate limits.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
import traceback
import warnings
from datetime import datetime, timezone
from pathlib import Path

from src.pipeline.store import Store

log = logging.getLogger(__name__)

LOG_TAIL_LINES = 300
_TOKEN_PATTERNS = [
    re.compile(r"hf_[A-Za-z0-9]{8,}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9]{16,}"),
    re.compile(r"(?i)(authorization:\s*bearer\s+)\S+"),
]


def redact(text: str) -> str:
    token = os.environ.get("HF_TOKEN")
    if token:
        text = text.replace(token, "[REDACTED]")
    for pattern in _TOKEN_PATTERNS:
        text = pattern.sub(lambda m: (m.group(1) if m.groups() else "") + "[REDACTED]", text)
    return text


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def new_run_id(notebook: str, mode: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}_nb{notebook[:2]}_{mode}"


def setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%H:%M:%S")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in list(root.handlers):
        root.removeHandler(h)
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(log_path)):
        handler.setFormatter(fmt)
        root.addHandler(handler)
    logging.captureWarnings(True)
    # The private work repo is a model repo that holds Parquet checkpoints on purpose.
    warnings.filterwarnings("ignore", message="It seems that you are about to commit a data file")
    for noisy in ("httpx", "httpcore", "urllib3", "filelock", "huggingface_hub.file_download"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


class RunReporter:
    def __init__(
        self,
        store: Store,
        workdir: Path,
        notebook: str,
        mode: str,
        git_commit: str,
        log_path: Path,
        run_id: str | None = None,
        min_push_interval: float = 300.0,
    ):
        self.store = store
        self.notebook = notebook
        self.run_id = run_id or new_run_id(notebook, mode)
        self.dir = Path(workdir) / "runs"
        self.log_path = Path(log_path)
        self.min_push_interval = min_push_interval
        self._last_push = 0.0
        self._t0: dict[str, float] = {}
        self.events: list[dict] = []
        self.summary: dict = {
            "run_id": self.run_id,
            "notebook": notebook,
            "mode": mode,
            "git_commit": git_commit,
            "status": "running",
            "started_at": _now(),
            "updated_at": _now(),
            "failed_stage": None,
            "stages_done": [],
            "stages_skipped": [],
            "stages": {},
            "metrics": {},
            "versions": {},
            "error": None,
            "traceback": None,
        }

    # ---- events --------------------------------------------------------------------------

    def event(self, kind: str, stage: str | None = None, **data) -> None:
        self.events.append({"time": _now(), "event": kind, "stage": stage, **data})

    def stage_start(self, stage: str) -> None:
        self._t0[stage] = time.time()
        self.summary["stages"][stage] = {"status": "running", "started_at": _now()}
        self.event("stage_start", stage)
        log.info("=== stage %s: start", stage)

    def stage_end(self, stage: str, info: dict | None = None) -> None:
        seconds = round(time.time() - self._t0.get(stage, time.time()), 1)
        entry = self.summary["stages"].setdefault(stage, {})
        entry.update(status="done", ended_at=_now(), seconds=seconds, info=info or {})
        self.summary["stages_done"].append(stage)
        self.event("stage_end", stage, seconds=seconds)
        log.info("=== stage %s: done in %.1fs", stage, seconds)
        self.push()

    def stage_skipped(self, stage: str) -> None:
        self.summary["stages"][stage] = {"status": "skipped"}
        self.summary["stages_skipped"].append(stage)
        self.summary["stages_done"].append(stage)
        self.event("stage_skipped", stage)
        log.info("=== stage %s: already done, skipped", stage)

    def progress(self, stage: str, **data) -> None:
        """Record progress inside a long stage and push if the last push is old enough."""
        self.summary["stages"].setdefault(stage, {})["progress"] = data
        self.event("progress", stage, **data)
        if time.time() - self._last_push >= self.min_push_interval:
            self.push()

    def set(self, key: str, value) -> None:
        self.summary[key] = value

    def add_metrics(self, metrics: dict) -> None:
        self.summary["metrics"].update(metrics)

    # ---- outcomes ------------------------------------------------------------------------

    def fail(self, stage: str | None, exc: BaseException) -> None:
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        self.summary.update(
            status="failed",
            failed_stage=stage,
            error=redact(f"{type(exc).__name__}: {exc}")[:2000],
            traceback=redact(tb)[-8000:],
        )
        if stage:
            self.summary["stages"].setdefault(stage, {})["status"] = "failed"
        self.event("run_failed", stage, error=self.summary["error"])
        log.error("run failed in stage %s: %s", stage, self.summary["error"])
        self.push()

    def succeed(self) -> None:
        self.summary["status"] = "passed"
        self.event("run_passed")
        log.info("run passed")
        self.push()

    # ---- push ----------------------------------------------------------------------------

    def _write_files(self) -> list[str]:
        self.summary["updated_at"] = _now()
        run_dir = self.dir / self.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        rel = f"runs/{self.run_id}"
        (run_dir / "events.jsonl").write_text(
            redact("".join(json.dumps(e, default=str) + "\n" for e in self.events))
        )
        (run_dir / "summary.json").write_text(
            redact(json.dumps(self.summary, indent=1, default=str))
        )
        tail = ""
        if self.log_path.exists():
            lines = self.log_path.read_text(errors="replace").splitlines()[-LOG_TAIL_LINES:]
            tail = "\n".join(lines) + "\n"
        (run_dir / "log_tail.txt").write_text(redact(tail))
        latest = {
            "run_id": self.run_id,
            "notebook": self.summary["notebook"],
            "mode": self.summary["mode"],
            "git_commit": self.summary["git_commit"],
            "status": self.summary["status"],
            "failed_stage": self.summary["failed_stage"],
            "stages_done": self.summary["stages_done"],
            "error": self.summary["error"],
            "updated_at": self.summary["updated_at"],
            "space_url": self.summary.get("space_url"),
            "summary": f"{rel}/summary.json",
            "log_tail": f"{rel}/log_tail.txt",
        }
        (self.dir / "latest.json").write_text(redact(json.dumps(latest, indent=1, default=str)))
        return [f"{rel}/events.jsonl", f"{rel}/summary.json", f"{rel}/log_tail.txt", "runs/latest.json"]

    def push(self) -> None:
        """Best effort: a failed report push is logged and never stops the run."""
        files = self._write_files()
        try:
            self.store.commit(self.dir.parent, files, message=f"run report {self.run_id}")
            self._last_push = time.time()
        except Exception as err:  # noqa: BLE001 - reporting must not break the pipeline
            log.warning("could not push run report: %s", redact(str(err))[:300])


def report_crash(store: Store, workdir: Path) -> None:
    """Called by the notebook when the pipeline process exits without reporting a failure,
    for example when it is killed for running out of memory."""
    current = Path(workdir) / "current_run.json"
    if not current.exists():
        return
    info = json.loads(current.read_text())
    report_dir = Path(info["report_dir"])
    summary_path = report_dir / info["run_id"] / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    if summary.get("status") in ("failed", "passed"):
        return
    rep = RunReporter(
        store, report_dir.parent, info["notebook"], info["mode"], info["git_commit"],
        Path(info["log_path"]), run_id=info["run_id"],
    )
    if summary:
        rep.summary.update(summary)
    rep.summary.update(
        status="failed",
        error="pipeline process exited without a Python traceback (killed, or out of memory)",
    )
    rep.push()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--crashed", action="store_true")
    parser.add_argument("--workdir", required=True)
    args = parser.parse_args()
    if args.crashed:
        from src.config import load_config
        from src.pipeline.store import HubStore

        cfg = load_config("full")
        report_crash(HubStore(cfg["project"]["reports_repo"], "model"), Path(args.workdir))


if __name__ == "__main__":
    main()
