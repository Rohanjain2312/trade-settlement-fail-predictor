"""Shared stage loop for the two pipelines (generate and train).

A stage function receives the Context and returns the list of output files (relative to
the working directory) to record in the manifest. Stages marked `always_run` prepare the
local disk (for example pulling data) and leave nothing in the work repo.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from src.pipeline.checkpoint import Checkpoint, StageSpec
from src.pipeline.run_report import RunReporter

log = logging.getLogger(__name__)


@dataclass
class Context:
    cfg: dict
    workdir: Path
    ckpt: Checkpoint
    report: RunReporter
    stores: dict  # "dataset", "model", "work", "reports" -> Store
    device: str = "cpu"
    state: dict = field(default_factory=dict)  # values shared between stages in one process


@dataclass(frozen=True)
class Stage:
    spec: StageSpec
    fn: Callable[[Context], list[str]]
    always_run: bool = False

    @property
    def name(self) -> str:
        return self.spec.name


def run_stages(stages: list[Stage], ctx: Context, from_stage: str | None, force: list[str]) -> None:
    names = [s.name for s in stages]
    unknown = [s for s in [from_stage, *force] if s is not None and s not in names]
    if unknown:
        raise ValueError(f"config/run.yaml names unknown stages {unknown}; stages are {names}")
    rerun = False
    current = None
    try:
        for stage in stages:
            current = stage.name
            if stage.always_run:
                ctx.report.stage_start(stage.name)
                stage.fn(ctx)
                ctx.report.stage_end(stage.name, ctx.state.get(f"info:{stage.name}"))
                continue
            forced = stage.name in force or stage.name == from_stage
            if not rerun and not forced and ctx.ckpt.is_done(stage.spec):
                ctx.report.stage_skipped(stage.name)
                continue
            rerun = True
            ctx.ckpt.begin(stage.spec, fresh=forced)
            ctx.report.stage_start(stage.name)
            outputs = stage.fn(ctx)
            ctx.ckpt.finish(stage.name, outputs)
            ctx.report.stage_end(stage.name, ctx.state.get(f"info:{stage.name}"))
    except BaseException as exc:
        ctx.report.fail(current, exc)
        raise


def write_current_run(workdir: Path, report: RunReporter, mode: str, git_commit: str) -> None:
    """Lets the notebook report a crash that killed the process before Python could."""
    (Path(workdir) / "current_run.json").write_text(
        json.dumps(
            {
                "run_id": report.run_id,
                "notebook": report.notebook,
                "mode": mode,
                "git_commit": git_commit,
                "log_path": str(report.log_path),
            }
        )
    )
