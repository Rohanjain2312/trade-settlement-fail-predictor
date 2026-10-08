"""Shared stage loop for the two pipelines (generate and train).

A stage function receives the Context and returns the list of output files (relative to
the working directory) to record in the manifest. Stages marked `always_run` prepare the
local disk (for example pulling data) and leave nothing in the work repo.
"""

from __future__ import annotations

import json
import logging
import os
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
                "report_dir": str(report.dir),
            }
        )
    )


# ---- running a notebook's pipeline --------------------------------------------------------

def default_workdir() -> Path:
    return Path(os.environ.get("TSFP_WORKDIR", "/content/tsfp_work"))


def hub_stores(cfg: dict) -> dict:
    from src.config import repo_ids
    from src.pipeline.store import HubStore

    ids = repo_ids(cfg)
    return {
        "work": HubStore(ids["work"], "model"),
        "dataset": HubStore(ids["dataset"], "dataset"),
        "model": HubStore(ids["model"], "model"),
        "space": HubStore(ids["space"], "space"),
        "reports": HubStore(ids["reports"], "model"),
    }


def local_stores(root: Path, cfg: dict) -> dict:
    """Directories standing in for the Hub repos (CI and tests)."""
    from src.config import repo_ids
    from src.pipeline.store import LocalDirStore

    ids = repo_ids(cfg)
    return {k: LocalDirStore(Path(root) / ids[k].replace("/", "__")) for k in ("work", "dataset", "model", "space", "reports")}


def smoke_marker(notebook: str, git_commit: str) -> str:
    return f"runs/smoke_passed/{notebook}_{git_commit}.json"


def resolve_modes(mode: str, notebook: str, git_commit: str, reports_store) -> list[str]:
    if mode != "auto":
        return [mode]
    if reports_store.exists(smoke_marker(notebook, git_commit)):
        return ["full"]
    return ["smoke", "full"]


def run_mode(notebook: str, mode: str, stages: list[Stage], stores: dict, workdir: Path,
             from_stage: str | None = None, force: list[str] = (), on_hub: bool = True,
             cfg_overrides: dict | None = None) -> Context:
    from src.config import git_commit, load_config, repo_ids
    from src.pipeline.env_check import versions
    from src.pipeline.run_report import new_run_id, setup_logging

    cfg = load_config(mode, overrides=cfg_overrides)
    commit = git_commit()
    mode_dir = Path(workdir) / mode
    run_id = new_run_id(notebook, mode)
    log_path = mode_dir / "logs" / f"{run_id}.log"
    setup_logging(log_path)
    report = RunReporter(stores["reports"], mode_dir, notebook, mode, commit, log_path, run_id=run_id)
    write_current_run(workdir, report, mode, commit)
    report.set("versions", versions())
    report.set("repos", repo_ids(cfg))
    log.info("run %s: notebook %s, mode %s, commit %s", run_id, notebook, mode, commit[:8])
    try:
        if on_hub:
            from src.hub.repos import ensure_repos

            ensure_repos(cfg)
        report.push()
        ckpt = Checkpoint(stores["work"], mode_dir, cfg, commit, [s.name for s in stages])
        ctx = Context(cfg=cfg, workdir=mode_dir, ckpt=ckpt, report=report, stores=stores)
        ctx.state["on_hub"] = on_hub
        run_stages(stages, ctx, from_stage, list(force))
        if mode == "smoke":
            marker = mode_dir / smoke_marker(notebook, commit)
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text(json.dumps({"run_id": run_id, "git_commit": commit}))
            stores["reports"].commit(mode_dir, [smoke_marker(notebook, commit)], message="smoke passed")
    except BaseException as exc:
        if report.summary["status"] != "failed":
            report.fail(None, exc)
        raise
    report.succeed()
    return ctx


def run_notebook(notebook: str, build_stages) -> None:
    """Entry point used by `python -m src.pipeline.run_generate` and `run_train` in Colab."""
    from src.config import git_commit, load_config, load_run_config

    run = load_run_config()
    workdir = default_workdir()
    reports = hub_stores(load_config("full"))["reports"]
    for mode in resolve_modes(run["mode"], notebook, git_commit(), reports):
        cfg = load_config(mode)
        run_mode(notebook, mode, build_stages(), hub_stores(cfg), workdir, run["from_stage"], run["force"])
