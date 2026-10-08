"""Manifest-based checkpointing on top of a Store.

Colab's local disk is the working copy and the work repo is the saved copy. Every save
commits a unit's files together with the updated manifest.json, so a checkpoint is all or
nothing. A stage is skipped only if it is marked done, its outputs match their checksums,
and neither its config hash nor its code hash has changed.
"""

from __future__ import annotations

import glob
import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from src.config import ROOT, config_hash
from src.pipeline.store import MANIFEST, Store

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class StageSpec:
    name: str
    sources: tuple[str, ...]  # repo-relative globs whose contents form the code hash
    config_keys: tuple[str, ...]  # top-level config sections that form the config hash
    uses_dataset: bool = False  # fingerprint includes the dataset revision


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def code_hash(sources: tuple[str, ...]) -> str:
    h = hashlib.sha256()
    files = sorted({p for pattern in sources for p in glob.glob(str(ROOT / pattern), recursive=True)})
    for path in files:
        if Path(path).is_file():
            h.update(Path(path).relative_to(ROOT).as_posix().encode())
            h.update(Path(path).read_bytes())
    return h.hexdigest()[:16]


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class Checkpoint:
    def __init__(
        self,
        store: Store,
        workdir: Path,
        cfg: dict,
        git_commit: str,
        stage_order: list[str],
    ):
        self.store = store
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.cfg = cfg
        self.git_commit = git_commit
        self.stage_order = stage_order
        self.dataset_revision: str | None = None
        self.manifest: dict = {"version": 1, "stages": {}}
        self._stale: dict[str, set[str]] = {}

    # ---- restore and inspect -------------------------------------------------------------

    def restore(self, allow_patterns: list[str] | None = None) -> None:
        """Download the saved copy into the working directory and load the manifest."""
        patterns = None if allow_patterns is None else [MANIFEST, *allow_patterns]
        self.store.download_all(self.workdir, patterns)
        path = self.workdir / MANIFEST
        if path.exists():
            self.manifest = json.loads(path.read_text())
        log.info("restored %d stage entries from the work store", len(self.manifest["stages"]))

    def fingerprint(self, spec: StageSpec) -> dict:
        return {
            "config_hash": config_hash(self.cfg, spec.config_keys),
            "code_hash": code_hash(spec.sources),
            "dataset_revision": self.dataset_revision if spec.uses_dataset else None,
        }

    def entry(self, name: str) -> dict | None:
        return self.manifest["stages"].get(name)

    def _files_ok(self, files: dict) -> bool:
        for rel, meta in files.items():
            path = self.workdir / rel
            if not path.is_file() or path.stat().st_size != meta["size"]:
                return False
            if sha256_file(path) != meta["sha256"]:
                return False
        return True

    def is_done(self, spec: StageSpec) -> bool:
        e = self.entry(spec.name)
        return (
            e is not None
            and e["status"] == "done"
            and e["fingerprint"] == self.fingerprint(spec)
            and self._files_ok(e["outputs"])
        )

    def unit_done(self, stage: str, unit: str) -> bool:
        e = self.entry(stage)
        return e is not None and unit in e["units"] and self._files_ok(e["units"][unit])

    def done_units(self, stage: str) -> list[str]:
        e = self.entry(stage)
        return [] if e is None else [u for u in e["units"] if self.unit_done(stage, u)]

    # ---- record progress -----------------------------------------------------------------

    def begin(self, spec: StageSpec, fresh: bool = False) -> None:
        """Start or resume a stage. Units saved under the same fingerprint are kept."""
        fp = self.fingerprint(spec)
        e = self.entry(spec.name)
        if fresh or e is None or e["fingerprint"] != fp:
            if e is not None:
                self._stale.setdefault(spec.name, set()).update(_entry_files(e))
            e = {"fingerprint": fp, "units": {}, "outputs": {}, "started_at": _now()}
            log.info("stage %s starts fresh", spec.name)
        else:
            log.info("stage %s resumes with %d saved units", spec.name, len(e["units"]))
        e.update(status="in_progress", git_commit=self.git_commit)
        self.manifest["stages"][spec.name] = e
        # Every later stage has to re-run from scratch, even after a restart, because its
        # inputs are about to change.
        if spec.name in self.stage_order:
            for later in self.stage_order[self.stage_order.index(spec.name) + 1 :]:
                le = self.manifest["stages"].get(later)
                if le is not None:
                    self._stale.setdefault(later, set()).update(_entry_files(le))
                    le.update(status="stale", units={}, outputs={})

    def save_unit(self, stage: str, unit: str, files: list[str], delete: list[str] = ()) -> None:
        """Commit one unit of work and the manifest together."""
        e = self.manifest["stages"][stage]
        for u in e["units"].values():
            for rel in delete:
                u.pop(rel, None)
        e["units"][unit] = self._record(files)
        e["updated_at"] = _now()
        self._commit(files, list(delete), f"{stage}: {unit}")

    def finish(self, stage: str, outputs: list[str]) -> None:
        """Mark a stage done, commit any outputs not saved yet, then squash history."""
        e = self.manifest["stages"][stage]
        e["outputs"] = self._record(outputs)
        e["status"] = "done"
        e["finished_at"] = _now()
        e["dataset_revision"] = self.dataset_revision
        saved = {rel: meta["sha256"] for u in e["units"].values() for rel, meta in u.items()}
        new = [rel for rel, meta in e["outputs"].items() if saved.get(rel) != meta["sha256"]]
        stale = self._stale.pop(stage, set()) - set(_entry_files(e))
        self._commit(new, sorted(stale), f"{stage}: done")
        self.store.squash(f"squash after {stage}")

    def _record(self, files: list[str]) -> dict:
        out = {}
        for rel in files:
            path = self.workdir / rel
            out[rel] = {"size": path.stat().st_size, "sha256": sha256_file(path)}
        return out

    def _commit(self, files: list[str], delete: list[str], message: str) -> None:
        (self.workdir / MANIFEST).write_text(json.dumps(self.manifest, indent=1, sort_keys=True))
        self.store.commit(self.workdir, [*files, MANIFEST], delete, message)


def _entry_files(e: dict) -> list[str]:
    files = set(e.get("outputs", {}))
    for u in e.get("units", {}).values():
        files |= set(u)
    return sorted(files)
