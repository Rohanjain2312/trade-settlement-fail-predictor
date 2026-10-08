"""Publishing to the dataset and model repos, resumable batch by batch.

Files are staged under the target repo's layout, then committed in batches. Each finished
batch is recorded in the work manifest, so a restart skips batches already on the Hub.
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

from src.pipeline.store import Store

log = logging.getLogger(__name__)


def stage_files(workdir: Path, mapping: dict[str, str], staging: str) -> Path:
    """Place local files (relative to workdir) at their repo paths under workdir/staging."""
    root = Path(workdir) / staging
    for repo_path, local in mapping.items():
        dst = root / repo_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            dst.unlink()
        try:
            os.link(Path(workdir) / local, dst)
        except OSError:
            shutil.copy2(Path(workdir) / local, dst)
    return root


def publish(ctx, stage: str, store: Store, mapping: dict[str, str], staging: str,
            batch_size: int = 8, message: str = "publish") -> None:
    """Commit mapping {repo_path: local_path} to store in batches, skipping finished batches."""
    root = stage_files(ctx.workdir, mapping, staging)
    paths = sorted(mapping)
    for i in range(0, len(paths), batch_size):
        batch = paths[i : i + batch_size]
        unit = f"batch_{i // batch_size:03d}"
        local = [str(Path(staging) / p) for p in batch]
        if ctx.ckpt.unit_done(stage, unit):
            continue
        store.commit(root, batch, message=f"{message} ({unit})")
        ctx.ckpt.save_unit(stage, unit, local, upload=False)
        log.info("published %s: %d files", unit, len(batch))
