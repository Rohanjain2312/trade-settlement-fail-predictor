"""File stores with one interface: HubStore (Colab) and LocalDirStore (CI, no network).

A store is a flat namespace of files addressed by POSIX paths. `commit` writes a batch of
files and deletions together. On the Hub that is one atomic multi-file commit. Locally,
`manifest.json` is written after the new files and before deletions, so a crash mid-commit
leaves the old manifest valid and the checkpoint logic treats the unit as not done.
"""

from __future__ import annotations

import fnmatch
import logging
import os
import shutil
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path

log = logging.getLogger(__name__)

MANIFEST = "manifest.json"


class Store(ABC):
    @abstractmethod
    def list_files(self) -> list[str]: ...

    @abstractmethod
    def exists(self, path: str) -> bool: ...

    @abstractmethod
    def download(self, path: str, local_dir: Path) -> Path: ...

    @abstractmethod
    def download_all(self, local_dir: Path, allow_patterns: list[str] | None = None) -> None: ...

    @abstractmethod
    def commit(
        self, local_dir: Path, add: Iterable[str], delete: Iterable[str] = (), message: str = ""
    ) -> None: ...

    @abstractmethod
    def squash(self, message: str = "") -> None: ...


class LocalDirStore(Store):
    """A directory on disk standing in for a Hub repo. Used by CI and tests."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def list_files(self) -> list[str]:
        out = []
        for p in self.root.rglob("*"):
            if p.is_file() and ".tmp" not in p.relative_to(self.root).parts:
                out.append(p.relative_to(self.root).as_posix())
        return sorted(out)

    def exists(self, path: str) -> bool:
        return (self.root / path).is_file()

    def download(self, path: str, local_dir: Path) -> Path:
        dst = Path(local_dir) / path
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.root / path, dst)
        return dst

    def download_all(self, local_dir: Path, allow_patterns: list[str] | None = None) -> None:
        for path in self.list_files():
            if allow_patterns is None or any(fnmatch.fnmatch(path, pat) for pat in allow_patterns):
                self.download(path, local_dir)

    def commit(
        self, local_dir: Path, add: Iterable[str], delete: Iterable[str] = (), message: str = ""
    ) -> None:
        # Order: new files, then the manifest, then deletions. A crash before the manifest
        # leaves the previous manifest valid, and a crash after it only leaves stray files.
        add = list(dict.fromkeys(add))
        for path in add:
            if path != MANIFEST:
                self._put(local_dir, path)
        if MANIFEST in add:
            self._put(local_dir, MANIFEST)
        for path in delete:
            if path != MANIFEST and path not in add and (self.root / path).exists():
                (self.root / path).unlink()

    def _put(self, local_dir: Path, path: str) -> None:
        tmp_dir = self.root / ".tmp"
        tmp_dir.mkdir(exist_ok=True)
        dst = self.root / path
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = tmp_dir / uuid.uuid4().hex
        shutil.copy2(Path(local_dir) / path, tmp)
        os.replace(tmp, dst)

    def squash(self, message: str = "") -> None:
        return None


class HubStore(Store):
    """A Hugging Face Hub repo. Reads the token from the HF_TOKEN environment variable."""

    def __init__(self, repo_id: str, repo_type: str = "model", token: str | None = None):
        from huggingface_hub import HfApi

        self.repo_id = repo_id
        self.repo_type = repo_type
        self.api = HfApi(token=token)

    def _retry(self, fn, what: str, attempts: int = 6):
        import httpx
        from huggingface_hub.errors import HfHubHTTPError

        for attempt in range(1, attempts + 1):
            try:
                return fn()
            except HfHubHTTPError as err:
                status = err.response.status_code if err.response is not None else None
                if status not in (429, 500, 502, 503, 504) or attempt == attempts:
                    raise
                wait = _retry_after(err) if status == 429 else 10 * 2**attempt
            except (OSError, httpx.TransportError) as err:
                if attempt == attempts:
                    raise
                wait = 10 * 2**attempt
                log.warning("%s: %s", what, type(err).__name__)
            wait = min(wait, 900)
            log.warning("%s failed (attempt %d/%d), retrying in %ds", what, attempt, attempts, wait)
            time.sleep(wait)

    def list_files(self) -> list[str]:
        return self._retry(
            lambda: self.api.list_repo_files(self.repo_id, repo_type=self.repo_type),
            f"list {self.repo_id}",
        )

    def exists(self, path: str) -> bool:
        return self._retry(
            lambda: self.api.file_exists(self.repo_id, path, repo_type=self.repo_type),
            f"exists {path}",
        )

    def download(self, path: str, local_dir: Path) -> Path:
        from huggingface_hub import hf_hub_download

        local = self._retry(
            lambda: hf_hub_download(
                self.repo_id,
                path,
                repo_type=self.repo_type,
                local_dir=str(local_dir),
                token=self.api.token,
            ),
            f"download {path}",
        )
        return Path(local)

    def download_all(self, local_dir: Path, allow_patterns: list[str] | None = None) -> None:
        self._retry(
            lambda: self.api.snapshot_download(
                self.repo_id,
                repo_type=self.repo_type,
                local_dir=str(local_dir),
                allow_patterns=allow_patterns,
            ),
            f"snapshot {self.repo_id}",
        )

    def commit(
        self, local_dir: Path, add: Iterable[str], delete: Iterable[str] = (), message: str = ""
    ) -> None:
        from huggingface_hub import CommitOperationAdd, CommitOperationDelete

        add = list(dict.fromkeys(add))
        delete = [p for p in dict.fromkeys(delete) if p not in add]
        if delete:
            existing = set(self.list_files())
            delete = [p for p in delete if p in existing]
        ops = [CommitOperationAdd(path_in_repo=p, path_or_fileobj=str(Path(local_dir) / p)) for p in add]
        ops += [CommitOperationDelete(path_in_repo=p) for p in delete]
        if not ops:
            return
        self._retry(
            lambda: self.api.create_commit(
                self.repo_id,
                operations=ops,
                commit_message=message or "checkpoint",
                repo_type=self.repo_type,
            ),
            f"commit to {self.repo_id}",
        )

    def squash(self, message: str = "") -> None:
        self._retry(
            lambda: self.api.super_squash_history(
                self.repo_id, repo_type=self.repo_type, commit_message=message or "squash"
            ),
            f"squash {self.repo_id}",
        )


def _retry_after(err) -> int:
    """Seconds to wait after a 429, from the RateLimit or Retry-After header."""
    headers = err.response.headers if err.response is not None else {}
    value = headers.get("Retry-After")
    if value and value.isdigit():
        return int(value) + 5
    ratelimit = headers.get("RateLimit", "")
    for part in ratelimit.split(";"):
        part = part.strip()
        if part.startswith("t=") and part[2:].isdigit():
            return int(part[2:]) + 5
    return 120
