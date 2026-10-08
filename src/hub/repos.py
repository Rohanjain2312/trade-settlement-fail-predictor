"""Create the Hugging Face repos if they do not exist yet.

Full run: public dataset, model, and Space; private work repo.
Smoke run: every repo has a -smoke suffix and is private.
Run reports always go to the public model repo, so they can be read without a token.
"""

from __future__ import annotations

import logging

from src.config import repo_ids

log = logging.getLogger(__name__)


def ensure_repos(cfg: dict, api=None) -> dict[str, str]:
    from huggingface_hub import HfApi

    api = api or HfApi()
    ids = repo_ids(cfg)
    private = cfg["mode"] == "smoke"
    api.create_repo(ids["dataset"], repo_type="dataset", private=private, exist_ok=True)
    api.create_repo(ids["model"], repo_type="model", private=private, exist_ok=True)
    api.create_repo(ids["work"], repo_type="model", private=True, exist_ok=True)
    api.create_repo(ids["space"], repo_type="space", space_sdk="docker", private=private, exist_ok=True)
    api.create_repo(ids["reports"], repo_type="model", private=False, exist_ok=True)
    log.info("hub repos ready: %s", ", ".join(sorted(set(ids.values()))))
    return ids
