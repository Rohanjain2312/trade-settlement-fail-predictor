"""Notebook cell 1 helper: check the environment and create the Hub repos if missing.

The work repo itself is restored by the pipeline's first stage, so every restore shows up
in the run report.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

from src.config import load_config, load_run_config
from src.hub.repos import ensure_repos
from src.pipeline import env_check


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--notebook", required=True, choices=["01_generate_data", "02_train_explain"])
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", stream=sys.stdout)
    if not os.environ.get("HF_TOKEN"):
        raise SystemExit("HF_TOKEN is not set. Add it in Colab Secrets with notebook access on.")
    run = load_run_config()
    modes = ["smoke", "full"] if run["mode"] == "auto" else [run["mode"]]
    needs_gpu = args.notebook == "02_train_explain" and any(
        load_config(m)["train"]["require_gpu"] for m in modes
    )
    env_check.check(require_gpu=needs_gpu)
    for mode in modes:
        ensure_repos(load_config(mode))
    print(f"Setup done. Mode: {run['mode']}. Now run the next cell.")


if __name__ == "__main__":
    main()
