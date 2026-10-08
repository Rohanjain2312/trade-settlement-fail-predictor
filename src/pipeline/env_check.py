"""Environment check: versions, GPU, a tiny GPU XGBoost fit, and a tiny SHAP call.

Stops with a clear message if SHAP fails, or if the GPU fit fails when a GPU is required.
Without a required GPU it falls back to CPU and says so.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import logging
import platform
import shutil
import subprocess
import sys

import numpy as np

log = logging.getLogger(__name__)

PACKAGES = (
    "numpy", "pandas", "scipy", "scikit-learn", "imbalanced-learn", "xgboost", "shap",
    "optuna", "pyarrow", "huggingface_hub", "duckdb", "matplotlib", "pyyaml", "cuml-cu12",
)


class EnvCheckError(SystemExit):
    pass


def versions() -> dict[str, str]:
    out = {"python": platform.python_version()}
    for name in PACKAGES:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return out


def gpu_name() -> str | None:
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30, check=True,
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _tiny_data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 5)).astype("float32")
    y = (X[:, 0] + 0.5 * X[:, 1] + rng.normal(scale=0.5, size=400) > 0).astype(int)
    return X, y


def xgb_fit(device: str):
    import xgboost as xgb

    X, y = _tiny_data()
    booster = xgb.train(
        {"objective": "binary:logistic", "tree_method": "hist", "device": device, "max_depth": 3},
        xgb.DMatrix(X, label=y),
        num_boost_round=10,
    )
    contribs = booster.predict(xgb.DMatrix(X[:20]), pred_contribs=True)
    assert contribs.shape == (20, 6)
    return booster


def shap_call(booster) -> None:
    import shap

    X, _ = _tiny_data()
    booster.set_param({"device": "cpu"})
    values = shap.TreeExplainer(booster).shap_values(X[:20])
    assert np.asarray(values).shape[-1] == 5


def check(require_gpu: bool) -> dict:
    info = {"versions": versions(), "gpu": gpu_name(), "device": "cpu", "cuml": False}
    log.info("python %s on %s", info["versions"]["python"], platform.platform())
    log.info("package versions: %s", json.dumps(info["versions"]))
    log.info("GPU: %s", info["gpu"] or "none")

    booster = None
    if info["gpu"]:
        try:
            booster = xgb_fit("cuda")
            info["device"] = "cuda"
            log.info("tiny XGBoost fit on the GPU: ok")
        except Exception as err:  # noqa: BLE001 - reported below with a clear message
            log.warning("tiny XGBoost fit on the GPU failed: %s", err)
    if info["device"] != "cuda":
        if require_gpu:
            raise EnvCheckError(
                "ENV CHECK FAILED: this stage needs a GPU. In Colab choose Runtime > Change "
                "runtime type > G4 GPU, then press Run all again."
            )
        log.warning("no working GPU, falling back to CPU")
        booster = xgb_fit("cpu")

    try:
        shap_call(booster)
        log.info("tiny SHAP call: ok")
    except Exception as err:  # noqa: BLE001
        raise EnvCheckError(f"ENV CHECK FAILED: SHAP does not work here: {err!r}") from err

    try:
        import cuml  # noqa: F401

        info["cuml"] = True
        log.info("cuML imports: yes")
    except Exception:  # noqa: BLE001 - cuML is optional
        log.info("cuML imports: no (RBF SVM uses scikit-learn on a CPU subsample)")
    return info


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-gpu", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", stream=sys.stdout)
    info = check(args.require_gpu)
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
