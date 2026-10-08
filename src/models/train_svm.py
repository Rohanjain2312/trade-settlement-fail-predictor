"""Model 2: support vector machines.

LinearSVC trains on the full training set. The RBF SVC trains on a stratified subsample,
because kernel SVM training cost grows roughly with the square of the row count. Neither
outputs a probability, so the decision function is calibrated into one on the validation
set (Platt scaling), keeping the fitted SVM frozen.
"""

from __future__ import annotations

import logging

from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.svm import SVC, LinearSVC

from src.models.preprocess import build_pipeline

log = logging.getLogger(__name__)


def make_linear(C: float, resampling: str, seed: int):
    est = LinearSVC(
        C=C,
        class_weight="balanced" if resampling == "class_weight" else None,
        dual="auto",
        max_iter=20000,
        random_state=seed,
    )
    return build_pipeline(est, resampling, seed)


def make_rbf(C: float, gamma: float | str, resampling: str, seed: int, use_cuml: bool = False):
    if use_cuml:
        from cuml.svm import SVC as CuSVC

        est = CuSVC(C=C, gamma=gamma, kernel="rbf", class_weight="balanced" if resampling == "class_weight" else None)
    else:
        est = SVC(
            C=C,
            gamma=gamma,
            kernel="rbf",
            class_weight="balanced" if resampling == "class_weight" else None,
            cache_size=2000,
            random_state=seed,
        )
    return build_pipeline(est, resampling, seed)


def calibrate(fitted_pipeline, X_val, y_val, method: str = "sigmoid"):
    """Fit only the score-to-probability map on validation data; the SVM stays frozen."""
    cal = CalibratedClassifierCV(FrozenEstimator(fitted_pipeline), method=method)
    return cal.fit(X_val, y_val)


def support_vector_count(fitted_pipeline) -> int | None:
    model = fitted_pipeline.named_steps["model"]
    n = getattr(model, "n_support_", None)
    return None if n is None else int(sum(n))
