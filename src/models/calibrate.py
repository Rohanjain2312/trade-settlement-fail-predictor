"""Turn each fitted model's score into a calibrated fail probability, on validation data.

Uses scikit-learn's API for calibrating an already fitted model: the model is wrapped in
FrozenEstimator, so CalibratedClassifierCV fits only the score-to-probability map.
SVMs output a decision score, not a probability, so they need this before their scores
can serve as risk scores. Class weights and SMOTE also inflate probabilities, which
calibration corrects. Platt (sigmoid) for LR and SVM, isotonic for XGBoost.
"""

from __future__ import annotations

from pathlib import Path

import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator


def xgb_classifier(model_path: Path) -> xgb.XGBClassifier:
    """A scikit-learn wrapper around a saved booster, so it can be calibrated like the rest."""
    clf = xgb.XGBClassifier(enable_categorical=True)
    clf.load_model(str(model_path))
    return clf


def calibrate(model, X_val, y_val, family: str) -> CalibratedClassifierCV:
    method = "isotonic" if family == "xgb" else "sigmoid"
    return CalibratedClassifierCV(FrozenEstimator(model), method=method).fit(X_val, y_val)


def raw_score(model, X):
    """The model's own score: a probability for LR and XGBoost, a margin for SVMs."""
    if hasattr(model, "predict_proba") and not _is_svm(model):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def _is_svm(model) -> bool:
    final = model.steps[-1][1] if hasattr(model, "steps") else model
    return type(final).__name__ in ("LinearSVC", "SVC")
