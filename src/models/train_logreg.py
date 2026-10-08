"""Model 1: L2 logistic regression, the baseline."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from src.features.definitions import BY_NAME, CATEGORICAL
from src.models.preprocess import build_pipeline, expanded_feature_names


def make(C: float, resampling: str, seed: int):
    est = LogisticRegression(
        C=C,
        max_iter=3000,
        class_weight="balanced" if resampling == "class_weight" else None,
    )
    return build_pipeline(est, resampling, seed)


def fit(X, y, C: float, resampling: str, seed: int):
    return make(C, resampling, seed).fit(X, y)


def coefficient_table(pipeline) -> list[dict]:
    """Coefficients per standardized feature, with odds ratios and a plain-English reading."""
    model = pipeline.named_steps["model"]
    names = expanded_feature_names()
    rows = []
    for name, coef in zip(names, model.coef_[0]):
        odds = float(np.exp(coef))
        if "=" in name:
            feature, level = name.split("=", 1)
            ref = BY_NAME[feature].reference_level
            reading = (
                f"{BY_NAME[feature].label} = {level}: odds of failing x{odds:.2f} "
                f"(one-hot column; compare with the other levels of this feature, the reference is {ref})"
            )
        else:
            feature, level = name, None
            direction = "raises" if coef > 0 else "lowers"
            reading = (
                f"One standard deviation more {BY_NAME[feature].label.lower()} {direction} the odds of "
                f"failing by x{odds:.2f}, other features held fixed"
            )
        rows.append({
            "column": name,
            "feature": feature,
            "level": level,
            "coefficient": float(coef),
            "odds_ratio": odds,
            "reading": reading,
            "is_categorical": feature in CATEGORICAL,
        })
    return rows
