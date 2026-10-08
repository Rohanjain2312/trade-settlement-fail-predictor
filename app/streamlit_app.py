"""Trade Settlement Fail Predictor: an interactive walkthrough of three models on synthetic data.

The Space ships with precomputed assets (assets/), produced by notebook 02, so it starts fast
on free CPU hardware. Set APP_ASSETS to point at another assets folder.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

ASSETS = Path(os.environ.get("APP_ASSETS", Path(__file__).parent / "assets"))
SYNTHETIC_NOTE = (
    "Synthetic data. Every number here comes from a seeded generator in this project, not from "
    "real trades. The numbers show the pipeline and the modeling, not real-world results."
)
FAMILY_LABEL = {"logreg": "Logistic regression", "svm_linear": "Linear SVM", "svm_rbf": "RBF SVM", "xgb": "XGBoost"}


@st.cache_data
def load_json(name: str) -> dict:
    return json.loads((ASSETS / name).read_text())


def synthetic_note() -> None:
    st.caption(f"ℹ️ {SYNTHETIC_NOTE}")


def metrics_table(metrics: dict) -> pd.DataFrame:
    rows = []
    for name, m in metrics["models"].items():
        t = m["test"]
        rows.append({
            "Model": FAMILY_LABEL.get(m["family"], m["family"]),
            "Resampling": m["resampling"],
            "PR-AUC": t["pr_auc"],
            "Recall in top 2%": t["recall_top_2pct"],
            "Recall at precision 0.5": t["recall_at_precision_50"],
            "Recall at precision 0.7": t["recall_at_precision_70"],
            "Brier": t["brier"],
            "Accuracy (not a headline metric)": t["accuracy"],
        })
    return pd.DataFrame(rows)


def tab_model_comparison(metrics: dict) -> None:
    st.subheader("Model comparison on the test period")
    synthetic_note()
    st.dataframe(metrics_table(metrics), hide_index=True)
    base = next(iter(metrics["models"].values()))["test"]
    st.markdown(
        f"Always predicting *settles* would score {base['accuracy_if_always_settles']:.1%} accuracy, "
        "which is why the table leads with PR-AUC and recall among the highest-scored trades."
    )


def main() -> None:
    st.set_page_config(page_title="Trade Settlement Fail Predictor", page_icon="📉", layout="wide")
    st.title("Trade Settlement Fail Predictor")
    st.info(SYNTHETIC_NOTE)
    metrics = load_json("metrics.json")
    (comparison,) = st.tabs(["Model comparison"])
    with comparison:
        tab_model_comparison(metrics)


main()
