"""Trade Settlement Fail Predictor: an interactive walkthrough of three models on synthetic data.

The Space ships with precomputed assets (assets/), produced by notebook 02, so it starts fast
on free CPU hardware. Set APP_ASSETS to point at another assets folder.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).parent))
from simulate_queue import compare, day_ranks  # noqa: E402

ASSETS = Path(os.environ.get("APP_ASSETS", Path(__file__).parent / "assets"))
SYNTHETIC_NOTE = (
    "Synthetic data. Every number here comes from a seeded generator in this project, not from "
    "real trades. The numbers show the pipeline and the modeling, not real-world results."
)
FAMILY = {"logreg": "Logistic regression", "svm_linear": "Linear SVM", "svm_rbf": "RBF SVM", "xgb": "XGBoost"}
RESAMPLING = {"none": "no resampling", "class_weight": "class weights", "scale_pos_weight": "class weights",
              "smote": "SMOTENC"}
REASON = {"ssi_problem": "SSI problem", "shortfall": "Shortfall", "unmatched": "Unmatched", "other": "Other"}
FAIL_COLOR, SETTLE_COLOR, SYNTH_COLOR = "#d62728", "#7f7f7f", "#ff9f1c"
STRENGTH_COLOR = {"strong": "#1f4e79", "medium": "#5b9bd5", "weak": "#bdd7ee"}


# ---- loading -----------------------------------------------------------------------------------

@st.cache_data
def load_json(name: str):
    return json.loads((ASSETS / name).read_text())


@st.cache_data
def load_parquet(name: str) -> pd.DataFrame:
    return pd.read_parquet(ASSETS / name)


def label(name: str) -> str:
    family, resampling = name.split("__")
    return f"{FAMILY[family]} ({RESAMPLING[resampling]})"


def note() -> None:
    st.caption(f"ℹ️ {SYNTHETIC_NOTE}")


def features() -> dict[str, dict]:
    return {f["name"]: f for f in load_json("features.json")}


def pct(x: float) -> str:
    return f"{x:.1%}"


# ---- tab 1: data and features ------------------------------------------------------------------

def tab_data() -> None:
    s = load_json("data_summary.json")
    feats = load_json("features.json")
    st.subheader("The problem")
    st.markdown(
        "A trade **fails** when the securities or the cash do not move on the intended settlement date. "
        "Fails tie up capital, trigger penalties, and land in an operations queue. Ops can only work a "
        "limited number of trades each day, so the goal is a **risk score for every open trade**: work the "
        "riskiest ones first and fix them before the settlement date."
    )
    note()
    c = st.columns(4)
    c[0].metric("Trades", f"{s['n_trades']:,}")
    c[1].metric("Fail rate", pct(s["fail_rate"]))
    c[2].metric("Period", f"{s['first_date'][:7]} to {s['last_date'][:7]}")
    c[3].metric("Counterparties / securities", f"{s['n_counterparties']} / {s['n_securities']:,}")

    st.markdown("#### Splits by time")
    splits = pd.DataFrame([{"Split": k, "Trades": v["rows"], "Fail rate": pct(v["fail_rate"]),
                            "From": v["first"], "To": v["last"]} for k, v in s["splits"].items()])
    st.dataframe(splits, hide_index=True)
    st.caption(
        f"Splits follow trade date, never a random shuffle, so the model is always tested on a later period. "
        f"The last {s['gap_days']} business days of train and validation are dropped, so no trade that is "
        "still settling straddles a boundary. The test period contains a volatility spike, which is why "
        "its fail rate is higher."
    )

    st.markdown("#### How the data was produced")
    st.markdown(
        "- A seeded **numpy generator** produces trades day by day for fictional counterparties and securities.\n"
        "- Each trade gets the **20 features** below. Rolling features (counterparty and security fail rates, "
        "pair history, size versus median) use only outcomes known on the trade date.\n"
        "- The fail probability is a **planted logistic function**: each feature has a known effect, plus three "
        "interactions, hidden counterparty and security risk, and noise. Because the truth is known, we can "
        "check later whether the explanations recover it.\n"
        "- **Scenario injectors** make sure rare situations (SSI breaks, shortfalls, chain cascades, holidays, "
        "corporate actions, amendments) appear often enough in every split, and a coverage report checks it."
    )

    st.markdown("#### The 20 features")
    table = pd.DataFrame([{"#": f["index"], "Feature": f["label"], "Group": f["group"], "Type": f["kind"],
                           "What it is": f["description"], "How it affects settlement": f["mechanism"],
                           "Planted strength": f["strength"]} for f in feats])
    st.dataframe(table, hide_index=True, height=420)

    st.markdown("#### Explore one feature")
    names = {f["label"]: f["name"] for f in feats}
    pick = names[st.selectbox("Feature", list(names), key="feature_explore")]
    f = features()[pick]
    bins = pd.DataFrame(load_json("feature_bins.json")[pick]).dropna(subset=["fail_rate"])
    fig = go.Figure()
    fig.add_bar(x=bins["bin"], y=bins["fail_rate"], marker_color="#5b9bd5",
                customdata=bins["trades"], hovertemplate="%{x}<br>fail rate %{y:.2%}<br>%{customdata:,} trades")
    fig.add_hline(y=s["fail_rate"], line_dash="dot", annotation_text="overall fail rate")
    fig.update_layout(height=320, yaxis_tickformat=".1%", xaxis_title=f["label"], yaxis_title="Fail rate",
                      margin=dict(t=20, b=40))
    st.plotly_chart(fig, key="feature_bins")
    st.caption(f"**Why it matters:** {f['mechanism']}  Planted strength: **{f['strength']}**.")

    st.markdown("#### Scenario coverage")
    cov = load_json("coverage.json")
    (st.success if cov["passed"] else st.error)(
        "Coverage report: " + ("every scenario and feature bin meets its minimum in every split."
                               if cov["passed"] else f"{len(cov['failures'])} cells below their minimum."))
    rows = [r for r in cov["rows"] if r["kind"] == "scenario"]
    st.dataframe(pd.DataFrame([{
        "ID": r["id"], "Scenario": r["name"],
        "Train fails": r["train_failed"], "Train settled": r["train_settled"],
        "Val fails": r["val_failed"], "Test fails": r["test_failed"], "Test settled": r["test_settled"],
    } for r in rows]), hide_index=True, height=380)
    st.caption(
        "S = root causes, R = market and calendar regimes, H = hard negatives (look risky, mostly settle) and "
        "unexplained fails. Each trade carries a bitmask of its scenarios, used only for these reports and "
        "per-scenario evaluation, never as a model input."
    )
    real = cov["realism"]
    fig = go.Figure()
    reasons = list(real["reason_mix"])
    fig.add_bar(x=[REASON[r] for r in reasons], y=[real["reason_mix"][r] for r in reasons], name="Generated",
                marker_color="#1f4e79")
    lo = [real["targets"]["reason_share"][r][0] for r in reasons]
    hi = [real["targets"]["reason_share"][r][1] for r in reasons]
    fig.add_scatter(x=[REASON[r] for r in reasons], y=[(a + b) / 2 for a, b in zip(lo, hi)], mode="markers",
                    name="Target range (public sources)", marker=dict(color="black", symbol="line-ew-open", size=18),
                    error_y=dict(type="data", array=[(b - a) / 2 for a, b in zip(lo, hi)]))
    fig.update_layout(height=300, yaxis_tickformat=".0%", title="Root-cause mix of fails versus rough public ranges",
                      margin=dict(t=40, b=30))
    st.plotly_chart(fig, key="realism")
    t = cov["time"]
    st.caption(
        f"Fail rate {pct(real['fail_rate'])} (public figures put euro-area fails near 2 to 3%). "
        f"{pct(t['cold_start_test_share'])} of test trades come from counterparties never seen in training. "
        "Out of scope: partial settlements, buy-ins, regulation-specific penalty mechanics, market-wide outages. "
        "Real production data always contains cases nobody planned for."
    )


# ---- tab 2: model comparison -------------------------------------------------------------------

def tab_comparison() -> None:
    metrics = load_json("metrics.json")["models"]
    primary = load_json("models.json")["primary"]
    st.subheader("Model comparison on the test period")
    note()
    base = next(iter(metrics.values()))["test"]
    st.markdown(
        f"Only **{pct(base['base_rate'])}** of test trades fail, so always predicting *settles* scores "
        f"**{pct(base['accuracy_if_always_settles'])} accuracy** while catching nothing. The table therefore leads "
        "with ranking metrics: **PR-AUC** (precision against recall over all thresholds), **recall in the top 1% "
        "and 2%** of trades by score (how many fails ops would see if they worked only that share), recall at "
        "fixed precision, and the **Brier score** (how good the probabilities are; lower is better)."
    )
    show_all = st.toggle("Show every resampling variant", value=True, key="all_variants")
    names = list(metrics) if show_all else list(primary.values())
    df = pd.DataFrame([{
        "Model": FAMILY[metrics[n]["family"]], "Resampling": RESAMPLING[metrics[n]["resampling"]],
        "PR-AUC": metrics[n]["test"]["pr_auc"], "Recall top 1%": metrics[n]["test"]["recall_top_1pct"],
        "Recall top 2%": metrics[n]["test"]["recall_top_2pct"],
        "Recall @ precision 0.5": metrics[n]["test"]["recall_at_precision_50"],
        "Recall @ precision 0.7": metrics[n]["test"]["recall_at_precision_70"],
        "Brier": metrics[n]["test"]["brier"], "Accuracy": metrics[n]["test"]["accuracy"],
    } for n in names])
    st.dataframe(df.style.format({c: "{:.3f}" for c in df.columns[2:]}).highlight_max(
        subset=["PR-AUC", "Recall top 2%"], color="#d9ead3"), hide_index=True)

    st.markdown("#### Precision-recall curves")
    picks = st.multiselect("Models", list(metrics), default=list(primary.values()), format_func=label, key="pr_models")
    fig = go.Figure()
    for n in picks:
        c = metrics[n]["test"]["pr_curve"]
        fig.add_scatter(x=c["recall"], y=c["precision"], mode="lines", name=label(n))
    fig.add_hline(y=base["base_rate"], line_dash="dot", annotation_text="random ranking")
    fig.update_layout(height=380, xaxis_title="Recall (share of fails caught)", yaxis_title="Precision",
                      margin=dict(t=20))
    st.plotly_chart(fig, key="pr_curves")

    st.markdown("#### Does SMOTE help? No resampling vs class weights vs SMOTENC")
    metric = st.radio("Metric", ["PR-AUC", "Recall top 2%", "Brier"], horizontal=True, key="resample_metric")
    key = {"PR-AUC": "pr_auc", "Recall top 2%": "recall_top_2pct", "Brier": "brier"}[metric]
    fig = go.Figure()
    for res in ("none", "class_weight", "smote"):
        xs, ys = [], []
        for fam in FAMILY:
            n = next((m for m in metrics if m.startswith(fam + "__") and
                      metrics[m]["resampling"] in ((res, "scale_pos_weight") if res == "class_weight" else (res,))), None)
            if n:
                xs.append(FAMILY[fam])
                ys.append(metrics[n]["test"][key])
        fig.add_bar(x=xs, y=ys, name=RESAMPLING[res])
    fig.update_layout(barmode="group", height=340, yaxis_title=metric, margin=dict(t=20))
    st.plotly_chart(fig, key="resampling")
    lines = []
    for fam in FAMILY:
        fam_models = {metrics[m]["resampling"]: metrics[m]["test"]["pr_auc"] for m in metrics if m.startswith(fam + "__")}
        if "smote" in fam_models:
            best_other = max(v for k, v in fam_models.items() if k != "smote")
            verb = "helped" if fam_models["smote"] > best_other else "did not help"
            lines.append(f"- {FAMILY[fam]}: SMOTENC {verb} (PR-AUC {fam_models['smote']:.3f} vs best other "
                         f"{best_other:.3f}).")
    st.markdown("\n".join(lines))
    st.caption(
        "SMOTE changes where a model draws its default decision line, which matters when a model is used with "
        "its own threshold (see the SVM tab). Ranking quality is what PR-AUC measures, and class weights reach "
        "the same goal without inventing data. SMOTE only ever runs on training rows, inside the pipeline."
    )

    st.markdown("#### Calibration: can the score be read as a probability?")
    cal_pick = st.selectbox("Model", list(metrics), index=list(metrics).index(primary["xgb"]), format_func=label,
                            key="cal_model")
    m = metrics[cal_pick]
    fig = go.Figure()
    fig.add_scatter(x=[0, 0.5], y=[0, 0.5], mode="lines", line=dict(dash="dot", color="gray"), name="perfect")
    cc = m["test"]["calibration_curve"]
    fig.add_scatter(x=cc["mean_predicted"], y=cc["fraction_failed"], mode="lines+markers", name="calibrated")
    if "calibration_curve" in m["raw_test"]:
        rc = m["raw_test"]["calibration_curve"]
        fig.add_scatter(x=rc["mean_predicted"], y=rc["fraction_failed"], mode="lines+markers", name="raw model output")
    fig.update_layout(height=360, xaxis_title="Predicted fail probability (bin average)",
                      yaxis_title="Observed fail rate", margin=dict(t=20))
    st.plotly_chart(fig, key="calibration")
    raw_brier = m["raw_test"].get("brier")
    st.caption(
        f"Brier score after calibration {m['test']['brier']:.4f}"
        + (f", raw {raw_brier:.4f}." if raw_brier is not None else ". SVMs output a margin, not a probability, so "
           "they have no raw curve: Platt scaling on the validation set turns the margin into a probability.")
        + " Calibration is fitted on validation data only. The test period is more stressed than validation, "
        "so some miscalibration remains: a real-world reason to recalibrate after drift."
    )

    st.markdown("#### Where each model is strong or weak: per scenario")
    which = st.radio("Metric per scenario", ["Recall at ops capacity", "PR-AUC"], horizontal=True, key="scen_metric")
    mk = "recall_at_threshold" if which.startswith("Recall") else "pr_auc"
    scen = list(metrics[primary["xgb"]]["per_scenario"])
    z = [[metrics[n]["per_scenario"][s].get(mk) for n in primary.values()] for s in scen]
    fig = go.Figure(go.Heatmap(z=z, x=[label(n) for n in primary.values()],
                               y=[f"{s}: {metrics[primary['xgb']]['per_scenario'][s]['name'][:40]}" for s in scen],
                               colorscale="Blues", zmin=0, zmax=1, text=[[f"{v:.2f}" if v is not None else "" for v in r] for r in z],
                               texttemplate="%{text}", hovertemplate="%{y}<br>%{x}<br>%{z:.3f}<extra></extra>"))
    fig.update_layout(height=640, margin=dict(t=20, l=10))
    st.plotly_chart(fig, key="per_scenario")
    st.caption(f"Recall at ops capacity: the share of each scenario's fails among the {pct(load_json('metrics.json')['ops_capacity_share'])} "
               "highest-scored trades (threshold picked on validation). Hard negatives (H1 to H3) mostly settle, "
               "so a low flag rate there is good. H4 fails have no visible cause by design.")

    st.markdown("#### Held-out scenario: what if a cause never appeared in training?")
    held = load_json("heldout.json")
    fig = go.Figure()
    fig.add_bar(x=[f"{h['scenario']}: {h['name'][:30]}" for h in held],
                y=[h["full_model"]["recall_at_capacity"] for h in held], name="trained on everything")
    fig.add_bar(x=[f"{h['scenario']}: {h['name'][:30]}" for h in held],
                y=[h["without_scenario"]["recall_at_capacity"] for h in held], name="scenario removed from training")
    fig.update_layout(barmode="group", height=320, yaxis_tickformat=".0%", yaxis_title="Recall at ops capacity",
                      margin=dict(t=20))
    st.plotly_chart(fig, key="heldout")
    st.caption("XGBoost retrained without every training trade in the scenario, then tested on that scenario. "
               "The drop shows the limits of generalization: a model can only partly recognize a cause it never saw.")


# ---- tab 3: how the SVM works -----------------------------------------------------------------

def tab_svm() -> None:
    v = load_json("svm_views.json")
    b = v["boundary"]
    st.subheader("How the SVM works")
    note()
    st.markdown(
        "A support vector machine draws the boundary with the **widest margin** between the classes. Only the "
        "trades on or inside the margin, the **support vectors**, decide where it goes; moving any other trade "
        "changes nothing. **C** sets how much the SVM pays for trades on the wrong side (small C: wide, tolerant "
        "margin; large C: tight fit). An **RBF kernel** lets the boundary curve; **gamma** sets how far each "
        "trade's influence reaches (large gamma: wiggly boundary around single trades)."
    )
    st.info(b["note"])
    c1, c2, c3 = st.columns(3)
    kernel = c1.radio("Kernel", ["linear", "rbf"], horizontal=True, key="svm_kernel")
    C = c2.select_slider("C", options=sorted({s["C"] for s in b["settings"]}), value=1.0, key="svm_C")
    gammas = sorted({s["gamma"] for s in b["settings"] if s["gamma"] is not None})
    gamma = c3.select_slider("gamma (RBF only)", options=gammas, value=1.0, key="svm_gamma", disabled=kernel == "linear")
    s = next(x for x in b["settings"] if x["kernel"] == kernel and x["C"] == C and
             (kernel == "linear" or x["gamma"] == gamma))
    pts = b["points"]
    y = np.array(pts["failed"])
    fig = go.Figure()
    fig.add_contour(x=b["grid"]["x"], y=b["grid"]["y"], z=s["decision"], colorscale="RdBu_r", opacity=0.35,
                    contours=dict(start=-3, end=3, size=0.5), showscale=False, hoverinfo="skip")
    fig.add_contour(x=b["grid"]["x"], y=b["grid"]["y"], z=s["decision"], showscale=False, hoverinfo="skip",
                    contours=dict(coloring="lines", start=-1, end=1, size=1), line=dict(width=2, color="black"))
    for cls, name, color in ((0, "settled", SETTLE_COLOR), (1, "failed", FAIL_COLOR)):
        m = y == cls
        fig.add_scatter(x=np.array(pts["x"])[m], y=np.array(pts["y"])[m], mode="markers", name=name,
                        marker=dict(color=color, size=5, opacity=0.6))
    sv = np.array(s["support"])
    fig.add_scatter(x=np.array(pts["x"])[sv], y=np.array(pts["y"])[sv], mode="markers", name="support vectors",
                    marker=dict(size=10, color="rgba(0,0,0,0)", line=dict(width=1.5, color="black")))
    fig.update_layout(height=480, xaxis_title=f"{b['labels'][0]} (standardized)",
                      yaxis_title=f"{b['labels'][1]} (standardized)", margin=dict(t=20))
    st.plotly_chart(fig, key="svm_boundary")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Support vectors", f"{s['n_support']} of {len(y)}")
    m2.metric("Train accuracy", pct(s["train_accuracy"]))
    m3.metric("Holdout accuracy", pct(s["holdout_accuracy"]))
    m4.metric("Holdout PR-AUC", f"{s['holdout_pr_auc']:.3f}")
    st.caption("Solid lines: the boundary (0) and the margin edges (-1 and +1). Circled trades are support vectors. "
               "This illustration uses a balanced sample, so accuracy here is meaningful; on real class balance it is not.")

    st.markdown("#### C and gamma: under- and overfitting")
    rbf = [x for x in b["settings"] if x["kernel"] == "rbf"]
    Cs = sorted({x["C"] for x in rbf})
    fig = make_subplots(rows=len(gammas), cols=len(Cs), shared_xaxes=True, shared_yaxes=True,
                        subplot_titles=[f"C={c:g}, gamma={g:g}" for g in gammas for c in Cs],
                        horizontal_spacing=0.02, vertical_spacing=0.06)
    for i, g in enumerate(gammas):
        for j, c in enumerate(Cs):
            x = next(t for t in rbf if t["C"] == c and t["gamma"] == g)
            fig.add_trace(go.Contour(x=b["grid"]["x"], y=b["grid"]["y"], z=x["decision"], colorscale="RdBu_r",
                                     showscale=False, contours=dict(start=-2, end=2, size=1), opacity=0.6,
                                     hoverinfo="skip"), row=i + 1, col=j + 1)
    fig.update_layout(height=560, margin=dict(t=40), font=dict(size=10))
    st.plotly_chart(fig, key="svm_grid")
    fig = go.Figure()
    for g in gammas:
        xs = [t for t in rbf if t["gamma"] == g]
        fig.add_scatter(x=[t["C"] for t in xs], y=[t["holdout_pr_auc"] for t in xs], mode="lines+markers",
                        name=f"holdout, gamma={g:g}")
        fig.add_scatter(x=[t["C"] for t in xs], y=[t["train_accuracy"] for t in xs], mode="lines", name=f"train acc, gamma={g:g}",
                        line=dict(dash="dot"))
    fig.update_layout(height=320, xaxis_type="log", xaxis_title="C", margin=dict(t=20))
    st.plotly_chart(fig, key="svm_bias_variance")
    st.caption("Small C and gamma: a smooth boundary that misses structure (underfitting). Large C and gamma: "
               "training accuracy climbs while holdout quality falls, because the boundary wraps single trades "
               "(overfitting). Tuning picks the middle with time-series cross-validation.")

    st.markdown("#### SMOTE: making synthetic fails")
    sm = v["smote"]
    n = st.slider("Synthetic fails shown", 0, len(sm["synthetic"]["x"]), 60, key="smote_n")
    fig = go.Figure()
    fig.add_scatter(x=sm["settled"]["x"], y=sm["settled"]["y"], mode="markers", name="settled",
                    marker=dict(color=SETTLE_COLOR, size=3, opacity=0.25))
    fig.add_scatter(x=sm["failed"]["x"], y=sm["failed"]["y"], mode="markers", name="real fails",
                    marker=dict(color=FAIL_COLOR, size=6))
    fx, fy = np.array(sm["failed"]["x"]), np.array(sm["failed"]["y"])
    syn = sm["synthetic"]
    lx, ly = [], []
    for i in range(n):
        p, q = syn["parent"][i], syn["partner"][i]
        lx += [fx[p], fx[q], None]
        ly += [fy[p], fy[q], None]
    fig.add_scatter(x=lx, y=ly, mode="lines", name="fail and its neighbor", line=dict(color=SYNTH_COLOR, width=1),
                    opacity=0.5)
    fig.add_scatter(x=syn["x"][:n], y=syn["y"][:n], mode="markers", name="synthetic fails",
                    marker=dict(color=SYNTH_COLOR, size=7, symbol="diamond"))
    fig.update_layout(height=440, xaxis_title=f"{sm['labels'][0]} (standardized)",
                      yaxis_title=f"{sm['labels'][1]} (standardized)", margin=dict(t=20))
    st.plotly_chart(fig, key="smote")
    st.markdown(
        f"Each synthetic fail sits on the line between a real fail and one of its {sm['k']} nearest fails, at a "
        "random point. **SMOTENC** does the same for numeric features and takes the most common value among the "
        "neighbors for categorical ones (SSI status, counterparty type, and so on)."
    )
    lk = v["leakage"]
    st.markdown("#### Why SMOTE runs only on training data")
    c1, c2 = st.columns(2)
    c1.metric("SMOTE on training rows only", f"{lk['median_distance_right']:.2f}",
              help="Median distance from a test fail to its nearest synthetic fail")
    c2.metric("SMOTE before the split (wrong)", f"{lk['median_distance_wrong']:.2f}",
              delta=f"{(lk['ratio'] - 1):.0%}" if lk["ratio"] else None, delta_color="inverse")
    st.markdown(
        "If SMOTE runs **before** the split, synthetic fails are built from test fails, so training data contains "
        "near copies of the test set and the scores look better than they are. Here the median distance from a "
        "test fail to the nearest synthetic point shrinks sharply when it is done the wrong way. In this project "
        "SMOTENC is a step inside an `imblearn` pipeline, so it runs only when the pipeline is fitted, on training "
        "rows or training folds:"
    )
    st.code("Pipeline([\n  ('encode', ordinal-encode categoricals + median-impute numerics),\n"
            "  ('smote',  SMOTENC(categorical_features=[0, 1, 2, 3])),  # fit only\n"
            "  ('expand', one-hot encode + scale),\n  ('model',  LinearSVC(...)),\n])", language="text")

    st.markdown("#### Why SVM scores need calibrating, and what SMOTE did for the SVM")
    sample = load_parquet("app_sample.parquet")
    sample = sample[~sample["pool"]]
    fig = go.Figure()
    for cls, name, color in ((0, "settled", SETTLE_COLOR), (1, "failed", FAIL_COLOR)):
        fig.add_histogram(x=sample.loc[sample["failed"] == cls, "score__svm_linear_raw"], name=name, opacity=0.65,
                          marker_color=color, histnorm="probability density", nbinsx=60)
    platt = v.get("platt")
    if platt:
        xs = np.linspace(sample["score__svm_linear_raw"].quantile(0.001), sample["score__svm_linear_raw"].quantile(0.999), 100)
        fig.add_scatter(x=xs, y=1 / (1 + np.exp(platt["a"] * xs + platt["b"])), name="Platt: probability", yaxis="y2",
                        line=dict(color="black"))
    fig.update_layout(barmode="overlay", height=360, xaxis_title=f"Raw SVM score ({label(v['svm_model'])})",
                      yaxis2=dict(overlaying="y", side="right", range=[0, 1], title="calibrated probability"),
                      margin=dict(t=20))
    st.plotly_chart(fig, key="svm_scores")
    st.dataframe(pd.DataFrame([{"Raw SVM": label(d["model"]), "Flags (score > 0)": pct(d["flag_rate"]),
                                "Recall": pct(d["recall"]), "Precision": pct(d["precision"])}
                               for d in v["default_threshold"]]), hide_index=True)
    st.caption(
        "An SVM outputs a distance from the boundary, not a probability, so Platt scaling (a logistic curve fitted on "
        "validation data) maps it to one. The table shows each raw SVM with its own decision rule: trained on "
        "the natural 3% fail rate, the boundary flags almost nothing. SMOTE (or class weights) moves the boundary "
        "so the default rule flags enough trades. Once scores are ranked or calibrated, that advantage goes away."
    )


# ---- tab 4: XGBoost and SHAP -------------------------------------------------------------------

def _tree_figure(nodes: list[dict]) -> go.Figure:
    children: dict = {}
    for n in nodes:
        children.setdefault(n["parent"], []).append(n)
    pos, counter = {}, [0]

    def place(n):
        kids = children.get(n["id"], [])
        if not kids:
            pos[n["id"]] = counter[0]
            counter[0] += 1
        else:
            for k in kids:
                place(k)
            pos[n["id"]] = float(np.mean([pos[k["id"]] for k in kids]))

    place(children[None][0])
    fig = go.Figure()
    ex, ey = [], []
    for n in nodes:
        if n["parent"] is not None:
            p = next(m for m in nodes if m["id"] == n["parent"])
            ex += [pos[p["id"]], pos[n["id"]], None]
            ey += [-p["depth"], -n["depth"], None]
    fig.add_scatter(x=ex, y=ey, mode="lines", line=dict(color="#aaa"), hoverinfo="skip", showlegend=False)
    fig.add_scatter(x=[pos[n["id"]] for n in nodes], y=[-n["depth"] for n in nodes], mode="markers+text",
                    text=[n["label"].replace(" in ", "<br>in ").replace(" < ", "<br>< ") for n in nodes],
                    textposition="bottom center", textfont=dict(size=9),
                    marker=dict(size=12, color=["#2ca02c" if n["leaf"] else "#1f4e79" for n in nodes]),
                    hovertext=[f"{n['label']}<br>cover {n['cover']:.0f}" for n in nodes], hoverinfo="text",
                    showlegend=False)
    fig.update_layout(height=430, xaxis=dict(visible=False), yaxis=dict(visible=False), margin=dict(t=10, b=10))
    return fig


def tab_xgb() -> None:
    xv = load_json("xgb_views.json")
    feats = features()
    st.subheader("XGBoost and SHAP")
    note()
    st.markdown(
        "**Boosting** builds many small trees one after another. Each new tree is fitted to what the trees so far "
        "still get wrong (the gradient of the loss, the residuals), and its output is added with a small "
        "learning rate. Regularization (tree depth, minimum leaf weight, L2 penalty, row and column sampling) "
        "keeps each tree weak; **early stopping** ends training when validation PR-AUC stops improving."
    )
    c1, c2 = st.columns(2)
    with c1:
        h = xv["val_aucpr"]
        fig = go.Figure(go.Scatter(x=list(range(1, len(h) + 1)), y=h, mode="lines", name="validation PR-AUC"))
        fig.add_vline(x=xv["best_iteration"] + 1, line_dash="dot", annotation_text=f"best: {xv['best_iteration'] + 1} trees")
        fig.update_layout(height=320, xaxis_title="Trees", yaxis_title="Validation PR-AUC", margin=dict(t=30),
                          title="Early stopping")
        st.plotly_chart(fig, key="xgb_curve")
    with c2:
        fig = go.Figure()
        for name, margins in xv["staged_margin"].items():
            fig.add_scatter(x=xv["staged_steps"], y=1 / (1 + np.exp(-np.array(margins))), mode="lines+markers", name=name)
        fig.update_layout(height=320, xaxis_type="log", xaxis_title="Trees added", yaxis_title="Predicted probability (raw)",
                          margin=dict(t=30), title="Three trades as trees are added")
        st.plotly_chart(fig, key="xgb_staged")
    st.markdown("#### The first tree (top levels)")
    st.plotly_chart(_tree_figure(xv["tree0"]), key="xgb_tree")
    st.caption(f"Tree 1 of {xv['n_trees']}. Categorical features split on groups of levels natively, and missing "
               "values follow a learned default branch, so no imputation or one-hot encoding is needed.")

    st.markdown("#### SHAP: which features drive fails, and in which direction")
    st.markdown(
        "A **SHAP value** splits one trade's prediction (in log-odds) into a contribution per feature, so that "
        "base value plus contributions equals the model output. It is the feature's average marginal contribution "
        "over all orders of adding features (the Shapley value from game theory). XGBoost computes them exactly "
        "for trees, here on the GPU for every test trade."
    )
    gi = load_json("global_importance.json")["importance"]
    order = sorted(gi, key=lambda n: gi[n])
    fig = go.Figure(go.Bar(x=[gi[n] for n in order], y=[feats[n]["label"] for n in order], orientation="h",
                           marker_color=[STRENGTH_COLOR[feats[n]["strength"]] for n in order]))
    fig.update_layout(height=520, xaxis_title="Mean |SHAP| (log-odds), all test trades", margin=dict(t=20))
    c1, c2 = st.columns([1, 1.3])
    c1.plotly_chart(fig, key="shap_bar")
    sample = load_parquet("app_sample.parquet")
    rand = sample[~sample["pool"]]
    top = [n for n in sorted(gi, key=lambda n: -gi[n])][:12]
    fig = go.Figure()
    rng = np.random.default_rng(0)
    for i, n in enumerate(reversed(top)):
        vals = rand[n]
        color = (vals.cat.codes.astype(float) if hasattr(vals, "cat") else vals.astype(float))
        color = color.rank(pct=True)
        fig.add_scatter(x=rand[f"shap__{n}"], y=i + rng.uniform(-0.3, 0.3, len(rand)), mode="markers",
                        marker=dict(size=3, color=color, colorscale="Bluered", cmin=0, cmax=1,
                                    showscale=i == 0, colorbar=dict(title="feature value<br>(low to high)")),
                        name=feats[n]["label"], showlegend=False,
                        hovertemplate=f"{feats[n]['label']}<br>SHAP %{{x:.3f}}<extra></extra>")
    fig.update_layout(height=520, yaxis=dict(tickvals=list(range(len(top))), ticktext=[feats[n]["label"] for n in reversed(top)]),
                      xaxis_title="SHAP value (log-odds); right = pushes toward failing", margin=dict(t=20))
    c2.plotly_chart(fig, key="beeswarm")
    st.caption("Bar colors show the planted strength (dark: strong, light: weak). In the beeswarm each dot is a "
               "test trade; red means a high value of that feature.")

    st.markdown("#### SHAP interactions: what a linear model misses")
    inter = load_json("interaction_summary.json")
    planted = {frozenset((p["a"], p["b"])): p for p in inter["planted"]}
    options = {"ssi_mismatch_x_cross_border": ("ssi_match_status", "is_cross_border"),
               "shortfall_x_security_fail_rate": ("obligation_coverage_ratio", "security_fail_rate_30d"),
               "confirmation_x_overnight": ("hours_to_confirmation", "instruction_hour_bucket")}
    pick = st.selectbox("Planted interaction", list(options), format_func=lambda k: " x ".join(
        feats[f]["label"] for f in options[k]), key="interaction")
    a, b2 = options[pick]
    fig = go.Figure()
    xa = rand[a].astype(str) if hasattr(rand[a], "cat") else rand[a]
    cb = rand[b2].cat.codes if hasattr(rand[b2], "cat") else rand[b2]
    fig.add_scatter(x=xa, y=rand[f"inter__{pick}"], mode="markers",
                    marker=dict(size=4, color=cb, colorscale="Viridis", showscale=True, colorbar=dict(title=feats[b2]["label"])),
                    hovertemplate="%{x}<br>interaction %{y:.3f}<extra></extra>")
    fig.update_layout(height=380, xaxis_title=feats[a]["label"], yaxis_title="SHAP interaction value (log-odds)",
                      margin=dict(t=20))
    st.plotly_chart(fig, key="dependence")
    p = planted.get(frozenset((a, b2)))
    if p:
        st.markdown(f"Among all {inter['n_pairs']} feature pairs, SHAP ranks this planted interaction **#{p['rank']}** "
                    "by mean absolute interaction value. A logistic regression adds each feature's effect separately, "
                    "so it cannot represent that two problems together are worse than the sum of each.")

    st.markdown("#### SHAP versus the planted truth")
    sv = load_json("shap_vs_truth.json")
    rows = pd.DataFrame(sv["rows"])
    fig = go.Figure()
    fig.add_bar(y=[feats[n]["label"] for n in rows["feature"]][::-1], x=(rows["shap_importance"] / rows["shap_importance"].max())[::-1],
                name="Model SHAP importance", orientation="h", marker_color="#1f4e79")
    fig.add_bar(y=[feats[n]["label"] for n in rows["feature"]][::-1], x=(rows["true_importance"] / rows["true_importance"].max())[::-1],
                name="True planted importance", orientation="h", marker_color="#ff9f1c")
    fig.update_layout(barmode="group", height=620, xaxis_title="Importance (scaled to the top feature)", margin=dict(t=20))
    st.plotly_chart(fig, key="truth")
    c = st.columns(2)
    c[0].metric("Rank correlation with the truth", f"{sv['spearman']:.2f}")
    c[1].metric("Every Strong feature above every Weak one", "yes" if sv["strong_above_weak"] else "no")
    gaps = rows.assign(gap=rows["shap_rank"] - rows["true_rank"])
    over = gaps.sort_values("gap").iloc[0]
    under = gaps.sort_values("gap").iloc[-1]
    st.markdown(
        "Because the data is synthetic, each feature's true contribution to the planted log-odds is known, so the "
        "explanations can be checked. Agreement means SHAP is telling the truth about this model and this data. "
        f"The biggest gaps: **{feats[over['feature']]['label']}** ranks higher than its planted effect (rank "
        f"{over['shap_rank']} vs {over['true_rank']}), and **{feats[under['feature']]['label']}** ranks lower "
        f"({under['shap_rank']} vs {under['true_rank']}). Rolling fail rates also stand in for hidden counterparty "
        "and security risk that no feature shows, and a day-level feature can only be learned within the range "
        "seen in training."
    )
    st.dataframe(rows.assign(feature=[feats[n]["label"] for n in rows["feature"]])[
        ["feature", "strength", "shap_rank", "true_rank", "shap_importance", "true_importance"]],
        hide_index=True, height=300)

    st.markdown("#### One flagged trade: why it was flagged and what ops should check first")
    pool = sample[sample["pool"]].sort_values("score__xgb", ascending=False)
    choices = {f"Trade {int(r.trade_id)}: XGBoost {r.score__xgb:.0%}, "
               f"{'failed (' + REASON.get(str(r.fail_reason), '') + ')' if r.failed else 'settled'}": i
               for i, r in pool.head(60).iterrows()}
    t = sample.loc[choices[st.selectbox("Trade (highest XGBoost scores in the test sample)", list(choices), key="trade")]]
    c = st.columns(4)
    c[0].metric("Logistic regression", pct(t["score__logreg"]))
    c[1].metric("SVM", pct(t["score__svm_linear"]))
    c[2].metric("XGBoost", pct(t["score__xgb"]))
    c[3].metric("Outcome", "failed" if t["failed"] else "settled")
    contrib = pd.Series({n: t[f"shap__{n}"] for n in feats}).sort_values(key=np.abs, ascending=False)
    top = contrib.head(8)
    rest = contrib.iloc[8:].sum()
    fig = go.Figure(go.Waterfall(
        orientation="h", measure=["absolute"] + ["relative"] * (len(top) + 1) + ["total"],
        y=["base value"] + [f"{feats[n]['label']} = {_fmt(t[n])}" for n in top.index] + ["other features", "model output"],
        x=[t["shap__bias"]] + list(top.values) + [rest, 0],
        increasing=dict(marker_color=FAIL_COLOR), decreasing=dict(marker_color="#1f77b4"),
    ))
    fig.update_layout(height=430, xaxis_title="Log-odds of failing", margin=dict(t=20), yaxis=dict(autorange="reversed"))
    c1, c2 = st.columns([1.4, 1])
    c1.plotly_chart(fig, key="waterfall")
    truth = pd.Series({n: t[f"true__{n}"] for n in top.index})
    fig = go.Figure(go.Bar(x=truth.values, y=[feats[n]["label"] for n in truth.index], orientation="h",
                           marker_color=["#ff9f1c" if v > 0 else "#9ecae1" for v in truth.values]))
    fig.update_layout(height=430, xaxis_title="True planted contribution", margin=dict(t=20), yaxis=dict(autorange="reversed"))
    c2.plotly_chart(fig, key="true_contrib")
    st.markdown("**What ops should check first**")
    for n in contrib[contrib > 0].head(3).index:
        st.markdown(f"- **{feats[n]['label']}** ({_fmt(t[n])}): {feats[n]['ops_check']}")


def _fmt(v) -> str:
    if isinstance(v, (float, np.floating)):
        return "missing" if np.isnan(v) else f"{v:.3g}"
    return str(v)


# ---- tab 5: ops queue simulator ----------------------------------------------------------------

@st.cache_data
def _ranks(model: str):
    sim = load_parquet("sim_trades.parquet")
    return day_ranks(sim["day"].to_numpy(), sim[model].to_numpy())


def tab_simulator() -> None:
    st.subheader("Ops queue simulator (what-if)")
    st.warning("A what-if on synthetic data under the assumptions you choose below. It shows the mechanism by which "
               "a risk score reduces pending trades and penalties. It is not a measured result.")
    sim = load_parquet("sim_trades.parquet")
    primary = load_json("models.json")["primary"]
    c1, c2 = st.columns(2)
    model = c1.selectbox("Risk model", ["xgb", "svm_linear", "logreg"], format_func=lambda f: label(primary[f]),
                         key="sim_model")
    capacity = c1.slider("Ops capacity: share of each day's trades ops can work", 0.5, 10.0, 2.0, 0.5,
                         format="%.1f%%", key="sim_capacity") / 100
    fix_rate = c1.slider("Fix rate: share of worked would-be fails that ops fix in time", 0, 100, 60, 5,
                         format="%d%%", key="sim_fix") / 100
    cost = c2.number_input("Handling cost per failed trade ($)", 0, 10000, 150, 50, key="sim_cost")
    penalty = c2.number_input("Penalty per failed trade ($)", 0, 100000, 250, 50, key="sim_penalty")
    r = compare(sim["day"].to_numpy(), sim["failed"].to_numpy(), sim[model].to_numpy(), capacity, fix_rate,
                cost, penalty, ranks=_ranks(model))
    q = 63 / r["n_days"]
    m = st.columns(4)
    m[0].metric("Pending fails per quarter, no model", f"{r['pending_without'] * q:,.0f}")
    m[1].metric("Pending fails per quarter, with model", f"{r['pending_with'] * q:,.0f}",
                delta=f"-{r['pending_reduction']:.0%}", delta_color="inverse")
    m[2].metric("Quarterly exposure, no model", f"${r['quarterly_exposure_without']:,.0f}")
    m[3].metric("Quarterly exposure, with model", f"${r['quarterly_exposure_with']:,.0f}",
                delta=f"-${r['quarterly_exposure_without'] - r['quarterly_exposure_with']:,.0f}", delta_color="inverse")
    base, mod = r["without_model"], r["with_model"]
    fig = go.Figure()
    fig.add_scatter(x=base["days"], y=pd.Series(base["pending"]).rolling(5, min_periods=1).mean(), name="no model")
    fig.add_scatter(x=mod["days"], y=pd.Series(mod["pending"]).rolling(5, min_periods=1).mean(), name="with model")
    fig.add_scatter(x=base["days"], y=pd.Series(base["would_fail"]).rolling(5, min_periods=1).mean(),
                    name="would fail if nobody acted", line=dict(dash="dot", color="gray"))
    fig.update_layout(height=360, xaxis_title="Calendar day in the test period", yaxis_title="Pending fails per day (5-day avg)",
                      margin=dict(t=20))
    st.plotly_chart(fig, key="sim_daily")
    st.markdown(
        f"**How it works.** Each day ops work {pct(capacity)} of the day's trades. Without a model they have no "
        f"risk ranking, so on average they reach {pct(capacity)} of the would-be fails. With the model they work "
        f"the highest-scored trades first and reach {pct(r['share_of_fails_worked_with_model'])} of them. "
        f"{pct(fix_rate)} of the would-be fails they reach get fixed in time. Exposure is pending fails times "
        f"(handling cost + penalty), scaled to a 63-business-day quarter. The test period includes a volatility "
        "spike, so its fail rate is higher than in calm months."
    )
    st.caption("Assumptions: ops capacity and fix rate are the same every day; worked trades are fixed independently; "
               "trades nobody works settle or fail as they would have. Scores are calibrated XGBoost, SVM, or logistic "
               "regression probabilities on the synthetic test period.")


def main() -> None:
    st.set_page_config(page_title="Trade Settlement Fail Predictor", page_icon="📉", layout="wide")
    st.title("Trade Settlement Fail Predictor")
    st.info(SYNTHETIC_NOTE)
    tabs = st.tabs(["Data and features", "Model comparison", "How the SVM works", "XGBoost and SHAP",
                    "Ops queue simulator"])
    with tabs[0]:
        tab_data()
    with tabs[1]:
        tab_comparison()
    with tabs[2]:
        tab_svm()
    with tabs[3]:
        tab_xgb()
    with tabs[4]:
        tab_simulator()


main()
