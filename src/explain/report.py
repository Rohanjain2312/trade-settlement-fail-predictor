"""Static fallback report: RESULTS.md with plots, published to the model repo under report/.

Built from the same precomputed assets the app uses, so it tells the same story if the Space
or the network is down.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.features.definitions import BY_NAME, CATEGORICAL, FEATURE_NAMES
from src.sim.simulate_queue import compare

FAMILY = {"logreg": "Logistic regression", "svm_linear": "Linear SVM", "svm_rbf": "RBF SVM", "xgb": "XGBoost"}
RES = {"none": "no resampling", "class_weight": "class weights", "scale_pos_weight": "class weights", "smote": "SMOTENC"}
SIM_DEFAULTS = {"capacity_share": 0.02, "fix_rate": 0.6, "cost_per_fail": 150, "penalty_per_fail": 250}


def _label(name: str) -> str:
    f, r = name.split("__")
    return f"{FAMILY[f]} ({RES[r]})"


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def build(app: Path, metrics_path: Path, out: Path) -> list[str]:
    """Write the plots and RESULTS.md under out from the app assets and metrics.json."""
    out.mkdir(parents=True, exist_ok=True)
    J = lambda name: json.loads((app / name).read_text())  # noqa: E731
    metrics = json.loads(metrics_path.read_text())["models"]
    primary = J("models.json")["primary"]
    s, truth, held = J("data_summary.json"), J("shap_vs_truth.json"), J("heldout.json")
    svm, inter, lr = J("svm_views.json"), J("interaction_summary.json"), J("lr_coefficients.json")
    sample = pd.read_parquet(app / "app_sample.parquet")
    rand = sample[~sample["pool"]]
    files, results = [], {}

    def plot(name, draw, size=(7, 4.2)):
        fig, ax = plt.subplots(figsize=size)
        draw(fig, ax)
        _save(fig, out / name)
        files.append(name)

    def pr(fig, ax):
        for n in primary.values():
            c = metrics[n]["test"]["pr_curve"]
            ax.plot(c["recall"], c["precision"], label=f"{_label(n)}  PR-AUC {metrics[n]['test']['pr_auc']:.3f}")
        ax.axhline(metrics[primary["xgb"]]["test"]["base_rate"], ls=":", c="gray", label="random ranking")
        ax.set(xlabel="Recall", ylabel="Precision", title="Precision-recall on the test period")
        ax.legend(fontsize=8)

    def resampling(fig, ax):
        fams = list(FAMILY)
        width = 0.25
        for i, res in enumerate(("none", "class_weight", "smote")):
            vals = []
            for fam in fams:
                n = next((m for m in metrics if m.startswith(fam + "__") and metrics[m]["resampling"] in
                          ((res, "scale_pos_weight") if res == "class_weight" else (res,))), None)
                vals.append(metrics[n]["test"]["pr_auc"] if n else np.nan)
            ax.bar(np.arange(len(fams)) + (i - 1) * width, vals, width, label=RES[res])
        ax.set_xticks(range(len(fams)), [FAMILY[f] for f in fams])
        ax.set(ylabel="Test PR-AUC", title="No resampling vs class weights vs SMOTENC")
        ax.legend(fontsize=8)

    def calibration(fig, ax):
        ax.plot([0, 0.5], [0, 0.5], ":", c="gray")
        for fam in ("logreg", "xgb", "svm_linear"):
            m = metrics[primary[fam]]
            c = m["test"]["calibration_curve"]
            ax.plot(c["mean_predicted"], c["fraction_failed"], "o-", label=f"{_label(primary[fam])} calibrated")
            if "calibration_curve" in m["raw_test"]:
                r = m["raw_test"]["calibration_curve"]
                ax.plot(r["mean_predicted"], r["fraction_failed"], "x--", label=f"{FAMILY[fam]} raw")
        ax.set(xlabel="Predicted fail probability", ylabel="Observed fail rate", title="Calibration (test period)")
        ax.legend(fontsize=7)

    def scenarios(fig, ax):
        scen = list(metrics[primary["xgb"]]["per_scenario"])
        z = np.array([[metrics[n]["per_scenario"][k].get("recall_at_threshold", np.nan) or np.nan
                       for n in primary.values()] for k in scen], dtype=float)
        im = ax.imshow(z, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        ax.set_yticks(range(len(scen)), scen, fontsize=8)
        ax.set_xticks(range(len(primary)), [FAMILY[f] for f in primary], fontsize=8)
        for i in range(len(scen)):
            for j in range(len(primary)):
                if np.isfinite(z[i, j]):
                    ax.text(j, i, f"{z[i, j]:.2f}", ha="center", va="center", fontsize=7)
        fig.colorbar(im, ax=ax)
        ax.set_title("Recall at ops capacity, per scenario")

    def heldout(fig, ax):
        x = np.arange(len(held))
        ax.bar(x - 0.2, [h["full_model"]["recall_at_capacity"] for h in held], 0.4, label="trained on everything")
        ax.bar(x + 0.2, [h["without_scenario"]["recall_at_capacity"] for h in held], 0.4, label="scenario removed")
        ax.set_xticks(x, [h["scenario"] for h in held])
        ax.set(ylabel="Recall at ops capacity", title="Held-out scenario experiment (XGBoost)")
        ax.legend(fontsize=8)

    def shap_truth(fig, ax):
        rows = pd.DataFrame(truth["rows"]).iloc[::-1]
        y = np.arange(len(rows))
        ax.barh(y + 0.2, rows["shap_importance"] / rows["shap_importance"].max(), 0.4, label="model SHAP")
        ax.barh(y - 0.2, rows["true_importance"] / rows["true_importance"].max(), 0.4, label="planted truth")
        ax.set_yticks(y, [BY_NAME[n].label for n in rows["feature"]], fontsize=7)
        ax.set(xlabel="Importance (scaled to the top feature)",
               title=f"SHAP vs truth: rank correlation {truth['spearman']:.2f}")
        ax.legend(fontsize=8)

    def lr_coef(fig, ax):
        df = pd.DataFrame(lr).assign(abs=lambda d: d["coefficient"].abs()).sort_values("abs").tail(18)
        ax.barh(df["column"], df["coefficient"], color=np.where(df["coefficient"] > 0, "#d62728", "#1f77b4"))
        ax.tick_params(axis="y", labelsize=7)
        ax.set(xlabel="Coefficient (log-odds per standard deviation or per level)",
               title="Logistic regression baseline: largest coefficients")

    def boundary(fig, axes):
        b = svm["boundary"]
        pts = b["points"]
        for ax, pick in zip(axes, (("linear", 1.0, None), ("rbf", 1.0, 1.0))):
            st = next(x for x in b["settings"] if (x["kernel"], x["C"], x["gamma"]) == pick)
            ax.contourf(b["grid"]["x"], b["grid"]["y"], np.array(st["decision"]), levels=12, cmap="RdBu_r", alpha=0.35)
            ax.contour(b["grid"]["x"], b["grid"]["y"], np.array(st["decision"]), levels=[-1, 0, 1], colors="k",
                       linestyles=["--", "-", "--"])
            y = np.array(pts["failed"])
            ax.scatter(np.array(pts["x"])[y == 0], np.array(pts["y"])[y == 0], s=5, c="gray", alpha=0.5)
            ax.scatter(np.array(pts["x"])[y == 1], np.array(pts["y"])[y == 1], s=5, c="#d62728", alpha=0.6)
            sv = np.array(st["support"])
            ax.scatter(np.array(pts["x"])[sv], np.array(pts["y"])[sv], s=25, facecolors="none", edgecolors="k", lw=0.5)
            ax.set(title=f"{pick[0]} kernel, C={pick[1]:g}" + (f", gamma={pick[2]:g}" if pick[2] else "")
                   + f": {st['n_support']} support vectors", xlabel=b["labels"][0], ylabel=b["labels"][1])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    boundary(fig, axes)
    _save(fig, out / "svm_boundary.png")
    files.append("svm_boundary.png")

    def smote(fig, ax):
        sm = svm["smote"]
        ax.scatter(sm["settled"]["x"], sm["settled"]["y"], s=3, c="gray", alpha=0.2, label="settled")
        ax.scatter(sm["failed"]["x"], sm["failed"]["y"], s=12, c="#d62728", label="real fails")
        k = 80
        fx, fy, syn = np.array(sm["failed"]["x"]), np.array(sm["failed"]["y"]), sm["synthetic"]
        for i in range(k):
            p, q = syn["parent"][i], syn["partner"][i]
            ax.plot([fx[p], fx[q]], [fy[p], fy[q]], c="#ff9f1c", lw=0.5, alpha=0.6)
        ax.scatter(syn["x"][:k], syn["y"][:k], s=18, marker="D", c="#ff9f1c", label="synthetic fails")
        ax.set(xlabel=sm["labels"][0], ylabel=sm["labels"][1], title="SMOTE interpolates between neighboring fails")
        ax.legend(fontsize=8)

    def sim(fig, ax):
        tr = pd.read_parquet(app / "sim_trades.parquet")
        r = compare(tr["day"].to_numpy(), tr["failed"].to_numpy(), tr["xgb"].to_numpy(),
                    SIM_DEFAULTS["capacity_share"], SIM_DEFAULTS["fix_rate"], SIM_DEFAULTS["cost_per_fail"],
                    SIM_DEFAULTS["penalty_per_fail"])
        results["sim"] = r
        roll = lambda v: pd.Series(v).rolling(5, min_periods=1).mean()  # noqa: E731
        ax.plot(r["without_model"]["days"], roll(r["without_model"]["pending"]), label="no model")
        ax.plot(r["with_model"]["days"], roll(r["with_model"]["pending"]), label="with XGBoost ranking")
        ax.set(xlabel="Calendar day in the test period", ylabel="Pending fails per day (5-day avg)",
               title="Ops queue what-if (2% capacity, 60% fix rate)")
        ax.legend(fontsize=8)

    for name, draw in (("pr_curves.png", pr), ("resampling.png", resampling), ("calibration.png", calibration),
                       ("per_scenario.png", scenarios), ("heldout.png", heldout), ("shap_vs_truth.png", shap_truth),
                       ("lr_coefficients.png", lr_coef), ("smote.png", smote), ("simulator.png", sim)):
        plot(name, draw, size=(7, 6.5) if name in ("per_scenario.png", "shap_vs_truth.png", "lr_coefficients.png") else (7, 4.2))

    import shap

    names = list(FEATURE_NAMES)
    data = np.column_stack([rand[n].cat.codes.to_numpy() if n in CATEGORICAL else rand[n].to_numpy(dtype=float)
                            for n in names])
    exp = shap.Explanation(values=rand[[f"shap__{n}" for n in names]].to_numpy(),
                           base_values=rand["shap__bias"].to_numpy(), data=data,
                           feature_names=[BY_NAME[n].label for n in names])
    shap.plots.beeswarm(exp, max_display=15, show=False)
    plt.gcf().set_size_inches(8, 6)
    _save(plt.gcf(), out / "shap_beeswarm.png")
    files.append("shap_beeswarm.png")

    (out / "RESULTS.md").write_text(results_md(s, metrics, primary, truth, held, svm, inter, results["sim"]))
    files.append("RESULTS.md")
    return files


def results_md(s, metrics, primary, truth, held, svm, inter, sim) -> str:
    t = {f: metrics[n]["test"] for f, n in primary.items()}
    rows = ["| Model | Resampling | PR-AUC | Recall top 2% | Recall @ precision 0.5 | Brier |", "|---|---|---|---|---|---|"]
    for m in metrics.values():
        x = m["test"]
        rows.append(f"| {FAMILY[m['family']]} | {RES[m['resampling']]} | {x['pr_auc']:.3f} | "
                    f"{x['recall_top_2pct']:.3f} | {x['recall_at_precision_50']:.3f} | {x['brier']:.4f} |")
    held_rows = [f"| {h['scenario']} {h['name']} | {h['full_model']['recall_at_capacity']:.2f} | "
                 f"{h['without_scenario']['recall_at_capacity']:.2f} |" for h in held]
    planted = ", ".join(f"{p['a']} x {p['b']} (#{p['rank']})" for p in inter["planted"])
    dt = {d["model"]: d for d in svm["default_threshold"]}
    lk = svm["leakage"]
    return "\n".join([
        "# Results: Trade Settlement Fail Predictor",
        "",
        "> **All data is synthetic**, generated by this project's seeded generator. These results show the "
        "pipeline and the modeling approach, not performance on real trades. This page is the static fallback "
        "for the demo Space.",
        "",
        f"Data: {s['n_trades']:,} trades from {s['first_date']} to {s['last_date']}, fail rate {s['fail_rate']:.2%}. "
        f"Splits by trade date (train, validation, test) with a {s['gap_days']}-business-day gap. The test period "
        f"has a volatility spike, so its fail rate is {t['xgb']['base_rate']:.2%}.",
        "",
        "## Model comparison (test period)",
        "",
        *rows,
        "",
        f"Always predicting 'settles' would score {t['xgb']['accuracy_if_always_settles']:.1%} accuracy, so the "
        "headline metrics are ranking metrics.",
        "",
        "![PR curves](pr_curves.png)",
        "",
        "## Does SMOTE help?",
        "",
        "![Resampling comparison](resampling.png)",
        "",
        "On ranking quality (PR-AUC), SMOTENC did not beat no resampling or class weights in any model family here. "
        "What SMOTE changes is the default decision rule. Trained on the natural fail rate, a raw linear SVM flags "
        f"{dt['svm_linear__none']['flag_rate']:.2%} of trades and catches {dt['svm_linear__none']['recall']:.1%} of fails "
        f"with its own threshold; with SMOTENC it flags {dt['svm_linear__smote']['flag_rate']:.1%} and catches "
        f"{dt['svm_linear__smote']['recall']:.1%}.",
        "",
        f"Leakage check: the median distance from a test fail to the nearest synthetic fail is "
        f"{lk['median_distance_right']:.2f} when SMOTE runs on training rows only, and {lk['median_distance_wrong']:.2f} "
        "when it runs before the split. SMOTENC therefore sits inside the imblearn pipeline.",
        "",
        "## How the SVM works (2-feature illustration)",
        "",
        "![SVM boundary](svm_boundary.png)",
        "",
        "![SMOTE](smote.png)",
        "",
        "## Calibration",
        "",
        "![Calibration](calibration.png)",
        "",
        "## Logistic regression baseline",
        "",
        "![LR coefficients](lr_coefficients.png)",
        "",
        "## XGBoost and SHAP",
        "",
        "![SHAP beeswarm](shap_beeswarm.png)",
        "",
        f"SHAP versus the planted truth: rank correlation **{truth['spearman']:.2f}**; every Strong feature ranks above "
        f"every Weak one: **{truth['strong_above_weak']}**. Planted interactions rank among all feature pairs: {planted}.",
        "",
        "![SHAP vs truth](shap_vs_truth.png)",
        "",
        "## Scenario slices and held-out scenarios",
        "",
        "![Per scenario](per_scenario.png)",
        "",
        "| Held-out scenario | Recall at capacity, full model | Without the scenario in training |",
        "|---|---|---|",
        *held_rows,
        "",
        "![Held-out](heldout.png)",
        "",
        "## Ops queue what-if",
        "",
        f"Assumptions: ops work {SIM_DEFAULTS['capacity_share']:.0%} of each day's trades and fix "
        f"{SIM_DEFAULTS['fix_rate']:.0%} of the would-be fails they reach. Under these assumptions, ranking by XGBoost "
        f"cuts pending fails by {sim['pending_reduction']:.0%} versus working trades without a ranking. This is a "
        "what-if on synthetic data, not a measured result; the app lets you change every assumption.",
        "",
        "![Simulator](simulator.png)",
        "",
    ])
