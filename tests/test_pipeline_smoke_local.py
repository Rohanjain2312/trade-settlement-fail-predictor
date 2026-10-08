"""End-to-end smoke run of both notebook pipelines on LocalDirStore (no network), with an
injected crash and a resume in each, then the app is run on the deployed Space files."""

from __future__ import annotations

import json

import pytest

from src.config import load_config
from src.pipeline import run_generate, run_train
from src.pipeline.runner import local_stores, run_mode

pytestmark = pytest.mark.smoke


class Crash(RuntimeError):
    pass


@pytest.fixture(scope="module")
def smoke_run(tmp_path_factory):
    """Run notebook 01 then notebook 02 once each (with crashes and resumes)."""
    mp = pytest.MonkeyPatch()
    root = tmp_path_factory.mktemp("smoke")
    cfg = load_config("smoke")
    stores = local_stores(root / "hub", cfg)
    workdir = root / "colab"
    calls = {"gen_days": [], "resumed_xgb": []}

    # Notebook 01: crash inside month 4 (day 70) of the real generation, then resume.
    from src.data import generator

    real_day = generator.generate_day
    crashed = {"gen": False, "xgb": False}

    def flaky_day(t, *a, **k):
        real = len(a) < 5 or a[4] is None
        if real and k.get("trades_per_day") is None:
            calls["gen_days"].append(t)
        if t == 70 and real and k.get("trades_per_day") is None and not crashed["gen"]:
            crashed["gen"] = True
            raise Crash("runtime disconnected during month 4")
        return real_day(t, *a, **k)

    mp.setattr(generator, "generate_day", flaky_day)
    with pytest.raises(Crash):
        run_mode("01_generate_data", "smoke", run_generate.build_stages(), stores, workdir, on_hub=False)
    first_latest = json.loads((stores["reports"].root / "runs/latest.json").read_text())
    calls["gen_days"].clear()
    run_mode("01_generate_data", "smoke", run_generate.build_stages(), stores, workdir, on_hub=False)
    resumed_gen_days = list(calls["gen_days"])

    # Notebook 02: crash on the second XGBoost checkpoint write, then resume.
    from src.models import train_xgb

    real_save, real_train = train_xgb.save_booster, train_xgb.train
    saves = {"n": 0}

    def flaky_save(*a, **k):
        saves["n"] += 1
        if saves["n"] == 2 and not crashed["xgb"]:
            crashed["xgb"] = True
            raise Crash("runtime disconnected during boosting")
        return real_save(*a, **k)

    def spy_train(*a, **k):
        calls["resumed_xgb"].append(a[7] if len(a) > 7 else k.get("resume"))
        return real_train(*a, **k)

    mp.setattr(train_xgb, "save_booster", flaky_save)
    mp.setattr(train_xgb, "train", spy_train)
    with pytest.raises(Crash):
        run_mode("02_train_explain", "smoke", run_train.build_stages(), stores, workdir, on_hub=False)
    run_mode("02_train_explain", "smoke", run_train.build_stages(), stores, workdir, on_hub=False)
    mp.undo()
    return {"stores": stores, "first_latest": first_latest, "resumed_gen_days": resumed_gen_days,
            "calls": calls, "workdir": workdir}


def test_generation_resumed_after_the_crash(smoke_run):
    assert smoke_run["first_latest"]["status"] == "failed"
    assert smoke_run["first_latest"]["failed_stage"] == "generate"
    # Months 1 to 3 (days 0 to 61) were checkpointed and not generated again.
    assert min(smoke_run["resumed_gen_days"]) == 62


def test_dataset_was_published(smoke_run):
    files = set(smoke_run["stores"]["dataset"].list_files())
    assert len([f for f in files if f.startswith("data/trades/")]) == 6
    for f in ("README.md", "samples/test_sample.parquet", "truth/effect_weights.json",
              "truth/counterparty_latent.parquet", "data/reference/counterparties.parquet",
              "reports/validation_summary.json"):
        assert f in files
    summary = json.loads((smoke_run["stores"]["dataset"].root / "reports/validation_summary.json").read_text())
    assert summary["passed"], summary["checks"]


def test_xgboost_resumed_from_its_checkpoint(smoke_run):
    resumes = [r for r in smoke_run["calls"]["resumed_xgb"] if r is not None]
    assert len(resumes) == 1, "exactly one boosting run resumed from a checkpoint"
    assert resumes[0][1]["next_iteration"] > 0


def test_models_metrics_and_reports_were_published(smoke_run):
    stores = smoke_run["stores"]
    files = set(stores["model"].list_files())
    for f in ("README.md", "logreg.joblib", "svm_linear_calibrated.joblib", "svm_rbf_calibrated.joblib",
              "xgb_model.json", "xgb_calibrated.joblib", "feature_schema.json", "metrics.json",
              "preprocessor.joblib", "shap/shap_vs_truth.json", "shap/global_importance.json",
              "shap/interaction_summary.json", "heldout/S1.json", "report/RESULTS.md", "report/shap_beeswarm.png",
              "report/svm_boundary.png", "report/simulator.png"):
        assert f in files
    metrics = json.loads((stores["model"].root / "metrics.json").read_text())
    assert len(metrics["models"]) == 12  # 4 model families x 3 resampling variants
    for m in metrics["models"].values():
        assert 0.0 < m["test"]["pr_auc"] <= 1.0
        assert m["test"]["pr_auc"] > m["test"]["base_rate"]  # better than random ranking
        assert "S1" in m["per_scenario"] and "S1+S6" in m["per_scenario"]
    results = (stores["model"].root / "report/RESULTS.md").read_text()
    assert "synthetic" in results.lower() and "![SHAP vs truth](shap_vs_truth.png)" in results
    truth = json.loads((stores["model"].root / "shap/shap_vs_truth.json").read_text())
    assert len(truth["rows"]) == 20 and -1 <= truth["spearman"] <= 1
    latest = json.loads((stores["reports"].root / "runs/latest.json").read_text())
    assert latest["status"] == "passed" and latest["notebook"] == "02_train_explain"
    assert any(p.startswith("runs/smoke_passed/02_train_explain_") for p in stores["reports"].list_files())


def test_space_files_and_app_run(smoke_run, monkeypatch):
    from streamlit.testing.v1 import AppTest

    space = smoke_run["stores"]["space"]
    files = set(space.list_files())
    for f in ("README.md", "Dockerfile", "requirements.txt", "streamlit_app.py", "simulate_queue.py",
              "assets/metrics.json", "assets/app_sample.parquet", "assets/svm_views.json", "assets/xgb_views.json",
              "assets/sim_trades.parquet", "assets/coverage.json", "assets/shap_vs_truth.json",
              "assets/lr_coefficients.json"):
        assert f in files
    readme = (space.root / "README.md").read_text()
    assert "sdk: docker" in readme and "app_port: 8501" in readme and "synthetic" in readme.lower()
    monkeypatch.setenv("APP_ASSETS", str(space.root / "assets"))
    at = AppTest.from_file(str(space.root / "streamlit_app.py"), default_timeout=120)
    at.run()
    assert not at.exception, [e.message for e in at.exception]
    assert any("synthetic" in i.value.lower() for i in at.info)
    assert len(at.tabs) == 5
    # Exercise the interactive views: RBF kernel, another trade, another model in the simulator.
    at.radio(key="svm_kernel").set_value("rbf").run()
    at.select_slider(key="svm_C").set_value(100.0).run()
    trades = at.selectbox(key="trade").options
    at.selectbox(key="trade").set_value(trades[-1]).run()
    at.selectbox(key="sim_model").set_value("logreg").run()
    at.selectbox(key="interaction").set_value("confirmation_x_overnight").run()
    at.toggle(key="all_variants").set_value(False).run()
    at.slider(key="lr_n").set_value(8).run()
    assert not at.exception, [e.message for e in at.exception]
