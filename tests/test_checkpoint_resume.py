"""Checkpoint and resume on LocalDirStore, with injected crashes.

A toy two-stage pipeline stands in for the real ones: stage "make" writes one part per unit
plus a rolling state file, stage "combine" reads every part. The real generator and trainer
use the same Checkpoint API, and tests/test_pipeline_smoke_local.py exercises them.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.pipeline import checkpoint as checkpoint_mod
from src.pipeline import store as store_mod
from src.pipeline.checkpoint import Checkpoint, StageSpec
from src.pipeline.run_report import RunReporter, redact
from src.pipeline.runner import Context, Stage, run_stages
from src.pipeline.store import LocalDirStore

CALLS: list[tuple[str, int]] = []
CRASH: dict = {}


class InjectedCrash(RuntimeError):
    pass


def _make(ctx: Context) -> list[str]:
    n = ctx.cfg["toy_make"]["n_units"]
    outputs = []
    for i in range(n):
        part, state = f"toy/part_{i}.txt", f"toy/state_{i}.json"
        outputs.append(part)
        if ctx.ckpt.unit_done("make", f"u{i}"):
            continue
        if CRASH.get("make_unit") == i:
            CRASH.clear()
            raise InjectedCrash(f"crash before unit {i}")
        CALLS.append(("make", i))
        prev = json.loads((ctx.workdir / f"toy/state_{i - 1}.json").read_text()) if i else {"acc": 0}
        value = int(hashlib.sha256(f"{ctx.cfg['toy_make']['salt']}-{i}".encode()).hexdigest()[:8], 16)
        (ctx.workdir / part).parent.mkdir(parents=True, exist_ok=True)
        (ctx.workdir / part).write_text(f"{i},{value}\n")
        (ctx.workdir / state).write_text(json.dumps({"acc": prev["acc"] + value}))
        delete = [f"toy/state_{i - 1}.json"] if i else []
        ctx.ckpt.save_unit("make", f"u{i}", [part, state], delete=delete)
    outputs.append(f"toy/state_{n - 1}.json")
    return outputs


def _combine(ctx: Context) -> list[str]:
    if CRASH.get("combine"):
        CRASH.clear()
        raise InjectedCrash("crash in combine")
    CALLS.append(("combine", 0))
    parts = sorted((ctx.workdir / "toy").glob("part_*.txt"))
    text = "".join(p.read_text() for p in parts) + ctx.cfg["toy_combine"]["footer"]
    (ctx.workdir / "toy/combined.txt").write_text(text)
    return ["toy/combined.txt"]


STAGES = [
    Stage(StageSpec("make", ("src/pipeline/checkpoint.py",), ("toy_make",)), _make),
    Stage(StageSpec("combine", ("src/pipeline/runner.py",), ("toy_combine",)), _combine),
]
ORDER = [s.name for s in STAGES]


def _cfg(n_units=4, salt="a", footer="end\n"):
    return {"toy_make": {"n_units": n_units, "salt": salt}, "toy_combine": {"footer": footer}}


def run_toy(base: Path, cfg: dict, name: str, from_stage=None, force=()):
    """One 'Colab session': fresh local disk, restore from the work store, run all stages."""
    workdir = base / f"session_{name}"
    work = LocalDirStore(base / "work_store")
    reports = LocalDirStore(base / "reports_store")
    ckpt = Checkpoint(work, workdir, cfg, "testcommit", ORDER)
    ckpt.restore()
    rep = RunReporter(reports, workdir, "01_toy", "smoke", "testcommit", workdir / "log.txt", run_id=name)
    ctx = Context(cfg=cfg, workdir=workdir, ckpt=ckpt, report=rep, stores={"work": work})
    run_stages(STAGES, ctx, from_stage, list(force))
    rep.succeed()
    return work, reports


def _store_digest(store: LocalDirStore) -> dict[str, str]:
    return {
        p: hashlib.sha256((store.root / p).read_bytes()).hexdigest()
        for p in store.list_files()
        if p != "manifest.json"
    }


@pytest.fixture(autouse=True)
def _reset():
    CALLS.clear()
    CRASH.clear()
    yield
    CALLS.clear()
    CRASH.clear()


def test_crash_and_resume_matches_uninterrupted_run(tmp_path):
    clean, _ = run_toy(tmp_path / "a", _cfg(), "clean")
    CALLS.clear()

    CRASH["make_unit"] = 2
    with pytest.raises(InjectedCrash):
        run_toy(tmp_path / "b", _cfg(), "first")
    assert CALLS == [("make", 0), ("make", 1)]
    CALLS.clear()

    resumed, _ = run_toy(tmp_path / "b", _cfg(), "second")
    # Units 0 and 1 were checkpointed, so only 2 and 3 run again.
    assert CALLS == [("make", 2), ("make", 3), ("combine", 0)]
    assert _store_digest(resumed) == _store_digest(clean)
    # Only the newest rolling state is kept in the store.
    assert [p for p in resumed.list_files() if "state_" in p] == ["toy/state_3.json"]


def test_crash_in_the_middle_of_a_commit_is_all_or_nothing(tmp_path, monkeypatch):
    clean, _ = run_toy(tmp_path / "a", _cfg(), "clean")
    CALLS.clear()

    real_copy = store_mod.shutil.copy2
    copies = {"n": 0}

    def flaky_copy(src, dst, *a, **k):
        # Unit 0 commits part, state, manifest (3 copies). Die after unit 1's first file.
        copies["n"] += 1
        if copies["n"] == 5:
            raise InjectedCrash("disk died mid-commit")
        return real_copy(src, dst, *a, **k)

    monkeypatch.setattr(store_mod.shutil, "copy2", flaky_copy)
    with pytest.raises(InjectedCrash):
        run_toy(tmp_path / "b", _cfg(), "first")
    monkeypatch.setattr(store_mod.shutil, "copy2", real_copy)

    manifest = json.loads((tmp_path / "b/work_store/manifest.json").read_text())
    assert list(manifest["stages"]["make"]["units"]) == ["u0"]
    CALLS.clear()
    resumed, _ = run_toy(tmp_path / "b", _cfg(), "second")
    assert CALLS == [("make", 1), ("make", 2), ("make", 3), ("combine", 0)]
    assert _store_digest(resumed) == _store_digest(clean)


def test_done_stages_are_skipped(tmp_path):
    run_toy(tmp_path, _cfg(), "one")
    CALLS.clear()
    _, reports = run_toy(tmp_path, _cfg(), "two")
    assert CALLS == []
    latest = json.loads((reports.root / "runs/latest.json").read_text())
    assert latest["status"] == "passed" and latest["run_id"] == "two"
    summary = json.loads((reports.root / "runs/two/summary.json").read_text())
    assert summary["stages_skipped"] == ["make", "combine"]


def test_config_change_reruns_that_stage_and_every_later_stage(tmp_path):
    run_toy(tmp_path, _cfg(), "one")
    CALLS.clear()
    run_toy(tmp_path, _cfg(footer="changed\n"), "two")
    assert CALLS == [("combine", 0)]
    CALLS.clear()
    run_toy(tmp_path, _cfg(salt="b", footer="changed\n"), "three")
    assert CALLS == [("make", 0), ("make", 1), ("make", 2), ("make", 3), ("combine", 0)]


def test_code_change_reruns(tmp_path, monkeypatch):
    run_toy(tmp_path, _cfg(), "one")
    CALLS.clear()
    real = checkpoint_mod.code_hash
    monkeypatch.setattr(
        checkpoint_mod, "code_hash", lambda s: "edited" if "src/pipeline/checkpoint.py" in s else real(s)
    )
    run_toy(tmp_path, _cfg(), "two")
    assert CALLS == [("make", 0), ("make", 1), ("make", 2), ("make", 3), ("combine", 0)]


def test_corrupted_output_triggers_rerun(tmp_path):
    work, _ = run_toy(tmp_path, _cfg(), "one")
    (work.root / "toy/combined.txt").write_text("tampered")
    CALLS.clear()
    run_toy(tmp_path, _cfg(), "two")
    assert CALLS == [("combine", 0)]
    assert (work.root / "toy/combined.txt").read_text().endswith("end\n")


def test_later_stage_stays_invalid_across_a_restart(tmp_path):
    """make re-runs (new salt) and the session dies before combine. combine must not be
    skipped on the next session, even though its own config and code did not change."""
    run_toy(tmp_path, _cfg(), "one")
    CALLS.clear()
    CRASH["combine"] = True
    with pytest.raises(InjectedCrash):
        run_toy(tmp_path, _cfg(salt="b"), "two")
    CALLS.clear()
    work, _ = run_toy(tmp_path, _cfg(salt="b"), "three")
    assert CALLS == [("combine", 0)]
    clean, _ = run_toy(tmp_path / "fresh", _cfg(salt="b"), "clean")
    assert _store_digest(work) == _store_digest(clean)


def test_force_and_from_stage(tmp_path):
    run_toy(tmp_path, _cfg(), "one")
    CALLS.clear()
    run_toy(tmp_path, _cfg(), "two", force=["combine"])
    assert CALLS == [("combine", 0)]
    CALLS.clear()
    run_toy(tmp_path, _cfg(), "three", from_stage="make")
    assert CALLS == [("make", 0), ("make", 1), ("make", 2), ("make", 3), ("combine", 0)]
    with pytest.raises(ValueError, match="unknown stages"):
        run_toy(tmp_path, _cfg(), "four", force=["nope"])


def test_failure_report_names_the_stage_and_redacts_tokens(tmp_path, monkeypatch):
    fake = "hf_" + "x" * 30
    monkeypatch.setenv("HF_TOKEN", fake)
    CRASH["make_unit"] = 1
    with pytest.raises(InjectedCrash):
        run_toy(tmp_path, _cfg(), "boom")
    latest = json.loads((tmp_path / "reports_store/runs/latest.json").read_text())
    assert latest["status"] == "failed"
    assert latest["failed_stage"] == "make"
    assert "InjectedCrash" in latest["error"]
    for p in (tmp_path / "reports_store").rglob("*"):
        if p.is_file():
            assert fake not in p.read_text()
    assert redact(f"token={fake} Authorization: Bearer abc") == "token=[REDACTED] Authorization: Bearer [REDACTED]"


# ---- the real generator: month-by-month resume ---------------------------------------------


def _tiny_generator_cfg():
    from src.config import load_config

    return load_config("smoke", overrides={
        "data": {"n_days": 62, "trades_per_day": 60},
        "reference": {"n_counterparties": 40, "n_securities": 300},
    })


def _run_generator(base: Path, cfg: dict, name: str):
    from src.data.generator import generate_months
    from src.data.reference_data import build_reference

    work = LocalDirStore(base / "work_store")
    workdir = base / f"session_{name}"
    ckpt = Checkpoint(work, workdir, cfg, "testcommit", ["generate"])
    ckpt.restore()
    spec = StageSpec("generate", ("src/data/*.py",), ("seed", "data", "reference"))
    ckpt.begin(spec)
    outputs = generate_months(build_reference(cfg), cfg, -4.0, workdir, ckpt, "generate")
    ckpt.finish("generate", outputs)
    return work


def test_generator_restart_continues_with_identical_data(tmp_path, monkeypatch):
    from src.data import generator

    cfg = _tiny_generator_cfg()
    clean = _run_generator(tmp_path / "a", cfg, "clean")
    assert len([p for p in clean.list_files() if p.startswith("gen/trades/")]) == 3

    real_day = generator.generate_day
    calls = []

    def crashing_day(t, *a, **k):
        calls.append(t)
        if t == 30:  # inside the second month
            raise InjectedCrash("runtime disconnected")
        return real_day(t, *a, **k)

    monkeypatch.setattr(generator, "generate_day", crashing_day)
    with pytest.raises(InjectedCrash):
        _run_generator(tmp_path / "b", cfg, "first")
    calls.clear()
    monkeypatch.setattr(generator, "generate_day", lambda t, *a, **k: (calls.append(t), real_day(t, *a, **k))[1])
    resumed = _run_generator(tmp_path / "b", cfg, "second")
    assert min(calls) == 21  # month 1 (21 business days) was not generated again
    # The data files are byte-identical. The final state is compared by content, because a
    # pickle of reloaded numpy arrays can differ in bytes while holding the same values.
    state = "gen/state/state_2024-03.pkl"
    a, b = _store_digest(clean), _store_digest(resumed)
    assert {k: v for k, v in a.items() if k != state} == {k: v for k, v in b.items() if k != state}
    _assert_same_state(generator.load_state(clean.root / state), generator.load_state(resumed.root / state))


def _assert_same_state(x, y):
    import numpy as np

    assert (x.day, x.next_trade_id) == (y.day, y.next_trade_id)
    np.testing.assert_array_equal(x.ssi_verified_ord, y.ssi_verified_ord)
    assert list(x.outcomes) == list(y.outcomes)
    for k in x.outcomes:
        for u, v in zip(x.outcomes[k], y.outcomes[k]):
            np.testing.assert_array_equal(u, v)
    for xs, ys in ((x.pair_days, y.pair_days), (x.notional_days, y.notional_days)):
        assert len(xs) == len(ys)
        for u, v in zip(xs, ys):
            assert u[0] == v[0]
            for p, q in zip(u[1:], v[1:]):
                np.testing.assert_array_equal(p, q)
