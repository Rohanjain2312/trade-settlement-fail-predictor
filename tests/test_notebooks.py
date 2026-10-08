"""Both notebooks are valid JSON with exactly the two expected code cells, so Rohan never
has to edit a notebook, and they request the right Colab runtime."""

from __future__ import annotations

import json
from pathlib import Path

import nbformat
import pytest

NB = Path(__file__).resolve().parents[1] / "notebooks"
EXPECTED = {
    "01_generate_data.ipynb": ("src.pipeline.run_generate", None),
    "02_train_explain.ipynb": ("src.pipeline.run_train", "G4"),
}


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_notebook_has_two_cells_and_runtime(name):
    raw = json.loads((NB / name).read_text())
    nb = nbformat.reads(json.dumps(raw), as_version=4)
    nbformat.validate(nb)
    module, gpu = EXPECTED[name]
    assert [c.cell_type for c in nb.cells] == ["code", "code"]
    setup, run = (c.source for c in nb.cells)
    assert '"git", "clone"' in setup and "requirements-colab.txt" in setup and "HF_TOKEN" in setup
    assert "src.pipeline.setup" in setup
    assert module in run and "--crashed" in run
    assert all(not c.outputs for c in nb.cells)
    assert nb.metadata.get("colab", {}).get("gpuType") == gpu
    assert (nb.metadata.get("accelerator") == "GPU") == (gpu is not None)
    for src in (setup, run):
        compile(src, name, "exec")  # valid Python
        assert "hf_" not in src  # no token in the notebook
