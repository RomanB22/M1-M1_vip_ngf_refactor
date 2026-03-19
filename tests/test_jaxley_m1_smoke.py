from __future__ import annotations

from pathlib import Path
import json
import subprocess
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_jaxley_fit_smoke(tmp_path):
    pytest.importorskip("jax")
    pytest.importorskip("jaxley")
    pytest.importorskip("optax")

    outdir = tmp_path / "optimization" / "jaxley"
    result = subprocess.run(
        [
            sys.executable,
            "src_test/jaxley_fit.py",
            "--steps",
            "3",
            "--lr",
            "0.01",
            "--seed",
            "0",
            "--outdir",
            str(outdir),
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    run_dirs = [path for path in outdir.iterdir() if path.is_dir()]
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    assert (run_dir / "best_params.json").exists()
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "history.json").exists()
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert "loss" in payload
