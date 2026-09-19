from __future__ import annotations

import os
import subprocess
import sys
import tomllib
from pathlib import Path

import invarlock.diagnostics as diagnostics

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_diagnostics_are_a_core_optional_feature() -> None:
    payload = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert payload["project"]["name"] == "invarlock"
    assert payload["project"]["optional-dependencies"]["diagnostics"] == ["numpy>=1.24"]
    assert not (REPO_ROOT / "addins").exists()


def test_public_api_is_closed_to_three_observations_and_one_input_error() -> None:
    assert diagnostics.__all__ == [
        "DiagnosticInputError",
        "canonical_observation_bytes",
        "rmt_observation",
        "spectral_observation",
        "variance_observation",
    ]


def test_diagnostics_import_is_lazy_and_missing_numpy_has_install_guidance() -> None:
    probe = """
import importlib.abc
import sys

attempts = []
class BlockNumpy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] == "numpy":
            attempts.append(fullname)
            raise ModuleNotFoundError("NumPy is unavailable", name=fullname)

sys.meta_path.insert(0, BlockNumpy())
import invarlock.diagnostics as diagnostics
assert attempts == []
assert "invarlock.diagnostics.observations" not in sys.modules
assert not hasattr(diagnostics, "unknown_observation")
assert attempts == []

for name in diagnostics.__all__:
    try:
        getattr(diagnostics, name)
    except ImportError as exc:
        assert "invarlock[diagnostics]" in str(exc), str(exc)
    else:
        raise AssertionError(f"{name} unexpectedly resolved without NumPy")
assert len(attempts) == len(diagnostics.__all__)
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(REPO_ROOT / "src")
    result = subprocess.run(
        [sys.executable, "-c", probe],
        env=environment,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_star_import_exposes_the_original_diagnostic_objects() -> None:
    import invarlock.diagnostics.observations as observations

    exported: dict[str, object] = {}
    exec("from invarlock.diagnostics import *", exported)

    assert set(exported) - {"__builtins__"} == set(diagnostics.__all__)
    for name in diagnostics.__all__:
        assert exported[name] is getattr(observations, name)
