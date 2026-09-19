from __future__ import annotations

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
