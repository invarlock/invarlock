from __future__ import annotations

import tomllib
from pathlib import Path

import invarlock_addins.diagnostics as diagnostics

ADDIN_ROOT = Path(__file__).resolve().parents[1]


def test_distribution_is_standalone_and_has_no_command_or_plugin_entrypoint() -> None:
    payload = tomllib.loads((ADDIN_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert payload["project"]["name"] == "invarlock-diagnostics"
    assert payload["project"]["dependencies"] == ["numpy>=1.24"]
    assert "scripts" not in payload["project"]
    assert "entry-points" not in payload["project"]


def test_public_api_is_closed_to_three_observations_and_one_input_error() -> None:
    assert diagnostics.__all__ == [
        "DiagnosticInputError",
        "canonical_observation_bytes",
        "rmt_observation",
        "spectral_observation",
        "variance_observation",
    ]
