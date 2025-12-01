import json
from pathlib import Path

import pytest
from click.exceptions import Exit as ClickExit

from invarlock.cli.commands import verify as v


def _cert_with_provenance() -> dict:
    base = {
        "meta": {"model_id": "m", "adapter": "hf", "seed": 1, "device": "cpu"},
        "primary_metric": {
            "kind": "ppl_causal",
            "preview": 100.0,
            "final": 101.0,
            "ratio_vs_baseline": 1.01,
        },
        "dataset": {
            "windows": {
                "preview": 1,
                "final": 1,
                "stats": {
                    "window_match_fraction": 1.0,
                    "window_overlap_fraction": 0.0,
                    "coverage": {"preview": {"used": 1}, "final": {"used": 1}},
                    "paired_windows": 1,
                },
            }
        },
        "baseline_ref": {"primary_metric": {"kind": "ppl_causal", "final": 100.0}},
        "provenance": {
            "provider_digest": {
                "ids_sha256": "abc",
                "tokenizer_sha256": "tok",
                "masking_sha256": "mask",
                "masking": {"is_masked": False},
            }
        },
    }
    return base


def test_verify_command_happy_ci_profile(tmp_path: Path, monkeypatch):
    # Monkeypatch schema validator
    monkeypatch.setattr(v, "validate_certificate", lambda c: True)
    cert = _cert_with_provenance()
    cert_path = tmp_path / "c.json"
    cert_path.write_text(json.dumps(cert))

    baseline = {
        "provenance": {
            "provider_digest": {
                "ids_sha256": "abc",
                "tokenizer_sha256": "tok",
                "masking_sha256": "mask",
                "masking": {"is_masked": False},
            }
        }
    }
    baseline_path = tmp_path / "b.json"
    baseline_path.write_text(json.dumps(baseline))

    # Expect Typer Exit (JSON emit) with code 0
    with pytest.raises(ClickExit) as ei:
        v.verify_command(
            [cert_path], baseline=baseline_path, profile="ci", json_out=True
        )
    assert getattr(ei.value, "exit_code", None) == 0


def test_verify_command_missing_digest_raises_in_ci(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(v, "validate_certificate", lambda c: True)
    cert = _cert_with_provenance()
    # remove provider digest
    cert.pop("provenance", None)
    cert_path = tmp_path / "c.json"
    cert_path.write_text(json.dumps(cert))

    with pytest.raises(ClickExit) as ei:
        v.verify_command([cert_path], baseline=None, profile="ci", json_out=True)
    assert getattr(ei.value, "exit_code", None) not in (None, 0)
