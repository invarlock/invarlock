from __future__ import annotations

import json
from pathlib import Path

import pytest
import typer

from invarlock.cli.commands import verify as verify_mod
from invarlock.cli.commands.verify import verify_command


def _w(p: Path, payload: dict) -> Path:
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


def _cert_min() -> dict:
    return {
        "schema_version": "v1",
        "run_id": "r",
        "artifacts": {"generated_at": "t"},
        "plugins": {},
        "meta": {},
        "dataset": {
            "provider": "p",
            "seq_len": 8,
            "windows": {
                "preview": 2,
                "final": 2,
                "stats": {
                    "window_match_fraction": 1.0,
                    "window_overlap_fraction": 0.0,
                    "coverage": {"preview": {"used": 2}, "final": {"used": 2}},
                    "paired_windows": 2,
                },
            },
        },
        "primary_metric": {
            "kind": "ppl_causal",
            "final": 10.0,
            "ratio_vs_baseline": 1.0,
            "display_ci": [1.0, 1.0],
            "analysis_point_final": 2.302585093,
        },
        "baseline_ref": {"primary_metric": {"kind": "ppl_causal", "final": 10.0}},
        "validation": {
            "primary_metric_acceptable": True,
            "preview_final_drift_acceptable": True,
            "invariants_pass": True,
            "spectral_stable": True,
            "rmt_stable": True,
        },
    }


def test_verify_exit_code_invarlockerror_ci_profile_parity(tmp_path: Path) -> None:
    c = _cert_min()
    p = _w(tmp_path / "c.json", c)
    # No provider_digest in subject; in CI this yields InvarlockError → exit code 3
    with pytest.raises(typer.Exit) as ei:
        verify_command([p], baseline=None, profile="ci", json_out=True)
    assert getattr(ei.value, "exit_code", getattr(ei.value, "code", None)) == 3


def test_verify_counts_and_drift_errors(tmp_path: Path, capsys) -> None:
    c = _cert_min()
    # Make coverage counts mismatched and drift out-of-band
    c["dataset"]["windows"]["stats"]["coverage"]["final"]["used"] = 1
    c["primary_metric"]["preview"] = 5.0
    c["primary_metric"]["final"] = 6.0  # drift 1.2 → out-of-band
    p = _w(tmp_path / "c.json", c)
    with pytest.raises(typer.Exit) as ei:
        verify_command([p], baseline=None, profile="dev", json_out=True)
    out = json.loads(capsys.readouterr().out)
    assert out["resolution"]["exit_code"] == 1
    assert getattr(ei.value, "exit_code", getattr(ei.value, "code", None)) == 1


def test_verify_drift_band_override_allows_wider_drift(tmp_path: Path, capsys) -> None:
    c = _cert_min()
    c["primary_metric"]["preview"] = 5.0
    c["primary_metric"]["final"] = 6.0  # drift 1.2
    c["primary_metric"]["ratio_vs_baseline"] = 0.6
    c["primary_metric"]["drift_band"] = {"min": 0.9, "max": 1.3}
    p = _w(tmp_path / "c.json", c)
    with pytest.raises(typer.Exit) as ei:
        verify_command([p], baseline=None, profile="dev", json_out=True)
    out = json.loads(capsys.readouterr().out)
    assert out["resolution"]["exit_code"] == 0
    assert getattr(ei.value, "exit_code", getattr(ei.value, "code", None)) == 0


def test_verify_json_results_handle_load_error_in_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    cert = _cert_min()
    # Make certificate fail policy (ratio mismatch) so summary path is exercised
    cert["primary_metric"]["ratio_vs_baseline"] = 2.0
    cert_path = _w(tmp_path / "c.json", cert)

    orig_load = verify_mod._load_certificate
    call_count = {"n": 0}

    def _wrapped_load(path: Path) -> dict:
        call_count["n"] += 1
        if call_count["n"] == 1:
            return orig_load(path)
        raise json.JSONDecodeError("boom", "{}", 0)

    monkeypatch.setattr(verify_mod, "_load_certificate", _wrapped_load, raising=True)

    with pytest.raises(typer.Exit) as ei:
        verify_command([cert_path], baseline=None, profile="dev", json_out=True)

    out = json.loads(capsys.readouterr().out)
    assert out["resolution"]["exit_code"] == 1
    # Even with load error in summary, we should still emit a result record
    assert isinstance(out.get("results"), list) and out["results"]
    assert getattr(ei.value, "exit_code", getattr(ei.value, "code", None)) == 1
