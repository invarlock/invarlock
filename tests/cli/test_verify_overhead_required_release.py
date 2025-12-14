from __future__ import annotations

import json
from pathlib import Path

import pytest
import typer

from invarlock.cli.commands.verify import verify_command


def _write_cert(tmp_path: Path, payload: dict, name: str) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _release_ready_cert(*, include_guard_overhead: bool) -> dict:
    cert: dict = {
        "schema_version": "v1",
        "run_id": "run-xyz",
        "artifacts": {"generated_at": "2024-01-01T00:00:00"},
        "plugins": {},
        "meta": {},
        "dataset": {
            "provider": "unit",
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
            "preview": 10.0,
            "final": 10.0,
            "ratio_vs_baseline": 1.0,
            "display_ci": [1.0, 1.0],
        },
        "baseline_ref": {
            "run_id": "base-xyz",
            "model_id": "m",
            "primary_metric": {"kind": "ppl_causal", "final": 10.0},
        },
        "validation": {
            "primary_metric_acceptable": True,
            "preview_final_drift_acceptable": True,
            "invariants_pass": True,
            "spectral_stable": True,
            "rmt_stable": True,
        },
        "provenance": {"provider_digest": {"ids_sha256": "deadbeef"}},
    }

    if include_guard_overhead:
        cert["guard_overhead"] = {
            "skipped": True,
            "mode": "skipped",
            "source": "env:INVARLOCK_SKIP_OVERHEAD_CHECK",
            "overhead_threshold": 0.01,
        }

    return cert


def test_verify_release_fails_when_guard_overhead_missing(
    tmp_path: Path, capsys
) -> None:
    cert = _release_ready_cert(include_guard_overhead=False)
    path = _write_cert(tmp_path, cert, "missing_overhead.json")

    with pytest.raises(typer.Exit) as ei:
        verify_command([path], baseline=None, profile="release", json_out=True)

    out = json.loads(capsys.readouterr().out)
    assert out["resolution"]["exit_code"] != 0
    assert getattr(ei.value, "exit_code", getattr(ei.value, "code", None)) != 0


def test_verify_release_allows_explicit_overhead_skip_marker(
    tmp_path: Path, capsys
) -> None:
    cert = _release_ready_cert(include_guard_overhead=True)
    path = _write_cert(tmp_path, cert, "skipped_overhead.json")

    with pytest.raises(typer.Exit) as ei:
        verify_command([path], baseline=None, profile="release", json_out=True)

    out = json.loads(capsys.readouterr().out)
    assert out["resolution"]["exit_code"] == 0
    assert getattr(ei.value, "exit_code", getattr(ei.value, "code", None)) == 0


def test_verify_release_requires_overhead_evaluated_when_not_skipped(
    tmp_path: Path, capsys
) -> None:
    cert = _release_ready_cert(include_guard_overhead=False)
    cert["guard_overhead"] = {
        "mode": "measured",
        "overhead_threshold": 0.01,
        "evaluated": False,
    }
    path = _write_cert(tmp_path, cert, "not_evaluated.json")

    with pytest.raises(typer.Exit) as ei:
        verify_command([path], baseline=None, profile="release", json_out=True)

    out = json.loads(capsys.readouterr().out)
    assert out["resolution"]["exit_code"] != 0
    assert getattr(ei.value, "exit_code", getattr(ei.value, "code", None)) != 0


def test_verify_release_requires_overhead_ratio_when_evaluated(
    tmp_path: Path, capsys
) -> None:
    cert = _release_ready_cert(include_guard_overhead=False)
    cert["guard_overhead"] = {
        "mode": "measured",
        "overhead_threshold": 0.01,
        "evaluated": True,
    }
    path = _write_cert(tmp_path, cert, "missing_ratio.json")

    with pytest.raises(typer.Exit) as ei:
        verify_command([path], baseline=None, profile="release", json_out=True)

    out = json.loads(capsys.readouterr().out)
    assert out["resolution"]["exit_code"] != 0
    assert getattr(ei.value, "exit_code", getattr(ei.value, "code", None)) != 0
