from __future__ import annotations

import json
from pathlib import Path

import pytest
import typer

from invarlock.cli.commands import verify as verify_mod
from invarlock.cli.commands.verify import verify_command


def _write(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_verify_ppl_baseline_zero_is_error(tmp_path: Path, capsys) -> None:
    cert = {
        "schema_version": "v1",
        "run_id": "r",
        "artifacts": {"generated_at": "t"},
        "plugins": {},
        "meta": {},
        "dataset": {
            "provider": "p",
            "seq_len": 8,
            "windows": {
                "preview": 1,
                "final": 1,
                "stats": {
                    "window_match_fraction": 1.0,
                    "window_overlap_fraction": 0.0,
                    "coverage": {"preview": {"used": 1}, "final": {"used": 1}},
                    "paired_windows": 1,
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
        "baseline_ref": {"primary_metric": {"kind": "ppl_causal", "final": 0.0}},
        "validation": {
            "primary_metric_acceptable": True,
            "preview_final_drift_acceptable": True,
            "invariants_pass": True,
            "spectral_stable": True,
            "rmt_stable": True,
        },
    }
    p = _write(tmp_path / "c.json", cert)
    with pytest.raises(typer.Exit):
        verify_command([p], baseline=None, profile="dev", json_out=True)
    out = json.loads(capsys.readouterr().out)
    assert out["resolution"]["exit_code"] != 0


def test_adapter_family_warning_swallows_internal_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    cert = {
        "schema_version": "v1",
        "run_id": "r-ok",
        "artifacts": {"generated_at": "t"},
        "plugins": {
            "adapter": {
                "name": "hf",
                "module": "invarlock.adapters.hf_causal",
                "version": "0",
                "provenance": {
                    "family": "hf",
                    "library": "transformers",
                    "version": "1",
                },
            }
        },
        "meta": {},
        "dataset": {
            "provider": "p",
            "seq_len": 8,
            "windows": {
                "preview": 1,
                "final": 1,
                "stats": {
                    "window_match_fraction": 1.0,
                    "window_overlap_fraction": 0.0,
                    "coverage": {"preview": {"used": 1}, "final": {"used": 1}},
                    "paired_windows": 1,
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
        "baseline_ref": {"primary_metric": {"kind": "ppl_causal", "final": 10.0}},
        "validation": {
            "primary_metric_acceptable": True,
            "preview_final_drift_acceptable": True,
            "invariants_pass": True,
            "spectral_stable": True,
            "rmt_stable": True,
        },
    }
    p = _write(tmp_path / "ok.json", cert)

    def _boom(*args, **kwargs):
        raise RuntimeError("adapter warning failed")

    monkeypatch.setattr(
        verify_mod, "_warn_adapter_family_mismatch", _boom, raising=True
    )

    # Human mode success: internal warning failure should be swallowed
    verify_command([p], baseline=None, profile="dev", json_out=False)
    out = capsys.readouterr().out
    assert "PASS" in out
