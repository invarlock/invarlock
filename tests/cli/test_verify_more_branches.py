from __future__ import annotations

import json
from pathlib import Path

import pytest
import typer

from invarlock.cli.commands.verify import verify_command


def _write_cert(tmp: Path, payload: dict, name: str) -> Path:
    p = tmp / name
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


def _minimal_cert() -> dict:
    return {
        "schema_version": "v1",
        "run_id": "r",
        "artifacts": {"generated_at": "t"},
        "plugins": {},
        "meta": {},
        "dataset": {
            "provider": "p",
            "seq_len": 8,
            "windows": {"preview": 0, "final": 0},
        },
        "primary_metric": {
            "kind": "ppl_causal",
            "final": 10.0,
            "ratio_vs_baseline": 1.0,
            "display_ci": [1.0, 1.0],
        },
        "validation": {
            "primary_metric_acceptable": True,
            "preview_final_drift_acceptable": True,
            "invariants_pass": True,
            "spectral_stable": True,
            "rmt_stable": True,
        },
        "baseline_ref": {"primary_metric": {"kind": "ppl_causal", "final": 10.0}},
    }


def test_verify_basis_mismatch_analysis_point(tmp_path: Path, capsys) -> None:
    c = _minimal_cert()
    # Provide analysis_point_final inconsistent with recompute from evaluation_windows.final
    c["primary_metric"]["analysis_point_final"] = 2.0  # mean logloss (basis)
    c["evaluation_windows"] = {"final": {"logloss": [3.0, 3.0], "token_counts": [1, 1]}}
    p = _write_cert(tmp_path, c, "c.json")
    with pytest.raises(typer.Exit):
        verify_command([p], baseline=None, profile="dev", json_out=True)
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["resolution"]["exit_code"] != 0


def test_verify_tokenizer_hash_mismatch_payload(tmp_path: Path, capsys) -> None:
    c = _minimal_cert()
    c.setdefault("meta", {})["tokenizer_hash"] = "TOK-X"
    c.setdefault("baseline_ref", {})["tokenizer_hash"] = "TOK-Y"
    p = _write_cert(tmp_path, c, "c.json")
    with pytest.raises(typer.Exit):
        verify_command([p], baseline=None, profile="dev", json_out=True)
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["resolution"]["exit_code"] != 0


def test_verify_parity_mask_mismatch_ci_profile(tmp_path: Path, capsys) -> None:
    # Subject with provider_digest matching tokenizer but differing masking
    c = _minimal_cert()
    c.setdefault("provenance", {})["provider_digest"] = {
        "ids_sha256": "ID",
        "tokenizer_sha256": "TOK",
        "masking_sha256": "MASK-A",
    }
    p_c = _write_cert(tmp_path, c, "subject.json")
    baseline = {
        "provenance": {
            "provider_digest": {
                "ids_sha256": "ID",
                "tokenizer_sha256": "TOK",
                "masking_sha256": "MASK-B",
            }
        }
    }
    p_b = _write_cert(tmp_path, baseline, "baseline.json")
    with pytest.raises(typer.Exit):
        verify_command([p_c], baseline=p_b, profile="ci", json_out=True)
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["resolution"]["exit_code"] != 0


def test_verify_malformed_schema_failure_code(tmp_path: Path, capsys) -> None:
    # Missing primary_metric block makes schema invalid
    bad = _minimal_cert()
    bad.pop("primary_metric", None)
    p = _write_cert(tmp_path, bad, "bad.json")
    with pytest.raises(typer.Exit):
        verify_command([p], baseline=None, profile="dev", json_out=True)
    out = capsys.readouterr().out
    payload = json.loads(out)
    # code should be non-zero; labeled malformed
    assert payload["resolution"]["exit_code"] != 0
    assert payload["summary"]["reason"] == "malformed"


def test_verify_adapter_family_warning_human(tmp_path: Path, capsys) -> None:
    # PASS case with adapter family mismatch should emit a warning in human mode
    c = _minimal_cert()
    c.setdefault("plugins", {})["adapter"] = {
        "name": "ad",
        "version": "0",
        "module": "m",
        "provenance": {"family": "hf", "library": "transformers", "version": "1.0"},
    }
    # Baseline report referenced by provenance
    base_report = {
        "meta": {
            "plugins": {
                "adapter": {
                    "provenance": {
                        "family": "ggml",
                        "library": "ggml-backend",
                        "version": "0.0",
                    }
                }
            }
        }
    }
    p_base_report = _write_cert(tmp_path, base_report, "base_report.json")
    c.setdefault("provenance", {})["baseline"] = {"report_path": str(p_base_report)}
    # Ensure PASS by providing pairing stats, coverage, and drift-ok preview/final
    c.setdefault("dataset", {}).setdefault("windows", {})
    c["dataset"]["windows"]["preview"] = 2
    c["dataset"]["windows"]["final"] = 2
    c["dataset"]["windows"]["stats"] = {
        "window_match_fraction": 1.0,
        "window_overlap_fraction": 0.0,
        "coverage": {"preview": {"used": 2}, "final": {"used": 2}},
        "paired_windows": 2,
    }
    c["primary_metric"]["preview"] = 10.0
    p = _write_cert(tmp_path, c, "c.json")
    # Human mode success should print warning lines
    verify_command([p], baseline=None, profile="dev", json_out=False)
    out = capsys.readouterr().out
    assert "Adapter family differs" in out
