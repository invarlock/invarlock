"""Tests for the `invarlock verify` CLI command."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

from typer.testing import CliRunner

os.environ["INVARLOCK_LIGHT_IMPORT"] = "1"
from invarlock.cli.app import app
from invarlock.cli.commands import verify as verify_mod

runner = CliRunner()


def _build_sample_certificate() -> dict:
    """Create a minimal v1 payload suitable for verification tests (PM-only)."""
    baseline_final = 46.5
    ppl_final = 47.3
    preview_ppl = 46.9
    ratio_vs_baseline = ppl_final / baseline_final
    preview_final_ratio = ppl_final / preview_ppl
    logloss_delta = math.log(preview_final_ratio)
    logloss_delta_ci = (-0.02, 0.03)
    ratio_ci = [math.exp(bound) for bound in logloss_delta_ci]

    dataset_block = {
        "provider": "wikitext2",
        "split": "validation",
        "seq_len": 768,
        "stride": 768,
        "windows": {"preview": 200, "final": 200, "seed": 42},
        "hash": {
            "preview": "sha256:previewhash",
            "final": "sha256:finalhash",
            "dataset": "sha256:datasethash",
            "total_tokens": 307200,
            "preview_tokens": 153600,
            "final_tokens": 153600,
        },
        "tokenizer": {
            "name": "gpt2",
            "hash": "tokhash123",
            "vocab_size": 50257,
            "bos_token": "Ġ",
            "eos_token": "",
            "pad_token": "",
            "add_prefix_space": False,
        },
    }

    cert = {
        "schema_version": "v1",
        "run_id": "run-001",
        "meta": {
            "model_id": "gpt2-small",
            "adapter": "hf_gpt2",
            "device": "cpu",
            "ts": "2025-01-01T00:00:00",
            "commit": "abcdef1234567890",
            "seed": 42,
            "seeds": {"python": 42, "numpy": 42, "torch": 42},
            "cuda_flags": {"deterministic_algorithms": True},
        },
        "auto": {
            "tier": "balanced",
            "probes_used": 0,
            "target_pm_ratio": None,
            "policy_digest": "deadbeefcafe1234",
        },
        "dataset": dataset_block,
        "baseline_ref": {
            "run_id": "baseline-001",
            "model_id": "gpt2-small",
            "primary_metric": {"final": baseline_final},
        },
        "invariants": {"status": "pass"},
        "spectral": {
            "caps_applied": 0,
            "summary": {
                "status": "stable",
                "sigma_quantile": 0.95,
            },
        },
        "rmt": {
            "epsilon": 0.1,
            "epsilon_by_family": {"ffn": 0.1, "attn": 0.1, "embed": 0.1, "other": 0.1},
            "status": "stable",
        },
        "variance": {
            "enabled": False,
            "gain": None,
            "tap": "transformer.h.*.mlp.c_proj",
            "target_modules": ["transformer.h.4.mlp.c_proj"],
            "predictive_gate": {
                "evaluated": True,
                "reason": "ci_gain_met",
                "delta_ci": [-0.001, 0.0005],
                "mean_delta": -0.00025,
            },
            "ab_test": {
                "seed": 1337,
                "windows_used": 16,
                "provenance": {"baseline": "baseline-001"},
                "point_estimates": {"no_ve": 53.2, "with_ve": 53.1},
            },
        },
        "structure": {"params_changed": 123, "layers_modified": 3},
        "policies": {
            "spectral": {
                "sigma_quantile": 0.95,
                "deadband": 0.1,
            },
            "rmt": {
                "deadband": 0.1,
                "margin": 1.5,
                "detection_threshold": 1.65,
                "q_method": "auto",
                "epsilon_default": 0.1,
                "epsilon_by_family": {
                    "ffn": 0.1,
                    "attn": 0.1,
                    "embed": 0.1,
                    "other": 0.1,
                },
            },
            "variance": {
                "deadband": 0.02,
                "min_abs_adjust": 0.012,
                "max_scale_step": 0.03,
                "policy_digest": "deadbeefcafe1234",
                "target_modules": ["transformer.h.4.mlp.c_proj"],
                "calibration": {"windows": 24, "seed": 1337},
            },
        },
        "artifacts": {
            "events_path": "/tmp/events.jsonl",
            "report_path": "/tmp/report.json",
            "generated_at": "2025-01-01T00:00:00",
        },
        "plugins": {
            "adapter": {
                "name": "hf_gpt2",
                "module": "invarlock.adapters.hf_gpt2",
                "version": "1.0.0",
                "entry_point": "hf_gpt2",
                "entry_point_group": "invarlock.adapters",
                "available": True,
            },
            "edit": {
                "name": "quant_rtn",
                "module": "invarlock.edits.quant_rtn",
                "version": "1.0.0",
                "entry_point": "quant_rtn",
                "entry_point_group": "invarlock.edits",
                "available": True,
            },
            "guards": [
                {
                    "name": "spectral",
                    "module": "invarlock.guards.spectral",
                    "version": "1.0.0",
                    "entry_point": "spectral",
                    "entry_point_group": "invarlock.guards",
                    "available": True,
                }
            ],
        },
        "validation": {
            "preview_final_drift_acceptable": True,
            "primary_metric_acceptable": True,
            "invariants_pass": True,
            "spectral_stable": True,
            "rmt_stable": True,
            "guard_overhead_acceptable": True,
        },
        "guard_overhead": {},
        "edit_name": "quant_rtn",
    }
    # PM-only: pairing and coverage live under dataset.windows.stats
    cert.setdefault("dataset", {}).setdefault("windows", {}).setdefault(
        "stats", {}
    ).update(
        {
            "metric_space": "log_nll",
            "pairing": "paired_baseline",
            "paired_windows": 200,
            "window_match_fraction": 1.0,
            "window_overlap_fraction": 0.0,
            "paired_delta_summary": {
                "mean": logloss_delta,
                "std": 0.01,
                "degenerate": False,
            },
            "coverage": {
                "tier": "balanced",
                "preview": {"used": 200, "required": 200, "ok": True},
                "final": {"used": 200, "required": 200, "ok": True},
                "replicates": {"used": 1200, "required": 1200, "ok": True},
            },
        }
    )
    # Attach canonical primary metric
    cert["primary_metric"] = {
        "kind": "ppl_causal",
        "preview": preview_ppl,
        "final": ppl_final,
        "ratio_vs_baseline": ratio_vs_baseline,
        "display_ci": ratio_ci,
    }
    return cert


def _write_certificate(tmp_path: Path) -> Path:
    cert_path = tmp_path / "cert.json"
    cert_path.write_text(json.dumps(_build_sample_certificate()))
    return cert_path


def test_verify_uses_dataset_windows_stats(tmp_path: Path):
    # Cert with pairing/coverage under dataset.windows.stats should verify without using legacy ppl stats
    cert = _build_sample_certificate()
    # ensure present
    assert isinstance(cert.get("dataset", {}).get("windows", {}).get("stats", {}), dict)
    cert_path = tmp_path / "cert_stats.json"
    cert_path.write_text(json.dumps(cert))
    result = runner.invoke(app, ["report", "verify", str(cert_path)])
    assert result.exit_code == 0


def test_verify_delta_basis_matches_pm_direction(tmp_path: Path):
    # Lower-is-better (ppl_*) uses ratio; higher-is-better (accuracy) uses Δpp
    cert = _build_sample_certificate()
    # Swap primary metric kind to accuracy and change basis
    cert["primary_metric"] = {
        "kind": "accuracy",
        "preview": 0.91,
        "final": 0.93,
        "ratio_vs_baseline": +2.0,  # nonsensical for accuracy; verify focuses on presence
        "display_ci": [0.92, 0.94],
    }
    # Verify should still process without error (basis rendered in markdown; functional check is ratio presence for non-ppl kinds)
    cert_path = tmp_path / "cert_accuracy.json"
    cert_path.write_text(json.dumps(cert))
    result = runner.invoke(app, ["report", "verify", str(cert_path)])
    assert result.exit_code == 0


def test_verify_command_passes(tmp_path: Path):
    cert_path = _write_certificate(tmp_path)
    result = runner.invoke(app, ["report", "verify", str(cert_path)])
    assert result.exit_code == 0
    # report.verify emits JSON payload; verify success should have ok=true
    payload = json.loads(result.stdout)
    assert payload.get("format_version") == "verify-v1"
    assert payload.get("summary", {}).get("ok") is True


def test_verify_command_detects_ratio_mismatch(tmp_path: Path):
    cert_path = _write_certificate(tmp_path)
    certificate = json.loads(cert_path.read_text())
    certificate["primary_metric"]["ratio_vs_baseline"] = 2.0  # impossible ratio
    cert_path.write_text(json.dumps(certificate))

    result = runner.invoke(app, ["report", "verify", str(cert_path)])
    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload.get("format_version") == "verify-v1"
    assert payload.get("summary", {}).get("ok") is False


def test_verify_command_detects_pairing_failure(tmp_path: Path):
    cert_path = _write_certificate(tmp_path)
    certificate = json.loads(cert_path.read_text())
    certificate["dataset"]["windows"]["stats"]["window_match_fraction"] = 0.95
    cert_path.write_text(json.dumps(certificate))

    result = runner.invoke(app, ["report", "verify", str(cert_path)])
    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload.get("format_version") == "verify-v1"
    assert payload.get("summary", {}).get("ok") is False


def test_verify_command_detects_count_mismatch(tmp_path: Path):
    cert_path = _write_certificate(tmp_path)
    certificate = json.loads(cert_path.read_text())
    certificate["dataset"]["windows"]["stats"]["coverage"]["preview"]["used"] = 180
    cert_path.write_text(json.dumps(certificate))

    result = runner.invoke(app, ["report", "verify", str(cert_path)])
    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload.get("format_version") == "verify-v1"
    assert payload.get("summary", {}).get("ok") is False


def test_verify_command_detects_drift_band_violation(tmp_path: Path):
    cert_path = _write_certificate(tmp_path)
    certificate = json.loads(cert_path.read_text())
    # Raise drift by manipulating PM values
    certificate["primary_metric"]["preview"] = 10.0
    certificate["primary_metric"]["final"] = 12.0
    cert_path.write_text(json.dumps(certificate))

    result = runner.invoke(app, ["report", "verify", str(cert_path)])
    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload.get("format_version") == "verify-v1"
    assert payload.get("summary", {}).get("ok") is False


def test_verify_family_mismatch_warning_includes_backends(tmp_path: Path):
    # Create a baseline report with GPTQ provenance
    baseline_report = {
        "meta": {
            "plugins": {
                "adapter": {
                    "provenance": {
                        "family": "gptq",
                        "library": "auto-gptq",
                        "version": "0.7.0",
                    }
                }
            }
        }
    }
    baseline_path = tmp_path / "baseline_report.json"
    baseline_path.write_text(json.dumps(baseline_report))

    # Build a certificate that references baseline and has AWQ in edited provenance
    cert = _build_sample_certificate()
    cert.setdefault("plugins", {}).setdefault("adapter", {}).setdefault(
        "provenance", {}
    ).update({"family": "awq", "library": "autoawq", "version": "0.2.0"})
    cert["provenance"] = {"baseline": {"report_path": str(baseline_path)}}

    cert_path = tmp_path / "cert_family_mismatch.json"
    cert_path.write_text(json.dumps(cert))

    result = runner.invoke(app, ["verify", str(cert_path)])
    assert result.exit_code == 0
    out = result.stdout
    assert "Adapter family differs" in out
    assert "baseline: family=gptq, backend=auto-gptq ==0.7.0" in out
    assert "edited  : family=awq, backend=autoawq ==0.2.0" in out


def test_validate_pairing_overlap_violation_direct() -> None:
    cert = _build_sample_certificate()
    windows = cert.setdefault("dataset", {}).setdefault("windows", {})
    stats = windows.setdefault("stats", {})
    stats["window_overlap_fraction"] = 0.5
    errors = verify_mod._validate_pairing(cert)
    assert any("window_overlap_fraction must be 0.0" in e for e in errors)


def test_validate_counts_final_mismatch_direct() -> None:
    cert = _build_sample_certificate()
    windows = cert.setdefault("dataset", {}).setdefault("windows", {})
    stats = windows.setdefault("stats", {})
    cov = stats.setdefault("coverage", {})
    cov.setdefault("preview", {})["used"] = windows.get("preview", 0)
    cov.setdefault("final", {})["used"] = 0
    errors = verify_mod._validate_counts(cert)
    assert any("Final window count mismatch" in e for e in errors)
