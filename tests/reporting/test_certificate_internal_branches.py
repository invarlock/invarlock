from __future__ import annotations

import math

import pytest

from invarlock.reporting import certificate as cert


def test_normalize_and_validate_report_raises_on_invalid(monkeypatch):
    monkeypatch.setattr(cert, "validate_report", lambda _: False, raising=False)
    with pytest.raises(ValueError, match="Invalid RunReport structure"):
        cert._normalize_and_validate_report({"meta": {}})


def test_normalize_baseline_v1_schema():
    baseline = {
        "schema_version": "baseline-v1",
        "meta": {"model_id": "demo", "commit_sha": "abcdef1234567890"},
        "metrics": {"ppl_final": 42.0},
        "spectral_base": {"caps": 1},
        "rmt_base": {"stable": True},
        "invariants": {"status": "pass"},
    }
    normalized = cert._normalize_baseline(baseline)
    assert normalized["run_id"] == "abcdef1234567890"
    assert normalized["ppl_final"] == 42.0


def test_normalize_baseline_run_report_invalid_ppl(capfd):
    baseline = {
        "meta": {"model_id": "demo", "tokenizer_hash": "hash"},
        "data": {},
        "edit": {
            "name": "quant_rtn",
            "plan_digest": "quant_ffn",
            "deltas": {"params_changed": 5},
        },
        "metrics": {
            "ppl_final": 0.5,
            "spectral": {},
            "rmt": {},
            "invariants": {},
            "moe": {},
            "bootstrap": {},
            "window_overlap_fraction": 0.0,
            "window_match_fraction": 0.0,
        },
        "evaluation_windows": {
            "final": {"window_ids": [1], "logloss": [0.1]},
        },
    }
    normalized = cert._normalize_baseline(baseline)
    out = capfd.readouterr().out
    assert "Invalid baseline detected" in out
    assert math.isclose(normalized["ppl_final"], 50.797)


def test_normalize_baseline_dict_soft_fallback(capfd):
    normalized = cert._normalize_baseline({"ppl_final": 0.5})
    out = capfd.readouterr().out
    assert "Invalid baseline PPL" in out
    assert math.isclose(normalized["ppl_final"], 50.797)


def test_normalize_baseline_invalid_type():
    with pytest.raises(ValueError):
        cert._normalize_baseline("not a baseline")  # type: ignore[arg-type]


def test_extract_structural_deltas_infers_scope_and_details():
    report = {
        "edit": {
            "name": "quant_rtn",
            "plan_digest": "energy_0.3_ffn",
            "plan": {"scope": "unknown"},
            "deltas": {
                "params_changed": 10,
                "layers_modified": 2,
                "bitwidth_map": {
                    "ffn.0": {"bitwidth": 4, "group_size": None, "params": 256},
                    "ffn.1": {"bitwidth": 8, "group_size": 8, "params": 128},
                },
                "rank_map": {
                    "ffn.0": {
                        "rank": 8,
                        "params_saved": 64,
                        "energy_retained": 0.9,
                        "deploy_mode": "recompose",
                        "savings_mode": "realized",
                        "realized_params_saved": 32,
                        "theoretical_params_saved": 40,
                        "realized_params": 512,
                        "theoretical_params": 520,
                        "skipped": False,
                    }
                },
                "savings": {"deploy_mode": "recompose"},
            },
        }
    }
    structure = cert._extract_structural_deltas(report)
    diagnostics = structure["compression_diagnostics"]
    assert diagnostics["target_analysis"]["scope"] == "all"
    assert diagnostics["rank_summary"]["modules_modified"] == 1


def test_compute_validation_flags_hysteresis_and_ci(monkeypatch):
    monkeypatch.delenv("INVARLOCK_TINY_RELAX", raising=False)
    flags = cert._compute_validation_flags(
        ppl={
            "preview_final_ratio": 1.0,
            "ratio_vs_baseline": 1.102,
            "ratio_ci": (1.02, 1.101),
        },
        spectral={"caps_applied": 1},
        rmt={"stable": True},
        invariants={"status": "pass"},
        tier="balanced",
        _ppl_metrics={"preview_total_tokens": 40000, "final_total_tokens": 40000},
        dataset_capacity={"tokens_available": 100000},
        target_ratio=None,
        guard_overhead={"butterfly": True},  # triggers soft-pass path
        primary_metric={"kind": "ppl_causal", "ratio_vs_baseline": 1.102},
    )
    assert flags["primary_metric_acceptable"] is True
    assert flags.get("hysteresis_applied")


def test_compute_validation_flags_tiny_relax_env(monkeypatch):
    monkeypatch.setenv("INVARLOCK_TINY_RELAX", "1")
    flags = cert._compute_validation_flags(
        ppl={"preview_final_ratio": 1.2, "ratio_vs_baseline": 1.5},
        spectral={"caps_applied": 10},
        rmt={"stable": False},
        invariants={"status": "fail"},
        tier="balanced",
        guard_overhead={"passed": False, "evaluated": False},
        primary_metric={"kind": "ppl_causal", "ratio_vs_baseline": float("nan")},
    )
    assert flags["preview_final_drift_acceptable"] is True
    assert flags["guard_overhead_acceptable"] is True


def test_prepare_guard_overhead_section_triggers_validation():
    bare = {"metrics": {"primary_metric": {"final": 10.0}}}
    guarded = {"metrics": {"primary_metric": {"final": 12.0}}}
    payload, passed = cert._prepare_guard_overhead_section(
        {"bare_report": bare, "guarded_report": guarded, "overhead_threshold": 0.01}
    )
    assert passed is False
    assert "errors" in payload


def test_compute_quality_overhead_from_guard_none_on_missing_data():
    assert (
        cert._compute_quality_overhead_from_guard(
            {"bare_report": {}, "guarded_report": {}}
        )
        is None
    )
