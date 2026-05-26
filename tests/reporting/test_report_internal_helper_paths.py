from __future__ import annotations

import pytest

from invarlock.reporting import report_edit_summary as report_edit_summary_mod
from invarlock.reporting import report_normalization as report_normalization_mod
from invarlock.reporting import report_overhead as report_overhead_mod
from invarlock.reporting import report_validation as report_validation_mod


def test_normalize_and_validate_report_raises_on_invalid(monkeypatch):
    del monkeypatch
    with pytest.raises(ValueError, match="Invalid RunReport structure"):
        report_normalization_mod.normalize_and_validate_run_report({"meta": {}})


def test_normalize_baseline_v1_schema():
    baseline = {
        "schema_version": "baseline-v1",
        "meta": {"model_id": "demo", "commit_sha": "abcdef1234567890"},
        "metrics": {"ppl_final": 42.0},
        "spectral_base": {"caps": 1},
        "rmt_base": {"stable": True},
        "invariants": {"status": "pass"},
    }
    normalized = report_normalization_mod.normalize_baseline(baseline)
    assert normalized["run_id"] == "abcdef1234567890"
    assert normalized["ppl_final"] == 42.0


def test_normalize_baseline_run_report_invalid_ppl():
    baseline = {
        "meta": {"model_id": "demo", "tokenizer_hash": "hash"},
        "data": {},
        "edit": {
            "name": "quant_rtn",
            "plan_digest": "quant_ffn",
            "deltas": {"params_changed": 5},
        },
        "metrics": {
            "ppl_final": 0.0,
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
    with pytest.raises(ValueError, match="Invalid baseline"):
        report_normalization_mod.normalize_baseline(baseline)


def test_normalize_baseline_dict_soft_fallback():
    with pytest.raises(ValueError, match="Invalid baseline"):
        report_normalization_mod.normalize_baseline({"ppl_final": 0.0})


def test_normalize_baseline_invalid_type():
    with pytest.raises(
        ValueError, match="Baseline must be a RunReport dict or canonical baseline dict"
    ):
        report_normalization_mod.normalize_baseline("not a baseline")


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
    structure = report_edit_summary_mod.extract_structural_deltas(report)
    diagnostics = structure["compression_diagnostics"]
    assert diagnostics["target_analysis"]["scope"] == "all"
    assert diagnostics["rank_summary"]["modules_modified"] == 1


def test_compute_validation_flags_hysteresis_and_ci() -> None:
    flags = report_validation_mod.compute_validation_flags(
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


def test_metric_specific_primary_metric_gate_handles_none_and_other_kinds() -> None:
    flags = {"primary_metric_acceptable": True}
    report_validation_mod._apply_metric_specific_primary_metric_gate(
        flags,
        primary_metric={"kind": None},
        metrics_policy={},
        ratio_limit_with_hyst=1.1,
        tokens_ok_eff=True,
        compression_acceptable=True,
        tiny_relax=False,
        dataset_capacity=None,
    )
    assert flags["primary_metric_acceptable"] is False

    flags = {"primary_metric_acceptable": True}
    report_validation_mod._apply_metric_specific_primary_metric_gate(
        flags,
        primary_metric={"kind": "bleu"},
        metrics_policy={},
        ratio_limit_with_hyst=1.1,
        tokens_ok_eff=True,
        compression_acceptable=True,
        tiny_relax=False,
        dataset_capacity=None,
    )
    assert flags["primary_metric_acceptable"] is False

    flags = {"primary_metric_acceptable": True}
    report_validation_mod._apply_ppl_primary_metric_reconcile(
        flags,
        primary_metric={"kind": None},
        ratio_limit=1.1,
        hysteresis_ratio=0.0,
        tokens_ok_eff=True,
    )
    assert flags["primary_metric_acceptable"] is False


def test_compute_validation_flags_lower_bound_exception_defaults_open(
    monkeypatch,
) -> None:
    class BadFloat(float):
        def __float__(self):
            raise ValueError("bad float")

    original_predicate = report_validation_mod._is_non_bool_finite_number

    def predicate(value):  # noqa: ANN001
        if isinstance(value, BadFloat):
            return True
        return original_predicate(value)

    monkeypatch.setattr(report_validation_mod, "_is_non_bool_finite_number", predicate)

    flags = report_validation_mod.compute_validation_flags(
        ppl={"preview_final_ratio": 1.0, "ratio_vs_baseline": BadFloat(1.0)},
        spectral={"caps_applied": 0},
        rmt={"stable": True},
        invariants={"status": "pass"},
        pm_acceptance_range={"min": 0.9},
        get_tier_policies_fn=lambda: {"balanced": {"metrics": {"pm_ratio": {}}}},
    )

    assert flags["primary_metric_acceptable"] is True


def test_compute_validation_flags_tiny_relax_mode() -> None:
    flags = report_validation_mod.compute_validation_flags(
        ppl={"preview_final_ratio": 1.2, "ratio_vs_baseline": 1.5},
        spectral={"caps_applied": 10},
        rmt={"stable": False},
        invariants={"status": "fail"},
        tier="balanced",
        guard_overhead={"passed": False, "evaluated": False},
        primary_metric={"kind": "ppl_causal", "ratio_vs_baseline": float("nan")},
        tiny_relax=True,
    )
    assert flags["preview_final_drift_acceptable"] is True
    assert flags["guard_overhead_acceptable"] is True


def test_prepare_guard_overhead_section_triggers_validation():
    bare = {"metrics": {"primary_metric": {"final": 10.0}}}
    guarded = {"metrics": {"primary_metric": {"final": 12.0}}}
    payload, passed = report_overhead_mod.prepare_guard_overhead_section(
        {"bare_report": bare, "guarded_report": guarded, "overhead_threshold": 0.01}
    )
    assert passed is False
    assert any(
        item.get("severity") == "error"
        and "guard overhead failed" in item.get("message", "").lower()
        for item in payload.get("diagnostics", [])
    )
    assert "errors" not in payload


def test_compute_quality_overhead_from_guard_none_on_missing_data():
    assert (
        report_overhead_mod.compute_quality_overhead_from_guard(
            {"bare_report": {}, "guarded_report": {}}
        )
        is None
    )
