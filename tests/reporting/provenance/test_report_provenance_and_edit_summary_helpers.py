from __future__ import annotations

from invarlock.reporting import report_edit_summary as report_edit_summary_mod
from invarlock.reporting import report_provenance as provenance_mod
from invarlock.reporting.report_enrichment import compute_confidence_label


def test_compute_edit_digest_detects_quantization():
    report = {
        "edit": {
            "name": "quant_rtn",
            "config": {"scope": "ffn", "plan": {"target_sparsity": 0.5}},
        }
    }
    digest = provenance_mod.compute_edit_digest(report)
    assert digest["family"] == "quantization"
    assert digest["version"] == 1


def test_compute_edit_digest_defaults_when_missing():
    digest = provenance_mod.compute_edit_digest({})
    assert digest["family"] == "report_only"


def test_compute_confidence_label_accuracy_medium():
    evaluation_report = {
        "validation": {"primary_metric_acceptable": True},
        "primary_metric": {
            "kind": "accuracy",
            "display_ci": (0.6, 0.9),
            "unstable": True,
        },
        "resolved_policy": {"confidence": {"accuracy_delta_pp_width_max": 0.5}},
    }
    label = compute_confidence_label(evaluation_report)
    assert label["basis"] == "accuracy"
    assert label["label"] == "Medium"


def test_compute_confidence_label_low_when_gate_fails():
    evaluation_report = {
        "validation": {"primary_metric_acceptable": False},
        "primary_metric": {"kind": "ppl_causal", "display_ci": (1.0, 1.02)},
    }
    label = compute_confidence_label(evaluation_report)
    assert label["basis"] == "ppl_ratio"
    assert label["label"] == "Low"


def test_extract_edit_metadata_infers_scope_from_budgets():
    report = {
        "edit": {
            "plan": {
                "head_budget": {"ratio": 0.5},
                "mlp_budget": {"ratio": 0.5},
            },
            "deltas": {"params_changed": 1},
            "name": "quant_rtn",
        }
    }
    metadata = report_edit_summary_mod.extract_edit_metadata(report, {})
    assert metadata["scope"] == "heads+ffn"


def test_extract_edit_metadata_uses_config_plan_fallback():
    report = {
        "edit": {
            "config": {"plan": {"scope": "ffn", "ranking": "l2"}},
            "deltas": {"params_changed": 1},
            "name": "quant_rtn",
        }
    }
    metadata = report_edit_summary_mod.extract_edit_metadata(report, {})
    assert metadata["scope"] == "ffn"
    assert metadata["ranking"] == "l2"


def test_extract_edit_metadata_uses_direct_config_fallback():
    report = {
        "edit": {
            "config": {
                "quantization_mode": "rtn_dequantized_weight_edit",
                "storage_format": "float_dequantized",
                "packed_quantized_storage": False,
                "runtime_memory_reduction": False,
                "scope": "all",
            },
            "deltas": {"params_changed": 1},
            "name": "quant_rtn",
        }
    }
    metadata = report_edit_summary_mod.extract_edit_metadata(report, {})

    assert metadata["plan"]["quantization_mode"] == "rtn_dequantized_weight_edit"
    assert metadata["plan"]["storage_format"] == "float_dequantized"
    assert metadata["scope"] == "all"


def test_extract_edit_metadata_preserves_quant_rtn_plan_payload():
    report = {
        "edit": {
            "plan": {
                "quantization_mode": "rtn_dequantized_weight_edit",
                "storage_format": "float_dequantized",
                "packed_quantized_storage": False,
                "runtime_memory_reduction": False,
                "scope": "attn",
            },
            "deltas": {"params_changed": 1},
            "name": "quant_rtn",
            "plan_digest": "sha256:abc",
        }
    }

    metadata = report_edit_summary_mod.extract_edit_metadata(report, {})

    assert metadata["plan"]["quantization_mode"] == "rtn_dequantized_weight_edit"
    assert metadata["plan"]["packed_quantized_storage"] is False
    assert metadata["scope"] == "attn"


def test_extract_edit_metadata_preserves_optional_edit_provenance_and_impact():
    report = {
        "edit": {
            "name": "custom",
            "plan_digest": "sha256:abc",
            "edit_provenance": {
                "edit_family": "knowledge_edit",
                "edit_method": "custom",
                "edit_count": 3,
                "target_set_digest": "sha256:" + "a" * 64,
                "editor_artifact_digest": "sha256:" + "b" * 64,
                "self_edit_data_digest": "sha256:" + "c" * 64,
                "dynamic_runtime_required": False,
            },
            "edit_impact": {
                "scenario_types": [
                    "target_success",
                    "near_neighbor",
                    "unrelated_locality",
                ]
            },
        }
    }

    metadata = report_edit_summary_mod.extract_edit_metadata(report, {})

    assert metadata["edit_provenance"]["edit_family"] == "knowledge_edit"
    assert metadata["edit_provenance"]["edit_count"] == 3
    assert metadata["edit_impact"]["scenario_types"] == [
        "target_success",
        "near_neighbor",
        "unrelated_locality",
    ]


def test_compute_report_digest_returns_none_for_non_dict():
    assert provenance_mod.compute_report_digest(None) is None


def test_compute_edit_digest_quantization_family():
    report = {"edit": {"name": "quant_rtn", "config": {"scope": "ffn"}}}
    digest = provenance_mod.compute_edit_digest(report)
    assert digest["family"] == "quantization"


def test_compute_edit_digest_handles_faulty_mapping():
    class Faulty:
        def get(self, *_args, **_kwargs):
            raise RuntimeError("boom")

    digest = provenance_mod.compute_edit_digest(Faulty())
    assert digest["family"] == "report_only"


def test_compute_confidence_label_accuracy_high():
    evaluation_report = {
        "validation": {"primary_metric_acceptable": True},
        "primary_metric": {"kind": "accuracy", "display_ci": (70.0, 72.0)},
        "resolved_policy": {"confidence": {"accuracy_delta_pp_width_max": 0.05}},
    }
    label = compute_confidence_label(evaluation_report)
    assert label["label"] == "Low"
    assert label["basis"] == "accuracy"


def test_extract_structural_deltas_captures_bitwidth_and_ranks():
    report = {
        "edit": {
            "name": "quant_rtn",
            "plan": {"scope": "heads"},
            "config": {"plan": {"seed": 19}},
            "deltas": {
                "params_changed": 10,
                "bitwidth_map": {
                    "layer1": {"bitwidth": 4, "group_size": 32, "params": 512}
                },
                "rank_map": {
                    "layer1": {
                        "rank": 8,
                        "params_saved": 128,
                        "energy_retained": 0.95,
                        "deploy_mode": "recompose",
                        "savings_mode": "realized",
                        "realized_params_saved": 64,
                        "theoretical_params_saved": 80,
                        "realized_params": 900,
                        "theoretical_params": 920,
                        "skipped": False,
                    }
                },
                "savings": {"deploy_mode": "recompose"},
            },
        },
        "meta": {"seed": 7},
    }
    structure = report_edit_summary_mod.extract_structural_deltas(report)
    assert "bitwidths" in structure
    assert "ranks" in structure
    diag = structure["compression_diagnostics"]
    assert diag["algorithm_details"]["seed"] == 7


def test_build_provenance_block_uses_schedule_digest(monkeypatch):
    report = {"provenance": {}, "meta": {"model_id": "model"}}
    baseline_ref = {"run_id": "baseline-1"}
    artifacts = {"generated_at": "now", "report_path": "/tmp/report"}
    policy = {"source": "auto"}
    ppl = {"window_plan": {"profile": "dev"}}

    provenance = provenance_mod.build_provenance_block(
        report,
        baseline_raw=None,
        baseline_ref=baseline_ref,
        artifacts_payload=artifacts,
        policy_provenance=policy,
        schedule_digest="abc123",
        ppl_analysis=ppl,
        current_run_id="edited-1",
        compute_report_digest_fn=lambda payload: (
            "digest" if payload is not None else None
        ),
        collect_backend_versions_fn=lambda: {"python": "x.y"},
        compute_edit_digest_fn=lambda report: {"family": "report_only"},
    )

    assert provenance["provider_digest"] == {"ids_sha256": "abc123"}
    assert provenance["window_plan_digest"] == "abc123"
    assert provenance["window_plan"]["profile"] == "dev"


def test_extract_compression_diagnostics_no_modifications():
    inference_record = {
        "flags": dict.fromkeys(("scope", "seed", "rank_policy", "frac"), False),
        "sources": {},
        "log": [],
    }
    diagnostics = report_edit_summary_mod.extract_compression_diagnostics(
        "quant_rtn",
        {"scope": "ffn", "clamp_ratio": 0.0},
        {"params_changed": 0},
        {},
        inference_record,
    )
    assert diagnostics["execution_status"] == "no_modifications"
    assert diagnostics["target_analysis"] == {
        "modules_found": 0,
        "modules_eligible": 0,
        "modules_modified": 0,
        "scope": "ffn",
    }
    assert diagnostics["parameter_analysis"] == {
        "bitwidth": {"value": "unknown", "effectiveness": "ineffective"},
        "clamp_ratio": {"value": 0.0, "effectiveness": "disabled"},
    }
    assert diagnostics["algorithm_details"] == {
        "scope_targeting": "ffn",
        "seed": "unknown",
    }
    assert diagnostics["warnings"] == [
        "No parameters were modified - algorithm may be too conservative",
        "Check scope configuration and parameter thresholds",
        "FFN scope may not match model architecture - try 'all' scope",
    ]
    assert diagnostics["inferred"] == dict.fromkeys(
        ("scope", "seed", "rank_policy", "frac"), False
    )
    assert "inference_source" not in diagnostics
    assert "inference_log" not in diagnostics


def test_extract_compression_diagnostics_quant_success():
    inference_record = {
        "flags": dict.fromkeys(("scope", "seed", "rank_policy", "frac"), False),
        "sources": {},
        "log": [],
    }
    deltas = {
        "params_changed": 5,
        "bitwidth_map": {"layer1": {"bitwidth": 8, "group_size": 32, "params": 256}},
    }
    diagnostics = report_edit_summary_mod.extract_compression_diagnostics(
        "quant_rtn",
        {"scope": "attn", "clamp_ratio": 0.5},
        deltas,
        {},
        inference_record,
    )
    assert diagnostics["execution_status"] == "successful"
    assert diagnostics["target_analysis"] == {
        "modules_found": 1,
        "modules_eligible": 1,
        "modules_modified": 1,
        "scope": "attn",
    }
    assert diagnostics["parameter_analysis"] == {
        "bitwidth": {"value": 8, "effectiveness": "applied"},
        "group_size": {"value": 32, "effectiveness": "used"},
        "clamp_ratio": {"value": 0.5, "effectiveness": "applied"},
    }
    assert diagnostics["algorithm_details"] == {
        "scope_targeting": "attn",
        "seed": "unknown",
        "modules_quantized": 1,
        "quantization_type": "grouped",
        "total_params_quantized": 256,
        "estimated_memory_saved_mb": 0.0,
    }
    assert diagnostics["warnings"] == []
    assert diagnostics["inferred"] == dict.fromkeys(
        ("scope", "seed", "rank_policy", "frac"), False
    )
    assert "inference_source" not in diagnostics
    assert "inference_log" not in diagnostics


def test_extract_rank_information_tracks_skipped_modules():
    deltas = {
        "rank_map": {
            "layer.0": {"rank": 8, "params_saved": 10},
            "layer.1": {"rank": 0, "params_saved": 0, "skipped": True},
        },
        "savings": {"deploy_mode": "recompose"},
    }
    info = report_edit_summary_mod.extract_rank_information({"frac": 0.2}, deltas)
    assert "per_module" in info
    assert info["skipped_modules"] == ["layer.1"]


def test_build_provenance_block_respects_existing_provider_digest():
    report = {
        "provenance": {"provider_digest": {"source": "pre"}},
        "artifacts": {},
    }
    provenance = provenance_mod.build_provenance_block(
        report,
        {"artifacts": {"logs_path": "/logs/base.log"}},
        {"run_id": "base-1"},
        {"generated_at": "now", "report_path": "/logs/run.log"},
        {"tier": "balanced"},
        "abc123",
        {},
        "run-1",
        compute_report_digest_fn=lambda payload: (
            "digest" if payload is not None else None
        ),
        collect_backend_versions_fn=lambda: {"python": "x.y"},
        compute_edit_digest_fn=lambda report: {"family": "report_only"},
    )
    assert provenance["provider_digest"] == {"source": "pre"}
    assert provenance["baseline"]["report_path"] == "/logs/base.log"


def test_build_provenance_block_fallbacks_to_schedule_digest():
    provenance = provenance_mod.build_provenance_block(
        {},
        {},
        {"run_id": "base-1"},
        {"generated_at": "now", "report_path": "/logs/run.log"},
        {"tier": "balanced"},
        "deadbeef",
        {},
        "run-2",
        compute_report_digest_fn=lambda payload: (
            "digest" if payload is not None else None
        ),
        collect_backend_versions_fn=lambda: {"python": "x.y"},
        compute_edit_digest_fn=lambda report: {"family": "report_only"},
    )
    assert provenance["provider_digest"] == {"ids_sha256": "deadbeef"}


def test_build_provenance_block_transfers_dataset_split_and_window_plan():
    report = {
        "provenance": {"dataset_split": "eval", "split_fallback": True},
        "artifacts": {},
    }
    ppl_analysis = {"window_plan": {"profile": "ci"}}
    provenance = provenance_mod.build_provenance_block(
        report,
        {},
        {"run_id": "base-2"},
        {"generated_at": "ts", "report_path": "/logs/run.log"},
        {"tier": "balanced"},
        "cafebabe",
        ppl_analysis,
        "run-3",
        compute_report_digest_fn=lambda payload: (
            "digest" if payload is not None else None
        ),
        collect_backend_versions_fn=lambda: {"python": "x.y"},
        compute_edit_digest_fn=lambda report: {"family": "report_only"},
    )
    assert provenance["dataset_split"] == "eval"
    assert provenance["split_fallback"] is True
    assert provenance["window_plan"]["profile"] == "ci"
    assert provenance["window_ids_digest"] == "cafebabe"


def test_compute_confidence_label_handles_unknown_metric_kind():
    evaluation_report = {
        "validation": {"primary_metric_acceptable": True},
        "primary_metric": {"kind": "custom_metric", "display_ci": (2.0, 2.5)},
        "resolved_policy": {},
    }
    label = compute_confidence_label(evaluation_report)
    assert label["basis"] == "primary_metric"
    assert label["label"] == "Low"
