from __future__ import annotations

from invarlock.reporting.guards_invariants import _extract_invariants


def _invariant_report(checks: dict, *, warnings: int = 0) -> dict:
    return {
        "guards": [
            {
                "name": "invariants",
                "passed": True,
                "metrics": {
                    "checks_performed": len(checks),
                    "violations_found": warnings,
                    "fatal_violations": 0,
                    "warning_violations": warnings,
                },
                "violations": [],
                "details": {
                    "baseline_checks": checks,
                    "current_checks": checks,
                },
            }
        ],
        "metrics": {"invariants": {}},
    }


def test_invariants_baseline_compare_tokenizer_mismatch_fails() -> None:
    baseline_checks = {
        "parameter_count": 100,
        "layer_norm_paths": ("ln",),
        "embedding_vocab_sizes": {"embed": 10},
        "structure_hash": "deadbeef",
        "weight_tying": True,
    }
    current_checks = {
        **baseline_checks,
        "embedding_vocab_sizes": {"embed": 11},
    }

    baseline_report = {
        "guards": [
            {
                "name": "invariants",
                "metrics": {
                    "checks_performed": 5,
                    "violations_found": 0,
                    "fatal_violations": 0,
                    "warning_violations": 0,
                },
                "violations": [],
                "details": {
                    "baseline_checks": baseline_checks,
                    "current_checks": baseline_checks,
                },
            }
        ],
        "metrics": {"invariants": {}},
    }

    report = {
        "guards": [
            {
                "name": "invariants",
                "metrics": {
                    "checks_performed": 5,
                    "violations_found": 0,
                    "fatal_violations": 0,
                    "warning_violations": 0,
                },
                "violations": [],
                "details": {
                    "baseline_checks": current_checks,
                    "current_checks": current_checks,
                },
            }
        ],
        "metrics": {"invariants": {}},
    }

    out = _extract_invariants(report, baseline=baseline_report)
    assert out["status"] == "fail"
    assert any(f.get("type") == "tokenizer_mismatch" for f in out["failures"])


def test_invariants_baseline_compare_allows_embedding_path_alias() -> None:
    baseline_checks = {
        "parameter_count": 100,
        "layer_norm_paths": ("ln",),
        "embedding_vocab_sizes": {"model.embed_tokens": 32000},
        "structure_hash": "baseline",
        "weight_tying": True,
    }
    current_checks = {
        **baseline_checks,
        "parameter_count": 75,
        "embedding_vocab_sizes": {"model.model.embed_tokens": 32000},
        "structure_hash": "current",
        "weight_tying": None,
    }

    baseline_report = {
        "guards": [
            {
                "name": "invariants",
                "passed": True,
                "metrics": {},
                "violations": [],
                "details": {
                    "baseline_checks": baseline_checks,
                    "current_checks": baseline_checks,
                },
            }
        ],
        "metrics": {"invariants": {}},
    }
    report = {
        "guards": [
            {
                "name": "invariants",
                "passed": True,
                "metrics": {},
                "violations": [],
                "details": {
                    "baseline_checks": current_checks,
                    "current_checks": current_checks,
                },
            }
        ],
        "metrics": {"invariants": {}},
    }

    out = _extract_invariants(report, baseline=baseline_report)
    assert out["status"] == "warn"
    assert not any(f.get("type") == "tokenizer_mismatch" for f in out["failures"])
    assert any(f.get("check") == "parameter_count" for f in out["failures"])


def test_invariants_baseline_compare_invariant_violation_warns() -> None:
    baseline_checks = {
        "parameter_count": 100,
        "layer_norm_paths": ("ln",),
        "embedding_vocab_sizes": {"embed": 10},
        "structure_hash": "deadbeef",
        "weight_tying": True,
    }
    current_checks = {
        **baseline_checks,
        "parameter_count": 101,
    }

    baseline_report = {
        "guards": [
            {
                "name": "invariants",
                "passed": True,
                "metrics": {
                    "checks_performed": 5,
                    "violations_found": 0,
                    "fatal_violations": 0,
                    "warning_violations": 0,
                },
                "violations": [],
                "details": {
                    "baseline_checks": baseline_checks,
                    "current_checks": baseline_checks,
                },
            }
        ],
        "metrics": {"invariants": {}},
    }

    report = {
        "guards": [
            {
                "name": "invariants",
                "passed": True,
                "metrics": {
                    "checks_performed": 5,
                    "violations_found": 0,
                    "fatal_violations": 0,
                    "warning_violations": 0,
                },
                "violations": [],
                "details": {
                    "baseline_checks": current_checks,
                    "current_checks": current_checks,
                },
            }
        ],
        "metrics": {"invariants": {}},
    }

    out = _extract_invariants(report, baseline=baseline_report)
    assert out["status"] == "warn"
    assert any(f.get("type") == "invariant_violation" for f in out["failures"])


def test_invariants_pre_stage_metrics_are_not_counted_twice() -> None:
    report = _invariant_report({"parameter_count": 100}, warnings=1)
    report["guards"][0]["stage"] = "pre"

    out = _extract_invariants(report)

    assert out["summary"]["warning_violations"] == 1
    assert out["summary"]["violations_found"] == 1


def test_invariants_allow_exact_bnb8_structure_substitution() -> None:
    baseline_checks = {
        "parameter_count": 32,
        "structure_hash": "dense",
        "module_type_paths": {
            "": "example.Model",
            "proj": "torch.nn.modules.linear.Linear",
        },
        "linear_dimensions": {"proj": [4, 8]},
        "parameter_shapes": {"proj.weight": [8, 4]},
    }
    current_checks = {
        "parameter_count": 32,
        "structure_hash": "bnb8",
        "module_type_paths": {
            "": "example.Model",
            "proj": "bitsandbytes.nn.modules.Linear8bitLt",
        },
        "linear_dimensions": {"proj": [4, 8]},
        "parameter_shapes": {"proj.weight": [8, 4]},
        "quantized_runtime_observation": {
            "schema": "invarlock/quantized-structure-observation-v1",
            "adapter": "hf_bnb",
            "count": 1,
            "types": ["bitsandbytes.nn.modules.Linear8bitLt"],
            "kinds": ["module"],
            "modules": {"proj": "bitsandbytes.nn.modules.Linear8bitLt"},
        },
    }

    out = _extract_invariants(
        _invariant_report(current_checks),
        baseline=_invariant_report(baseline_checks),
    )

    assert out["status"] == "pass"
    assert not out["failures"]


def test_invariants_allow_exact_bnb4_structure_substitution() -> None:
    baseline_checks = {
        "parameter_count": 32,
        "structure_hash": "dense",
        "module_type_paths": {"proj": "torch.nn.modules.linear.Linear"},
        "linear_dimensions": {"proj": [4, 8]},
        "parameter_shapes": {"proj.weight": [8, 4]},
    }
    current_checks = {
        "parameter_count": 16,
        "structure_hash": "bnb4",
        "module_type_paths": {"proj": "bitsandbytes.nn.modules.Linear4bit"},
        "linear_dimensions": {"proj": [4, 8]},
        "parameter_shapes": {"proj.weight": [16, 1]},
        "quantized_runtime_observation": {
            "schema": "invarlock/quantized-structure-observation-v1",
            "adapter": "hf_bnb",
            "count": 1,
            "types": ["bitsandbytes.nn.modules.Linear4bit"],
            "kinds": ["module"],
            "modules": {"proj": "bitsandbytes.nn.modules.Linear4bit"},
        },
    }

    out = _extract_invariants(
        _invariant_report(current_checks),
        baseline=_invariant_report(baseline_checks),
    )

    assert out["status"] == "pass"
    assert not out["failures"]


def test_invariants_reject_copied_bnb_observation_on_dense_structure() -> None:
    baseline_checks = {
        "structure_hash": "dense",
        "module_type_paths": {"proj": "torch.nn.modules.linear.Linear"},
        "linear_dimensions": {"proj": [4, 8]},
    }
    current_checks = {
        **baseline_checks,
        "quantized_runtime_observation": {
            "schema": "invarlock/quantized-structure-observation-v1",
            "adapter": "hf_bnb",
            "count": 1,
            "types": ["bitsandbytes.nn.modules.Linear8bitLt"],
            "kinds": ["module"],
            "modules": {"proj": "bitsandbytes.nn.modules.Linear8bitLt"},
        },
    }

    out = _extract_invariants(
        _invariant_report(current_checks),
        baseline=_invariant_report(baseline_checks),
    )

    assert out["status"] == "fail"
    assert out["passed"] is False
    assert out["decision"] == "block"
    assert out["summary"]["fatal_violations"] == 1
    assert out["summary"]["warning_violations"] == 0
    assert any(
        failure.get("type") == "quantized_structure_unproven"
        for failure in out["failures"]
    )


def test_invariants_reject_unrelated_drift_alongside_bnb_substitution() -> None:
    baseline_checks = {
        "structure_hash": "dense",
        "module_type_paths": {
            "proj": "torch.nn.modules.linear.Linear",
            "norm": "torch.nn.modules.normalization.LayerNorm",
        },
        "linear_dimensions": {"proj": [4, 8]},
    }
    current_checks = {
        "structure_hash": "mixed",
        "module_type_paths": {
            "proj": "bitsandbytes.nn.modules.Linear8bitLt",
            "norm": "example.UnrelatedNorm",
        },
        "linear_dimensions": {"proj": [4, 8]},
        "quantized_runtime_observation": {
            "schema": "invarlock/quantized-structure-observation-v1",
            "adapter": "hf_bnb",
            "count": 1,
            "types": ["bitsandbytes.nn.modules.Linear8bitLt"],
            "kinds": ["module"],
            "modules": {"proj": "bitsandbytes.nn.modules.Linear8bitLt"},
        },
    }

    out = _extract_invariants(
        _invariant_report(current_checks),
        baseline=_invariant_report(baseline_checks),
    )

    assert out["status"] == "fail"
    assert any(
        failure.get("type") == "quantized_structure_unproven"
        for failure in out["failures"]
    )


def test_invariants_reject_bnb_substitution_with_changed_logical_dimensions() -> None:
    baseline_checks = {
        "structure_hash": "dense",
        "module_type_paths": {"proj": "torch.nn.modules.linear.Linear"},
        "linear_dimensions": {"proj": [4, 8]},
    }
    current_checks = {
        "structure_hash": "bnb8",
        "module_type_paths": {"proj": "bitsandbytes.nn.modules.Linear8bitLt"},
        "linear_dimensions": {"proj": [4, 7]},
        "quantized_runtime_observation": {
            "schema": "invarlock/quantized-structure-observation-v1",
            "adapter": "hf_bnb",
            "count": 1,
            "types": ["bitsandbytes.nn.modules.Linear8bitLt"],
            "kinds": ["module"],
            "modules": {"proj": "bitsandbytes.nn.modules.Linear8bitLt"},
        },
    }

    out = _extract_invariants(
        _invariant_report(current_checks),
        baseline=_invariant_report(baseline_checks),
    )

    assert any(
        failure.get("detail", {}).get("reason") == "logical_linear_dimensions_changed"
        for failure in out["failures"]
    )


def test_invariants_reject_retired_quantized_structure_observation_shape() -> None:
    baseline_checks = {
        "structure_hash": "dense",
        "module_type_paths": {"proj": "torch.nn.modules.linear.Linear"},
        "linear_dimensions": {"proj": [4, 8]},
    }
    current_checks = {
        "structure_hash": "bnb8",
        "module_type_paths": {"proj": "bitsandbytes.nn.modules.Linear8bitLt"},
        "linear_dimensions": {"proj": [4, 8]},
        "quantized_runtime_observation": {
            "schema": "invarlock/quantized-structure-observation-v2",
            "adapter": "hf_bnb",
            "count": 1,
            "types": ["bitsandbytes.nn.modules.Linear8bitLt"],
            "kinds": ["module"],
            "modules": {"proj": "bitsandbytes.nn.modules.Linear8bitLt"},
        },
    }

    out = _extract_invariants(
        _invariant_report(current_checks),
        baseline=_invariant_report(baseline_checks),
    )

    assert any(
        failure.get("detail", {}).get("reason") == "runtime_observation_schema_mismatch"
        for failure in out["failures"]
    )


def test_invariants_reject_false_guard_verdict_even_without_violation_rows() -> None:
    report = _invariant_report({"parameter_count": 100})
    report["guards"][0]["passed"] = False
    report["guards"][0]["decision"] = "block"

    out = _extract_invariants(report)

    assert out["status"] == "fail"
    assert out["passed"] is False
    assert any(item.get("type") == "guard_verdict_failed" for item in out["failures"])


def test_invariants_reject_block_decision_with_true_passed_field() -> None:
    report = _invariant_report({"parameter_count": 100})
    report["guards"][0]["decision"] = "block"

    out = _extract_invariants(report)

    assert out["status"] == "fail"
    assert out["passed"] is False


def test_invariants_reject_dense_linear_dimension_drift_with_same_count() -> None:
    baseline_checks = {
        "parameter_count": 32,
        "structure_hash": "same",
        "module_type_paths": {"proj": "torch.nn.modules.linear.Linear"},
        "linear_dimensions": {"proj": [4, 8]},
        "parameter_shapes": {"proj.weight": [8, 4]},
    }
    current_checks = {
        "parameter_count": 32,
        "structure_hash": "same",
        "module_type_paths": {"proj": "torch.nn.modules.linear.Linear"},
        "linear_dimensions": {"proj": [8, 4]},
        "parameter_shapes": {"proj.weight": [4, 8]},
    }

    out = _extract_invariants(
        _invariant_report(current_checks),
        baseline=_invariant_report(baseline_checks),
    )

    assert out["status"] == "warn"
    assert any(item.get("check") == "linear_dimensions" for item in out["failures"])


def test_invariants_reports_dense_parameter_shape_drift_with_same_count() -> None:
    baseline_checks = {
        "parameter_count": 8,
        "structure_hash": "same",
        "module_type_paths": {"": "example.Model"},
        "parameter_shapes": {"table": [2, 4]},
    }
    current_checks = {
        **baseline_checks,
        "parameter_shapes": {"table": [4, 2]},
    }

    out = _extract_invariants(
        _invariant_report(current_checks),
        baseline=_invariant_report(baseline_checks),
    )

    assert out["status"] == "warn"
    assert any(item.get("check") == "parameter_shapes" for item in out["failures"])


def test_invariants_reject_unexplained_parameter_count_delta_for_bnb() -> None:
    baseline_checks = {
        "parameter_count": 33,
        "structure_hash": "dense",
        "module_type_paths": {"proj": "torch.nn.modules.linear.Linear"},
        "linear_dimensions": {"proj": [4, 8]},
        "parameter_shapes": {"proj.weight": [8, 4], "unrelated": [1]},
    }
    current_checks = {
        "parameter_count": 16,
        "structure_hash": "bnb4",
        "module_type_paths": {"proj": "bitsandbytes.nn.modules.Linear4bit"},
        "linear_dimensions": {"proj": [4, 8]},
        "parameter_shapes": {"proj.weight": [16, 1], "unrelated": [1]},
        "quantized_runtime_observation": {
            "schema": "invarlock/quantized-structure-observation-v1",
            "adapter": "hf_bnb",
            "count": 1,
            "types": ["bitsandbytes.nn.modules.Linear4bit"],
            "kinds": ["module"],
            "modules": {"proj": "bitsandbytes.nn.modules.Linear4bit"},
        },
    }

    out = _extract_invariants(
        _invariant_report(current_checks),
        baseline=_invariant_report(baseline_checks),
    )

    assert out["status"] == "fail"
    assert any(
        item.get("detail", {}).get("reason")
        == "current_parameter_inventory_count_mismatch"
        for item in out["failures"]
    )


def test_invariants_reject_out_of_scope_parameter_shape_drift_for_bnb() -> None:
    baseline_checks = {
        "parameter_count": 34,
        "structure_hash": "dense",
        "module_type_paths": {"proj": "torch.nn.modules.linear.Linear"},
        "linear_dimensions": {"proj": [4, 8]},
        "parameter_shapes": {"proj.weight": [8, 4], "unrelated": [2]},
    }
    current_checks = {
        "parameter_count": 18,
        "structure_hash": "bnb4",
        "module_type_paths": {"proj": "bitsandbytes.nn.modules.Linear4bit"},
        "linear_dimensions": {"proj": [4, 8]},
        "parameter_shapes": {"proj.weight": [16, 1], "unrelated": [1, 2]},
        "quantized_runtime_observation": {
            "schema": "invarlock/quantized-structure-observation-v1",
            "adapter": "hf_bnb",
            "count": 1,
            "types": ["bitsandbytes.nn.modules.Linear4bit"],
            "kinds": ["module"],
            "modules": {"proj": "bitsandbytes.nn.modules.Linear4bit"},
        },
    }

    out = _extract_invariants(
        _invariant_report(current_checks),
        baseline=_invariant_report(baseline_checks),
    )

    assert out["status"] == "fail"
    assert any(
        item.get("detail", {}).get("reason") == "out_of_scope_parameter_shape_changed"
        for item in out["failures"]
    )
