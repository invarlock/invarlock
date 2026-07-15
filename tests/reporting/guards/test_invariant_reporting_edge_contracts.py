from __future__ import annotations

from invarlock.reporting import guards_invariants


def test_invariant_inventory_parsers_fail_closed() -> None:
    assert guards_invariants._string_map([]) is None
    assert guards_invariants._string_map({1: "type"}) is None
    assert guards_invariants._linear_dimension_map([]) is None
    assert guards_invariants._linear_dimension_map({"m": [1]}) is None
    assert guards_invariants._linear_dimension_map({"m": [True, 2]}) is None
    assert guards_invariants._linear_dimension_map({"m": [0, 2]}) is None
    assert guards_invariants._parameter_shape_map([]) is None
    assert guards_invariants._parameter_shape_map({"p": [True]}) is None
    assert guards_invariants._parameter_shape_map({"p": [0]}) is None
    assert guards_invariants._shape_numel((2, 3, 4)) == 24


def test_bnb_parameter_transition_rejects_unbound_inventories() -> None:
    dimensions = {"layer": (2, 3)}
    baseline = {
        "parameter_shapes": {"layer.weight": [3, 2]},
        "parameter_count": 6,
    }
    current = {
        "parameter_shapes": {"layer.weight": [3, 2]},
        "parameter_count": 6,
    }
    changed = {"layer": "bitsandbytes.nn.modules.Linear8bitLt"}

    assert guards_invariants._bnb_parameter_transition(
        baseline_checks={},
        current_checks={},
        changed_modules=changed,
        baseline_dimensions=dimensions,
    ) == (False, "parameter_shape_map_missing")
    assert guards_invariants._bnb_parameter_transition(
        baseline_checks=baseline,
        current_checks={**current, "parameter_shapes": {"other": [3, 2]}},
        changed_modules=changed,
        baseline_dimensions=dimensions,
    ) == (False, "parameter_paths_changed")
    assert guards_invariants._bnb_parameter_transition(
        baseline_checks={**baseline, "parameter_count": True},
        current_checks=current,
        changed_modules=changed,
        baseline_dimensions=dimensions,
    ) == (False, "parameter_count_missing")
    assert guards_invariants._bnb_parameter_transition(
        baseline_checks={**baseline, "parameter_count": 7},
        current_checks=current,
        changed_modules=changed,
        baseline_dimensions=dimensions,
    ) == (False, "baseline_parameter_inventory_count_mismatch")
    assert guards_invariants._bnb_parameter_transition(
        baseline_checks=baseline,
        current_checks={**current, "parameter_count": 7},
        changed_modules=changed,
        baseline_dimensions=dimensions,
    ) == (False, "current_parameter_inventory_count_mismatch")


def test_bnb_parameter_transition_rejects_shape_forgery() -> None:
    baseline = {
        "parameter_shapes": {
            "layer.weight": [3, 2],
            "layer.extra": [1],
        },
        "parameter_count": 7,
    }
    current = {
        "parameter_shapes": {
            "layer.weight": [3, 2],
            "layer.extra": [1],
        },
        "parameter_count": 7,
    }
    result = guards_invariants._bnb_parameter_transition(
        baseline_checks=baseline,
        current_checks=current,
        changed_modules={"layer": "bitsandbytes.nn.modules.Linear8bitLt"},
        baseline_dimensions={"layer": (2, 3)},
    )
    assert result == (False, "unexpected_baseline_quantized_module_parameters")

    baseline = {
        "parameter_shapes": {"layer.weight": [2, 3]},
        "parameter_count": 6,
    }
    current = {
        "parameter_shapes": {"layer.weight": [2, 3]},
        "parameter_count": 6,
    }
    assert guards_invariants._bnb_parameter_transition(
        baseline_checks=baseline,
        current_checks=current,
        changed_modules={"layer": "bitsandbytes.nn.modules.Linear8bitLt"},
        baseline_dimensions={"layer": (2, 3)},
    ) == (False, "baseline_linear_weight_shape_mismatch")


def test_metric_invariant_failure_normalization_preserves_details() -> None:
    failures = guards_invariants._metric_invariant_failures(
        {
            "ok": True,
            "scalar": False,
            "passed": {"passed": True},
            "listed": {
                "passed": False,
                "violations": [
                    "noise",
                    {"type": "drift", "severity": "warning", "delta": 2},
                ],
            },
            "fallback": {
                "passed": False,
                "type": "missing",
                "message": "not measured",
                "reason": "offline",
            },
        }
    )
    assert [item["check"] for item in failures] == ["scalar", "listed", "fallback"]
    assert failures[1]["detail"]["delta"] == 2
    assert failures[2]["detail"] == {
        "message": "not measured",
        "reason": "offline",
    }


def test_guard_verdict_requires_explicit_boolean_and_failed_evidence() -> None:
    failures: list[dict] = []
    guards_invariants._append_guard_verdict_failure(failures, {})
    assert failures[0]["type"] == "missing_explicit_verdict"

    failures = []
    guards_invariants._append_guard_verdict_failure(
        failures, {"passed": False, "decision": "block", "violations": []}
    )
    assert failures[0]["type"] == "guard_verdict_failed"

    failures = []
    guards_invariants._append_guard_verdict_failure(
        failures,
        {
            "passed": False,
            "decision": "block",
            "violations": [{"severity": "fatal"}],
        },
    )
    assert failures == []


def test_staged_invariant_extraction_fails_pre_edit_and_keeps_violations() -> None:
    report = {
        "metrics": {"invariants": {}},
        "guards": [
            {
                "name": "invariants",
                "stage": "pre",
                "passed": False,
                "decision": "block",
                "metrics": {"checks_performed": "bad"},
                "violations": ["noise", {"type": "shape", "detail": "changed"}],
            },
            {
                "name": "invariants_post",
                "stage": "post",
                "passed": True,
                "decision": "allow",
                "metrics": {
                    "checks_performed": 1,
                    "violations_found": 0,
                    "fatal_violations": 0,
                    "warning_violations": 0,
                },
            },
        ],
    }

    extracted = guards_invariants._extract_invariants(report)  # type: ignore[arg-type]

    assert extracted["pre"] == "fail"
    assert extracted["post"] == "pass"
    assert extracted["status"] == "fail"
    assert extracted["decision"] == "block"
    assert any(
        item.get("detail", {}).get("source") == "pre_edit"
        for item in extracted["failures"]
    )


def test_staged_invariant_extraction_adds_failure_when_pre_stage_is_silent() -> None:
    report = {
        "metrics": {},
        "guards": [
            {"name": "invariants", "stage": "pre", "passed": False},
            {
                "name": "invariants_post",
                "stage": "post",
                "passed": True,
                "metrics": {},
            },
        ],
    }
    extracted = guards_invariants._extract_invariants(report)  # type: ignore[arg-type]
    assert any(item["type"] == "stage_failed" for item in extracted["failures"])


def _valid_bnb_checks() -> tuple[dict, dict]:
    baseline = {
        "module_type_paths": {"layer": "torch.nn.modules.linear.Linear"},
        "linear_dimensions": {"layer": [2, 3]},
        "parameter_shapes": {"layer.weight": [3, 2], "layer.bias": [3]},
        "parameter_count": 9,
    }
    current = {
        "module_type_paths": {"layer": "bitsandbytes.nn.modules.Linear8bitLt"},
        "linear_dimensions": {"layer": [2, 3]},
        "parameter_shapes": {"layer.weight": [3, 2], "layer.bias": [3]},
        "parameter_count": 9,
        "quantized_runtime_observation": {
            "schema": "invarlock/quantized-structure-observation-v1",
            "adapter": "hf_bnb",
            "count": 1,
            "types": ["bitsandbytes.nn.modules.Linear8bitLt"],
            "kinds": ["module"],
            "modules": {"layer": "bitsandbytes.nn.modules.Linear8bitLt"},
        },
    }
    return baseline, current


def test_bnb_structure_transition_accepts_bound_dense_to_quantized_change() -> None:
    baseline, current = _valid_bnb_checks()
    assert guards_invariants._bnb_structure_transition(baseline, current) == (
        True,
        "reported_bnb_linear_substitutions_consistent",
    )


def test_bnb_structure_transition_rejects_malformed_observation_contract() -> None:
    baseline, current = _valid_bnb_checks()
    cases = [
        (None, "runtime_observation_missing"),
        ({}, "runtime_observation_fields_invalid"),
        (
            {**current["quantized_runtime_observation"], "schema": "bad"},
            "runtime_observation_schema_mismatch",
        ),
        (
            {**current["quantized_runtime_observation"], "adapter": "bad"},
            "runtime_observation_adapter_mismatch",
        ),
        (
            {**current["quantized_runtime_observation"], "count": True},
            "recognized_runtime_module_count_missing",
        ),
        (
            {**current["quantized_runtime_observation"], "kinds": []},
            "recognized_runtime_observation_kind_mismatch",
        ),
        (
            {**current["quantized_runtime_observation"], "types": []},
            "recognized_runtime_types_invalid",
        ),
    ]
    for observation, reason in cases:
        candidate = {**current, "quantized_runtime_observation": observation}
        assert guards_invariants._bnb_structure_transition(baseline, candidate) == (
            False,
            reason,
        )


def test_bnb_structure_transition_rejects_inventory_binding_mismatches() -> None:
    baseline, current = _valid_bnb_checks()
    cases = [
        ({**current, "module_type_paths": None}, "module_type_map_missing"),
        (
            {**current, "module_type_paths": {"other": "type"}},
            "module_paths_changed",
        ),
        ({**current, "linear_dimensions": None}, "linear_dimensions_missing"),
        (
            {**current, "linear_dimensions": {"layer": [2, 4]}},
            "logical_linear_dimensions_changed",
        ),
        (
            {
                **current,
                "quantized_runtime_observation": {
                    **current["quantized_runtime_observation"],
                    "modules": {},
                },
            },
            "runtime_observation_does_not_bind_all_structure_changes",
        ),
        (
            {
                **current,
                "quantized_runtime_observation": {
                    **current["quantized_runtime_observation"],
                    "count": 2,
                },
            },
            "runtime_observation_count_mismatch",
        ),
    ]
    for candidate, reason in cases:
        assert guards_invariants._bnb_structure_transition(baseline, candidate) == (
            False,
            reason,
        )


def test_guard_summary_without_guard_is_empty_pass() -> None:
    assert guards_invariants._guard_summary(None, None, []) == ({}, "pass")


def test_baseline_guard_selection_prefers_post_edit_entry() -> None:
    report = {"guards": [{"name": "invariants"}]}
    baseline = {
        "guards": [
            {"name": "invariants", "marker": "pre"},
            {"name": "other", "stage": "post", "marker": "post"},
        ]
    }
    _, _, selected, _ = guards_invariants._select_guard_entries(  # type: ignore[arg-type]
        report,
        baseline,  # type: ignore[arg-type]
    )
    assert selected["marker"] == "post"
