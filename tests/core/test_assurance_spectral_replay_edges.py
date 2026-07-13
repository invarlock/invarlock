from __future__ import annotations

import pytest

from invarlock.core.assurance_spectral_replay_common import (
    _compare_tree,
    _compare_violations,
    _degeneracy_map,
    _family_map,
    _finite,
    _nonnegative_int,
    _numeric_map,
    _policy_number,
    _reject_families,
)
from invarlock.core.assurance_spectral_replay_correction import (
    _correction_bindings,
    _finding_bindings,
    _replay_correction_entries,
    replay_correction_ledger,
)
from invarlock.core.assurance_spectral_replay_decision import replay_selected_findings
from invarlock.core.assurance_spectral_replay_inventory import (
    _excluded_module_name,
    _module_list,
    replay_measurement_inventory,
)


def test_bh_replay_selects_the_largest_valid_prefix_not_only_individual_hits() -> None:
    pvalues = {"attention": 0.01, "ffn": 0.03, "embedding": 0.20}

    selected = _reject_families(pvalues, method="bh", alpha=0.05, m=3)

    assert selected == {"attention", "ffn"}


def test_violation_replay_rejects_duplicate_keys_even_if_values_are_identical() -> None:
    finding = {
        "type": "family_z_cap",
        "severity": "budgeted",
        "module": "layer.0",
        "family": "ffn",
        "z_score": 4.0,
    }
    errors: list[str] = []

    _compare_violations(errors, [finding, dict(finding)], [finding], "violations")

    assert any("duplicate violation" in error for error in errors)


def test_measurement_inventory_rejects_eligible_module_without_measurement() -> None:
    errors: list[str] = []
    entry = {
        "measurement_inventory": {
            phase: {
                "schema_version": 1,
                "phase": phase,
                "enumerated_modules": ["layer.0", "layer.1"],
                "eligible_modules": ["layer.0", "layer.1"],
                "measured_modules": ["layer.0"],
                "excluded_modules": [
                    {
                        "module": "layer.1",
                        "stage": "measurement",
                        "reason": "estimator_error",
                    }
                ],
                "identity_changed_modules": [],
                "discovery_errors": [],
                "enumerated_count": 2,
                "eligible_count": 2,
                "measured_count": 1,
                "excluded_count": 1,
                "identity_changed_count": 0,
                "discovery_error_count": 0,
            }
            for phase in ("prepare", "validate")
        },
        "correction_ledger": {"phase": "validate"},
    }

    replay_measurement_inventory(
        errors,
        entry,
        "guards[0]",
        baseline_modules={"layer.0"},
        final_modules={"layer.0"},
    )

    assert any(
        "eligible module without a valid measurement" in error for error in errors
    )


def test_correction_binding_rejects_cross_module_finding_references() -> None:
    errors: list[str] = []
    finding_ids = {"finding-0001:family_z_cap:layer.0", "finding-0002:max:layer.1"}
    finding_ids_by_module = {
        "layer.0": {"finding-0001:family_z_cap:layer.0"},
        "layer.1": {"finding-0002:max:layer.1"},
    }
    corrections = [
        {
            "correction_id": "correction-0001:layer.0",
            "module": "layer.0",
            "finding_ids": ["finding-0002:max:layer.1"],
        },
        {
            "correction_id": "correction-0002:layer.1",
            "module": "layer.1",
            "finding_ids": ["finding-0001:family_z_cap:layer.0"],
        },
    ]

    _correction_bindings(
        errors,
        corrections,
        "guards[0]",
        finding_ids,
        finding_ids_by_module,
    )

    assert (
        sum("finding_ids do not bind every module finding" in error for error in errors)
        == 2
    )
    assert any("reference every selected finding" in error for error in errors)


def test_applied_correction_requires_changed_weight_digest_and_remeasurement() -> None:
    errors: list[str] = []
    finding_id = "finding-0001:family_z_cap:layer.0"
    correction = {
        "correction_id": "correction-0001:layer.0",
        "finding_ids": [finding_id],
        "module": "layer.0",
        "operation": "relative_spectral_cap",
        "attempted": True,
        "mutation_applied": True,
        "outcome": "applied_and_remeasured",
        "pre_sigma": 3.0,
        "baseline_sigma": 1.0,
        "post_sigma": 2.2,
        "scale_factor": 2.0 / 3.0,
        "pre_weight_digest": "a" * 64,
        "post_weight_digest": "a" * 64,
    }

    applied, attempted, result = _replay_correction_entries(
        errors,
        source="guards[0]",
        corrections_by_module={"layer.0": correction},
        selected=[{"module": "layer.0", "finding_id": finding_id}],
        baseline={"layer.0": 1.0},
        pre_metrics={"layer.0": 3.0},
        post_metrics={"layer.0": 2.2},
        correction_enabled=True,
        correction_cap_ratio=2.0,
    )

    assert (applied, attempted, result) == (1, 1, "corrections_applied")
    assert any("does not prove the recorded mutation" in error for error in errors)
    assert any("did not change the weight digest" in error for error in errors)


def test_common_replay_parsers_reject_nonfinite_and_unbound_measurements() -> None:
    assert _finite(True) is None
    assert _finite(object()) is None
    assert _finite(float("inf")) is None
    assert _nonnegative_int(True) is None
    assert _nonnegative_int(-1) is None

    errors: list[str] = []
    assert _numeric_map(errors, {}, "metrics") is None
    parsed = _numeric_map(
        errors,
        {"": 1.0, "negative": -1.0, "valid": 2.0},
        "metrics",
        nonnegative=True,
    )
    assert parsed == {"valid": 2.0}
    assert any("keys must be non-empty" in error for error in errors)
    assert any("finite non-negative" in error for error in errors)

    errors = []
    assert _family_map(errors, {}, "families") is None
    assert _family_map(errors, {"": "ffn", "layer": " "}, "families") == {}
    assert len(errors) == 3

    errors = []
    assert _policy_number(errors, {}, "alpha", "policy", minimum=0.0) is None
    assert errors == ["policy.alpha must be a finite number >= 0.0."]


def test_common_tree_and_degeneracy_replay_detect_shape_and_value_tampering() -> None:
    errors: list[str] = []
    _compare_tree(errors, [], {"value": 1.0}, "tree")
    _compare_tree(errors, {"other": 1.0}, {"value": 1.0}, "tree")
    _compare_tree(errors, "bad", ["ok"], "list")
    _compare_tree(errors, "nan", 1.0, "number")
    _compare_tree(errors, "wrong", "expected", "scalar")
    assert len(errors) == 5

    errors = []
    assert _degeneracy_map(errors, {}, "degeneracy", {"layer"}) is None
    parsed = _degeneracy_map(
        errors,
        {"layer": {"stable_rank": -1.0, "norm_collapse": "bad"}},
        "degeneracy",
        {"layer"},
    )
    assert parsed == {}
    assert sum("finite non-negative" in error for error in errors) == 2


def test_violation_replay_rejects_nonarrays_nonobjects_and_inventory_drift() -> None:
    expected = [{"type": "max", "module": "layer", "family": "ffn"}]
    errors: list[str] = []
    _compare_violations(errors, {}, expected, "violations")
    _compare_violations(errors, ["bad"], expected, "violations")
    _compare_violations(
        errors,
        [{"type": "other", "module": "layer", "family": "ffn"}],
        expected,
        "violations",
    )
    assert any("must be an array" in error for error in errors)
    assert any("entries must be objects" in error for error in errors)
    assert any("missing=" in error and "unexpected=" in error for error in errors)


def test_inventory_primitives_reject_untyped_duplicate_and_aliasless_exclusions() -> (
    None
):
    errors: list[str] = []
    assert _module_list(errors, "layer", "modules") is None
    assert _module_list(errors, ["b", "a", "a"], "modules") is None
    assert _excluded_module_name(errors, [], "excluded[0]") is None
    assert _excluded_module_name(errors, {"module": 1}, "excluded[1]") is None
    assert (
        _excluded_module_name(
            errors,
            {"module": "alias", "stage": "unknown", "reason": "parameter_alias"},
            "excluded[2]",
        )
        == "alias"
    )
    assert any("alias_of" in error for error in errors)


def _inventory(phase: str, *, measured: list[str] | None = None) -> dict:
    measured = ["layer.0"] if measured is None else measured
    return {
        "schema_version": 1,
        "phase": phase,
        "enumerated_modules": ["layer.0"],
        "eligible_modules": ["layer.0"],
        "measured_modules": measured,
        "excluded_modules": [],
        "identity_changed_modules": [],
        "discovery_errors": [],
        "enumerated_count": 1,
        "eligible_count": 1,
        "measured_count": len(measured),
        "excluded_count": 0,
        "identity_changed_count": 0,
        "discovery_error_count": 0,
    }


def test_inventory_replay_rejects_missing_phases_and_cross_phase_drift() -> None:
    errors: list[str] = []
    entry = {
        "measurement_inventory": {
            "prepare": _inventory("prepare"),
            "validate": {
                **_inventory("validate"),
                "enumerated_modules": ["layer.0", "layer.1"],
                "eligible_modules": ["layer.0", "layer.1"],
                "enumerated_count": 2,
                "eligible_count": 2,
            },
        },
        "correction_ledger": {"phase": "final"},
    }

    replay_measurement_inventory(
        errors,
        entry,
        "guard",
        baseline_modules={"different"},
        final_modules={"layer.0"},
    )

    text = "\n".join(errors)
    assert "disagrees across measurement phases" in text
    assert "disagrees with baseline module sigmas" in text
    assert "lacks the correction-ledger final phase" in text


def test_inventory_replay_rejects_identity_discovery_count_and_partition_forgery() -> (
    None
):
    inventory = _inventory("prepare", measured=[])
    inventory.update(
        {
            "identity_changed_modules": ["layer.0"],
            "discovery_errors": ["adapter"],
            "excluded_modules": [
                {
                    "module": "layer.0",
                    "stage": "selection",
                    "reason": "parameter_alias",
                    "alias_of": "missing",
                }
            ],
            "measured_count": 99,
        }
    )
    errors: list[str] = []
    replay_measurement_inventory(
        errors,
        {
            "measurement_inventory": {"prepare": inventory},
            "correction_ledger": {"phase": "prepare"},
        },
        "guard",
        baseline_modules=set(),
        final_modules={"unexpected"},
    )

    text = "\n".join(errors)
    assert "identity changes" in text
    assert "incomplete adapter module discovery" in text
    assert "lacks an eligible primary" in text
    assert "measured_count disagrees" in text
    assert "disagrees with final_metrics" in text


def test_decision_replay_covers_absolute_and_degeneracy_fatal_findings() -> None:
    result = replay_selected_findings(
        baseline={"a": 1.0, "b": 1.0},
        current={"a": 3.0, "b": 1.1},
        families={"a": "ffn", "b": "ffn"},
        family_stats={"ffn": {"mean": 1.0, "std": 0.5}},
        family_caps={"ffn": 1.0},
        deadband=0.1,
        max_norm=2.0,
        method="bonferroni",
        alpha=0.05,
        configured_m=1,
        degeneracy_enabled=True,
        baseline_degeneracy={
            "a": {"stable_rank": 2.0, "norm_collapse": 1.0},
            "b": {"stable_rank": 0.0, "norm_collapse": 1.0},
        },
        current_degeneracy={
            "a": {"stable_rank": 0.2, "norm_collapse": 0.4},
            "b": {"stable_rank": 0.0, "norm_collapse": 1.0},
        },
        thresholds={"stable_rank": (0.8, 0.5), "norm_collapse": (0.8, 0.2)},
    )
    _z, budgeted, fatal, selection, selected = result

    assert {item["type"] for item in fatal} == {
        "max_spectral_norm",
        "degeneracy_stable_rank_drop",
    }
    assert any(item["type"] == "degeneracy_norm_collapse" for item in budgeted)
    assert selection["default_selected_without_pvalue"] == 1
    assert any(item["type"] == "degeneracy_norm_collapse" for item in selected)


def test_correction_binding_rejects_malformed_duplicate_and_noncanonical_entries() -> (
    None
):
    errors: list[str] = []
    ids, by_module = _finding_bindings(
        errors,
        [
            None,
            {},
            {"finding_id": "id", "module": "layer"},
            {"finding_id": "id", "module": "layer"},
        ],
        "guard",
    )
    corrections = _correction_bindings(
        errors,
        [
            None,
            {"module": "missing"},
            {
                "module": "layer",
                "correction_id": "wrong",
                "finding_ids": [],
                "unsupported": True,
            },
            {"module": "layer", "correction_id": "wrong", "finding_ids": ["id"]},
        ],
        "guard",
        ids,
        by_module,
    )

    assert set(corrections) == {"layer"}
    text = "\n".join(errors)
    assert "finding_id must be a non-empty string" in text
    assert "finding IDs must be unique" in text
    assert "must be an object" in text
    assert "unsupported fields" in text
    assert "duplicate module corrections" in text
    assert "correction_id is not canonical" in text

    errors = []
    _correction_bindings(
        errors,
        [],
        "guard",
        {"id"},
        {"layer": {"id"}},
    )
    assert any("cover every selected module" in error for error in errors)


def test_inventory_replay_rejects_malformed_phase_shapes() -> None:
    errors: list[str] = []
    assert (
        replay_measurement_inventory(
            errors,
            {"measurement_inventory": {}},
            "guard",
            baseline_modules=set(),
            final_modules=set(),
        )
        is None
    )
    assert "must be a non-empty object" in errors[0]

    errors = []
    replay_measurement_inventory(
        errors,
        {
            "measurement_inventory": {
                None: {},
                "prepare": None,
                "validate": {
                    **_inventory("wrong"),
                    "excluded_modules": "bad",
                },
            },
            "correction_ledger": {"phase": "validate"},
        },
        "guard",
        baseline_modules=set(),
        final_modules=set(),
    )
    text = "\n".join(errors)
    assert "phase names must be non-empty" in text
    assert "prepare must be an object" in text
    assert "invalid schema_version or phase binding" in text
    assert "excluded_modules must be an array" in text
    assert "measurement_inventory.prepare is required" in text


def test_inventory_replay_rejects_bad_exclusion_partition_and_eligibility() -> None:
    prepare = _inventory("prepare", measured=["layer.0"])
    prepare.update(
        {
            "enumerated_modules": ["layer.0", "layer.1"],
            "eligible_modules": ["layer.0"],
            "excluded_modules": [
                None,
                {"module": "layer.1", "stage": "measurement", "reason": "unknown"},
                {"module": "layer.1", "stage": "selection", "reason": "scope_mismatch"},
            ],
            "enumerated_count": 2,
            "eligible_count": 1,
            "excluded_count": 1,
        }
    )
    errors: list[str] = []
    replay_measurement_inventory(
        errors,
        {
            "measurement_inventory": {"prepare": prepare},
            "correction_ledger": {"phase": "prepare"},
        },
        "guard",
        baseline_modules={"layer.0"},
        final_modules={"layer.0"},
    )
    text = "\n".join(errors)
    assert "reason is not a recognized typed reason" in text
    assert "sorted and unique by module" in text
    assert "measurement exclusion 'layer.1' is not marked eligible" in text


def test_inventory_replay_rejects_inconsistent_eligibility_sets() -> None:
    inventory = _inventory("prepare", measured=["layer.0"])
    inventory["eligible_modules"] = []
    inventory["eligible_count"] = 0
    errors: list[str] = []

    replay_measurement_inventory(
        errors,
        {
            "measurement_inventory": {"prepare": inventory},
            "correction_ledger": {"phase": "prepare"},
        },
        "guard",
        baseline_modules={"layer.0"},
        final_modules={"layer.0"},
    )

    assert any("eligibility and measurement sets are inconsistent" in e for e in errors)


def test_correction_entry_replay_rejects_forged_applied_receipt_fields() -> None:
    errors: list[str] = []
    _replay_correction_entries(
        errors,
        source="guard",
        corrections_by_module={
            "layer": {
                "attempted": False,
                "operation": "none",
                "mutation_applied": True,
                "outcome": "forged",
                "pre_sigma": "bad",
                "baseline_sigma": "bad",
                "post_sigma": "bad",
                "scale_factor": "bad",
                "pre_weight_digest": "bad",
                "post_weight_digest": "bad",
            }
        },
        selected=[{"module": "layer"}],
        baseline={"layer": 1.0},
        pre_metrics={"layer": 3.0},
        post_metrics={"layer": 2.0},
        correction_enabled=True,
        correction_cap_ratio=2.0,
    )
    text = "\n".join(errors)
    for fragment in (
        "invalid attempted state",
        "invalid operation",
        "invalid outcome",
        "pre_weight_digest is invalid",
        "post_weight_digest is invalid",
        "pre_sigma is inconsistent",
        "baseline_sigma is inconsistent",
        "post_sigma is inconsistent",
        "scale_factor is inconsistent",
    ):
        assert fragment in text


def test_correction_entry_replay_rejects_changed_noop_receipt() -> None:
    errors: list[str] = []
    _replay_correction_entries(
        errors,
        source="guard",
        corrections_by_module={
            "layer": {
                "attempted": False,
                "operation": "none",
                "mutation_applied": False,
                "outcome": "not_attempted_policy_disabled",
                "pre_sigma": 1.0,
                "baseline_sigma": 1.0,
                "post_sigma": 2.0,
                "scale_factor": 2.0,
                "pre_weight_digest": "a" * 64,
                "post_weight_digest": "b" * 64,
            }
        },
        selected=[{"module": "layer"}],
        baseline={"layer": 1.0},
        pre_metrics={"layer": 1.0},
        post_metrics={"layer": 2.0},
        correction_enabled=False,
        correction_cap_ratio=2.0,
    )
    assert any("changed the retained measurement" in error for error in errors)
    assert any("changed the weight digest" in error for error in errors)


def _ledger_kwargs(entry: dict) -> dict:
    return {
        "entry": entry,
        "source": "guard",
        "metrics": {},
        "baseline": {"layer": 1.0},
        "final": {"layer": 1.0},
        "families": {"layer": "ffn"},
        "family_stats": {"ffn": {"mean": 1.0, "std": 0.0}},
        "family_caps": {"ffn": 2.0},
        "deadband": 0.1,
        "max_norm": None,
        "method": "bonferroni",
        "alpha": 0.05,
        "configured_m": 1,
        "degeneracy_enabled": False,
        "baseline_degeneracy": {},
        "thresholds": {},
        "correction_enabled": False,
        "correction_cap_ratio": 2.0,
        "final_caps_applied": 0,
        "final_caps_exceeded": False,
    }


def _valid_empty_ledger_entry() -> dict:
    return {
        "correction_ledger": {
            "schema_version": 1,
            "phase": "validate",
            "correction_enabled": False,
            "correction_cap_ratio": 2.0,
            "pre_correction_metrics": {"layer": 1.0},
            "post_correction_metrics": {"layer": 1.0},
            "pre_correction_z_scores": {"layer": 0.0},
            "pre_correction_degeneracy": {},
            "multiple_testing_selection": {
                "method": "bonferroni",
                "alpha": 0.05,
                "m": 1,
                "families_tested": [],
                "families_selected": [],
                "family_pvalues": {},
                "family_max_abs_z": {},
                "family_violation_counts": {},
                "default_selected_without_pvalue": 0,
            },
            "selected_findings": [],
            "corrections": [],
            "policy_result": "no_selected_findings",
        }
    }


def test_correction_ledger_rejects_missing_and_invalid_contract_header() -> None:
    errors: list[str] = []
    replay_correction_ledger(errors, **_ledger_kwargs({}))
    assert errors == ["guard.correction_ledger must be an object."]

    errors = []
    entry = {
        "correction_ledger": {
            "schema_version": 0,
            "phase": "",
            "correction_enabled": True,
            "correction_cap_ratio": "bad",
            "pre_correction_metrics": {},
            "post_correction_metrics": {},
            "pre_correction_z_scores": {},
            "unsupported": True,
        }
    }
    replay_correction_ledger(errors, **_ledger_kwargs(entry))
    text = "\n".join(errors)
    assert "contains unsupported fields" in text
    assert "schema_version must be 1" in text
    assert "phase must be a non-empty string" in text
    assert "correction_enabled disagrees" in text
    assert "correction_cap_ratio disagrees" in text


@pytest.mark.parametrize(
    ("mutation", "fragment"),
    [
        ("inventory", "module inventories must match baseline"),
        ("z", "disagrees with replayed pre-correction measurements"),
        ("findings", "selected_findings must be an array"),
        ("corrections", "corrections must be an array"),
        ("policy", "policy_result disagrees"),
    ],
)
def test_correction_ledger_rejects_replay_boundary_tampering(
    mutation: str, fragment: str
) -> None:
    entry = _valid_empty_ledger_entry()
    ledger = entry["correction_ledger"]
    if mutation == "inventory":
        ledger["pre_correction_metrics"] = {"other": 1.0}
    elif mutation == "z":
        ledger["pre_correction_z_scores"]["layer"] = 2.0
    elif mutation == "findings":
        ledger["selected_findings"] = {}
    elif mutation == "corrections":
        ledger["corrections"] = {}
    elif mutation == "policy":
        ledger["policy_result"] = "forged"

    errors: list[str] = []
    kwargs = _ledger_kwargs(entry)
    kwargs["metrics"] = {
        "selected_budgeted_findings": 0,
        "cap_budget_exceeded": False,
        "corrections_attempted": 0,
        "corrections_applied": 0,
        "correction_policy_result": "no_selected_findings",
    }
    replay_correction_ledger(errors, **kwargs)

    assert fragment in "\n".join(errors)


def test_correction_ledger_requires_degeneracy_object_when_replay_disabled() -> None:
    entry = _valid_empty_ledger_entry()
    entry["correction_ledger"]["pre_correction_degeneracy"] = []
    kwargs = _ledger_kwargs(entry)
    errors: list[str] = []

    replay_correction_ledger(errors, **kwargs)

    assert any("pre_correction_degeneracy must be an object" in e for e in errors)
