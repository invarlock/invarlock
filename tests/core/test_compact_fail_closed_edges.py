from __future__ import annotations

import pytest
import torch
from torch import nn

from invarlock.core.assurance_guard_validation_variance_binding import (
    _report_pairing_reference,
)
from invarlock.core.assurance_plugin_validation import (
    _validate_plugin_entry,
    strict_plugin_provenance_errors,
)
from invarlock.core.builtin_plugin_catalog import (
    builtin_plugin_specs,
    builtin_plugin_support_metadata,
)
from invarlock.core.runtime_observation import observe_model_runtime
from invarlock.guards.invariant_checks import assert_invariants
from invarlock.guards.quantized_weights import (
    is_packed_quantized_module,
    is_quantized_weight,
)


@pytest.mark.parametrize(
    "pairing",
    [
        {"preview": {"window_ids": "w1"}, "final": {"window_ids": ["w1"]}},
        {"preview": {"window_ids": []}, "final": {"window_ids": []}},
    ],
)
def test_pairing_reference_rejects_unreplayable_or_empty_windows(pairing: dict) -> None:
    assert _report_pairing_reference({"pairing_baseline": pairing}) is None


def test_pairing_reference_rejects_empty_evaluation_window_schedule() -> None:
    assert (
        _report_pairing_reference(
            {
                "evaluation_windows": {
                    "preview": {"window_ids": []},
                    "final": {"window_ids": []},
                }
            }
        )
        is None
    )


class _TruthyNonIteratingList(list):
    def __iter__(self):
        return iter(())


def test_pairing_reference_rejects_deceptive_list_subclass_without_ids() -> None:
    deceptive_ids = _TruthyNonIteratingList(["claimed-id"])

    assert (
        _report_pairing_reference(
            {
                "evaluation_windows": {
                    "preview": {"window_ids": deceptive_ids},
                    "final": {"window_ids": deceptive_ids},
                }
            }
        )
        is None
    )


def test_plugin_provenance_rejects_blank_names_before_catalog_resolution() -> None:
    errors: list[str] = []

    assert (
        _validate_plugin_entry(
            errors,
            entry={"name": "  "},
            path="plugins.adapter",
            plugin_type="adapters",
        )
        is None
    )
    assert errors == ["plugins.adapter.name must be a non-empty string."]


def test_demo_plugin_is_present_but_never_strict_assurance_eligible() -> None:
    demo = next(
        spec
        for spec in builtin_plugin_specs("guards")
        if spec.name == "demo_hello_guard"
    )
    errors: list[str] = []
    entry = {
        "name": demo.name,
        "type": "guards",
        "module": demo.module,
        "package": "invarlock",
        "support_tier": demo.support_tier,
        "strict_assurance_allowed": demo.strict_assurance_allowed,
        "available": True,
    }

    _validate_plugin_entry(
        errors,
        entry=entry,
        path="plugins.guards[0]",
        plugin_type="guards",
    )

    assert errors == ["plugins.guards[0] is not eligible for strict assurance."]
    assert (
        builtin_plugin_support_metadata("guards", "not-installed")["support_tier"]
        == "third_party"
    )
    with pytest.raises(ValueError, match="Unknown plugin catalog type"):
        builtin_plugin_specs("models")


def test_plugin_provenance_rejects_name_mismatch_and_missing_guard_inventory() -> None:
    errors: list[str] = []
    _validate_plugin_entry(
        errors,
        entry={"name": "spectral"},
        path="plugins.guards[0]",
        plugin_type="guards",
        expected_name="invariants",
    )
    assert any("name must be 'invariants'" in error for error in errors)

    errors = strict_plugin_provenance_errors(
        {"plugins": {"adapter": None, "edit": None, "guards": "invalid"}},
        canonical_guard_chain=("invariants",),
    )
    assert any("plugins.guards provenance array" in error for error in errors)


def test_plugin_provenance_rejects_guard_inventory_length_drift() -> None:
    errors = strict_plugin_provenance_errors(
        {"plugins": {"adapter": None, "edit": None, "guards": []}},
        canonical_guard_chain=("invariants",),
    )

    assert any("exactly cover the canonical guard chain" in error for error in errors)
    assert any("plugins.guards[0] plugin provenance" in error for error in errors)


class _ExplodingModel:
    def __getattribute__(self, name: str):
        if name in {"named_modules", "modules"}:
            raise RuntimeError("unobservable")
        return super().__getattribute__(name)


class _ExplodingWeight(nn.Module):
    @property
    def weight(self):
        raise RuntimeError("unreadable")


class _ExplodingTraversal:
    def named_modules(self):
        raise RuntimeError("unreadable")


def test_runtime_observation_fails_closed_on_object_and_traversal_failures() -> None:
    assert observe_model_runtime(_ExplodingModel()) == (False, ())
    assert observe_model_runtime(_ExplodingTraversal()) == (False, ())


def test_runtime_observation_skips_unreadable_direct_weight_without_losing_module() -> (
    None
):
    observed, inventory = observe_model_runtime(_ExplodingWeight())

    assert observed is True
    assert inventory
    assert all(item.kind == "module" for item in inventory)


def test_assert_invariants_raises_with_runtime_violation_details() -> None:
    model = nn.Linear(2, 2)
    with torch.no_grad():
        model.weight[0, 0] = float("nan")

    with pytest.raises(AssertionError, match="NaN detected in parameter weight"):
        assert_invariants(model)


class _ExplodingQuantizedFlag:
    @property
    def is_quantized(self):
        raise RuntimeError("metadata unavailable")


class _ExplodingPackedAttr:
    weight = None

    @property
    def qweight(self):
        raise RuntimeError("metadata unavailable")


class _ExplodingState:
    weight = None

    def state_dict(self):
        raise RuntimeError("metadata unavailable")


class _PackedState:
    weight = None

    def state_dict(self):
        return {"qweight": object()}


def test_unreadable_quantized_metadata_is_treated_as_packed_fail_closed() -> None:
    assert is_quantized_weight(_ExplodingQuantizedFlag()) is True
    assert is_packed_quantized_module(_ExplodingPackedAttr()) is True
    assert is_packed_quantized_module(_ExplodingState()) is True
    assert is_packed_quantized_module(_PackedState()) is True
