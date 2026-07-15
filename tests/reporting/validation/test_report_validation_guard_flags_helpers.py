from __future__ import annotations

from collections import UserDict
from types import MappingProxyType

import pytest

from invarlock.reporting.validation import guard_flags
from invarlock.reporting.validation.guard_flags import (
    resolve_invariants_pass,
    resolve_rmt_stable,
    resolve_spectral_stable,
)


def test_guard_flag_helpers_reject_non_numeric_and_non_mapping_caps() -> None:
    assert guard_flags._finite_float(object()) is None
    assert (
        resolve_spectral_stable({"caps_applied": object()}, tier_policy=None) is False
    )
    assert (
        resolve_spectral_stable(
            {"caps_applied": 0, "max_caps": object()}, tier_policy=None
        )
        is False
    )
    assert (
        resolve_spectral_stable(
            {"caps_applied": 0},
            tier_policy={"spectral": "not-a-policy"},
        )
        is True
    )


@pytest.mark.parametrize("bad_payload", [None, [], "ok", 1.0])
def test_guard_flag_helpers_fail_closed_on_non_mapping_inputs(bad_payload) -> None:
    assert (
        resolve_spectral_stable(
            bad_payload,
            tier_policy={"spectral": {"max_caps": 5}},
        )
        is False
    )
    assert resolve_rmt_stable(bad_payload) is False
    assert resolve_invariants_pass(bad_payload) is False


def test_spectral_stability_accepts_mapping_wrapped_evidence_and_policy() -> None:
    spectral = MappingProxyType({"caps_applied": 2})
    policy = UserDict({"spectral": MappingProxyType({"max_caps": 2})})

    assert resolve_spectral_stable(spectral, tier_policy=policy) is True


def test_spectral_stability_uses_mapping_wrapped_summary_budget() -> None:
    spectral = MappingProxyType(
        {
            "caps_applied": 3,
            "summary": MappingProxyType({"max_caps": 2}),
        }
    )

    assert resolve_spectral_stable(spectral, tier_policy=None) is False


def test_rmt_and_invariants_accept_mapping_wrapped_evidence() -> None:
    assert resolve_rmt_stable(MappingProxyType({"stable": True})) is True
    assert resolve_invariants_pass(MappingProxyType({"status": " PASS "})) is True
