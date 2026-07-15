"""Closed, finite payload schemas for verdict-driving probe sidecars."""

from __future__ import annotations

import math
import re
from typing import Any


class ProbePayloadError(ValueError):
    """A probe payload does not satisfy its exact public schema."""


RMT_FIELDS = frozenset(
    {
        "schema",
        "probe",
        "stable",
        "passed",
        "action",
        "stable_guard",
        "epsilon_by_family",
        "epsilon_default",
        "epsilon_violations",
        "violations",
        "metrics",
        "binding",
    }
)
VE_FIELDS = frozenset(
    {
        "schema",
        "probe",
        "signal",
        "signal_reasons",
        "would_enable",
        "gate_reason",
        "proposed_scales",
        "ppl_no_ve",
        "ppl_with_ve",
        "abs_improvement",
        "ab_gain",
        "ratio_ci",
        "predictive_gate",
        "calibration",
        "binding",
    }
)


def _finite(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, int | float)
        and math.isfinite(float(value))
    )


def _number(value: object) -> float:
    assert not isinstance(value, bool) and isinstance(value, int | float)
    return float(value)


def _finite_mapping(value: object) -> bool:
    return isinstance(value, dict) and all(
        isinstance(key, str) and key and _finite(item) for key, item in value.items()
    )


def valid_binding_shape(value: object) -> bool:
    if not isinstance(value, dict) or set(value) != {
        "report_sha256",
        "run_id",
        "model_id",
        "runtime",
        "toolchain",
        "provider_digest",
    }:
        return False
    runtime, toolchain, provider = (
        value.get("runtime"),
        value.get("toolchain"),
        value.get("provider_digest"),
    )
    return (
        isinstance(value.get("report_sha256"), str)
        and re.fullmatch(r"sha256:[a-f0-9]{64}", value["report_sha256"]) is not None
        and all(
            isinstance(value.get(field), str) and value[field]
            for field in ("run_id", "model_id")
        )
        and isinstance(runtime, dict)
        and set(runtime) == {"execution_mode"}
        and isinstance(runtime.get("execution_mode"), str)
        and bool(runtime["execution_mode"])
        and isinstance(toolchain, dict)
        and set(toolchain) == {"adapter", "profile"}
        and all(
            isinstance(toolchain.get(field), str) and toolchain[field]
            for field in ("adapter", "profile")
        )
        and isinstance(provider, dict)
        and bool(provider)
        and all(
            isinstance(key, str) and key and isinstance(item, str) and item
            for key, item in provider.items()
        )
    )


def _valid_rmt_violation(item: object) -> bool:
    return (
        isinstance(item, dict)
        and set(item)
        == {"family", "module", "edge_base", "edge_cur", "delta", "allowed", "epsilon"}
        and isinstance(item.get("family"), str)
        and isinstance(item.get("module"), str)
        and all(
            _finite(item.get(field))
            for field in ("edge_base", "edge_cur", "delta", "allowed", "epsilon")
        )
    )


def _valid_rmt_metrics(value: object) -> bool:
    return (
        isinstance(value, dict)
        and set(value)
        == {
            "stable",
            "epsilon_default",
            "epsilon_by_family",
            "edge_base_by_family",
            "edge_cur_by_family",
        }
        and isinstance(value.get("stable"), bool)
        and _finite(value.get("epsilon_default"))
        and all(
            _finite_mapping(value.get(field))
            for field in (
                "epsilon_by_family",
                "edge_base_by_family",
                "edge_cur_by_family",
            )
        )
    )


def _close(left: object, right: object) -> bool:
    return (
        _finite(left)
        and _finite(right)
        and math.isclose(_number(left), _number(right), rel_tol=1e-9, abs_tol=1e-12)
    )


def _expected_rmt_violation_families(payload: dict[str, Any]) -> set[str]:
    metrics = payload["metrics"]
    base_by_family = metrics["edge_base_by_family"]
    current_by_family = metrics["edge_cur_by_family"]
    if set(base_by_family) != set(current_by_family):
        raise ProbePayloadError("RMT family aggregates do not cover the same families")
    if not _close(payload["epsilon_default"], metrics["epsilon_default"]):
        raise ProbePayloadError("RMT epsilon policy disagrees with metrics")
    if payload["epsilon_by_family"] != metrics["epsilon_by_family"]:
        raise ProbePayloadError("RMT family epsilon policy disagrees with metrics")

    expected: set[str] = set()
    for family, base in base_by_family.items():
        current = current_by_family[family]
        epsilon = payload["epsilon_by_family"].get(family, payload["epsilon_default"])
        if base < 0 or current < 0 or epsilon < 0:
            raise ProbePayloadError(
                "RMT family aggregates and epsilon must be non-negative"
            )
        if base > 0 and current > base * (1.0 + epsilon):
            expected.add(family)
    return expected


def _validate_rmt_violations(
    payload: dict[str, Any], expected_families: set[str]
) -> None:
    metrics = payload["metrics"]
    observed_families: set[str] = set()
    for violation in payload["epsilon_violations"]:
        base = violation["edge_base"]
        current = violation["edge_cur"]
        epsilon = violation["epsilon"]
        family = violation["family"]
        if family in observed_families or family not in metrics["edge_base_by_family"]:
            raise ProbePayloadError(
                "RMT epsilon violations contain an unknown or duplicate family"
            )
        observed_families.add(family)
        if (
            base <= 0
            or epsilon < 0
            or not _close(base, metrics["edge_base_by_family"][family])
            or not _close(current, metrics["edge_cur_by_family"][family])
            or not _close(violation["allowed"], base * (1.0 + epsilon))
            or not _close(violation["delta"], current / base - 1.0)
            or current <= violation["allowed"]
        ):
            raise ProbePayloadError("RMT epsilon violation arithmetic is invalid")
        family_epsilon = payload["epsilon_by_family"].get(
            family, payload["epsilon_default"]
        )
        if not _close(epsilon, family_epsilon):
            raise ProbePayloadError("RMT epsilon violation is not policy-bound")
    if observed_families != expected_families:
        raise ProbePayloadError("RMT epsilon violations do not match family aggregates")


def _validate_rmt_semantics(payload: dict[str, Any]) -> None:
    stable = payload["stable"]
    violations = payload["epsilon_violations"]
    metrics = payload["metrics"]
    if (
        payload["passed"] is not payload["stable_guard"]
        or metrics["stable"] is not payload["stable_guard"]
        or bool(violations) is stable
    ):
        raise ProbePayloadError("RMT probe status is inconsistent with its violations")
    expected_families = _expected_rmt_violation_families(payload)
    _validate_rmt_violations(payload, expected_families)
    if stable is bool(expected_families):
        raise ProbePayloadError("RMT stable status disagrees with family aggregates")


def _validate_ve_semantics(payload: dict[str, Any]) -> None:
    signal = payload["signal"]
    reasons = payload["signal_reasons"]
    if signal:
        if (
            reasons
            or payload["proposed_scales"] <= 0
            or payload["ab_gain"] is None
            or payload["ab_gain"] <= 0
            or payload["abs_improvement"] is None
            or payload["abs_improvement"] <= 0
        ):
            raise ProbePayloadError("VE positive signal lacks positive measured gain")
    elif not reasons:
        raise ProbePayloadError("VE absent signal must record a reason")
    gate = payload["predictive_gate"]
    if payload["would_enable"] is not gate["would_enable"]:
        raise ProbePayloadError("VE enable decision disagrees with its predictive gate")
    if gate["reason"] != payload["gate_reason"]:
        raise ProbePayloadError("VE gate reason disagrees with its predictive gate")
    no_ve, with_ve = payload["ppl_no_ve"], payload["ppl_with_ve"]
    if (no_ve is None) is not (with_ve is None):
        raise ProbePayloadError("VE perplexity measurements must be paired")
    if signal and (no_ve is None or with_ve is None):
        raise ProbePayloadError("VE positive signal lacks paired measurements")
    if no_ve is not None and with_ve is not None:
        if no_ve <= 0 or with_ve <= 0:
            raise ProbePayloadError("VE perplexities must be positive")
        if payload["abs_improvement"] is None or not _close(
            payload["abs_improvement"], no_ve - with_ve
        ):
            raise ProbePayloadError("VE absolute improvement arithmetic is invalid")
        if payload["ab_gain"] is not None and not _close(
            payload["ab_gain"], (no_ve - with_ve) / no_ve
        ):
            raise ProbePayloadError("VE relative gain arithmetic is invalid")
    elif payload["abs_improvement"] is not None:
        raise ProbePayloadError("VE absolute improvement lacks paired measurements")
    ratio = payload["ratio_ci"]
    if ratio is not None:
        if ratio[0] <= 0 or ratio[1] <= 0:
            raise ProbePayloadError("VE ratio interval must be positive")
        if ratio[0] > ratio[1]:
            raise ProbePayloadError("VE ratio interval is reversed")
    calibration = payload["calibration"]
    if calibration["min_coverage"] > calibration["windows"]:
        raise ProbePayloadError("VE minimum coverage exceeds calibration windows")


def validate_rmt_payload(payload: dict[str, Any], *, schema: str) -> None:
    violations = payload.get("violations")
    if (
        set(payload) != RMT_FIELDS
        or payload.get("schema") != schema
        or payload.get("probe") != "rmt_cross_model_v1"
        or any(
            not isinstance(payload.get(field), bool)
            for field in ("stable", "passed", "stable_guard")
        )
        or not isinstance(payload.get("action"), str)
        or not payload["action"]
        or not _finite(payload.get("epsilon_default"))
        or not _finite_mapping(payload.get("epsilon_by_family"))
        or not isinstance(payload.get("epsilon_violations"), list)
        or any(not _valid_rmt_violation(item) for item in payload["epsilon_violations"])
        or not isinstance(violations, list)
        or any(
            not isinstance(item, dict)
            or set(item) != {"code", "message"}
            or any(
                not isinstance(item.get(field), str) or not item[field]
                for field in ("code", "message")
            )
            for item in violations
        )
        or not _valid_rmt_metrics(payload.get("metrics"))
        or not valid_binding_shape(payload.get("binding"))
    ):
        raise ProbePayloadError("RMT probe field types are invalid")
    _validate_rmt_semantics(payload)


def validate_ve_payload(payload: dict[str, Any], *, schema: str) -> None:
    gate, calibration = payload.get("predictive_gate"), payload.get("calibration")
    optional_numbers = (
        payload.get("ppl_no_ve"),
        payload.get("ppl_with_ve"),
        payload.get("abs_improvement"),
        payload.get("ab_gain"),
    )
    ratio = payload.get("ratio_ci")
    if (
        set(payload) != VE_FIELDS
        or payload.get("schema") != schema
        or payload.get("probe") != "ve_probe_v1"
        or any(
            not isinstance(payload.get(field), bool)
            for field in ("signal", "would_enable")
        )
        or not isinstance(payload.get("signal_reasons"), list)
        or any(not isinstance(item, str) for item in payload["signal_reasons"])
        or not isinstance(payload.get("gate_reason"), str)
        or isinstance(payload.get("proposed_scales"), bool)
        or not isinstance(payload.get("proposed_scales"), int)
        or payload["proposed_scales"] < 0
        or any(value is not None and not _finite(value) for value in optional_numbers)
        or (
            ratio is not None
            and (
                not isinstance(ratio, list)
                or len(ratio) != 2
                or any(not _finite(value) for value in ratio)
            )
        )
        or not isinstance(gate, dict)
        or set(gate) != {"would_enable", "reason"}
        or not isinstance(gate.get("would_enable"), bool)
        or not isinstance(gate.get("reason"), str)
        or not isinstance(calibration, dict)
        or set(calibration) != {"windows", "min_coverage", "tier", "profile"}
        or any(
            isinstance(calibration.get(field), bool)
            or not isinstance(calibration.get(field), int)
            or calibration[field] < 0
            for field in ("windows", "min_coverage")
        )
        or any(
            not isinstance(calibration.get(field), str) or not calibration[field]
            for field in ("tier", "profile")
        )
        or not valid_binding_shape(payload.get("binding"))
    ):
        raise ProbePayloadError("VE probe field types are invalid")
    _validate_ve_semantics(payload)
