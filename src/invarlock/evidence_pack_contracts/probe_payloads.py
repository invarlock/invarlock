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
