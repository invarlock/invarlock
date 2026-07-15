"""Bind reporting policy provenance to the policy used by the runtime."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from invarlock.guards.authority import (
    DEFAULT_GUARD_AUTHORITY,
    guard_authority_errors,
)

RUNTIME_POLICY_RECEIPT_FORMAT = "invarlock.runtime-policy-receipt.v1"
_RUNTIME_GUARDS = frozenset({"spectral", "rmt", "variance"})
_RECEIPT_FIELDS = frozenset(
    {
        "format_version",
        "source",
        "tier",
        "profile",
        "edit_name",
        "guard_policies",
        "resolved_policy_sha256",
    }
)


def _canonical_digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _guard_policies(
    guard_entries: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    observed: dict[str, dict[str, Any]] = {}
    for entry in guard_entries:
        name = str(entry.get("name") or "").strip().lower()
        if name not in _RUNTIME_GUARDS:
            continue
        raw_policy = entry.get("policy")
        if not isinstance(raw_policy, Mapping) or not raw_policy:
            raise ValueError(
                f"runtime guard {name!r} did not retain its applied policy"
            )
        policy = copy.deepcopy(dict(raw_policy))
        raw_metrics = entry.get("metrics")
        measurement_contract = (
            raw_metrics.get("measurement_contract")
            if isinstance(raw_metrics, Mapping)
            else None
        )
        if isinstance(measurement_contract, Mapping) and measurement_contract:
            policy["measurement_contract"] = copy.deepcopy(dict(measurement_contract))
        previous = observed.get(name)
        if previous is not None and previous != policy:
            raise ValueError(
                f"runtime guard {name!r} emitted inconsistent applied policies"
            )
        observed[name] = policy
    return observed


def build_runtime_policy_receipt(
    runtime_policies: Mapping[str, Any],
    guard_entries: Sequence[Mapping[str, Any]],
    *,
    tier: str,
    profile: str,
    edit_name: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Create an exact effective-policy snapshot at the runtime/report boundary."""

    if not runtime_policies:
        raise ValueError("runtime resolved policy is empty")
    normalized_tier = str(tier or "").strip().lower()
    normalized_profile = str(profile or "").strip().lower()
    normalized_edit = str(edit_name or "").strip()
    if normalized_tier not in {"aggressive", "balanced", "conservative"}:
        raise ValueError("runtime policy receipt tier is unsupported")
    if normalized_profile not in {"dev", "ci", "release"}:
        raise ValueError("runtime policy receipt profile is unsupported")
    if not normalized_edit:
        raise ValueError("runtime policy receipt edit_name is required")
    resolved = copy.deepcopy(dict(runtime_policies))
    resolved.setdefault("guard_authority", copy.deepcopy(DEFAULT_GUARD_AUTHORITY))
    authority_errors = guard_authority_errors(
        resolved.get("guard_authority"),
        path="runtime resolved_policy.guard_authority",
    )
    if authority_errors:
        raise ValueError("; ".join(authority_errors))
    applied = _guard_policies(guard_entries)
    for name, policy in applied.items():
        base_policy = resolved.get(name)
        merged = (
            copy.deepcopy(dict(base_policy)) if isinstance(base_policy, Mapping) else {}
        )
        merged.update(policy)
        resolved[name] = merged
    receipt = {
        "format_version": RUNTIME_POLICY_RECEIPT_FORMAT,
        "source": "runtime",
        "tier": normalized_tier,
        "profile": normalized_profile,
        "edit_name": normalized_edit,
        "guard_policies": sorted(applied),
        "resolved_policy_sha256": _canonical_digest(resolved),
    }
    return resolved, receipt


def runtime_policy_from_report(
    report: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, list[str]]:
    """Load and reconcile a runtime policy receipt from a canonical run report."""

    resolution = report.get("policy_resolution")
    if resolution is None:
        return None, []
    errors: list[str] = []
    if not isinstance(resolution, Mapping):
        return None, ["policy_resolution must be an object"]
    if set(resolution) != _RECEIPT_FIELDS:
        errors.append("policy_resolution fields do not match the current schema")
    if resolution.get("format_version") != RUNTIME_POLICY_RECEIPT_FORMAT:
        errors.append("policy_resolution has an unknown format_version")
    if resolution.get("source") != "runtime":
        errors.append("policy_resolution source must be runtime")
    if resolution.get("profile") not in {"dev", "ci", "release"}:
        errors.append("policy_resolution profile is unsupported")
    if resolution.get("tier") not in {"aggressive", "balanced", "conservative"}:
        errors.append("policy_resolution tier is unsupported")
    edit_name = resolution.get("edit_name")
    if not isinstance(edit_name, str) or not edit_name.strip():
        errors.append("policy_resolution edit_name must be a non-empty string")
    guard_policies = resolution.get("guard_policies")
    if not isinstance(guard_policies, list) or not all(
        isinstance(name, str) and name in _RUNTIME_GUARDS for name in guard_policies
    ):
        errors.append("policy_resolution guard_policies is malformed")
    raw_policy = report.get("resolved_policy")
    if not isinstance(raw_policy, Mapping) or not raw_policy:
        return None, [*errors, "runtime policy receipt requires resolved_policy"]
    resolved = copy.deepcopy(dict(raw_policy))
    if resolution.get("resolved_policy_sha256") != _canonical_digest(resolved):
        errors.append("runtime policy receipt digest does not match resolved_policy")
    raw_guards = report.get("guards")
    if not isinstance(raw_guards, list):
        errors.append("runtime policy receipt requires canonical guard entries")
        return resolved, errors
    try:
        observed = _guard_policies(
            [entry for entry in raw_guards if isinstance(entry, Mapping)]
        )
    except ValueError as exc:
        errors.append(str(exc))
        return resolved, errors
    recorded_guards = resolution.get("guard_policies")
    if recorded_guards != sorted(observed):
        errors.append("runtime policy receipt guard inventory does not match report")
    for name, policy in observed.items():
        retained = resolved.get(name)
        if not isinstance(retained, Mapping) or any(
            retained.get(key) != value for key, value in policy.items()
        ):
            errors.append(
                f"runtime policy receipt disagrees with applied {name!r} policy"
            )
    return resolved, errors


__all__ = [
    "RUNTIME_POLICY_RECEIPT_FORMAT",
    "build_runtime_policy_receipt",
    "runtime_policy_from_report",
]
