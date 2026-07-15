"""Strict public contracts for verdict-driving cross-model probe sidecars."""

from __future__ import annotations

import hashlib
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

from invarlock.evidence_pack_contracts.probe_payloads import (
    ProbePayloadError,
    validate_rmt_payload,
    validate_ve_payload,
)
from invarlock.evidence_pack_json import (
    StrictJsonError,
    read_json_object_snapshot,
)

RMT_PROBE_SCHEMA = "invarlock/rmt-probe-v1"
VE_PROBE_SCHEMA = "invarlock/ve-probe-v1"
PROBE_FILENAMES = ("rmt_probe.json", "ve_probe.json")

_WINDOWS_DRIVE = re.compile(r"^[A-Za-z]:[\\/]")


class ProbeValidationError(ValueError):
    """A public probe is malformed, ambiguous, or unsafe to publish."""


def _strings(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [
            text
            for key, item in value.items()
            for text in [*_strings(key), *_strings(item)]
        ]
    if isinstance(value, list):
        return [text for item in value for text in _strings(item)]
    return []


def _reject_host_paths(payload: dict[str, Any]) -> None:
    for value in _strings(payload):
        if value.startswith(("/", "~/", "~\\")) or _WINDOWS_DRIVE.match(value):
            raise ProbeValidationError("probe contains a host-local path")


def build_probe_binding(report: object, report_sha256: str) -> dict[str, Any]:
    """Derive the one closed identity record accepted for a probe sidecar."""

    if not isinstance(report_sha256, str) or not re.fullmatch(
        r"sha256:[a-f0-9]{64}", report_sha256
    ):
        raise ProbeValidationError("canonical report digest is invalid")
    if not isinstance(report, dict) or not isinstance(report.get("run_id"), str):
        raise ProbeValidationError("canonical report lacks a run identity")
    meta = report.get("meta")
    context = report.get("context")
    provenance = report.get("provenance")
    runtime = context.get("runtime") if isinstance(context, dict) else None
    provider = (
        provenance.get("provider_digest") if isinstance(provenance, dict) else None
    )
    model_id = meta.get("model_id") if isinstance(meta, dict) else None
    adapter = meta.get("adapter") if isinstance(meta, dict) else None
    profile = meta.get("profile") if isinstance(meta, dict) else None
    execution_mode = (
        runtime.get("execution_mode") if isinstance(runtime, dict) else None
    )
    if (
        not isinstance(model_id, str)
        or not model_id
        or not isinstance(adapter, str)
        or not adapter
        or not isinstance(profile, str)
        or not profile
        or not isinstance(execution_mode, str)
        or not execution_mode
        or not isinstance(provider, dict)
        or not provider
        or not all(
            isinstance(key, str) and key and isinstance(value, str) and value
            for key, value in provider.items()
        )
    ):
        raise ProbeValidationError("canonical report lacks probe identity coordinates")
    return {
        "report_sha256": report_sha256,
        "run_id": report["run_id"],
        "model_id": model_id,
        "runtime": {"execution_mode": execution_mode},
        "toolchain": {"adapter": adapter, "profile": profile},
        "provider_digest": deepcopy(provider),
    }


def validate_probe_binding(binding: object, report: object, report_sha256: str) -> None:
    if binding != build_probe_binding(report, report_sha256):
        raise ProbeValidationError("probe binding does not match the canonical report")


def validate_probe_payload(filename: str, payload: object) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ProbeValidationError("probe must be a JSON object")
    if filename not in PROBE_FILENAMES:
        raise ProbeValidationError("probe has missing or unsupported fields")
    _reject_host_paths(payload)
    try:
        if filename == "rmt_probe.json":
            validate_rmt_payload(payload, schema=RMT_PROBE_SCHEMA)
        else:
            validate_ve_payload(payload, schema=VE_PROBE_SCHEMA)
    except ProbePayloadError as exc:
        raise ProbeValidationError(str(exc)) from exc
    return payload


def load_probe_snapshot(
    path: Path, *, report_path: Path | None = None
) -> tuple[bytes, dict[str, Any]]:
    try:
        raw, payload = read_json_object_snapshot(
            path,
            label=f"probe sidecar {path.name}",
        )
    except StrictJsonError as exc:
        raise ProbeValidationError(str(exc)) from exc
    probe = validate_probe_payload(path.name, payload)
    report_path = report_path or path.parent / "evaluation.report.json"
    try:
        report_raw, report = read_json_object_snapshot(
            report_path,
            label=f"canonical report for probe sidecar {path.name}",
        )
    except StrictJsonError as exc:
        raise ProbeValidationError(str(exc)) from exc
    validate_probe_binding(
        probe.get("binding"),
        report,
        "sha256:" + hashlib.sha256(report_raw).hexdigest(),
    )
    return raw, probe


def load_probe_file(path: Path, *, report_path: Path | None = None) -> dict[str, Any]:
    return load_probe_snapshot(path, report_path=report_path)[1]


__all__ = [
    "PROBE_FILENAMES",
    "ProbeValidationError",
    "RMT_PROBE_SCHEMA",
    "VE_PROBE_SCHEMA",
    "build_probe_binding",
    "load_probe_file",
    "load_probe_snapshot",
    "validate_probe_payload",
    "validate_probe_binding",
]
