"""Closed internal protocol for executing one evaluation side in its runtime image.

This module is intentionally not registered as a public CLI command.  The host
transaction creates the canonical schedule and invokes this worker with a small,
canonical JSON job.  A worker can emit runtime-side evidence, but it never sees
the evidence-signing key and cannot publish the comparison evidence pack.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, cast

from invarlock.core.registry import CoreRegistry
from invarlock.core.runtime_provider import (
    ModelRuntimeSpec,
    RuntimeArtifactResources,
    load_runtime_behavioral_schedule,
    validate_runtime_evaluation_inputs,
)
from invarlock.evidence_pack_contract import canonical_json_bytes
from invarlock.evidence_pack_json import parse_json_bytes, read_regular_file_bytes
from invarlock.runtime_behavior.transaction import run_evidence_side

_JOB_FORMAT = "invarlock/runtime-side-job-v1"
_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_JOB_KEYS = {
    "format_version",
    "role",
    "provider",
    "model_id",
    "settings",
    "metric",
    "policy_digest",
    "resource_root",
    "primary_artifact",
    "support_resources",
    "device_kind",
    "image_digest",
    "schedule",
    "output",
}


class RuntimeSideWorkerError(ValueError):
    """Raised when an internal worker job is unsafe or malformed."""


def _closed_absolute_path(value: object, *, expected: str, label: str) -> Path:
    if not isinstance(value, str) or not value.startswith("/") or "\x00" in value:
        raise RuntimeSideWorkerError(f"{label} must be an absolute path")
    path = Path(value)
    if any(part in {"", ".", ".."} for part in path.parts[1:]):
        raise RuntimeSideWorkerError(f"{label} is not canonical")
    try:
        resolved = path.resolve(strict=True) if expected != "output" else path
    except OSError as exc:
        raise RuntimeSideWorkerError(f"{label} is unavailable") from exc
    if expected == "file" and not resolved.is_file():
        raise RuntimeSideWorkerError(f"{label} must be a regular file")
    if expected == "directory" and not resolved.is_dir():
        raise RuntimeSideWorkerError(f"{label} must be a directory")
    if expected == "output":
        parent = path.parent.resolve(strict=True)
        if not parent.is_dir() or path.exists() or path.is_symlink():
            raise RuntimeSideWorkerError(f"{label} must name a new directory")
        return parent / path.name
    return resolved


def _string_mapping(value: object, *, label: str) -> MappingProxyType[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise RuntimeSideWorkerError(f"{label} must be an object with string keys")
    return MappingProxyType(dict(value))


def execute_job(job_path: Path) -> Path:
    """Execute one canonical side job and return its published bundle path."""

    payload = read_regular_file_bytes(
        Path(job_path), label="runtime side job", max_bytes=1024 * 1024
    )
    value = parse_json_bytes(payload, label="runtime side job")
    if not isinstance(value, dict) or set(value) != _JOB_KEYS:
        raise RuntimeSideWorkerError("runtime side job has unexpected fields")
    if payload != canonical_json_bytes(value):
        raise RuntimeSideWorkerError("runtime side job must use canonical JSON")
    if value["format_version"] != _JOB_FORMAT:
        raise RuntimeSideWorkerError("runtime side job format is unsupported")
    role = value["role"]
    if role not in {"baseline", "subject"}:
        raise RuntimeSideWorkerError("runtime side job role is invalid")
    provider_name = value["provider"]
    model_id = value["model_id"]
    metric = value["metric"]
    policy_digest = value["policy_digest"]
    image_digest = value["image_digest"]
    device_kind = value["device_kind"]
    primary_artifact = value["primary_artifact"]
    if not all(
        isinstance(item, str)
        for item in (
            provider_name,
            model_id,
            metric,
            policy_digest,
            image_digest,
            device_kind,
            primary_artifact,
        )
    ):
        raise RuntimeSideWorkerError("runtime side job contains non-string bindings")
    if _DIGEST_RE.fullmatch(cast(str, policy_digest)) is None:
        raise RuntimeSideWorkerError("runtime side job policy digest is invalid")
    if _DIGEST_RE.fullmatch(cast(str, image_digest)) is None:
        raise RuntimeSideWorkerError("runtime side job image digest is invalid")
    if device_kind not in {"cpu", "cuda"}:
        raise RuntimeSideWorkerError("runtime side job device must be cpu or cuda")

    resource_root = _closed_absolute_path(
        value["resource_root"], expected="directory", label="resource root"
    )
    schedule = _closed_absolute_path(
        value["schedule"], expected="file", label="canonical schedule"
    )
    output = _closed_absolute_path(
        value["output"], expected="output", label="runtime side output"
    )
    settings = _string_mapping(value["settings"], label="runtime settings")
    support = _string_mapping(value["support_resources"], label="support resources")
    if any(not isinstance(item, str) for item in support.values()):
        raise RuntimeSideWorkerError("support resource values must be strings")

    registry = CoreRegistry()
    provider = registry.get_runtime_provider(cast(str, provider_name))
    spec = ModelRuntimeSpec(
        provider_name=cast(str, provider_name),
        model_id=cast(str, model_id),
        settings=settings,
    )
    resources = RuntimeArtifactResources(
        root=resource_root,
        primary_artifact=cast(str, primary_artifact),
        support_resources=cast(dict[str, str], dict(support)),
        device_kind=cast(Literal["cpu", "cuda"], device_kind),
        container_image_digest=cast(str, image_digest),
    )
    try:
        validated_schedule = load_runtime_behavioral_schedule(schedule)
        validate_runtime_evaluation_inputs(
            provider, spec, resources, validated_schedule
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise RuntimeSideWorkerError(
            f"runtime provider input preflight failed: {exc}"
        ) from exc
    context = provider.prepare_execution(spec, resources)
    bundle = run_evidence_side(
        role=cast(Literal["baseline", "subject"], role),
        provider=provider,
        spec=spec,
        context=context,
        schedule_path=schedule,
        policy_digest=cast(str, policy_digest),
        output_directory=output,
        metric=cast(Any, metric),
        _validated_schedule=validated_schedule,
    )
    return bundle.directory


def main(argv: list[str] | None = None) -> int:
    """Run the internal worker without exposing it through the public CLI."""

    arguments = sys.argv[1:] if argv is None else argv
    if len(arguments) != 1:
        print("runtime side worker requires exactly one job path", file=sys.stderr)
        return 2
    if any(
        os.environ.get(variable) is not None
        for variable in ("INVARLOCK_SIGNING_KEY", "INVARLOCK_EVIDENCE_SIGNING_KEY")
    ):
        print(
            "runtime side worker refuses evidence-signing key material", file=sys.stderr
        )
        return 2
    try:
        execute_job(Path(arguments[0]))
    except Exception as exc:  # closed process boundary converts all failures
        print(str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover - subprocess entry point
    raise SystemExit(main())


__all__ = ["RuntimeSideWorkerError", "execute_job", "main"]
