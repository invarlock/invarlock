"""Strict runtime integration execution for one paired-evaluation side.

This is intentionally an evidence primitive, not a policy engine.  It executes
one authenticated provider context, persists the typed provider sidecars, and
binds them to the canonical schedule, policy input digest, and outer image.
Comparison thresholds remain solely in the canonical evidence transaction.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAliasType

from invarlock.core.runtime_provider import (
    ModelRuntimeSpec,
    RuntimeBehavioralSchedule,
    RuntimeExecutionContext,
    RuntimeMetric,
    RuntimeProvider,
    RuntimeSession,
    artifact_identity_sha256,
    load_runtime_behavioral_schedule,
)
from invarlock.filesystem.atomic_directory import (
    AtomicDirectoryExistsError,
    AtomicDirectoryPublicationError,
    publish_directory_no_replace,
)
from invarlock.runtime_manifest import write_runtime_manifest
from invarlock.runtime_provider_evidence import (
    PersistedRuntimeProviderEvidence,
    RuntimeProviderEvidencePaths,
    load_runtime_provider_evidence,
    write_runtime_provider_evidence,
)
from invarlock.runtime_security_helpers import (
    RuntimeManifestExecution,
    RuntimeProviderManifestFiles,
    network_allowed,
    remote_code_allowed,
    resolve_runtime_image,
    resolve_runtime_image_digest,
    strict_container_boundary_present,
    third_party_plugins_allowed,
)
from invarlock.runtime_verify import verify_runtime_manifest_snapshot

RuntimeSideRole = TypeAliasType(  # noqa: UP040
    "RuntimeSideRole", Literal["baseline", "subject"]
)


class RuntimeEvidenceError(ValueError):
    """Raised when strict runtime side evidence cannot be produced."""


_RUNTIME_CLEANUP_FAILURE_NOTE = "runtime provider session cleanup also failed"


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _resolved_output(output_directory: Path) -> tuple[Path, Path]:
    output = Path(output_directory)
    if output.name in {"", ".", ".."}:
        raise RuntimeEvidenceError("runtime evidence output must name a directory")
    try:
        parent = output.parent.resolve(strict=True)
    except OSError as exc:
        raise RuntimeEvidenceError(
            "runtime evidence output parent must be a real directory"
        ) from exc
    if not parent.is_dir():
        raise RuntimeEvidenceError(
            "runtime evidence output parent must be a real directory"
        )
    destination = parent / output.name
    if destination.exists() or destination.is_symlink():
        raise RuntimeEvidenceError("runtime evidence output already exists")
    return destination, parent


_REPORT_FILENAME = "report.json"
_CONFIG_FILENAME = "run.yaml"


def _report_payload(
    *,
    provider_name: str,
    artifact_sha256: str,
    schedule_sha256: str,
    scoring_observation_sha256: str,
    record_count: int,
) -> dict[str, object]:
    return {
        "format": "invarlock/runtime-side-report-v1",
        "provider": provider_name,
        "artifact_identity_sha256": artifact_sha256,
        "scoring_observation_sha256": scoring_observation_sha256,
        "schedule_sha256": schedule_sha256,
        "record_count": record_count,
    }


@dataclass(frozen=True)
class RuntimeEvidenceSideBundle:
    """Exact files emitted for one strictly executed comparison side."""

    role: RuntimeSideRole
    directory: Path
    report_path: Path
    config_path: Path
    manifest_path: Path
    evidence: PersistedRuntimeProviderEvidence


def _image_ref_matches_digest(image_ref: object, image_digest: str) -> bool:
    if image_ref == image_digest:
        return True
    if not isinstance(image_ref, str):
        return False
    repository, separator, digest = image_ref.rpartition("@")
    return bool(repository and separator and digest == image_digest)


def _observed_execution(context: RuntimeExecutionContext) -> RuntimeManifestExecution:
    if not context.strict or context.allow_network:
        raise RuntimeEvidenceError("runtime evidence requires strict offline execution")
    allow_network = network_allowed()
    if allow_network:
        raise RuntimeEvidenceError(
            "runtime evidence requires network access to be disabled"
        )
    allow_remote_code = remote_code_allowed()
    if allow_remote_code:
        raise RuntimeEvidenceError(
            "runtime evidence requires remote code loading to be disabled"
        )
    allow_third_party_plugins = third_party_plugins_allowed()
    if allow_third_party_plugins:
        raise RuntimeEvidenceError(
            "runtime evidence requires third-party provider discovery to be disabled"
        )
    if not strict_container_boundary_present():
        raise RuntimeEvidenceError(
            "runtime evidence requires an actual container execution boundary"
        )
    image_digest = context.container_image_digest
    if image_digest is None:
        raise RuntimeEvidenceError("runtime evidence requires a pinned outer image")
    observed_digest = resolve_runtime_image_digest()
    image_ref = resolve_runtime_image()
    if observed_digest != image_digest:
        raise RuntimeEvidenceError(
            "observed container image digest does not match runtime resources"
        )
    if not _image_ref_matches_digest(image_ref, image_digest):
        raise RuntimeEvidenceError(
            "runtime image reference must embed the observed image digest"
        )
    return RuntimeManifestExecution(
        execution_mode="container",
        container_execution=True,
        image_ref=image_ref,
        image_digest=image_digest,
        allow_network=allow_network,
        allow_remote_code=allow_remote_code,
        allow_third_party_plugins=allow_third_party_plugins,
    )


def _validate_observation_bindings(
    *,
    schedule,
    provider_name: str,
    artifact_sha256: str,
    observation,
) -> None:
    if observation.provider_name != provider_name:
        raise RuntimeEvidenceError("scoring observation provider is invalid")
    if observation.artifact_identity_sha256 != artifact_sha256:
        raise RuntimeEvidenceError("scoring observation artifact binding is invalid")
    if observation.schedule_sha256 != schedule.schedule_sha256:
        raise RuntimeEvidenceError("scoring observation schedule binding is invalid")
    expected = tuple(
        (record.record_id, record.input_sha256) for record in schedule.records
    )
    observed = tuple(
        (record.record_id, record.input_sha256) for record in observation.records
    )
    if observed != expected:
        raise RuntimeEvidenceError(
            "scoring observation order does not match the canonical schedule"
        )


def run_evidence_side(
    *,
    role: RuntimeSideRole,
    provider: RuntimeProvider,
    spec: ModelRuntimeSpec,
    context: RuntimeExecutionContext,
    schedule_path: Path,
    policy_digest: str,
    output_directory: Path,
    metric: RuntimeMetric = "exact_match",
    _validated_schedule: RuntimeBehavioralSchedule | None = None,
) -> RuntimeEvidenceSideBundle:
    """Execute, authenticate, and atomically publish one side evidence bundle."""

    if role not in {"baseline", "subject"}:
        raise RuntimeEvidenceError("role must be baseline or subject")
    if not isinstance(spec, ModelRuntimeSpec):
        raise TypeError("spec must be ModelRuntimeSpec")
    if _validated_schedule is not None and not isinstance(
        _validated_schedule, RuntimeBehavioralSchedule
    ):
        raise TypeError("_validated_schedule must be RuntimeBehavioralSchedule")
    if (
        not policy_digest.startswith("sha256:")
        or len(policy_digest) != 71
        or any(character not in "0123456789abcdef" for character in policy_digest[7:])
    ):
        raise RuntimeEvidenceError("policy_digest must be a sha256 digest")
    execution = _observed_execution(context)
    schedule = (
        load_runtime_behavioral_schedule(Path(schedule_path))
        if _validated_schedule is None
        else _validated_schedule
    )
    provider.validate_config(spec)
    if provider.name != spec.provider_name:
        raise RuntimeEvidenceError("provider and model spec do not agree")
    if spec.settings.get("batch_size") != 1:
        raise RuntimeEvidenceError("strict paired execution requires batch_size=1")
    capabilities = provider.capabilities()
    if capabilities.provider_name != provider.name:
        raise RuntimeEvidenceError("provider capabilities identity is invalid")
    if metric not in capabilities.metrics:
        raise RuntimeEvidenceError(
            f"provider {provider.name!r} does not support metric {metric!r}"
        )
    artifact_identity = provider.identify_artifact(spec)
    artifact_sha256 = artifact_identity_sha256(artifact_identity)
    if context.artifact_identity_sha256 != artifact_sha256:
        raise RuntimeEvidenceError(
            "runtime context artifact identity does not match the model spec"
        )
    image_digest = context.container_image_digest
    assert image_digest is not None

    session: RuntimeSession | None = None
    primary_error: BaseException | None = None
    try:
        session = provider.open(spec, context)
        observation = session.score(schedule.evaluation_batch(metric))
        receipt = session.runtime_receipt()
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        try:
            if session is not None:
                session.close()
            elif context.close_callback is not None:
                context.close_callback()
        except Exception:
            if primary_error is None:
                raise
            primary_error.add_note(_RUNTIME_CLEANUP_FAILURE_NOTE)
    _validate_observation_bindings(
        schedule=schedule,
        provider_name=provider.name,
        artifact_sha256=artifact_sha256,
        observation=observation,
    )
    if receipt.capabilities != capabilities:
        raise RuntimeEvidenceError(
            "runtime receipt capabilities do not match the prepared provider"
        )
    if receipt.execution_settings.allow_network:
        raise RuntimeEvidenceError("runtime receipt must bind offline execution")
    for field_name in (
        "seed",
        "context_length",
        "batch_size",
        "max_output_tokens",
        "timeout_seconds",
    ):
        if field_name in spec.settings and spec.settings[field_name] != getattr(
            receipt.execution_settings, field_name
        ):
            raise RuntimeEvidenceError(
                f"runtime receipt does not match setting {field_name!r}"
            )
    if receipt.device.device_kind != context.device_kind:
        raise RuntimeEvidenceError(
            "runtime receipt device does not match caller-owned resources"
        )

    output, parent = _resolved_output(output_directory)
    staging = Path(tempfile.mkdtemp(dir=parent, prefix=f".{output.name}.staging."))
    try:
        evidence = write_runtime_provider_evidence(
            staging,
            artifact_identity=artifact_identity,
            scoring_observation=observation,
            receipt=receipt,
            expected_outer_image_digest=image_digest,
        )
        report_path = staging / _REPORT_FILENAME
        report_path.write_bytes(
            _canonical_json_bytes(
                _report_payload(
                    provider_name=provider.name,
                    artifact_sha256=artifact_sha256,
                    schedule_sha256=schedule.schedule_sha256,
                    scoring_observation_sha256=evidence.scoring_observation_sha256,
                    record_count=len(observation.records),
                )
            )
        )
        config_path = staging / _CONFIG_FILENAME
        config_path.write_bytes(
            _canonical_json_bytes(
                {
                    "format": "invarlock/runtime-side-config-v1",
                    "role": role,
                    "provider": provider.name,
                    "artifact_identity_sha256": artifact_sha256,
                    "schedule_sha256": schedule.schedule_sha256,
                    "policy_digest": policy_digest,
                }
            )
        )
        manifest_path = write_runtime_manifest(
            report_path,
            provider_files=RuntimeProviderManifestFiles(
                receipt=evidence.paths.receipt,
                scoring_observation=evidence.paths.scoring_observation,
                artifact_identity=evidence.paths.artifact_identity,
            ),
            config_path=config_path,
            execution=execution,
        )
        manifest = json.loads(manifest_path.read_bytes())
        verification = verify_runtime_manifest_snapshot(
            report_path.read_bytes(),
            manifest,
            report=report_path,
            manifest=manifest_path,
            expected_image_digest=image_digest,
            require_strict_runtime=True,
        )
        if verification.errors:
            raise RuntimeEvidenceError(
                "runtime evidence manifest is invalid: "
                + "; ".join(verification.errors)
            )
        publish_directory_no_replace(staging, output)
    except AtomicDirectoryExistsError as exc:
        raise RuntimeEvidenceError(
            "could not publish runtime evidence without clobber"
        ) from exc
    except AtomicDirectoryPublicationError as exc:
        raise RuntimeEvidenceError(
            f"runtime evidence atomic publication failed: {exc}"
        ) from exc
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)

    report_path = output / _REPORT_FILENAME
    config_path = output / _CONFIG_FILENAME
    manifest_path = output / "runtime.manifest.json"
    evidence = load_runtime_provider_evidence(
        RuntimeProviderEvidencePaths.in_directory(output),
        expected_outer_image_digest=image_digest,
    )
    return RuntimeEvidenceSideBundle(
        role=role,
        directory=output,
        report_path=report_path,
        config_path=config_path,
        manifest_path=manifest_path,
        evidence=evidence,
    )


__all__ = [
    "RuntimeEvidenceError",
    "RuntimeEvidenceSideBundle",
    "RuntimeSideRole",
    "run_evidence_side",
]
