"""Typed authoring helpers for complete runtime-import evidence.

This module accepts per-record backend facts, never provider-supplied aggregates.
It materializes the same closed side evidence used by native execution, reloads it
through the independent runtime verifier, and derives paired records from those
authenticated side files.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

from invarlock.core.runtime_provider import (
    ModelArtifactIdentity,
    RuntimeBackendIdentity,
    RuntimeBehavioralSchedule,
    RuntimeDeviceFacts,
    RuntimeExecutionSettings,
    RuntimeProviderCapabilities,
    RuntimeProviderPluginIdentity,
    RuntimeProviderReceipt,
    RuntimeScoringRecord,
    ScoringObservation,
    artifact_identity_sha256,
    runtime_scoring_records_sha256,
)
from invarlock.core.scorer_extension import (
    ScorerExtensionBinding,
    ScorerExtensionRegistry,
)
from invarlock.evidence_pack_contract import (
    RuntimeSideEvidence,
    canonical_json_bytes,
    derive_paired_records,
    sha256_digest,
)
from invarlock.evidence_pack_json import (
    StrictJsonError,
    parse_json_bytes,
    read_regular_file_bytes,
)
from invarlock.filesystem.atomic_directory import (
    AtomicDirectoryPublicationError,
    publish_directory_no_replace,
)
from invarlock.runtime_manifest import write_runtime_manifest
from invarlock.runtime_provider_evidence import (
    MAX_RUNTIME_PROVIDER_SIDECAR_BYTES,
    PersistedRuntimeProviderEvidence,
    RuntimeProviderEvidenceError,
    RuntimeProviderEvidencePaths,
    encode_scoring_observation,
    load_runtime_provider_evidence,
    write_runtime_provider_evidence,
)
from invarlock.runtime_security_helpers import (
    RuntimeManifestExecution,
    RuntimeProviderManifestFiles,
)
from invarlock.runtime_verify import verify_runtime_manifest_snapshot

RUNTIME_IMPORT_REPORT_FILENAME = "report.json"
RUNTIME_IMPORT_CONFIG_FILENAME = "run.yaml"
RUNTIME_IMPORT_MANIFEST_FILENAME = "runtime.manifest.json"
MAX_EXTERNAL_RECORDS = 10_000

_RECORD_FIELDS = frozenset(
    {
        "record_id",
        "input_sha256",
        "status",
        "output_text",
        "output_sha256",
        "logprob_sum",
        "token_count",
        "utf8_byte_count",
        "error_code",
    }
)
_REQUIRED_RECORD_FIELDS = frozenset({"record_id", "input_sha256", "status"})


class RuntimeImportAuthoringError(ValueError):
    """Raised when external per-record facts cannot form strict import evidence."""


@dataclass(frozen=True)
class RuntimeImportSideEvidence:
    """One complete, independently reloaded runtime-import side directory."""

    role: Literal["baseline", "subject"]
    directory: Path
    report_path: Path
    config_path: Path
    manifest_path: Path
    provider_evidence: PersistedRuntimeProviderEvidence
    runtime_image_digest: str
    side_evidence: RuntimeSideEvidence

    def evidence_pack_value(self) -> RuntimeSideEvidence:
        """Return exact bytes for verifier-owned pair derivation and publication."""

        return self.side_evidence


@dataclass(frozen=True)
class RuntimeImportPairedRecords:
    """Canonical paired-record artifact derived from two verified sides."""

    path: Path
    payload: Mapping[str, object]
    sha256: str


def _record_from_object(
    value: Mapping[str, object], *, line_number: int
) -> RuntimeScoringRecord:
    fields = set(value)
    missing = sorted(_REQUIRED_RECORD_FIELDS - fields)
    if missing:
        raise RuntimeImportAuthoringError(
            f"external record line {line_number} is missing {missing[0]!r}"
        )
    unexpected = sorted(fields - _RECORD_FIELDS)
    if unexpected:
        raise RuntimeImportAuthoringError(
            f"external record line {line_number} has unsupported field "
            f"{unexpected[0]!r}; aggregate summaries are not accepted"
        )
    try:
        return RuntimeScoringRecord(
            record_id=cast(str, value["record_id"]),
            input_sha256=cast(str, value["input_sha256"]),
            status=cast(Any, value["status"]),
            output_text=cast(str | None, value.get("output_text")),
            output_sha256=cast(str | None, value.get("output_sha256")),
            logprob_sum=cast(float | None, value.get("logprob_sum")),
            token_count=cast(int | None, value.get("token_count")),
            utf8_byte_count=cast(int | None, value.get("utf8_byte_count")),
            error_code=cast(str | None, value.get("error_code")),
        )
    except (TypeError, ValueError) as exc:
        raise RuntimeImportAuthoringError(
            f"external record line {line_number} is invalid: {exc}"
        ) from exc


def load_external_scoring_records_jsonl(
    path: str | Path,
    *,
    schedule: RuntimeBehavioralSchedule,
) -> tuple[RuntimeScoringRecord, ...]:
    """Load strict per-record JSONL and bind it exactly to a schedule.

    Each line represents one backend observation. Objects containing aggregate or
    unknown fields fail closed; callers cannot import a score summary in place of
    replayable record facts.
    """

    try:
        payload = read_regular_file_bytes(
            Path(path),
            label="external scoring records",
            max_bytes=MAX_RUNTIME_PROVIDER_SIDECAR_BYTES,
        )
    except StrictJsonError as exc:
        raise RuntimeImportAuthoringError(str(exc)) from exc
    if not payload or not payload.endswith(b"\n"):
        raise RuntimeImportAuthoringError(
            "external scoring records must be non-empty newline-terminated JSONL"
        )
    lines = payload.splitlines()
    if len(lines) > MAX_EXTERNAL_RECORDS:
        raise RuntimeImportAuthoringError(
            f"external scoring records exceed the {MAX_EXTERNAL_RECORDS}-record limit"
        )
    records: list[RuntimeScoringRecord] = []
    for line_number, line in enumerate(lines, start=1):
        if not line:
            raise RuntimeImportAuthoringError(
                f"external record line {line_number} must not be blank"
            )
        try:
            value = parse_json_bytes(line, label=f"external record line {line_number}")
        except StrictJsonError as exc:
            raise RuntimeImportAuthoringError(str(exc)) from exc
        if not isinstance(value, dict):
            raise RuntimeImportAuthoringError(
                f"external record line {line_number} must be a JSON object"
            )
        records.append(_record_from_object(value, line_number=line_number))
    result = tuple(records)
    _require_schedule_bound_records(schedule=schedule, records=result)
    return result


def _require_schedule_bound_records(
    *,
    schedule: RuntimeBehavioralSchedule,
    records: tuple[RuntimeScoringRecord, ...],
) -> None:
    if not isinstance(schedule, RuntimeBehavioralSchedule):
        raise TypeError("schedule must be RuntimeBehavioralSchedule")
    if not isinstance(records, tuple) or not records:
        raise RuntimeImportAuthoringError("records must be a non-empty tuple")
    if len(records) > MAX_EXTERNAL_RECORDS:
        raise RuntimeImportAuthoringError(
            f"records exceed the {MAX_EXTERNAL_RECORDS}-record limit"
        )
    expected = tuple(
        (record.record_id, record.input_sha256) for record in schedule.records
    )
    observed = tuple((record.record_id, record.input_sha256) for record in records)
    if observed != expected:
        raise RuntimeImportAuthoringError(
            "external record order and input identities must exactly match the schedule"
        )
    for record in records:
        if record.status != "ok":
            raise RuntimeImportAuthoringError(
                f"external record {record.record_id!r} is not successful"
            )
        if record.output_text is not None:
            expected_output_sha256 = hashlib.sha256(
                record.output_text.encode("utf-8")
            ).hexdigest()
            if record.output_sha256 != expected_output_sha256:
                raise RuntimeImportAuthoringError(
                    f"external record {record.record_id!r} output digest is invalid"
                )


def build_runtime_import_observation(
    *,
    provider_name: str,
    artifact_identity: ModelArtifactIdentity,
    schedule: RuntimeBehavioralSchedule,
    records: tuple[RuntimeScoringRecord, ...],
) -> ScoringObservation:
    """Build one schedule-bound observation from typed per-record facts."""

    _require_schedule_bound_records(schedule=schedule, records=records)
    return ScoringObservation(
        provider_name=provider_name,
        artifact_identity_sha256=artifact_identity_sha256(artifact_identity),
        schedule_sha256=schedule.schedule_sha256,
        records=records,
        aggregate_source_sha256=runtime_scoring_records_sha256(
            [asdict(record) for record in records]
        ),
    )


def build_runtime_import_receipt(
    *,
    plugin: RuntimeProviderPluginIdentity,
    backend: RuntimeBackendIdentity,
    capabilities: RuntimeProviderCapabilities,
    artifact_identity: ModelArtifactIdentity,
    execution_settings: RuntimeExecutionSettings,
    device: RuntimeDeviceFacts,
    runtime_image_digest: str,
    observation: ScoringObservation,
) -> RuntimeProviderReceipt:
    """Build provenance bound to exact canonical observation bytes."""

    if len({plugin.name, capabilities.provider_name, observation.provider_name}) != 1:
        raise RuntimeImportAuthoringError(
            "plugin, capabilities, and observation provider names must agree"
        )
    if artifact_identity.artifact_format not in capabilities.artifact_formats:
        raise RuntimeImportAuthoringError(
            "artifact format is not declared by provider capabilities"
        )
    if execution_settings.allow_network:
        raise RuntimeImportAuthoringError(
            "strict runtime import requires offline execution settings"
        )
    if observation.artifact_identity_sha256 != artifact_identity_sha256(
        artifact_identity
    ):
        raise RuntimeImportAuthoringError(
            "observation does not bind the supplied artifact identity"
        )
    observation_bytes = encode_scoring_observation(observation)
    try:
        return RuntimeProviderReceipt(
            plugin=plugin,
            backend=backend,
            capabilities=capabilities,
            artifact_identity=artifact_identity,
            execution_settings=execution_settings,
            device=device,
            outer_image_digest=runtime_image_digest,
            scoring_observation_sha256=hashlib.sha256(observation_bytes).hexdigest(),
        )
    except (TypeError, ValueError) as exc:
        raise RuntimeImportAuthoringError(
            f"runtime import receipt is invalid: {exc}"
        ) from exc


def _destination(path: str | Path) -> tuple[Path, Path]:
    candidate = Path(path)
    if candidate.name in {"", ".", ".."}:
        raise RuntimeImportAuthoringError("runtime import output must name a directory")
    try:
        parent = candidate.parent.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise RuntimeImportAuthoringError(
            "runtime import output parent must be an existing directory"
        ) from exc
    if not parent.is_dir():
        raise RuntimeImportAuthoringError(
            "runtime import output parent must be an existing directory"
        )
    destination = parent / candidate.name
    if destination.exists() or destination.is_symlink():
        raise RuntimeImportAuthoringError("runtime import output already exists")
    return destination, parent


def _report_payload(
    observation: ScoringObservation,
    *,
    scoring_observation_sha256: str,
) -> dict[str, object]:
    return {
        "format": "invarlock/runtime-side-report-v1",
        "provider": observation.provider_name,
        "artifact_identity_sha256": observation.artifact_identity_sha256,
        "scoring_observation_sha256": scoring_observation_sha256,
        "schedule_sha256": observation.schedule_sha256,
        "record_count": len(observation.records),
    }


def _config_payload(
    *,
    role: Literal["baseline", "subject"],
    observation: ScoringObservation,
    policy_digest: str,
) -> dict[str, object]:
    return {
        "format": "invarlock/runtime-side-config-v1",
        "role": role,
        "provider": observation.provider_name,
        "artifact_identity_sha256": observation.artifact_identity_sha256,
        "schedule_sha256": observation.schedule_sha256,
        "policy_digest": policy_digest,
    }


def _validate_side_bindings(*, role: object, policy_digest: object) -> None:
    if role not in {"baseline", "subject"}:
        raise RuntimeImportAuthoringError("role must be baseline or subject")
    if not isinstance(policy_digest, str) or not policy_digest.startswith("sha256:"):
        raise RuntimeImportAuthoringError("policy_digest must be a sha256 digest")
    if len(policy_digest) != 71 or any(
        character not in "0123456789abcdef" for character in policy_digest[7:]
    ):
        raise RuntimeImportAuthoringError("policy_digest must be a sha256 digest")


def write_runtime_import_side(
    directory: str | Path,
    *,
    role: Literal["baseline", "subject"],
    schedule: RuntimeBehavioralSchedule,
    policy_digest: str,
    artifact_identity: ModelArtifactIdentity,
    records: tuple[RuntimeScoringRecord, ...],
    plugin: RuntimeProviderPluginIdentity,
    backend: RuntimeBackendIdentity,
    capabilities: RuntimeProviderCapabilities,
    execution_settings: RuntimeExecutionSettings,
    device: RuntimeDeviceFacts,
    runtime_image_ref: str,
    runtime_image_digest: str,
    generated_at_utc: str,
) -> RuntimeImportSideEvidence:
    """Atomically write and independently verify one complete import side."""

    _validate_side_bindings(role=role, policy_digest=policy_digest)
    observation = build_runtime_import_observation(
        provider_name=plugin.name,
        artifact_identity=artifact_identity,
        schedule=schedule,
        records=records,
    )
    receipt = build_runtime_import_receipt(
        plugin=plugin,
        backend=backend,
        capabilities=capabilities,
        artifact_identity=artifact_identity,
        execution_settings=execution_settings,
        device=device,
        runtime_image_digest=runtime_image_digest,
        observation=observation,
    )
    output, parent = _destination(directory)
    staging = Path(tempfile.mkdtemp(dir=parent, prefix=f".{output.name}.staging."))
    try:
        provider_evidence = write_runtime_provider_evidence(
            staging,
            artifact_identity=artifact_identity,
            scoring_observation=observation,
            receipt=receipt,
            expected_outer_image_digest=runtime_image_digest,
        )
        report_path = staging / RUNTIME_IMPORT_REPORT_FILENAME
        report_path.write_bytes(
            canonical_json_bytes(
                _report_payload(
                    observation,
                    scoring_observation_sha256=(
                        provider_evidence.scoring_observation_sha256
                    ),
                )
            )
        )
        config_path = staging / RUNTIME_IMPORT_CONFIG_FILENAME
        config_path.write_bytes(
            canonical_json_bytes(
                _config_payload(
                    role=role,
                    observation=observation,
                    policy_digest=policy_digest,
                )
            )
        )
        manifest_path = write_runtime_manifest(
            report_path,
            provider_files=RuntimeProviderManifestFiles(
                receipt=provider_evidence.paths.receipt,
                scoring_observation=provider_evidence.paths.scoring_observation,
                artifact_identity=provider_evidence.paths.artifact_identity,
            ),
            config_path=config_path,
            execution=RuntimeManifestExecution(
                execution_mode="container",
                container_execution=True,
                image_ref=runtime_image_ref,
                image_digest=runtime_image_digest,
                allow_network=False,
                allow_remote_code=False,
                allow_third_party_plugins=False,
            ),
            generated_at_utc=generated_at_utc,
        )
        manifest = cast(dict[str, object], json.loads(manifest_path.read_bytes()))
        verification = verify_runtime_manifest_snapshot(
            report_path.read_bytes(),
            manifest,
            report=report_path,
            manifest=manifest_path,
            expected_image_digest=runtime_image_digest,
            require_strict_runtime=True,
        )
        if verification.errors:
            raise RuntimeImportAuthoringError(
                "runtime import manifest is invalid: " + "; ".join(verification.errors)
            )
        publish_directory_no_replace(staging, output)
    except RuntimeImportAuthoringError:
        raise
    except (
        RuntimeProviderEvidenceError,
        AtomicDirectoryPublicationError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        raise RuntimeImportAuthoringError(str(exc)) from exc
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
    return load_runtime_import_side(
        output,
        role=role,
        schedule=schedule,
        policy_digest=policy_digest,
        expected_runtime_image_digest=runtime_image_digest,
    )


def _canonical_object_file(
    path: Path, *, label: str
) -> tuple[bytes, dict[str, object]]:
    try:
        raw = read_regular_file_bytes(path, label=label)
        parsed = parse_json_bytes(raw, label=label)
    except StrictJsonError as exc:
        raise RuntimeImportAuthoringError(str(exc)) from exc
    if not isinstance(parsed, dict):
        raise RuntimeImportAuthoringError(f"{label} must be a JSON object")
    value = cast(dict[str, object], parsed)
    if raw != canonical_json_bytes(value):
        raise RuntimeImportAuthoringError(f"{label} must use canonical JSON")
    return raw, value


def load_runtime_import_side(
    directory: str | Path,
    *,
    role: Literal["baseline", "subject"],
    schedule: RuntimeBehavioralSchedule,
    policy_digest: str,
    expected_runtime_image_digest: str,
) -> RuntimeImportSideEvidence:
    """Reload a complete side and replay every file and schedule binding."""

    _validate_side_bindings(role=role, policy_digest=policy_digest)
    supplied_root = Path(directory)
    if supplied_root.is_symlink():
        raise RuntimeImportAuthoringError(
            "runtime import side must be a real directory"
        )
    try:
        root = supplied_root.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise RuntimeImportAuthoringError(
            "runtime import side must be a real directory"
        ) from exc
    if not root.is_dir():
        raise RuntimeImportAuthoringError(
            "runtime import side must be a real directory"
        )
    try:
        provider_evidence = load_runtime_provider_evidence(
            RuntimeProviderEvidencePaths.in_directory(root),
            expected_outer_image_digest=expected_runtime_image_digest,
        )
    except RuntimeProviderEvidenceError as exc:
        raise RuntimeImportAuthoringError(str(exc)) from exc
    observation = provider_evidence.scoring_observation
    _require_schedule_bound_records(schedule=schedule, records=observation.records)
    if observation.schedule_sha256 != schedule.schedule_sha256:
        raise RuntimeImportAuthoringError(
            "runtime import observation does not bind the supplied schedule"
        )
    report_path = root / RUNTIME_IMPORT_REPORT_FILENAME
    report_raw, report = _canonical_object_file(
        report_path, label="runtime import report"
    )
    expected_report = _report_payload(
        observation,
        scoring_observation_sha256=provider_evidence.scoring_observation_sha256,
    )
    if report != expected_report:
        raise RuntimeImportAuthoringError(
            "runtime import report does not bind its observation"
        )
    config_path = root / RUNTIME_IMPORT_CONFIG_FILENAME
    config_raw, config = _canonical_object_file(
        config_path, label="runtime import config"
    )
    expected_config = _config_payload(
        role=role,
        observation=observation,
        policy_digest=policy_digest,
    )
    if config != expected_config:
        raise RuntimeImportAuthoringError(
            "runtime import config does not bind role, schedule, artifact, and policy"
        )
    manifest_path = root / RUNTIME_IMPORT_MANIFEST_FILENAME
    try:
        manifest_raw = read_regular_file_bytes(
            manifest_path, label="runtime import manifest"
        )
        manifest_value = parse_json_bytes(manifest_raw, label="runtime import manifest")
    except StrictJsonError as exc:
        raise RuntimeImportAuthoringError(str(exc)) from exc
    if not isinstance(manifest_value, dict):
        raise RuntimeImportAuthoringError(
            "runtime import manifest must be a JSON object"
        )
    manifest = cast(dict[str, object], manifest_value)
    verification = verify_runtime_manifest_snapshot(
        report_raw,
        manifest,
        report=report_path,
        manifest=manifest_path,
        expected_image_digest=expected_runtime_image_digest,
        require_strict_runtime=True,
    )
    if verification.errors:
        raise RuntimeImportAuthoringError(
            "runtime import manifest is invalid: " + "; ".join(verification.errors)
        )
    return RuntimeImportSideEvidence(
        role=role,
        directory=root,
        report_path=report_path,
        config_path=config_path,
        manifest_path=manifest_path,
        provider_evidence=provider_evidence,
        runtime_image_digest=expected_runtime_image_digest,
        side_evidence=RuntimeSideEvidence(
            run_report=report_raw,
            runtime_manifest=manifest_raw,
            runtime_config=config_raw,
            artifact_identity=provider_evidence.artifact_identity_bytes,
            provider_receipt=provider_evidence.receipt_bytes,
            scoring_observation=provider_evidence.scoring_observation_bytes,
        ),
    )


def write_runtime_import_paired_records(
    path: str | Path,
    *,
    schedule: RuntimeBehavioralSchedule,
    metric: str,
    baseline: RuntimeImportSideEvidence,
    subject: RuntimeImportSideEvidence,
    scorer_binding: ScorerExtensionBinding | None = None,
    scorer_registry: ScorerExtensionRegistry | None = None,
) -> RuntimeImportPairedRecords:
    """Derive and no-clobber write canonical pairs from two verified sides."""

    if baseline.role != "baseline" or subject.role != "subject":
        raise RuntimeImportAuthoringError(
            "paired records require baseline and subject side roles"
        )
    collection_metric = "exact_match" if scorer_binding is not None else metric
    for side in (baseline, subject):
        if collection_metric not in side.provider_evidence.receipt.capabilities.metrics:
            raise RuntimeImportAuthoringError(
                f"{side.role} provider evidence does not declare metric "
                f"{collection_metric!r}"
            )
    try:
        paired = derive_paired_records(
            schedule=schedule,
            metric=metric,
            baseline=baseline.evidence_pack_value(),
            subject=subject.evidence_pack_value(),
            baseline_identity_digest=sha256_digest(
                baseline.provider_evidence.artifact_identity_bytes
            ),
            subject_identity_digest=sha256_digest(
                subject.provider_evidence.artifact_identity_bytes
            ),
            baseline_runtime_digest=baseline.runtime_image_digest,
            subject_runtime_digest=subject.runtime_image_digest,
            scorer_binding=scorer_binding,
            scorer_registry=scorer_registry,
        )
    except ValueError as exc:
        raise RuntimeImportAuthoringError(
            f"runtime import paired records are invalid: {exc}"
        ) from exc
    output_candidate = Path(path)
    if output_candidate.name in {"", ".", ".."}:
        raise RuntimeImportAuthoringError(
            "runtime import paired-record destination must name a file"
        )
    try:
        output_parent = output_candidate.parent.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise RuntimeImportAuthoringError(
            "runtime import paired-record destination parent must exist"
        ) from exc
    if not output_parent.is_dir():
        raise RuntimeImportAuthoringError(
            "runtime import paired-record destination parent must exist"
        )
    output = output_parent / output_candidate.name
    payload = canonical_json_bytes(paired)
    parent_fd: int | None = None
    staging_fd: int | None = None
    staging_name: str | None = None
    try:
        parent_fd = os.open(
            output_parent,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        staging_fd, staging_path = tempfile.mkstemp(
            dir=output_parent,
            prefix=f".{output.name}.staging.",
        )
        staging_name = Path(staging_path).name
        with os.fdopen(staging_fd, "wb") as handle:
            staging_fd = None
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(
            staging_name,
            output.name,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
            follow_symlinks=False,
        )
        os.fsync(parent_fd)
    except OSError as exc:
        raise RuntimeImportAuthoringError(
            "runtime import paired-record destination must be new and writable"
        ) from exc
    finally:
        if staging_fd is not None:
            os.close(staging_fd)
        if staging_name is not None and parent_fd is not None:
            try:
                os.unlink(staging_name, dir_fd=parent_fd)
            except FileNotFoundError:
                pass
        if parent_fd is not None:
            os.close(parent_fd)
    try:
        reloaded, value = _canonical_object_file(
            output, label="runtime import paired records"
        )
        if value != paired or reloaded != payload:
            raise RuntimeImportAuthoringError(
                "runtime import paired records changed during publication"
            )
    except RuntimeImportAuthoringError:
        output.unlink(missing_ok=True)
        raise
    return RuntimeImportPairedRecords(
        path=output.resolve(),
        payload=paired,
        sha256=hashlib.sha256(payload).hexdigest(),
    )


__all__ = [
    "MAX_EXTERNAL_RECORDS",
    "RUNTIME_IMPORT_CONFIG_FILENAME",
    "RUNTIME_IMPORT_MANIFEST_FILENAME",
    "RUNTIME_IMPORT_REPORT_FILENAME",
    "RuntimeImportAuthoringError",
    "RuntimeImportPairedRecords",
    "RuntimeImportSideEvidence",
    "build_runtime_import_observation",
    "build_runtime_import_receipt",
    "load_external_scoring_records_jsonl",
    "load_runtime_import_side",
    "write_runtime_import_paired_records",
    "write_runtime_import_side",
]
