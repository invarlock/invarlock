"""Authenticated execute-or-import transaction for paired evaluation."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives.asymmetric import ed25519

from invarlock.core.evaluation_request import (
    ComparisonSideRequest,
    EvaluationRequest,
    EvaluationRequestError,
    ImportSideRequest,
    load_evaluation_request,
)
from invarlock.core.registry import CoreRegistry
from invarlock.core.runtime_provider import (
    ModelRuntimeSpec,
    RuntimeBehavioralSchedule,
    RuntimeProvider,
    RuntimeProviderInputPreflight,
    artifact_identity_sha256,
    build_runtime_behavioral_schedule,
    validate_runtime_evaluation_inputs,
)
from invarlock.core.schedule_preparation import (
    LocalDatasetRequest,
    local_dataset_preparation_payload,
    prepare_local_evaluation_schedule_bytes,
)
from invarlock.core.scorer_extension import (
    SCORER_REPLAY_OUTPUT_KIND,
    ScorerExtensionError,
    ScorerExtensionRegistry,
    scorer_binding_payload,
)
from invarlock.evaluation_run import (
    RuntimeComparisonExecutor,
    execute_runtime_comparison,
)
from invarlock.evaluation_runtime import (
    RuntimeResourceResolver,
    RuntimeSideRole,
    caller_runtime_resources_from_environment,
)
from invarlock.evidence_pack import (
    EVIDENCE_PATHS,
    EvidenceObservation,
    EvidencePackError,
    InputIdentity,
    RuntimeSideEvidence,
    derive_paired_records,
    publish_comparison_evidence,
)
from invarlock.evidence_pack_contract import (
    MAX_OBSERVATION_BYTES,
    PAIRED_RECORDS_FORMAT,
    build_comparison_report,
    canonical_json_bytes,
    normalize_digest,
    policy_sample_requirements,
    runtime_side_config_errors,
    schedule_bytes,
    sha256_digest,
)
from invarlock.evidence_pack_integrity import public_key_fingerprint
from invarlock.evidence_pack_json import parse_json_bytes
from invarlock.evidence_pack_publication import _load_private_key
from invarlock.runtime_provider_evidence import (
    RuntimeProviderEvidenceError,
    decode_artifact_identity,
    decode_runtime_provider_receipt,
    encode_artifact_identity,
)
from invarlock.runtime_security_helpers import (
    network_allowed,
    remote_code_allowed,
    third_party_plugins_allowed,
)

_MAX_POLICY_BYTES = 4 * 1024 * 1024
_MAX_REQUEST_INPUT_BYTES = 64 * 1024 * 1024
_WINDOWS_DRIVE_RE = re.compile(r"^[A-Za-z]:")
_DIRECTORY_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_NOFOLLOW", 0)
)
_FILE_FLAGS = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)


class EvaluationTransactionError(ValueError):
    """Raised when a request cannot produce authenticated evidence."""

    def __init__(self, message: str, *, exit_code: int = 2) -> None:
        super().__init__(message)
        self.exit_code = exit_code

    def as_json(self) -> str:
        return json.dumps(
            {
                "format_version": "invarlock/evaluation-result-v1",
                "ok": False,
                "errors": [str(self)],
            },
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )


class EvaluationPreflightError(ValueError):
    """Raised when execution-free evaluation qualification fails."""

    exit_code = 2

    def as_json(self) -> str:
        return json.dumps(
            {
                "format_version": "invarlock/evaluation-preflight-v2",
                "ok": False,
                "errors": [str(self)],
            },
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )


def _require_closed_runtime_switches() -> None:
    """Reject process-wide execution opt-ins before provider discovery."""

    enabled: list[str] = []
    if network_allowed():
        enabled.append("network access")
    if remote_code_allowed():
        enabled.append("remote code")
    if third_party_plugins_allowed():
        enabled.append("third-party provider discovery")
    if enabled:
        raise EvaluationTransactionError(
            "strict evaluation requires disabled runtime opt-ins: " + ", ".join(enabled)
        )


@dataclass(frozen=True)
class EvaluationTransactionResult:
    """Successful publication of one immutable evidence pack."""

    evidence_path: Path
    comparison_id: str
    pack_manifest_digest: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "pack_manifest_digest",
            normalize_digest(
                self.pack_manifest_digest,
                label="published evidence manifest digest",
            ),
        )

    def as_json(self) -> str:
        return json.dumps(
            {
                "format_version": "invarlock/evaluation-result-v1",
                "ok": True,
                "comparison_id": self.comparison_id,
                "evidence": str(self.evidence_path),
                "pack_manifest_digest": self.pack_manifest_digest,
            },
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )


@dataclass(frozen=True)
class EvaluationPreflightResult:
    """Successful execution-free qualification of one evaluation request."""

    execution_mode: str
    output: str
    schedule_digest: str
    policy_digest: str
    artifact_digests: dict[str, str]
    evidence_signer_fingerprint: str
    request_digest: str
    record_count: int
    providers: dict[str, str]
    checks: tuple[str, ...]
    runtime_image_digests: dict[str, str] | None = None
    sample_qualification: dict[str, object] | None = None
    format_version: str = "invarlock/evaluation-preflight-v2"

    def as_json(self) -> str:
        payload: dict[str, object] = {
            "format_version": self.format_version,
            "ok": True,
            "execution_mode": self.execution_mode,
            "output": self.output,
            "schedule_digest": self.schedule_digest,
            "policy_digest": self.policy_digest,
            "artifact_digests": self.artifact_digests,
            "evidence_signer_fingerprint": self.evidence_signer_fingerprint,
            "request_digest": self.request_digest,
            "record_count": self.record_count,
            "providers": self.providers,
            "checks": list(self.checks),
        }
        if self.runtime_image_digests is not None:
            payload["runtime_image_digests"] = self.runtime_image_digests
        if self.sample_qualification is not None:
            payload["sample_qualification"] = self.sample_qualification
        return json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )


@dataclass(frozen=True)
class _PreparedEvaluation:
    signing_key: ed25519.Ed25519PrivateKey
    registry: CoreRegistry
    request: EvaluationRequest
    schedule: RuntimeBehavioralSchedule
    canonical_schedule: bytes
    policy_bytes: bytes
    policy_digest: str
    artifact_digests: dict[str, str] | None
    evidence_signer_fingerprint: str
    selected_metric: str
    sample_requirements: dict[str, int | float]
    observations: tuple[EvidenceObservation, ...]


@dataclass
class _OutputParentAnchor:
    descriptor: int
    device: int
    inode: int
    destination_name: str

    def close(self) -> None:
        os.close(self.descriptor)


def _root_relative_parts(root: Path, path: Path, *, label: str) -> tuple[str, ...]:
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise EvaluationTransactionError(f"{label} escapes the request root") from exc
    if not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        raise EvaluationTransactionError(f"{label} is not a safe request reference")
    return relative.parts


def _read_request_file(
    root: Path,
    path: Path,
    *,
    label: str,
    max_bytes: int = _MAX_REQUEST_INPUT_BYTES,
) -> bytes:
    """Repeat root-confined no-follow traversal at the exact read boundary."""

    parts = _root_relative_parts(root, path, label=label)
    root_fd = os.open(root, _DIRECTORY_FLAGS)
    current_fd = root_fd
    try:
        for index, component in enumerate(parts):
            final = index == len(parts) - 1
            try:
                child_fd = os.open(
                    component,
                    _FILE_FLAGS if final else _DIRECTORY_FLAGS,
                    dir_fd=current_fd,
                )
            except OSError as exc:
                raise EvaluationTransactionError(
                    f"{label} could not be opened without following links"
                ) from exc
            if current_fd != root_fd:
                os.close(current_fd)
            current_fd = child_fd
        opened = os.fstat(current_fd)
        if not stat.S_ISREG(opened.st_mode):
            raise EvaluationTransactionError(f"{label} must be a regular file")
        if opened.st_size > max_bytes:
            raise EvaluationTransactionError(
                f"{label} exceeds the {max_bytes}-byte size limit"
            )
        chunks: list[bytes] = []
        remaining = max_bytes + 1
        while remaining > 0:
            chunk = os.read(current_fd, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        if len(payload) > max_bytes:
            raise EvaluationTransactionError(
                f"{label} exceeds the {max_bytes}-byte size limit"
            )
        after = os.fstat(current_fd)
        identity = lambda value: (  # noqa: E731 - compact immutable stat projection
            value.st_dev,
            value.st_ino,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )
        if identity(opened) != identity(after):
            raise EvaluationTransactionError(f"{label} changed while being read")
        return payload
    finally:
        if current_fd != root_fd:
            os.close(current_fd)
        os.close(root_fd)


def _prepare_output_parent(root: Path, destination: Path) -> _OutputParentAnchor:
    """Create and retain an inode anchor for the no-follow output parent."""

    parts = _root_relative_parts(root, destination, label="output.evidence")
    root_fd = os.open(root, _DIRECTORY_FLAGS)
    current_fd = root_fd
    anchor_fd: int | None = None
    try:
        for component in parts[:-1]:
            try:
                child_fd = os.open(component, _DIRECTORY_FLAGS, dir_fd=current_fd)
            except FileNotFoundError:
                try:
                    os.mkdir(component, mode=0o755, dir_fd=current_fd)
                    child_fd = os.open(component, _DIRECTORY_FLAGS, dir_fd=current_fd)
                except OSError as exc:
                    raise EvaluationTransactionError(
                        "output.evidence parent could not be created safely"
                    ) from exc
            except OSError as exc:
                raise EvaluationTransactionError(
                    "output.evidence parent traverses an unsafe component"
                ) from exc
            if current_fd != root_fd:
                os.close(current_fd)
            current_fd = child_fd
        try:
            os.stat(parts[-1], dir_fd=current_fd, follow_symlinks=False)
        except FileNotFoundError:
            anchor_fd = os.dup(current_fd)
            parent_stat = os.fstat(anchor_fd)
            return _OutputParentAnchor(
                descriptor=anchor_fd,
                device=parent_stat.st_dev,
                inode=parent_stat.st_ino,
                destination_name=parts[-1],
            )
        else:
            raise EvaluationTransactionError("output.evidence already exists")
    finally:
        if current_fd != root_fd:
            os.close(current_fd)
        os.close(root_fd)


def _revalidate_output_parent(
    anchor: _OutputParentAnchor,
    destination: Path,
    *,
    published: bool,
) -> None:
    """Fail closed if publication changed or escaped the anchored parent."""

    try:
        pathname_fd = os.open(destination.parent, _DIRECTORY_FLAGS)
    except OSError as exc:
        raise EvaluationTransactionError(
            "output.evidence parent changed during publication"
        ) from exc
    try:
        pathname_stat = os.fstat(pathname_fd)
    finally:
        os.close(pathname_fd)
    if (pathname_stat.st_dev, pathname_stat.st_ino) != (anchor.device, anchor.inode):
        raise EvaluationTransactionError(
            "output.evidence parent changed during publication"
        )
    try:
        entry_stat = os.stat(
            anchor.destination_name,
            dir_fd=anchor.descriptor,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        if published:
            raise EvaluationTransactionError(
                "evidence was not published beneath the anchored output parent"
            ) from None
        return
    if not published:
        raise EvaluationTransactionError("output.evidence already exists")
    if not stat.S_ISDIR(entry_stat.st_mode):
        raise EvaluationTransactionError(
            "published evidence is not a directory beneath the anchored parent"
        )
    try:
        pathname_entry = destination.lstat()
    except OSError as exc:
        raise EvaluationTransactionError(
            "published evidence path changed during publication"
        ) from exc
    if (pathname_entry.st_dev, pathname_entry.st_ino) != (
        entry_stat.st_dev,
        entry_stat.st_ino,
    ):
        raise EvaluationTransactionError(
            "published evidence escaped the anchored output parent"
        )


def _parse_object(payload: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = parse_json_bytes(payload, label=label)
    except ValueError as exc:
        raise EvaluationTransactionError(str(exc)) from exc
    if not isinstance(value, dict):
        raise EvaluationTransactionError(f"{label} must be a JSON object")
    return value


def _stable_locator(value: str | None, *, label: str) -> str:
    locator = value.strip() if isinstance(value, str) else ""
    if (
        not locator
        or locator.startswith(("/", "~", "\\"))
        or _WINDOWS_DRIVE_RE.match(locator) is not None
        or locator.lower().startswith("file://")
    ):
        raise EvaluationTransactionError(f"{label} must be a stable non-file locator")
    return locator


def _side_evidence(
    request: EvaluationRequest,
    imported: ImportSideRequest,
    *,
    side: str,
) -> RuntimeSideEvidence:
    return RuntimeSideEvidence(
        run_report=_read_request_file(
            request.root, imported.run_report, label=f"{side} run report"
        ),
        runtime_manifest=_read_request_file(
            request.root,
            imported.runtime_manifest,
            label=f"{side} runtime manifest",
        ),
        runtime_config=_read_request_file(
            request.root, imported.runtime_config, label=f"{side} runtime config"
        ),
        artifact_identity=_read_request_file(
            request.root, imported.identity, label=f"{side} artifact identity"
        ),
        provider_receipt=_read_request_file(
            request.root, imported.receipt, label=f"{side} provider receipt"
        ),
        scoring_observation=_read_request_file(
            request.root, imported.observation, label=f"{side} scoring observation"
        ),
    )


def _validate_import_side(
    comparison: ComparisonSideRequest,
    evidence: RuntimeSideEvidence,
    *,
    side: str,
    provider: RuntimeProvider,
    task: str,
    metric: str,
    schedule: RuntimeBehavioralSchedule,
    policy_digest: str,
) -> str:
    try:
        identity = decode_artifact_identity(evidence.artifact_identity)
        receipt = decode_runtime_provider_receipt(evidence.provider_receipt)
    except RuntimeProviderEvidenceError as exc:
        raise EvaluationTransactionError(
            f"{side} provider evidence is invalid: {exc}"
        ) from exc
    providers = {
        receipt.plugin.name,
        receipt.capabilities.provider_name,
        comparison.runtime.provider,
        provider.name,
    }
    if len(providers) != 1:
        raise EvaluationTransactionError(
            f"{side} provider evidence does not match the requested runtime"
        )
    installed_capabilities = provider.capabilities()
    if (
        task not in receipt.capabilities.tasks
        or task not in installed_capabilities.tasks
    ):
        raise EvaluationTransactionError(
            f"{side} provider evidence does not declare requested task {task!r}"
        )
    if (
        metric not in receipt.capabilities.metrics
        or metric not in installed_capabilities.metrics
    ):
        raise EvaluationTransactionError(
            f"{side} provider evidence does not declare requested metric {metric!r}"
        )
    try:
        expected_identity = provider.identify_artifact(
            ModelRuntimeSpec(
                provider_name=comparison.runtime.provider,
                model_id=comparison.artifact.model_id,
                settings=comparison.runtime.settings,
            )
        )
    except (TypeError, ValueError) as exc:
        raise EvaluationTransactionError(
            f"{side} requested runtime cannot reproduce an artifact identity: {exc}"
        ) from exc
    if identity != expected_identity:
        raise EvaluationTransactionError(
            f"{side} artifact identity does not match the requested runtime spec"
        )
    config_errors = runtime_side_config_errors(
        evidence.runtime_config,
        role=side,
        provider_name=receipt.plugin.name,
        artifact_identity_sha256=artifact_identity_sha256(identity),
        schedule_sha256=schedule.schedule_sha256,
        policy_digest=policy_digest,
    )
    if config_errors:
        raise EvaluationTransactionError(config_errors[0])
    if receipt.outer_image_digest is None:
        raise EvaluationTransactionError(
            f"{side} provider receipt lacks a strict outer runtime image digest"
        )
    execution = receipt.execution_settings
    settings = comparison.runtime.settings
    for field in (
        "seed",
        "context_length",
        "batch_size",
        "max_output_tokens",
        "timeout_seconds",
        "allow_network",
    ):
        if field in settings and settings[field] != getattr(execution, field):
            raise EvaluationTransactionError(
                f"{side} provider receipt does not match runtime setting {field!r}"
            )
    return receipt.outer_image_digest


def _normalized_side(side: ComparisonSideRequest) -> dict[str, object]:
    return {
        "artifact": {
            "model_id": side.artifact.model_id,
            "locator": _stable_locator(
                side.artifact.locator,
                label=f"{side.artifact.model_id} artifact locator",
            ),
        },
        "runtime": {
            "provider": side.runtime.provider,
            "settings": dict(side.runtime.settings),
        },
    }


def _normalized_request(
    request: EvaluationRequest,
    schedule: RuntimeBehavioralSchedule,
    observations: tuple[EvidenceObservation, ...],
) -> dict[str, object]:
    def imported(side: str) -> dict[str, str]:
        return {
            "identity": EVIDENCE_PATHS[f"{side}_provider_identity"],
            "receipt": EVIDENCE_PATHS[f"{side}_provider_receipt"],
            "observation": EVIDENCE_PATHS[f"{side}_scoring_observation"],
            "run_report": EVIDENCE_PATHS[f"{side}_run_report"],
            "runtime_manifest": EVIDENCE_PATHS[f"{side}_runtime_manifest"],
            "runtime_config": EVIDENCE_PATHS[f"{side}_runtime_config"],
        }

    execution: dict[str, object]
    dataset: object
    if request.execution.mode == "run":
        execution = {"mode": "run"}
        source = request.comparison.dataset
        assert isinstance(source, LocalDatasetRequest)
        dataset = local_dataset_preparation_payload(source, schedule)
    else:
        execution = {
            "mode": "import",
            "records": "records/paired-records.json",
            "schedule": EVIDENCE_PATHS["schedule"],
            "baseline": imported("baseline"),
            "subject": imported("subject"),
        }
        dataset = EVIDENCE_PATHS["schedule"]
    normalized: dict[str, object] = {
        "format_version": request.format_version,
        "comparison": {
            "baseline": _normalized_side(request.comparison.baseline),
            "subject": _normalized_side(request.comparison.subject),
            "dataset": dataset,
            "policy": "inputs/policy.json",
            "task": request.comparison.task,
        },
        "execution": execution,
        "output": {"evidence": "evidence"},
    }
    if request.comparison.scorer_extension is None:
        normalized_comparison = normalized["comparison"]
        assert isinstance(normalized_comparison, dict)
        normalized_comparison["metric"] = request.comparison.metric
    else:
        normalized_comparison = normalized["comparison"]
        assert isinstance(normalized_comparison, dict)
        normalized_comparison["scorer_extension"] = scorer_binding_payload(
            request.comparison.scorer_extension
        )
    if observations:
        normalized["observations"] = [
            {
                "id": observation.observation_id,
                "kind": observation.kind,
                "scope": observation.scope,
                "payload_digest": sha256_digest(observation.payload),
            }
            for observation in observations
        ]
    return normalized


def _load_request_observations(
    request: EvaluationRequest,
) -> tuple[EvidenceObservation, ...]:
    """Read and validate every optional observation before runtime launch."""

    return tuple(
        sorted(
            (
                EvidenceObservation(
                    observation_id=observation.observation_id,
                    kind=observation.kind,
                    scope=observation.scope,
                    payload=_read_request_file(
                        request.root,
                        observation.path,
                        label=f"observation {observation.observation_id!r}",
                        max_bytes=MAX_OBSERVATION_BYTES,
                    ),
                )
                for observation in request.observations
            ),
            key=lambda observation: observation.observation_id,
        )
    )


def _comparison_id(
    normalized_request: dict[str, object],
    *,
    baseline_identity: bytes,
    subject_identity: bytes,
    schedule_payload: bytes,
    policy_payload: bytes,
    baseline_runtime_digest: str,
    subject_runtime_digest: str,
    paired_records: dict[str, object],
) -> str:
    digest = hashlib.sha256()
    for payload in (
        canonical_json_bytes(normalized_request),
        baseline_identity,
        subject_identity,
        schedule_payload,
        policy_payload,
        baseline_runtime_digest.encode("ascii"),
        subject_runtime_digest.encode("ascii"),
        canonical_json_bytes(paired_records),
    ):
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return f"comparison-{digest.hexdigest()[:32]}"


def _preflight_policy(
    policy: dict[str, Any],
    *,
    metric: str,
    policy_digest: str,
    scorer_binding=None,
) -> None:
    """Validate the selected metric policy before either runtime is launched."""

    record: dict[str, object] = {
        "record_id": "policy-preflight",
        "input_sha256": "0" * 64,
        "baseline": {"score": 1.0},
        "subject": {"score": 1.0},
    }
    paired: dict[str, object] = {
        "format": PAIRED_RECORDS_FORMAT,
        "metric": metric,
        "schedule_sha256": "0" * 64,
        "records": [record],
    }
    if metric == "normalized_nll_per_utf8_byte":
        paired["derived_measurements"] = {
            "perplexity_ratio": {
                "status": "available",
                "basis": "authenticated_target_likelihood",
                "method": "target_token_weighted_perplexity_ratio_v1",
                "tokenizer_metadata_sha256": "0" * 64,
                "target_token_count": 1,
                "baseline_perplexity": 1.0,
                "subject_perplexity": 1.0,
                "ratio": 1.0,
            }
        }
    if scorer_binding is not None:
        paired["scorer_extension"] = scorer_binding_payload(scorer_binding)
        paired["scorer_replay"] = {"baseline": {}, "subject": {}}
    build_comparison_report(
        comparison_id="policy-preflight",
        paired_records=paired,
        policy=policy,
        policy_digest=policy_digest,
    )


def _validate_output_destination(request: EvaluationRequest) -> None:
    """Check that publication can begin without creating any output component."""

    destination = request.output.evidence
    if os.path.lexists(destination):
        raise EvaluationTransactionError("output.evidence already exists")
    candidate = destination.parent
    while not candidate.exists():
        if candidate == request.root or candidate.parent == candidate:
            break
        candidate = candidate.parent
    if candidate.is_symlink() or not candidate.is_dir():
        raise EvaluationTransactionError(
            "output.evidence parent is not a real directory"
        )
    try:
        candidate.relative_to(request.root)
    except ValueError as exc:
        raise EvaluationTransactionError(
            "output.evidence parent escapes the request root"
        ) from exc
    if not os.access(candidate, os.W_OK | os.X_OK):
        raise EvaluationTransactionError(
            "output.evidence parent is not writable by the caller"
        )


def _prepare_evaluation_inputs(
    request_path: Path | EvaluationRequest,
    *,
    signing_key_path: Path | None,
    scorer_registry: ScorerExtensionRegistry | None,
    authenticate_artifacts: bool = False,
    registry: CoreRegistry | None = None,
) -> _PreparedEvaluation:
    if signing_key_path is None:
        raise EvaluationTransactionError("an Ed25519 evidence-signing key is required")
    _require_closed_runtime_switches()
    signing_key = _load_private_key(Path(signing_key_path))
    selected_registry = registry if registry is not None else CoreRegistry()
    request = (
        request_path
        if isinstance(request_path, EvaluationRequest)
        else load_evaluation_request(
            request_path,
            provider_resolver=selected_registry.get_runtime_provider,
        )
    )
    artifact_digests: dict[str, str] | None = None
    if authenticate_artifacts and request.execution.mode == "run":
        artifact_digests = {}
        for side_name, side in (
            ("baseline", request.comparison.baseline),
            ("subject", request.comparison.subject),
        ):
            provider = selected_registry.get_runtime_provider(side.runtime.provider)
            artifact_path = side.artifact.path
            assert artifact_path is not None
            try:
                identity = provider.authenticate_artifact(
                    ModelRuntimeSpec(
                        provider_name=side.runtime.provider,
                        model_id=side.artifact.model_id,
                        settings=side.runtime.settings,
                    ),
                    artifact_path,
                )
                artifact_digests[side_name] = sha256_digest(
                    encode_artifact_identity(identity)
                )
            except (TypeError, ValueError) as exc:
                raise EvaluationTransactionError(
                    f"{side_name} artifact could not be authenticated: {exc}"
                ) from exc

    if request.execution.mode == "import":
        schedule_path = request.execution.schedule
        assert schedule_path is not None
        schedule_raw = _read_request_file(
            request.root, schedule_path, label="canonical schedule"
        )
        schedule_payload = _parse_object(schedule_raw, label="canonical schedule")
        schedule = build_runtime_behavioral_schedule(schedule_payload)
        if schedule.task != request.comparison.task:
            raise EvaluationTransactionError(
                "imported schedule task does not match comparison.task"
            )
        canonical_schedule = schedule_bytes(schedule)
        if schedule_raw != canonical_schedule:
            raise EvaluationTransactionError(
                "canonical schedule bytes are not canonical"
            )
        dataset_path = request.comparison.dataset
        assert isinstance(dataset_path, Path)
        dataset_raw = _read_request_file(
            request.root, dataset_path, label="dataset input"
        )
        if dataset_raw != canonical_schedule:
            raise EvaluationTransactionError(
                "dataset input must be the exact canonical imported schedule"
            )
    else:
        dataset_source = request.comparison.dataset
        assert isinstance(dataset_source, LocalDatasetRequest)
        dataset_raw = _read_request_file(
            request.root,
            dataset_source.path,
            label="local evaluation dataset",
        )
        schedule = prepare_local_evaluation_schedule_bytes(
            dataset_source,
            dataset_raw,
            task=request.comparison.task,
        )
        canonical_schedule = schedule_bytes(schedule)

    policy_bytes = _read_request_file(
        request.root,
        request.comparison.policy,
        label="policy input",
        max_bytes=_MAX_POLICY_BYTES,
    )
    policy_payload = _parse_object(policy_bytes, label="policy input")
    policy_digest = sha256_digest(policy_bytes)
    selected_metric = (
        request.comparison.scorer_extension.scorer_id
        if request.comparison.scorer_extension is not None
        else request.comparison.metric
    )
    assert selected_metric is not None
    _preflight_policy(
        policy_payload,
        metric=selected_metric,
        policy_digest=policy_digest,
        scorer_binding=request.comparison.scorer_extension,
    )
    sample_requirements = policy_sample_requirements(
        policy_payload,
        metric=selected_metric,
        scorer_binding=request.comparison.scorer_extension,
    )
    minimum_record_count = sample_requirements.get("minimum_record_count")
    if (
        isinstance(minimum_record_count, int)
        and len(schedule.records) < minimum_record_count
    ):
        raise EvaluationTransactionError(
            f"schedule has {len(schedule.records)} records but policy requires at "
            f"least {minimum_record_count}"
        )
    if request.comparison.scorer_extension is not None:
        if scorer_registry is None:
            raise EvaluationTransactionError(
                "request requires an explicitly authorized scorer registry"
            )
        try:
            available_scorers = scorer_registry.list_scorers()
        except ScorerExtensionError as exc:
            raise EvaluationTransactionError(str(exc)) from exc
        if request.comparison.scorer_extension.scorer_id not in available_scorers:
            raise EvaluationTransactionError(
                "required scorer extension is not installed or enabled"
            )
    observations = _load_request_observations(request)
    _validate_output_destination(request)
    return _PreparedEvaluation(
        signing_key=signing_key,
        registry=selected_registry,
        request=request,
        schedule=schedule,
        canonical_schedule=canonical_schedule,
        policy_bytes=policy_bytes,
        policy_digest=policy_digest,
        artifact_digests=artifact_digests,
        evidence_signer_fingerprint=public_key_fingerprint(signing_key.public_key()),
        selected_metric=selected_metric,
        sample_requirements=sample_requirements,
        observations=observations,
    )


def preflight_evaluation_request(
    request_path: Path | EvaluationRequest,
    *,
    signing_key_path: Path | None,
    scorer_registry: ScorerExtensionRegistry | None = None,
    runtime_image_digests: Mapping[str, str] | None = None,
    resource_resolver: RuntimeResourceResolver | None = None,
    registry: CoreRegistry | None = None,
) -> EvaluationPreflightResult:
    """Validate an evaluation transaction without execution or filesystem mutation."""

    try:
        prepared = _prepare_evaluation_inputs(
            request_path,
            signing_key_path=signing_key_path,
            scorer_registry=scorer_registry,
            authenticate_artifacts=True,
            registry=registry,
        )
        request = prepared.request
        checks = [
            "request",
            "artifacts",
            "dataset_schedule",
            "policy",
            "provider_capabilities",
            "signing_key",
            "output_destination",
        ]
        scorer_binding = request.comparison.scorer_extension
        if scorer_binding is not None:
            assert scorer_registry is not None
            input_kinds = tuple(
                sorted(
                    {
                        part.kind
                        for scheduled in prepared.schedule.records
                        for part in scheduled.input_parts
                    }
                    or {"text"}
                )
            )
            try:
                scorer_registry.validate_binding(
                    scorer_binding,
                    task=prepared.schedule.task,
                    input_kinds=input_kinds,
                    output_kind=SCORER_REPLAY_OUTPUT_KIND,
                )
            except ScorerExtensionError as exc:
                raise EvaluationTransactionError(str(exc)) from exc
            checks.append("scorer_binding")
        if prepared.sample_requirements:
            checks.append("sample_record_count")
        normalized_runtime_digests: dict[str, str] | None = None
        artifact_digests = prepared.artifact_digests
        if request.execution.mode == "run":
            if runtime_image_digests is None or set(runtime_image_digests) != {
                "baseline",
                "subject",
            }:
                raise EvaluationTransactionError(
                    "run preflight requires both locally available runtime images"
                )
            normalized_runtime_digests = {
                side: normalize_digest(
                    str(runtime_image_digests[side]),
                    label=f"{side} runtime image digest",
                )
                for side in ("baseline", "subject")
            }
            checks.append("runtime_images")
            sides: tuple[tuple[RuntimeSideRole, ComparisonSideRequest], ...] = (
                ("baseline", request.comparison.baseline),
                ("subject", request.comparison.subject),
            )
            providers = tuple(
                (
                    side_name,
                    side,
                    prepared.registry.get_runtime_provider(side.runtime.provider),
                )
                for side_name, side in sides
            )
            if resource_resolver is None:
                if any(
                    isinstance(provider, RuntimeProviderInputPreflight)
                    for _side_name, _side, provider in providers
                ):
                    raise EvaluationTransactionError(
                        "run preflight requires caller-owned runtime resources for "
                        "provider input validation"
                    )
            else:
                for side_name, side, provider in providers:
                    spec = ModelRuntimeSpec(
                        provider_name=side.runtime.provider,
                        model_id=side.artifact.model_id,
                        settings=side.runtime.settings,
                    )
                    try:
                        resources = resource_resolver.resolve(
                            request_root=request.root,
                            role=side_name,
                            side=side,
                            provider=provider,
                        )
                    except (OSError, RuntimeError, TypeError, ValueError) as exc:
                        raise EvaluationTransactionError(
                            f"{side_name} caller-owned runtime resources are "
                            f"invalid: {exc}"
                        ) from exc
                    expected_runtime_digest = normalized_runtime_digests[side_name]
                    if resources.container_image_digest != expected_runtime_digest:
                        raise EvaluationTransactionError(
                            f"{side_name} caller-owned runtime resources do not "
                            "match the inspected runtime image digest"
                        )
                    try:
                        validate_runtime_evaluation_inputs(
                            provider, spec, resources, prepared.schedule
                        )
                    except (OSError, RuntimeError, TypeError, ValueError) as exc:
                        raise EvaluationTransactionError(
                            f"{side_name} provider {provider.name!r} input "
                            f"preflight failed: {exc}"
                        ) from exc
                checks.append("runtime_resources")
        else:
            assert request.execution.baseline is not None
            assert request.execution.subject is not None
            imported_sides = {
                "baseline": _side_evidence(
                    request,
                    request.execution.baseline,
                    side="baseline",
                ),
                "subject": _side_evidence(
                    request,
                    request.execution.subject,
                    side="subject",
                ),
            }
            artifact_digests = {
                side: sha256_digest(evidence.artifact_identity)
                for side, evidence in imported_sides.items()
            }
            for side_name, side in (
                ("baseline", request.comparison.baseline),
                ("subject", request.comparison.subject),
            ):
                _validate_import_side(
                    side,
                    imported_sides[side_name],
                    side=side_name,
                    provider=prepared.registry.get_runtime_provider(
                        side.runtime.provider
                    ),
                    task=request.comparison.task,
                    metric=request.comparison.collection_metric,
                    schedule=prepared.schedule,
                    policy_digest=prepared.policy_digest,
                )
            assert request.execution.records is not None
            records_raw = _read_request_file(
                request.root,
                request.execution.records,
                label="imported paired records",
            )
            records_payload = _parse_object(
                records_raw,
                label="imported paired records",
            )
            if records_raw != canonical_json_bytes(records_payload):
                raise EvaluationTransactionError(
                    "imported paired records must use canonical JSON"
                )
        assert artifact_digests is not None
        assert set(artifact_digests) == {"baseline", "subject"}
        normalized_request = _normalized_request(
            request,
            prepared.schedule,
            prepared.observations,
        )
        sample_qualification: dict[str, object] | None = None
        if prepared.sample_requirements:
            minimum = prepared.sample_requirements["minimum_record_count"]
            width_field = next(
                field
                for field in prepared.sample_requirements
                if field != "minimum_record_count"
            )
            sample_qualification = {
                "record_count": {
                    "minimum": minimum,
                    "observed": len(prepared.schedule.records),
                    "status": "pass",
                },
                "interval_width": {
                    "maximum": prepared.sample_requirements[width_field],
                    "unit": (
                        "ratio"
                        if width_field.endswith("_ratio")
                        else "percentage_points"
                    ),
                    "status": "pending_execution",
                },
            }
        return EvaluationPreflightResult(
            execution_mode=request.execution.mode,
            output=request.output.evidence.relative_to(request.root).as_posix(),
            schedule_digest=f"sha256:{prepared.schedule.schedule_sha256}",
            policy_digest=prepared.policy_digest,
            artifact_digests=artifact_digests,
            evidence_signer_fingerprint=prepared.evidence_signer_fingerprint,
            request_digest=sha256_digest(canonical_json_bytes(normalized_request)),
            record_count=len(prepared.schedule.records),
            providers={
                "baseline": request.comparison.baseline.runtime.provider,
                "subject": request.comparison.subject.runtime.provider,
            },
            checks=tuple(checks),
            runtime_image_digests=normalized_runtime_digests,
            sample_qualification=sample_qualification,
        )
    except EvaluationPreflightError:
        raise
    except (
        EvaluationRequestError,
        EvaluationTransactionError,
        EvidencePackError,
        KeyError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        raise EvaluationPreflightError(str(exc)) from exc


def evaluate_request_file(
    request_path: Path | EvaluationRequest,
    *,
    signing_key_path: Path | None,
    resource_resolver: RuntimeResourceResolver | None = None,
    runtime_executor: RuntimeComparisonExecutor | None = None,
    runtime_image_digests: Mapping[str, str] | None = None,
    scorer_registry: ScorerExtensionRegistry | None = None,
    registry: CoreRegistry | None = None,
) -> EvaluationTransactionResult:
    """Execute or import, authenticate, and publish one closed request."""

    try:
        prepared = _prepare_evaluation_inputs(
            request_path,
            signing_key_path=signing_key_path,
            scorer_registry=scorer_registry,
            registry=registry,
        )
        signing_key = prepared.signing_key
        registry = prepared.registry
        request = prepared.request
        schedule = prepared.schedule
        canonical_schedule = prepared.canonical_schedule
        policy_bytes = prepared.policy_bytes
        policy_digest = prepared.policy_digest
        selected_metric = prepared.selected_metric
        observations = prepared.observations
        execution_resolver = resource_resolver
        preflight_resolver = resource_resolver
        if request.execution.mode == "run":
            if runtime_executor is not None and resource_resolver is not None:
                raise EvaluationTransactionError(
                    "runtime executor and direct resource resolver are mutually exclusive"
                )
            if runtime_executor is not None and callable(
                getattr(runtime_executor, "resolve", None)
            ):
                preflight_resolver = runtime_executor  # type: ignore[assignment]
            if runtime_executor is None and execution_resolver is None:
                execution_resolver = caller_runtime_resources_from_environment()
                preflight_resolver = execution_resolver
            if runtime_image_digests is None:
                raise EvaluationTransactionError(
                    "run evaluation requires both preflight runtime image digests"
                )
        preflight_result = preflight_evaluation_request(
            request,
            signing_key_path=signing_key_path,
            scorer_registry=scorer_registry,
            runtime_image_digests=runtime_image_digests,
            resource_resolver=preflight_resolver,
            registry=registry,
        )
        output_probe = _prepare_output_parent(request.root, request.output.evidence)
        try:
            _revalidate_output_parent(
                output_probe,
                request.output.evidence,
                published=False,
            )
        finally:
            os.close(output_probe.descriptor)
        if request.execution.mode == "import":
            assert request.execution.records is not None
            assert request.execution.baseline is not None
            assert request.execution.subject is not None
            baseline_evidence = _side_evidence(
                request, request.execution.baseline, side="baseline"
            )
            subject_evidence = _side_evidence(
                request, request.execution.subject, side="subject"
            )
        else:
            if runtime_executor is not None:
                executed = runtime_executor.execute(
                    request,
                    registry=registry,
                    schedule_bytes=canonical_schedule,
                    policy_digest=policy_digest,
                )
            else:
                assert execution_resolver is not None
                executed = execute_runtime_comparison(
                    request,
                    registry=registry,
                    resource_resolver=execution_resolver,
                    schedule_bytes=canonical_schedule,
                    policy_digest=policy_digest,
                )
            baseline_evidence = executed.baseline
            subject_evidence = executed.subject
        # Import and live workers converge at the same host-side verifier.  A
        # worker-reported digest is never accepted without independently
        # validating all six files and reproducing the artifact identity.
        baseline_runtime = _validate_import_side(
            request.comparison.baseline,
            baseline_evidence,
            side="baseline",
            provider=registry.get_runtime_provider(
                request.comparison.baseline.runtime.provider
            ),
            task=request.comparison.task,
            metric=request.comparison.collection_metric,
            schedule=schedule,
            policy_digest=policy_digest,
        )
        subject_runtime = _validate_import_side(
            request.comparison.subject,
            subject_evidence,
            side="subject",
            provider=registry.get_runtime_provider(
                request.comparison.subject.runtime.provider
            ),
            task=request.comparison.task,
            metric=request.comparison.collection_metric,
            schedule=schedule,
            policy_digest=policy_digest,
        )
        if request.execution.mode == "run":
            expected_runtime_digests = preflight_result.runtime_image_digests
            assert expected_runtime_digests is not None
            if baseline_runtime != expected_runtime_digests["baseline"]:
                raise EvaluationTransactionError(
                    "baseline validated runtime digest does not match preflight"
                )
            if subject_runtime != expected_runtime_digests["subject"]:
                raise EvaluationTransactionError(
                    "subject validated runtime digest does not match preflight"
                )
            if baseline_runtime != executed.baseline_runtime_digest:
                raise EvaluationTransactionError(
                    "baseline worker runtime digest does not match its validated receipt"
                )
            if subject_runtime != executed.subject_runtime_digest:
                raise EvaluationTransactionError(
                    "subject worker runtime digest does not match its validated receipt"
                )
        normalized = _normalized_request(request, schedule, observations)
        baseline_digest = sha256_digest(baseline_evidence.artifact_identity)
        subject_digest = sha256_digest(subject_evidence.artifact_identity)
        derived = derive_paired_records(
            schedule=schedule,
            metric=selected_metric,
            baseline=baseline_evidence,
            subject=subject_evidence,
            baseline_identity_digest=baseline_digest,
            subject_identity_digest=subject_digest,
            baseline_runtime_digest=baseline_runtime,
            subject_runtime_digest=subject_runtime,
            scorer_binding=request.comparison.scorer_extension,
            scorer_registry=scorer_registry,
        )
        if request.execution.mode == "import":
            assert request.execution.records is not None
            imported_records_raw = _read_request_file(
                request.root,
                request.execution.records,
                label="imported paired records",
            )
            imported_records = _parse_object(
                imported_records_raw, label="imported paired records"
            )
            if imported_records_raw != canonical_json_bytes(imported_records):
                raise EvaluationTransactionError(
                    "imported paired records must use canonical JSON"
                )
            if imported_records != derived:
                raise EvaluationTransactionError(
                    "imported paired records do not equal verifier-derived pairs"
                )

        comparison_id = _comparison_id(
            normalized,
            baseline_identity=baseline_evidence.artifact_identity,
            subject_identity=subject_evidence.artifact_identity,
            schedule_payload=canonical_schedule,
            policy_payload=policy_bytes,
            baseline_runtime_digest=baseline_runtime,
            subject_runtime_digest=subject_runtime,
            paired_records=derived,
        )
        output_anchor = _prepare_output_parent(request.root, request.output.evidence)
        try:
            _revalidate_output_parent(
                output_anchor, request.output.evidence, published=False
            )
            publication = publish_comparison_evidence(
                request.output.evidence,
                comparison_id=comparison_id,
                baseline=InputIdentity(
                    digest=baseline_digest,
                    locator=_stable_locator(
                        request.comparison.baseline.artifact.locator,
                        label="baseline artifact locator",
                    ),
                    media_type="application/json",
                ),
                subject=InputIdentity(
                    digest=subject_digest,
                    locator=_stable_locator(
                        request.comparison.subject.artifact.locator,
                        label="subject artifact locator",
                    ),
                    media_type="application/json",
                ),
                dataset=InputIdentity(
                    digest=f"sha256:{schedule.schedule_sha256}",
                    locator=EVIDENCE_PATHS["schedule"],
                    media_type="application/json",
                ),
                baseline_runtime=InputIdentity(
                    digest=baseline_runtime,
                    locator=f"runtime:{baseline_runtime}",
                ),
                subject_runtime=InputIdentity(
                    digest=subject_runtime,
                    locator=f"runtime:{subject_runtime}",
                ),
                policy=InputIdentity(
                    digest=sha256_digest(policy_bytes),
                    locator="inputs/policy.json",
                    media_type="application/json",
                ),
                normalized_request=normalized,
                schedule=schedule,
                policy_bytes=policy_bytes,
                baseline_evidence=baseline_evidence,
                subject_evidence=subject_evidence,
                signing_key_path=signing_key,
                observations=observations,
                scorer_registry=scorer_registry,
                expected_paired_records=derived,
            )
            _revalidate_output_parent(
                output_anchor, request.output.evidence, published=True
            )
        finally:
            output_anchor.close()
    except EvaluationTransactionError:
        raise
    except (
        EvaluationRequestError,
        EvidencePackError,
        KeyError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        raise EvaluationTransactionError(str(exc)) from exc
    return EvaluationTransactionResult(
        evidence_path=publication.evidence_path.resolve(),
        comparison_id=comparison_id,
        pack_manifest_digest=publication.pack_manifest_digest,
    )


__all__ = [
    "EvaluationPreflightError",
    "EvaluationPreflightResult",
    "EvaluationTransactionError",
    "EvaluationTransactionResult",
    "evaluate_request_file",
    "preflight_evaluation_request",
]
