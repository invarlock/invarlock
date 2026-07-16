"""Authenticated execute-or-import transaction for paired evaluation."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
    artifact_identity_sha256,
    build_runtime_behavioral_schedule,
)
from invarlock.core.schedule_preparation import (
    LocalDatasetRequest,
    local_dataset_preparation_payload,
    prepare_local_evaluation_schedule_bytes,
)
from invarlock.core.scorer_extension import (
    ScorerExtensionRegistry,
    scorer_binding_payload,
)
from invarlock.evaluation_run import (
    RuntimeComparisonExecutor,
    execute_runtime_comparison,
)
from invarlock.evaluation_runtime import (
    RuntimeResourceResolver,
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
    runtime_side_config_errors,
    schedule_bytes,
    sha256_digest,
)
from invarlock.evidence_pack_json import parse_json_bytes
from invarlock.evidence_pack_publication import _load_private_key
from invarlock.runtime_provider_evidence import (
    RuntimeProviderEvidenceError,
    decode_artifact_identity,
    decode_runtime_provider_receipt,
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

    def as_json(self) -> str:
        return json.dumps(
            {
                "format_version": "invarlock/evaluation-result-v1",
                "ok": True,
                "comparison_id": self.comparison_id,
                "evidence": str(self.evidence_path),
            },
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )


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


def evaluate_request_file(
    request_path: Path,
    *,
    signing_key_path: Path | None,
    resource_resolver: RuntimeResourceResolver | None = None,
    runtime_executor: RuntimeComparisonExecutor | None = None,
    scorer_registry: ScorerExtensionRegistry | None = None,
) -> EvaluationTransactionResult:
    """Execute or import, authenticate, and publish one closed request."""

    if signing_key_path is None:
        raise EvaluationTransactionError("an Ed25519 evidence-signing key is required")
    _require_closed_runtime_switches()
    try:
        signing_key = _load_private_key(Path(signing_key_path))
        registry = CoreRegistry()
        request = load_evaluation_request(
            request_path, provider_resolver=registry.get_runtime_provider
        )
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
        observations = _load_request_observations(request)
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
            if runtime_executor is not None and resource_resolver is not None:
                raise EvaluationTransactionError(
                    "runtime executor and direct resource resolver are mutually exclusive"
                )
            if runtime_executor is not None:
                executed = runtime_executor.execute(
                    request,
                    registry=registry,
                    schedule_bytes=canonical_schedule,
                    policy_digest=policy_digest,
                )
            else:
                resolver = (
                    resource_resolver
                    if resource_resolver is not None
                    else caller_runtime_resources_from_environment()
                )
                executed = execute_runtime_comparison(
                    request,
                    registry=registry,
                    resource_resolver=resolver,
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
            published = publish_comparison_evidence(
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
        evidence_path=published.resolve(), comparison_id=comparison_id
    )


__all__ = [
    "EvaluationTransactionError",
    "EvaluationTransactionResult",
    "evaluate_request_file",
]
