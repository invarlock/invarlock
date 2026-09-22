"""Offline judge collection through authenticated first-party runtime providers."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

from invarlock.core.runtime_provider import (
    EvaluationBatch,
    EvaluationInputPart,
    EvaluationRecord,
    ModelRuntimeSpec,
    RuntimeArtifactResources,
    RuntimeProvider,
    RuntimeScoringRecord,
    artifact_identity_sha256,
    evaluation_input_parts_sha256,
)
from invarlock.core.runtime_provider.behavioral_observation import (
    runtime_scoring_records_sha256,
)
from invarlock.evidence_pack_json import parse_json_bytes, read_regular_file_bytes
from invarlock.filesystem.atomic_file import write_file_no_replace
from invarlock.judge_measurement_types import JudgeMeasurementPlan, JudgeMeasurements
from invarlock.runtime_provider_evidence import (
    encode_artifact_identity,
    encode_runtime_provider_receipt,
    encode_scoring_observation,
    runtime_provider_evidence_errors,
    runtime_request_binding_errors,
)
from invarlock.runtime_security_helpers import (
    network_allowed,
    remote_code_allowed,
    strict_container_boundary_present,
    third_party_plugins_allowed,
)

from .contracts import (
    JUDGE_REQUEST_MAX_BYTES,
    MEASUREMENTS_FORMAT,
    MEASUREMENTS_MAX_BYTES,
    RUNTIME_PROVIDER_SOURCE_FORMAT,
    _check_attempts,
    _runtime_provider_source_errors,
    _validate_frozen_answer_bindings,
    _validate_measurement_trial_shape,
    canonical_payload,
    expected_trial_id,
    measurement_plan_digest,
    render_judge_request,
    validate_measurement_plan,
    validate_measurements,
)

_SUPPORTED_PROVIDERS = frozenset({"hf_transformers", "llama_cpp"})
RUNTIME_PROVIDER_COLLECTION_PROFILE = "runtime-provider-text-frozen-answer-v1"
_SOURCE_MAX_BYTES = 16 * 1024 * 1024
# A source is embedded as a JSON string and its trial is repeated in the outer
# contract. Reserving twice the source maximum plus one source maximum for the
# repeated trial is deliberately conservative and makes final serialization
# admission-safe even for escape-heavy model output.
_RETAINED_BYTES_PER_TRIAL = 3 * _SOURCE_MAX_BYTES
_MEASUREMENTS_ENVELOPE_BYTES = 1024 * 1024
_MAX_CONTEXT_LENGTH = 1024 * 1024
_MAX_TIMEOUT_SECONDS = 604800


class RuntimeProviderJudgeError(ValueError):
    """A local judge cannot be collected under the strict runtime profile."""


@dataclass(frozen=True)
class RuntimeProviderJudgeOptions:
    """Bounded retained-source identity for one local collection."""

    source_id: str = "runtime-provider-judge"
    scorer_id: str = "judge"
    checkpoint_directory: Path | None = None

    def validate(self) -> None:
        for label, value in (
            ("source_id", self.source_id),
            ("scorer_id", self.scorer_id),
        ):
            if (
                not isinstance(value, str)
                or not 1 <= len(value) <= 128
                or any(
                    ord(character) < 32 or ord(character) == 127 for character in value
                )
            ):
                raise RuntimeProviderJudgeError(
                    f"runtime judge {label} must be a bounded printable identifier"
                )
        if self.checkpoint_directory is not None and not isinstance(
            self.checkpoint_directory, Path
        ):
            raise RuntimeProviderJudgeError(
                "runtime judge checkpoint_directory must be a Path"
            )


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _decoded_object(payload: bytes, *, label: str) -> dict[str, Any]:
    value = parse_json_bytes(payload, label=label)
    if not isinstance(value, dict):  # pragma: no cover - typed encoders guarantee this
        raise RuntimeProviderJudgeError(f"{label} must encode an object")
    return cast(dict[str, Any], value)


def _checkpoint_identity(
    *,
    plan_digest: str,
    spec: ModelRuntimeSpec,
    resources: RuntimeArtifactResources,
    artifact_digest: str,
    options: RuntimeProviderJudgeOptions,
) -> bytes:
    return canonical_payload(
        {
            "format": "invarlock/runtime-provider-judge-checkpoint-v1",
            "plan_sha256": plan_digest,
            "runtime_spec": {
                "provider_name": spec.provider_name,
                "model_id": spec.model_id,
                "settings": dict(spec.settings),
            },
            "artifact_identity_sha256": artifact_digest,
            "outer_image_digest": resources.container_image_digest,
            "device_kind": resources.device_kind,
            "source_id": options.source_id,
            "scorer_id": options.scorer_id,
        }
    )


def _prepare_checkpoint(
    options: RuntimeProviderJudgeOptions, identity: bytes, *, batch_count: int
) -> list[dict[str, Any]]:
    directory = options.checkpoint_directory
    if directory is None:
        return []
    try:
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        metadata = directory.stat(follow_symlinks=False)
    except OSError as exc:
        raise RuntimeProviderJudgeError(
            "runtime judge checkpoint directory is unavailable"
        ) from exc
    if not directory.is_dir() or metadata.st_mode & 0o077:
        raise RuntimeProviderJudgeError(
            "runtime judge checkpoint directory must be private (0700)"
        )
    identity_path = directory / "identity.json"
    if identity_path.exists():
        observed = read_regular_file_bytes(
            identity_path,
            label="runtime judge checkpoint identity",
            max_bytes=1024 * 1024,
        )
        if observed != identity:
            raise RuntimeProviderJudgeError(
                "runtime judge checkpoint belongs to a different collection"
            )
    else:
        write_file_no_replace(identity_path, identity)
    allowed_names = {"identity.json"}
    for index in range(batch_count):
        suffix = f"{index + 1:06d}.json"
        allowed_names.update({f"admission-{suffix}", f"result-{suffix}"})
    try:
        observed_names = {entry.name for entry in directory.iterdir()}
    except OSError as exc:
        raise RuntimeProviderJudgeError(
            "runtime judge checkpoint directory cannot be enumerated"
        ) from exc
    unexpected = observed_names - allowed_names
    if unexpected:
        raise RuntimeProviderJudgeError(
            "runtime judge checkpoint contains an unexpected entry"
        )
    completed: list[dict[str, Any]] = []
    gap = False
    for index in range(batch_count):
        suffix = f"{index + 1:06d}.json"
        result_path = directory / f"result-{suffix}"
        admission_path = directory / f"admission-{suffix}"
        if gap and (result_path.exists() or admission_path.exists()):
            raise RuntimeProviderJudgeError(
                "runtime judge checkpoint contains a non-contiguous shard"
            )
        if result_path.exists():
            if not admission_path.exists():
                raise RuntimeProviderJudgeError(
                    "runtime judge checkpoint result lacks prior admission"
                )
            admission = read_regular_file_bytes(
                admission_path,
                label="runtime judge checkpoint admission",
                max_bytes=1024,
            )
            if admission != canonical_payload(
                {
                    "format": "invarlock/runtime-provider-judge-admission-v1",
                    "batch": index + 1,
                }
            ):
                raise RuntimeProviderJudgeError(
                    "runtime judge checkpoint admission is invalid"
                )
            raw = read_regular_file_bytes(
                result_path,
                label="runtime judge checkpoint result",
                max_bytes=_SOURCE_MAX_BYTES,
            )
            decoded = parse_json_bytes(raw, label="runtime judge checkpoint result")
            if not isinstance(decoded, dict):
                raise RuntimeProviderJudgeError(
                    "runtime judge checkpoint result must be an object"
                )
            if canonical_payload(decoded) != raw:
                raise RuntimeProviderJudgeError(
                    "runtime judge checkpoint result must use canonical JSON"
                )
            completed.append(cast(dict[str, Any], decoded))
            continue
        if admission_path.exists():
            raise RuntimeProviderJudgeError(
                "runtime judge has an admitted inference without a retained result; "
                "the outcome is ambiguous and cannot be retried"
            )
        gap = True
    return completed


def _source_id(options: RuntimeProviderJudgeOptions, index: int, count: int) -> str:
    return (
        options.source_id
        if count == 1
        else f"{options.source_id[:115]}-{index + 1:06d}"
    )


def _validate_cached_sources(
    sources: list[dict[str, Any]],
    *,
    pending: list[tuple[dict[str, Any], bytes, EvaluationRecord]],
    plan: JudgeMeasurementPlan,
    spec: ModelRuntimeSpec,
    resources: RuntimeArtifactResources,
    options: RuntimeProviderJudgeOptions,
) -> None:
    required_source = {
        "format",
        "runtime_spec",
        "artifact_identity",
        "scoring_observation",
        "provider_receipt",
        "trials",
    }
    for index, source in enumerate(sources):
        if (
            set(source) != required_source
            or source.get("format") != RUNTIME_PROVIDER_SOURCE_FORMAT
        ):
            raise RuntimeProviderJudgeError(
                "runtime judge checkpoint source has an unsupported shape"
            )
        if source["runtime_spec"] != {
            "provider_name": spec.provider_name,
            "model_id": spec.model_id,
            "settings": dict(spec.settings),
        }:
            raise RuntimeProviderJudgeError(
                "runtime judge checkpoint spec differs from the current request"
            )
        errors = _runtime_provider_source_errors(source, cast(dict[str, Any], plan))
        if errors:
            raise RuntimeProviderJudgeError(errors[0])
        receipt = cast(dict[str, Any], source["provider_receipt"])
        device = receipt.get("device")
        if (
            receipt.get("outer_image_digest") != resources.container_image_digest
            or not isinstance(device, dict)
            or device.get("device_kind") != resources.device_kind
        ):
            raise RuntimeProviderJudgeError(
                "runtime judge checkpoint execution resources differ from the current request"
            )
        trials = source.get("trials")
        if (
            not isinstance(trials, list)
            or len(trials) != 1
            or not isinstance(trials[0], dict)
        ):
            raise RuntimeProviderJudgeError(
                "runtime judge checkpoint source must contain one trial"
            )
        trial = cast(dict[str, Any], trials[0])
        try:
            _validate_measurement_trial_shape(trial)
        except ValueError as exc:
            raise RuntimeProviderJudgeError(str(exc)) from exc
        expected, request, _ = pending[index]
        for name, value in expected.items():
            if trial.get(name) != value:
                raise RuntimeProviderJudgeError(
                    "runtime judge checkpoint trial differs from its planned shard"
                )
        source_id = _source_id(options, index, len(pending))
        try:
            _check_attempts(
                trial,
                plan=cast(dict[str, Any], plan),
                source_ids={source_id},
                expected_request_sha256=_sha256(request),
            )
        except ValueError as exc:
            raise RuntimeProviderJudgeError(str(exc)) from exc
        attempts = trial.get("attempts")
        if (
            not isinstance(attempts, list)
            or len(attempts) != 1
            or attempts[0].get("source")
            != {
                "source_id": source_id,
                "scorer_id": options.scorer_id,
                "model_event_id": trial["trial_id"],
                "record_index": 0,
                "attempt_index": 0,
            }
        ):
            raise RuntimeProviderJudgeError(
                "runtime judge checkpoint source mapping is invalid"
            )


def _admit_checkpoint_batch(options: RuntimeProviderJudgeOptions, index: int) -> None:
    if options.checkpoint_directory is None:
        return
    write_file_no_replace(
        options.checkpoint_directory / f"admission-{index + 1:06d}.json",
        canonical_payload(
            {
                "format": "invarlock/runtime-provider-judge-admission-v1",
                "batch": index + 1,
            }
        ),
    )


def _retain_checkpoint_batch(
    options: RuntimeProviderJudgeOptions, index: int, source: dict[str, Any]
) -> None:
    if options.checkpoint_directory is None:
        return
    write_file_no_replace(
        options.checkpoint_directory / f"result-{index + 1:06d}.json",
        canonical_payload(source),
    )


def _local_plan_constraints(
    plan: JudgeMeasurementPlan,
    *,
    spec: ModelRuntimeSpec,
    artifact_digest: str,
) -> None:
    judge = plan["judge"]
    config = judge["config"]
    schedule = plan["schedule"]
    if spec.provider_name not in _SUPPORTED_PROVIDERS:
        raise RuntimeProviderJudgeError(
            "runtime judge provider must be hf_transformers or llama_cpp"
        )
    if judge["provider"] != spec.provider_name:
        raise RuntimeProviderJudgeError(
            "runtime judge provider differs from the approved plan"
        )
    if judge["requested_model"] != spec.model_id or judge[
        "approved_resolved_models"
    ] != [spec.model_id]:
        raise RuntimeProviderJudgeError(
            "runtime judge model must be the sole approved resolved model"
        )
    identity = judge["model_identity"]
    if identity != {"kind": "local_weights", "weights_sha256": artifact_digest}:
        raise RuntimeProviderJudgeError(
            "runtime judge plan must bind the complete artifact identity digest"
        )
    if (
        config["temperature"] != "0"
        or config["top_p"] != "1"
        or config["reasoning_effort"] is not None
        or config["seed"] is None
    ):
        raise RuntimeProviderJudgeError(
            "runtime judge requires temperature=0, top_p=1, an explicit seed, "
            "and reasoning_effort=null"
        )
    if (
        schedule["max_attempts"] != 1
        or schedule["retry_on"]
        or schedule["cache"] != "forbid"
    ):
        raise RuntimeProviderJudgeError(
            "runtime judge requires one uncached attempt without retries"
        )


def validate_runtime_provider_collection(
    configuration: dict[str, Any], plan: JudgeMeasurementPlan
) -> dict[str, int]:
    """Validate closed local collection reserves without loading a model."""

    if set(configuration) != {"profile", "max_calls", "max_output_tokens"}:
        raise RuntimeProviderJudgeError(
            "runtime judge collection must contain profile, max_calls, and "
            "max_output_tokens"
        )
    if configuration["profile"] != RUNTIME_PROVIDER_COLLECTION_PROFILE:
        raise RuntimeProviderJudgeError("unsupported runtime judge collection profile")
    limits: dict[str, int] = {}
    for name, maximum in (("max_calls", 200_000), ("max_output_tokens", 10**12)):
        value = configuration[name]
        if type(value) is not int or not 1 <= value <= maximum:
            raise RuntimeProviderJudgeError(
                f"runtime judge {name} must be a bounded positive integer"
            )
        limits[name] = value
    expected = plan["schedule"]["expected_trials"]
    reserved_output = expected * plan["judge"]["config"]["max_output_tokens"]
    if limits["max_calls"] < expected or limits["max_output_tokens"] < reserved_output:
        raise RuntimeProviderJudgeError(
            "runtime judge collection must reserve every planned inference"
        )
    return limits


def _retained_measurement_size(
    sources: list[dict[str, Any]],
    *,
    plan: JudgeMeasurementPlan,
    plan_digest: str,
) -> int:
    retained_sources = []
    trials = []
    for source in sources:
        encoded = canonical_payload(source)
        trial = source["trials"][0]
        source_id = trial["attempts"][0]["source"]["source_id"]
        retained_sources.append(
            {
                "source_id": source_id,
                "profile": RUNTIME_PROVIDER_SOURCE_FORMAT,
                "encoding": "utf-8",
                "byte_size": len(encoded),
                "media_type": "application/json",
                "content": encoded.decode("utf-8"),
                "sha256": _sha256(encoded),
            }
        )
        trials.append(trial)
    return len(
        canonical_payload(
            {
                "format": MEASUREMENTS_FORMAT,
                "profile_id": plan["profile_id"],
                "plan_sha256": plan_digest,
                "source_profile": RUNTIME_PROVIDER_SOURCE_FORMAT,
                "sources": retained_sources,
                "trials": trials,
                "completeness": {
                    "status": "incomplete",
                    "expected_trials": plan["schedule"]["expected_trials"],
                    "recorded_trials": len(trials),
                    "completed_trials": sum(
                        trial["status"] == "complete" for trial in trials
                    ),
                },
            }
        )
    )


def preflight_runtime_provider(
    plan: JudgeMeasurementPlan,
    *,
    provider: RuntimeProvider,
    spec: ModelRuntimeSpec,
    resources: RuntimeArtifactResources,
) -> dict[str, Any]:
    """Authenticate a local judge and strict boundary without loading the model."""

    validate_measurement_plan(plan)
    if (
        not strict_container_boundary_present()
        or network_allowed()
        or remote_code_allowed()
        or third_party_plugins_allowed()
    ):
        raise RuntimeProviderJudgeError(
            "runtime judge collection requires execution inside the authenticated "
            "strict offline container"
        )
    if provider.name != spec.provider_name:
        raise RuntimeProviderJudgeError(
            "runtime judge provider instance differs from its model spec"
        )
    provider.validate_config(spec)
    settings = spec.settings
    context_length = settings.get("context_length")
    batch_size = settings.get("batch_size")
    max_output_tokens = settings.get("max_output_tokens")
    timeout_seconds = settings.get("timeout_seconds")
    if (
        type(context_length) is not int
        or not 1 <= context_length <= _MAX_CONTEXT_LENGTH
        or type(batch_size) is not int
        or batch_size != 1
        or type(max_output_tokens) is not int
        or max_output_tokens != plan["judge"]["config"]["max_output_tokens"]
        or type(settings.get("seed")) is not int
        or settings.get("seed") != plan["judge"]["config"]["seed"]
        or type(timeout_seconds) is not int
        or not 1 <= timeout_seconds <= _MAX_TIMEOUT_SECONDS
    ):
        raise RuntimeProviderJudgeError(
            "runtime judge requires bounded context_length and timeout_seconds, "
            "batch_size=1, and the planned max_output_tokens"
        )
    if max_output_tokens > context_length:
        raise RuntimeProviderJudgeError(
            "runtime judge max_output_tokens cannot exceed context_length"
        )
    artifact = provider.authenticate_artifact(spec, resources.primary_path())
    artifact_digest = artifact_identity_sha256(artifact)
    _local_plan_constraints(plan, spec=spec, artifact_digest=artifact_digest)
    capabilities = provider.capabilities()
    if (
        capabilities.provider_name != spec.provider_name
        or "text_causal" not in capabilities.tasks
        or "exact_match" not in capabilities.metrics
    ):
        raise RuntimeProviderJudgeError(
            "runtime judge provider lacks text_causal exact-match collection"
        )
    return {
        "provider": spec.provider_name,
        "model_id": spec.model_id,
        "artifact_identity_sha256": artifact_digest,
        "outer_image_digest": resources.container_image_digest,
        "device_kind": resources.device_kind,
        "strict_offline_container": True,
        "model_loaded": False,
    }


def _input_record(trial_id: str, request: bytes) -> EvaluationRecord:
    text = request.decode("utf-8")
    part = EvaluationInputPart(
        kind="text",
        role="prompt",
        text=text,
        sha256=_sha256(request),
    )
    parts = (part,)
    return EvaluationRecord(
        record_id=trial_id,
        input_text=text,
        input_sha256=evaluation_input_parts_sha256(parts),
        input_parts=parts,
    )


def _parse_result(output: str, plan: JudgeMeasurementPlan) -> dict[str, str | None]:
    ratings = {item["label"]: item["value"] for item in plan["scale"]["ratings"]}
    try:
        decoded = parse_json_bytes(
            output.encode("utf-8"), label="runtime judge response"
        )
    except ValueError:
        decoded = None
    if (
        isinstance(decoded, dict)
        and set(decoded) == {"rating"}
        and isinstance(decoded["rating"], str)
        and decoded["rating"] in ratings
    ):
        rating = decoded["rating"]
        return {"status": "ok", "rating": rating, "value": ratings[rating]}
    return {"status": "invalid", "rating": None, "value": None}


def _attempt(
    *,
    record: RuntimeScoringRecord,
    request: bytes,
    spec: ModelRuntimeSpec,
    options: RuntimeProviderJudgeOptions,
    source_id: str,
    record_index: int,
) -> dict[str, Any]:
    request_blob = {
        "media_type": "application/json",
        "text": request.decode("utf-8"),
        "sha256": _sha256(request),
    }
    response = None
    error = None
    if record.status == "ok":
        if record.output_text is None or record.output_sha256 != _sha256(
            record.output_text.encode("utf-8")
        ):
            raise RuntimeProviderJudgeError(
                "runtime judge output text and digest are incomplete"
            )
        if len(record.output_text.encode("utf-8")) > JUDGE_REQUEST_MAX_BYTES:
            raise RuntimeProviderJudgeError("runtime judge response exceeds 1 MiB")
        response = {
            "media_type": "application/json",
            "text": record.output_text,
            "sha256": record.output_sha256,
        }
        status = "completed"
    else:
        status = "cancelled"
        error = {
            "code": "runtime-provider-error",
            "message": f"runtime provider record failed with code {record.error_code}",
        }
    return {
        "attempt": 1,
        "role": "judge",
        "resolved_model": spec.model_id if status == "completed" else None,
        "status": status,
        "request": request_blob,
        "response": response,
        "request_id": None,
        "finish_reason": None,
        "error": error,
        "usage": None,
        "cache": "none",
        "source": {
            "source_id": source_id,
            "scorer_id": options.scorer_id,
            "model_event_id": record.record_id,
            "record_index": record_index,
            "attempt_index": 0,
        },
    }


def collect_runtime_provider(
    plan: JudgeMeasurementPlan,
    *,
    provider: RuntimeProvider,
    spec: ModelRuntimeSpec,
    resources: RuntimeArtifactResources,
    baseline_run: dict[str, Any],
    subject_run: dict[str, Any],
    options: RuntimeProviderJudgeOptions = RuntimeProviderJudgeOptions(),
) -> JudgeMeasurements:
    """Collect one complete frozen-answer schedule without network access.

    The exact normalized judge request JSON is the model prompt. The retained
    source embeds the provider's artifact identity, scoring observation and
    receipt so later verification requires no provider installation.
    """

    options.validate()
    _validate_frozen_answer_bindings(plan, baseline_run, subject_run)

    runs = {
        "baseline": {row["id"]: row for row in baseline_run.get("records", [])},
        "subject": {row["id"]: row for row in subject_run.get("records", [])},
    }
    plan_digest = measurement_plan_digest(plan)
    pending: list[tuple[dict[str, Any], bytes, EvaluationRecord]] = []
    for binding in sorted(plan["answer_bindings"], key=lambda item: item["case_id"]):
        case_id = binding["case_id"]
        for side in ("baseline", "subject"):
            try:
                frozen = runs[side][case_id]
            except KeyError as exc:
                raise RuntimeProviderJudgeError(
                    "runtime judge frozen runs do not match the plan"
                ) from exc
            for repetition in range(1, plan["schedule"]["repetitions"] + 1):
                trial_id = expected_trial_id(plan_digest, case_id, side, repetition)
                request = render_judge_request(
                    plan,
                    input_text=frozen.get("input"),
                    answer_text=frozen.get("output"),
                    reference_text=frozen.get("expected"),
                )
                context_length = cast(int, spec.settings["context_length"])
                if (
                    len(request) + plan["judge"]["config"]["max_output_tokens"]
                    > context_length
                ):
                    raise RuntimeProviderJudgeError(
                        "runtime judge prompt and output reservation exceed context_length"
                    )
                pending.append(
                    (
                        {
                            "trial_id": trial_id,
                            "case_id": case_id,
                            "side": side,
                            "repetition": repetition,
                            "answer_sha256": (
                                binding["baseline_answer_sha256"]
                                if side == "baseline"
                                else binding["subject_answer_sha256"]
                            ),
                            "plan_sha256": plan_digest,
                        },
                        request,
                        _input_record(trial_id, request),
                    )
                )

    # All frozen bindings, normalized prompts, context bounds, and aggregate
    # retained capacity are checked before artifact scanning or model loading.
    validate_runtime_provider_collection(
        {
            "profile": RUNTIME_PROVIDER_COLLECTION_PROFILE,
            "max_calls": len(pending),
            "max_output_tokens": len(pending)
            * plan["judge"]["config"]["max_output_tokens"],
        },
        plan,
    )
    preflight = preflight_runtime_provider(
        plan, provider=provider, spec=spec, resources=resources
    )
    artifact = provider.identify_artifact(spec)
    artifact_digest = artifact_identity_sha256(artifact)
    if artifact_digest != preflight["artifact_identity_sha256"]:
        raise RuntimeProviderJudgeError(
            "runtime judge artifact identity changed after authentication"
        )

    checkpoint_identity = _checkpoint_identity(
        plan_digest=plan_digest,
        spec=spec,
        resources=resources,
        artifact_digest=artifact_digest,
        options=options,
    )
    artifact_bytes = encode_artifact_identity(artifact)
    if len(pending) > 1000:
        raise RuntimeProviderJudgeError(
            "runtime judge collection exceeds the 1000-source retained limit"
        )
    if (
        _MEASUREMENTS_ENVELOPE_BYTES + len(pending) * _RETAINED_BYTES_PER_TRIAL
        > MEASUREMENTS_MAX_BYTES
        and options.checkpoint_directory is None
    ):
        raise RuntimeProviderJudgeError(
            "runtime judge schedules above the in-memory worst-case retained capacity "
            "require a checkpoint directory"
        )
    source_values = _prepare_checkpoint(
        options, checkpoint_identity, batch_count=len(pending)
    )
    _validate_cached_sources(
        source_values,
        pending=pending,
        plan=plan,
        spec=spec,
        resources=resources,
        options=options,
    )
    start = len(source_values)
    session = None
    context = None
    if start < len(pending):
        context = provider.prepare_execution(spec, resources)
        if (
            not context.strict
            or context.allow_network
            or context.container_image_digest != resources.container_image_digest
            or context.device_kind != resources.device_kind
            or context.artifact_identity_sha256 != artifact_digest
        ):
            if context.close_callback is not None:
                context.close_callback()
            raise RuntimeProviderJudgeError(
                "runtime judge requires a strict offline execution context"
            )
        try:
            session = provider.open(spec, context)
        except BaseException:
            if context.close_callback is not None:
                context.close_callback()
            raise
    try:
        for index in range(start, len(pending)):
            trial, request, expected_record = pending[index]
            source_id = _source_id(options, index, len(pending))
            retained_bytes = _retained_measurement_size(
                source_values,
                plan=plan,
                plan_digest=plan_digest,
            )
            if (
                retained_bytes
                + _RETAINED_BYTES_PER_TRIAL
                + _MEASUREMENTS_ENVELOPE_BYTES
                > MEASUREMENTS_MAX_BYTES
            ):
                raise RuntimeProviderJudgeError(
                    "runtime judge retained capacity was exhausted before the next "
                    "inference admission; completed checkpoint shards remain durable"
                )
            _admit_checkpoint_batch(options, index)
            assert session is not None
            observation = session.score(
                EvaluationBatch(
                    schedule_sha256=plan_digest,
                    records=(expected_record,),
                    metric="exact_match",
                    task="text_causal",
                )
            )
            receipt = session.runtime_receipt()
            observation_bytes = encode_scoring_observation(observation)
            receipt_bytes = encode_runtime_provider_receipt(receipt)
            if (
                len(artifact_bytes) > 64 * 1024
                or len(observation_bytes) > 2 * 1024 * 1024
                or len(receipt_bytes) > 256 * 1024
            ):
                raise RuntimeProviderJudgeError(
                    "runtime judge provider evidence exceeds its retained reservation"
                )
            evidence_errors = runtime_provider_evidence_errors(
                artifact_identity=artifact,
                scoring_observation=observation,
                receipt=receipt,
                scoring_observation_bytes=observation_bytes,
                expected_outer_image_digest=resources.container_image_digest,
            )
            evidence_errors += runtime_request_binding_errors(
                provider_name=spec.provider_name,
                settings=spec.settings,
                artifact_identity=artifact,
                receipt=receipt,
            )
            if evidence_errors:
                raise RuntimeProviderJudgeError(evidence_errors[0])
            if receipt.plugin.distribution != "invarlock":
                raise RuntimeProviderJudgeError(
                    "runtime judge requires a first-party InvarLock provider receipt"
                )
            execution = receipt.execution_settings
            if (
                execution.allow_network
                or execution.seed != plan["judge"]["config"]["seed"]
                or execution.max_output_tokens
                != plan["judge"]["config"]["max_output_tokens"]
            ):
                raise RuntimeProviderJudgeError(
                    "runtime judge receipt differs from the approved generation settings"
                )
            if (
                observation.schedule_sha256 != plan_digest
                or len(observation.records) != 1
            ):
                raise RuntimeProviderJudgeError(
                    "runtime judge observation differs from the plan"
                )
            aggregate = runtime_scoring_records_sha256(
                [
                    cast(dict[str, object], asdict(record))
                    for record in observation.records
                ]
            )
            if observation.aggregate_source_sha256 != aggregate:
                raise RuntimeProviderJudgeError(
                    "runtime judge observation aggregate digest is invalid"
                )
            record = observation.records[0]
            if (
                record.record_id != expected_record.record_id
                or record.input_sha256 != expected_record.input_sha256
            ):
                raise RuntimeProviderJudgeError(
                    "runtime judge observation does not match its planned prompt"
                )
            attempt = _attempt(
                record=record,
                request=request,
                spec=spec,
                options=options,
                source_id=source_id,
                record_index=0,
            )
            parsed = (
                _parse_result(record.output_text, plan)
                if record.status == "ok" and record.output_text is not None
                else {"status": "unavailable", "rating": None, "value": None}
            )
            trial.update(
                status="complete" if parsed["status"] == "ok" else "incomplete",
                attempts=[attempt],
                selected_attempt=1 if record.status == "ok" else None,
                parse=parsed,
            )
            source_value = {
                "format": RUNTIME_PROVIDER_SOURCE_FORMAT,
                "runtime_spec": {
                    "provider_name": spec.provider_name,
                    "model_id": spec.model_id,
                    "settings": dict(spec.settings),
                },
                "artifact_identity": _decoded_object(
                    artifact_bytes, label="runtime judge artifact identity"
                ),
                "scoring_observation": _decoded_object(
                    observation_bytes, label="runtime judge scoring observation"
                ),
                "provider_receipt": _decoded_object(
                    receipt_bytes, label="runtime judge provider receipt"
                ),
                "trials": [trial],
            }
            source_bytes = canonical_payload(source_value)
            if len(source_bytes) > _SOURCE_MAX_BYTES:
                raise RuntimeProviderJudgeError(
                    f"runtime judge retained source exceeds {_SOURCE_MAX_BYTES} bytes"
                )
            _retain_checkpoint_batch(options, index, source_value)
            source_values.append(source_value)
    finally:
        if session is not None:
            session.close()

    trials = [source["trials"][0] for source in source_values]
    retained_sources = []
    for source in source_values:
        source_bytes = canonical_payload(source)
        source_id = source["trials"][0]["attempts"][0]["source"]["source_id"]
        retained_sources.append(
            {
                "source_id": source_id,
                "profile": RUNTIME_PROVIDER_SOURCE_FORMAT,
                "encoding": "utf-8",
                "byte_size": len(source_bytes),
                "media_type": "application/json",
                "content": source_bytes.decode("utf-8"),
                "sha256": _sha256(source_bytes),
            }
        )
    completed = sum(trial["status"] == "complete" for trial in trials)
    result = cast(
        JudgeMeasurements,
        {
            "format": MEASUREMENTS_FORMAT,
            "profile_id": plan["profile_id"],
            "plan_sha256": plan_digest,
            "source_profile": RUNTIME_PROVIDER_SOURCE_FORMAT,
            "sources": retained_sources,
            "trials": trials,
            "completeness": {
                "status": "complete" if completed == len(trials) else "incomplete",
                "expected_trials": plan["schedule"]["expected_trials"],
                "recorded_trials": len(trials),
                "completed_trials": completed,
            },
        },
    )
    if len(canonical_payload(result)) > MEASUREMENTS_MAX_BYTES:
        raise RuntimeProviderJudgeError(
            "runtime judge retained measurements exceed the global byte limit"
        )
    validate_measurements(
        result,
        plan,
        baseline_run=baseline_run,
        subject_run=subject_run,
    )
    return result


__all__ = [
    "RUNTIME_PROVIDER_COLLECTION_PROFILE",
    "RuntimeProviderJudgeError",
    "RuntimeProviderJudgeOptions",
    "collect_runtime_provider",
    "preflight_runtime_provider",
    "validate_runtime_provider_collection",
]
