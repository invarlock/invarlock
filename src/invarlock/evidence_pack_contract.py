"""Closed contracts and verifier-owned replay for evidence-pack v1."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal, cast

from jsonschema import Draft202012Validator

from invarlock.core.runtime_provider import (
    RuntimeBehavioralSchedule,
    RuntimeScoringRecord,
    canonical_runtime_behavioral_schedule_json,
)
from invarlock.core.schedule_preparation import validate_local_dataset_preparation
from invarlock.core.scorer_extension import (
    SCORER_REPLAY_OUTPUT_KIND,
    AuthenticatedScorerRecord,
    ScorerExtensionBinding,
    ScorerExtensionError,
    ScorerExtensionRegistry,
    ScorerReplayRequest,
    decode_scorer_binding,
    scorer_binding_payload,
    scorer_result_payload,
)
from invarlock.evidence_pack_json import parse_json_bytes
from invarlock.paired_exact_match import (
    PAIRED_CONFIDENCE_INTERVAL_METHOD_V1,
    PAIRED_CONFIDENCE_INTERVAL_METHOD_V2,
    PairedExactMatchError,
    paired_exact_match_statistics,
)
from invarlock.public_contracts import (
    EVIDENCE_OBSERVATION_FORMAT_VERSION,
    load_evaluation_request_schema,
    load_evidence_observation_schema,
)
from invarlock.public_contracts import (
    EVIDENCE_PACK_FORMAT_VERSION as EVIDENCE_PACK_FORMAT,
)
from invarlock.runtime_provider_evidence import (
    RuntimeProviderEvidenceError,
    decode_artifact_identity,
    decode_runtime_provider_receipt,
    decode_scoring_observation,
    runtime_provider_evidence_errors,
)

EVIDENCE_PACK_VERIFY_FORMAT = "invarlock/evidence-pack-verify-v1"
EVIDENCE_INPUT_IDENTITY_FORMAT = "invarlock/evidence-input-identity-v1"
PAIRED_RECORDS_FORMAT = "invarlock/paired-records-v1"
LEGACY_COMPARISON_REPORT_FORMAT = "invarlock/comparison-report-v1"
COMPARISON_REPORT_FORMAT_V2 = "invarlock/comparison-report-v2"
COMPARISON_REPORT_FORMAT = "invarlock/comparison-report-v3"
COMPARISON_REPORT_FORMATS = frozenset(
    {
        LEGACY_COMPARISON_REPORT_FORMAT,
        COMPARISON_REPORT_FORMAT_V2,
        COMPARISON_REPORT_FORMAT,
    }
)
RUNTIME_SIDE_REPORT_FORMAT = "invarlock/runtime-side-report-v1"
RUNTIME_SIDE_CONFIG_FORMAT = "invarlock/runtime-side-config-v1"
EVIDENCE_OBSERVATION_FORMAT = EVIDENCE_OBSERVATION_FORMAT_VERSION

INPUT_ROLES = (
    "baseline",
    "subject",
    "dataset",
    "baseline_runtime",
    "subject_runtime",
    "policy",
)
EVIDENCE_PATHS = {
    "request": "request.json",
    "schedule": "schedule/runtime-behavioral-schedule.json",
    "evaluation_report": "reports/evaluation.report.json",
    "baseline_run_report": "runs/baseline/report.json",
    "subject_run_report": "runs/subject/report.json",
    "baseline_runtime_manifest": "providers/baseline/runtime.manifest.json",
    "subject_runtime_manifest": "providers/subject/runtime.manifest.json",
    "baseline_runtime_config": "providers/baseline/run.yaml",
    "subject_runtime_config": "providers/subject/run.yaml",
    "baseline_provider_identity": "providers/baseline/model-artifact.identity.json",
    "subject_provider_identity": "providers/subject/model-artifact.identity.json",
    "baseline_provider_receipt": "providers/baseline/runtime-provider.receipt.json",
    "subject_provider_receipt": "providers/subject/runtime-provider.receipt.json",
    "baseline_scoring_observation": (
        "providers/baseline/runtime-scoring.observation.json"
    ),
    "subject_scoring_observation": (
        "providers/subject/runtime-scoring.observation.json"
    ),
}

_DIGEST_RE = re.compile(r"sha256:[a-f0-9]{64}\Z")
_IDENTIFIER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
_OBSERVATION_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_OBSERVATION_KIND_RE = re.compile(r"[a-z][a-z0-9]*(?:[._:-][a-z0-9]+)*\Z")
MAX_IDENTITY_BYTES = 64 * 1024
MAX_EVIDENCE_BYTES = 64 * 1024 * 1024
MAX_OBSERVATION_BYTES = 1024 * 1024
MAX_OBSERVATIONS = 64
MAX_RECORDS = 10_000
PAIRED_INTERVAL_CONFIDENCE = 0.95
PAIRED_INTERVAL_REPLICATES = 2_048
PAIRED_INTERVAL_METHOD = "paired_percentile_bootstrap_sha256_v1"
DERIVED_PERPLEXITY_METHOD = "target_token_weighted_perplexity_ratio_v1"


class EvidencePackError(ValueError):
    """Raised when canonical evidence evidence is malformed or inconsistent."""


@dataclass(frozen=True)
class InputIdentity:
    """Authenticated digest and optional descriptive locator for one input."""

    digest: str
    locator: str | None = None
    media_type: str | None = None


@dataclass(frozen=True)
class RuntimeSideEvidence:
    """Exact evaluation bytes for one side of a comparison.

    These bytes are copied, not regenerated, so the runtime manifest continues
    to bind the same report, config, provider receipt, observation, and artifact
    identity that evaluation emitted.
    """

    run_report: bytes
    runtime_manifest: bytes
    runtime_config: bytes
    artifact_identity: bytes
    provider_receipt: bytes
    scoring_observation: bytes


@dataclass(frozen=True)
class EvidenceObservation:
    """One typed, observation-only JSON payload attached to a comparison.

    ``payload`` must already be canonical JSON bytes for an object. Publication
    wraps those bytes in a signed envelope with comparison, schedule, policy,
    and artifact bindings. Observations are intentionally absent from policy
    replay and therefore cannot change the comparison verdict.
    """

    observation_id: str
    scope: Literal["baseline", "subject", "comparison"]
    kind: str
    payload: bytes

    def __post_init__(self) -> None:
        if _OBSERVATION_ID_RE.fullmatch(self.observation_id) is None:
            raise EvidencePackError("observation_id is invalid")
        if _OBSERVATION_KIND_RE.fullmatch(self.kind) is None:
            raise EvidencePackError(
                f"observation {self.observation_id!r} kind is invalid"
            )
        if self.scope not in {"baseline", "subject", "comparison"}:
            raise EvidencePackError(
                f"observation {self.observation_id!r} scope is invalid"
            )
        if not isinstance(self.payload, bytes) or not self.payload:
            raise EvidencePackError(
                f"observation {self.observation_id!r} payload must be non-empty bytes"
            )
        if len(self.payload) > MAX_OBSERVATION_BYTES:
            raise EvidencePackError(
                f"observation {self.observation_id!r} exceeds the "
                f"{MAX_OBSERVATION_BYTES}-byte limit"
            )
        try:
            parsed = parse_json_bytes(
                self.payload, label=f"observation {self.observation_id} payload"
            )
        except ValueError as exc:
            raise EvidencePackError(str(exc)) from exc
        if not isinstance(parsed, dict):
            raise EvidencePackError(
                f"observation {self.observation_id!r} payload must be a JSON object"
            )
        if canonical_json_bytes(parsed) != self.payload:
            raise EvidencePackError(
                f"observation {self.observation_id!r} payload must use canonical JSON"
            )


def canonical_json_bytes(value: object, *, newline: bool = True) -> bytes:
    try:
        suffix = "\n" if newline else ""
        return (
            json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + suffix
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise EvidencePackError(f"value is not canonical JSON: {exc}") from exc


def sha256_digest(payload: bytes) -> str:
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _observation_schema_errors(payload: Mapping[str, object]) -> list[str]:
    errors = sorted(
        Draft202012Validator(load_evidence_observation_schema()).iter_errors(payload),
        key=lambda error: tuple(str(part) for part in error.absolute_path),
    )
    if not errors:
        return []
    first = errors[0]
    path = ".".join(str(part) for part in first.absolute_path) or "<root>"
    return [f"observation schema failed at {path}: {first.message}"]


def evidence_observation_bytes(
    observation: EvidenceObservation,
    *,
    comparison_id: str,
    schedule_digest: str,
    policy_digest: str,
    artifact_digests: Mapping[str, str],
) -> bytes:
    """Return one canonical observation envelope with complete bindings."""

    observation_id = observation.observation_id
    if _OBSERVATION_ID_RE.fullmatch(observation_id) is None:
        raise EvidencePackError("observation_id is invalid")
    if _OBSERVATION_KIND_RE.fullmatch(observation.kind) is None:
        raise EvidencePackError(f"observation {observation_id!r} kind is invalid")
    if observation.scope not in {"baseline", "subject", "comparison"}:
        raise EvidencePackError(f"observation {observation_id!r} scope is invalid")
    if set(artifact_digests) != {"baseline", "subject"}:
        raise EvidencePackError(
            "observation artifact digests must contain baseline and subject"
        )
    try:
        payload = parse_json_bytes(
            observation.payload,
            label=f"observation {observation_id} payload",
        )
    except ValueError as exc:
        raise EvidencePackError(str(exc)) from exc
    if not isinstance(payload, dict):
        raise EvidencePackError(
            f"observation {observation_id!r} payload must be a JSON object"
        )
    if canonical_json_bytes(payload) != observation.payload:
        raise EvidencePackError(
            f"observation {observation_id!r} payload must use canonical JSON"
        )
    envelope: dict[str, object] = {
        "format": EVIDENCE_OBSERVATION_FORMAT,
        "observation_id": observation_id,
        "kind": observation.kind,
        "scope": observation.scope,
        "authority": "observation",
        "bindings": {
            "comparison_id": comparison_id,
            "schedule_digest": normalize_digest(
                schedule_digest, label="observation schedule digest"
            ),
            "policy_digest": normalize_digest(
                policy_digest, label="observation policy digest"
            ),
            "artifact_digests": {
                side: normalize_digest(
                    artifact_digests[side],
                    label=f"observation {side} artifact digest",
                )
                for side in ("baseline", "subject")
            },
        },
        "payload": payload,
    }
    if schema_errors := _observation_schema_errors(envelope):
        raise EvidencePackError(schema_errors[0])
    return canonical_json_bytes(envelope)


def evidence_observation_errors(
    payload: Mapping[str, object],
    *,
    observation_id: str,
    reference: Mapping[str, object],
    comparison_id: str,
    schedule_digest: str,
    policy_digest: str,
    artifact_digests: Mapping[str, str],
) -> list[str]:
    """Validate one observation envelope and all signed-pack bindings."""

    errors = _observation_schema_errors(payload)
    observation_payload = payload.get("payload")
    if (
        isinstance(observation_payload, Mapping)
        and len(canonical_json_bytes(observation_payload)) > MAX_OBSERVATION_BYTES
    ):
        errors.append(
            f"observation {observation_id!r} payload exceeds the "
            f"{MAX_OBSERVATION_BYTES}-byte limit"
        )
    expected_reference = {
        "path": f"observations/{observation_id}.json",
        "digest": reference.get("digest"),
        "kind": payload.get("kind"),
        "scope": payload.get("scope"),
    }
    if dict(reference) != expected_reference:
        errors.append(f"observation {observation_id!r} manifest reference is invalid")
    if payload.get("observation_id") != observation_id:
        errors.append(f"observation {observation_id!r} identifier binding is invalid")
    if payload.get("kind") != reference.get("kind"):
        errors.append(f"observation {observation_id!r} kind binding is invalid")
    if payload.get("scope") != reference.get("scope"):
        errors.append(f"observation {observation_id!r} scope binding is invalid")
    bindings = payload.get("bindings")
    expected_bindings = {
        "comparison_id": comparison_id,
        "schedule_digest": schedule_digest,
        "policy_digest": policy_digest,
        "artifact_digests": dict(artifact_digests),
    }
    if bindings != expected_bindings:
        errors.append(
            f"observation {observation_id!r} does not bind the comparison, "
            "schedule, policy, and artifacts"
        )
    if payload.get("authority") != "observation":
        errors.append(f"observation {observation_id!r} authority is invalid")
    return list(dict.fromkeys(errors))


def runtime_side_config_errors(
    payload: bytes,
    *,
    role: str,
    provider_name: str,
    artifact_identity_sha256: str,
    schedule_sha256: str,
    policy_digest: str,
) -> list[str]:
    """Validate the semantic bindings in one authenticated runtime config."""

    try:
        parsed = parse_json_bytes(payload, label=f"{role} runtime config")
    except ValueError as exc:
        return [str(exc)]
    if not isinstance(parsed, dict):
        return [f"{role} runtime config must be a JSON object"]
    expected = {
        "format": RUNTIME_SIDE_CONFIG_FORMAT,
        "role": role,
        "provider": provider_name,
        "artifact_identity_sha256": artifact_identity_sha256,
        "schedule_sha256": schedule_sha256,
        "policy_digest": policy_digest,
    }
    if parsed != expected:
        return [
            f"{role} runtime config does not bind role, provider, artifact, "
            "schedule, and policy"
        ]
    if payload != canonical_json_bytes(parsed):
        return [f"{role} runtime config must use canonical JSON"]
    return []


def normalize_digest(value: str, *, label: str) -> str:
    normalized = value.strip().lower() if isinstance(value, str) else ""
    if _DIGEST_RE.fullmatch(normalized) is None:
        raise EvidencePackError(f"{label} must be a sha256:... digest")
    return normalized


def identity_payload(role: str, identity: InputIdentity) -> dict[str, object]:
    if role not in INPUT_ROLES:
        raise EvidencePackError(f"unsupported input role: {role!r}")
    payload: dict[str, object] = {
        "format": EVIDENCE_INPUT_IDENTITY_FORMAT,
        "role": role,
        "digest": normalize_digest(identity.digest, label=f"{role} digest"),
    }
    if identity.locator is not None:
        locator = identity.locator.strip()
        if not locator or len(locator.encode("utf-8")) > 4096:
            raise EvidencePackError(f"{role} locator is invalid")
        payload["locator"] = locator
    if identity.media_type is not None:
        media_type = identity.media_type.strip()
        if not media_type or len(media_type) > 255:
            raise EvidencePackError(f"{role} media_type is invalid")
        payload["media_type"] = media_type
    return payload


def evaluation_request_errors(payload: Mapping[str, object]) -> list[str]:
    errors = sorted(
        Draft202012Validator(load_evaluation_request_schema()).iter_errors(payload),
        key=lambda error: tuple(str(part) for part in error.absolute_path),
    )
    if not errors:
        return []
    first = errors[0]
    path = ".".join(str(part) for part in first.absolute_path) or "<root>"
    return [f"normalized request schema failed at {path}: {first.message}"]


def dataset_preparation_binding_errors(
    request: Mapping[str, object],
    schedule: RuntimeBehavioralSchedule,
) -> list[str]:
    """Validate the mode-specific dataset intent preserved in signed evidence."""

    comparison = request.get("comparison")
    execution = request.get("execution")
    if not isinstance(comparison, Mapping) or not isinstance(execution, Mapping):
        return ["normalized request dataset binding is invalid"]
    mode = execution.get("mode")
    dataset = comparison.get("dataset")
    if mode == "import":
        return (
            []
            if dataset == EVIDENCE_PATHS["schedule"]
            else ["import dataset must name the canonical schedule"]
        )
    if mode != "run" or not isinstance(dataset, Mapping):
        return ["run dataset must contain its path-free preparation descriptor"]
    try:
        validate_local_dataset_preparation(dataset, schedule)
    except ValueError as exc:
        return [str(exc)]
    return []


def request_metric(request: Mapping[str, object]) -> str:
    comparison = request.get("comparison")
    metric = comparison.get("metric") if isinstance(comparison, Mapping) else None
    if metric == "multiple_choice_accuracy":
        raise EvidencePackError(
            "multiple_choice_accuracy is not supported by "
            "invarlock/evidence-pack-v1; "
            "its option contract is not yet canonical"
        )
    if metric == "exact_match":
        return "exact_match"
    if metric == "normalized_nll_per_utf8_byte":
        return "normalized_nll_per_utf8_byte"
    if isinstance(comparison, Mapping) and "scorer_extension" in comparison:
        try:
            return decode_scorer_binding(comparison.get("scorer_extension")).scorer_id
        except ScorerExtensionError as exc:
            raise EvidencePackError(str(exc)) from exc
    raise EvidencePackError("request comparison.metric is unsupported")


def request_scorer_binding(
    request: Mapping[str, object],
) -> ScorerExtensionBinding | None:
    """Return the mutually exclusive extension binding, when selected."""

    comparison = request.get("comparison")
    if not isinstance(comparison, Mapping) or "scorer_extension" not in comparison:
        return None
    try:
        return decode_scorer_binding(comparison.get("scorer_extension"))
    except ScorerExtensionError as exc:
        raise EvidencePackError(str(exc)) from exc


def parse_json_object(payload: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = parse_json_bytes(payload, label=label)
    except ValueError as exc:
        raise EvidencePackError(str(exc)) from exc
    if not isinstance(value, dict):
        raise EvidencePackError(f"{label} must be a JSON object")
    return value


def _record_digest(record: RuntimeScoringRecord) -> str:
    return sha256_digest(canonical_json_bytes(asdict(record), newline=False))


def _validate_observation_records(
    records: Sequence[RuntimeScoringRecord], *, side: str
) -> None:
    for record in records:
        if record.status != "ok":
            raise EvidencePackError(
                f"{side} observation record {record.record_id!r} is not successful"
            )
        if record.output_text is not None:
            expected = hashlib.sha256(record.output_text.encode("utf-8")).hexdigest()
            if record.output_sha256 != expected:
                raise EvidencePackError(
                    f"{side} observation record {record.record_id!r} output digest "
                    "does not match output text"
                )
    if not records:
        raise EvidencePackError(f"{side} observation has no records")


def _score_record(
    record: RuntimeScoringRecord, *, expected_output: str, metric: str, side: str
) -> float:
    if metric == "exact_match":
        if record.output_text is None:
            raise EvidencePackError(
                f"{side} record {record.record_id!r} lacks output text"
            )
        return 1.0 if record.output_text == expected_output else 0.0
    if record.logprob_sum is None or (
        metric == "normalized_nll_per_utf8_byte"
        and (record.utf8_byte_count is None or record.utf8_byte_count <= 0)
    ):
        raise EvidencePackError(
            f"{side} record {record.record_id!r} lacks normalized NLL facts"
        )
    assert record.utf8_byte_count is not None
    expected_byte_count = len(expected_output.encode("utf-8"))
    if record.utf8_byte_count != expected_byte_count:
        raise EvidencePackError(
            f"{side} record {record.record_id!r} utf8_byte_count does not match "
            "expected_output"
        )
    score = -float(record.logprob_sum) / record.utf8_byte_count
    if not math.isfinite(score) or score < 0:
        raise EvidencePackError(
            f"{side} record {record.record_id!r} has invalid normalized NLL"
        )
    return score


def _derived_perplexity_measurement(
    *,
    baseline_records: Sequence[RuntimeScoringRecord],
    subject_records: Sequence[RuntimeScoringRecord],
    baseline_tokenizer_sha256: str,
    subject_tokenizer_sha256: str,
) -> dict[str, object]:
    """Derive a non-authoritative perplexity interpretation when comparable."""

    unavailable: dict[str, object] = {
        "status": "unavailable",
        "basis": "authenticated_target_likelihood",
        "method": DERIVED_PERPLEXITY_METHOD,
    }
    if baseline_tokenizer_sha256 != subject_tokenizer_sha256:
        return {**unavailable, "reason": "tokenizer_contracts_differ"}
    baseline_logprobs: list[float] = []
    subject_logprobs: list[float] = []
    total_tokens = 0
    for baseline, subject in zip(baseline_records, subject_records, strict=True):
        if (
            baseline.token_count is None
            or baseline.token_count <= 0
            or subject.token_count is None
            or subject.token_count <= 0
        ):
            return {**unavailable, "reason": "target_token_counts_unavailable"}
        if baseline.token_count != subject.token_count:
            return {**unavailable, "reason": "target_token_counts_differ"}
        assert baseline.logprob_sum is not None
        assert subject.logprob_sum is not None
        baseline_logprobs.append(float(baseline.logprob_sum))
        subject_logprobs.append(float(subject.logprob_sum))
        total_tokens += baseline.token_count
    baseline_nll = -math.fsum(baseline_logprobs) / total_tokens
    subject_nll = -math.fsum(subject_logprobs) / total_tokens
    try:
        baseline_perplexity = math.exp(baseline_nll)
        subject_perplexity = math.exp(subject_nll)
        ratio = math.exp(subject_nll - baseline_nll)
    except OverflowError:
        return {**unavailable, "reason": "derived_value_non_finite"}
    if not all(
        math.isfinite(value) and value > 0
        for value in (baseline_perplexity, subject_perplexity, ratio)
    ):
        return {**unavailable, "reason": "derived_value_non_finite"}
    return {
        "status": "available",
        "basis": "authenticated_target_likelihood",
        "method": DERIVED_PERPLEXITY_METHOD,
        "tokenizer_metadata_sha256": baseline_tokenizer_sha256,
        "target_token_count": total_tokens,
        "baseline_perplexity": baseline_perplexity,
        "subject_perplexity": subject_perplexity,
        "ratio": ratio,
    }


def derive_paired_records(
    *,
    schedule: RuntimeBehavioralSchedule,
    metric: str,
    baseline: RuntimeSideEvidence,
    subject: RuntimeSideEvidence,
    baseline_identity_digest: str,
    subject_identity_digest: str,
    baseline_runtime_digest: str,
    subject_runtime_digest: str,
    scorer_binding: ScorerExtensionBinding | None = None,
    scorer_registry: ScorerExtensionRegistry | None = None,
) -> dict[str, object]:
    """Derive canonical pairs solely from typed, cross-bound observations."""

    if len(schedule.records) > MAX_RECORDS:
        raise EvidencePackError(f"schedule exceeds the {MAX_RECORDS}-record limit")
    side_values = {"baseline": baseline, "subject": subject}
    expected_artifacts = {
        "baseline": normalize_digest(
            baseline_identity_digest, label="baseline artifact identity"
        ),
        "subject": normalize_digest(
            subject_identity_digest, label="subject artifact identity"
        ),
    }
    expected_runtimes = {
        "baseline": normalize_digest(
            baseline_runtime_digest, label="baseline runtime identity"
        ),
        "subject": normalize_digest(
            subject_runtime_digest, label="subject runtime identity"
        ),
    }
    observations: dict[str, Any] = {}
    artifact_identities: dict[str, Any] = {}
    for side, values in side_values.items():
        try:
            artifact = decode_artifact_identity(values.artifact_identity)
            observation = decode_scoring_observation(values.scoring_observation)
            receipt = decode_runtime_provider_receipt(values.provider_receipt)
            provider_errors = runtime_provider_evidence_errors(
                artifact_identity=artifact,
                scoring_observation=observation,
                receipt=receipt,
                scoring_observation_bytes=values.scoring_observation,
                expected_outer_image_digest=expected_runtimes[side],
            )
        except RuntimeProviderEvidenceError as exc:
            raise EvidencePackError(
                f"{side} runtime provider evidence is invalid: {exc}"
            ) from exc
        if provider_errors:
            raise EvidencePackError(
                f"{side} runtime provider evidence is invalid: "
                + "; ".join(provider_errors)
            )
        collection_metric = "exact_match" if scorer_binding is not None else metric
        if collection_metric not in receipt.capabilities.metrics:
            raise EvidencePackError(
                f"{side} runtime provider evidence does not declare metric "
                f"{collection_metric!r}"
            )
        if sha256_digest(values.artifact_identity) != expected_artifacts[side]:
            raise EvidencePackError(
                f"{side} input identity does not match provider artifact identity"
            )
        if observation.schedule_sha256 != schedule.schedule_sha256:
            raise EvidencePackError(
                f"{side} observation does not bind the canonical schedule"
            )
        _validate_observation_records(observation.records, side=side)
        aggregate = hashlib.sha256(
            canonical_json_bytes(
                [asdict(record) for record in observation.records], newline=False
            )
        ).hexdigest()
        if observation.aggregate_source_sha256 != aggregate:
            raise EvidencePackError(
                f"{side} observation aggregate_source_sha256 is invalid"
            )
        run_report = parse_json_object(
            values.run_report, label=f"{side} runtime side report"
        )
        expected_run_report = {
            "format": RUNTIME_SIDE_REPORT_FORMAT,
            "provider": observation.provider_name,
            "artifact_identity_sha256": observation.artifact_identity_sha256,
            "scoring_observation_sha256": hashlib.sha256(
                values.scoring_observation
            ).hexdigest(),
            "schedule_sha256": schedule.schedule_sha256,
            "record_count": len(observation.records),
        }
        if run_report != expected_run_report:
            raise EvidencePackError(
                f"{side} runtime side report does not bind its provider observation"
            )
        observations[side] = observation
        artifact_identities[side] = artifact

    schedule_ids = [record.record_id for record in schedule.records]
    for side, observation in observations.items():
        observed_ids = [record.record_id for record in observation.records]
        if observed_ids != schedule_ids:
            raise EvidencePackError(
                f"{side} observation record order does not match the schedule"
            )

    replay_results = None
    replay_values: dict[str, tuple[float, ...]] = {}
    if scorer_binding is not None:
        if metric != scorer_binding.scorer_id:
            raise EvidencePackError("selected scorer ID does not match paired metric")
        if scorer_registry is None:
            raise EvidencePackError(
                "an explicitly authorized scorer registry is required"
            )
        input_kinds = tuple(
            sorted(
                {
                    part.kind
                    for scheduled in schedule.records
                    for part in scheduled.input_parts
                }
                or {"text"}
            )
        )
        replay_payloads: dict[str, object] = {}
        for side in ("baseline", "subject"):
            records: list[AuthenticatedScorerRecord] = []
            for scheduled, observed in zip(
                schedule.records, observations[side].records, strict=True
            ):
                if scheduled.expected_output is None or observed.output_text is None:
                    raise EvidencePackError(
                        f"{side} record {scheduled.record_id!r} lacks text scorer facts"
                    )
                if observed.output_sha256 is None:
                    raise EvidencePackError(
                        f"{side} record {scheduled.record_id!r} lacks output digest"
                    )
                records.append(
                    AuthenticatedScorerRecord(
                        record_id=scheduled.record_id,
                        input_sha256=scheduled.input_sha256,
                        facts={
                            "expected_output": scheduled.expected_output,
                            "output_text": observed.output_text,
                            "output_sha256": observed.output_sha256,
                        },
                    )
                )
            try:
                replay = scorer_registry.replay(
                    ScorerReplayRequest(
                        binding=scorer_binding,
                        task=schedule.task,
                        input_kinds=input_kinds,
                        output_kind=SCORER_REPLAY_OUTPUT_KIND,
                        schedule_sha256=schedule.schedule_sha256,
                        records=tuple(records),
                    )
                )
            except ScorerExtensionError as exc:
                raise EvidencePackError(str(exc)) from exc
            replay_payloads[side] = scorer_result_payload(replay)
            replay_values[side] = tuple(item.value for item in replay.record_results)
        replay_results = replay_payloads

    paired: list[dict[str, object]] = []
    for index, scheduled in enumerate(schedule.records):
        baseline_record = observations["baseline"].records[index]
        subject_record = observations["subject"].records[index]
        for side, record in (
            ("baseline", baseline_record),
            ("subject", subject_record),
        ):
            if record.input_sha256 != scheduled.input_sha256:
                raise EvidencePackError(
                    f"{side} record {record.record_id!r} input digest does not "
                    "match the schedule"
                )
        expected_output = scheduled.expected_output
        if expected_output is None:
            raise EvidencePackError(
                f"schedule record {scheduled.record_id!r} lacks expected output"
            )
        paired_record: dict[str, object] = {
            "record_id": scheduled.record_id,
            "input_sha256": scheduled.input_sha256,
            "baseline": {
                "observation_record_digest": _record_digest(baseline_record),
                "score": (
                    replay_values["baseline"][index]
                    if scorer_binding is not None
                    else _score_record(
                        baseline_record,
                        expected_output=expected_output,
                        metric=metric,
                        side="baseline",
                    )
                ),
            },
            "subject": {
                "observation_record_digest": _record_digest(subject_record),
                "score": (
                    replay_values["subject"][index]
                    if scorer_binding is not None
                    else _score_record(
                        subject_record,
                        expected_output=expected_output,
                        metric=metric,
                        side="subject",
                    )
                ),
            },
        }
        paired.append(paired_record)
    result: dict[str, object] = {
        "format": PAIRED_RECORDS_FORMAT,
        "metric": metric,
        "schedule_sha256": schedule.schedule_sha256,
        "records": paired,
    }
    if metric == "normalized_nll_per_utf8_byte":
        result["derived_measurements"] = {
            "perplexity_ratio": _derived_perplexity_measurement(
                baseline_records=observations["baseline"].records,
                subject_records=observations["subject"].records,
                baseline_tokenizer_sha256=artifact_identities[
                    "baseline"
                ].tokenizer_metadata_sha256,
                subject_tokenizer_sha256=artifact_identities[
                    "subject"
                ].tokenizer_metadata_sha256,
            )
        }
    if scorer_binding is not None:
        assert replay_results is not None
        result["scorer_extension"] = scorer_binding_payload(scorer_binding)
        result["scorer_replay"] = replay_results
    return result


def _finite_number(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvidencePackError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise EvidencePackError(f"{label} must be a finite number")
    return result


def _comparison_value(
    metric: str,
    baseline_scores: Sequence[float],
    subject_scores: Sequence[float],
) -> float:
    if metric != "normalized_nll_per_utf8_byte":
        return (
            math.fsum(
                subject - baseline
                for baseline, subject in zip(
                    baseline_scores, subject_scores, strict=True
                )
            )
            / len(baseline_scores)
            * 100.0
        )
    baseline_mean = math.fsum(baseline_scores) / len(baseline_scores)
    if baseline_mean <= 0:
        raise EvidencePackError(
            "normalized NLL baseline mean must be greater than zero"
        )
    return (math.fsum(subject_scores) / len(subject_scores)) / baseline_mean


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _paired_resampling_interval(
    *,
    metric: str,
    baseline_scores: Sequence[float],
    subject_scores: Sequence[float],
    schedule_sha256: str,
) -> tuple[float, float]:
    """Return the fixed finite-schedule paired resampling interval.

    Index draws come only from SHA-256 over the authenticated schedule digest,
    replicate number, and draw number.  This makes verifier replay independent
    of a platform PRNG while keeping the resampling algorithm versioned in the
    canonical report.
    """

    if re.fullmatch(r"[a-f0-9]{64}", schedule_sha256) is None:
        raise EvidencePackError("paired records schedule_sha256 is invalid")
    count = len(baseline_scores)
    if count == 1:
        value = _comparison_value(metric, baseline_scores, subject_scores)
        return value, value
    seed = bytes.fromhex(schedule_sha256)
    distribution: list[float] = []
    for replicate in range(PAIRED_INTERVAL_REPLICATES):
        indexes = []
        for draw in range(count):
            digest = hashlib.sha256(
                seed + replicate.to_bytes(8, "big") + draw.to_bytes(8, "big")
            ).digest()
            indexes.append(int.from_bytes(digest[:8], "big") % count)
        distribution.append(
            _comparison_value(
                metric,
                [baseline_scores[index] for index in indexes],
                [subject_scores[index] for index in indexes],
            )
        )
    alpha = (1.0 - PAIRED_INTERVAL_CONFIDENCE) / 2.0
    return _percentile(distribution, alpha), _percentile(distribution, 1.0 - alpha)


def _resolved_metric_policy(
    policy: Mapping[str, object],
    *,
    metric: str,
    scorer_binding: ScorerExtensionBinding | None = None,
) -> Mapping[str, object]:
    if set(policy) != {"resolved_policy"}:
        raise EvidencePackError("policy fields must contain exactly resolved_policy")
    resolved = policy.get("resolved_policy")
    if not isinstance(resolved, Mapping) or set(resolved) != {"metrics"}:
        raise EvidencePackError(
            "policy resolved_policy fields must contain exactly metrics"
        )
    metrics = resolved.get("metrics") if isinstance(resolved, Mapping) else None
    if not isinstance(metrics, Mapping):
        raise EvidencePackError("policy resolved_policy.metrics must be an object")
    if scorer_binding is not None:
        metric_name = "scorer_extension"
        threshold_name = "delta_min_pp"
    elif metric == "exact_match":
        metric_name = "exact_match"
        threshold_name = "delta_min_pp"
    else:
        metric_name = "normalized_nll_per_utf8_byte"
        threshold_name = "ratio_max"
    if set(metrics) != {metric_name}:
        raise EvidencePackError(
            f"policy metrics must contain exactly metrics.{metric_name}"
        )
    selected = metrics.get(metric_name)
    if not isinstance(selected, Mapping):
        raise EvidencePackError(f"policy metrics.{metric_name} must be an object")
    expected_fields = {threshold_name}
    if scorer_binding is not None:
        expected_fields.update(
            {
                "scorer_id",
                "scorer_version",
                "descriptor_sha256",
                "configuration_sha256",
            }
        )
    interval_width_field = (
        "maximum_interval_width_ratio"
        if metric == "normalized_nll_per_utf8_byte" and scorer_binding is None
        else "maximum_interval_width_pp"
    )
    sample_fields = {"minimum_record_count", interval_width_field}
    side_accuracy_field = {"minimum_side_accuracy"}
    selected_fields = set(selected)
    if selected_fields & sample_fields and not sample_fields <= selected_fields:
        raise EvidencePackError(
            f"policy metrics.{metric_name}.minimum_record_count and "
            f"{interval_width_field} must be provided together"
        )
    optional_fields = set()
    if selected_fields & sample_fields:
        optional_fields.update(sample_fields)
    if "minimum_side_accuracy" in selected_fields:
        if metric != "exact_match" or scorer_binding is not None:
            raise EvidencePackError(
                f"policy metrics.{metric_name}.minimum_side_accuracy is only "
                "supported for exact_match"
            )
        optional_fields.update(side_accuracy_field)
    if selected_fields != expected_fields | optional_fields:
        raise EvidencePackError(
            f"policy metrics.{metric_name} must contain exactly the authorized fields"
        )
    if scorer_binding is not None:
        expected_pin = {
            "scorer_id": scorer_binding.scorer_id,
            "scorer_version": scorer_binding.scorer_version,
            "descriptor_sha256": scorer_binding.descriptor_sha256,
            "configuration_sha256": scorer_binding.configuration_sha256,
        }
        if any(selected.get(key) != value for key, value in expected_pin.items()):
            raise EvidencePackError(
                "policy scorer_extension pin does not match the request binding"
            )
    if sample_fields <= selected_fields:
        minimum_record_count = selected.get("minimum_record_count")
        if (
            isinstance(minimum_record_count, bool)
            or not isinstance(minimum_record_count, int)
            or not 1 <= minimum_record_count <= MAX_RECORDS
        ):
            raise EvidencePackError(
                f"policy metrics.{metric_name}.minimum_record_count must be an "
                f"integer between 1 and {MAX_RECORDS}"
            )
        maximum_interval_width = _finite_number(
            selected.get(interval_width_field),
            label=f"policy metrics.{metric_name}.{interval_width_field}",
        )
        maximum_width = (
            200.0 if interval_width_field == "maximum_interval_width_pp" else None
        )
        if maximum_interval_width <= 0.0 or (
            maximum_width is not None and maximum_interval_width > maximum_width
        ):
            suffix = " and at most 200" if maximum_width is not None else ""
            raise EvidencePackError(
                f"policy metrics.{metric_name}.{interval_width_field} must be "
                f"positive{suffix}"
            )
    if "minimum_side_accuracy" in selected:
        minimum_side_accuracy = _finite_number(
            selected["minimum_side_accuracy"],
            label=f"policy metrics.{metric_name}.minimum_side_accuracy",
        )
        if not 0.0 <= minimum_side_accuracy <= 1.0:
            raise EvidencePackError(
                f"policy metrics.{metric_name}.minimum_side_accuracy must be "
                "between 0 and 1"
            )
    return selected


def policy_sample_requirements(
    policy: Mapping[str, object],
    *,
    metric: str,
    scorer_binding: ScorerExtensionBinding | None = None,
) -> dict[str, int | float]:
    """Return validated sample and precision requirements for a metric policy."""

    selected = _resolved_metric_policy(
        policy,
        metric=metric,
        scorer_binding=scorer_binding,
    )
    if "minimum_record_count" not in selected:
        return {}
    width_field = (
        "maximum_interval_width_ratio"
        if metric == "normalized_nll_per_utf8_byte" and scorer_binding is None
        else "maximum_interval_width_pp"
    )
    return {
        "minimum_record_count": cast(int, selected["minimum_record_count"]),
        width_field: cast(float, selected[width_field]),
    }


def _sample_qualification(
    selected_policy: Mapping[str, object],
    *,
    metric: str,
    scorer_binding: ScorerExtensionBinding | None,
    record_count: int,
    interval_lower: float,
    interval_upper: float,
) -> dict[str, object] | None:
    """Resolve verifier-replayable sample sufficiency from an authorized policy."""

    if "minimum_record_count" not in selected_policy:
        return None
    width_field = (
        "maximum_interval_width_ratio"
        if metric == "normalized_nll_per_utf8_byte" and scorer_binding is None
        else "maximum_interval_width_pp"
    )
    unit = "ratio" if width_field.endswith("_ratio") else "percentage_points"
    minimum = cast(int, selected_policy["minimum_record_count"])
    maximum = cast(float, selected_policy[width_field])
    observed_width = interval_upper - interval_lower
    record_count_passed = record_count >= minimum
    interval_width_passed = observed_width <= maximum
    return {
        "record_count": {
            "minimum": minimum,
            "observed": record_count,
            "passed": record_count_passed,
        },
        "interval_width": {
            "maximum": maximum,
            "observed": observed_width,
            "unit": unit,
            "passed": interval_width_passed,
        },
        "passed": record_count_passed and interval_width_passed,
    }


def _side_accuracy_qualification(
    selected_policy: Mapping[str, object],
    *,
    baseline_mean: float,
    subject_mean: float,
) -> tuple[dict[str, object] | None, bool]:
    if "minimum_side_accuracy" not in selected_policy:
        return None, True
    minimum = cast(float, selected_policy["minimum_side_accuracy"])
    baseline_passed = baseline_mean >= minimum
    subject_passed = subject_mean >= minimum
    qualification = {
        "minimum": minimum,
        "baseline": {"observed": baseline_mean, "passed": baseline_passed},
        "subject": {"observed": subject_mean, "passed": subject_passed},
        "passed": baseline_passed and subject_passed,
    }
    return qualification, baseline_passed and subject_passed


def validated_derived_measurements(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != {"perplexity_ratio"}:
        raise EvidencePackError(
            "normalized NLL derived_measurements must contain exactly perplexity_ratio"
        )
    measurement = value.get("perplexity_ratio")
    if not isinstance(measurement, Mapping):
        raise EvidencePackError("derived perplexity_ratio must be an object")
    common = {
        "status",
        "basis",
        "method",
    }
    if measurement.get("basis") != "authenticated_target_likelihood":
        raise EvidencePackError("derived perplexity_ratio basis is invalid")
    if measurement.get("method") != DERIVED_PERPLEXITY_METHOD:
        raise EvidencePackError("derived perplexity_ratio method is invalid")
    if measurement.get("status") == "unavailable":
        if set(measurement) != common | {"reason"} or measurement.get("reason") not in {
            "tokenizer_contracts_differ",
            "target_token_counts_unavailable",
            "target_token_counts_differ",
            "derived_value_non_finite",
        }:
            raise EvidencePackError(
                "derived perplexity_ratio unavailability is invalid"
            )
        return {"perplexity_ratio": dict(measurement)}
    expected = common | {
        "tokenizer_metadata_sha256",
        "target_token_count",
        "baseline_perplexity",
        "subject_perplexity",
        "ratio",
    }
    if measurement.get("status") != "available" or set(measurement) != expected:
        raise EvidencePackError("derived perplexity_ratio fields are invalid")
    tokenizer = measurement.get("tokenizer_metadata_sha256")
    token_count = measurement.get("target_token_count")
    if (
        not isinstance(tokenizer, str)
        or re.fullmatch(r"[a-f0-9]{64}", tokenizer) is None
        or isinstance(token_count, bool)
        or not isinstance(token_count, int)
        or token_count <= 0
    ):
        raise EvidencePackError("derived perplexity_ratio bindings are invalid")
    baseline = _finite_number(
        measurement.get("baseline_perplexity"),
        label="derived baseline_perplexity",
    )
    subject = _finite_number(
        measurement.get("subject_perplexity"),
        label="derived subject_perplexity",
    )
    ratio = _finite_number(measurement.get("ratio"), label="derived perplexity_ratio")
    if min(baseline, subject, ratio) <= 0 or not math.isclose(
        ratio,
        subject / baseline,
        rel_tol=1e-12,
        abs_tol=0.0,
    ):
        raise EvidencePackError("derived perplexity_ratio values are invalid")
    return {"perplexity_ratio": dict(measurement)}


def build_comparison_report(
    *,
    comparison_id: str,
    paired_records: Mapping[str, object],
    policy: Mapping[str, object],
    policy_digest: str,
    report_format: str = COMPARISON_REPORT_FORMAT,
) -> dict[str, object]:
    """Replay the one closed comparison report from verifier-derived pairs."""

    if not _IDENTIFIER_RE.fullmatch(comparison_id):
        raise EvidencePackError("comparison_id is invalid")
    if report_format not in COMPARISON_REPORT_FORMATS:
        raise EvidencePackError("comparison report format is unsupported")
    metric = paired_records.get("metric")
    expected_fields = {"format", "metric", "schedule_sha256", "records"}
    if metric == "normalized_nll_per_utf8_byte":
        expected_fields.add("derived_measurements")
    scorer_binding: ScorerExtensionBinding | None = None
    scorer_replay: dict[str, object] | None = None
    if (
        isinstance(metric, str)
        and metric not in {"exact_match", "normalized_nll_per_utf8_byte"}
        and {"scorer_extension", "scorer_replay"}.issubset(paired_records)
    ):
        expected_fields.update({"scorer_extension", "scorer_replay"})
        try:
            scorer_binding = decode_scorer_binding(
                paired_records.get("scorer_extension")
            )
        except ScorerExtensionError as exc:
            raise EvidencePackError(str(exc)) from exc
        if metric != scorer_binding.scorer_id:
            raise EvidencePackError("paired records scorer ID is invalid")
        replay_value = paired_records.get("scorer_replay")
        if (
            not isinstance(replay_value, Mapping)
            or set(replay_value) != {"baseline", "subject"}
            or not all(
                isinstance(replay_value.get(side), Mapping) for side in replay_value
            )
        ):
            raise EvidencePackError("paired records scorer replay is invalid")
        scorer_replay = dict(replay_value)
    if set(paired_records) != expected_fields:
        raise EvidencePackError("paired records fields are invalid")
    if paired_records.get("format") != PAIRED_RECORDS_FORMAT:
        raise EvidencePackError("paired records format is invalid")
    if not isinstance(metric, str) or (
        metric not in {"exact_match", "normalized_nll_per_utf8_byte"}
        and scorer_binding is None
    ):
        raise EvidencePackError("paired records metric is unsupported")
    records = paired_records.get("records")
    if not isinstance(records, list) or not records:
        raise EvidencePackError("paired records must be a non-empty array")
    baseline_scores: list[float] = []
    subject_scores: list[float] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise EvidencePackError(f"paired record {index} must be an object")
        for side, target in (
            ("baseline", baseline_scores),
            ("subject", subject_scores),
        ):
            value = record.get(side)
            if not isinstance(value, Mapping):
                raise EvidencePackError(f"paired record {index} {side} is invalid")
            target.append(
                _finite_number(value.get("score"), label=f"record {index} {side}.score")
            )
        if "weight" in record:
            raise EvidencePackError(f"paired record {index} weight is invalid")
    baseline_mean = math.fsum(baseline_scores) / len(baseline_scores)
    subject_mean = math.fsum(subject_scores) / len(subject_scores)
    value = _comparison_value(metric, baseline_scores, subject_scores)
    schedule_sha256 = paired_records.get("schedule_sha256")
    if not isinstance(schedule_sha256, str):
        raise EvidencePackError("paired records schedule_sha256 is invalid")
    selected_policy = _resolved_metric_policy(
        policy, metric=metric, scorer_binding=scorer_binding
    )
    if (
        "minimum_side_accuracy" in selected_policy
        and report_format != COMPARISON_REPORT_FORMAT
    ):
        raise EvidencePackError(
            "minimum_side_accuracy requires invarlock/comparison-report-v3"
        )
    paired_binary: dict[str, object] | None = None
    if metric == "exact_match":
        try:
            paired_statistics = paired_exact_match_statistics(
                baseline_scores,
                subject_scores,
                confidence_interval_method=(
                    PAIRED_CONFIDENCE_INTERVAL_METHOD_V1
                    if report_format == LEGACY_COMPARISON_REPORT_FORMAT
                    else PAIRED_CONFIDENCE_INTERVAL_METHOD_V2
                ),
            )
        except PairedExactMatchError as exc:
            raise EvidencePackError(
                f"exact-match paired statistics are invalid: {exc}"
            ) from exc
        interval = paired_statistics.effect_size_confidence_interval
        interval_lower = interval.lower_pp
        interval_upper = interval.upper_pp
        uncertainty = {
            "method": interval.method,
            "scope": "paired_binary_outcomes",
            "interval_mass": interval.confidence_level,
            "lower": interval_lower,
            "upper": interval_upper,
        }
        paired_binary = {
            "baseline_pass_subject_fail": (
                paired_statistics.baseline_pass_subject_fail_count
            ),
            "baseline_fail_subject_pass": (
                paired_statistics.baseline_fail_subject_pass_count
            ),
            "both_pass": paired_statistics.both_pass_count,
            "both_fail": paired_statistics.both_fail_count,
            "discordant_pairs": (
                paired_statistics.baseline_pass_subject_fail_count
                + paired_statistics.baseline_fail_subject_pass_count
            ),
            "mcnemar_exact_two_sided_p_value": (
                paired_statistics.mcnemar_exact_two_sided_p_value
            ),
            "effect_size_pp": paired_statistics.effect_size_pp,
            "effect_size_confidence_interval": asdict(interval),
        }
        limit = _finite_number(
            selected_policy.get("delta_min_pp"),
            label="policy exact_match.delta_min_pp",
        )
        if not -100.0 <= limit <= 100.0:
            raise EvidencePackError(
                "policy exact_match.delta_min_pp must be between -100 and 100"
            )
        value = 0.0 if value == 0 else value
        comparison = {
            "kind": "exact_match_delta_pp",
            "value": value,
            "minimum": limit,
        }
        passed = interval_lower >= limit
    elif metric == "normalized_nll_per_utf8_byte":
        interval_lower, interval_upper = _paired_resampling_interval(
            metric=metric,
            baseline_scores=baseline_scores,
            subject_scores=subject_scores,
            schedule_sha256=schedule_sha256,
        )
        uncertainty = {
            "method": PAIRED_INTERVAL_METHOD,
            "scope": "authenticated_schedule",
            "interval_mass": PAIRED_INTERVAL_CONFIDENCE,
            "replicates": PAIRED_INTERVAL_REPLICATES,
            "lower": interval_lower,
            "upper": interval_upper,
        }
        limit = _finite_number(
            selected_policy.get("ratio_max"),
            label="policy normalized_nll_per_utf8_byte.ratio_max",
        )
        kind = "normalized_nll_ratio"
        if limit <= 0:
            raise EvidencePackError(f"policy {metric}.ratio_max must be positive")
        comparison = {
            "kind": kind,
            "value": value,
            "maximum": limit,
        }
        passed = interval_upper <= limit
    else:
        interval_lower, interval_upper = _paired_resampling_interval(
            metric=metric,
            baseline_scores=baseline_scores,
            subject_scores=subject_scores,
            schedule_sha256=schedule_sha256,
        )
        uncertainty = {
            "method": PAIRED_INTERVAL_METHOD,
            "scope": "authenticated_schedule",
            "interval_mass": PAIRED_INTERVAL_CONFIDENCE,
            "replicates": PAIRED_INTERVAL_REPLICATES,
            "lower": interval_lower,
            "upper": interval_upper,
        }
        limit = _finite_number(
            selected_policy.get("delta_min_pp"),
            label="policy scorer_extension.delta_min_pp",
        )
        if not -100.0 <= limit <= 100.0:
            raise EvidencePackError(
                "policy scorer_extension.delta_min_pp must be between -100 and 100"
            )
        comparison = {
            "kind": "scorer_extension_delta_pp",
            "value": 0.0 if value == 0 else value,
            "minimum": limit,
        }
        passed = interval_lower >= limit
    sample_qualification = _sample_qualification(
        selected_policy,
        metric=metric,
        scorer_binding=scorer_binding,
        record_count=len(records),
        interval_lower=interval_lower,
        interval_upper=interval_upper,
    )
    if sample_qualification is not None:
        passed = passed and cast(bool, sample_qualification["passed"])
    side_accuracy_qualification, side_accuracy_passed = _side_accuracy_qualification(
        selected_policy,
        baseline_mean=baseline_mean,
        subject_mean=subject_mean,
    )
    passed = passed and side_accuracy_passed
    report: dict[str, object] = {
        "format": report_format,
        "comparison_id": comparison_id,
        "metric": metric,
        "record_count": len(records),
        "baseline": {"mean_score": baseline_mean},
        "subject": {"mean_score": subject_mean},
        "comparison": comparison,
        "uncertainty": uncertainty,
        "policy_digest": normalize_digest(policy_digest, label="policy digest"),
        "verdict": "pass" if passed else "fail",
    }
    if metric == "normalized_nll_per_utf8_byte":
        report["derived_measurements"] = validated_derived_measurements(
            paired_records.get("derived_measurements")
        )
    if paired_binary is not None:
        report["paired_binary"] = paired_binary
    if sample_qualification is not None:
        report["sample_qualification"] = sample_qualification
    report.update(
        {"side_accuracy": side_accuracy_qualification}
        if side_accuracy_qualification is not None
        else {}
    )
    if scorer_binding is not None:
        report["scorer_extension"] = scorer_binding_payload(scorer_binding)
        assert scorer_replay is not None
        report["scorer_replay"] = scorer_replay
    return report


def schedule_bytes(schedule: RuntimeBehavioralSchedule) -> bytes:
    payload = canonical_runtime_behavioral_schedule_json(schedule)
    if hashlib.sha256(payload).hexdigest() != schedule.schedule_sha256:
        raise EvidencePackError("schedule digest does not match canonical bytes")
    return payload


__all__ = [
    "COMPARISON_REPORT_FORMAT",
    "COMPARISON_REPORT_FORMAT_V2",
    "COMPARISON_REPORT_FORMATS",
    "DERIVED_PERPLEXITY_METHOD",
    "EVIDENCE_PACK_FORMAT",
    "EVIDENCE_OBSERVATION_FORMAT",
    "EVIDENCE_PACK_VERIFY_FORMAT",
    "EVIDENCE_PATHS",
    "EvidencePackError",
    "EvidenceObservation",
    "INPUT_ROLES",
    "InputIdentity",
    "LEGACY_COMPARISON_REPORT_FORMAT",
    "MAX_EVIDENCE_BYTES",
    "MAX_IDENTITY_BYTES",
    "MAX_OBSERVATION_BYTES",
    "MAX_OBSERVATIONS",
    "PAIRED_RECORDS_FORMAT",
    "RUNTIME_SIDE_REPORT_FORMAT",
    "RuntimeSideEvidence",
    "build_comparison_report",
    "canonical_json_bytes",
    "dataset_preparation_binding_errors",
    "derive_paired_records",
    "evaluation_request_errors",
    "evidence_observation_bytes",
    "evidence_observation_errors",
    "validated_derived_measurements",
    "identity_payload",
    "normalize_digest",
    "parse_json_object",
    "policy_sample_requirements",
    "request_metric",
    "request_scorer_binding",
    "schedule_bytes",
    "sha256_digest",
]
