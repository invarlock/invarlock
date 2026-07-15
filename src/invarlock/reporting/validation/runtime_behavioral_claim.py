"""Paired verification for the runtime behavioral regression claim.

The provider-facing observation verifier authenticates one set of measured facts.
This module owns the claim-level decision: it binds both sides to one verifier-owned
schedule and dataset identity, authorizes their capabilities and artifacts through a
policy-pack-v3, independently replays both observations, and applies the policy
thresholds.  Provider-supplied aggregate values are never consumed.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass

from invarlock.core.runtime_provider import (
    RUNTIME_BEHAVIORAL_CLAIM_SET,
    EvaluationBatch,
    ModelArtifactIdentity,
    RuntimeProviderCapabilities,
    ScoringObservation,
    artifact_identity_sha256,
    evaluate_runtime_claim_compatibility,
)
from invarlock.core.runtime_provider.behavioral_schedule import (
    RuntimeBehavioralSchedule,
    build_runtime_behavioral_schedule,
)
from invarlock.policy_pack import (
    BEHAVIORAL_POLICY_PACK_FORMAT,
    verify_policy_pack,
)

from .runtime_behavioral_observation import (
    RuntimeBehavioralMetricResult,
    RuntimeBehavioralObservationError,
    verify_runtime_behavioral_observation,
)

PAIRED_BEHAVIORAL_METRICS = frozenset({"exact_match"})


@dataclass(frozen=True)
class RuntimeBehavioralClaimVerificationResult:
    """Explicit outcome of one paired behavioral policy decision."""

    ok: bool
    errors: tuple[str, ...]
    claim_set: str | None
    metric: str | None
    baseline_score: float | None
    subject_score: float | None
    regression: float | None
    schedule_sha256: str
    policy_digest: str | None


def _deduplicated(errors: list[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(errors))


def _dataset_identity_errors(
    observed: Mapping[str, object],
    *,
    expected: object,
) -> list[str]:
    if not isinstance(expected, Mapping):
        return ["policy-pack-v3 is missing compatibility.dataset_identity"]
    expected_payload = dict(expected)
    observed_payload = dict(observed)
    required_fields = {
        "provider",
        "dataset_name",
        "config_name",
        "revision",
        "split",
    }
    errors: list[str] = []
    if set(observed_payload) != required_fields:
        errors.append(
            "authenticated dataset identity must contain exactly "
            + ", ".join(sorted(required_fields))
        )
        return errors
    if set(expected_payload) != required_fields:
        errors.append(
            "policy dataset identity must contain exactly "
            + ", ".join(sorted(required_fields))
        )
        return errors
    if observed_payload != expected_payload:
        errors.append("authenticated dataset identity does not match policy-pack-v3")
    return errors


def _artifact_binding(
    role: str,
    identity: ModelArtifactIdentity,
    *,
    capabilities: RuntimeProviderCapabilities,
    allowed_provider_names: frozenset[str],
    allowed_artifact_formats: frozenset[str],
    required_tasks: frozenset[str],
    required_metrics: frozenset[str],
    required_surfaces: frozenset[str],
) -> tuple[str | None, list[str]]:
    errors: list[str] = []
    provider_name = capabilities.provider_name
    artifact_format = identity.artifact_format
    if provider_name not in allowed_provider_names:
        errors.append(
            f"{role} provider {provider_name!r} is not authorized by policy-pack-v3"
        )
    if artifact_format not in allowed_artifact_formats:
        errors.append(
            f"{role} artifact format {artifact_format!r} is not authorized by "
            "policy-pack-v3"
        )
    if artifact_format not in capabilities.artifact_formats:
        errors.append(
            f"{role} provider {provider_name!r} does not declare artifact format "
            f"{artifact_format!r}"
        )

    for label, required, available in (
        ("tasks", required_tasks, frozenset(capabilities.tasks)),
        ("metrics", required_metrics, frozenset(capabilities.metrics)),
        (
            "evidence surfaces",
            required_surfaces,
            frozenset(capabilities.evidence_surfaces),
        ),
    ):
        missing = sorted(required.difference(available))
        if missing:
            errors.append(
                f"{role} provider {provider_name!r} lacks required {label}: "
                + ", ".join(missing)
            )
    return artifact_identity_sha256(identity), errors


def _observation_payload(
    observation: Mapping[str, object] | ScoringObservation,
) -> Mapping[str, object] | None:
    if isinstance(observation, Mapping):
        return observation
    # JSON round-tripping converts dataclass tuples to the arrays required by the
    # public schema and rejects non-finite values before verification.
    try:
        payload = json.loads(
            json.dumps(asdict(observation), allow_nan=False, ensure_ascii=False)
        )
    except (TypeError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _verify_one_observation(
    role: str,
    observation: Mapping[str, object] | ScoringObservation,
    *,
    capabilities: RuntimeProviderCapabilities,
    artifact_sha256: str | None,
    evaluation_batch: EvaluationBatch,
    metric: str | None,
) -> tuple[RuntimeBehavioralMetricResult | None, list[str]]:
    if artifact_sha256 is None or metric not in PAIRED_BEHAVIORAL_METRICS:
        return None, []
    payload = _observation_payload(observation)
    if payload is None:
        return None, [f"{role} scoring observation must be a JSON object"]
    try:
        result = verify_runtime_behavioral_observation(
            payload,
            expected_provider_name=capabilities.provider_name,
            expected_artifact_identity_sha256=artifact_sha256,
            expected_batch=evaluation_batch,
            metric=metric,
        )
    except RuntimeBehavioralObservationError as exc:
        return None, [f"{role} scoring observation failed verification: {exc}"]
    return result, []


def _string_set(value: object) -> frozenset[str]:
    if not isinstance(value, list):
        return frozenset()
    return frozenset(item for item in value if isinstance(item, str))


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def verify_runtime_behavioral_claim(
    *,
    baseline_capabilities: RuntimeProviderCapabilities,
    subject_capabilities: RuntimeProviderCapabilities,
    baseline_artifact_identity: ModelArtifactIdentity,
    subject_artifact_identity: ModelArtifactIdentity,
    baseline_observation: Mapping[str, object] | ScoringObservation,
    subject_observation: Mapping[str, object] | ScoringObservation,
    schedule: RuntimeBehavioralSchedule,
    policy_pack: Mapping[str, object],
) -> RuntimeBehavioralClaimVerificationResult:
    """Verify and decide one paired runtime behavioral claim, fail closed."""

    errors: list[str] = []
    if not isinstance(schedule, RuntimeBehavioralSchedule):
        raise TypeError("schedule must be a RuntimeBehavioralSchedule")
    authenticated_schedule = build_runtime_behavioral_schedule(schedule.to_payload())
    if schedule.schedule_sha256 != authenticated_schedule.schedule_sha256:
        errors.append(
            "runtime behavioral schedule digest does not match its authenticated "
            "dataset and ordered record material"
        )
    evaluation_batch = authenticated_schedule.evaluation_batch()

    policy = dict(policy_pack)
    errors.extend(f"policy-pack-v3: {error}" for error in verify_policy_pack(policy))
    if policy.get("format") != BEHAVIORAL_POLICY_PACK_FORMAT:
        errors.append(
            f"runtime behavioral claims require {BEHAVIORAL_POLICY_PACK_FORMAT}"
        )

    behavioral_claim = policy.get("behavioral_claim")
    claim = behavioral_claim if isinstance(behavioral_claim, Mapping) else {}
    claim_set_value = claim.get("claim_set")
    claim_set = claim_set_value if isinstance(claim_set_value, str) else None
    if claim_set is not None:
        try:
            compatibility = evaluate_runtime_claim_compatibility(
                claim_set,
                baseline=baseline_capabilities,
                subject=subject_capabilities,
            )
        except ValueError as exc:
            errors.append(f"runtime claim compatibility: {exc}")
        else:
            errors.extend(
                f"runtime claim compatibility: {error}"
                for error in compatibility.errors
            )
    if claim_set != RUNTIME_BEHAVIORAL_CLAIM_SET:
        errors.append(
            f"paired runtime verification requires {RUNTIME_BEHAVIORAL_CLAIM_SET}"
        )

    compatibility_block = policy.get("compatibility")
    compatibility_payload = (
        compatibility_block if isinstance(compatibility_block, Mapping) else {}
    )
    errors.extend(
        _dataset_identity_errors(
            authenticated_schedule.dataset_identity.to_payload(),
            expected=compatibility_payload.get("dataset_identity"),
        )
    )

    allowed_provider_names = _string_set(claim.get("allowed_provider_names"))
    allowed_artifact_formats = _string_set(claim.get("allowed_artifact_formats"))
    required_capabilities = claim.get("required_capabilities")
    required_payload = (
        required_capabilities if isinstance(required_capabilities, Mapping) else {}
    )
    required_tasks = _string_set(required_payload.get("tasks"))
    required_metrics = _string_set(required_payload.get("metrics"))
    required_surfaces = _string_set(required_payload.get("evidence_surfaces"))

    metric_policy = claim.get("metric_policy")
    metric_payload = metric_policy if isinstance(metric_policy, Mapping) else {}
    metric_value = metric_payload.get("kind")
    metric = metric_value if isinstance(metric_value, str) else None
    if metric not in PAIRED_BEHAVIORAL_METRICS:
        errors.append(
            "paired runtime behavioral verification currently supports only exact_match"
        )

    baseline_artifact_sha256, baseline_artifact_errors = _artifact_binding(
        "baseline",
        baseline_artifact_identity,
        capabilities=baseline_capabilities,
        allowed_provider_names=allowed_provider_names,
        allowed_artifact_formats=allowed_artifact_formats,
        required_tasks=required_tasks,
        required_metrics=required_metrics,
        required_surfaces=required_surfaces,
    )
    subject_artifact_sha256, subject_artifact_errors = _artifact_binding(
        "subject",
        subject_artifact_identity,
        capabilities=subject_capabilities,
        allowed_provider_names=allowed_provider_names,
        allowed_artifact_formats=allowed_artifact_formats,
        required_tasks=required_tasks,
        required_metrics=required_metrics,
        required_surfaces=required_surfaces,
    )
    errors.extend(baseline_artifact_errors)
    errors.extend(subject_artifact_errors)

    baseline_result, baseline_errors = _verify_one_observation(
        "baseline",
        baseline_observation,
        capabilities=baseline_capabilities,
        artifact_sha256=baseline_artifact_sha256,
        evaluation_batch=evaluation_batch,
        metric=metric,
    )
    subject_result, subject_errors = _verify_one_observation(
        "subject",
        subject_observation,
        capabilities=subject_capabilities,
        artifact_sha256=subject_artifact_sha256,
        evaluation_batch=evaluation_batch,
        metric=metric,
    )
    errors.extend(baseline_errors)
    errors.extend(subject_errors)

    baseline_score = baseline_result.value if baseline_result is not None else None
    subject_score = subject_result.value if subject_result is not None else None
    regression = (
        baseline_score - subject_score
        if baseline_score is not None and subject_score is not None
        else None
    )
    minimum_subject_score = _number(metric_payload.get("minimum_subject_score"))
    maximum_regression = _number(metric_payload.get("maximum_regression"))
    if (
        subject_score is not None
        and minimum_subject_score is not None
        and subject_score < minimum_subject_score
    ):
        errors.append(
            "subject score is below policy minimum: "
            f"observed={subject_score:.12g} minimum={minimum_subject_score:.12g}"
        )
    if (
        regression is not None
        and maximum_regression is not None
        and regression > maximum_regression
    ):
        errors.append(
            "baseline-to-subject regression exceeds policy maximum: "
            f"observed={regression:.12g} maximum={maximum_regression:.12g}"
        )

    final_errors = _deduplicated(errors)
    policy_digest_value = policy.get("policy_digest")
    return RuntimeBehavioralClaimVerificationResult(
        ok=not final_errors,
        errors=final_errors,
        claim_set=claim_set,
        metric=metric,
        baseline_score=baseline_score,
        subject_score=subject_score,
        regression=regression,
        schedule_sha256=authenticated_schedule.schedule_sha256,
        policy_digest=(
            policy_digest_value if isinstance(policy_digest_value, str) else None
        ),
    )


__all__ = [
    "PAIRED_BEHAVIORAL_METRICS",
    "RuntimeBehavioralClaimVerificationResult",
    "verify_runtime_behavioral_claim",
]
