from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace
from typing import Any

import pytest

from invarlock.core.assurance_contract import ASSURANCE_CLAIM_SET
from invarlock.core.runtime_provider import (
    EvaluationBatch,
    GGUFArtifactIdentity,
    HFSnapshotArtifactIdentity,
    RuntimeProviderCapabilities,
    artifact_identity_sha256,
)
from invarlock.core.runtime_provider.behavioral_schedule import (
    RUNTIME_BEHAVIORAL_SCHEDULE_FORMAT,
    RuntimeBehavioralSchedule,
    build_runtime_behavioral_schedule,
)
from invarlock.policy_pack import build_behavioral_policy_pack
from invarlock.reporting.validation.runtime_behavioral_claim import (
    verify_runtime_behavioral_claim,
)
from invarlock.reporting.validation.runtime_behavioral_observation import (
    runtime_scoring_records_sha256,
)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _dataset_identity() -> dict[str, object]:
    return {
        "provider": "local_jsonl",
        "dataset_name": "runtime-regression-v1",
        "config_name": None,
        "revision": "d" * 40,
        "split": "validation",
    }


def _schedule(
    *, dataset_identity: dict[str, object] | None = None
) -> RuntimeBehavioralSchedule:
    records = [
        {
            "record_id": f"sample-{index}",
            "input_text": f"Prompt {index}",
            "input_sha256": _sha256(f"Prompt {index}"),
            "expected_output": answer,
        }
        for index, answer in enumerate(("alpha", "beta", "gamma", "delta"), start=1)
    ]
    return build_runtime_behavioral_schedule(
        {
            "format_version": RUNTIME_BEHAVIORAL_SCHEDULE_FORMAT,
            "dataset_identity": dataset_identity or _dataset_identity(),
            "records": records,
        }
    )


def _capabilities(
    name: str,
    *,
    artifact_format: str,
    execution_modes: tuple[str, ...],
    evidence_surfaces: tuple[str, ...],
) -> RuntimeProviderCapabilities:
    return RuntimeProviderCapabilities(
        provider_name=name,
        artifact_formats=(artifact_format,),  # type: ignore[arg-type]
        tasks=("text_causal",),
        metrics=("exact_match",),
        execution_modes=execution_modes,  # type: ignore[arg-type]
        required_extra=None,
        required_image=None,
        platform_constraints=(),
        evidence_surfaces=evidence_surfaces,  # type: ignore[arg-type]
        supported_claim_sets=("invarlock-runtime-behavioral-regression-v1",),
    )


def _baseline_capabilities() -> RuntimeProviderCapabilities:
    return _capabilities(
        "hf_transformers",
        artifact_format="hf_snapshot",
        execution_modes=("in_process",),
        evidence_surfaces=(
            "behavior",
            "tokenizer",
            "weights",
            "modules",
            "activations",
        ),
    )


def _subject_capabilities() -> RuntimeProviderCapabilities:
    return _capabilities(
        "llama_cpp",
        artifact_format="gguf",
        execution_modes=("local_process", "container"),
        evidence_surfaces=("behavior", "tokenizer", "build"),
    )


def _baseline_artifact() -> HFSnapshotArtifactIdentity:
    return HFSnapshotArtifactIdentity(
        model_id="org/model",
        immutable_revision="a" * 40,
        checkpoint_tree_sha256=None,
        tokenizer_metadata_sha256="b" * 64,
    )


def _subject_artifact() -> GGUFArtifactIdentity:
    return GGUFArtifactIdentity(
        artifact_name="model.gguf",
        sha256="c" * 64,
        byte_length=4096,
        gguf_metadata_sha256="d" * 64,
        tensor_inventory_sha256="e" * 64,
        tokenizer_metadata_sha256="b" * 64,
    )


def _observation(
    *,
    provider_name: str,
    artifact_sha256: str,
    outputs: tuple[str, ...],
    batch: EvaluationBatch | None = None,
) -> dict[str, object]:
    effective_batch = batch or _schedule().evaluation_batch()
    records: list[dict[str, object]] = []
    for expected, output in zip(effective_batch.records, outputs, strict=True):
        records.append(
            {
                "record_id": expected.record_id,
                "input_sha256": expected.input_sha256,
                "status": "ok",
                "output_text": output,
                "output_sha256": _sha256(output),
                "logprob_sum": None,
                "token_count": None,
                "utf8_byte_count": None,
                "error_code": None,
            }
        )
    return {
        "format_version": "invarlock/runtime-scoring-observation-v1",
        "provider_name": provider_name,
        "artifact_identity_sha256": artifact_sha256,
        "schedule_sha256": effective_batch.schedule_sha256,
        "records": records,
        "aggregate_source_sha256": runtime_scoring_records_sha256(records),
    }


def _policy(
    *,
    minimum_subject_score: float = 0.75,
    maximum_regression: float = 0.25,
    metric_kind: str = "exact_match",
    providers: list[str] | None = None,
    formats: list[str] | None = None,
) -> dict[str, Any]:
    return build_behavioral_policy_pack(
        tier="balanced",
        allowed_provider_names=providers or ["hf_transformers", "llama_cpp"],
        allowed_artifact_formats=formats or ["gguf", "hf_snapshot"],
        metric_kind=metric_kind,
        minimum_subject_score=minimum_subject_score,
        maximum_regression=maximum_regression,
        dataset_identity=_dataset_identity(),
    )


def _redigest(policy: dict[str, Any]) -> None:
    payload = {key: value for key, value in policy.items() if key != "policy_digest"}
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    policy["policy_digest"] = f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _verify(
    *,
    policy: dict[str, Any] | None = None,
    schedule: RuntimeBehavioralSchedule | None = None,
    baseline_capabilities: RuntimeProviderCapabilities | None = None,
    subject_capabilities: RuntimeProviderCapabilities | None = None,
    baseline_artifact: HFSnapshotArtifactIdentity | None = None,
    subject_artifact: GGUFArtifactIdentity | None = None,
    baseline_outputs: tuple[str, ...] = ("alpha", "beta", "gamma", "delta"),
    subject_outputs: tuple[str, ...] = ("alpha", "beta", "wrong", "delta"),
    subject_observation: dict[str, object] | None = None,
):
    effective_schedule = schedule or _schedule()
    effective_batch = effective_schedule.evaluation_batch()
    effective_baseline_artifact = baseline_artifact or _baseline_artifact()
    effective_subject_artifact = subject_artifact or _subject_artifact()
    return verify_runtime_behavioral_claim(
        baseline_capabilities=baseline_capabilities or _baseline_capabilities(),
        subject_capabilities=subject_capabilities or _subject_capabilities(),
        baseline_artifact_identity=effective_baseline_artifact,
        subject_artifact_identity=effective_subject_artifact,
        baseline_observation=_observation(
            provider_name="hf_transformers",
            artifact_sha256=artifact_identity_sha256(effective_baseline_artifact),
            outputs=baseline_outputs,
            batch=effective_batch,
        ),
        subject_observation=subject_observation
        or _observation(
            provider_name="llama_cpp",
            artifact_sha256=artifact_identity_sha256(effective_subject_artifact),
            outputs=subject_outputs,
            batch=effective_batch,
        ),
        schedule=effective_schedule,
        policy_pack=policy or _policy(),
    )


def test_paired_behavioral_claim_replays_both_sides_and_applies_policy() -> None:
    result = _verify()

    assert result.ok is True
    assert result.errors == ()
    assert result.claim_set == "invarlock-runtime-behavioral-regression-v1"
    assert result.metric == "exact_match"
    assert result.baseline_score == 1.0
    assert result.subject_score == 0.75
    assert result.regression == 0.25
    assert result.schedule_sha256 == _schedule().schedule_sha256


def test_paired_behavioral_claim_rejects_policy_digest_tampering() -> None:
    policy = _policy()
    policy["behavioral_claim"]["metric_policy"]["minimum_subject_score"] = 0.5

    result = _verify(policy=policy)

    assert result.ok is False
    assert any("policy digest mismatch" in error for error in result.errors)


def test_paired_behavioral_claim_rejects_unauthorized_provider_and_format() -> None:
    provider_result = _verify(policy=_policy(providers=["hf_transformers"]))
    format_result = _verify(policy=_policy(formats=["hf_snapshot"]))

    assert any(
        "subject provider 'llama_cpp'" in error for error in provider_result.errors
    )
    assert any(
        "subject artifact format 'gguf'" in error for error in format_result.errors
    )


def test_paired_behavioral_claim_rejects_missing_required_capability() -> None:
    subject = replace(
        _subject_capabilities(),
        evidence_surfaces=("behavior", "build"),
    )

    result = _verify(subject_capabilities=subject)

    assert result.ok is False
    assert any("tokenizer" in error for error in result.errors)


def test_paired_behavioral_claim_rejects_dataset_identity_drift() -> None:
    identity = _dataset_identity()
    identity["split"] = "test"

    result = _verify(schedule=_schedule(dataset_identity=identity))

    assert result.ok is False
    assert any("does not match policy-pack-v3" in error for error in result.errors)


def test_paired_behavioral_claim_rejects_artifact_observation_drift() -> None:
    observation = _observation(
        provider_name="llama_cpp",
        artifact_sha256="f" * 64,
        outputs=("alpha", "beta", "wrong", "delta"),
    )

    result = _verify(subject_observation=observation)

    assert result.ok is False
    assert any("artifact identity" in error for error in result.errors)


def test_paired_behavioral_claim_rejects_unbound_schedule_digest() -> None:
    unbound = replace(_schedule(), schedule_sha256="f" * 64)

    result = _verify(schedule=unbound)

    assert result.ok is False
    assert any(
        "does not match its authenticated dataset" in error for error in result.errors
    )


def test_paired_behavioral_claim_enforces_minimum_and_regression_thresholds() -> None:
    minimum_result = _verify(
        policy=_policy(minimum_subject_score=0.75, maximum_regression=1.0),
        subject_outputs=("alpha", "wrong", "wrong", "delta"),
    )
    regression_result = _verify(
        policy=_policy(minimum_subject_score=0.5, maximum_regression=0.25),
        subject_outputs=("alpha", "wrong", "wrong", "delta"),
    )

    assert any("below policy minimum" in error for error in minimum_result.errors)
    assert any("regression exceeds" in error for error in regression_result.errors)


def test_behavioral_policy_builder_rejects_multiple_choice_without_contract() -> None:
    with pytest.raises(ValueError, match="must be exact_match"):
        _policy(metric_kind="multiple_choice_accuracy")


def test_same_entrypoint_rejects_opaque_weight_edit_claim() -> None:
    dishonest_subject = replace(
        _subject_capabilities(),
        supported_claim_sets=(ASSURANCE_CLAIM_SET,),
        evidence_surfaces=(
            "behavior",
            "tokenizer",
            "weights",
            "modules",
            "activations",
        ),
        execution_modes=("in_process",),
    )
    policy = copy.deepcopy(_policy())
    policy["behavioral_claim"]["claim_set"] = ASSURANCE_CLAIM_SET
    _redigest(policy)

    result = _verify(policy=policy, subject_capabilities=dishonest_subject)

    assert result.ok is False
    assert any(
        "subject provider must be hf_transformers" in error for error in result.errors
    )
    assert any(
        "paired runtime verification requires" in error for error in result.errors
    )
