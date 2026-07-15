from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace
from typing import Any, cast

import pytest

from invarlock.core.assurance_contract import ASSURANCE_CLAIM_SET
from invarlock.core.runtime_provider import (
    EvaluationBatch,
    GGUFArtifactIdentity,
    HFSnapshotArtifactIdentity,
    RuntimeBackendIdentity,
    RuntimeDeviceFacts,
    RuntimeExecutionSettings,
    RuntimeProviderCapabilities,
    RuntimeProviderPluginIdentity,
    RuntimeProviderReceipt,
    artifact_identity_sha256,
)
from invarlock.core.runtime_provider.behavioral_schedule import (
    RUNTIME_BEHAVIORAL_SCHEDULE_FORMAT,
    RuntimeBehavioralSchedule,
    build_runtime_behavioral_schedule,
)
from invarlock.policy_pack import build_behavioral_policy_pack
from invarlock.reporting.validation.runtime_behavioral_claim import (
    _artifact_binding,
    _dataset_identity_errors,
    _observation_payload,
    _receipt_binding_errors,
    runtime_execution_settings_sha256,
    verify_runtime_behavioral_claim,
)
from invarlock.reporting.validation.runtime_behavioral_observation import (
    runtime_scoring_records_sha256,
)
from invarlock.runtime_behavioral_claim_receipt import (
    RuntimeBehavioralEvidenceBindings,
    build_runtime_behavioral_claim_receipt,
    verify_runtime_behavioral_claim_receipt,
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


def _settings(*, seed: int = 7) -> RuntimeExecutionSettings:
    return RuntimeExecutionSettings(
        seed=seed,
        context_length=512,
        batch_size=1,
        max_output_tokens=32,
        timeout_seconds=120,
        allow_network=False,
    )


def _observation_sha256(observation: dict[str, object]) -> str:
    encoded = json.dumps(
        observation,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _receipt(
    *,
    capabilities: RuntimeProviderCapabilities,
    artifact: HFSnapshotArtifactIdentity | GGUFArtifactIdentity,
    observation: dict[str, object],
    image_marker: str,
    settings: RuntimeExecutionSettings | None = None,
) -> RuntimeProviderReceipt:
    return RuntimeProviderReceipt(
        plugin=RuntimeProviderPluginIdentity(
            name=capabilities.provider_name,
            distribution="invarlock",
            distribution_version="0.13.0",
        ),
        backend=RuntimeBackendIdentity(
            name="runtime-backend",
            version="1",
            source_sha256="9" * 64,
            binary_sha256=None,
            build_sha256=None,
        ),
        capabilities=capabilities,
        artifact_identity=artifact,
        execution_settings=settings or _settings(),
        device=RuntimeDeviceFacts(device_kind="cpu", device_name="test-cpu"),
        outer_image_digest="sha256:" + image_marker * 64,
        scoring_observation_sha256=_observation_sha256(observation),
    )


def _binding(receipt: RuntimeProviderReceipt) -> dict[str, object]:
    return {
        "provider_name": receipt.capabilities.provider_name,
        "artifact_format": receipt.artifact_identity.artifact_format,
        "artifact_identity_sha256": artifact_identity_sha256(receipt.artifact_identity),
        "outer_image_digest": receipt.outer_image_digest,
        "execution_settings_sha256": runtime_execution_settings_sha256(
            receipt.execution_settings
        ),
    }


def _policy(
    *,
    minimum_subject_score: float = 0.75,
    maximum_regression: float = 0.25,
    metric_kind: str = "exact_match",
    baseline_binding: dict[str, object] | None = None,
    subject_binding: dict[str, object] | None = None,
    schedule_sha256: str | None = None,
) -> dict[str, Any]:
    schedule = _schedule()
    batch = schedule.evaluation_batch()
    baseline_artifact = _baseline_artifact()
    subject_artifact = _subject_artifact()
    baseline_observation = _observation(
        provider_name="hf_transformers",
        artifact_sha256=artifact_identity_sha256(baseline_artifact),
        outputs=("alpha", "beta", "gamma", "delta"),
        batch=batch,
    )
    subject_observation = _observation(
        provider_name="llama_cpp",
        artifact_sha256=artifact_identity_sha256(subject_artifact),
        outputs=("alpha", "beta", "wrong", "delta"),
        batch=batch,
    )
    baseline_receipt = _receipt(
        capabilities=_baseline_capabilities(),
        artifact=baseline_artifact,
        observation=baseline_observation,
        image_marker="1",
    )
    subject_receipt = _receipt(
        capabilities=_subject_capabilities(),
        artifact=subject_artifact,
        observation=subject_observation,
        image_marker="2",
    )
    return build_behavioral_policy_pack(
        tier="balanced",
        schedule_sha256=schedule_sha256 or schedule.schedule_sha256,
        baseline=baseline_binding or _binding(baseline_receipt),
        subject=subject_binding or _binding(subject_receipt),
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
    baseline_receipt: RuntimeProviderReceipt | None = None,
    subject_receipt: RuntimeProviderReceipt | None = None,
    subject_observation: dict[str, object] | None = None,
):
    effective_schedule = schedule or _schedule()
    effective_batch = effective_schedule.evaluation_batch()
    effective_baseline_artifact = baseline_artifact or _baseline_artifact()
    effective_subject_artifact = subject_artifact or _subject_artifact()
    effective_baseline_capabilities = baseline_capabilities or _baseline_capabilities()
    effective_subject_capabilities = subject_capabilities or _subject_capabilities()
    baseline_observation = _observation(
        provider_name="hf_transformers",
        artifact_sha256=artifact_identity_sha256(effective_baseline_artifact),
        outputs=baseline_outputs,
        batch=effective_batch,
    )
    effective_subject_observation = subject_observation or _observation(
        provider_name="llama_cpp",
        artifact_sha256=artifact_identity_sha256(effective_subject_artifact),
        outputs=subject_outputs,
        batch=effective_batch,
    )
    effective_baseline_receipt = baseline_receipt or _receipt(
        capabilities=effective_baseline_capabilities,
        artifact=effective_baseline_artifact,
        observation=baseline_observation,
        image_marker="1",
    )
    effective_subject_receipt = subject_receipt or _receipt(
        capabilities=effective_subject_capabilities,
        artifact=effective_subject_artifact,
        observation=effective_subject_observation,
        image_marker="2",
    )
    return verify_runtime_behavioral_claim(
        baseline_capabilities=effective_baseline_capabilities,
        subject_capabilities=effective_subject_capabilities,
        baseline_artifact_identity=effective_baseline_artifact,
        subject_artifact_identity=effective_subject_artifact,
        baseline_receipt=effective_baseline_receipt,
        subject_receipt=effective_subject_receipt,
        baseline_observation=baseline_observation,
        subject_observation=effective_subject_observation,
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


def test_strict_paired_replay_builds_a_portable_digest_only_receipt() -> None:
    result = _verify()
    baseline = RuntimeBehavioralEvidenceBindings(
        runtime_manifest_sha256=_sha256("baseline-manifest"),
        evaluation_report_sha256=_sha256("baseline-report"),
        provider_receipt_sidecar_sha256=_sha256("baseline-receipt"),
        scoring_observation_sidecar_sha256=_sha256("baseline-observation"),
        artifact_identity_sidecar_sha256=_sha256("baseline-artifact"),
    )
    subject = RuntimeBehavioralEvidenceBindings(
        runtime_manifest_sha256=_sha256("subject-manifest"),
        evaluation_report_sha256=_sha256("subject-report"),
        provider_receipt_sidecar_sha256=_sha256("subject-receipt"),
        scoring_observation_sidecar_sha256=_sha256("subject-observation"),
        artifact_identity_sidecar_sha256=_sha256("subject-artifact"),
    )

    receipt = build_runtime_behavioral_claim_receipt(
        baseline=baseline,
        subject=subject,
        verification=result,
    )

    assert receipt.baseline_score == 1.0
    assert receipt.subject_score == 0.75
    assert receipt.regression == 0.25
    assert receipt.verdict == "pass"
    assert (
        verify_runtime_behavioral_claim_receipt(
            receipt.to_payload(),
            expected_baseline=baseline,
            expected_subject=subject,
            expected_verification=result,
        )
        == receipt
    )


def test_paired_behavioral_claim_rejects_policy_digest_tampering() -> None:
    policy = _policy()
    policy["behavioral_claim"]["metric_policy"]["minimum_subject_score"] = 0.5

    result = _verify(policy=policy)

    assert result.ok is False
    assert any("policy digest mismatch" in error for error in result.errors)


def test_paired_behavioral_claim_rejects_direction_swap() -> None:
    policy = _policy()
    claim = policy["behavioral_claim"]
    claim["baseline"], claim["subject"] = claim["subject"], claim["baseline"]
    _redigest(policy)

    result = _verify(policy=policy)

    assert result.ok is False
    assert any(
        "baseline provider_name does not match the directed policy binding" in error
        for error in result.errors
    )
    assert any(
        "subject provider_name does not match the directed policy binding" in error
        for error in result.errors
    )


@pytest.mark.parametrize("field", ["artifact_identity_sha256", "outer_image_digest"])
def test_paired_behavioral_claim_rejects_unrelated_artifact_or_image(
    field: str,
) -> None:
    policy = _policy()
    replacement = "sha256:" + "f" * 64 if field == "outer_image_digest" else "f" * 64
    policy["behavioral_claim"]["subject"][field] = replacement
    _redigest(policy)

    result = _verify(policy=policy)

    assert result.ok is False
    assert any(
        f"subject {field} does not match the directed policy binding" in error
        for error in result.errors
    )


def test_paired_behavioral_claim_rejects_policy_schedule_drift() -> None:
    result = _verify(policy=_policy(schedule_sha256="f" * 64))

    assert result.ok is False
    assert any(
        "schedule does not match the directed policy binding" in error
        for error in result.errors
    )


def test_paired_behavioral_claim_rejects_settings_mismatch() -> None:
    schedule = _schedule()
    artifact = _subject_artifact()
    observation = _observation(
        provider_name="llama_cpp",
        artifact_sha256=artifact_identity_sha256(artifact),
        outputs=("alpha", "beta", "wrong", "delta"),
        batch=schedule.evaluation_batch(),
    )
    receipt = _receipt(
        capabilities=_subject_capabilities(),
        artifact=artifact,
        observation=observation,
        image_marker="2",
        settings=_settings(seed=8),
    )
    policy = _policy(subject_binding=_binding(receipt))

    result = _verify(policy=policy, subject_receipt=receipt)

    assert result.ok is False
    assert any("settings must be equal" in error for error in result.errors)


def test_paired_behavioral_claim_rejects_receipt_observation_mismatch() -> None:
    schedule = _schedule()
    artifact = _subject_artifact()
    observation = _observation(
        provider_name="llama_cpp",
        artifact_sha256=artifact_identity_sha256(artifact),
        outputs=("alpha", "beta", "wrong", "delta"),
        batch=schedule.evaluation_batch(),
    )
    receipt = replace(
        _receipt(
            capabilities=_subject_capabilities(),
            artifact=artifact,
            observation=observation,
            image_marker="2",
        ),
        scoring_observation_sha256="f" * 64,
    )

    result = _verify(subject_receipt=receipt)

    assert result.ok is False
    assert any("receipt scoring observation digest" in error for error in result.errors)


@pytest.mark.parametrize("field", ["capabilities", "artifact_identity"])
def test_paired_behavioral_claim_rejects_receipt_input_mismatch(field: str) -> None:
    schedule = _schedule()
    artifact = _subject_artifact()
    observation = _observation(
        provider_name="llama_cpp",
        artifact_sha256=artifact_identity_sha256(artifact),
        outputs=("alpha", "beta", "wrong", "delta"),
        batch=schedule.evaluation_batch(),
    )
    receipt = _receipt(
        capabilities=_subject_capabilities(),
        artifact=artifact,
        observation=observation,
        image_marker="2",
    )
    if field == "capabilities":
        receipt = replace(
            receipt,
            capabilities=replace(
                _subject_capabilities(), execution_modes=("container",)
            ),
        )
    else:
        receipt = replace(
            receipt,
            artifact_identity=replace(artifact, byte_length=artifact.byte_length + 1),
        )

    result = _verify(subject_receipt=receipt)

    assert result.ok is False
    assert any(f"receipt {field.replace('_', ' ')}" in error for error in result.errors)


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


def test_dataset_identity_validation_rejects_missing_and_extra_fields() -> None:
    identity = _dataset_identity()

    assert _dataset_identity_errors(identity, expected=None) == [
        "policy-pack-v3 is missing compatibility.dataset_identity"
    ]
    assert _dataset_identity_errors(
        {key: value for key, value in identity.items() if key != "split"},
        expected=identity,
    ) == [
        "authenticated dataset identity must contain exactly "
        "config_name, dataset_name, provider, revision, split"
    ]
    assert _dataset_identity_errors(
        identity,
        expected={**identity, "unexpected": True},
    ) == [
        "policy dataset identity must contain exactly "
        "config_name, dataset_name, provider, revision, split"
    ]


def test_claim_verifier_rejects_malformed_policy_contract_fields() -> None:
    policy = copy.deepcopy(_policy())
    policy["format"] = "invarlock/policy-pack-v2"
    claim = policy["behavioral_claim"]
    assert isinstance(claim, dict)
    claim["claim_set"] = "unknown-runtime-claim"
    claim["required_capabilities"] = {
        "tasks": None,
        "metrics": None,
        "evidence_surfaces": None,
    }
    claim["metric_policy"] = {
        "kind": "provider_accuracy",
        "minimum_subject_score": True,
        "maximum_regression": "0.25",
    }
    claim.pop("baseline")
    compatibility = policy["compatibility"]
    assert isinstance(compatibility, dict)
    compatibility["dataset_identity"] = None
    _redigest(policy)

    result = _verify(policy=policy)

    assert result.ok is False
    assert any(
        "runtime behavioral claims require policy-pack-v3" in error
        for error in result.errors
    )
    assert any(
        "unsupported runtime claim set" in error.lower() for error in result.errors
    )
    assert any("compatibility.dataset_identity" in error for error in result.errors)
    assert any(
        "currently supports only exact_match" in error for error in result.errors
    )
    assert any("missing behavioral_claim.baseline" in error for error in result.errors)


def test_artifact_binding_rejects_undeclared_format_and_capabilities() -> None:
    _, errors = _artifact_binding(
        "subject",
        _subject_artifact(),
        capabilities=replace(
            _subject_capabilities(),
            artifact_formats=("hf_snapshot",),
        ),
        required_tasks=frozenset({"image_text"}),
        required_metrics=frozenset({"normalized_nll_per_utf8_byte"}),
        required_surfaces=frozenset({"activations"}),
    )

    assert any("does not declare artifact format" in error for error in errors)
    assert any("lacks required tasks" in error for error in errors)
    assert any("lacks required metrics" in error for error in errors)
    assert any("lacks required evidence surfaces" in error for error in errors)


def test_observation_and_receipt_binding_fail_closed_on_non_json_values() -> None:
    artifact = _subject_artifact()
    observation = _observation(
        provider_name="llama_cpp",
        artifact_sha256=artifact_identity_sha256(artifact),
        outputs=("alpha", "beta", "wrong", "delta"),
    )
    receipt = _receipt(
        capabilities=_subject_capabilities(),
        artifact=artifact,
        observation=observation,
        image_marker="2",
    )

    assert _observation_payload(cast(Any, object())) is None
    non_json_errors = _receipt_binding_errors(
        "subject",
        receipt=receipt,
        capabilities=_subject_capabilities(),
        artifact_identity=artifact,
        observation=cast(Any, object()),
        expected_binding=None,
    )
    noncanonical_errors = _receipt_binding_errors(
        "subject",
        receipt=receipt,
        capabilities=_subject_capabilities(),
        artifact_identity=artifact,
        observation={**observation, "unexpected": object()},
        expected_binding=_binding(receipt),
    )

    assert (
        "subject receipt cannot bind a non-JSON scoring observation" in non_json_errors
    )
    assert "policy-pack-v3 is missing behavioral_claim.subject" in non_json_errors
    assert (
        "subject receipt cannot bind a non-canonical observation" in noncanonical_errors
    )


def test_claim_verifier_rejects_wrong_settings_and_receipt_types() -> None:
    with pytest.raises(TypeError, match="RuntimeExecutionSettings"):
        runtime_execution_settings_sha256(cast(RuntimeExecutionSettings, object()))
    with pytest.raises(TypeError, match="baseline_receipt"):
        _verify(baseline_receipt=cast(RuntimeProviderReceipt, object()))
    with pytest.raises(TypeError, match="subject_receipt"):
        _verify(subject_receipt=cast(RuntimeProviderReceipt, object()))
