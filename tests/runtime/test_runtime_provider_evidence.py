from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from invarlock.core.runtime_provider import (
    GGUFArtifactIdentity,
    HFSnapshotArtifactIdentity,
    RuntimeBackendIdentity,
    RuntimeDeviceFacts,
    RuntimeExecutionSettings,
    RuntimeProviderCapabilities,
    RuntimeProviderPluginIdentity,
    RuntimeProviderReceipt,
    RuntimeScoringRecord,
    ScoringObservation,
    TensorRTLLMArtifactIdentity,
    artifact_identity_sha256,
    canonical_artifact_identity_json,
)
from invarlock.runtime_provider_evidence import (
    ARTIFACT_IDENTITY_FILENAME,
    MAX_RUNTIME_PROVIDER_SIDECAR_BYTES,
    PROVIDER_RECEIPT_FILENAME,
    SCORING_OBSERVATION_FILENAME,
    RuntimeProviderEvidenceError,
    RuntimeProviderEvidencePaths,
    decode_artifact_identity,
    decode_runtime_provider_capabilities,
    decode_runtime_provider_receipt,
    decode_scoring_observation,
    encode_artifact_identity,
    encode_runtime_provider_capabilities,
    encode_runtime_provider_receipt,
    encode_scoring_observation,
    load_runtime_provider_evidence,
    runtime_provider_evidence_errors,
    write_runtime_provider_evidence,
)

_IMAGE_DIGEST = "sha256:" + "a" * 64


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _artifact() -> GGUFArtifactIdentity:
    return GGUFArtifactIdentity(
        artifact_name="tiny-model.gguf",
        sha256="1" * 64,
        byte_length=123,
        gguf_metadata_sha256="2" * 64,
        tensor_inventory_sha256="3" * 64,
        tokenizer_metadata_sha256="4" * 64,
    )


def _capabilities(*, provider_name: str = "llama_cpp") -> RuntimeProviderCapabilities:
    return RuntimeProviderCapabilities(
        provider_name=provider_name,
        artifact_formats=("gguf",),
        tasks=("text_causal",),
        metrics=("exact_match",),
        execution_modes=("container",),
        required_extra=None,
        required_image="ghcr.io/invarlock/runtime-llama-cpp@sha256:" + "b" * 64,
        platform_constraints=("linux",),
        evidence_surfaces=("behavior", "tokenizer"),
        supported_claim_sets=("invarlock-runtime-behavioral-regression-v1",),
    )


def _observation(
    artifact: GGUFArtifactIdentity,
    *,
    provider_name: str = "llama_cpp",
) -> ScoringObservation:
    output = "A"
    record = RuntimeScoringRecord(
        record_id="sample-1",
        input_sha256="5" * 64,
        status="ok",
        output_text=output,
        output_sha256=_sha256(output.encode("utf-8")),
    )
    records_payload = [asdict(record)]
    aggregate = _sha256(
        json.dumps(
            records_payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    return ScoringObservation(
        provider_name=provider_name,
        artifact_identity_sha256=artifact_identity_sha256(artifact),
        schedule_sha256="6" * 64,
        records=(record,),
        aggregate_source_sha256=aggregate,
    )


def _receipt(
    artifact: GGUFArtifactIdentity,
    observation: ScoringObservation,
    *,
    capabilities: RuntimeProviderCapabilities | None = None,
    observation_bytes: bytes | None = None,
    outer_image_digest: str | None = _IMAGE_DIGEST,
) -> RuntimeProviderReceipt:
    encoded_observation = observation_bytes or encode_scoring_observation(observation)
    return RuntimeProviderReceipt(
        plugin=RuntimeProviderPluginIdentity(
            name="llama_cpp",
            distribution="invarlock",
            distribution_version="0.13.0",
        ),
        backend=RuntimeBackendIdentity(
            name="llama.cpp",
            version="b1234",
            source_sha256="7" * 64,
            binary_sha256="8" * 64,
            build_sha256="9" * 64,
        ),
        capabilities=capabilities or _capabilities(),
        artifact_identity=artifact,
        execution_settings=RuntimeExecutionSettings(
            seed=42,
            context_length=512,
            batch_size=1,
            max_output_tokens=32,
            timeout_seconds=120,
            allow_network=False,
        ),
        device=RuntimeDeviceFacts(
            device_kind="cpu",
            device_name="x86_64",
        ),
        outer_image_digest=outer_image_digest,
        scoring_observation_sha256=_sha256(encoded_observation),
    )


def _bundle_values() -> tuple[
    GGUFArtifactIdentity, ScoringObservation, RuntimeProviderReceipt
]:
    artifact = _artifact()
    observation = _observation(artifact)
    return artifact, observation, _receipt(artifact, observation)


def test_typed_codecs_round_trip_canonical_contract_values() -> None:
    artifact, observation, receipt = _bundle_values()
    capabilities = receipt.capabilities

    artifact_bytes = encode_artifact_identity(artifact)

    assert artifact_bytes == canonical_artifact_identity_json(artifact)
    assert decode_artifact_identity(artifact_bytes) == artifact
    assert (
        decode_scoring_observation(encode_scoring_observation(observation))
        == observation
    )
    assert (
        decode_runtime_provider_capabilities(
            encode_runtime_provider_capabilities(capabilities)
        )
        == capabilities
    )
    assert (
        decode_runtime_provider_receipt(encode_runtime_provider_receipt(receipt))
        == receipt
    )


@pytest.mark.parametrize(
    "identity",
    [
        HFSnapshotArtifactIdentity(
            model_id="org/model",
            immutable_revision="a" * 40,
            checkpoint_tree_sha256=None,
            tokenizer_metadata_sha256="b" * 64,
        ),
        _artifact(),
        TensorRTLLMArtifactIdentity(
            bundle_name="engine-bundle",
            engine_bundle_tree_sha256="c" * 64,
            file_inventory_sha256="d" * 64,
            builder_config_sha256="e" * 64,
            tokenizer_metadata_sha256="a" * 64,
            engine_metadata_sha256="f" * 64,
            target_compute_capability="9.0",
        ),
    ],
    ids=("hf-snapshot", "gguf", "tensorrt-llm"),
)
def test_artifact_identity_codec_reconstructs_every_supported_type(
    identity: HFSnapshotArtifactIdentity
    | GGUFArtifactIdentity
    | TensorRTLLMArtifactIdentity,
) -> None:
    encoded = encode_artifact_identity(identity)

    assert encoded == canonical_artifact_identity_json(identity)
    assert decode_artifact_identity(encoded) == identity
    assert type(decode_artifact_identity(encoded)) is type(identity)


def test_write_and_reload_produces_cross_bound_canonical_sidecars(
    tmp_path: Path,
) -> None:
    artifact, observation, receipt = _bundle_values()

    bundle = write_runtime_provider_evidence(
        tmp_path,
        artifact_identity=artifact,
        scoring_observation=observation,
        receipt=receipt,
        expected_outer_image_digest=_IMAGE_DIGEST,
    )

    assert bundle.artifact_identity == artifact
    assert bundle.scoring_observation == observation
    assert bundle.receipt == receipt
    assert bundle.capabilities == receipt.capabilities
    assert bundle.artifact_identity_bytes == encode_artifact_identity(artifact)
    assert bundle.scoring_observation_bytes == encode_scoring_observation(observation)
    assert bundle.receipt_bytes == encode_runtime_provider_receipt(receipt)
    assert bundle.artifact_identity_sha256 == _sha256(bundle.artifact_identity_bytes)
    assert bundle.scoring_observation_sha256 == receipt.scoring_observation_sha256
    assert bundle.receipt_sha256 == _sha256(bundle.receipt_bytes)
    assert {path.name for path in tmp_path.iterdir()} == {
        ARTIFACT_IDENTITY_FILENAME,
        SCORING_OBSERVATION_FILENAME,
        PROVIDER_RECEIPT_FILENAME,
    }


def test_write_rejects_cross_binding_drift_without_creating_sidecars(
    tmp_path: Path,
) -> None:
    artifact, observation, receipt = _bundle_values()
    mismatched = replace(receipt, scoring_observation_sha256="0" * 64)

    with pytest.raises(
        RuntimeProviderEvidenceError,
        match="scoring_observation_sha256 does not match",
    ):
        write_runtime_provider_evidence(
            tmp_path,
            artifact_identity=artifact,
            scoring_observation=observation,
            receipt=mismatched,
        )

    assert list(tmp_path.iterdir()) == []


def test_cross_binding_reports_provider_artifact_capability_and_image_drift() -> None:
    artifact, observation, receipt = _bundle_values()
    other_artifact = replace(artifact, sha256="0" * 64)
    other_observation = replace(
        observation,
        provider_name="other_provider",
        artifact_identity_sha256="d" * 64,
    )
    unsupported_capabilities = replace(
        receipt.capabilities,
        provider_name="llama_cpp",
        artifact_formats=("hf_snapshot",),
    )
    drifted_receipt = replace(
        receipt,
        artifact_identity=other_artifact,
        capabilities=unsupported_capabilities,
    )

    errors = runtime_provider_evidence_errors(
        artifact_identity=artifact,
        scoring_observation=other_observation,
        receipt=drifted_receipt,
        expected_outer_image_digest="sha256:" + "c" * 64,
    )

    assert errors == (
        "receipt artifact_identity does not match the bound artifact",
        "receipt scoring_observation_sha256 does not match observation bytes",
        "receipt and observation provider names do not agree",
        "observation artifact_identity_sha256 does not match bound artifact",
        "bound artifact format is not declared by provider capabilities",
        "receipt outer_image_digest does not match expected runtime image",
    )


def test_cross_binding_rejects_noncanonical_expected_image_digest() -> None:
    artifact, observation, receipt = _bundle_values()

    errors = runtime_provider_evidence_errors(
        artifact_identity=artifact,
        scoring_observation=observation,
        receipt=receipt,
        expected_outer_image_digest="a" * 64,
    )

    assert errors == ("expected outer image digest is not canonical",)


@pytest.mark.parametrize(
    ("filename", "payload", "message"),
    [
        (
            SCORING_OBSERVATION_FILENAME,
            b'{"provider_name":"a","provider_name":"b"}',
            "duplicate key",
        ),
        (PROVIDER_RECEIPT_FILENAME, b'{"value":NaN}', "non-standard constant"),
        (ARTIFACT_IDENTITY_FILENAME, b"[]", "must be a JSON object"),
    ],
)
def test_reload_rejects_ambiguous_or_nonobject_json(
    tmp_path: Path,
    filename: str,
    payload: bytes,
    message: str,
) -> None:
    artifact, observation, receipt = _bundle_values()
    write_runtime_provider_evidence(
        tmp_path,
        artifact_identity=artifact,
        scoring_observation=observation,
        receipt=receipt,
    )
    (tmp_path / filename).write_bytes(payload)

    with pytest.raises(RuntimeProviderEvidenceError, match=message):
        load_runtime_provider_evidence(
            RuntimeProviderEvidencePaths.in_directory(tmp_path)
        )


def test_reload_rejects_schema_invalid_sidecar(tmp_path: Path) -> None:
    artifact, observation, receipt = _bundle_values()
    write_runtime_provider_evidence(
        tmp_path,
        artifact_identity=artifact,
        scoring_observation=observation,
        receipt=receipt,
    )
    (tmp_path / PROVIDER_RECEIPT_FILENAME).write_bytes(b"{}")

    with pytest.raises(RuntimeProviderEvidenceError, match="schema validation failed"):
        load_runtime_provider_evidence(
            RuntimeProviderEvidencePaths.in_directory(tmp_path)
        )


def test_reload_rejects_symlink_and_hardlink_role_aliases(tmp_path: Path) -> None:
    paths = RuntimeProviderEvidencePaths.in_directory(tmp_path)
    paths.artifact_identity.write_bytes(encode_artifact_identity(_artifact()))
    paths.receipt.write_bytes(b"{}")
    paths.scoring_observation.symlink_to(paths.artifact_identity)

    with pytest.raises(RuntimeProviderEvidenceError, match="regular file"):
        load_runtime_provider_evidence(paths)

    paths.scoring_observation.unlink()
    os.link(paths.artifact_identity, paths.scoring_observation)
    with pytest.raises(RuntimeProviderEvidenceError, match="must not alias"):
        load_runtime_provider_evidence(paths)


def test_reload_rejects_oversized_regular_sidecar(tmp_path: Path) -> None:
    paths = RuntimeProviderEvidencePaths.in_directory(tmp_path)
    paths.artifact_identity.write_bytes(b"{}")
    paths.scoring_observation.write_bytes(b"{}")
    paths.receipt.write_bytes(b"{}")
    with paths.artifact_identity.open("r+b") as handle:
        handle.truncate(MAX_RUNTIME_PROVIDER_SIDECAR_BYTES + 1)

    with pytest.raises(RuntimeProviderEvidenceError, match="exceeds"):
        load_runtime_provider_evidence(paths)


def test_atomic_write_failure_removes_temporary_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    artifact, observation, receipt = _bundle_values()

    def _fail_link(_source: Path, _target: Path, *, follow_symlinks: bool) -> None:
        assert follow_symlinks is False
        raise OSError("injected link failure")

    monkeypatch.setattr("invarlock.runtime_provider_evidence.os.link", _fail_link)

    with pytest.raises(RuntimeProviderEvidenceError, match="atomically write"):
        write_runtime_provider_evidence(
            tmp_path,
            artifact_identity=artifact,
            scoring_observation=observation,
            receipt=receipt,
        )

    assert list(tmp_path.iterdir()) == []


def test_late_publication_failure_rolls_back_earlier_sidecars(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    artifact, observation, receipt = _bundle_values()
    real_link = os.link
    calls = 0

    def _fail_second_link(source: Path, target: Path, *, follow_symlinks: bool) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected second-link failure")
        real_link(source, target, follow_symlinks=follow_symlinks)

    monkeypatch.setattr(
        "invarlock.runtime_provider_evidence.os.link", _fail_second_link
    )

    with pytest.raises(RuntimeProviderEvidenceError, match="atomically write"):
        write_runtime_provider_evidence(
            tmp_path,
            artifact_identity=artifact,
            scoring_observation=observation,
            receipt=receipt,
        )

    assert list(tmp_path.iterdir()) == []


def test_publication_race_does_not_replace_existing_target(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    artifact, observation, receipt = _bundle_values()
    real_link = os.link
    raced_target = tmp_path / ARTIFACT_IDENTITY_FILENAME

    def _race_link(source: Path, target: Path, *, follow_symlinks: bool) -> None:
        raced_target.write_text("created concurrently", encoding="utf-8")
        real_link(source, target, follow_symlinks=follow_symlinks)

    monkeypatch.setattr("invarlock.runtime_provider_evidence.os.link", _race_link)

    with pytest.raises(RuntimeProviderEvidenceError, match="atomically write"):
        write_runtime_provider_evidence(
            tmp_path,
            artifact_identity=artifact,
            scoring_observation=observation,
            receipt=receipt,
        )

    assert raced_target.read_text(encoding="utf-8") == "created concurrently"


def test_write_rejects_existing_canonical_sidecar(tmp_path: Path) -> None:
    artifact, observation, receipt = _bundle_values()
    (tmp_path / ARTIFACT_IDENTITY_FILENAME).write_text("do not replace")

    with pytest.raises(RuntimeProviderEvidenceError, match="already exist"):
        write_runtime_provider_evidence(
            tmp_path,
            artifact_identity=artifact,
            scoring_observation=observation,
            receipt=receipt,
        )

    assert (tmp_path / ARTIFACT_IDENTITY_FILENAME).read_text() == "do not replace"
