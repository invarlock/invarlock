from __future__ import annotations

import base64
import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

from invarlock.core.runtime_provider import (
    GGUFArtifactIdentity,
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
    build_runtime_behavioral_schedule_from_material,
)
from invarlock.core.runtime_provider.behavioral_observation import (
    runtime_scoring_records_sha256,
)
from invarlock.evidence_pack import (
    EVIDENCE_PACK_FORMAT,
    EvidenceObservation,
    EvidencePackError,
    InputIdentity,
    RuntimeSideEvidence,
    build_comparison_report,
    derive_paired_records,
    publish_comparison_evidence,
    verify_comparison_evidence,
)
from invarlock.evidence_pack_contract import (
    MAX_OBSERVATION_BYTES,
    canonical_json_bytes,
    evidence_observation_bytes,
    sha256_digest,
)
from invarlock.evidence_pack_integrity import (
    EVIDENCE_PACK_SIGNATURE_FORMAT,
    public_key_fingerprint,
)
from invarlock.runtime_manifest import write_runtime_manifest
from invarlock.runtime_provider_evidence import write_runtime_provider_evidence
from invarlock.runtime_security_helpers import (
    RuntimeManifestExecution,
    RuntimeProviderManifestFiles,
)


def _digest(character: str) -> str:
    return "sha256:" + character * 64


def _signing_key(tmp_path: Path) -> tuple[Path, str]:
    key = ed25519.Ed25519PrivateKey.generate()
    path = tmp_path / "release-key.pem"
    path.write_bytes(
        key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    path.chmod(0o600)
    return path, public_key_fingerprint(key.public_key())


def _schedule() -> RuntimeBehavioralSchedule:
    return build_runtime_behavioral_schedule_from_material(
        dataset_identity={
            "provider": "local",
            "dataset_name": "evidence-pack-fixture",
            "config_name": "evaluation-records-jsonl-v1",
            "revision": "e" * 64,
            "split": "validation",
        },
        records=[
            {
                "record_id": "example-001",
                "input_text": "Return A",
                "expected_output": "A",
            },
            {
                "record_id": "example-002",
                "input_text": "Return B",
                "expected_output": "B",
            },
        ],
    )


def _side_evidence(
    root: Path,
    *,
    schedule: RuntimeBehavioralSchedule,
    image_digest: str,
    artifact_marker: str,
    outputs: tuple[str, str] | None = None,
    logprob_sums: tuple[float, float] | None = None,
    token_counts: tuple[int, int] = (1, 1),
    utf8_byte_counts: tuple[int, int] = (1, 1),
    tokenizer_marker: str = "3",
    capability_metrics: tuple[str, ...] = (
        "exact_match",
        "normalized_nll_per_utf8_byte",
    ),
    role: str = "baseline",
    policy_digest: str | None = None,
) -> RuntimeSideEvidence:
    root.mkdir(parents=True)
    artifact = GGUFArtifactIdentity(
        artifact_name=f"model-{artifact_marker}.gguf",
        sha256=artifact_marker * 64,
        byte_length=123,
        gguf_metadata_sha256="1" * 64,
        tensor_inventory_sha256="2" * 64,
        tokenizer_metadata_sha256=tokenizer_marker * 64,
    )
    if (outputs is None) == (logprob_sums is None):
        raise ValueError("exactly one scoring fact source is required")
    if outputs is not None:
        records = tuple(
            RuntimeScoringRecord(
                record_id=scheduled.record_id,
                input_sha256=scheduled.input_sha256,
                status="ok",
                output_text=output,
                output_sha256=hashlib.sha256(output.encode("utf-8")).hexdigest(),
            )
            for scheduled, output in zip(schedule.records, outputs, strict=True)
        )
    else:
        assert logprob_sums is not None
        records = tuple(
            RuntimeScoringRecord(
                record_id=scheduled.record_id,
                input_sha256=scheduled.input_sha256,
                status="ok",
                logprob_sum=logprob_sum,
                token_count=token_count,
                utf8_byte_count=utf8_byte_count,
            )
            for scheduled, logprob_sum, token_count, utf8_byte_count in zip(
                schedule.records,
                logprob_sums,
                token_counts,
                utf8_byte_counts,
                strict=True,
            )
        )
    observation = ScoringObservation(
        provider_name="llama_cpp",
        artifact_identity_sha256=artifact_identity_sha256(artifact),
        schedule_sha256=schedule.schedule_sha256,
        records=records,
        aggregate_source_sha256=runtime_scoring_records_sha256(
            [asdict(record) for record in records]
        ),
    )
    capabilities = RuntimeProviderCapabilities(
        provider_name="llama_cpp",
        artifact_formats=("gguf",),
        tasks=("text_causal",),
        metrics=capability_metrics,  # type: ignore[arg-type]
        execution_modes=("container",),
        required_extra=None,
        required_image=None,
    )
    observation_bytes = json.dumps(
        asdict(observation),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    receipt = RuntimeProviderReceipt(
        plugin=RuntimeProviderPluginIdentity(
            name="llama_cpp",
            distribution="invarlock-runtime-llama-cpp",
            distribution_version="1.0.0",
        ),
        backend=RuntimeBackendIdentity(
            name="llama.cpp",
            version="test",
            source_sha256="4" * 64,
            binary_sha256=None,
            build_sha256=None,
        ),
        capabilities=capabilities,
        artifact_identity=artifact,
        execution_settings=RuntimeExecutionSettings(
            seed=42,
            context_length=512,
            batch_size=1,
            max_output_tokens=16,
            timeout_seconds=120,
            allow_network=False,
        ),
        device=RuntimeDeviceFacts(device_kind="cpu", device_name="test-cpu"),
        outer_image_digest=image_digest,
        scoring_observation_sha256=hashlib.sha256(observation_bytes).hexdigest(),
    )
    persisted = write_runtime_provider_evidence(
        root,
        artifact_identity=artifact,
        scoring_observation=observation,
        receipt=receipt,
        expected_outer_image_digest=image_digest,
    )
    run_report = {
        "format": "invarlock/runtime-side-report-v1",
        "provider": observation.provider_name,
        "artifact_identity_sha256": observation.artifact_identity_sha256,
        "scoring_observation_sha256": persisted.scoring_observation_sha256,
        "schedule_sha256": observation.schedule_sha256,
        "record_count": len(records),
    }
    report_path = root / "report.json"
    report_path.write_text(
        json.dumps(run_report, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    config_path = root / "run.yaml"
    config_path.write_bytes(
        canonical_json_bytes(
            {
                "format": "invarlock/runtime-side-config-v1",
                "role": role,
                "provider": "llama_cpp",
                "artifact_identity_sha256": observation.artifact_identity_sha256,
                "schedule_sha256": schedule.schedule_sha256,
                "policy_digest": policy_digest or _digest("f"),
            }
        )
    )
    manifest_path = write_runtime_manifest(
        report_path,
        config_path=config_path,
        provider_files=RuntimeProviderManifestFiles(
            receipt=persisted.paths.receipt,
            scoring_observation=persisted.paths.scoring_observation,
            artifact_identity=persisted.paths.artifact_identity,
        ),
        execution=RuntimeManifestExecution(
            execution_mode="container",
            container_execution=True,
            image_ref="ghcr.io/invarlock/runtime:test",
            image_digest=image_digest,
            allow_network=False,
            allow_remote_code=False,
            allow_third_party_plugins=False,
        ),
    )
    return RuntimeSideEvidence(
        run_report=report_path.read_bytes(),
        runtime_manifest=manifest_path.read_bytes(),
        runtime_config=config_path.read_bytes(),
        artifact_identity=persisted.artifact_identity_bytes,
        provider_receipt=persisted.receipt_bytes,
        scoring_observation=persisted.scoring_observation_bytes,
    )


def _request(metric: str = "exact_match") -> dict[str, object]:
    def side(model_id: str) -> dict[str, object]:
        return {
            "artifact": {
                "model_id": model_id,
                "locator": f"artifact://{model_id}",
            },
            "runtime": {"provider": "llama_cpp", "settings": {}},
        }

    return {
        "format_version": "invarlock/evaluation-request-v1",
        "comparison": {
            "baseline": side("model-c.gguf"),
            "subject": side("model-d.gguf"),
            "dataset": {
                "format_version": "invarlock/local-jsonl-preparation-v1",
                "source_sha256": "e" * 64,
                "source_format": "jsonl",
                "name": "evidence-pack-fixture",
                "split": "validation",
                "input_field": "prompt",
                "expected_output_field": "expected",
                "id_field": "case_id",
                "content_role": None,
                "content_id_field": None,
                "content_sha256_field": None,
                "content_byte_length_field": None,
                "content_media_type_field": None,
                "limit": 2,
                "selected_record_count": 2,
            },
            "task": "text_causal",
            "metric": metric,
            "policy": "inputs/policy.json",
        },
        "execution": {"mode": "run"},
        "output": {"evidence": "evidence"},
    }


def _policy_bytes(delta_min_pp: float) -> bytes:
    return (
        json.dumps(
            {
                "resolved_policy": {
                    "metrics": {"exact_match": {"delta_min_pp": delta_min_pp}}
                }
            },
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _nll_policy_bytes(ratio_limit: float) -> bytes:
    return (
        json.dumps(
            {
                "resolved_policy": {
                    "metrics": {
                        "normalized_nll_per_utf8_byte": {"ratio_max": ratio_limit}
                    }
                }
            },
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _publish(
    tmp_path: Path,
    *,
    destination_name: str = "evidence",
    delta_min_pp: float = -100.0,
    observations: tuple[EvidenceObservation, ...] = (),
) -> tuple[Path, Path, str, dict[str, str], Path, dict[str, object]]:
    schedule = _schedule()
    baseline_runtime = _digest("a")
    subject_runtime = _digest("b")
    policy_path = tmp_path / "acceptance-policy.json"
    policy_path.write_bytes(_policy_bytes(delta_min_pp))
    policy_digest = sha256_digest(policy_path.read_bytes())
    baseline_side = _side_evidence(
        tmp_path / "evaluation-baseline",
        schedule=schedule,
        image_digest=baseline_runtime,
        artifact_marker="c",
        outputs=("A", "B"),
        role="baseline",
        policy_digest=policy_digest,
    )
    subject_side = _side_evidence(
        tmp_path / "evaluation-subject",
        schedule=schedule,
        image_digest=subject_runtime,
        artifact_marker="d",
        outputs=("A", "wrong"),
        role="subject",
        policy_digest=policy_digest,
    )
    key_path, fingerprint = _signing_key(tmp_path)
    normalized_request = _request()
    if observations:
        normalized_request["observations"] = [
            {
                "id": observation.observation_id,
                "kind": observation.kind,
                "scope": observation.scope,
                "payload_digest": sha256_digest(observation.payload),
            }
            for observation in sorted(
                observations, key=lambda item: item.observation_id
            )
        ]
    arguments: dict[str, object] = {
        "comparison_id": "single-comparison",
        "baseline": InputIdentity(
            sha256_digest(baseline_side.artifact_identity),
            locator="artifact://model-c.gguf",
        ),
        "subject": InputIdentity(
            sha256_digest(subject_side.artifact_identity),
            locator="artifact://model-d.gguf",
        ),
        "dataset": InputIdentity(
            "sha256:" + schedule.schedule_sha256,
            locator="schedule/runtime-behavioral-schedule.json",
        ),
        "baseline_runtime": InputIdentity(
            baseline_runtime, locator=f"runtime:{baseline_runtime}"
        ),
        "subject_runtime": InputIdentity(
            subject_runtime, locator=f"runtime:{subject_runtime}"
        ),
        "policy": InputIdentity(
            sha256_digest(policy_path.read_bytes()), locator="inputs/policy.json"
        ),
        "normalized_request": normalized_request,
        "schedule": schedule,
        "policy_bytes": policy_path.read_bytes(),
        "baseline_evidence": baseline_side,
        "subject_evidence": subject_side,
        "signing_key_path": key_path,
        "observations": observations,
    }
    destination = tmp_path / destination_name
    publish_comparison_evidence(destination, **arguments)
    return (
        destination,
        policy_path,
        fingerprint,
        {"baseline": baseline_runtime, "subject": subject_runtime},
        key_path,
        arguments,
    )


def _verification_anchors(arguments: dict[str, object]) -> dict[str, Any]:
    baseline = arguments["baseline"]
    subject = arguments["subject"]
    dataset = arguments["dataset"]
    assert isinstance(baseline, InputIdentity)
    assert isinstance(subject, InputIdentity)
    assert isinstance(dataset, InputIdentity)
    return {
        "expected_artifact_digests": {
            "baseline": baseline.digest,
            "subject": subject.digest,
        },
        "expected_schedule_digest": dataset.digest,
    }


def _rebind_and_resign_pack(pack: Path, key_path: Path) -> None:
    checksums_path = pack / "checksums.sha256"
    manifest_path = pack / "manifest.json"
    signature_path = pack / "manifest.signature.json"
    for path in (checksums_path, manifest_path, signature_path):
        path.chmod(0o644)
    checksum_lines: list[str] = []
    for line in checksums_path.read_text(encoding="utf-8").splitlines():
        _digest_value, relative = line.split("  ", maxsplit=1)
        payload_digest = hashlib.sha256((pack / relative).read_bytes()).hexdigest()
        checksum_lines.append(f"{payload_digest}  {relative}\n")
    checksums = "".join(checksum_lines).encode("utf-8")
    checksums_path.write_bytes(checksums)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for reference in manifest["inputs"].values():
        identity_bytes = (pack / reference["path"]).read_bytes()
        reference["digest"] = sha256_digest(identity_bytes)
        reference["material_digest"] = json.loads(identity_bytes)["digest"]
    for reference in manifest["evidence"].values():
        reference["digest"] = sha256_digest((pack / reference["path"]).read_bytes())
    for reference in manifest.get("observations", {}).values():
        reference["digest"] = sha256_digest((pack / reference["path"]).read_bytes())
    paired_bytes = (pack / "records/paired-records.json").read_bytes()
    manifest["paired_records"]["digest"] = sha256_digest(paired_bytes)
    manifest["checksums_sha256_digest"] = hashlib.sha256(checksums).hexdigest()
    manifest_bytes = (
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    manifest_path.write_bytes(manifest_bytes)
    key = serialization.load_pem_private_key(key_path.read_bytes(), password=None)
    assert isinstance(key, ed25519.Ed25519PrivateKey)
    public_key = key.public_key()
    signature_path.write_text(
        json.dumps(
            {
                "format": EVIDENCE_PACK_SIGNATURE_FORMAT,
                "algorithm": "ed25519",
                "signing_key_fingerprint": public_key_fingerprint(public_key),
                "public_key": {
                    "encoding": "pem",
                    "value": public_key.public_bytes(
                        encoding=serialization.Encoding.PEM,
                        format=serialization.PublicFormat.SubjectPublicKeyInfo,
                    ).decode("ascii"),
                },
                "signature": {
                    "encoding": "base64",
                    "value": base64.b64encode(key.sign(manifest_bytes)).decode("ascii"),
                },
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )


def test_publish_and_verify_real_runtime_provider_snapshots(tmp_path: Path) -> None:
    pack, policy, fingerprint, runtimes, _key, arguments = _publish(tmp_path)

    manifest = json.loads((pack / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["format"] == EVIDENCE_PACK_FORMAT
    assert set(manifest["inputs"]) == {
        "baseline",
        "subject",
        "dataset",
        "baseline_runtime",
        "subject_runtime",
        "policy",
    }
    assert (pack / "schedule/runtime-behavioral-schedule.json").is_file()
    assert (pack / "providers/baseline/run.yaml").is_file()
    paired = json.loads(
        (pack / "records/paired-records.json").read_text(encoding="utf-8")
    )
    assert [item["baseline"]["score"] for item in paired["records"]] == [1.0, 1.0]
    assert [item["subject"]["score"] for item in paired["records"]] == [1.0, 0.0]

    result = verify_comparison_evidence(
        pack,
        policy_path=policy,
        **_verification_anchors(arguments),
        expected_runtime_digests=runtimes,
        expected_signer_fingerprint=fingerprint,
    )

    assert result.status == 0
    assert result.payload["ok"] is True
    assert result.payload["authenticity"] == "pinned"
    assert result.payload["policy_verdict"] == "pass"
    assert result.payload["observations"] == []


def test_authenticated_observation_is_verified_without_decision_authority(
    tmp_path: Path,
) -> None:
    observation = EvidenceObservation(
        observation_id="spectral-summary",
        scope="subject",
        kind="spectral",
        payload=canonical_json_bytes(
            {
                "format": "invarlock/diagnostic-observation-v1",
                "status": "observation",
                "verdict": "fail",
                "stable_rank": 1.25,
            }
        ),
    )
    pack, policy, fingerprint, runtimes, _key, arguments = _publish(
        tmp_path,
        observations=(observation,),
    )

    result = verify_comparison_evidence(
        pack,
        policy_path=policy,
        **_verification_anchors(arguments),
        expected_runtime_digests=runtimes,
        expected_signer_fingerprint=fingerprint,
    )

    assert result.status == 0
    assert result.payload["policy_verdict"] == "pass"
    assert result.payload["ok"] is True
    assert result.payload["observations"] == [
        {
            "observation_id": "spectral-summary",
            "kind": "spectral",
            "scope": "subject",
            "digest": sha256_digest(
                (pack / "observations/spectral-summary.json").read_bytes()
            ),
        }
    ]


def test_authenticated_observation_with_rebound_acceptance_authority_fails(
    tmp_path: Path,
) -> None:
    observation = EvidenceObservation(
        observation_id="variance-summary",
        scope="comparison",
        kind="variance",
        payload=canonical_json_bytes({"population_variance": 0.5}),
    )
    pack, policy, fingerprint, runtimes, key, arguments = _publish(
        tmp_path,
        observations=(observation,),
    )
    path = pack / "observations/variance-summary.json"
    path.chmod(0o644)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["authority"] = "acceptance"
    path.write_bytes(canonical_json_bytes(payload))
    _rebind_and_resign_pack(pack, key)

    result = verify_comparison_evidence(
        pack,
        policy_path=policy,
        **_verification_anchors(arguments),
        expected_runtime_digests=runtimes,
        expected_signer_fingerprint=fingerprint,
    )

    assert result.payload["ok"] is False
    assert result.payload["integrity_ok"] is False
    assert "authority" in " ".join(result.payload["errors"])


def test_publication_rejects_noncanonical_observation_payload(tmp_path: Path) -> None:
    with pytest.raises(EvidencePackError, match="canonical JSON"):
        EvidenceObservation(
            observation_id="variance-summary",
            scope="comparison",
            kind="variance",
            payload=b'{"population_variance": 0.5}\n',
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"observation_id": "bad/id"}, "observation_id is invalid"),
        ({"kind": "Bad Kind"}, "kind is invalid"),
        ({"scope": "runtime"}, "scope is invalid"),
        ({"payload": b""}, "non-empty bytes"),
        ({"payload": canonical_json_bytes(["not-an-object"])}, "JSON object"),
        ({"payload": b'{"value":1,"value":2}\n'}, "duplicate key"),
        (
            {"payload": canonical_json_bytes({"blob": "x" * MAX_OBSERVATION_BYTES})},
            "byte limit",
        ),
    ],
)
def test_observation_contract_rejects_ambiguous_or_unbounded_inputs(
    overrides: dict[str, object],
    message: str,
) -> None:
    arguments: dict[str, object] = {
        "observation_id": "subject-variance",
        "scope": "subject",
        "kind": "variance",
        "payload": canonical_json_bytes({"population_variance": 0.5}),
    }
    arguments.update(overrides)

    with pytest.raises(EvidencePackError, match=message):
        EvidenceObservation(**cast(Any, arguments))


def test_observation_envelope_requires_both_artifact_bindings() -> None:
    observation = EvidenceObservation(
        observation_id="subject-variance",
        scope="subject",
        kind="variance",
        payload=canonical_json_bytes({"population_variance": 0.5}),
    )

    with pytest.raises(EvidencePackError, match="baseline and subject"):
        evidence_observation_bytes(
            observation,
            comparison_id="comparison",
            schedule_digest=_digest("a"),
            policy_digest=_digest("b"),
            artifact_digests={"baseline": _digest("c")},
        )


def test_observation_id_is_a_portable_evidence_filename() -> None:
    with pytest.raises(EvidencePackError, match="observation_id is invalid"):
        EvidenceObservation(
            observation_id="windows:drive",
            scope="comparison",
            kind="context",
            payload=canonical_json_bytes({"value": 1}),
        )


def test_publication_rejects_observation_substituted_after_request_normalization(
    tmp_path: Path,
) -> None:
    _pack, _policy, _fingerprint, _runtimes, _key, arguments = _publish(tmp_path)
    observation = EvidenceObservation(
        observation_id="subject-variance",
        scope="subject",
        kind="variance",
        payload=canonical_json_bytes({"population_variance": 0.5}),
    )
    normalized_request_value = arguments["normalized_request"]
    assert isinstance(normalized_request_value, dict)
    normalized_request = dict(normalized_request_value)
    normalized_request["observations"] = [
        {
            "id": observation.observation_id,
            "kind": observation.kind,
            "scope": observation.scope,
            "payload_digest": _digest("f"),
        }
    ]
    arguments["normalized_request"] = normalized_request
    arguments["observations"] = (observation,)

    with pytest.raises(
        EvidencePackError,
        match="normalized request observations do not match publication inputs",
    ):
        publish_comparison_evidence(
            tmp_path / "substituted-observation", **cast(Any, arguments)
        )


def test_publication_rejects_duplicate_observation_ids(tmp_path: Path) -> None:
    _pack, _policy, _fingerprint, _runtimes, _key, arguments = _publish(tmp_path)
    observation = EvidenceObservation(
        observation_id="duplicate-observation",
        scope="comparison",
        kind="variance",
        payload=canonical_json_bytes({"population_variance": 0.5}),
    )
    descriptor = {
        "id": observation.observation_id,
        "kind": observation.kind,
        "scope": observation.scope,
        "payload_digest": sha256_digest(observation.payload),
    }
    normalized_request_value = arguments["normalized_request"]
    assert isinstance(normalized_request_value, dict)
    normalized_request = dict(normalized_request_value)
    normalized_request["observations"] = [descriptor, dict(descriptor)]
    arguments["normalized_request"] = normalized_request
    arguments["observations"] = (observation, observation)

    with pytest.raises(EvidencePackError, match="duplicate observation_id"):
        publish_comparison_evidence(
            tmp_path / "duplicate-observations", **cast(Any, arguments)
        )


@pytest.mark.parametrize(
    ("binding_path", "replacement"),
    [
        (("comparison_id",), "other-comparison"),
        (("schedule_digest",), _digest("e")),
        (("policy_digest",), _digest("e")),
        (("artifact_digests", "subject"), _digest("e")),
    ],
)
def test_signed_observation_cannot_be_rebound_to_other_evaluation_inputs(
    tmp_path: Path,
    binding_path: tuple[str, ...],
    replacement: str,
) -> None:
    observation = EvidenceObservation(
        observation_id="subject-variance",
        scope="subject",
        kind="variance",
        payload=canonical_json_bytes({"population_variance": 0.5}),
    )
    pack, policy, fingerprint, runtimes, key, arguments = _publish(
        tmp_path,
        observations=(observation,),
    )
    path = pack / "observations/subject-variance.json"
    path.chmod(0o644)
    envelope = json.loads(path.read_text(encoding="utf-8"))
    target = envelope["bindings"]
    assert isinstance(target, dict)
    for component in binding_path[:-1]:
        target = target[component]
        assert isinstance(target, dict)
    target[binding_path[-1]] = replacement
    path.write_bytes(canonical_json_bytes(envelope))
    _rebind_and_resign_pack(pack, key)

    result = verify_comparison_evidence(
        pack,
        policy_path=policy,
        **_verification_anchors(arguments),
        expected_runtime_digests=runtimes,
        expected_signer_fingerprint=fingerprint,
    )

    assert result.payload["ok"] is False
    assert "does not bind the comparison, schedule, policy, and artifacts" in " ".join(
        result.payload["errors"]
    )


def test_signed_observation_must_match_normalized_request_descriptor(
    tmp_path: Path,
) -> None:
    observation = EvidenceObservation(
        observation_id="subject-variance",
        scope="subject",
        kind="variance",
        payload=canonical_json_bytes({"population_variance": 0.5}),
    )
    pack, policy, fingerprint, runtimes, key, arguments = _publish(
        tmp_path,
        observations=(observation,),
    )
    request_path = pack / "request.json"
    request_path.chmod(0o644)
    request = json.loads(request_path.read_text(encoding="utf-8"))
    request["observations"][0]["payload_digest"] = _digest("e")
    request_path.write_bytes(canonical_json_bytes(request))
    _rebind_and_resign_pack(pack, key)

    result = verify_comparison_evidence(
        pack,
        policy_path=policy,
        **_verification_anchors(arguments),
        expected_runtime_digests=runtimes,
        expected_signer_fingerprint=fingerprint,
    )

    assert result.payload["ok"] is False
    assert "does not match normalized request" in " ".join(result.payload["errors"])


def test_verifier_rejects_signed_noncanonical_observation_envelope(
    tmp_path: Path,
) -> None:
    observation = EvidenceObservation(
        observation_id="subject-variance",
        scope="subject",
        kind="variance",
        payload=canonical_json_bytes({"population_variance": 0.5}),
    )
    pack, policy, fingerprint, runtimes, key, arguments = _publish(
        tmp_path,
        observations=(observation,),
    )
    observation_path = pack / "observations/subject-variance.json"
    observation_path.chmod(0o644)
    envelope = json.loads(observation_path.read_text(encoding="utf-8"))
    observation_path.write_text(
        json.dumps(envelope, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _rebind_and_resign_pack(pack, key)

    result = verify_comparison_evidence(
        pack,
        policy_path=policy,
        **_verification_anchors(arguments),
        expected_runtime_digests=runtimes,
        expected_signer_fingerprint=fingerprint,
    )

    assert result.payload["ok"] is False
    assert "must use canonical JSON" in " ".join(result.payload["errors"])


def test_verifier_rejects_duplicate_signed_observation_request_entries(
    tmp_path: Path,
) -> None:
    observation = EvidenceObservation(
        observation_id="subject-variance",
        scope="subject",
        kind="variance",
        payload=canonical_json_bytes({"population_variance": 0.5}),
    )
    pack, policy, fingerprint, runtimes, key, arguments = _publish(
        tmp_path,
        observations=(observation,),
    )
    request_path = pack / "request.json"
    request_path.chmod(0o644)
    request = json.loads(request_path.read_text(encoding="utf-8"))
    request["observations"].append(dict(request["observations"][0]))
    request_path.write_bytes(canonical_json_bytes(request))
    _rebind_and_resign_pack(pack, key)

    result = verify_comparison_evidence(
        pack,
        policy_path=policy,
        **_verification_anchors(arguments),
        expected_runtime_digests=runtimes,
        expected_signer_fingerprint=fingerprint,
    )

    assert result.payload["ok"] is False
    assert "normalized request observation entry is invalid" in " ".join(
        result.payload["errors"]
    )


def test_verifier_rejects_signed_observation_payload_over_publisher_limit(
    tmp_path: Path,
) -> None:
    observation = EvidenceObservation(
        observation_id="subject-variance",
        scope="subject",
        kind="variance",
        payload=canonical_json_bytes({"population_variance": 0.5}),
    )
    pack, policy, fingerprint, runtimes, key, arguments = _publish(
        tmp_path,
        observations=(observation,),
    )
    observation_path = pack / "observations/subject-variance.json"
    request_path = pack / "request.json"
    observation_path.chmod(0o644)
    request_path.chmod(0o644)
    envelope = json.loads(observation_path.read_text(encoding="utf-8"))
    envelope["payload"] = {"blob": "x" * MAX_OBSERVATION_BYTES}
    observation_path.write_bytes(canonical_json_bytes(envelope))
    request = json.loads(request_path.read_text(encoding="utf-8"))
    request["observations"][0]["payload_digest"] = sha256_digest(
        canonical_json_bytes(envelope["payload"])
    )
    request_path.write_bytes(canonical_json_bytes(request))
    _rebind_and_resign_pack(pack, key)

    result = verify_comparison_evidence(
        pack,
        policy_path=policy,
        **_verification_anchors(arguments),
        expected_runtime_digests=runtimes,
        expected_signer_fingerprint=fingerprint,
    )

    assert result.payload["ok"] is False
    assert f"payload exceeds the {MAX_OBSERVATION_BYTES}-byte limit" in " ".join(
        result.payload["errors"]
    )


@pytest.mark.parametrize("side", ["baseline", "subject"])
def test_each_runtime_anchor_is_independent(tmp_path: Path, side: str) -> None:
    pack, policy, fingerprint, runtimes, _key, arguments = _publish(tmp_path)
    wrong = dict(runtimes)
    wrong[side] = _digest("e")

    result = verify_comparison_evidence(
        pack,
        policy_path=policy,
        **_verification_anchors(arguments),
        expected_runtime_digests=wrong,
        expected_signer_fingerprint=fingerprint,
    )

    assert result.payload["ok"] is False
    assert f"{side} runtime identity" in " ".join(result.payload["errors"])


@pytest.mark.parametrize("side", ["baseline", "subject"])
def test_each_artifact_anchor_rejects_the_wrong_model(
    tmp_path: Path, side: str
) -> None:
    pack, policy, fingerprint, runtimes, _key, arguments = _publish(tmp_path)
    anchors = _verification_anchors(arguments)
    artifacts = dict(anchors["expected_artifact_digests"])
    artifacts[side] = _digest("e")

    result = verify_comparison_evidence(
        pack,
        policy_path=policy,
        expected_artifact_digests=artifacts,
        expected_schedule_digest=anchors["expected_schedule_digest"],
        expected_runtime_digests=runtimes,
        expected_signer_fingerprint=fingerprint,
    )

    assert result.payload["ok"] is False
    assert f"{side} artifact identity" in " ".join(result.payload["errors"])


def test_schedule_anchor_rejects_the_wrong_evaluation_schedule(tmp_path: Path) -> None:
    pack, policy, fingerprint, runtimes, _key, arguments = _publish(tmp_path)
    anchors = _verification_anchors(arguments)

    result = verify_comparison_evidence(
        pack,
        policy_path=policy,
        expected_artifact_digests=anchors["expected_artifact_digests"],
        expected_schedule_digest=_digest("e"),
        expected_runtime_digests=runtimes,
        expected_signer_fingerprint=fingerprint,
    )

    assert result.payload["ok"] is False
    assert "canonical schedule identity" in " ".join(result.payload["errors"])


def test_caller_policy_bytes_override_embedded_identity(tmp_path: Path) -> None:
    pack, _policy, fingerprint, runtimes, _key, arguments = _publish(tmp_path)
    other = tmp_path / "other-policy.json"
    other.write_bytes(_policy_bytes(-50.0))

    result = verify_comparison_evidence(
        pack,
        policy_path=other,
        **_verification_anchors(arguments),
        expected_runtime_digests=runtimes,
        expected_signer_fingerprint=fingerprint,
    )

    assert result.payload["ok"] is False
    assert "caller policy anchor" in " ".join(result.payload["errors"])


def test_policy_failure_is_not_an_integrity_failure(tmp_path: Path) -> None:
    pack, policy, fingerprint, runtimes, _key, arguments = _publish(
        tmp_path, delta_min_pp=-10.0
    )

    result = verify_comparison_evidence(
        pack,
        policy_path=policy,
        **_verification_anchors(arguments),
        expected_runtime_digests=runtimes,
        expected_signer_fingerprint=fingerprint,
    )

    assert result.payload["ok"] is False
    assert result.payload["integrity_ok"] is True
    assert result.payload["policy_verdict"] == "fail"
    assert result.payload["errors"] == []


def test_signed_tampering_of_pair_score_is_rejected_by_observation_replay(
    tmp_path: Path,
) -> None:
    pack, policy, fingerprint, runtimes, key, arguments = _publish(tmp_path)
    paired_path = pack / "records/paired-records.json"
    paired_path.chmod(0o644)
    paired = json.loads(paired_path.read_text(encoding="utf-8"))
    paired["records"][1]["subject"]["score"] = 1.0
    paired_path.write_text(
        json.dumps(paired, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    _rebind_and_resign_pack(pack, key)

    result = verify_comparison_evidence(
        pack,
        policy_path=policy,
        **_verification_anchors(arguments),
        expected_runtime_digests=runtimes,
        expected_signer_fingerprint=fingerprint,
    )

    assert result.payload["ok"] is False
    assert "scores derived from provider observations" in " ".join(
        result.payload["errors"]
    )


def test_resigned_input_locator_rebind_is_rejected(tmp_path: Path) -> None:
    pack, policy, fingerprint, runtimes, key, arguments = _publish(tmp_path)
    identity_path = pack / "inputs/baseline.json"
    identity_path.chmod(0o644)
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    identity["locator"] = "artifact://model-d.gguf"
    identity_path.write_bytes(canonical_json_bytes(identity))
    _rebind_and_resign_pack(pack, key)

    result = verify_comparison_evidence(
        pack,
        policy_path=policy,
        **_verification_anchors(arguments),
        expected_runtime_digests=runtimes,
        expected_signer_fingerprint=fingerprint,
    )

    assert result.payload["ok"] is False
    assert "baseline input locator" in " ".join(result.payload["errors"])


def test_resigned_request_locator_swap_is_rejected(tmp_path: Path) -> None:
    pack, policy, fingerprint, runtimes, key, arguments = _publish(tmp_path)
    request_path = pack / "request.json"
    request_path.chmod(0o644)
    request = json.loads(request_path.read_text(encoding="utf-8"))
    request["comparison"]["baseline"]["artifact"]["locator"] = "artifact://model-d.gguf"
    request_path.write_bytes(canonical_json_bytes(request))
    _rebind_and_resign_pack(pack, key)

    result = verify_comparison_evidence(
        pack,
        policy_path=policy,
        **_verification_anchors(arguments),
        expected_runtime_digests=runtimes,
        expected_signer_fingerprint=fingerprint,
    )

    assert result.payload["ok"] is False
    assert "baseline input locator" in " ".join(result.payload["errors"])


def test_resigned_request_model_id_swap_is_rejected(tmp_path: Path) -> None:
    pack, policy, fingerprint, runtimes, key, arguments = _publish(tmp_path)
    request_path = pack / "request.json"
    request_path.chmod(0o644)
    request = json.loads(request_path.read_text(encoding="utf-8"))
    request["comparison"]["baseline"]["artifact"]["model_id"] = "model-d.gguf"
    request_path.write_bytes(canonical_json_bytes(request))
    _rebind_and_resign_pack(pack, key)

    result = verify_comparison_evidence(
        pack,
        policy_path=policy,
        **_verification_anchors(arguments),
        expected_runtime_digests=runtimes,
        expected_signer_fingerprint=fingerprint,
    )

    assert result.payload["ok"] is False
    assert "request model_id" in " ".join(result.payload["errors"])


def test_runtime_side_report_must_bind_same_observation(tmp_path: Path) -> None:
    schedule = _schedule()
    side = _side_evidence(
        tmp_path / "side",
        schedule=schedule,
        image_digest=_digest("a"),
        artifact_marker="c",
        outputs=("A", "B"),
    )
    bad_report = json.loads(side.run_report)
    bad_report["schedule_sha256"] = "0" * 64
    bad_side = RuntimeSideEvidence(
        **{
            **asdict(side),
            "run_report": (
                json.dumps(bad_report, sort_keys=True, separators=(",", ":")) + "\n"
            ).encode("utf-8"),
        }
    )

    with pytest.raises(EvidencePackError, match="does not bind"):
        from invarlock.evidence_pack import derive_paired_records

        derive_paired_records(
            schedule=schedule,
            metric="exact_match",
            baseline=bad_side,
            subject=side,
            baseline_identity_digest=sha256_digest(side.artifact_identity),
            subject_identity_digest=sha256_digest(side.artifact_identity),
            baseline_runtime_digest=_digest("a"),
            subject_runtime_digest=_digest("a"),
        )


def test_multiple_choice_is_explicitly_out_of_scope(tmp_path: Path) -> None:
    _pack, _policy, _fingerprint, _runtimes, _key, arguments = _publish(tmp_path)
    arguments["normalized_request"] = _request("multiple_choice_accuracy")

    with pytest.raises(EvidencePackError, match="is not one of"):
        publish_comparison_evidence(tmp_path / "mc", **arguments)


def test_normalized_nll_is_replayed_from_typed_observations(tmp_path: Path) -> None:
    schedule = _schedule()
    baseline_runtime = _digest("a")
    subject_runtime = _digest("b")
    policy = tmp_path / "nll-policy.json"
    policy.write_bytes(_nll_policy_bytes(1.3))
    policy_digest = sha256_digest(policy.read_bytes())
    baseline_side = _side_evidence(
        tmp_path / "nll-baseline",
        schedule=schedule,
        image_digest=baseline_runtime,
        artifact_marker="c",
        logprob_sums=(-2.0, -2.0),
        utf8_byte_counts=(1, 1),
        role="baseline",
        policy_digest=policy_digest,
    )
    subject_side = _side_evidence(
        tmp_path / "nll-subject",
        schedule=schedule,
        image_digest=subject_runtime,
        artifact_marker="d",
        logprob_sums=(-2.5, -2.0),
        utf8_byte_counts=(1, 1),
        role="subject",
        policy_digest=policy_digest,
    )
    key, fingerprint = _signing_key(tmp_path)
    request = _request("normalized_nll_per_utf8_byte")
    destination = tmp_path / "nll-evidence"

    publish_comparison_evidence(
        destination,
        comparison_id="nll-comparison",
        baseline=InputIdentity(
            sha256_digest(baseline_side.artifact_identity),
            locator="artifact://model-c.gguf",
        ),
        subject=InputIdentity(
            sha256_digest(subject_side.artifact_identity),
            locator="artifact://model-d.gguf",
        ),
        dataset=InputIdentity(
            "sha256:" + schedule.schedule_sha256,
            locator="schedule/runtime-behavioral-schedule.json",
        ),
        baseline_runtime=InputIdentity(
            baseline_runtime, locator=f"runtime:{baseline_runtime}"
        ),
        subject_runtime=InputIdentity(
            subject_runtime, locator=f"runtime:{subject_runtime}"
        ),
        policy=InputIdentity(
            sha256_digest(policy.read_bytes()), locator="inputs/policy.json"
        ),
        normalized_request=request,
        schedule=schedule,
        policy_bytes=policy.read_bytes(),
        baseline_evidence=baseline_side,
        subject_evidence=subject_side,
        signing_key_path=key,
    )

    result = verify_comparison_evidence(
        destination,
        policy_path=policy,
        expected_artifact_digests={
            "baseline": sha256_digest(baseline_side.artifact_identity),
            "subject": sha256_digest(subject_side.artifact_identity),
        },
        expected_schedule_digest="sha256:" + schedule.schedule_sha256,
        expected_runtime_digests={
            "baseline": baseline_runtime,
            "subject": subject_runtime,
        },
        expected_signer_fingerprint=fingerprint,
    )
    paired = json.loads(
        (destination / "records/paired-records.json").read_text(encoding="utf-8")
    )

    assert result.payload["ok"] is True
    assert [record["baseline"]["score"] for record in paired["records"]] == [
        2.0,
        2.0,
    ]
    assert [record["subject"]["score"] for record in paired["records"]] == [
        2.5,
        2.0,
    ]


def test_normalized_nll_reports_derived_perplexity_when_token_facts_are_comparable(
    tmp_path: Path,
) -> None:
    schedule = _schedule()
    policy = {
        "resolved_policy": {
            "metrics": {"normalized_nll_per_utf8_byte": {"ratio_max": 1.25}}
        }
    }
    baseline = _side_evidence(
        tmp_path / "relative-baseline",
        schedule=schedule,
        image_digest=_digest("a"),
        artifact_marker="c",
        logprob_sums=(-2.0, -6.0),
        token_counts=(2, 3),
        capability_metrics=("normalized_nll_per_utf8_byte",),
    )
    subject = _side_evidence(
        tmp_path / "relative-subject",
        schedule=schedule,
        image_digest=_digest("b"),
        artifact_marker="d",
        logprob_sums=(-2.2, -6.6),
        token_counts=(2, 3),
        capability_metrics=("normalized_nll_per_utf8_byte",),
    )

    paired = derive_paired_records(
        schedule=schedule,
        metric="normalized_nll_per_utf8_byte",
        baseline=baseline,
        subject=subject,
        baseline_identity_digest=sha256_digest(baseline.artifact_identity),
        subject_identity_digest=sha256_digest(subject.artifact_identity),
        baseline_runtime_digest=_digest("a"),
        subject_runtime_digest=_digest("b"),
    )
    report = build_comparison_report(
        comparison_id="likelihood-comparison",
        paired_records=paired,
        policy=policy,
        policy_digest=_digest("f"),
    )

    derived = paired["derived_measurements"]["perplexity_ratio"]
    assert derived["status"] == "available"
    assert derived["tokenizer_metadata_sha256"] == "3" * 64
    assert derived["target_token_count"] == 5
    assert derived["baseline_perplexity"] == pytest.approx(math.exp(1.6))
    assert derived["subject_perplexity"] == pytest.approx(math.exp(1.76))
    assert derived["ratio"] == pytest.approx(math.exp(0.16))
    assert report["baseline"]["mean_score"] == pytest.approx(4.0)
    assert report["subject"]["mean_score"] == pytest.approx(4.4)
    assert report["comparison"] == {
        "kind": "normalized_nll_ratio",
        "value": pytest.approx(1.1),
        "maximum": 1.25,
    }
    assert report["derived_measurements"] == paired["derived_measurements"]
    assert report["verdict"] == "pass"


@pytest.mark.parametrize("mismatch", ["tokenizer", "token_count"])
def test_normalized_nll_marks_derived_perplexity_unavailable_when_incomparable(
    tmp_path: Path, mismatch: str
) -> None:
    schedule = _schedule()
    baseline = _side_evidence(
        tmp_path / f"{mismatch}-baseline",
        schedule=schedule,
        image_digest=_digest("a"),
        artifact_marker="c",
        logprob_sums=(-2.0, -6.0),
        token_counts=(2, 3),
        capability_metrics=("normalized_nll_per_utf8_byte",),
    )
    subject = _side_evidence(
        tmp_path / f"{mismatch}-subject",
        schedule=schedule,
        image_digest=_digest("b"),
        artifact_marker="d",
        logprob_sums=(-2.2, -6.6),
        token_counts=(2, 4) if mismatch == "token_count" else (2, 3),
        tokenizer_marker="4" if mismatch == "tokenizer" else "3",
        capability_metrics=("normalized_nll_per_utf8_byte",),
    )

    reason = (
        "tokenizer_contracts_differ"
        if mismatch == "tokenizer"
        else "target_token_counts_differ"
    )
    paired = derive_paired_records(
        schedule=schedule,
        metric="normalized_nll_per_utf8_byte",
        baseline=baseline,
        subject=subject,
        baseline_identity_digest=sha256_digest(baseline.artifact_identity),
        subject_identity_digest=sha256_digest(subject.artifact_identity),
        baseline_runtime_digest=_digest("a"),
        subject_runtime_digest=_digest("b"),
    )

    assert paired["derived_measurements"]["perplexity_ratio"] == {
        "status": "unavailable",
        "basis": "authenticated_target_likelihood",
        "method": "target_token_weighted_perplexity_ratio_v1",
        "reason": reason,
    }


def test_resigned_runtime_config_role_swap_is_rejected(tmp_path: Path) -> None:
    pack, policy, fingerprint, runtimes, key, arguments = _publish(tmp_path)
    config_path = pack / "providers/baseline/run.yaml"
    manifest_path = pack / "providers/baseline/runtime.manifest.json"
    config_path.chmod(0o644)
    manifest_path.chmod(0o644)
    config = json.loads(config_path.read_bytes())
    config["role"] = "subject"
    config_path.write_bytes(canonical_json_bytes(config))
    runtime_manifest = json.loads(manifest_path.read_bytes())
    runtime_manifest["config"]["sha256"] = hashlib.sha256(
        config_path.read_bytes()
    ).hexdigest()
    manifest_path.write_bytes(canonical_json_bytes(runtime_manifest))
    _rebind_and_resign_pack(pack, key)

    result = verify_comparison_evidence(
        pack,
        policy_path=policy,
        **_verification_anchors(arguments),
        expected_runtime_digests=runtimes,
        expected_signer_fingerprint=fingerprint,
    )

    assert result.payload["ok"] is False
    assert "runtime config does not bind" in " ".join(result.payload["errors"])


def test_normalized_nll_rejects_receipt_without_metric_capability(
    tmp_path: Path,
) -> None:
    schedule = _schedule()
    side = _side_evidence(
        tmp_path / "exact-only-side",
        schedule=schedule,
        image_digest=_digest("a"),
        artifact_marker="c",
        logprob_sums=(-2.0, -2.0),
        capability_metrics=("exact_match",),
    )

    with pytest.raises(EvidencePackError, match="does not declare metric"):
        derive_paired_records(
            schedule=schedule,
            metric="normalized_nll_per_utf8_byte",
            baseline=side,
            subject=side,
            baseline_identity_digest=sha256_digest(side.artifact_identity),
            subject_identity_digest=sha256_digest(side.artifact_identity),
            baseline_runtime_digest=_digest("a"),
            subject_runtime_digest=_digest("a"),
        )


def test_publication_is_no_clobber(tmp_path: Path) -> None:
    _pack, _policy, _fingerprint, _runtimes, _key, arguments = _publish(tmp_path)
    destination = tmp_path / "owned"
    destination.mkdir()
    marker = destination / "caller.txt"
    marker.write_text("preserve", encoding="utf-8")

    with pytest.raises(EvidencePackError, match="already exists"):
        publish_comparison_evidence(destination, **arguments)

    assert marker.read_text(encoding="utf-8") == "preserve"


def test_same_frozen_inputs_and_key_publish_identical_bytes(tmp_path: Path) -> None:
    pack_a, _policy, _fingerprint, _runtimes, _key, arguments = _publish(tmp_path)
    pack_b = tmp_path / "evidence-b"
    publish_comparison_evidence(pack_b, **arguments)

    files_a = {
        str(path.relative_to(pack_a)): path.read_bytes()
        for path in pack_a.rglob("*")
        if path.is_file()
    }
    files_b = {
        str(path.relative_to(pack_b)): path.read_bytes()
        for path in pack_b.rglob("*")
        if path.is_file()
    }
    assert files_a == files_b
