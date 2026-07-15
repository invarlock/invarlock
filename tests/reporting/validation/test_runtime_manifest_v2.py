from __future__ import annotations

import hashlib
import json
from pathlib import Path

import jsonschema
import pytest

from invarlock import public_contracts, runtime_verify
from invarlock.runtime_security_helpers import (
    RuntimeManifestExecution,
    RuntimeProviderManifestFiles,
    write_runtime_manifest,
    write_runtime_manifest_v2,
)

_IMAGE_DIGEST = "sha256:" + "a" * 64


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json(payload: object) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _artifact_payload() -> dict[str, object]:
    return {
        "format_version": "invarlock/model-artifact-identity-v1",
        "artifact_format": "gguf",
        "artifact_name": "tiny-model.gguf",
        "sha256": "1" * 64,
        "byte_length": 123,
        "gguf_metadata_sha256": "2" * 64,
        "tensor_inventory_sha256": "3" * 64,
        "tokenizer_metadata_sha256": "4" * 64,
    }


def _capabilities_payload() -> dict[str, object]:
    return {
        "format_version": "runtime-provider-capabilities-v1",
        "provider_name": "llama_cpp",
        "provider_abi": "1",
        "artifact_formats": ["gguf"],
        "tasks": ["text_causal"],
        "metrics": ["exact_match"],
        "execution_modes": ["container"],
        "required_extra": None,
        "required_image": None,
        "platform_constraints": ["linux"],
        "evidence_surfaces": ["behavior", "tokenizer", "weights"],
        "supported_claim_sets": ["invarlock-runtime-behavioral-regression-v1"],
        "degraded_modes": [],
        "unavailable_modes": [],
    }


def _observation_payload(artifact: dict[str, object]) -> dict[str, object]:
    output = "A"
    return {
        "format_version": "invarlock/runtime-scoring-observation-v1",
        "provider_name": "llama_cpp",
        "artifact_identity_sha256": _sha256(_canonical_json(artifact)),
        "schedule_sha256": "5" * 64,
        "records": [
            {
                "record_id": "sample-1",
                "input_sha256": "6" * 64,
                "status": "ok",
                "output_text": output,
                "output_sha256": _sha256(output.encode("utf-8")),
                "logprob_sum": None,
                "token_count": None,
                "utf8_byte_count": None,
                "error_code": None,
            }
        ],
        "aggregate_source_sha256": "7" * 64,
    }


def _receipt_payload(
    artifact: dict[str, object], observation_bytes: bytes
) -> dict[str, object]:
    return {
        "format_version": "invarlock/runtime-provider-receipt-v1",
        "plugin": {
            "name": "llama_cpp",
            "provider_abi": "1",
            "distribution": "invarlock-runtime-llama-cpp",
            "distribution_version": "1.0.0",
        },
        "backend": {
            "name": "llama.cpp",
            "version": "b1234",
            "source_sha256": "8" * 64,
            "binary_sha256": None,
            "build_sha256": None,
        },
        "capabilities": _capabilities_payload(),
        "artifact_identity": artifact,
        "execution_settings": {
            "seed": 43,
            "context_length": 512,
            "batch_size": 1,
            "max_output_tokens": 32,
            "timeout_seconds": 120,
            "allow_network": False,
        },
        "device": {
            "device_kind": "cpu",
            "device_name": "x86_64",
            "compute_capability": None,
            "driver_version": None,
        },
        "outer_image_digest": _IMAGE_DIGEST,
        "scoring_observation_sha256": _sha256(observation_bytes),
    }


def _rebind_sidecar(
    manifest: Path,
    sidecars: dict[str, Path],
    role: str,
    payload: bytes,
) -> None:
    sidecars[role].write_bytes(payload)
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    manifest_payload["runtime_provider"][role]["sha256"] = _sha256(payload)
    manifest.write_bytes(_canonical_json(manifest_payload))


def _load_sidecar(sidecars: dict[str, Path], role: str) -> dict[str, object]:
    payload = json.loads(sidecars[role].read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _rebind_observation_and_receipt(
    manifest: Path,
    sidecars: dict[str, Path],
    observation: dict[str, object],
) -> None:
    observation_bytes = _canonical_json(observation)
    _rebind_sidecar(manifest, sidecars, "scoring_observation", observation_bytes)
    receipt = _load_sidecar(sidecars, "receipt")
    receipt["scoring_observation_sha256"] = _sha256(observation_bytes)
    _rebind_sidecar(manifest, sidecars, "receipt", _canonical_json(receipt))


def _write_v2_inputs(tmp_path: Path) -> tuple[Path, Path, dict[str, Path]]:
    report = tmp_path / "evaluation.report.json"
    report.write_bytes(b'{"schema_version":"v1"}\n')
    config = tmp_path / "run.yaml"
    config.write_bytes(b"model:\n  id: tiny\n")
    sidecars = {
        "receipt": tmp_path / "runtime-provider.receipt.json",
        "scoring_observation": tmp_path / "runtime-scoring.observation.json",
        "artifact_identity": tmp_path / "model-artifact.identity.json",
    }
    artifact = _artifact_payload()
    artifact_bytes = _canonical_json(artifact)
    observation_bytes = _canonical_json(_observation_payload(artifact))
    receipt_bytes = _canonical_json(_receipt_payload(artifact, observation_bytes))
    sidecars["receipt"].write_bytes(receipt_bytes)
    sidecars["scoring_observation"].write_bytes(observation_bytes)
    sidecars["artifact_identity"].write_bytes(artifact_bytes)
    manifest = write_runtime_manifest_v2(
        report,
        config_path=config,
        provider_files=RuntimeProviderManifestFiles(
            receipt=sidecars["receipt"],
            scoring_observation=sidecars["scoring_observation"],
            artifact_identity=sidecars["artifact_identity"],
        ),
        execution=RuntimeManifestExecution(
            execution_mode="container",
            container_execution=True,
            image_ref="ghcr.io/invarlock/runtime:test",
            image_digest=_IMAGE_DIGEST,
            allow_network=False,
            allow_remote_code=False,
            allow_third_party_plugins=False,
        ),
    )
    return report, manifest, sidecars


def test_runtime_manifest_v2_writer_binds_portable_sibling_inputs(
    tmp_path: Path,
) -> None:
    report, manifest, sidecars = _write_v2_inputs(tmp_path)

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    jsonschema.validate(
        payload,
        public_contracts.load_runtime_manifest_v2_schema(),
    )
    assert payload["manifest_version"] == 2
    assert payload["verifier_contract_version"] == "runtime-manifest-v2"
    assert payload["report"] == {
        "path": report.name,
        "filename": report.name,
        "sha256": _sha256(report.read_bytes()),
    }
    assert payload["config"]["path"] == "run.yaml"
    assert payload["outer_container"]["image_digest"] == _IMAGE_DIGEST
    for role, path in sidecars.items():
        assert payload["runtime_provider"][role] == {
            "filename": path.name,
            "sha256": _sha256(path.read_bytes()),
        }
    assert str(tmp_path) not in manifest.read_text(encoding="utf-8")


def test_runtime_manifest_v2_public_contract_is_valid_and_packaged() -> None:
    schema = public_contracts.load_runtime_manifest_v2_schema()

    jsonschema.Draft202012Validator.check_schema(schema)
    assert schema["title"] == "InvarLock Runtime Manifest v2"
    assert (
        public_contracts.public_subcontract_catalog()["runtime_manifest_v2"]["version"]
        == "runtime-manifest-v2"
    )
    assert public_contracts.contract_catalog()["runtime_manifest_v2"]["path"] == (
        "contracts/runtime_manifest_v2.schema.json"
    )
    assert json.loads(
        (
            public_contracts.PACKAGE_CONTRACTS_ROOT / "runtime_manifest_v2.schema.json"
        ).read_text(encoding="utf-8")
    ) == json.loads(
        (public_contracts.CONTRACTS_ROOT / "runtime_manifest_v2.schema.json").read_text(
            encoding="utf-8"
        )
    )


def test_runtime_manifest_v2_verifies_all_bound_files(tmp_path: Path) -> None:
    report, manifest, _sidecars = _write_v2_inputs(tmp_path)

    result = runtime_verify.verify_runtime_manifest(
        report,
        manifest,
        expected_image_digest=_IMAGE_DIGEST,
        require_strict_runtime=True,
    )

    assert result.ok is True
    assert result.binding_verified is True
    assert result.expected_digest_matched is True
    assert result.declared_image_digest == _IMAGE_DIGEST


@pytest.mark.parametrize(
    "role", ["receipt", "scoring_observation", "artifact_identity"]
)
def test_runtime_manifest_v2_rejects_tampered_bound_file(
    tmp_path: Path, role: str
) -> None:
    report, manifest, sidecars = _write_v2_inputs(tmp_path)
    sidecars[role].write_bytes(b'{"tampered":true}\n')

    result = runtime_verify.verify_runtime_manifest(report, manifest)

    assert result.ok is False
    assert any(
        f"runtime_provider.{role} digest mismatch" in error for error in result.errors
    )


@pytest.mark.parametrize(
    ("role", "malformed", "reason"),
    [
        ("receipt", b'{"value":1,"value":2}', "duplicate key"),
        ("scoring_observation", b'{"value":NaN}', "non-standard constant"),
        ("artifact_identity", b"[", "not valid JSON"),
    ],
)
def test_runtime_manifest_v2_rejects_ambiguous_or_malformed_sidecar_json(
    tmp_path: Path,
    role: str,
    malformed: bytes,
    reason: str,
) -> None:
    report, manifest, sidecars = _write_v2_inputs(tmp_path)
    _rebind_sidecar(manifest, sidecars, role, malformed)

    result = runtime_verify.verify_runtime_manifest(
        report, manifest, require_strict_runtime=True
    )

    assert result.ok is False
    assert any(
        f"runtime_provider.{role}" in error and reason in error
        for error in result.errors
    )


@pytest.mark.parametrize(
    "role", ["receipt", "scoring_observation", "artifact_identity"]
)
def test_runtime_manifest_v2_rejects_schema_invalid_sidecars(
    tmp_path: Path, role: str
) -> None:
    report, manifest, sidecars = _write_v2_inputs(tmp_path)
    _rebind_sidecar(manifest, sidecars, role, b"{}")

    result = runtime_verify.verify_runtime_manifest(
        report, manifest, require_strict_runtime=True
    )

    assert result.ok is False
    assert any(
        f"runtime_provider.{role} schema validation failed" in error
        for error in result.errors
    )


def test_runtime_manifest_v2_cross_binds_receipt_artifact_identity(
    tmp_path: Path,
) -> None:
    report, manifest, sidecars = _write_v2_inputs(tmp_path)
    receipt = _load_sidecar(sidecars, "receipt")
    receipt_artifact = receipt["artifact_identity"]
    assert isinstance(receipt_artifact, dict)
    receipt_artifact["byte_length"] = 124
    _rebind_sidecar(manifest, sidecars, "receipt", _canonical_json(receipt))

    result = runtime_verify.verify_runtime_manifest(report, manifest)

    assert result.ok is False
    assert "receipt artifact_identity does not match the bound artifact file" in (
        result.errors
    )


def test_runtime_manifest_v2_cross_binds_observation_bytes_to_receipt(
    tmp_path: Path,
) -> None:
    report, manifest, sidecars = _write_v2_inputs(tmp_path)
    receipt = _load_sidecar(sidecars, "receipt")
    receipt["scoring_observation_sha256"] = "9" * 64
    _rebind_sidecar(manifest, sidecars, "receipt", _canonical_json(receipt))

    result = runtime_verify.verify_runtime_manifest(report, manifest)

    assert result.ok is False
    assert "receipt scoring_observation_sha256 does not match observation bytes" in (
        result.errors
    )


@pytest.mark.parametrize("surface", ["plugin", "capabilities", "observation"])
def test_runtime_manifest_v2_cross_binds_provider_name(
    tmp_path: Path, surface: str
) -> None:
    report, manifest, sidecars = _write_v2_inputs(tmp_path)
    if surface == "observation":
        observation = _load_sidecar(sidecars, "scoring_observation")
        observation["provider_name"] = "other_provider"
        _rebind_observation_and_receipt(manifest, sidecars, observation)
    else:
        receipt = _load_sidecar(sidecars, "receipt")
        provider = receipt[surface]
        assert isinstance(provider, dict)
        provider["name" if surface == "plugin" else "provider_name"] = "other_provider"
        _rebind_sidecar(manifest, sidecars, "receipt", _canonical_json(receipt))

    result = runtime_verify.verify_runtime_manifest(report, manifest)

    assert result.ok is False
    assert "receipt and observation provider names do not agree" in result.errors


def test_runtime_manifest_v2_cross_binds_artifact_digest_to_observation(
    tmp_path: Path,
) -> None:
    report, manifest, sidecars = _write_v2_inputs(tmp_path)
    observation = _load_sidecar(sidecars, "scoring_observation")
    observation["artifact_identity_sha256"] = "9" * 64
    _rebind_observation_and_receipt(manifest, sidecars, observation)

    result = runtime_verify.verify_runtime_manifest(report, manifest)

    assert result.ok is False
    assert "observation artifact_identity_sha256 does not match bound artifact" in (
        result.errors
    )


def test_runtime_manifest_v2_cross_binds_receipt_outer_image_digest(
    tmp_path: Path,
) -> None:
    report, manifest, sidecars = _write_v2_inputs(tmp_path)
    receipt = _load_sidecar(sidecars, "receipt")
    receipt["outer_image_digest"] = "sha256:" + "b" * 64
    _rebind_sidecar(manifest, sidecars, "receipt", _canonical_json(receipt))

    result = runtime_verify.verify_runtime_manifest(report, manifest)

    assert result.ok is False
    assert "receipt outer_image_digest does not match outer_container" in result.errors


def test_runtime_manifest_v2_requires_capability_for_bound_artifact_format(
    tmp_path: Path,
) -> None:
    report, manifest, sidecars = _write_v2_inputs(tmp_path)
    receipt = _load_sidecar(sidecars, "receipt")
    capabilities = receipt["capabilities"]
    assert isinstance(capabilities, dict)
    capabilities["artifact_formats"] = ["hf_snapshot"]
    _rebind_sidecar(manifest, sidecars, "receipt", _canonical_json(receipt))

    result = runtime_verify.verify_runtime_manifest(report, manifest)

    assert result.ok is False
    assert "bound artifact format is not declared by provider capabilities" in (
        result.errors
    )


@pytest.mark.parametrize(
    "config",
    [
        {"path": None, "sha256": None, "source": "missing"},
        {"path": None, "sha256": "9" * 64, "source": "inline"},
    ],
)
def test_runtime_manifest_v2_strict_verification_requires_file_config(
    tmp_path: Path, config: dict[str, object]
) -> None:
    report, manifest, _sidecars = _write_v2_inputs(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["config"] = config
    manifest.write_bytes(_canonical_json(payload))

    result = runtime_verify.verify_runtime_manifest(
        report, manifest, require_strict_runtime=True
    )

    assert result.ok is False
    assert "strict runtime manifest v2 requires a verifiable file config" in (
        result.errors
    )


def test_runtime_manifest_v2_rejects_unknown_and_tampered_manifest_fields(
    tmp_path: Path,
) -> None:
    report, manifest, _sidecars = _write_v2_inputs(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["unexpected"] = True
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    result = runtime_verify.verify_runtime_manifest(report, manifest)

    assert result.ok is False
    assert any("schema validation failed" in error for error in result.errors)


def test_runtime_verify_dispatch_rejects_unknown_version_pair(tmp_path: Path) -> None:
    report, manifest, _sidecars = _write_v2_inputs(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["manifest_version"] = 3
    payload["verifier_contract_version"] = "runtime-manifest-v3"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    result = runtime_verify.verify_runtime_manifest(report, manifest)

    assert result.ok is False
    assert result.errors[0] == (
        "unsupported runtime manifest version pair: "
        "manifest_version=3, verifier_contract_version='runtime-manifest-v3'"
    )


@pytest.mark.parametrize(
    ("manifest_version", "contract_version"),
    [(1, "runtime-manifest-v2"), (2, "runtime-manifest-v1")],
)
def test_runtime_verify_dispatch_rejects_mismatched_known_contract(
    tmp_path: Path, manifest_version: int, contract_version: str
) -> None:
    report, manifest, _sidecars = _write_v2_inputs(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["manifest_version"] = manifest_version
    payload["verifier_contract_version"] = contract_version
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    result = runtime_verify.verify_runtime_manifest(report, manifest)

    assert result.ok is False
    assert result.errors[0].startswith("runtime manifest schema validation failed:")


def test_runtime_manifest_v2_rejects_tampered_file_config(tmp_path: Path) -> None:
    report, manifest, _sidecars = _write_v2_inputs(tmp_path)
    (tmp_path / "run.yaml").write_text("tampered: true\n", encoding="utf-8")

    result = runtime_verify.verify_runtime_manifest(report, manifest)

    assert result.ok is False
    assert any("config digest mismatch" in error for error in result.errors)


def test_runtime_manifest_v2_rejects_report_filename_mismatch(tmp_path: Path) -> None:
    report, manifest, _sidecars = _write_v2_inputs(tmp_path)
    renamed_report = tmp_path / "renamed.report.json"
    renamed_report.write_bytes(report.read_bytes())

    result = runtime_verify.verify_runtime_manifest(renamed_report, manifest)

    assert result.ok is False
    assert any("report.path does not match" in error for error in result.errors)
    assert any("report.filename does not match" in error for error in result.errors)


def test_runtime_manifest_v2_rejects_colliding_provider_bindings(
    tmp_path: Path,
) -> None:
    report, manifest, _sidecars = _write_v2_inputs(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["runtime_provider"]["receipt"] = payload["runtime_provider"][
        "artifact_identity"
    ]
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    result = runtime_verify.verify_runtime_manifest(report, manifest)

    assert result.ok is False
    assert "runtime provider binding filenames must be distinct" in result.errors


def test_runtime_manifest_v2_writer_rejects_provider_report_collision(
    tmp_path: Path,
) -> None:
    report, _manifest, sidecars = _write_v2_inputs(tmp_path)

    with pytest.raises(ValueError, match="distinct from the report"):
        write_runtime_manifest_v2(
            report,
            provider_files=RuntimeProviderManifestFiles(
                receipt=report,
                scoring_observation=sidecars["scoring_observation"],
                artifact_identity=sidecars["artifact_identity"],
            ),
            execution=RuntimeManifestExecution(
                execution_mode="container",
                container_execution=True,
                image_ref="image",
                image_digest=_IMAGE_DIGEST,
                allow_network=False,
                allow_remote_code=False,
                allow_third_party_plugins=False,
            ),
        )


def test_runtime_manifest_v2_writer_rejects_non_sibling_provider_file(
    tmp_path: Path,
) -> None:
    report, _manifest, sidecars = _write_v2_inputs(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    external = outside / "receipt.json"
    external.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="sibling"):
        write_runtime_manifest_v2(
            report,
            provider_files=RuntimeProviderManifestFiles(
                receipt=external,
                scoring_observation=sidecars["scoring_observation"],
                artifact_identity=sidecars["artifact_identity"],
            ),
            execution=RuntimeManifestExecution(
                execution_mode="container",
                container_execution=True,
                image_ref="image",
                image_digest=_IMAGE_DIGEST,
                allow_network=False,
                allow_remote_code=False,
                allow_third_party_plugins=False,
            ),
        )


def test_runtime_manifest_v1_writer_and_verifier_remain_unchanged(
    tmp_path: Path,
) -> None:
    report = tmp_path / "evaluation.report.json"
    report.write_text("{}\n", encoding="utf-8")
    manifest = write_runtime_manifest(
        report,
        execution=RuntimeManifestExecution(
            execution_mode="container",
            container_execution=True,
            image_ref="image",
            image_digest=_IMAGE_DIGEST,
            allow_network=False,
            allow_remote_code=False,
            allow_third_party_plugins=False,
        ),
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))

    assert payload["manifest_version"] == 1
    assert payload["verifier_contract_version"] == "runtime-manifest-v1"
    assert "runtime" in payload
    assert "outer_container" not in payload
    assert runtime_verify.verify_runtime_manifest(report, manifest).ok is True
