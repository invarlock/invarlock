from __future__ import annotations

import copy
import json
from pathlib import Path

import jsonschema
import pytest

from invarlock import public_contracts
from invarlock.core.runtime_provider import types as runtime_provider_types


def _digest(character: str) -> str:
    return character * 64


def _capabilities_payload() -> dict[str, object]:
    return {
        "format_version": "runtime-provider-capabilities-v1",
        "provider_name": "hf_transformers",
        "provider_abi": "1",
        "artifact_formats": ["hf_snapshot"],
        "tasks": ["text_causal"],
        "metrics": ["exact_match"],
        "execution_modes": ["in_process"],
        "required_extra": "hf",
        "required_image": None,
        "platform_constraints": ["python"],
        "evidence_surfaces": [
            "behavior",
            "tokenizer",
            "weights",
            "modules",
            "activations",
            "build",
        ],
        "supported_claim_sets": ["invarlock-weight-edit-regression-v2"],
        "degraded_modes": [],
        "unavailable_modes": [],
    }


def _artifact_payload() -> dict[str, object]:
    return {
        "format_version": "invarlock/model-artifact-identity-v1",
        "artifact_format": "gguf",
        "artifact_name": "tinyllama-q4.gguf",
        "sha256": _digest("a"),
        "byte_length": 123,
        "gguf_metadata_sha256": _digest("b"),
        "tensor_inventory_sha256": _digest("c"),
        "tokenizer_metadata_sha256": _digest("d"),
    }


def _tensorrt_artifact_payload() -> dict[str, object]:
    return {
        "format_version": "invarlock/model-artifact-identity-v1",
        "artifact_format": "tensorrt_llm_engine",
        "bundle_name": "tensorrt-llm-sha256-" + _digest("a"),
        "engine_bundle_tree_sha256": _digest("a"),
        "file_inventory_sha256": _digest("b"),
        "builder_config_sha256": _digest("c"),
        "tokenizer_metadata_sha256": _digest("d"),
        "engine_metadata_sha256": _digest("e"),
        "target_compute_capability": "9.0",
    }


def _settings_payload() -> dict[str, object]:
    return {
        "seed": 43,
        "context_length": 512,
        "batch_size": 1,
        "max_output_tokens": 32,
        "timeout_seconds": 120,
        "allow_network": False,
    }


def _scoring_payload() -> dict[str, object]:
    return {
        "format_version": "invarlock/runtime-scoring-observation-v1",
        "provider_name": "llama_cpp",
        "artifact_identity_sha256": _digest("a"),
        "schedule_sha256": _digest("b"),
        "records": [
            {
                "record_id": "sample-1",
                "input_sha256": _digest("c"),
                "status": "ok",
                "output_text": "A",
                "output_sha256": _digest("d"),
                "logprob_sum": -1.5,
                "token_count": 1,
                "utf8_byte_count": 1,
                "error_code": None,
            }
        ],
        "aggregate_source_sha256": _digest("e"),
    }


def _receipt_payload() -> dict[str, object]:
    return {
        "format_version": "invarlock/runtime-provider-receipt-v1",
        "plugin": {
            "name": "llama_cpp",
            "provider_abi": "1",
            "distribution": "invarlock",
            "distribution_version": "0.13.0",
        },
        "backend": {
            "name": "llama.cpp",
            "version": "b1234",
            "source_sha256": _digest("a"),
            "binary_sha256": _digest("b"),
            "build_sha256": _digest("c"),
        },
        "capabilities": _capabilities_payload(),
        "artifact_identity": _artifact_payload(),
        "execution_settings": _settings_payload(),
        "device": {
            "device_kind": "cpu",
            "device_name": "x86_64",
            "compute_capability": None,
            "driver_version": None,
            "cuda_runtime_version": None,
        },
        "outer_image_digest": "sha256:" + _digest("d"),
        "scoring_observation_sha256": _digest("e"),
    }


@pytest.mark.parametrize(
    ("loader", "payload"),
    [
        (
            public_contracts.load_runtime_provider_capabilities_schema,
            _capabilities_payload(),
        ),
        (public_contracts.load_model_artifact_identity_schema, _artifact_payload()),
        (public_contracts.load_runtime_provider_receipt_schema, _receipt_payload()),
        (public_contracts.load_runtime_scoring_observation_schema, _scoring_payload()),
    ],
)
def test_runtime_provider_public_schemas_accept_canonical_payloads(
    loader, payload
) -> None:
    schema = loader()
    assert isinstance(schema, dict)
    jsonschema.validate(payload, schema)


@pytest.mark.parametrize(
    ("loader", "payload"),
    [
        (
            public_contracts.load_runtime_provider_capabilities_schema,
            _capabilities_payload(),
        ),
        (public_contracts.load_model_artifact_identity_schema, _artifact_payload()),
        (public_contracts.load_runtime_provider_receipt_schema, _receipt_payload()),
        (public_contracts.load_runtime_scoring_observation_schema, _scoring_payload()),
    ],
)
def test_runtime_provider_public_schemas_reject_unknown_fields(loader, payload) -> None:
    malformed = copy.deepcopy(payload)
    malformed["unexpected"] = True

    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(malformed, loader())


@pytest.mark.parametrize(
    "payload",
    [
        {**_artifact_payload(), "artifact_name": "/private/model.gguf"},
        {**_artifact_payload(), "artifact_name": "../model.gguf"},
        {**_artifact_payload(), "sha256": "not-a-digest"},
        {**_artifact_payload(), "byte_length": True},
    ],
)
def test_model_artifact_identity_schema_rejects_paths_and_malformed_facts(
    payload,
) -> None:
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(
            payload, public_contracts.load_model_artifact_identity_schema()
        )


def test_tensorrt_artifact_schemas_require_canonical_tokenizer_binding() -> None:
    artifact = _tensorrt_artifact_payload()
    jsonschema.validate(
        artifact,
        public_contracts.load_model_artifact_identity_schema(),
    )

    receipt = _receipt_payload()
    receipt["artifact_identity"] = artifact
    jsonschema.validate(
        receipt, public_contracts.load_runtime_provider_receipt_schema()
    )

    for malformed_digest in (None, "bad", "A" * 64):
        malformed_artifact = copy.deepcopy(artifact)
        if malformed_digest is None:
            malformed_artifact.pop("tokenizer_metadata_sha256")
        else:
            malformed_artifact["tokenizer_metadata_sha256"] = malformed_digest
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(
                malformed_artifact,
                public_contracts.load_model_artifact_identity_schema(),
            )

        malformed_receipt = _receipt_payload()
        malformed_receipt["artifact_identity"] = malformed_artifact
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(
                malformed_receipt,
                public_contracts.load_runtime_provider_receipt_schema(),
            )

    malformed_capability = copy.deepcopy(artifact)
    malformed_capability["target_compute_capability"] = "09.0"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(
            malformed_capability,
            public_contracts.load_model_artifact_identity_schema(),
        )


def test_runtime_scoring_observation_rejects_non_finite_or_incomplete_error() -> None:
    malformed = _scoring_payload()
    malformed["records"][0]["logprob_sum"] = "NaN"  # type: ignore[index]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(
            malformed,
            public_contracts.load_runtime_scoring_observation_schema(),
        )

    error = _scoring_payload()
    error["records"][0].update(  # type: ignore[union-attr,index]
        {"status": "error", "error_code": None}
    )
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(
            error, public_contracts.load_runtime_scoring_observation_schema()
        )


def test_runtime_provider_contracts_are_cataloged_and_packaged_byte_identically() -> (
    None
):
    expected = {
        "runtime_provider_capabilities",
        "model_artifact_identity",
        "runtime_provider_receipt",
        "runtime_scoring_observation",
    }
    catalog = public_contracts.contract_catalog()
    assert expected <= set(catalog)

    for name in expected:
        relative = catalog[name]["path"]
        repository = Path(relative)
        packaged = public_contracts.PACKAGE_CONTRACTS_ROOT.joinpath(repository.name)
        assert packaged.is_file()
        assert packaged.read_bytes() == repository.read_bytes()
        assert isinstance(json.loads(packaged.read_text(encoding="utf-8")), dict)


def test_runtime_provider_contract_versions_are_single_sourced() -> None:
    assert (
        public_contracts.RUNTIME_PROVIDER_ABI_VERSION
        == runtime_provider_types.RUNTIME_PROVIDER_ABI_VERSION
    )
    assert (
        public_contracts.RUNTIME_PROVIDER_CAPABILITIES_FORMAT_VERSION
        == runtime_provider_types.RUNTIME_PROVIDER_CAPABILITIES_FORMAT
    )
    assert (
        public_contracts.MODEL_ARTIFACT_IDENTITY_FORMAT_VERSION
        == runtime_provider_types.MODEL_ARTIFACT_IDENTITY_FORMAT
    )
    assert (
        public_contracts.RUNTIME_PROVIDER_RECEIPT_FORMAT_VERSION
        == runtime_provider_types.RUNTIME_PROVIDER_RECEIPT_FORMAT
    )
    assert (
        public_contracts.RUNTIME_SCORING_OBSERVATION_FORMAT_VERSION
        == runtime_provider_types.RUNTIME_SCORING_OBSERVATION_FORMAT
    )


@pytest.mark.parametrize(
    ("filename", "loader"),
    [
        (
            "runtime_provider_capabilities.json",
            public_contracts.load_runtime_provider_capabilities_schema,
        ),
        (
            "model_artifact_identity.schema.json",
            public_contracts.load_model_artifact_identity_schema,
        ),
        (
            "runtime_provider_receipt.schema.json",
            public_contracts.load_runtime_provider_receipt_schema,
        ),
        (
            "runtime_scoring_observation.schema.json",
            public_contracts.load_runtime_scoring_observation_schema,
        ),
    ],
)
def test_runtime_provider_contract_loaders_fail_closed_on_non_objects(
    monkeypatch: pytest.MonkeyPatch, filename: str, loader
) -> None:
    monkeypatch.setattr(
        public_contracts,
        "_load_contract_or_raise",
        lambda requested: [requested],
    )

    with pytest.raises(public_contracts.ContractLoadError, match=filename):
        loader()
