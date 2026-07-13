from __future__ import annotations

import copy

import pytest

from invarlock.core.backend_inventory import BACKEND_INVENTORY_SCHEMA
from invarlock.core.runtime_quantization.validation import (
    runtime_type_name_matches_adapter,
    validate_backend_inventory_payload,
    validate_runtime_quantization_proof_payload,
)
from invarlock.core.runtime_quantization_proof import (
    RUNTIME_QUANTIZATION_PROOF_KIND,
    RUNTIME_QUANTIZATION_PROOF_SCHEMA,
)


def _bnb_proof() -> dict[str, object]:
    return {
        "schema": RUNTIME_QUANTIZATION_PROOF_SCHEMA,
        "proof_kind": RUNTIME_QUANTIZATION_PROOF_KIND,
        "adapter": "hf_bnb",
        "backend": "bitsandbytes",
        "backend_version": "0.49.0",
        "ok": True,
        "status": "verified_live_runtime_types",
        "reason": "recognized_live_quantized_runtime_types",
        "live_model_observed": True,
        "module_inventory_observed": True,
        "recognized_quantized_runtime_type_count": 1,
        "recognized_quantized_runtime_types": ["bitsandbytes.nn.modules.Linear8bitLt"],
        "recognized_quantized_runtime_observation_kinds": ["module"],
        "live_model_quantization_method": None,
        "backend_runtime_importable": None,
        "backend_runtime_import_error_type": None,
        "backend_runtime_version": None,
        "backend_runtime_compatibility_bridge_required": None,
        "backend_runtime_compatibility_bridge_applied": None,
        "backend_runtime_compatibility_bridge_error_type": None,
        "packed_storage_artifact_proof_required": False,
        "artifact_binding": "not_attempted",
    }


def _bnb_inventory() -> dict[str, object]:
    return {
        "schema": BACKEND_INVENTORY_SCHEMA,
        "adapter": "hf_bnb",
        "backend": "bitsandbytes",
        "backend_version": "0.49.0",
        "quantized_module_count": 1,
        "quantized_module_types": ["bitsandbytes.nn.modules.Linear8bitLt"],
        "quantized_observation_kinds": ["module"],
        "transformers_version": "5.12.0",
        "quantization_config": {"load_in_8bit": True},
        "device_map": "cuda",
        "memory_footprint": {"reported_bytes": 1024},
        "load_smoke": True,
        "inference_smoke": True,
    }


def _proof_errors(payload: dict[str, object]) -> list[str]:
    return validate_runtime_quantization_proof_payload(
        payload=payload,
        expected_adapter="hf_bnb",
        expected_backend="bitsandbytes",
        expected_schema=RUNTIME_QUANTIZATION_PROOF_SCHEMA,
        expected_proof_kind=RUNTIME_QUANTIZATION_PROOF_KIND,
    )


@pytest.mark.parametrize(
    ("field", "forged_value", "fragment"),
    [
        (
            "backend_runtime_importable",
            False,
            "must record backend_runtime_importable as null",
        ),
        (
            "backend_runtime_import_error_type",
            "ImportError",
            "must record backend_runtime_import_error_type as null",
        ),
        (
            "backend_runtime_compatibility_bridge_applied",
            False,
            "must record backend_runtime_compatibility_bridge_applied as null",
        ),
    ],
)
def test_non_gptq_receipt_cannot_smuggle_gptq_runtime_state(
    field: str,
    forged_value: object,
    fragment: str,
) -> None:
    payload = _bnb_proof()
    payload[field] = forged_value

    assert any(fragment in error for error in _proof_errors(payload))


def test_positive_receipt_rejects_cross_family_type_even_when_other_claims_match() -> (
    None
):
    payload = _bnb_proof()
    payload["recognized_quantized_runtime_types"] = [
        "gptqmodel.nn_modules.qlinear.marlin.MarlinLinear"
    ]

    errors = _proof_errors(payload)

    assert any("cross-family runtime type" in error for error in errors)


@pytest.mark.parametrize("field", ["load_smoke", "inference_smoke"])
def test_inventory_requires_literal_boolean_smoke_success(field: str) -> None:
    payload = _bnb_inventory()
    payload[field] = 1

    errors = validate_backend_inventory_payload(
        payload=payload,
        expected_adapter="hf_bnb",
        expected_backend="bitsandbytes",
    )

    assert any(f"record {field}: true" in error for error in errors)


def test_inventory_rejects_duplicate_and_unsupported_observation_kinds() -> None:
    duplicate = _bnb_inventory()
    duplicate["quantized_observation_kinds"] = ["module", "module"]
    unsupported = copy.deepcopy(_bnb_inventory())
    unsupported["quantized_observation_kinds"] = ["packed_storage"]

    duplicate_errors = validate_backend_inventory_payload(
        payload=duplicate,
        expected_adapter="hf_bnb",
        expected_backend="bitsandbytes",
    )
    unsupported_errors = validate_backend_inventory_payload(
        payload=unsupported,
        expected_adapter="hf_bnb",
        expected_backend="bitsandbytes",
    )

    assert any("sorted and unique" in error for error in duplicate_errors)
    assert any("non-empty supported list" in error for error in unsupported_errors)


@pytest.mark.parametrize(
    ("adapter", "type_name", "method", "matches"),
    [
        ("hf_bnb", "bitsandbytes.nn.modules.Linear4bit", None, True),
        ("hf_torchao", "torchao.quantization.Int8Tensor", None, True),
        ("hf_hqq", "hqq.core.quantize.HQQLinear", None, True),
        ("hf_quanto", "optimum.quanto.nn.qlinear.QLinear", None, True),
        ("unknown", "vendor.Linear", None, False),
        ("hf_awq", "torch.nn.Linear", "awq", False),
        ("hf_awq", "gptqmodel.nn_modules.qlinear.QLinear", "awq", True),
        ("hf_gptq", "gptqmodel.nn_modules.qlinear.QLinear", "awq", False),
        (
            "hf_awq",
            "gptqmodel.nn_modules.qlinear.marlin_awq.MarlinLinear",
            None,
            True,
        ),
        (
            "hf_gptq",
            "gptqmodel.nn_modules.qlinear.marlin.MarlinLinear",
            None,
            True,
        ),
        (
            "hf_awq",
            "gptqmodel.nn_modules.qlinear.marlin.MarlinLinear",
            None,
            False,
        ),
    ],
)
def test_runtime_type_recognition_is_adapter_and_family_specific(
    adapter: str, type_name: str, method: str | None, matches: bool
) -> None:
    assert (
        runtime_type_name_matches_adapter(
            adapter=adapter,
            type_name=type_name,
            quantization_method=method,
        )
        is matches
    )


@pytest.mark.parametrize(
    ("mutation", "fragment"),
    [
        (("remove", "schema", None), "missing required fields"),
        (("set", "legacy", True), "unsupported v1 fields"),
        (("set", "schema", "runtime-proof-v0"), "schema does not match"),
        (("set", "proof_kind", "metadata"), "kind is not"),
        (("set", "adapter", "hf_hqq"), "adapter does not match"),
        (("set", "backend", "torch"), "backend does not match"),
        (("set", "backend_version", " "), "backend_version must be non-empty"),
        (("set", "ok", 1), "must record ok: true"),
        (("set", "status", "observed"), "status must be"),
        (("set", "reason", "claimed"), "reason must be"),
        (("set", "live_model_observed", 1), "live model observation"),
        (("set", "module_inventory_observed", False), "module inventory"),
        (
            ("set", "packed_storage_artifact_proof_required", True),
            "cannot stand in",
        ),
        (("set", "artifact_binding", "claimed"), "must be not_attempted"),
        (("set", "recognized_quantized_runtime_type_count", True), "must be positive"),
        (("set", "recognized_quantized_runtime_types", []), "must contain"),
        (("set", "recognized_quantized_runtime_types", [" bad "]), "must be strings"),
        (
            (
                "set",
                "recognized_quantized_runtime_types",
                [
                    "bitsandbytes.nn.modules.Linear8bitLt",
                    "bitsandbytes.nn.modules.Linear8bitLt",
                ],
            ),
            "sorted and unique",
        ),
        (
            (
                "set_many",
                "recognized_quantized_runtime_type_count",
                (
                    1,
                    "recognized_quantized_runtime_types",
                    [
                        "bitsandbytes.nn.modules.Linear4bit",
                        "bitsandbytes.nn.modules.Linear8bitLt",
                    ],
                ),
            ),
            "smaller than its runtime type inventory",
        ),
        (
            ("set", "recognized_quantized_runtime_observation_kinds", []),
            "non-empty supported list",
        ),
        (
            (
                "set",
                "recognized_quantized_runtime_observation_kinds",
                ["module", "module"],
            ),
            "sorted and unique",
        ),
        (("set", "live_model_quantization_method", "int8"), "must be awq, gptq"),
    ],
)
def test_runtime_proof_rejects_forged_contract_claims(
    mutation: tuple[str, str, object], fragment: str
) -> None:
    payload = _bnb_proof()
    operation, field, value = mutation
    if operation == "remove":
        payload.pop(field)
    elif operation == "set_many":
        first, second_field, second = value
        payload[field] = first
        payload[second_field] = second
    else:
        payload[field] = value

    assert any(fragment in error for error in _proof_errors(payload))


def _gptq_proof() -> dict[str, object]:
    payload = _bnb_proof()
    payload.update(
        {
            "adapter": "hf_gptq",
            "backend": "gptqmodel",
            "backend_version": "7.1.0",
            "recognized_quantized_runtime_types": [
                "gptqmodel.nn_modules.qlinear.marlin.MarlinLinear"
            ],
            "live_model_quantization_method": "gptq",
            "backend_runtime_importable": True,
            "backend_runtime_version": "7.1.0",
            "backend_runtime_compatibility_bridge_required": False,
            "backend_runtime_compatibility_bridge_applied": False,
        }
    )
    return payload


def _gptq_errors(payload: dict[str, object]) -> list[str]:
    return validate_runtime_quantization_proof_payload(
        payload=payload,
        expected_adapter="hf_gptq",
        expected_backend="gptqmodel",
        expected_schema=RUNTIME_QUANTIZATION_PROOF_SCHEMA,
        expected_proof_kind=RUNTIME_QUANTIZATION_PROOF_KIND,
    )


@pytest.mark.parametrize(
    ("field", "value", "fragment"),
    [
        ("backend_runtime_importable", False, "importable: true"),
        ("backend_runtime_import_error_type", "ImportError", "runtime import error"),
        ("backend_runtime_version", "", "non-empty runtime version"),
        ("backend_runtime_compatibility_bridge_required", 1, "must be boolean"),
        ("backend_runtime_compatibility_bridge_applied", None, "must be boolean"),
        (
            "backend_runtime_compatibility_bridge_error_type",
            "RuntimeError",
            "compatibility bridge error",
        ),
    ],
)
def test_gptq_runtime_proof_rejects_unusable_backend_state(
    field: str, value: object, fragment: str
) -> None:
    payload = _gptq_proof()
    payload[field] = value

    assert any(fragment in error for error in _gptq_errors(payload))


def test_gptq_runtime_proof_requires_an_applied_required_bridge() -> None:
    payload = _gptq_proof()
    payload["backend_runtime_compatibility_bridge_required"] = True
    payload["backend_runtime_compatibility_bridge_applied"] = False

    assert any("required bridge was not applied" in e for e in _gptq_errors(payload))


def test_gptq_runtime_proof_rejects_live_method_from_other_family() -> None:
    payload = _gptq_proof()
    payload["live_model_quantization_method"] = "awq"

    assert any(
        "quantization method does not match" in error for error in _gptq_errors(payload)
    )


@pytest.mark.parametrize(
    ("mutation", "fragment"),
    [
        (("remove", "schema", None), "missing required fields"),
        (("set", "legacy", True), "unsupported v1 fields"),
        (("set", "schema", "inventory-v0"), "schema does not match"),
        (("set", "adapter", "hf_hqq"), "adapter does not match"),
        (("set", "backend", "torch"), "backend does not match"),
        (("set", "backend_version", " "), "backend_version must be non-empty"),
        (("set", "quantized_module_count", True), "must be non-negative"),
        (("set", "quantized_module_types", [" bad "]), "must be a string list"),
        (
            (
                "set",
                "quantized_module_types",
                [
                    "bitsandbytes.nn.modules.Linear8bitLt",
                    "bitsandbytes.nn.modules.Linear8bitLt",
                ],
            ),
            "sorted and unique",
        ),
        (
            (
                "set_many",
                "quantized_module_count",
                (
                    1,
                    "quantized_module_types",
                    [
                        "bitsandbytes.nn.modules.Linear4bit",
                        "bitsandbytes.nn.modules.Linear8bitLt",
                    ],
                ),
            ),
            "smaller than its type inventory",
        ),
    ],
)
def test_backend_inventory_rejects_unbound_or_malformed_claims(
    mutation: tuple[str, str, object], fragment: str
) -> None:
    payload = _bnb_inventory()
    operation, field, value = mutation
    if operation == "remove":
        payload.pop(field)
    elif operation == "set_many":
        first, second_field, second = value
        payload[field] = first
        payload[second_field] = second
    else:
        payload[field] = value

    errors = validate_backend_inventory_payload(
        payload=payload,
        expected_adapter="hf_bnb",
        expected_backend="bitsandbytes",
    )
    assert any(fragment in error for error in errors)
