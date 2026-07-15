from __future__ import annotations

import json
from pathlib import Path

import pytest

from invarlock.core.runtime_quantization_proof import (
    RUNTIME_QUANTIZATION_PROOF_KIND,
    RUNTIME_QUANTIZATION_PROOF_SCHEMA,
    main,
    validate_runtime_quantization_proof_sidecars,
)


def _valid_proof() -> dict[str, object]:
    return {
        "schema": RUNTIME_QUANTIZATION_PROOF_SCHEMA,
        "proof_kind": RUNTIME_QUANTIZATION_PROOF_KIND,
        "adapter": "hf_hqq",
        "backend": "hqq",
        "backend_version": "3.0.0",
        "ok": True,
        "status": "verified_live_runtime_types",
        "reason": "recognized_live_quantized_runtime_types",
        "live_model_observed": True,
        "module_inventory_observed": True,
        "recognized_quantized_runtime_type_count": 2,
        "recognized_quantized_runtime_types": ["hqq.core.quantize.HQQLinear"],
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


def _valid_inventory() -> dict[str, object]:
    return {
        "schema": "invarlock/backend-inventory-v1",
        "adapter": "hf_hqq",
        "backend": "hqq",
        "backend_version": "3.0.0",
        "transformers_version": "5.12.0",
        "quantization_config": {},
        "quantized_module_count": 2,
        "quantized_module_types": ["hqq.core.quantize.HQQLinear"],
        "quantized_observation_kinds": ["module"],
        "device_map": "cuda",
        "memory_footprint": {"reported_bytes": 1, "method": "test"},
        "load_smoke": True,
        "inference_smoke": True,
    }


def _write_sidecars(
    tmp_path: Path,
    proof: object,
    inventory: object,
) -> tuple[Path, Path]:
    proof_path = tmp_path / "runtime_quantization_proof.json"
    inventory_path = tmp_path / "backend_inventory.json"
    proof_path.write_text(json.dumps(proof, sort_keys=True) + "\n", encoding="utf-8")
    inventory_path.write_text(
        json.dumps(inventory, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return proof_path, inventory_path


def test_runtime_quantization_sidecar_validator_accepts_complete_cross_bound_v1(
    tmp_path: Path,
) -> None:
    proof_path, inventory_path = _write_sidecars(
        tmp_path,
        _valid_proof(),
        _valid_inventory(),
    )

    assert (
        validate_runtime_quantization_proof_sidecars(
            proof_path=proof_path,
            expected_adapter="hf_hqq",
            backend_inventory_path=inventory_path,
        )
        == []
    )


def test_runtime_quantization_sidecar_validator_rejects_fake_green_payload(
    tmp_path: Path,
) -> None:
    proof_path, inventory_path = _write_sidecars(
        tmp_path,
        {"ok": True},
        _valid_inventory(),
    )

    errors = validate_runtime_quantization_proof_sidecars(
        proof_path=proof_path,
        expected_adapter="hf_hqq",
        backend_inventory_path=inventory_path,
    )

    assert any("missing required fields" in error for error in errors)
    assert any("schema does not match v1" in error for error in errors)
    assert any(
        "must contain recognized runtime type names" in error for error in errors
    )


def test_runtime_quantization_sidecar_validator_rejects_integer_boolean_claims(
    tmp_path: Path,
) -> None:
    proof = _valid_proof()
    proof["ok"] = 1
    proof["live_model_observed"] = 1
    proof["module_inventory_observed"] = 1
    proof["packed_storage_artifact_proof_required"] = 0
    proof_path, inventory_path = _write_sidecars(
        tmp_path,
        proof,
        _valid_inventory(),
    )

    errors = validate_runtime_quantization_proof_sidecars(
        proof_path=proof_path,
        expected_adapter="hf_hqq",
        backend_inventory_path=inventory_path,
    )

    assert "runtime quantization proof must record ok: true" in errors
    assert any("live model observation" in error for error in errors)
    assert any("module inventory observation" in error for error in errors)
    assert any(
        "cannot stand in for packed-storage evidence" in error for error in errors
    )


def test_runtime_quantization_sidecar_validator_rejects_structured_observation_kinds(
    tmp_path: Path,
) -> None:
    proof = _valid_proof()
    proof["recognized_quantized_runtime_observation_kinds"] = [{"kind": "module"}]
    inventory = _valid_inventory()
    inventory["quantized_observation_kinds"] = [{"kind": "module"}]
    proof_path, inventory_path = _write_sidecars(tmp_path, proof, inventory)

    errors = validate_runtime_quantization_proof_sidecars(
        proof_path=proof_path,
        expected_adapter="hf_hqq",
        backend_inventory_path=inventory_path,
    )

    assert any(
        "runtime quantization proof observation kinds" in error for error in errors
    )
    assert any("backend inventory observation kinds" in error for error in errors)


def test_runtime_quantization_sidecar_validator_rejects_cross_bound_inventory(
    tmp_path: Path,
) -> None:
    inventory = _valid_inventory()
    inventory["adapter"] = "hf_bnb"
    inventory["backend"] = "bitsandbytes"
    inventory["backend_version"] = "0.49.0"
    inventory["quantized_module_types"] = ["bitsandbytes.nn.modules.Linear8bitLt"]
    proof_path, inventory_path = _write_sidecars(tmp_path, _valid_proof(), inventory)

    errors = validate_runtime_quantization_proof_sidecars(
        proof_path=proof_path,
        expected_adapter="hf_hqq",
        backend_inventory_path=inventory_path,
    )

    assert any("backend inventory adapter does not match" in error for error in errors)
    assert any("backend inventory backend does not match" in error for error in errors)
    assert any(
        "proof adapter does not match backend inventory" in error for error in errors
    )
    assert any(
        "proof backend does not match backend inventory" in error for error in errors
    )
    assert any(
        "backend_version does not match backend inventory" in error for error in errors
    )


def test_runtime_quantization_sidecar_validator_rejects_nonoverlapping_runtime_inventory(
    tmp_path: Path,
) -> None:
    inventory = _valid_inventory()
    inventory["quantized_module_types"] = ["hqq.core.quantize.OtherHQQLinear"]
    proof_path, inventory_path = _write_sidecars(tmp_path, _valid_proof(), inventory)

    errors = validate_runtime_quantization_proof_sidecars(
        proof_path=proof_path,
        expected_adapter="hf_hqq",
        backend_inventory_path=inventory_path,
    )

    assert any(
        "runtime types do not exactly match backend inventory" in error
        for error in errors
    )


def test_runtime_quantization_sidecar_validator_rejects_torchao_weight_inventory_mismatch(
    tmp_path: Path,
) -> None:
    proof = _valid_proof()
    proof.update(
        {
            "adapter": "hf_torchao",
            "backend": "torchao",
            "recognized_quantized_runtime_types": ["torchao.quantization.Int8Tensor"],
            "recognized_quantized_runtime_observation_kinds": ["direct_weight"],
        }
    )
    inventory = _valid_inventory()
    inventory.update(
        {
            "adapter": "hf_torchao",
            "backend": "torchao",
            "quantized_module_count": 0,
            "quantized_module_types": [],
            "quantized_observation_kinds": [],
        }
    )
    proof_path, inventory_path = _write_sidecars(tmp_path, proof, inventory)

    errors = validate_runtime_quantization_proof_sidecars(
        proof_path=proof_path,
        expected_adapter="hf_torchao",
        backend_inventory_path=inventory_path,
    )
    assert any("observation count does not match" in error for error in errors)
    assert any("runtime types do not exactly match" in error for error in errors)


def test_runtime_quantization_sidecar_validator_accepts_cross_bound_torchao_weight(
    tmp_path: Path,
) -> None:
    proof = _valid_proof()
    proof.update(
        {
            "adapter": "hf_torchao",
            "backend": "torchao",
            "recognized_quantized_runtime_type_count": 1,
            "recognized_quantized_runtime_types": ["torchao.quantization.Int8Tensor"],
            "recognized_quantized_runtime_observation_kinds": ["direct_weight"],
        }
    )
    inventory = _valid_inventory()
    inventory.update(
        {
            "adapter": "hf_torchao",
            "backend": "torchao",
            "quantized_module_count": 1,
            "quantized_module_types": ["torchao.quantization.Int8Tensor"],
            "quantized_observation_kinds": ["direct_weight"],
        }
    )
    proof_path, inventory_path = _write_sidecars(tmp_path, proof, inventory)
    assert (
        validate_runtime_quantization_proof_sidecars(
            proof_path=proof_path,
            expected_adapter="hf_torchao",
            backend_inventory_path=inventory_path,
        )
        == []
    )


def test_runtime_quantization_sidecar_validator_rejects_duplicate_proof_keys(
    tmp_path: Path,
) -> None:
    proof_path, inventory_path = _write_sidecars(
        tmp_path,
        _valid_proof(),
        _valid_inventory(),
    )
    proof_path.write_text(
        proof_path.read_text(encoding="utf-8").replace(
            '"ok": true',
            '"ok": false, "ok": true',
        ),
        encoding="utf-8",
    )

    errors = validate_runtime_quantization_proof_sidecars(
        proof_path=proof_path,
        expected_adapter="hf_hqq",
        backend_inventory_path=inventory_path,
    )

    assert any("duplicate JSON key 'ok'" in error for error in errors)


@pytest.mark.parametrize("payload", ["[1, 2]", '{"value": NaN}'])
def test_runtime_quantization_sidecar_validator_rejects_nonobject_or_nonfinite_json(
    tmp_path: Path, payload: str
) -> None:
    proof_path, inventory_path = _write_sidecars(
        tmp_path,
        _valid_proof(),
        _valid_inventory(),
    )
    proof_path.write_text(payload, encoding="utf-8")

    errors = validate_runtime_quantization_proof_sidecars(
        proof_path=proof_path,
        expected_adapter="hf_hqq",
        backend_inventory_path=inventory_path,
    )

    assert errors


def test_runtime_quantization_sidecar_validator_cli_accepts_valid_pair(
    tmp_path: Path,
) -> None:
    proof_path, inventory_path = _write_sidecars(
        tmp_path,
        _valid_proof(),
        _valid_inventory(),
    )

    assert (
        main(
            [
                "validate-sidecars",
                "--proof",
                str(proof_path),
                "--backend-inventory",
                str(inventory_path),
                "--adapter",
                "hf_hqq",
            ]
        )
        == 0
    )


def test_runtime_quantization_sidecar_validator_cli_rejects_unsupported_adapter(
    tmp_path: Path,
    capsys,
) -> None:
    proof_path, inventory_path = _write_sidecars(
        tmp_path,
        _valid_proof(),
        _valid_inventory(),
    )

    assert (
        main(
            [
                "validate-sidecars",
                "--proof",
                str(proof_path),
                "--backend-inventory",
                str(inventory_path),
                "--adapter",
                "auto",
            ]
        )
        == 1
    )
    assert "explicit supported module-backed quantized subject adapter" in (
        capsys.readouterr().err
    )
