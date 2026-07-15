from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from tests.scripts._support_source_matrix_artifact_validator import (
    _acceptance_inputs,
    _load_validator,
    _report_dir,
    _write_matrix_artifact_set,
    _write_test_source_matrix,
    valid_hqq_backend_inventory,
    valid_hqq_runtime_quantization_proof,
)


def test_source_matrix_artifact_validator_accepts_complete_artifacts(
    tmp_path: Path,
) -> None:
    validator = _load_validator()
    matrix_path = _write_test_source_matrix(tmp_path)
    baseline_path, policy_path = _write_matrix_artifact_set(_report_dir(tmp_path))

    selected, issues = validator.validate_matrix(
        repo_root=tmp_path,
        matrix_path=matrix_path,
        targets={"hqq"},
        acceptance_inputs=_acceptance_inputs(validator, baseline_path, policy_path),
    )

    assert selected == ["hqq"]
    assert issues == []


def test_source_matrix_repaired_v1_rejects_retired_v2_shape(tmp_path: Path) -> None:
    validator = _load_validator()
    matrix_path = _write_test_source_matrix(tmp_path)
    payload = json.loads(matrix_path.read_text(encoding="utf-8"))
    payload["schema"] = "invarlock.integration_source_matrix.v2"
    matrix_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported source matrix schema"):
        validator.validate_matrix(
            repo_root=tmp_path,
            matrix_path=matrix_path,
            targets={"hqq"},
        )


def test_source_matrix_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    validator = _load_validator()
    matrix_path = _write_test_source_matrix(tmp_path)
    text = matrix_path.read_text(encoding="utf-8")
    matrix_path.write_text(
        text.replace(
            '"schema": "invarlock.integration_source_matrix.v1"',
            '"schema": "invarlock.integration_source_matrix.v1", '
            '"schema": "invarlock.integration_source_matrix.v1"',
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate JSON key 'schema'"):
        validator.validate_matrix(
            repo_root=tmp_path,
            matrix_path=matrix_path,
            targets={"hqq"},
        )


def test_source_matrix_requires_regular_snapshot(tmp_path: Path) -> None:
    validator = _load_validator()
    matrix_path = _write_test_source_matrix(tmp_path)
    symlink_path = tmp_path / "source-matrix-link.json"
    symlink_path.symlink_to(matrix_path)

    with pytest.raises(ValueError, match="readable regular file"):
        validator.validate_matrix(
            repo_root=tmp_path,
            matrix_path=symlink_path,
            targets={"hqq"},
        )


def test_regular_snapshot_rejects_mutation_during_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    validator = _load_validator()
    path = tmp_path / "authority.json"
    path.write_text("{}\n", encoding="utf-8")
    original_fstat = validator.os.fstat
    calls = 0

    def changing_fstat(descriptor: int):
        nonlocal calls
        value = original_fstat(descriptor)
        calls += 1
        if calls == 2:

            class Changed:
                st_mode = value.st_mode
                st_dev = value.st_dev
                st_ino = value.st_ino
                st_size = value.st_size
                st_mtime_ns = value.st_mtime_ns + 1
                st_ctime_ns = value.st_ctime_ns

            return Changed()
        return value

    monkeypatch.setattr(validator.os, "fstat", changing_fstat)
    with pytest.raises(ValueError, match="changed while it was being read"):
        validator._read_regular_snapshot(path, label="authority")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload: payload.update({"extra": True}), "source matrix fields"),
        (lambda payload: payload.update({"entries": []}), "nonempty list"),
        (
            lambda payload: payload["entries"][0].update({"extra": True}),
            "entry 0 fields",
        ),
        (
            lambda payload: payload["entries"][0]["runtime_image"].update(
                {"extra": True}
            ),
            "runtime_image fields",
        ),
        (
            lambda payload: payload["entries"][0]["expected"].update({"extra": True}),
            "expected fields",
        ),
    ],
)
def test_source_matrix_rejects_noncanonical_shapes(
    tmp_path: Path,
    mutation,
    message: str,
) -> None:
    validator = _load_validator()
    matrix_path = _write_test_source_matrix(tmp_path)
    payload = json.loads(matrix_path.read_text(encoding="utf-8"))
    mutation(payload)
    matrix_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        validator.validate_matrix(
            repo_root=tmp_path,
            matrix_path=matrix_path,
            targets={"hqq"},
        )


def test_source_matrix_rejects_nonmapping_and_duplicate_entries(tmp_path: Path) -> None:
    validator = _load_validator()
    matrix_path = _write_test_source_matrix(tmp_path)
    payload = json.loads(matrix_path.read_text(encoding="utf-8"))
    payload["entries"].append("hqq")
    matrix_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="entry 1 must be an object"):
        validator.validate_matrix(
            repo_root=tmp_path,
            matrix_path=matrix_path,
            targets={"hqq"},
        )

    payload["entries"][1] = copy.deepcopy(payload["entries"][0])
    matrix_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="target is duplicated"):
        validator.validate_matrix(
            repo_root=tmp_path,
            matrix_path=matrix_path,
            targets={"hqq"},
        )


def test_source_matrix_requires_full_target_coverage_for_full_validation(
    tmp_path: Path,
) -> None:
    validator = _load_validator()
    matrix_path = _write_test_source_matrix(tmp_path)

    with pytest.raises(ValueError, match="target coverage must be exact"):
        validator.validate_matrix(repo_root=tmp_path, matrix_path=matrix_path)


def test_source_matrix_artifact_validator_rejects_fabricated_green_json(
    tmp_path: Path,
) -> None:
    validator = _load_validator()
    matrix_path = _write_test_source_matrix(tmp_path)
    report_dir = _report_dir(tmp_path)
    baseline_path, policy_path = _write_matrix_artifact_set(report_dir)
    (report_dir / "evaluation.report.json").write_text("{}\n", encoding="utf-8")
    (report_dir / "verify.json").write_text(
        json.dumps(
            {
                "summary": {"ok": True, "reason": "ok"},
                "results": [
                    {
                        "ok": True,
                        "reason": "ok",
                        "verification": {
                            "runtime_provenance": {
                                "declared_mode": "container",
                                "verified": True,
                                "status": "expected_image_digest_matched",
                                "expected_digest_matched": True,
                            }
                        },
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    _, issues = validator.validate_matrix(
        repo_root=tmp_path,
        matrix_path=matrix_path,
        targets={"hqq"},
        acceptance_inputs=_acceptance_inputs(validator, baseline_path, policy_path),
    )
    messages = [issue.message for issue in issues]

    assert "verify artifact is missing its cryptographic input receipt" in messages
    assert any("canonical strict verifier replay failed" in msg for msg in messages)


def test_source_matrix_artifact_validator_reports_artifact_and_status_mismatches(
    tmp_path: Path,
) -> None:
    validator = _load_validator()
    matrix_path = _write_test_source_matrix(tmp_path)
    report_dir = _report_dir(tmp_path)
    baseline_path, policy_path = _write_matrix_artifact_set(report_dir)
    (report_dir / "backend_inventory.json").unlink()
    (report_dir / "verify.json").write_text(
        json.dumps(
            {
                "summary": {"ok": False, "reason": "policy_fail"},
                "results": [
                    {
                        "verification": {
                            "runtime_provenance": {
                                "declared_mode": "host",
                                "verified": False,
                            }
                        }
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    _, issues = validator.validate_matrix(
        repo_root=tmp_path,
        matrix_path=matrix_path,
        targets={"hqq"},
        acceptance_inputs=_acceptance_inputs(validator, baseline_path, policy_path),
    )
    messages = [issue.message for issue in issues]

    assert "required artifact is missing" in messages
    assert any("verify status mismatch" in message for message in messages)
    assert any(
        "runtime provenance declared mode mismatch" in message for message in messages
    )
    assert any(
        "runtime provenance verified flag mismatch" in message for message in messages
    )
    assert any("runtime provenance status mismatch" in message for message in messages)
    assert any(
        "runtime expected-digest match flag mismatch" in message for message in messages
    )


def test_source_matrix_artifact_validator_checks_inventory_and_runtime_manifest(
    tmp_path: Path,
) -> None:
    validator = _load_validator()
    matrix_path = _write_test_source_matrix(tmp_path)
    report_dir = _report_dir(tmp_path)
    baseline_path, policy_path = _write_matrix_artifact_set(report_dir)
    (report_dir / "backend_inventory.json").write_text(
        json.dumps(
            {
                "schema": "other",
                "adapter": "hf_other",
                "backend": "not-hqq",
                "quantized_module_count": -1,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (report_dir / "runtime.manifest.json").write_text(
        json.dumps({"runtime": {"image_digest": "", "image_ref": ""}}) + "\n",
        encoding="utf-8",
    )

    _, issues = validator.validate_matrix(
        repo_root=tmp_path,
        matrix_path=matrix_path,
        targets={"hqq"},
        acceptance_inputs=_acceptance_inputs(validator, baseline_path, policy_path),
    )
    messages = [issue.message for issue in issues]

    assert any("backend inventory schema mismatch" in message for message in messages)
    assert any("backend inventory adapter mismatch" in message for message in messages)
    assert any("backend inventory backend mismatch" in message for message in messages)
    assert any("quantized_module_count" in message for message in messages)
    assert "runtime manifest runtime.image_digest must be present" in messages
    assert "runtime manifest runtime.image_ref must be present" in messages


def test_source_matrix_rejects_extra_runtime_manifest_v1_fields(
    tmp_path: Path,
) -> None:
    validator = _load_validator()
    matrix_path = _write_test_source_matrix(tmp_path)
    report_dir = _report_dir(tmp_path)
    _write_matrix_artifact_set(report_dir)
    manifest_path = report_dir / "runtime.manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["claimed_green"] = True
    payload["runtime"]["claimed_container"] = True
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    entry = json.loads(matrix_path.read_text(encoding="utf-8"))["entries"][0]

    issues = validator.validate_entry(tmp_path, entry, acceptance_inputs=None)
    messages = [issue.message for issue in issues]

    assert any(
        "runtime manifest root fields do not match v1" in message
        for message in messages
    )
    assert any(
        "runtime manifest runtime fields do not match v1" in message
        for message in messages
    )


def test_runtime_quantization_proof_rejects_fake_green_and_cross_bound_data() -> None:
    validator = _load_validator()
    proof = valid_hqq_runtime_quantization_proof()
    inventory = valid_hqq_backend_inventory()
    path = Path("runtime_quantization_proof.json")

    assert (
        validator._validate_runtime_quantization_proof(
            target="hqq",
            path=path,
            payload=proof,
            expected_adapter="hf_hqq",
            expected_backend="hqq",
            backend_inventory=inventory,
        )
        == []
    )

    forged = dict(proof)
    forged.update(
        {
            "ok": False,
            "recognized_quantized_runtime_type_count": 0,
            "recognized_quantized_runtime_types": ["torch.nn.modules.linear.Linear"],
            "artifact_binding": "claimed",
        }
    )
    messages = [
        issue.message
        for issue in validator._validate_runtime_quantization_proof(
            target="hqq",
            path=path,
            payload=forged,
            expected_adapter="hf_hqq",
            expected_backend="hqq",
            backend_inventory=inventory,
        )
    ]
    assert "runtime quantization proof does not record ok: true" in messages
    assert any("must be a positive integer" in message for message in messages)
    assert any(
        "unrecognized or cross-family runtime type" in message for message in messages
    )
    assert any(
        "artifact_binding must be not_attempted" in message for message in messages
    )

    copied_adapter = dict(proof)
    copied_adapter["adapter"] = "hf_bnb"
    copied_adapter["backend"] = "bitsandbytes"
    copied_messages = [
        issue.message
        for issue in validator._validate_runtime_quantization_proof(
            target="hqq",
            path=path,
            payload=copied_adapter,
            expected_adapter="hf_hqq",
            expected_backend="hqq",
            backend_inventory=inventory,
        )
    ]
    assert any("proof adapter mismatch" in message for message in copied_messages)
    assert any("proof backend mismatch" in message for message in copied_messages)
    assert any(
        "does not match backend inventory" in message for message in copied_messages
    )


def test_runtime_quantization_proof_accepts_each_strict_adapter_family() -> None:
    validator = _load_validator()
    cases = [
        (
            "hf_bnb",
            "bitsandbytes",
            "bitsandbytes.nn.modules.Linear8bitLt",
            None,
        ),
        (
            "hf_awq",
            "gptqmodel",
            "gptqmodel.nn_modules.qlinear.machete_awq.AwqMacheteLinear",
            "awq",
        ),
        (
            "hf_gptq",
            "gptqmodel",
            "gptqmodel.nn_modules.qlinear.machete.MacheteLinear",
            "gptq",
        ),
        (
            "hf_torchao",
            "torchao",
            "torchao.quantization.Int8Tensor",
            None,
        ),
        ("hf_hqq", "hqq", "hqq.core.quantize.HQQLinear", None),
        (
            "hf_quanto",
            "optimum-quanto",
            "optimum.quanto.nn.qlinear.QLinear",
            None,
        ),
    ]
    for adapter, backend, type_name, method in cases:
        proof = valid_hqq_runtime_quantization_proof()
        proof.update(
            {
                "adapter": adapter,
                "backend": backend,
                "recognized_quantized_runtime_types": [type_name],
                "recognized_quantized_runtime_observation_kinds": [
                    "direct_weight" if adapter == "hf_torchao" else "module"
                ],
                "live_model_quantization_method": method,
            }
        )
        if adapter in {"hf_awq", "hf_gptq"}:
            proof.update(
                {
                    "backend_runtime_importable": True,
                    "backend_runtime_import_error_type": None,
                    "backend_runtime_version": "7.0.0",
                    "backend_runtime_compatibility_bridge_required": False,
                    "backend_runtime_compatibility_bridge_applied": False,
                    "backend_runtime_compatibility_bridge_error_type": None,
                }
            )
        inventory = valid_hqq_backend_inventory()
        inventory.update(
            {
                "adapter": adapter,
                "backend": backend,
                "quantized_module_types": [type_name],
                "quantized_observation_kinds": [
                    "direct_weight" if adapter == "hf_torchao" else "module"
                ],
            }
        )
        assert (
            validator._validate_runtime_quantization_proof(
                target=adapter,
                path=Path("runtime_quantization_proof.json"),
                payload=proof,
                expected_adapter=adapter,
                expected_backend=backend,
                backend_inventory=inventory,
            )
            == []
        )


def test_runtime_quantization_proof_rejects_torchao_inventory_observation_mismatch() -> (
    None
):
    validator = _load_validator()
    proof = valid_hqq_runtime_quantization_proof()
    proof.update(
        {
            "adapter": "hf_torchao",
            "backend": "torchao",
            "recognized_quantized_runtime_types": ["torchao.quantization.Int8Tensor"],
            "recognized_quantized_runtime_observation_kinds": ["direct_weight"],
        }
    )
    inventory = valid_hqq_backend_inventory()
    inventory.update(
        {
            "adapter": "hf_torchao",
            "backend": "torchao",
            "quantized_module_count": 0,
            "quantized_module_types": [],
            "quantized_observation_kinds": ["module"],
        }
    )
    messages = [
        issue.message
        for issue in validator._validate_runtime_quantization_proof(
            target="hf_torchao",
            path=Path("runtime_quantization_proof.json"),
            payload=proof,
            expected_adapter="hf_torchao",
            expected_backend="torchao",
            backend_inventory=inventory,
        )
    ]
    assert any("observation count does not match" in message for message in messages)
    assert any("runtime types do not exactly match" in message for message in messages)
    assert any("observation kinds do not match" in message for message in messages)


def test_repaired_v1_rejects_old_inventory_and_proof_shapes() -> None:
    validator = _load_validator()
    inventory = valid_hqq_backend_inventory()
    inventory.pop("quantized_observation_kinds")
    inventory_issues = validator._validate_backend_inventory(
        target="hqq",
        path=Path("backend_inventory.json"),
        payload=inventory,
        expected_adapter="hf_hqq",
        expected_backend="hqq",
    )
    assert any(
        "missing required fields: quantized_observation_kinds" in issue.message
        for issue in inventory_issues
    )

    proof = valid_hqq_runtime_quantization_proof()
    proof.pop("recognized_quantized_runtime_observation_kinds")
    proof_issues = validator._validate_runtime_quantization_proof(
        target="hqq",
        path=Path("runtime_quantization_proof.json"),
        payload=proof,
        expected_adapter="hf_hqq",
        expected_backend="hqq",
        backend_inventory=valid_hqq_backend_inventory(),
    )
    assert any(
        "missing required fields: recognized_quantized_runtime_observation_kinds"
        in issue.message
        for issue in proof_issues
    )
