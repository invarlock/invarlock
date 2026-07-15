from __future__ import annotations

import json
from pathlib import Path

import pytest

from invarlock.core.checkpoint_identity import checkpoint_tree_sha256
from invarlock.evidence_pack_contracts.deployable_coverage import (
    canonical_names_sha256,
)
from scripts.evidence_packs.python import (
    create_edits_batch as batch_edit_mod,
)
from scripts.evidence_packs.python.editing import (
    validate_artifact as edit_artifact_mod,
)
from scripts.evidence_packs.python.editing import validate_deployable as deployable_impl
from scripts.evidence_packs.python.editing.implementations import (
    DEPLOYABLE_OPTIMIZED_SUBJECT,
    EDIT_SEMANTICS_DEPLOYABLE,
    build_edit_metadata,
)

save_artifact_mod = edit_artifact_mod
deployable_validator_mod = edit_artifact_mod


def _logical_coverage() -> dict[str, object]:
    names = ["layer.weight"]
    return {
        "basis": "dense_baseline_unique_parameters",
        "weight_tensor_names": names,
        "weight_tensor_names_sha256": canonical_names_sha256(names),
        "weight_tensor_count": 1,
        "parameter_elements": 1,
        "total_unique_parameter_elements": 1,
    }


def _packed_facts() -> dict[str, object]:
    names = ["layer"]
    return {
        "quantized_module_count": 1,
        "quantized_module_names": names,
        "quantized_module_names_sha256": canonical_names_sha256(names),
        "quantized_module_types": ["bitsandbytes.nn.Linear8bitLt"],
        "packed_weight_storage_elements": 1,
        "logical_coverage": _logical_coverage(),
    }


def _write_minimal_artifact(path: Path, metadata: dict[str, object] | None) -> None:
    path.mkdir(parents=True)
    (path / "config.json").write_text("{}", encoding="utf-8")
    (path / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (path / "pytorch_model.bin").write_text("weights", encoding="utf-8")
    if metadata is not None:
        (path / "edit_metadata.json").write_text(
            json.dumps(metadata),
            encoding="utf-8",
        )


def _deployable_metadata() -> dict[str, object]:
    return build_edit_metadata(
        edit_type="bnb_8bit",
        scope="ffn",
        artifact_class=DEPLOYABLE_OPTIMIZED_SUBJECT,
        edit_semantics=EDIT_SEMANTICS_DEPLOYABLE,
        optimized_deployment_backend=True,
        backend="bitsandbytes",
        storage_format="bitsandbytes_8bit_packed",
        actual_storage_format="bitsandbytes_8bit_packed",
        packed_quantized_storage=True,
        runtime_memory_reduction=True,
        runtime_memory_reduction_expected=True,
        parameters={"bits": 8},
        coverage={"edited_tensors": 1, "edited_params": 1, "total_params": 1},
        extra={"logical_coverage": _logical_coverage()},
    )


def _write_deployable_sidecars(
    report_dir: Path, artifact: Path, baseline: Path | None = None
) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    binding = {
        "artifact_identity": {
            "kind": "local_checkpoint_tree",
            "sha256": checkpoint_tree_sha256(artifact),
        },
        "baseline_identity": {
            "kind": "local_checkpoint_tree",
            "sha256": (
                checkpoint_tree_sha256(baseline)
                if baseline is not None
                else "sha256:" + "b" * 64
            ),
        },
        "bits": 8,
        "trust_remote_code": False,
    }
    (report_dir / "backend_inventory.json").write_text(
        json.dumps(
            {
                "schema": "invarlock/backend-inventory-v1",
                "adapter": "hf_bnb",
                "backend": "bitsandbytes",
                "backend_version": "0.1",
                "transformers_version": "1.0",
                "quantization_config": {
                    "quant_method": "bitsandbytes",
                    "load_in_8bit": True,
                    "load_in_4bit": False,
                },
                **_packed_facts(),
                "device_map": "cuda:0",
                "memory_footprint": {
                    "reported_bytes": 1024,
                    "method": "get_memory_footprint",
                },
                "load_smoke": True,
                "inference_smoke": True,
                **binding,
            }
        ),
        encoding="utf-8",
    )
    (report_dir / "memory_report.json").write_text(
        json.dumps(
            {
                "schema": "invarlock/deployable-memory-report-v1",
                "ok": True,
                "runtime_memory_reduction_observed": True,
                "baseline_reported_bytes": 2048,
                "quantized_reported_bytes": 1024,
                "reduction_bytes": 1024,
                "reduction_ratio": 0.5,
                **binding,
            }
        ),
        encoding="utf-8",
    )
    (report_dir / "load_smoke.json").write_text(
        json.dumps(
            {
                "schema": "invarlock/deployable-load-smoke-v1",
                "ok": True,
                "loaded_from_saved_checkpoint": True,
                "load_time_quantization_override": False,
                **_packed_facts(),
                **binding,
            }
        ),
        encoding="utf-8",
    )
    (report_dir / "inference_smoke.json").write_text(
        json.dumps(
            {
                "schema": "invarlock/deployable-inference-smoke-v1",
                "ok": True,
                "all_logits_finite": True,
                "logits_sha256": "sha256:" + "a" * 64,
                "logits_shape": [1, 2, 3],
                **binding,
            }
        ),
        encoding="utf-8",
    )


def test_validate_deployable_artifact_checks_sidecar_schemas_and_ok(
    monkeypatch, tmp_path: Path
) -> None:
    artifact = tmp_path / "deployable"
    baseline = tmp_path / "baseline"
    report_dir = tmp_path / "report"
    _write_minimal_artifact(artifact, _deployable_metadata())
    _write_minimal_artifact(baseline, None)
    _write_deployable_sidecars(report_dir, artifact, baseline)
    monkeypatch.setattr(
        deployable_impl,
        "_package_version",
        lambda _package_name: "0.1",
        raising=True,
    )
    monkeypatch.setattr(
        deployable_validator_mod,
        "_runtime_bitsandbytes_proof",
        lambda artifact_dir, **_kwargs: {
            "artifact_identity": {
                "kind": "local_checkpoint_tree",
                "sha256": checkpoint_tree_sha256(artifact_dir),
            },
            "baseline_identity": {
                "kind": "local_checkpoint_tree",
                "sha256": checkpoint_tree_sha256(baseline),
            },
            "trust_remote_code": False,
            **_packed_facts(),
            "logits_sha256": "sha256:" + "a" * 64,
            "logits_shape": [1, 2, 3],
            "all_logits_finite": True,
            "load_time_quantization_override": False,
            "baseline_reported_bytes": 2048,
            "quantized_reported_bytes": 1024,
            "reduction_bytes": 1024,
            "reduction_ratio": 0.5,
            "runtime_memory_reduction_observed": True,
        },
        raising=True,
    )

    payload = deployable_validator_mod.validate_deployable_artifact(
        artifact,
        backend="bitsandbytes",
        report_dir=report_dir,
        smoke=True,
        baseline_dir=baseline,
    )

    assert payload["ok"] is True
    assert payload["validation_scope"] == "runtime_reproof"
    assert payload["runtime_proof_authoritative"] is True
    assert payload["load_smoke"] is True
    assert payload["inference_smoke"] is True
    assert payload["runtime_memory_reduction_observed"] is True

    structural_payload = deployable_validator_mod.validate_deployable_artifact(
        artifact,
        backend="bitsandbytes",
        report_dir=report_dir,
        smoke=False,
        baseline_dir=baseline,
    )
    assert structural_payload["ok"] is True
    assert structural_payload["validation_scope"] == "structural_only"
    assert structural_payload["runtime_proof_authoritative"] is False

    (report_dir / "load_smoke.json").write_text(
        json.dumps({"schema": "invarlock/deployable-load-smoke-v1", "ok": False}),
        encoding="utf-8",
    )
    payload = deployable_validator_mod.validate_deployable_artifact(
        artifact,
        backend="bitsandbytes",
        report_dir=report_dir,
        smoke=True,
        baseline_dir=baseline,
    )
    assert payload["ok"] is False
    assert payload["load_smoke"] is False
    assert "load_smoke.json ok must be true" in payload["issues"]

    _write_deployable_sidecars(report_dir, artifact, baseline)
    (report_dir / "backend_inventory.json").write_text(
        json.dumps(
            {
                "schema": "wrong",
                "backend": "bitsandbytes",
                "load_smoke": True,
                "inference_smoke": True,
                "quantized_module_count": 1,
                "quantized_module_types": [],
                "memory_footprint": {},
            }
        ),
        encoding="utf-8",
    )
    payload = deployable_validator_mod.validate_deployable_artifact(
        artifact,
        backend="bitsandbytes",
        report_dir=report_dir,
        smoke=True,
        baseline_dir=baseline,
    )
    assert payload["ok"] is False
    assert any(
        issue.startswith("backend_inventory.json schema mismatch")
        for issue in payload["issues"]
    )

    _write_deployable_sidecars(report_dir, artifact, baseline)
    backend_inventory = json.loads(
        (report_dir / "backend_inventory.json").read_text(encoding="utf-8")
    )
    backend_inventory["backend"] = "other_backend"
    (report_dir / "backend_inventory.json").write_text(
        json.dumps(backend_inventory),
        encoding="utf-8",
    )
    payload = deployable_validator_mod.validate_deployable_artifact(
        artifact,
        backend="bitsandbytes",
        report_dir=report_dir,
        smoke=True,
        baseline_dir=baseline,
    )
    assert payload["ok"] is False
    assert any(
        issue.startswith("backend_inventory.json backend mismatch")
        for issue in payload["issues"]
    )

    _write_deployable_sidecars(report_dir, artifact, baseline)
    backend_inventory = json.loads(
        (report_dir / "backend_inventory.json").read_text(encoding="utf-8")
    )
    backend_inventory["load_smoke"] = False
    (report_dir / "backend_inventory.json").write_text(
        json.dumps(backend_inventory),
        encoding="utf-8",
    )
    payload = deployable_validator_mod.validate_deployable_artifact(
        artifact,
        backend="bitsandbytes",
        report_dir=report_dir,
        smoke=True,
        baseline_dir=baseline,
    )
    assert payload["ok"] is False
    assert "backend_inventory.json load_smoke must be true" in payload["issues"]

    _write_deployable_sidecars(report_dir, artifact, baseline)
    backend_inventory = json.loads(
        (report_dir / "backend_inventory.json").read_text(encoding="utf-8")
    )
    backend_inventory["inference_smoke"] = False
    (report_dir / "backend_inventory.json").write_text(
        json.dumps(backend_inventory),
        encoding="utf-8",
    )
    payload = deployable_validator_mod.validate_deployable_artifact(
        artifact,
        backend="bitsandbytes",
        report_dir=report_dir,
        smoke=True,
        baseline_dir=baseline,
    )
    assert payload["ok"] is False
    assert "backend_inventory.json inference_smoke must be true" in payload["issues"]

    _write_deployable_sidecars(report_dir, artifact, baseline)
    (report_dir / "inference_smoke.json").write_text(
        json.dumps({"schema": "invarlock/deployable-inference-smoke-v1", "ok": False}),
        encoding="utf-8",
    )
    payload = deployable_validator_mod.validate_deployable_artifact(
        artifact,
        backend="bitsandbytes",
        report_dir=report_dir,
        smoke=True,
        baseline_dir=baseline,
    )
    assert payload["ok"] is False
    assert payload["inference_smoke"] is False
    assert "inference_smoke.json ok must be true" in payload["issues"]

    payload = deployable_validator_mod.validate_deployable_artifact(
        artifact,
        backend="bitsandbytes",
        report_dir=None,
        smoke=False,
    )
    assert payload["ok"] is False
    assert payload["load_smoke"] is False
    assert payload["inference_smoke"] is False
    assert "deployable validation requires --report-dir sidecars" in payload["issues"]


def test_deployable_validator_remote_code_requires_reviewed_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("INVARLOCK_ALLOW_REMOTE_CODE", raising=False)

    assert deployable_validator_mod._resolve_remote_code_request(False) is False
    with pytest.raises(RuntimeError, match="INVARLOCK_ALLOW_REMOTE_CODE=1"):
        deployable_validator_mod._resolve_remote_code_request(True)

    monkeypatch.setenv("INVARLOCK_ALLOW_REMOTE_CODE", "true")
    assert deployable_validator_mod._resolve_remote_code_request(True) is True


def test_deployable_validation_rejects_remote_code_provenance_drift(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    artifact = tmp_path / "deployable"
    report_dir = tmp_path / "report"
    _write_minimal_artifact(artifact, _deployable_metadata())
    _write_deployable_sidecars(report_dir, artifact)
    monkeypatch.setattr(deployable_impl, "_package_version", lambda _name: "0.1")
    inventory_path = report_dir / "backend_inventory.json"
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    inventory["trust_remote_code"] = True
    inventory_path.write_text(json.dumps(inventory), encoding="utf-8")

    payload = deployable_validator_mod.validate_deployable_artifact(
        artifact,
        backend="bitsandbytes",
        report_dir=report_dir,
        expected_bits=8,
    )

    assert payload["ok"] is False
    assert "backend_inventory.json trust_remote_code mismatch" in payload["issues"]


def test_deployable_smoke_rejects_dense_checkpoint_with_fabricated_metadata(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    artifact = tmp_path / "fabricated-dense"
    baseline = tmp_path / "baseline"
    report_dir = tmp_path / "report"
    _write_minimal_artifact(artifact, _deployable_metadata())
    _write_minimal_artifact(baseline, None)
    _write_deployable_sidecars(report_dir, artifact, baseline)
    monkeypatch.setattr(deployable_impl, "_package_version", lambda _name: "0.1")

    def reject_dense(*_args, **_kwargs):
        raise RuntimeError("reloaded artifact did not expose packed modules")

    monkeypatch.setattr(
        deployable_validator_mod, "_runtime_bitsandbytes_proof", reject_dense
    )

    payload = deployable_validator_mod.validate_deployable_artifact(
        artifact,
        backend="bitsandbytes",
        report_dir=report_dir,
        smoke=True,
        expected_bits=8,
        baseline_dir=baseline,
    )

    assert payload["ok"] is False
    assert payload["validation_scope"] == "runtime_reproof"
    assert payload["runtime_proof_authoritative"] is False
    assert any(
        "runtime deployability smoke failed" in issue for issue in payload["issues"]
    )


def test_deployable_validation_rejects_artifact_and_sidecar_tampering(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    artifact = tmp_path / "deployable"
    report_dir = tmp_path / "report"
    _write_minimal_artifact(artifact, _deployable_metadata())
    _write_deployable_sidecars(report_dir, artifact)
    monkeypatch.setattr(deployable_impl, "_package_version", lambda _name: "0.1")

    (artifact / "weights.safetensors").write_bytes(b"tampered")
    payload = deployable_validator_mod.validate_deployable_artifact(
        artifact,
        backend="bitsandbytes",
        report_dir=report_dir,
        expected_bits=8,
    )
    assert payload["ok"] is False
    assert any("artifact_identity mismatch" in issue for issue in payload["issues"])

    _write_deployable_sidecars(report_dir, artifact)
    inference = json.loads((report_dir / "inference_smoke.json").read_text())
    inference.pop("logits_sha256")
    (report_dir / "inference_smoke.json").write_text(json.dumps(inference))
    payload = deployable_validator_mod.validate_deployable_artifact(
        artifact,
        backend="bitsandbytes",
        report_dir=report_dir,
        expected_bits=8,
    )
    assert payload["ok"] is False
    assert (
        "inference_smoke.json logits_sha256 must be a sha256 digest"
        in payload["issues"]
    )


def test_save_subject_replace_restores_existing_output_on_swap_failure(
    monkeypatch, tmp_path: Path
) -> None:
    output = tmp_path / "subject"
    output.mkdir()
    (output / "marker.txt").write_text("original", encoding="utf-8")
    staging = save_artifact_mod.staging_path_for(output)
    staging.mkdir()
    (staging / "marker.txt").write_text("new", encoding="utf-8")
    original_rename = Path.rename

    def _rename_with_staging_failure(self: Path, target: Path) -> Path:
        if self == staging:
            raise OSError("simulated staging swap failure")
        return original_rename(self, target)

    monkeypatch.setattr(Path, "rename", _rename_with_staging_failure)

    try:
        try:
            save_artifact_mod._replace_output(staging, output)
        except OSError as exc:
            assert "simulated staging swap failure" in str(exc)
        else:  # pragma: no cover - defensive assertion
            raise AssertionError("expected staging swap failure")
    finally:
        monkeypatch.setattr(Path, "rename", original_rename)

    assert output.is_dir()
    assert (output / "marker.txt").read_text(encoding="utf-8") == "original"
    assert staging.is_dir()


def test_batch_transform_artifact_uses_streaming_materializer_without_model_deepcopy(
    monkeypatch, tmp_path: Path
) -> None:
    observed: dict[str, object] = {}
    monkeypatch.setattr(
        batch_edit_mod,
        "materialize_transformation_artifact",
        lambda **kwargs: (
            observed.update(kwargs)
            or {
                "selected_tensors": 1,
                "actual_changes": {"value_changed_params": 1},
            }
        ),
    )

    batch_edit_mod._create_streaming_transformation_artifact(
        baseline_path=tmp_path / "baseline",
        parsed_spec={"type": "quant_rtn", "bits": 4, "group_size": 32, "scope": "ffn"},
        edit_path=tmp_path / "edit",
    )
    assert observed == {
        "baseline_path": tmp_path / "baseline",
        "output_path": tmp_path / "edit",
        "edit_type": "quant_rtn",
        "parameters": {"bits": 4, "group_size": 32},
        "scope": "ffn",
    }
