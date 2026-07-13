from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from enum import Enum
from pathlib import Path

import pytest
import torch

import invarlock.evidence_pack_contracts.deployable_coverage as logical_coverage_mod
from invarlock.evidence_pack_contracts.deployable_coverage import (
    canonical_names_sha256,
    dense_parameter_catalog,
    inspect_bitsandbytes_modules,
    logical_coverage_from_inventory,
)
from scripts.evidence_packs.python.editing.validate_artifact import (
    DEPLOYABLE_SMOKE_PROMPT,
)

ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = (
    ROOT
    / "scripts"
    / "evidence_packs"
    / "python"
    / "editing"
    / "deployable_quantization.py"
)


class _AuthenticatedInt8Params:
    def numel(self) -> int:
        return 6


class _AuthenticatedLinear8bitLt:
    pass


@pytest.fixture(autouse=True)
def _authenticate_fake_bitsandbytes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        logical_coverage_mod,
        "_bitsandbytes_type_contract",
        lambda bits: (
            (_AuthenticatedLinear8bitLt, _AuthenticatedInt8Params)
            if bits == 8
            else (type("UnusedLinear4bit", (), {}), type("UnusedParams4bit", (), {}))
        ),
    )


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "deployable_quantization", MODULE_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _logical_coverage(*, count: int = 1) -> dict[str, object]:
    names = [f"model.layers.{index}.mlp.weight" for index in range(count)]
    return {
        "basis": "dense_baseline_unique_parameters",
        "weight_tensor_names": names,
        "weight_tensor_names_sha256": canonical_names_sha256(names),
        "weight_tensor_count": count,
        "parameter_elements": 24 * count,
        "total_unique_parameter_elements": 32 * count,
    }


def _inventory(*, bits: int = 8, count: int = 1) -> dict[str, object]:
    names = [f"model.layers.{index}.mlp" for index in range(count)]
    module = "Linear8bitLt" if bits == 8 else "Linear4bit"
    return {
        "count": count,
        "names": names,
        "names_sha256": canonical_names_sha256(names),
        "types": [f"bitsandbytes.nn.modules.{module}"],
        "packed_weight_storage_elements": 12,
    }


def test_build_metadata_identifies_real_packed_backend() -> None:
    module = _load_module()

    metadata = module.build_bitsandbytes_metadata(
        logical_coverage=_logical_coverage(),
        runtime_memory_reduction=True,
    )

    assert metadata["artifact_class"] == "deployable_optimized_subject"
    assert metadata["edit_semantics"] == "backend_deployable_edit"
    assert metadata["edit_type"] == "bnb_8bit"
    assert metadata["backend"] == "bitsandbytes"
    assert metadata["packed_quantized_storage"] is True
    assert metadata["actual_storage_format"] == "bitsandbytes_8bit_packed"
    assert metadata["edit_provenance"]["synthetic"] is False
    assert metadata["coverage"] == {
        "edited_tensors": 1,
        "edited_params": 24,
        "total_params": 32,
        "coverage_ratio": 0.75,
    }


def test_generation_and_runtime_validation_share_one_smoke_prompt() -> None:
    module = _load_module()

    assert module.DEPLOYABLE_SMOKE_PROMPT == DEPLOYABLE_SMOKE_PROMPT
    assert DEPLOYABLE_SMOKE_PROMPT == "InvarLock quantized checkpoint verification"
    assert not hasattr(
        module._parse_args(
            [
                "--baseline",
                "/baseline",
                "--output",
                "/output",
                "--report-dir",
                "/proof",
            ]
        ),
        "prompt",
    )


def test_remote_code_defaults_false_and_requires_reviewed_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    monkeypatch.delenv("INVARLOCK_ALLOW_REMOTE_CODE", raising=False)

    assert module._resolve_remote_code_request(False) is False
    with pytest.raises(RuntimeError, match="INVARLOCK_ALLOW_REMOTE_CODE=1"):
        module._resolve_remote_code_request(True)

    monkeypatch.setenv("INVARLOCK_ALLOW_REMOTE_CODE", "1")
    assert module._resolve_remote_code_request(True) is True


def test_cli_remote_code_flag_fails_before_model_or_output_access_without_authorization(
    tmp_path: Path,
) -> None:
    environment = dict(os.environ)
    environment.pop("INVARLOCK_ALLOW_REMOTE_CODE", None)

    result = subprocess.run(
        [
            sys.executable,
            str(MODULE_PATH),
            "--baseline",
            str(tmp_path / "missing-baseline"),
            "--output",
            str(tmp_path / "missing-output"),
            "--report-dir",
            str(tmp_path / "missing-report"),
            "--trust-remote-code",
        ],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
    )

    assert result.returncode != 0
    assert "requires INVARLOCK_ALLOW_REMOTE_CODE=1" in result.stderr
    assert not (tmp_path / "missing-output").exists()
    assert not (tmp_path / "missing-report").exists()


def test_quantized_inventory_fails_closed_without_real_backend_modules() -> None:
    module = _load_module()

    class DenseModel:
        def named_modules(self):
            return [("", object()), ("dense", object())]

    with pytest.raises(RuntimeError, match="no bitsandbytes packed linear modules"):
        module.inspect_bitsandbytes_modules(DenseModel(), bits=8)


def _packed_model(module_name: str = "layer") -> object:
    packed_module = _AuthenticatedLinear8bitLt()
    packed_module.weight = _AuthenticatedInt8Params()
    model_type = type(
        "PackedModel",
        (),
        {"named_modules": lambda self: [("", self), (module_name, packed_module)]},
    )
    return model_type()


def test_logical_coverage_maps_modules_to_dense_weight_tensors() -> None:
    dense = torch.nn.Module()
    dense.layer = torch.nn.Linear(4, 3, bias=False)

    inventory = inspect_bitsandbytes_modules(_packed_model(), bits=8)
    logical = logical_coverage_from_inventory(dense_parameter_catalog(dense), inventory)

    assert inventory["count"] == 1
    assert inventory["names"] == ["layer"]
    assert inventory["packed_weight_storage_elements"] == 6
    assert logical["weight_tensor_names"] == ["layer.weight"]
    assert logical["weight_tensor_count"] == 1
    assert logical["parameter_elements"] == 12
    assert logical["total_unique_parameter_elements"] == 12


def test_logical_coverage_rejects_tied_dense_target_ambiguity() -> None:
    dense = torch.nn.Module()
    dense.layer = torch.nn.Linear(4, 3, bias=False)
    dense.alias = torch.nn.Linear(4, 3, bias=False)
    dense.alias.weight = dense.layer.weight

    with pytest.raises(RuntimeError, match="tied or ambiguous"):
        logical_coverage_from_inventory(
            dense_parameter_catalog(dense),
            inspect_bitsandbytes_modules(_packed_model(), bits=8),
        )


def test_logical_coverage_rejects_unmatched_packed_module_name() -> None:
    dense = torch.nn.Module()
    dense.layer = torch.nn.Linear(4, 3, bias=False)

    with pytest.raises(RuntimeError, match="no dense baseline weight"):
        logical_coverage_from_inventory(
            dense_parameter_catalog(dense),
            inspect_bitsandbytes_modules(_packed_model("missing"), bits=8),
        )


def test_memory_reduction_must_be_observed() -> None:
    module = _load_module()

    with pytest.raises(RuntimeError, match="did not reduce runtime model footprint"):
        module.require_memory_reduction(100, 100)
    with pytest.raises(RuntimeError, match="positive"):
        module.require_memory_reduction(0, 50)

    result = module.require_memory_reduction(100, 60)
    assert result["runtime_memory_reduction_observed"] is True
    assert result["reduction_bytes"] == 40
    assert result["reduction_ratio"] == pytest.approx(0.4)


def test_serialized_quantization_config_accepts_transformers_enum() -> None:
    module = _load_module()

    class Method(Enum):
        BITS_AND_BYTES = "bitsandbytes"

    class Config:
        quantization_config = {
            "quant_method": Method.BITS_AND_BYTES,
            "load_in_8bit": True,
        }

    class Model:
        config = Config()

    assert module._config_quantization_payload(Model())["load_in_8bit"] is True


def test_sidecars_record_observed_smokes_and_backend_inventory(tmp_path: Path) -> None:
    module = _load_module()
    inventory = _inventory(count=2)
    memory = module.require_memory_reduction(100, 60)

    module.write_deployable_sidecars(
        tmp_path,
        backend_version="0.49.2",
        transformers_version="5.12.0",
        inventory=inventory,
        logical_coverage=_logical_coverage(count=2),
        quantized_footprint=60,
        memory=memory,
        load_details={"config_quant_method": "bitsandbytes"},
        inference_details={
            "logits_sha256": "sha256:" + "a" * 64,
            "logits_shape": [1, 2, 3],
            "all_logits_finite": True,
        },
        artifact_identity={
            "kind": "local_checkpoint_tree",
            "sha256": "sha256:" + "b" * 64,
        },
        baseline_identity={
            "kind": "local_checkpoint_tree",
            "sha256": "sha256:" + "c" * 64,
        },
    )

    backend = json.loads((tmp_path / "backend_inventory.json").read_text())
    assert backend["load_smoke"] is True
    assert backend["inference_smoke"] is True
    assert backend["quantized_module_count"] == 2
    assert backend["quantized_module_types"] == inventory["types"]
    assert backend["packed_weight_storage_elements"] == 12
    assert backend["logical_coverage"]["parameter_elements"] == 48
    assert backend["trust_remote_code"] is False

    for name in ("memory_report.json", "load_smoke.json", "inference_smoke.json"):
        assert json.loads((tmp_path / name).read_text())["ok"] is True
        assert json.loads((tmp_path / name).read_text())["trust_remote_code"] is False


def test_sidecars_reject_memory_inventory_footprint_drift(tmp_path: Path) -> None:
    module = _load_module()
    memory = module.require_memory_reduction(100, 60)

    with pytest.raises(RuntimeError, match="footprints disagree"):
        module.write_deployable_sidecars(
            tmp_path,
            backend_version="0.49.2",
            transformers_version="5.12.0",
            inventory=_inventory(),
            logical_coverage=_logical_coverage(),
            quantized_footprint=59,
            memory=memory,
            load_details={},
            inference_details={},
            artifact_identity={
                "kind": "local_checkpoint_tree",
                "sha256": "sha256:" + "b" * 64,
            },
            baseline_identity={
                "kind": "local_checkpoint_tree",
                "sha256": "sha256:" + "c" * 64,
            },
        )


def test_sidecars_bind_reviewed_remote_code_choice(tmp_path: Path) -> None:
    module = _load_module()
    module.write_deployable_sidecars(
        tmp_path,
        backend_version="0.49.2",
        transformers_version="5.12.0",
        inventory=_inventory(),
        logical_coverage=_logical_coverage(),
        quantized_footprint=60,
        memory=module.require_memory_reduction(100, 60),
        load_details={"config_quant_method": "bitsandbytes"},
        inference_details={
            "logits_sha256": "sha256:" + "a" * 64,
            "logits_shape": [1, 2, 3],
            "all_logits_finite": True,
        },
        artifact_identity={
            "kind": "local_checkpoint_tree",
            "sha256": "sha256:" + "b" * 64,
        },
        baseline_identity={
            "kind": "local_checkpoint_tree",
            "sha256": "sha256:" + "c" * 64,
        },
        trust_remote_code=True,
    )

    for name in module.PROOF_SIDECARS:
        payload = json.loads((tmp_path / name).read_text(encoding="utf-8"))
        assert payload["trust_remote_code"] is True


def test_promotion_rolls_back_artifact_if_report_promotion_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_module()
    artifact_stage = tmp_path / ".artifact-stage"
    report_stage = tmp_path / ".report-stage"
    output = tmp_path / "artifact"
    report = tmp_path / "report"
    artifact_stage.mkdir()
    report_stage.mkdir()
    (artifact_stage / "model.safetensors").write_text("packed")
    (report_stage / "load_smoke.json").write_text("{}")
    (report_stage / "publication_commit.json").write_text(
        json.dumps(
            {
                "committed": True,
                "validation_scope": "structural_only",
                "runtime_proof_authoritative": False,
            }
        ),
        encoding="utf-8",
    )
    real_rename = Path.rename

    def fail_report_rename(self: Path, target: Path) -> Path:
        if self == report_stage:
            raise OSError("injected report promotion failure")
        return real_rename(self, target)

    monkeypatch.setattr(Path, "rename", fail_report_rename)

    with pytest.raises(OSError, match="injected report promotion failure"):
        module.promote_staged_outputs(
            artifact_stage,
            report_stage,
            output,
            report,
        )

    assert not output.exists()
    assert not report.exists()
    assert artifact_stage.is_dir()
    assert report_stage.is_dir()


def test_promotion_requires_structural_non_authoritative_commit_marker(
    tmp_path: Path,
) -> None:
    module = _load_module()
    artifact_stage = tmp_path / ".artifact-stage"
    report_stage = tmp_path / ".report-stage"
    artifact_stage.mkdir()
    report_stage.mkdir()

    with pytest.raises(RuntimeError, match="commit marker"):
        module.promote_staged_outputs(
            artifact_stage,
            report_stage,
            tmp_path / "artifact",
            tmp_path / "report",
        )

    (report_stage / "publication_commit.json").write_text(
        json.dumps(
            {
                "committed": True,
                "validation_scope": "runtime_reproof",
                "runtime_proof_authoritative": True,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="structural and non-authoritative"):
        module.promote_staged_outputs(
            artifact_stage,
            report_stage,
            tmp_path / "artifact",
            tmp_path / "report",
        )


def test_retry_completes_interrupted_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_module()
    output = tmp_path / "artifact"
    output.mkdir()
    (output / "config.json").write_text("{}", encoding="utf-8")
    (output / "model.safetensors").write_bytes(b"packed")
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    (baseline / "config.json").write_text("{}", encoding="utf-8")
    (baseline / "model.safetensors").write_bytes(b"baseline")
    baseline_identity = {
        "kind": "local_checkpoint_tree",
        "sha256": module.checkpoint_tree_sha256(baseline),
    }
    report = tmp_path / "proof"
    report_stage = tmp_path / ".proof.staging-1234"
    report_stage.mkdir()
    identity = {
        "kind": "local_checkpoint_tree",
        "sha256": module.checkpoint_tree_sha256(output),
    }
    (report_stage / "publication_commit.json").write_text(
        json.dumps(
            {
                "schema": "invarlock/deployable-publication-commit-v1",
                "committed": True,
                "validation_scope": "structural_only",
                "runtime_proof_authoritative": False,
                "artifact_identity": identity,
                "baseline_identity": baseline_identity,
                "bits": 8,
                "trust_remote_code": False,
            }
        ),
        encoding="utf-8",
    )
    (report_stage / "runtime_deployability_validation.json").write_text(
        json.dumps(
            {
                "validation_scope": "runtime_reproof",
                "runtime_proof_authoritative": True,
                "artifact_identity": {"sha256": "sha256:" + "f" * 64},
            }
        ),
        encoding="utf-8",
    )
    validation_calls: list[bool] = []

    def validate_recovery(*_args, **kwargs):
        smoke = bool(kwargs["smoke"])
        validation_calls.append(smoke)
        if smoke:
            return {
                "ok": True,
                "validation_scope": "runtime_reproof",
                "runtime_proof_authoritative": True,
                "runtime_proof": {"packed_module_reloaded": True},
                "artifact_identity": identity,
                "baseline_identity": baseline_identity,
                "bits": 8,
                "trust_remote_code": False,
            }
        return {
            "ok": True,
            "validation_scope": "structural_only",
            "runtime_proof_authoritative": False,
            "artifact_identity": identity,
        }

    monkeypatch.setattr(module, "validate_deployable_artifact", validate_recovery)

    recovered = module.recover_interrupted_publication(
        output,
        report,
        baseline_path=baseline,
        bits=8,
        trust_remote_code=False,
    )

    assert recovered is not None
    assert recovered["runtime_proof_authoritative"] is True
    assert recovered["validation_scope"] == "runtime_reproof"
    assert validation_calls == [False, True]
    assert report.is_dir()
    assert not report_stage.exists()
    persisted = json.loads(
        (report / "runtime_deployability_validation.json").read_text(encoding="utf-8")
    )
    assert persisted["artifact_identity"] == identity
    assert persisted["baseline_identity"] == baseline_identity
    assert persisted["bits"] == 8
    assert persisted["validation_scope"] == "runtime_reproof"
    assert persisted["runtime_proof_authoritative"] is True


def test_recovery_rejects_dense_artifact_with_recomputed_sidecars_and_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_module()
    output = tmp_path / "dense-artifact"
    output.mkdir()
    (output / "config.json").write_text("{}", encoding="utf-8")
    (output / "model.safetensors").write_bytes(b"dense weights")
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    (baseline / "config.json").write_text("{}", encoding="utf-8")
    (baseline / "model.safetensors").write_bytes(b"baseline weights")
    report = tmp_path / "proof"
    report_stage = tmp_path / ".proof.staging-dense"
    report_stage.mkdir()
    identity = {
        "kind": "local_checkpoint_tree",
        "sha256": module.checkpoint_tree_sha256(output),
    }
    baseline_identity = {
        "kind": "local_checkpoint_tree",
        "sha256": module.checkpoint_tree_sha256(baseline),
    }
    for sidecar in module.PROOF_SIDECARS:
        (report_stage / sidecar).write_text(
            json.dumps({"recomputed": True}), encoding="utf-8"
        )
    (report_stage / "deployable_artifact_validation.json").write_text(
        json.dumps(
            {
                "ok": True,
                "validation_scope": "structural_only",
                "runtime_proof_authoritative": False,
            }
        ),
        encoding="utf-8",
    )
    (report_stage / "publication_commit.json").write_text(
        json.dumps(
            {
                "schema": "invarlock/deployable-publication-commit-v1",
                "committed": True,
                "validation_scope": "structural_only",
                "runtime_proof_authoritative": False,
                "artifact_identity": identity,
                "baseline_identity": baseline_identity,
                "bits": 8,
                "trust_remote_code": False,
            }
        ),
        encoding="utf-8",
    )
    validation_calls: list[bool] = []

    def reject_dense_recovery(*_args, **kwargs):
        smoke = bool(kwargs["smoke"])
        validation_calls.append(smoke)
        if smoke:
            return {
                "ok": False,
                "validation_scope": "runtime_reproof",
                "runtime_proof_authoritative": False,
                "runtime_proof": None,
                "issues": ["dense artifact did not expose packed modules"],
            }
        return {
            "ok": True,
            "validation_scope": "structural_only",
            "runtime_proof_authoritative": False,
        }

    monkeypatch.setattr(module, "validate_deployable_artifact", reject_dense_recovery)

    with pytest.raises(FileExistsError, match="unproven interrupted"):
        module.recover_interrupted_publication(
            output,
            report,
            baseline_path=baseline,
            bits=8,
            trust_remote_code=False,
        )

    assert validation_calls == [False, True]
    assert output.is_dir()
    assert report_stage.is_dir()
    assert not report.exists()
    assert not (report_stage / "runtime_deployability_validation.json").exists()


def test_retry_rejects_orphan_without_bound_proof(tmp_path: Path) -> None:
    module = _load_module()
    output = tmp_path / "artifact"
    output.mkdir()
    (output / "config.json").write_text("{}", encoding="utf-8")
    (output / "model.safetensors").write_bytes(b"packed")
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    (baseline / "config.json").write_text("{}", encoding="utf-8")
    (baseline / "model.safetensors").write_bytes(b"baseline")

    with pytest.raises(FileExistsError, match="unproven interrupted"):
        module.recover_interrupted_publication(
            output,
            tmp_path / "proof",
            baseline_path=baseline,
            bits=8,
            trust_remote_code=False,
        )


def test_retry_rejects_proof_from_different_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_module()
    output = tmp_path / "artifact"
    output.mkdir()
    (output / "config.json").write_text("{}", encoding="utf-8")
    (output / "model.safetensors").write_bytes(b"packed")
    baseline_a = tmp_path / "baseline-a"
    baseline_b = tmp_path / "baseline-b"
    for baseline, weights in ((baseline_a, b"a"), (baseline_b, b"b")):
        baseline.mkdir()
        (baseline / "config.json").write_text("{}", encoding="utf-8")
        (baseline / "model.safetensors").write_bytes(weights)
    report_stage = tmp_path / ".proof.staging-1234"
    report_stage.mkdir()
    (report_stage / "publication_commit.json").write_text(
        json.dumps(
            {
                "schema": "invarlock/deployable-publication-commit-v1",
                "committed": True,
                "artifact_identity": {
                    "kind": "local_checkpoint_tree",
                    "sha256": module.checkpoint_tree_sha256(output),
                },
                "baseline_identity": {
                    "kind": "local_checkpoint_tree",
                    "sha256": module.checkpoint_tree_sha256(baseline_a),
                },
                "bits": 8,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        module,
        "validate_deployable_artifact",
        lambda *_args, **_kwargs: {"ok": True},
    )

    with pytest.raises(FileExistsError, match="unproven interrupted"):
        module.recover_interrupted_publication(
            output,
            tmp_path / "proof",
            baseline_path=baseline_b,
            bits=8,
            trust_remote_code=False,
        )


def test_publication_flushes_staged_trees_and_parent_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_module()
    artifact_stage = tmp_path / ".artifact-stage"
    report_stage = tmp_path / ".report-stage"
    artifact_stage.mkdir()
    report_stage.mkdir()
    (artifact_stage / "model.safetensors").write_bytes(b"packed")
    (report_stage / "publication_commit.json").write_text(
        json.dumps(
            {
                "committed": True,
                "validation_scope": "structural_only",
                "runtime_proof_authoritative": False,
            }
        ),
        encoding="utf-8",
    )
    flushed_trees: list[Path] = []
    flushed_directories: list[Path] = []
    monkeypatch.setattr(module, "_fsync_tree", flushed_trees.append)
    monkeypatch.setattr(module, "_fsync_directory", flushed_directories.append)

    module.promote_staged_outputs(
        artifact_stage,
        report_stage,
        tmp_path / "artifact",
        tmp_path / "report",
    )

    assert flushed_trees == [artifact_stage, report_stage]
    assert flushed_directories.count(tmp_path) >= 3


def test_output_paths_must_not_exist(tmp_path: Path) -> None:
    module = _load_module()
    output = tmp_path / "subject"
    report = tmp_path / "report"
    output.mkdir()

    with pytest.raises(FileExistsError, match="refusing to replace"):
        module.require_fresh_outputs(output, report)


def test_four_bit_metadata_is_not_labeled_as_dense_rtn() -> None:
    module = _load_module()

    metadata = module.build_bitsandbytes_metadata(
        bits=4,
        logical_coverage=_logical_coverage(),
        runtime_memory_reduction=True,
    )

    assert metadata["edit_type"] == "bnb_4bit"
    assert metadata["storage_format"] == "bitsandbytes_4bit_packed"
    assert metadata["quantization_mode"] == "packed_backend_checkpoint"
    assert (
        metadata["edit_provenance"]["edit_method"]
        == "transformers_bitsandbytes_4bit_checkpoint"
    )


def test_serialized_quantization_config_rejects_opposite_bit_flag() -> None:
    module = _load_module()

    class Config:
        quantization_config = {
            "quant_method": "bitsandbytes",
            "load_in_8bit": True,
            "load_in_4bit": True,
        }

    class Model:
        config = Config()

    with pytest.raises(RuntimeError, match="does not identify"):
        module._config_quantization_payload(Model(), bits=8)
