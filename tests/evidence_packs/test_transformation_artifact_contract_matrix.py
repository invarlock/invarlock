from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from invarlock.clean_selection.common import canonical_json_sha256
from invarlock.evidence_pack_transformation_validation import (
    _clean_transformation_selection_errors,
    _safe_checkpoint_relative_path,
    _transformation_change_errors,
    _transformation_materialization_receipt_errors,
    _transformation_metadata_errors,
    _transformation_output_weights_errors,
    _transformation_shard_plan_errors,
    _transformation_target_manifest_errors,
)
from tests.evidence_packs._support_transformation_pack import _make_pack

_DIGEST = "sha256:" + "a" * 64


def _replay_arguments(
    tmp_path: Path, *, edit_type: str = "quant_rtn"
) -> dict[str, object]:
    pack, report_dir, replay = _make_pack(tmp_path, edit_type=edit_type)
    return {
        "scenario_id": report_dir.parent.name,
        "report": json.loads(
            (report_dir / "evaluation.report.json").read_text(encoding="utf-8")
        ),
        "metadata": json.loads(
            (report_dir / "edit_metadata.json").read_text(encoding="utf-8")
        ),
        "payload": replay,
        "spec": json.loads(
            (pack / "metadata" / "scenarios.json").read_text(encoding="utf-8")
        )["scenarios"][0],
        "pack_dir": pack,
        "report_dir": report_dir,
        "report_model_name": report_dir.parent.parent.name,
    }


def _validation_values(
    tmp_path: Path,
) -> tuple[dict[str, object], dict[str, object], str]:
    arguments = _replay_arguments(tmp_path)
    payload = arguments["payload"]
    transformation = payload["transformation"]
    scope = payload["scope"]
    assert isinstance(payload, dict)
    assert isinstance(transformation, dict)
    assert isinstance(scope, str)
    return payload, transformation, scope


def test_target_manifest_adversarial_shape_matrix(tmp_path: Path) -> None:
    payload, transformation, scope = _validation_values(tmp_path)
    bad = deepcopy(payload)
    bad["target_manifest"] = None
    assert _transformation_target_manifest_errors(
        prefix="x: ", payload=bad, transformation=transformation, scope=scope
    ) == ["x: transformation replay target_manifest must be an object"]

    bad = deepcopy(payload)
    bad["target_manifest_sha256"] = "bad"
    errors = _transformation_target_manifest_errors(
        prefix="x: ", payload=bad, transformation=transformation, scope=scope
    )
    assert any(
        "target_manifest_sha256 must be a sha256 digest" in error for error in errors
    )

    bad = deepcopy(payload)
    manifest = bad["target_manifest"]
    manifest["extra"] = {1, 2}
    errors = _transformation_target_manifest_errors(
        prefix="x: ", payload=bad, transformation=transformation, scope=scope
    )
    assert any("target_manifest is not JSON-safe" in error for error in errors)
    assert any("target_manifest has unbound fields" in error for error in errors)

    bad = deepcopy(payload)
    bad["target_manifest"]["scope"] = "attn"
    errors = _transformation_target_manifest_errors(
        prefix="x: ", payload=bad, transformation=transformation, scope=scope
    )
    assert any("target_manifest scope mismatch" in error for error in errors)
    assert any("target_manifest digest mismatch" in error for error in errors)

    bad = deepcopy(payload)
    bad["target_manifest"]["targets"] = []
    errors = _transformation_target_manifest_errors(
        prefix="x: ", payload=bad, transformation=transformation, scope=scope
    )
    assert any("targets must be a non-empty list" in error for error in errors)

    bad = deepcopy(payload)
    targets = [
        None,
        {
            "name": "",
            "dtype": "torch.float8_e4m3fn",
            "shape": [True],
            "numel": False,
            "role": "vision",
            "layer": True,
            "extra": 1,
        },
        {
            "name": "z.weight",
            "dtype": "torch.float32",
            "shape": [2, 2],
            "numel": 3,
            "role": "ffn",
            "layer": 1,
        },
        {
            "name": "z.weight",
            "dtype": "torch.float32",
            "shape": [1, 1],
            "numel": 1,
            "role": "ffn",
            "layer": 0,
        },
    ]
    bad["target_manifest"]["targets"] = targets
    bad["selected_params"] = 999
    errors = _transformation_target_manifest_errors(
        prefix="x: ",
        payload=bad,
        transformation=transformation,
        scope="all@layers=1,layer=0",
    )
    expected = (
        "targets[0] must be an object",
        "targets[1] has unbound fields",
        "targets[1].name must be a non-empty string",
        "targets[1].dtype must be regular floating-point storage",
        "targets[1].shape must be a positive matrix shape",
        "targets[1].numel must be a positive int",
        "targets[1].role is outside the declared scope",
        "targets[1].layer must be a non-negative int",
        "targets[2].numel does not match shape",
        "targets[2].layer is outside the layers qualifier",
        "targets must be sorted and unique",
        "selected_tensors does not match target manifest",
        "selected_params does not match target manifest",
    )
    for fragment in expected:
        assert any(fragment in error for error in errors), fragment

    bad = deepcopy(payload)
    bad["target_manifest"]["targets"][0]["layer"] = 1
    errors = _transformation_target_manifest_errors(
        prefix="x: ",
        payload=bad,
        transformation=transformation,
        scope="all@layer=0",
    )
    assert any("layer is outside the layer qualifier" in error for error in errors)


def test_output_weights_and_path_contract_matrix(tmp_path: Path) -> None:
    assert _transformation_output_weights_errors(prefix="x: ", output_weights=None) == [
        "x: transformation replay output_weights must be an object"
    ]
    assert _transformation_output_weights_errors(prefix="x: ", output_weights={}) == [
        "x: transformation replay output_weights has unbound fields"
    ]
    errors = _transformation_output_weights_errors(
        prefix="x: ",
        output_weights={"sha256": "bad", "index_sha256": "bad", "shards": []},
    )
    assert len(errors) == 3
    errors = _transformation_output_weights_errors(
        prefix="x: ",
        output_weights={
            "sha256": _DIGEST,
            "index_sha256": _DIGEST,
            "shards": [
                None,
                {"name": None, "sha256": _DIGEST},
                {"name": "../bad.safetensors", "sha256": "bad"},
                {"name": "z.safetensors", "sha256": _DIGEST},
                {"name": "z.safetensors", "sha256": _DIGEST},
            ],
        },
    )
    for fragment in (
        "must contain only name and sha256",
        "name must be a safe safetensors filename",
        "sha256 must be a sha256 digest",
        "shards must be sorted and unique",
    ):
        assert any(fragment in error for error in errors)

    payload, _, _ = _validation_values(tmp_path)
    weights = deepcopy(payload["output_weights"])
    weights["sha256"] = _DIGEST
    assert any(
        "output_weights digest mismatch" in error
        for error in _transformation_output_weights_errors(
            prefix="x: ", output_weights=weights
        )
    )
    for unsafe in (None, "", "/abs", "a\\b", ".", "..", "a//b", "a/../b"):
        assert not _safe_checkpoint_relative_path(unsafe)
    assert _safe_checkpoint_relative_path("weights/model.safetensors")


def test_shard_plan_adversarial_matrix(tmp_path: Path) -> None:
    payload, _, _ = _validation_values(tmp_path)
    bad = deepcopy(payload)
    bad["source_shard_plan"] = {}
    bad["output_shard_plan"] = {}
    errors = _transformation_shard_plan_errors(prefix="x: ", payload=bad)
    for fragment in (
        "source_shard_plan is invalid",
        "source_shard_plan digest mismatch",
        "output_shard_plan is invalid",
        "output_shard_plan digest mismatch",
        "output weights do not match output shard plan",
    ):
        assert any(fragment in error for error in errors), fragment

    bad = deepcopy(payload)
    source = bad["source_shard_plan"]["source_shards"][0]
    source.update(
        path="../bad.bin", sha256="bad", tensor_names=["z", "z"], byte_count=0
    )
    errors = _transformation_shard_plan_errors(prefix="x: ", payload=bad)
    for fragment in (
        "path is not a safe safetensors path",
        "sha256 must be a sha256 digest",
        "tensor_names must be sorted and unique",
        "byte_count must be positive",
    ):
        assert any(fragment in error for error in errors), fragment

    bad = deepcopy(payload)
    bad["source_shard_plan"]["source_shards"][0]["sha256"] = None
    errors = _transformation_shard_plan_errors(prefix="x: ", payload=bad)
    assert any("sha256 must be a sha256 digest" in error for error in errors)

    bad = deepcopy(payload)
    bad["source_shard_plan"]["source_shards"].append(
        deepcopy(bad["source_shard_plan"]["source_shards"][0])
    )
    errors = _transformation_shard_plan_errors(prefix="x: ", payload=bad)
    assert any(
        "source shard paths must be sorted and unique" in error for error in errors
    )

    bad = deepcopy(payload)
    chunk = bad["output_shard_plan"]["chunks"][0]
    chunk["source_path"] = "../bad"
    chunk["byte_count"] = 0
    errors = _transformation_shard_plan_errors(prefix="x: ", payload=bad)
    assert any("source_path is invalid" in error for error in errors)
    assert any("byte_count must be positive" in error for error in errors)

    bad = deepcopy(payload)
    chunk = bad["output_shard_plan"]["chunks"][0]
    chunk["source_sha256"] = "bad"
    chunk["tensor_names"] = ["not.in.source"]
    errors = _transformation_shard_plan_errors(prefix="x: ", payload=bad)
    assert any("source_sha256 mismatch" in error for error in errors)
    assert any("tensor_names are not in source shard" in error for error in errors)

    bad = deepcopy(payload)
    bad["output_shard_plan"]["chunks"].append(
        deepcopy(bad["output_shard_plan"]["chunks"][0])
    )
    errors = _transformation_shard_plan_errors(prefix="x: ", payload=bad)
    assert any(
        "output shard names must be sorted and unique" in error for error in errors
    )


def test_change_metadata_and_materialization_contract_matrix(tmp_path: Path) -> None:
    assert _transformation_change_errors(prefix="x: ", actual_changes=None) == [
        "x: transformation replay actual_changes must be an object"
    ]
    errors = _transformation_change_errors(
        prefix="x: ",
        actual_changes={
            "value_changed_tensors": True,
            "value_changed_params": 0,
            "byte_changed_tensors": -1,
            "byte_changed_params": 1,
            "extra": 1,
        },
    )
    assert any("actual_changes has unbound fields" in error for error in errors)
    assert any("must be a non-negative int" in error for error in errors)
    assert any("must be positive" in error for error in errors)

    arguments = _replay_arguments(tmp_path)
    payload = arguments["payload"]
    transformation = payload["transformation"]
    scope = payload["scope"]
    errors = _transformation_metadata_errors(
        prefix="x: ",
        metadata={},
        payload=payload,
        transformation=transformation,
        scope=scope,
    )
    assert any("metadata coverage missing" in error for error in errors)
    metadata = deepcopy(arguments["metadata"])
    metadata["coverage"]["coverage_ratio"] = float("nan")
    errors = _transformation_metadata_errors(
        prefix="x: ",
        metadata=metadata,
        payload=payload,
        transformation=transformation,
        scope=scope,
    )
    assert any("coverage.coverage_ratio mismatch" in error for error in errors)

    report_dir = arguments["report_dir"]
    receipt = json.loads(
        (report_dir / "transformation_materialization.json").read_text(encoding="utf-8")
    )
    receipt["extra"] = True
    receipt["output_shards"] = 0
    receipt["resume_count"] = True
    errors = _transformation_materialization_receipt_errors(
        prefix="x: ",
        receipt=receipt,
        payload=payload,
        transformation=transformation,
        scope=scope,
    )
    assert any("receipt has unbound fields" in error for error in errors)
    assert any("output_shards must be positive" in error for error in errors)
    assert any("resume_count must be a non-negative int" in error for error in errors)
    receipt = json.loads(
        (report_dir / "transformation_materialization.json").read_text(encoding="utf-8")
    )
    receipt["output_shards"] = 2
    errors = _transformation_materialization_receipt_errors(
        prefix="x: ",
        receipt=receipt,
        payload=payload,
        transformation=transformation,
        scope=scope,
    )
    assert any("output_shards mismatch" in error for error in errors)
    no_weights_payload = deepcopy(payload)
    no_weights_payload["output_weights"] = None
    receipt["output_shards"] = 1
    _transformation_materialization_receipt_errors(
        prefix="x: ",
        receipt=receipt,
        payload=no_weights_payload,
        transformation=transformation,
        scope=scope,
    )

    bad = deepcopy(payload)
    bad["source_shard_plan"]["source_shards"] = []
    bad["output_shard_plan"]["chunks"] = []
    errors = _transformation_shard_plan_errors(prefix="x: ", payload=bad)
    assert any("source_shards is empty" in error for error in errors)
    assert any("chunks is empty" in error for error in errors)

    bad = deepcopy(payload)
    source = bad["source_shard_plan"]["source_shards"][0]
    source.update(
        path="../bad.bin", sha256="bad", tensor_names=["z", "z"], byte_count=0
    )
    source["extra"] = True
    bad["source_shard_plan"]["source_shards"].append(None)
    errors = _transformation_shard_plan_errors(prefix="x: ", payload=bad)
    assert any("has unbound fields" in error for error in errors)

    bad = deepcopy(payload)
    output = bad["output_shard_plan"]
    output["source_shard_plan_sha256"] = "bad"
    output["target_manifest_sha256"] = "bad"
    chunk = output["chunks"][0]
    chunk.update(
        name="../bad.bin",
        source_path="missing.safetensors",
        source_sha256="bad",
        tensor_names=["missing", "missing"],
        byte_count=2 * 1024 * 1024,
    )
    output["chunks"].append(None)
    errors = _transformation_shard_plan_errors(prefix="x: ", payload=bad)
    for fragment in (
        "source plan digest mismatch",
        "target manifest digest mismatch",
        "name is not a safe output shard name",
        "source_path is not in source plan",
        "tensor_names must be sorted and unique",
        "byte_count exceeds the bound",
        "chunks[1] has unbound fields",
    ):
        assert any(fragment in error for error in errors), fragment


def _clean_selection_values(
    tmp_path: Path,
) -> tuple[Path, str, str, dict[str, object], dict[str, object], str]:
    pack, report_dir, payload = _make_pack(tmp_path, clean=True)
    transformation = payload["transformation"]
    scope = payload["scope"]
    assert isinstance(transformation, dict)
    assert isinstance(scope, str)
    return (
        pack,
        report_dir.parent.name,
        report_dir.parent.parent.name,
        payload,
        transformation,
        scope,
    )


def test_clean_selection_receipt_adversarial_matrix(tmp_path: Path) -> None:
    pack, scenario_id, model_name, payload, transformation, scope = (
        _clean_selection_values(tmp_path)
    )
    assert (
        _clean_transformation_selection_errors(
            pack_dir=pack,
            scenario_id=scenario_id,
            report_model_name=model_name,
            payload=payload,
            transformation=transformation,
            scope=scope,
        )
        == []
    )

    bundle_path = pack / "metadata/clean_selection/selection_bundle.json"
    bundle_bytes = bundle_path.read_bytes()
    bundle_path.unlink()
    errors = _clean_transformation_selection_errors(
        pack_dir=pack,
        scenario_id=scenario_id,
        report_model_name=model_name,
        payload=payload,
        transformation=transformation,
        scope=scope,
    )
    assert any("selection bundle invalid" in error for error in errors)
    bundle_path.write_bytes(bundle_bytes)

    bad = deepcopy(payload)
    bad.pop("selection_receipt")
    errors = _clean_transformation_selection_errors(
        pack_dir=pack,
        scenario_id=scenario_id,
        report_model_name=model_name,
        payload=bad,
        transformation=transformation,
        scope=scope,
    )
    assert any("selection_receipt is missing" in error for error in errors)

    bad = deepcopy(payload)
    bad["selection_receipt_sha256"] = "bad"
    errors = _clean_transformation_selection_errors(
        pack_dir=pack,
        scenario_id=scenario_id,
        report_model_name=model_name,
        payload=bad,
        transformation=transformation,
        scope=scope,
    )
    assert any("selection_receipt_sha256 must be" in error for error in errors)

    bad = deepcopy(payload)
    bad["selection_receipt"]["scope"] = "attn"
    errors = _clean_transformation_selection_errors(
        pack_dir=pack,
        scenario_id=scenario_id,
        report_model_name=model_name,
        payload=bad,
        transformation=transformation,
        scope=scope,
    )
    assert any("selection receipt digest mismatch" in error for error in errors)

    bad = deepcopy(payload)
    bad["selection_receipt"]["extra"] = True
    bad["selection_receipt_sha256"] = canonical_json_sha256(bad["selection_receipt"])
    errors = _clean_transformation_selection_errors(
        pack_dir=pack,
        scenario_id=scenario_id,
        report_model_name=model_name,
        payload=bad,
        transformation=transformation,
        scope=scope,
    )
    assert any("receipt has unbound fields" in error for error in errors)

    bad = deepcopy(payload)
    bad["selection_receipt"]["original_model_key"] = "ORG/MODEL"
    bad["selection_receipt_sha256"] = canonical_json_sha256(bad["selection_receipt"])
    errors = _clean_transformation_selection_errors(
        pack_dir=pack,
        scenario_id=scenario_id,
        report_model_name=model_name,
        payload=bad,
        transformation=transformation,
        scope=scope,
    )
    assert any("no unique model/edit entry" in error for error in errors)

    alternate_transformation = {**transformation, "edit_type": "synthetic_dense_update"}
    errors = _clean_transformation_selection_errors(
        pack_dir=pack,
        scenario_id=scenario_id,
        report_model_name=model_name,
        payload=payload,
        transformation=alternate_transformation,
        scope=scope,
    )
    assert any("no unique model/edit entry" in error for error in errors)

    bad = deepcopy(payload)
    bad["selection_receipt"]["original_model_key"] = "wrong/model"
    bad["selection_receipt_sha256"] = canonical_json_sha256(bad["selection_receipt"])
    errors = _clean_transformation_selection_errors(
        pack_dir=pack,
        scenario_id=scenario_id,
        report_model_name=model_name,
        payload=bad,
        transformation=transformation,
        scope=scope,
    )
    assert any("original_model_key mismatch" in error for error in errors)


def test_clean_selection_final_cross_bindings(tmp_path: Path) -> None:
    pack, scenario_id, model_name, payload, transformation, scope = (
        _clean_selection_values(tmp_path)
    )
    errors = _clean_transformation_selection_errors(
        pack_dir=pack,
        scenario_id=scenario_id,
        report_model_name=model_name,
        payload=payload,
        transformation=transformation,
        scope="attn",
    )
    assert any(
        "selected candidate differs from final replay" in error for error in errors
    )

    bad = deepcopy(payload)
    bad["baseline_identity"] = {
        "kind": "local_checkpoint_tree",
        "sha256": "sha256:" + "0" * 64,
    }
    bad["artifact_identity"] = {
        "kind": "local_checkpoint_tree",
        "sha256": "sha256:" + "9" * 64,
    }
    errors = _clean_transformation_selection_errors(
        pack_dir=pack,
        scenario_id=scenario_id,
        report_model_name=model_name,
        payload=bad,
        transformation=transformation,
        scope=scope,
    )
    assert any("baseline identity mismatch" in error for error in errors)
    assert any("artifact identity mismatch" in error for error in errors)
