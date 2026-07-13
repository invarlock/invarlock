from __future__ import annotations

import json
from pathlib import Path

import pytest

from invarlock.evidence_pack_edit_common import (
    RUNTIME_LOAD_DIAGNOSTICS_SCHEMA,
    RUNTIME_RELOAD_PROOF_SIDECAR,
    RUNTIME_STORAGE_KEY_AUDIT_SCHEMA,
)
from invarlock.evidence_pack_transformation_contract import (
    _canonical_json_sha256,
    _canonical_transformation_parameters,
    _canonical_transformation_scope,
    _expected_literal_transformation,
    _is_clean_transformation_scenario,
    _is_exact_json_value,
    _require_runtime_reload_proof,
    _runtime_load_diagnostics_errors,
    _runtime_reload_identity_errors,
    _runtime_reload_proof_errors,
    _runtime_storage_key_audit_errors,
    _transformation_identity_errors,
)
from invarlock.evidence_pack_transformation_replay import _transformation_replay_errors
from tests.evidence_packs._support_transformation_pack import _make_pack

_DIGEST = "sha256:" + "a" * 64


@pytest.mark.parametrize(
    ("edit_type", "parameters", "diagnostic"),
    [
        (None, {}, "has no verifier-grade generated-lane contract"),
        ("quant_rtn", [], "parameters must be a JSON object"),
        (
            "quant_rtn",
            {"bits": 4},
            "quant_rtn parameters must contain exactly ['bits', 'group_size']",
        ),
        (
            "quant_rtn",
            {"bits": True, "group_size": 2},
            "quant_rtn.bits must be a positive integer",
        ),
        (
            "quant_rtn",
            {"bits": 9, "group_size": 2},
            "quant_rtn.bits must be in [2, 8]",
        ),
        (
            "quant_rtn",
            {"bits": 4, "group_size": 0},
            "quant_rtn.group_size must be a positive integer",
        ),
        (
            "synthetic_lowrank_delta",
            {"rank": 1},
            "synthetic_lowrank_delta parameters must contain exactly ['rank', 'scale']",
        ),
        (
            "synthetic_lowrank_delta",
            {"rank": False, "scale": 1.0},
            "synthetic_lowrank_delta.rank must be a positive integer",
        ),
        (
            "synthetic_lowrank_delta",
            {"rank": 33, "scale": 1.0},
            "synthetic_lowrank_delta.rank must not exceed 32",
        ),
        (
            "synthetic_lowrank_delta",
            {"rank": 1, "scale": float("nan")},
            "synthetic_lowrank_delta.scale must be a finite positive number",
        ),
        (
            "synthetic_dense_update",
            {"iterations": 1},
            "synthetic_dense_update parameters must contain exactly ['iterations', 'step_size']",
        ),
        (
            "synthetic_dense_update",
            {"step_size": "0.1", "iterations": 1},
            "synthetic_dense_update.step_size must be a finite positive number",
        ),
        (
            "synthetic_dense_update",
            {"step_size": 0.1, "iterations": False},
            "synthetic_dense_update.iterations must be a positive integer",
        ),
        (
            "synthetic_dense_update",
            {"step_size": 0.1, "iterations": 17},
            "synthetic_dense_update.iterations must not exceed 16",
        ),
    ],
)
def test_canonical_parameter_failures_are_exact(
    edit_type: object,
    parameters: object,
    diagnostic: str,
) -> None:
    assert _canonical_transformation_parameters(edit_type, parameters) == (
        None,
        diagnostic,
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, (None, "transformation scope must be a string")),
        ("", (None, "transformation scope syntax is invalid")),
        ("ffn@@layer=0", (None, "transformation scope syntax is invalid")),
        ("vision", (None, "transformation scope base is invalid")),
        ("ffn@", (None, "transformation scope qualifier is invalid")),
        ("ffn@layer", (None, "transformation scope qualifier is invalid")),
        ("ffn@unknown=1", (None, "transformation scope qualifier is invalid")),
        ("ffn@layer=-1", (None, "transformation scope qualifier is invalid")),
        ("ffn@layers=0", (None, "layers qualifier must be greater than zero")),
        (
            "ffn@layers=1,layer=1",
            (None, "layer qualifier must be smaller than the layers qualifier"),
        ),
        (" ATTN ", ("attn", None)),
        ("attn@layer=0", ("attn@layer=0", None)),
        ("ffn@layers=2", ("ffn@layers=2", None)),
        ("ffn@layers=2,layer=1", ("ffn@layers=2,layer=1", None)),
    ],
)
def test_scope_contract_matrix(
    value: object, expected: tuple[str | None, str | None]
) -> None:
    assert _canonical_transformation_scope(value) == expected


@pytest.mark.parametrize(
    ("edit_spec", "diagnostic"),
    [
        (None, "generated transformation scenario edit_spec is missing"),
        ("fp8_quant:4:2:ffn", "generated transformation scenario is unsupported"),
        (
            "quant_rtn:clean:extra",
            "clean generated transformation edit_spec is invalid",
        ),
        ("quant_rtn:4:2", "generated transformation edit_spec has the wrong arity"),
        (
            "quant_rtn:nope:2:ffn",
            "generated transformation edit_spec has invalid parameters",
        ),
        ("quant_rtn:9:2:ffn", "quant_rtn.bits must be in [2, 8]"),
        (
            "quant_rtn:4:2:FFN",
            "generated transformation scenario scope is not canonical",
        ),
    ],
)
def test_literal_scenario_contract_matrix(
    edit_spec: object,
    diagnostic: str,
) -> None:
    result = _expected_literal_transformation({"generation": {"edit_spec": edit_spec}})
    assert result[0] is None
    assert result[2] == diagnostic


def test_exact_json_and_clean_scenario_helpers_reject_type_aliases() -> None:
    assert _canonical_json_sha256({"bad": {1, 2}}) is None
    assert not _is_exact_json_value(1, 1.0)
    assert not _is_exact_json_value({"a": 1}, {"b": 1})
    assert not _is_exact_json_value([1], [1, 2])
    assert _is_exact_json_value({"a": [1, True]}, {"a": [1, True]})
    assert not _is_clean_transformation_scenario(None)
    assert not _is_clean_transformation_scenario({"generation": {"edit_spec": []}})
    assert _is_clean_transformation_scenario(
        {"generation": {"edit_spec": "quant_rtn:clean"}}
    )


@pytest.mark.parametrize(
    ("helper", "value", "diagnostic"),
    [
        (
            _transformation_identity_errors,
            None,
            "transformation replay artifact must be an object",
        ),
        (
            _transformation_identity_errors,
            {"kind": "", "sha256": _DIGEST},
            "transformation replay artifact.kind must be a non-empty string",
        ),
        (
            _transformation_identity_errors,
            {"kind": "local_checkpoint_tree", "sha256": "bad"},
            "transformation replay artifact.sha256 must be a sha256 digest",
        ),
        (
            _runtime_reload_identity_errors,
            {"kind": "local_checkpoint_tree", "sha256": _DIGEST, "extra": True},
            "runtime reload proof artifact must be a local identity",
        ),
        (
            _runtime_reload_identity_errors,
            {"kind": "remote", "sha256": _DIGEST},
            "runtime reload proof artifact.kind must be local_checkpoint_tree",
        ),
        (
            _runtime_reload_identity_errors,
            {"kind": "local_checkpoint_tree", "sha256": "bad"},
            "runtime reload proof artifact.sha256 must be a sha256 digest",
        ),
    ],
)
def test_identity_contract_matrix(helper, value: object, diagnostic: str) -> None:  # noqa: ANN001
    assert helper(prefix="", label="artifact", value=value) == [diagnostic]


def _empty_diagnostic() -> dict[str, list[object]]:
    return {
        "unexpected_keys": [],
        "missing_keys": [],
        "mismatched_keys": [],
        "error_msgs": [],
    }


def _audit() -> dict[str, object]:
    return {
        "artifact_storage_key_count": 1,
        "artifact_storage_keys_sha256": _DIGEST,
        "model_state_key_count": 1,
        "model_state_keys_sha256": _DIGEST,
        "unexpected_storage_keys": [],
    }


def test_runtime_diagnostic_and_storage_audit_adversarial_matrix() -> None:
    assert _runtime_load_diagnostics_errors(prefix="x: ", value={}) == [
        "x: runtime reload proof load diagnostics are invalid"
    ]
    assert _runtime_load_diagnostics_errors(
        prefix="x: ",
        value={"schema": RUNTIME_LOAD_DIAGNOSTICS_SCHEMA, "reloads": []},
    ) == ["x: runtime reload proof load diagnostics must bind exactly two reloads"]
    errors = _runtime_load_diagnostics_errors(
        prefix="x: ",
        value={
            "schema": RUNTIME_LOAD_DIAGNOSTICS_SCHEMA,
            "reloads": [{}, {**_empty_diagnostic(), "missing_keys": ["x.weight"]}],
        },
    )
    assert errors == [
        "x: runtime reload proof load diagnostics reload 0 is invalid",
        "x: runtime reload proof load diagnostics reload 1 reports missing_keys",
    ]

    assert _runtime_storage_key_audit_errors(prefix="x: ", value={}) == [
        "x: runtime reload proof storage-key audit is invalid"
    ]
    assert _runtime_storage_key_audit_errors(
        prefix="x: ",
        value={"schema": RUNTIME_STORAGE_KEY_AUDIT_SCHEMA, "reloads": []},
    ) == ["x: runtime reload proof storage-key audit must bind exactly two reloads"]
    invalid_shape_errors = _runtime_storage_key_audit_errors(
        prefix="x: ",
        value={
            "schema": RUNTIME_STORAGE_KEY_AUDIT_SCHEMA,
            "reloads": [{}, _audit()],
        },
    )
    assert invalid_shape_errors == [
        "x: runtime reload proof storage-key audit reload 0 is invalid"
    ]
    bad = {
        **_audit(),
        "artifact_storage_key_count": True,
        "model_state_key_count": 0,
        "artifact_storage_keys_sha256": "bad",
        "unexpected_storage_keys": ["injected.weight"],
    }
    disagree = {**_audit(), "artifact_storage_key_count": 2}
    errors = _runtime_storage_key_audit_errors(
        prefix="x: ",
        value={
            "schema": RUNTIME_STORAGE_KEY_AUDIT_SCHEMA,
            "reloads": [bad, disagree],
        },
    )
    assert (
        "x: runtime reload proof storage-key audit reload 0 artifact_storage_key_count is invalid"
        in errors
    )
    assert (
        "x: runtime reload proof storage-key audit reload 0 model_state_key_count is invalid"
        in errors
    )
    assert (
        "x: runtime reload proof storage-key audit reload 0 artifact_storage_keys_sha256 is invalid"
        in errors
    )
    assert (
        "x: runtime reload proof storage-key audit reload 0 has unexpected storage keys"
        in errors
    )
    assert (
        "x: runtime reload proof storage-key audit reload 1 has more artifact storage keys than model state keys"
        in errors
    )
    assert "x: runtime reload proof storage-key audits disagree" in errors


def test_runtime_reload_proof_file_and_semantic_matrix(tmp_path: Path) -> None:
    pack, report_dir, replay = _make_pack(tmp_path)
    report = json.loads(
        (report_dir / "evaluation.report.json").read_text(encoding="utf-8")
    )
    proof_path = report_dir / RUNTIME_RELOAD_PROOF_SIDECAR
    proof = json.loads(proof_path.read_text(encoding="utf-8"))

    proof_path.unlink()
    assert _require_runtime_reload_proof(
        scenario_id="quant",
        report_dir=report_dir,
        report=report,
        replay=replay,
        expected_edit_type="quant_rtn",
    ) == ["quant: runtime reload proof sidecar missing"]
    proof_path.write_text("[]", encoding="utf-8")
    assert _require_runtime_reload_proof(
        scenario_id="quant",
        report_dir=report_dir,
        report=report,
        replay=replay,
        expected_edit_type="quant_rtn",
    ) == ["quant: runtime reload proof sidecar is invalid"]

    bad = dict(proof)
    bad.update(
        {
            "extra": True,
            "schema": "wrong",
            "ok": False,
            "replay_schema": "wrong",
            "edit_type": "wrong",
            "artifact_identity": {"kind": "remote", "sha256": "bad"},
            "replay_artifact_identity": None,
            "prompt_sha256": "bad",
            "device": "tpu",
            "input_device": "cuda:x",
            "reload_runs": 1,
            "token_ids_shape": [True],
            "logits_shape": [],
            "all_logits_finite": False,
            "repeat_deterministic": False,
            "load_diagnostics": {},
            "storage_key_audit": {},
        }
    )
    errors = _runtime_reload_proof_errors(
        scenario_id="quant",
        report={},
        replay={
            "schema": "replay",
            "edit_type": "quant_rtn",
            "artifact_identity": None,
        },
        proof=bad,
        expected_edit_type="quant_rtn",
    )
    expected_fragments = (
        "has unbound fields",
        "unrecognized schema",
        "did not pass",
        "replay schema mismatch",
        "replay edit type mismatch",
        "scenario edit type mismatch",
        "prompt_sha256 must be a sha256 digest",
        "device is invalid",
        "input device is invalid",
        "exactly two reloads",
        "token_ids_shape is invalid",
        "logits_shape is invalid",
        "finite logits evidence missing",
        "determinism evidence missing",
    )
    for fragment in expected_fragments:
        assert any(fragment in error for error in errors), fragment

    cpu_mismatch = {**proof, "device": "cpu", "input_device": "cuda:0"}
    assert any(
        "input device mismatches CPU run" in error
        for error in _runtime_reload_proof_errors(
            scenario_id="quant",
            report=report,
            replay=replay,
            proof=cpu_mismatch,
            expected_edit_type="quant_rtn",
        )
    )
    cuda_mismatch = {**proof, "device": "cuda", "input_device": "cpu"}
    assert any(
        "input device mismatches CUDA run" in error
        for error in _runtime_reload_proof_errors(
            scenario_id="quant",
            report=report,
            replay=replay,
            proof=cuda_mismatch,
            expected_edit_type="quant_rtn",
        )
    )


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


@pytest.mark.parametrize(
    ("mutation", "fragment"),
    [
        (lambda payload: payload.update(extra=True), "has unbound fields"),
        (
            lambda payload: payload.update(algorithm="forged"),
            "algorithm mismatch",
        ),
        (
            lambda payload: payload.update(selected_tensors=0),
            "selected no tensors",
        ),
        (
            lambda payload: payload.update(total_tensors=0),
            "total tensor count invalid",
        ),
        (
            lambda payload: payload.update(total_params=0),
            "total parameter count invalid",
        ),
        (
            lambda payload: payload.update(support_files_checked=0),
            "checked no support files",
        ),
        (
            lambda payload: payload["actual_changes"].update(value_changed_tensors=2),
            "value_changed_tensors exceeds targets",
        ),
        (
            lambda payload: payload["actual_changes"].update(byte_changed_params=5),
            "byte_changed_params exceeds targets",
        ),
        (
            lambda payload: payload.update(issues=["generator drift"]),
            "issues must be empty when ok",
        ),
    ],
)
def test_replay_adversarial_claim_matrix(
    tmp_path: Path,
    mutation,  # noqa: ANN001
    fragment: str,
) -> None:
    arguments = _replay_arguments(tmp_path)
    mutation(arguments["payload"])

    errors = _transformation_replay_errors(**arguments)

    assert any(fragment in error for error in errors), errors


def test_replay_numeric_canonicalization_and_scenario_error_paths(
    tmp_path: Path,
) -> None:
    arguments = _replay_arguments(tmp_path, edit_type="synthetic_lowrank_delta")
    payload = arguments["payload"]
    payload["parameters"]["scale"] = 1
    errors = _transformation_replay_errors(**arguments)
    assert any("parameters are not canonical" in error for error in errors)

    arguments = _replay_arguments(tmp_path / "malformed")
    arguments["spec"] = {"generation": {"edit_spec": "quant_rtn:bad"}}
    errors = _transformation_replay_errors(**arguments)
    assert any("wrong arity" in error for error in errors)

    arguments = _replay_arguments(tmp_path / "no-spec")
    arguments["spec"] = None
    assert not any(
        "scenario mismatch" in error
        for error in _transformation_replay_errors(**arguments)
    )


def test_replay_crosslink_file_failures_are_real_filesystem_failures(
    tmp_path: Path,
) -> None:
    arguments = _replay_arguments(tmp_path)
    arguments["report_dir"] = None
    assert any(
        "cannot verify report-sidecar cross-links" in error
        for error in _transformation_replay_errors(**arguments)
    )

    arguments = _replay_arguments(tmp_path / "metadata-unreadable")
    report_dir = arguments["report_dir"]
    metadata_path = report_dir / "edit_metadata.json"
    metadata_path.chmod(0)
    try:
        errors = _transformation_replay_errors(**arguments)
    finally:
        metadata_path.chmod(0o600)
    assert any("edit metadata is unreadable" in error for error in errors)

    arguments = _replay_arguments(tmp_path / "receipt-unreadable")
    report_dir = arguments["report_dir"]
    receipt_path = report_dir / "transformation_materialization.json"
    receipt_path.chmod(0)
    try:
        errors = _transformation_replay_errors(**arguments)
    finally:
        receipt_path.chmod(0o600)
    assert any("materialization receipt is unreadable" in error for error in errors)

    arguments = _replay_arguments(tmp_path / "receipt-invalid")
    report_dir = arguments["report_dir"]
    receipt_path = report_dir / "transformation_materialization.json"
    receipt_path.write_text("[]", encoding="utf-8")
    errors = _transformation_replay_errors(**arguments)
    assert any("receipt sidecar is invalid" in error for error in errors)
    assert any("receipt digest mismatch" in error for error in errors)

    arguments = _replay_arguments(tmp_path / "metadata-symlink")
    report_dir = arguments["report_dir"]
    metadata_path = report_dir / "edit_metadata.json"
    external_metadata = tmp_path / "external-edit-metadata.json"
    external_metadata.write_bytes(metadata_path.read_bytes())
    metadata_path.unlink()
    metadata_path.symlink_to(external_metadata)
    errors = _transformation_replay_errors(**arguments)
    assert any("edit metadata sidecar missing" in error for error in errors)

    arguments = _replay_arguments(tmp_path / "invalid-transform")
    arguments["payload"]["edit_type"] = "unsupported"
    errors = _transformation_replay_errors(**arguments)
    assert any("canonical parameters invalid" in error for error in errors)


def test_replay_clean_selection_and_nonclean_selection_boundaries(
    tmp_path: Path,
) -> None:
    arguments = _replay_arguments(tmp_path)
    payload = arguments["payload"]
    payload["selection_receipt"] = {}
    payload["selection_receipt_sha256"] = _DIGEST
    errors = _transformation_replay_errors(**arguments)
    assert any(
        "non-clean transformation replay must not carry" in error for error in errors
    )

    arguments = _replay_arguments(tmp_path / "clean")
    arguments["spec"] = {
        "generation": {"edit_spec": "quant_rtn:clean"},
        "artifact_class": "validation_subject_checkpoint",
    }
    arguments["pack_dir"] = None
    errors = _transformation_replay_errors(**arguments)
    assert any(
        "selection cannot be verified without pack metadata" in error
        for error in errors
    )
