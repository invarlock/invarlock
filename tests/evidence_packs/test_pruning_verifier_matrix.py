from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from invarlock.evidence_pack_pruning_validation import (
    _clean_pruning_selection_errors,
    _is_clean_pruning_scenario,
    _pruning_identity_errors,
    _pruning_replay_errors,
    _pruning_target_manifest_errors,
)
from tests.evidence_packs._support_clean_pruning_selection import (
    _final_clean_pruning_pack,
)
from tests.evidence_packs._support_pruning_replay_validation import (
    _metadata,
    _pruning_scenario,
    _replay_payload,
)

_ARTIFACT = {
    "kind": "local_checkpoint_tree",
    "sha256": "sha256:" + "a" * 64,
}
_BASELINE = {
    "kind": "local_checkpoint_tree",
    "sha256": "sha256:" + "b" * 64,
}


def _arguments() -> dict[str, object]:
    payload = _replay_payload(
        artifact_identity=_ARTIFACT,
        baseline_identity=_BASELINE,
    )
    return {
        "scenario_id": "prune",
        "report": {
            "meta": {"model_identity": _ARTIFACT},
            "baseline_ref": {"model_identity": _BASELINE},
        },
        "metadata": _metadata(target_manifest=payload["target_manifest"]),
        "payload": payload,
        "spec": _pruning_scenario("prune", "magnitude_prune:0.5:ffn"),
    }


def test_pruning_helper_contract_matrix() -> None:
    assert not _is_clean_pruning_scenario(None)
    assert not _is_clean_pruning_scenario({"generation": {"edit_spec": []}})
    assert _is_clean_pruning_scenario(
        {"generation": {"edit_spec": "magnitude_prune:clean"}}
    )
    assert _pruning_identity_errors(prefix="x: ", label="artifact", value=None) == [
        "x: pruning replay artifact must be an object"
    ]
    assert _pruning_identity_errors(
        prefix="x: ", label="artifact", value={"kind": "", "sha256": "bad"}
    ) == ["x: pruning replay artifact.kind must be a non-empty string"]
    assert _pruning_identity_errors(
        prefix="x: ",
        label="artifact",
        value={"kind": "local_checkpoint_tree", "sha256": "bad"},
    ) == ["x: pruning replay artifact.sha256 must be a sha256 digest"]


def test_pruning_target_manifest_adversarial_matrix() -> None:
    arguments = _arguments()
    payload = arguments["payload"]
    bad = deepcopy(payload)
    bad["target_manifest"] = None
    assert _pruning_target_manifest_errors(prefix="x: ", payload=bad) == [
        "x: pruning replay target_manifest must be an object"
    ]

    bad = deepcopy(payload)
    bad["target_manifest_sha256"] = "bad"
    errors = _pruning_target_manifest_errors(prefix="x: ", payload=bad)
    assert any("target_manifest_sha256 must be" in error for error in errors)

    bad = deepcopy(payload)
    bad["target_manifest"]["scope"] = "attn"
    errors = _pruning_target_manifest_errors(prefix="x: ", payload=bad)
    assert any("policy violation" in error for error in errors)
    assert any("target_manifest scope mismatch" in error for error in errors)

    bad = deepcopy(payload)
    bad["target_manifest"]["targets"] = []
    errors = _pruning_target_manifest_errors(prefix="x: ", payload=bad)
    assert any("targets must be a non-empty list" in error for error in errors)

    bad = deepcopy(payload)
    bad["target_manifest"]["targets"] = [
        None,
        {"name": "", "dtype": "", "shape": [True], "numel": False},
        {"name": "z", "dtype": "float32", "shape": [2, 2], "numel": 3},
        {"name": "z", "dtype": "float32", "shape": [1], "numel": 1},
        {"name": "y", "dtype": "float32", "shape": [], "numel": 1},
    ]
    bad["selected_tensors"] = 99
    bad["selected_params"] = 99
    errors = _pruning_target_manifest_errors(prefix="x: ", payload=bad)
    for fragment in (
        "targets[0] must be an object",
        "name must be a non-empty string",
        "dtype must be a non-empty string",
        "shape must be a non-empty integer list",
        "numel must be a positive int",
        "numel does not match shape",
        "name is duplicated",
        "selected_tensors does not match",
        "selected_params does not match",
    ):
        assert any(fragment in error for error in errors), fragment


@pytest.mark.parametrize(
    ("mutation", "fragment"),
    [
        (lambda args: args["payload"].update(target_sparsity=1.0), "must be in (0, 1)"),
        (
            lambda args: args["metadata"]["parameters"].update(target_sparsity=0.4),
            "target_sparsity metadata mismatch",
        ),
        (
            lambda args: args.update(
                spec={"generation": {"edit_spec": "magnitude_prune:bad:ffn"}}
            ),
            "scenario sparsity is invalid",
        ),
        (
            lambda args: args["payload"].update(selected_tensors=0),
            "selected no tensors",
        ),
        (
            lambda args: args["payload"].update(expected_pruned_params=0),
            "expected no pruned parameters",
        ),
        (
            lambda args: args["payload"].update(expected_changed_params=0),
            "made no effective parameter changes",
        ),
        (
            lambda args: args["payload"].update(observed_changed_params=0),
            "changed parameter count mismatch",
        ),
        (
            lambda args: args["payload"].update(expected_pruned_params=999),
            "expected pruned parameter count invalid",
        ),
        (
            lambda args: args["payload"].update(expected_changed_params=999),
            "expected changed parameter count invalid",
        ),
        (
            lambda args: args["payload"].update(support_files_checked=0),
            "checked no support files",
        ),
        (
            lambda args: args["metadata"].update(coverage=None),
            "metadata coverage must be an object",
        ),
        (
            lambda args: args["metadata"]["coverage"].update(total_params=0),
            "metadata total parameter count mismatch",
        ),
        (
            lambda args: args["payload"].update(total_params=9),
            "metadata total parameter count mismatch",
        ),
        (
            lambda args: args["metadata"]["coverage"].update(coverage_ratio=0.25),
            "metadata coverage.coverage_ratio mismatch",
        ),
        (lambda args: args["payload"].update(issues=["drift"]), "issues must be empty"),
    ],
)
def test_pruning_replay_adversarial_matrix(mutation, fragment: str) -> None:  # noqa: ANN001
    arguments = _arguments()
    mutation(arguments)
    errors = _pruning_replay_errors(**arguments)
    assert any(fragment in error for error in errors), (fragment, errors)


def test_pruning_replay_valid_control_and_missing_spec() -> None:
    arguments = _arguments()
    assert _pruning_replay_errors(**arguments) == []
    arguments["spec"] = None
    assert _pruning_replay_errors(**arguments) == []


def _clean_arguments(tmp_path: Path) -> dict[str, object]:
    pack, report_path, _, _ = _final_clean_pruning_pack(tmp_path)
    report_dir = report_path.parent
    return {
        "pack_dir": pack,
        "scenario_id": report_dir.parent.name,
        "report_path": report_path,
        "report_dir": report_dir,
        "report_model_name": report_dir.parent.parent.name,
        "report": json.loads(report_path.read_text(encoding="utf-8")),
        "metadata": json.loads(
            (report_dir / "edit_metadata.json").read_text(encoding="utf-8")
        ),
        "payload": json.loads(
            (report_dir / "pruning_replay.json").read_text(encoding="utf-8")
        ),
    }


def test_clean_pruning_cross_binding_adversarial_matrix(tmp_path: Path) -> None:
    arguments = _clean_arguments(tmp_path)
    assert _clean_pruning_selection_errors(**arguments) == []

    bad = deepcopy(arguments)
    bad["report_model_name"] = None
    assert _clean_pruning_selection_errors(**bad) == [
        f"{arguments['scenario_id']}: clean pruning report has no model path"
    ]

    bad = deepcopy(arguments)
    bad["report_model_name"] = "wrong_model"
    assert any(
        "no unique matching model entry" in error
        for error in _clean_pruning_selection_errors(**bad)
    )

    bad = deepcopy(arguments)
    bad["payload"]["scope"] = "attn"
    bad["payload"]["target_sparsity"] = 0.25
    bad["payload"]["artifact_identity"] = _BASELINE
    bad["payload"]["baseline_identity"] = _ARTIFACT
    bad["metadata"]["scope"] = "attn"
    bad["metadata"]["parameters"] = {"target_sparsity": 0.25}
    bad["report"] = {}
    errors = _clean_pruning_selection_errors(**bad)
    for fragment in (
        "selected scope mismatch",
        "selected sparsity mismatch",
        "metadata sparsity mismatch",
        "metadata scope mismatch",
        "selected artifact identity mismatch",
        "selected baseline identity mismatch",
        "report model identity mismatch",
        "report artifact identity mismatch",
        "report baseline identity mismatch",
        "final report changed during verification",
    ):
        assert any(fragment in error for error in errors), fragment


@pytest.mark.parametrize(
    ("relative", "fragment"),
    [
        ("evaluation.report.json", "final report is unavailable"),
        ("runtime.manifest.json", "runtime manifest is unavailable"),
        ("pruning_replay.json", "final replay is unavailable"),
        ("runtime_reload_proof.json", "final runtime proof is unavailable"),
    ],
)
def test_clean_pruning_missing_final_files(
    tmp_path: Path,
    relative: str,
    fragment: str,
) -> None:
    arguments = _clean_arguments(tmp_path)
    (arguments["report_dir"] / relative).unlink()
    errors = _clean_pruning_selection_errors(**arguments)
    assert any(fragment in error for error in errors), errors
