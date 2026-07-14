from __future__ import annotations

from pathlib import Path

import pytest

from invarlock import clean_pruning_selection_runtime as pruning
from invarlock import clean_selection_runtime as selection
from invarlock.clean_selection import artifacts


def _selection_context(tmp_path: Path) -> selection.CleanSelectionEvaluationContext:
    return selection.CleanSelectionEvaluationContext(
        selection_config_path=tmp_path / "config.json",
        selection_config_bytes=b"{}",
        selection_config={},
        execution_receipt_path=tmp_path / "receipt.json",
        execution_receipt_bytes=b"{}",
        execution_receipt={},
        execution_receipt_sha256="sha256:" + "a" * 64,
        replay_path=tmp_path / "replay.json",
        replay_bytes=b"{}",
        replay={},
        runtime_proof_path=tmp_path / "runtime.json",
        runtime_proof_bytes=b"{}",
        runtime_proof={},
        original_model_key="model",
        candidate_id="candidate",
        transformation={"edit_type": "quant_rtn"},
        baseline_identity={
            "kind": "local_checkpoint_tree",
            "sha256": "sha256:" + "b" * 64,
        },
        repeat_index=0,
    )


def _pruning_context(
    tmp_path: Path,
) -> pruning.CleanPruningSelectionEvaluationContext:
    return pruning.CleanPruningSelectionEvaluationContext(
        selection_config_path=tmp_path / "config.json",
        selection_config_bytes=b"{}",
        selection_config={},
        execution_receipt_path=tmp_path / "receipt.json",
        execution_receipt_bytes=b"{}",
        execution_receipt={},
        execution_receipt_sha256="sha256:" + "a" * 64,
        replay_path=tmp_path / "replay.json",
        replay_bytes=b"{}",
        replay={},
        runtime_proof_path=tmp_path / "runtime.json",
        runtime_proof_bytes=b"{}",
        runtime_proof={},
        original_model_key="model",
        candidate_id="candidate",
        pruning={"edit_type": "magnitude_prune"},
        baseline_identity={
            "kind": "local_checkpoint_tree",
            "sha256": "sha256:" + "b" * 64,
        },
        repeat_index=0,
    )


@pytest.mark.parametrize(
    ("runtime", "error"),
    [
        (selection, selection.CleanSelectionRuntimeError),
        (pruning, pruning.CleanPruningSelectionRuntimeError),
    ],
)
def test_runtime_helpers_reject_nonobjects_and_changed_snapshots(
    runtime: object,
    error: type[ValueError],
    tmp_path: Path,
) -> None:
    with pytest.raises(error, match="must be an object"):
        runtime._mapping([], label="value")  # type: ignore[attr-defined]
    path = tmp_path / "snapshot.json"
    path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(error, match="changed after evaluator startup"):
        runtime._snapshot_unchanged(  # type: ignore[attr-defined]
            path, expected=b'{"old":true}\n', label="snapshot"
        )


@pytest.mark.parametrize("runtime", [selection, pruning])
def test_runtime_snapshot_and_atomic_write_success_paths(
    runtime: object, tmp_path: Path
) -> None:
    snapshot = tmp_path / "snapshot.json"
    snapshot.write_text("{}\n", encoding="utf-8")
    assert (
        runtime._snapshot_unchanged(  # type: ignore[attr-defined]
            snapshot, expected=b"{}\n", label="snapshot"
        )
        == {}
    )

    output = tmp_path / "output.json"
    runtime._atomic_write_json(output, {"ok": True})  # type: ignore[attr-defined]
    assert output.read_text(encoding="utf-8") == '{\n  "ok": true\n}\n'


@pytest.mark.parametrize(
    ("runtime", "error"),
    [
        (selection, selection.CleanSelectionRuntimeError),
        (pruning, pruning.CleanPruningSelectionRuntimeError),
    ],
)
def test_runtime_checkpoint_identity_rejects_missing_unreadable_and_mismatch(
    runtime: object,
    error: type[ValueError],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    with pytest.raises(error, match="regular checkpoint directory"):
        runtime._assert_checkpoint_identity(  # type: ignore[attr-defined]
            tmp_path / "missing", expected={"sha256": "x"}, label="subject"
        )
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    monkeypatch.setattr(
        runtime,
        "checkpoint_tree_sha256",
        lambda _path: (_ for _ in ()).throw(ValueError("unreadable")),
    )
    with pytest.raises(error, match="identity is unavailable"):
        runtime._assert_checkpoint_identity(  # type: ignore[attr-defined]
            checkpoint, expected={"sha256": "x"}, label="subject"
        )
    monkeypatch.setattr(runtime, "checkpoint_tree_sha256", lambda _path: "observed")
    with pytest.raises(error, match="does not match"):
        runtime._assert_checkpoint_identity(  # type: ignore[attr-defined]
            checkpoint, expected={"sha256": "expected"}, label="subject"
        )
    runtime._assert_checkpoint_identity(  # type: ignore[attr-defined]
        checkpoint, expected={"sha256": "observed"}, label="subject"
    )


@pytest.mark.parametrize(
    ("runtime", "error", "message"),
    [
        (
            selection,
            selection.CleanSelectionRuntimeError,
            "candidate evaluation report",
        ),
        (
            pruning,
            pruning.CleanPruningSelectionRuntimeError,
            "pruning candidate evaluation report",
        ),
    ],
)
def test_runtime_atomic_report_write_translates_os_failures(
    runtime: object,
    error: type[ValueError],
    message: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        runtime.os,
        "replace",
        lambda *_args: (_ for _ in ()).throw(OSError("replace denied")),
    )
    with pytest.raises(error, match=message):
        runtime._atomic_write_json(tmp_path / "report.json", {"ok": True})  # type: ignore[attr-defined]
    assert not list(tmp_path.glob(".report.json.*.tmp"))


def _patch_selection_finalizer(
    monkeypatch: pytest.MonkeyPatch, report: dict[str, object]
) -> None:
    monkeypatch.setattr(selection, "_snapshot_unchanged", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        selection,
        "strict_json_object_snapshot",
        lambda *_args, **_kwargs: (b"{}", report),
    )
    monkeypatch.setattr(
        selection,
        "build_evaluator_execution_provenance",
        lambda **_kwargs: {"native": True},
    )
    monkeypatch.setattr(
        selection, "build_candidate_report_binding", lambda **_kwargs: {"bound": True}
    )
    monkeypatch.setattr(selection, "_atomic_write_json", lambda *_args, **_kwargs: None)


def test_clean_selection_finalizer_rejects_conflicting_existing_claims(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    context = _selection_context(tmp_path)
    report = {
        "run_id": "run",
        "meta": {},
        "provenance": {"clean_selection_execution": {"forged": True}},
    }
    _patch_selection_finalizer(monkeypatch, report)
    with pytest.raises(
        selection.CleanSelectionRuntimeError, match="incompatible evaluator"
    ):
        selection.finalize_clean_selection_evaluation_report(
            tmp_path / "report.json", context=context
        )

    report["provenance"] = {"clean_selection_execution": {"native": True}}
    report["clean_selection"] = {"forged": True}
    with pytest.raises(
        selection.CleanSelectionRuntimeError, match="incompatible selection"
    ):
        selection.finalize_clean_selection_evaluation_report(
            tmp_path / "report.json", context=context
        )

    report["clean_selection"] = {"bound": True}
    report["run_id"] = ""
    with pytest.raises(selection.CleanSelectionRuntimeError, match="run_id is invalid"):
        selection.finalize_clean_selection_evaluation_report(
            tmp_path / "report.json", context=context
        )


def _patch_pruning_finalizer(
    monkeypatch: pytest.MonkeyPatch, report: dict[str, object]
) -> None:
    monkeypatch.setattr(pruning, "_snapshot_unchanged", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        pruning,
        "strict_json_object_snapshot",
        lambda *_args, **_kwargs: (b"{}", report),
    )
    monkeypatch.setattr(
        pruning,
        "build_clean_pruning_evaluator_execution_provenance",
        lambda **_kwargs: {"native": True},
    )
    monkeypatch.setattr(
        pruning,
        "build_clean_pruning_candidate_report_binding",
        lambda **_kwargs: {"bound": True},
    )
    monkeypatch.setattr(pruning, "_atomic_write_json", lambda *_args, **_kwargs: None)


def test_clean_pruning_finalizer_rejects_conflicting_existing_claims(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    context = _pruning_context(tmp_path)
    report = {
        "run_id": "run",
        "meta": {},
        "provenance": {"clean_pruning_selection_execution": {"forged": True}},
    }
    _patch_pruning_finalizer(monkeypatch, report)
    with pytest.raises(
        pruning.CleanPruningSelectionRuntimeError,
        match="incompatible pruning evaluator",
    ):
        pruning.finalize_clean_pruning_selection_evaluation_report(
            tmp_path / "report.json", context=context
        )

    report["provenance"] = {"clean_pruning_selection_execution": {"native": True}}
    report["clean_pruning_selection"] = {"forged": True}
    with pytest.raises(
        pruning.CleanPruningSelectionRuntimeError,
        match="incompatible pruning selection",
    ):
        pruning.finalize_clean_pruning_selection_evaluation_report(
            tmp_path / "report.json", context=context
        )

    report["clean_pruning_selection"] = {"bound": True}
    report["run_id"] = ""
    with pytest.raises(
        pruning.CleanPruningSelectionRuntimeError, match="run_id is invalid"
    ):
        pruning.finalize_clean_pruning_selection_evaluation_report(
            tmp_path / "report.json", context=context
        )


def test_clean_selection_context_checks_repeat_and_optional_checkpoint_bindings(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = {"schedule": {"evaluation_repeats": 1}}
    receipt: dict[str, object] = {
        "baseline_identity": {
            "kind": "local_checkpoint_tree",
            "sha256": "sha256:" + "a" * 64,
        },
        "transformation": {"edit_type": "quant_rtn"},
        "original_model_key": "model",
        "candidate_id": "candidate",
    }

    def snapshot(path: Path, **_kwargs: object) -> tuple[bytes, dict[str, object]]:
        if path.name == "config.json":
            return b"config", config
        if path.name == "receipt.json":
            return b"receipt", receipt
        return b"{}", {}

    monkeypatch.setattr(selection, "strict_json_object_snapshot", snapshot)
    monkeypatch.setattr(
        selection,
        "validate_selection_execution_receipt",
        lambda *_args, **_kwargs: receipt,
    )
    monkeypatch.setattr(
        selection,
        "validate_candidate_replay_runtime",
        lambda **_kwargs: {
            "kind": "local_checkpoint_tree",
            "sha256": "sha256:" + "b" * 64,
        },
    )
    checked: list[str] = []
    monkeypatch.setattr(
        selection,
        "_assert_checkpoint_identity",
        lambda _path, **kwargs: checked.append(str(kwargs["label"])),
    )

    paths = {
        "selection_config_path": tmp_path / "config.json",
        "execution_receipt_path": tmp_path / "receipt.json",
        "replay_path": tmp_path / "replay.json",
        "runtime_proof_path": tmp_path / "runtime.json",
    }
    with pytest.raises(selection.CleanSelectionRuntimeError, match="repeat index"):
        selection.load_clean_selection_evaluation_context(**paths, repeat_index=-1)

    context = selection.load_clean_selection_evaluation_context(
        **paths,
        repeat_index=0,
        baseline_path=tmp_path,
        subject_path=tmp_path,
    )
    assert context.candidate_id == "candidate"
    assert checked == ["candidate baseline checkpoint", "candidate subject checkpoint"]

    receipt["baseline_identity"] = []
    with pytest.raises(selection.CleanSelectionRuntimeError, match="baseline identity"):
        selection.load_clean_selection_evaluation_context(**paths, repeat_index=0)
    receipt["baseline_identity"] = {"kind": "local_checkpoint_tree", "sha256": "x"}
    receipt["transformation"] = []
    with pytest.raises(selection.CleanSelectionRuntimeError, match="transformation"):
        selection.load_clean_selection_evaluation_context(**paths, repeat_index=0)
    receipt["transformation"] = {}
    receipt["candidate_id"] = None
    with pytest.raises(selection.CleanSelectionRuntimeError, match="receipt identity"):
        selection.load_clean_selection_evaluation_context(**paths, repeat_index=0)


def test_clean_pruning_context_checks_repeat_and_optional_checkpoint_bindings(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = {"schedule": {"evaluation_repeats": 1}}
    receipt: dict[str, object] = {
        "baseline_identity": {
            "kind": "local_checkpoint_tree",
            "sha256": "sha256:" + "a" * 64,
        },
        "pruning": {"edit_type": "magnitude_prune"},
        "original_model_key": "model",
        "candidate_id": "candidate",
    }

    def snapshot(path: Path, **_kwargs: object) -> tuple[bytes, dict[str, object]]:
        if path.name == "config.json":
            return b"config", config
        if path.name == "receipt.json":
            return b"receipt", receipt
        return b"{}", {}

    monkeypatch.setattr(pruning, "strict_json_object_snapshot", snapshot)
    monkeypatch.setattr(
        pruning,
        "validate_clean_pruning_execution_receipt",
        lambda *_args, **_kwargs: receipt,
    )
    monkeypatch.setattr(
        pruning,
        "validate_clean_pruning_candidate_replay_runtime",
        lambda **_kwargs: {
            "kind": "local_checkpoint_tree",
            "sha256": "sha256:" + "b" * 64,
        },
    )
    checked: list[str] = []
    monkeypatch.setattr(
        pruning,
        "_assert_checkpoint_identity",
        lambda _path, **kwargs: checked.append(str(kwargs["label"])),
    )
    paths = {
        "selection_config_path": tmp_path / "config.json",
        "execution_receipt_path": tmp_path / "receipt.json",
        "replay_path": tmp_path / "replay.json",
        "runtime_proof_path": tmp_path / "runtime.json",
    }
    with pytest.raises(pruning.CleanPruningSelectionRuntimeError, match="repeat index"):
        pruning.load_clean_pruning_selection_evaluation_context(
            **paths, repeat_index=True
        )
    context = pruning.load_clean_pruning_selection_evaluation_context(
        **paths,
        repeat_index=0,
        baseline_path=tmp_path,
        subject_path=tmp_path,
    )
    assert context.candidate_id == "candidate"
    assert checked == ["candidate baseline checkpoint", "candidate subject checkpoint"]

    receipt["candidate_id"] = None
    with pytest.raises(
        pruning.CleanPruningSelectionRuntimeError, match="receipt identity"
    ):
        pruning.load_clean_pruning_selection_evaluation_context(**paths, repeat_index=0)


def test_runtime_contexts_skip_optional_checkpoint_binding_when_absent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cases = [
        (
            selection,
            {
                "baseline_identity": {"kind": "tree", "sha256": "baseline"},
                "transformation": {"edit_type": "quant_rtn"},
                "original_model_key": "model",
                "candidate_id": "candidate",
            },
            "validate_selection_execution_receipt",
            "validate_candidate_replay_runtime",
            "load_clean_selection_evaluation_context",
        ),
        (
            pruning,
            {
                "baseline_identity": {"kind": "tree", "sha256": "baseline"},
                "pruning": {"edit_type": "magnitude_prune"},
                "original_model_key": "model",
                "candidate_id": "candidate",
            },
            "validate_clean_pruning_execution_receipt",
            "validate_clean_pruning_candidate_replay_runtime",
            "load_clean_pruning_selection_evaluation_context",
        ),
    ]
    for runtime, receipt, receipt_validator, replay_validator, loader_name in cases:
        config = {"schedule": {"evaluation_repeats": 1}}

        def snapshot(
            path: Path,
            *,
            _config: dict[str, object] = config,
            _receipt: dict[str, object] = receipt,
            **_kwargs: object,
        ) -> tuple[bytes, dict[str, object]]:
            if path.name == "config.json":
                return b"config", _config
            if path.name == "receipt.json":
                return b"receipt", _receipt
            return b"{}", {}

        monkeypatch.setattr(runtime, "strict_json_object_snapshot", snapshot)
        monkeypatch.setattr(
            runtime,
            receipt_validator,
            lambda *_args, _receipt=receipt, **_kwargs: _receipt,
        )
        monkeypatch.setattr(
            runtime,
            replay_validator,
            lambda **_kwargs: {"kind": "tree", "sha256": "artifact"},
        )
        monkeypatch.setattr(
            runtime,
            "_assert_checkpoint_identity",
            lambda *_args, **_kwargs: pytest.fail("optional checkpoint was inspected"),
        )
        loader = getattr(runtime, loader_name)
        context = loader(
            selection_config_path=tmp_path / "config.json",
            execution_receipt_path=tmp_path / "receipt.json",
            replay_path=tmp_path / "replay.json",
            runtime_proof_path=tmp_path / "runtime.json",
            repeat_index=0,
        )
        assert context.candidate_id == "candidate"


@pytest.mark.parametrize(
    ("runtime", "context_factory", "patcher", "finalizer"),
    [
        (
            selection,
            _selection_context,
            _patch_selection_finalizer,
            selection.finalize_clean_selection_evaluation_report,
        ),
        (
            pruning,
            _pruning_context,
            _patch_pruning_finalizer,
            pruning.finalize_clean_pruning_selection_evaluation_report,
        ),
    ],
)
def test_runtime_finalizers_create_missing_provenance_and_return_run_link(
    runtime: object,
    context_factory: object,
    patcher: object,
    finalizer: object,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    del runtime
    report = {"run_id": "run-1", "meta": {}}
    patcher(monkeypatch, report)  # type: ignore[operator]
    link = finalizer(  # type: ignore[operator]
        tmp_path / "report.json",
        context=context_factory(tmp_path),  # type: ignore[operator]
    )
    assert link["report_run_id"] == "run-1"


def test_clean_selection_runtime_diagnostics_reject_malformed_reload_records() -> None:
    diagnostics = {
        "schema": artifacts.RUNTIME_LOAD_DIAGNOSTICS_SCHEMA,
        "reloads": [{}, {}],
    }
    with pytest.raises(artifacts.CleanSelectionEvidenceError, match="reload 0"):
        artifacts._assert_clean_load_diagnostics(diagnostics)

    storage_audit = {
        "schema": artifacts.RUNTIME_STORAGE_KEY_AUDIT_SCHEMA,
        "reloads": [{}, {}],
    }
    with pytest.raises(artifacts.CleanSelectionEvidenceError, match="reload 0"):
        artifacts._assert_clean_storage_key_audit(storage_audit)
