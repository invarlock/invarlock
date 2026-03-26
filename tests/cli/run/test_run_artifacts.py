from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import invarlock.cli.run_artifacts as run_artifacts


def test_persist_ref_masks_from_dict_and_object_preserves_generated_at(
    tmp_path: Path,
) -> None:
    payload = {"keep": [1, 2], "meta": {"generated_at": "existing-ts"}}
    core_report = {"edit": {"artifacts": {"mask_payload": payload}}}

    mask_path = run_artifacts.persist_ref_masks(core_report, tmp_path)
    assert mask_path == tmp_path / "artifacts" / "edit_masks" / "masks.json"
    written = json.loads(mask_path.read_text(encoding="utf-8"))
    assert written == payload
    assert mask_path.read_text(encoding="utf-8").endswith("\n")

    obj = SimpleNamespace(edit={"artifacts": {"mask_payload": {"keep": [3]}}})
    object_mask_path = run_artifacts.persist_ref_masks(obj, tmp_path)
    object_written = json.loads(object_mask_path.read_text(encoding="utf-8"))
    assert object_written["keep"] == [3]
    assert "generated_at" in object_written["meta"]


def test_persist_ref_masks_returns_none_for_missing_sections(tmp_path: Path) -> None:
    assert run_artifacts.persist_ref_masks({}, tmp_path) is None
    assert run_artifacts.persist_ref_masks({"edit": []}, tmp_path) is None
    assert run_artifacts.persist_ref_masks({"edit": {}}, tmp_path) is None
    assert (
        run_artifacts.persist_ref_masks({"edit": {"artifacts": []}}, tmp_path) is None
    )
    assert (
        run_artifacts.persist_ref_masks({"edit": {"artifacts": {}}}, tmp_path) is None
    )
    assert (
        run_artifacts.persist_ref_masks(
            {"edit": {"artifacts": {"mask_payload": {}}}},
            tmp_path,
        )
        is None
    )
    assert (
        run_artifacts.persist_ref_masks(
            {"edit": {"artifacts": {"mask_payload": []}}},
            tmp_path,
        )
        is None
    )


def test_resolve_exit_code_covers_all_branches() -> None:
    class ConfigError(Exception): ...

    class ValidationError(Exception): ...

    class DataError(Exception): ...

    class InvarlockError(Exception): ...

    assert (
        run_artifacts.resolve_exit_code(
            ConfigError(),
            profile="dev",
            config_error_cls=ConfigError,
            validation_error_cls=ValidationError,
            data_error_cls=DataError,
            invarlock_error_cls=InvarlockError,
        )
        == 2
    )
    assert (
        run_artifacts.resolve_exit_code(
            ValidationError(),
            profile="dev",
            config_error_cls=ConfigError,
            validation_error_cls=ValidationError,
            data_error_cls=DataError,
            invarlock_error_cls=InvarlockError,
        )
        == 2
    )
    assert (
        run_artifacts.resolve_exit_code(
            DataError(),
            profile="dev",
            config_error_cls=ConfigError,
            validation_error_cls=ValidationError,
            data_error_cls=DataError,
            invarlock_error_cls=InvarlockError,
        )
        == 2
    )
    assert (
        run_artifacts.resolve_exit_code(
            ValueError("Invalid RunReport payload"),
            profile="dev",
            config_error_cls=ConfigError,
            validation_error_cls=ValidationError,
            data_error_cls=DataError,
            invarlock_error_cls=InvarlockError,
        )
        == 2
    )
    assert (
        run_artifacts.resolve_exit_code(
            InvarlockError(),
            profile="ci",
            config_error_cls=ConfigError,
            validation_error_cls=ValidationError,
            data_error_cls=DataError,
            invarlock_error_cls=InvarlockError,
        )
        == 3
    )
    assert (
        run_artifacts.resolve_exit_code(
            RuntimeError("boom"),
            profile=None,
            config_error_cls=ConfigError,
            validation_error_cls=ValidationError,
            data_error_cls=DataError,
            invarlock_error_cls=InvarlockError,
        )
        == 1
    )

    class _BadProfile:
        def strip(self) -> str:
            raise RuntimeError("boom")

    assert (
        run_artifacts.resolve_exit_code(
            RuntimeError("boom"),
            profile=_BadProfile(),  # type: ignore[arg-type]
            config_error_cls=ConfigError,
            validation_error_cls=ValidationError,
            data_error_cls=DataError,
            invarlock_error_cls=InvarlockError,
        )
        == 1
    )
