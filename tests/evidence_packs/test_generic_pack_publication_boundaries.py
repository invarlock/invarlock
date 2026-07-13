from __future__ import annotations

from pathlib import Path

import pytest

from scripts.evidence_packs.python.copy_report_owned_inputs import (
    OwnedInputError,
    copy_owned_inputs,
)
from scripts.evidence_packs.python.publication_privacy_check import (
    publication_privacy_errors,
)


def test_runtime_inputs_use_exact_flat_inventory(tmp_path: Path) -> None:
    source = tmp_path / "runtime_inputs"
    source.mkdir()
    (source / "baseline_report.json").write_text("{}", encoding="utf-8")
    (source / "calibrated_preset_model-a.yaml").write_text(
        "dataset: fixture\n", encoding="utf-8"
    )
    destination = tmp_path / "packed"
    copy_owned_inputs(
        source,
        destination,
        kind="runtime_inputs",
        report_relative_path="model/clean/run_1",
    )
    assert sorted(path.name for path in destination.iterdir()) == [
        "baseline_report.json",
        "calibrated_preset_model-a.yaml",
    ]


def test_runtime_inputs_reject_extra_files_and_directories(tmp_path: Path) -> None:
    source = tmp_path / "runtime_inputs"
    source.mkdir()
    (source / "secret.json").write_text("{}", encoding="utf-8")
    with pytest.raises(OwnedInputError, match="not scenario-owned"):
        copy_owned_inputs(
            source,
            tmp_path / "packed",
            kind="runtime_inputs",
            report_relative_path="model/clean/run_1",
        )

    (source / "secret.json").unlink()
    (source / "nested").mkdir()
    with pytest.raises(OwnedInputError, match="unowned directory"):
        copy_owned_inputs(
            source,
            tmp_path / "packed",
            kind="runtime_inputs",
            report_relative_path="model/clean/run_1",
        )


def test_source_and_edited_inputs_are_error_scenario_owned(tmp_path: Path) -> None:
    source = tmp_path / "source"
    run = source / "000000"
    run.mkdir(parents=True)
    (run / "report.json").write_text("{}", encoding="utf-8")
    destination = tmp_path / "packed"
    copy_owned_inputs(
        source,
        destination,
        kind="source",
        report_relative_path="model/errors/nan_injection",
    )
    assert (destination / "000000/report.json").is_file()

    with pytest.raises(OwnedInputError, match="only owned by error scenarios"):
        copy_owned_inputs(
            source,
            tmp_path / "not-error",
            kind="source",
            report_relative_path="model/clean/run_1",
        )


@pytest.mark.parametrize(
    "report_relative_path", ("../errors/injected", "/errors/injected")
)
def test_error_owned_inputs_reject_absolute_and_traversal_report_paths(
    tmp_path: Path, report_relative_path: str
) -> None:
    source = tmp_path / "source" / "000000"
    source.mkdir(parents=True)
    (source / "report.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(OwnedInputError, match="report-relative path is unsafe"):
        copy_owned_inputs(
            source.parent,
            tmp_path / "pack",
            kind="source",
            report_relative_path=report_relative_path,
        )


def test_error_owned_inputs_reject_symlinked_source_parent(tmp_path: Path) -> None:
    external = tmp_path / "external" / "source" / "000000"
    external.mkdir(parents=True)
    (external / "report.json").write_text("{}\n", encoding="utf-8")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "linked").symlink_to(external.parent, target_is_directory=True)

    with pytest.raises(OwnedInputError, match="symlink component"):
        copy_owned_inputs(
            run_dir / "linked",
            tmp_path / "pack",
            kind="source",
            report_relative_path="model/errors/nan_injection",
        )


def test_prepublication_privacy_gate_rejects_host_paths(tmp_path: Path) -> None:
    public_file = tmp_path / "reports/model/scenario/evaluation.report.json"
    public_file.parent.mkdir(parents=True)
    public_file.write_text(
        '{"checkpoint":"/Users/operator/private/model"}\n', encoding="utf-8"
    )
    errors = publication_privacy_errors(tmp_path)
    assert any("macos_user_home_path" in error for error in errors)


@pytest.mark.parametrize(
    ("relative_path", "contents", "expected_error"),
    (
        (
            "manifest.json",
            '{"dataset_provider":{"file":"/srv/private/input.jsonl"}}\n',
            "absolute_host_path",
        ),
        (
            "reports/model/scenario/evaluation.html",
            "<p>C:\\private\\input.jsonl</p>\n",
            "windows_host_path",
        ),
    ),
)
def test_prepublication_privacy_gate_scans_final_manifest_and_html(
    tmp_path: Path,
    relative_path: str,
    contents: str,
    expected_error: str,
) -> None:
    public_file = tmp_path / relative_path
    public_file.parent.mkdir(parents=True, exist_ok=True)
    public_file.write_text(contents, encoding="utf-8")

    errors = publication_privacy_errors(tmp_path)

    assert any(expected_error in error for error in errors)
