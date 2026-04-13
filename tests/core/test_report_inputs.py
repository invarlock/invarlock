from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from invarlock.core.report_inputs import (
    ReportInputError,
    load_report_input_json,
    resolve_report_input_path,
)


def test_resolve_report_input_path_accepts_explicit_file(tmp_path: Path) -> None:
    report = tmp_path / "custom-name.json"
    report.write_text("{}", encoding="utf-8")

    resolved = resolve_report_input_path(report)

    assert resolved == report.resolve()


def test_resolve_report_input_path_accepts_explicit_evaluation_report_file(
    tmp_path: Path,
) -> None:
    report = tmp_path / "evaluation.report.json"
    report.write_text("{}", encoding="utf-8")

    resolved = resolve_report_input_path(report)

    assert resolved == report.resolve()


def test_resolve_report_input_path_accepts_canonical_run_directory(
    tmp_path: Path,
) -> None:
    report = tmp_path / "report.json"
    report.write_text("{}", encoding="utf-8")

    resolved = resolve_report_input_path(tmp_path)

    assert resolved == report.resolve()


def test_resolve_report_input_path_accepts_canonical_evaluation_directory(
    tmp_path: Path,
) -> None:
    report = tmp_path / "evaluation.report.json"
    report.write_text("{}", encoding="utf-8")

    resolved = resolve_report_input_path(tmp_path)

    assert resolved == report.resolve()


def test_resolve_report_input_path_rejects_ambiguous_directory(tmp_path: Path) -> None:
    (tmp_path / "report.json").write_text("{}", encoding="utf-8")
    (tmp_path / "evaluation.report.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ReportInputError, match="Ambiguous report directory") as exc:
        resolve_report_input_path(tmp_path)

    assert "report.json and evaluation.report.json" in str(exc.value)


@pytest.mark.parametrize("expected_kind", ["run", "evaluation"])
def test_resolve_report_input_path_rejects_ambiguous_directory_for_expected_kind(
    tmp_path: Path, expected_kind: str
) -> None:
    (tmp_path / "report.json").write_text("{}", encoding="utf-8")
    (tmp_path / "evaluation.report.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ReportInputError, match="Ambiguous report directory"):
        resolve_report_input_path(tmp_path, expected_kind=expected_kind)  # type: ignore[arg-type]


def test_load_report_input_json_rejects_ambiguous_directory(tmp_path: Path) -> None:
    (tmp_path / "report.json").write_text("{}", encoding="utf-8")
    (tmp_path / "evaluation.report.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ReportInputError, match="Ambiguous report directory"):
        load_report_input_json(tmp_path)


def test_resolve_report_input_path_rejects_directory_without_canonical_file(
    tmp_path: Path,
) -> None:
    (tmp_path / "my_report.json").write_text("{}", encoding="utf-8")

    with pytest.raises(
        ReportInputError,
        match="does not contain a canonical report file",
    ):
        resolve_report_input_path(tmp_path)


def test_resolve_report_input_path_rejects_sidecar_report_like_directory(
    tmp_path: Path,
) -> None:
    (tmp_path / "other.json").write_text("{}", encoding="utf-8")
    (tmp_path / "subject_report.json").write_text("{}", encoding="utf-8")

    with pytest.raises(
        ReportInputError,
        match="does not contain a canonical report file",
    ):
        resolve_report_input_path(tmp_path)


def test_resolve_report_input_path_rejects_directory_when_explicit_file_required(
    tmp_path: Path,
) -> None:
    (tmp_path / "report.json").write_text("{}", encoding="utf-8")

    with pytest.raises(
        ReportInputError,
        match="explicit JSON file path, not a directory",
    ):
        resolve_report_input_path(tmp_path, allow_canonical_directory=False)


def test_load_report_input_json_rejects_invalid_json(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    report.write_text("{not-json", encoding="utf-8")

    with pytest.raises(ReportInputError, match="not valid JSON"):
        load_report_input_json(report)


def test_load_report_input_json_rejects_non_object_json(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    report.write_text(json.dumps(["not-an-object"]), encoding="utf-8")

    with pytest.raises(ReportInputError, match="must decode to a JSON object"):
        load_report_input_json(report)


def test_load_report_input_json_returns_resolved_path_and_payload(
    tmp_path: Path,
) -> None:
    report = tmp_path / "report.json"
    payload = {"ok": True}
    report.write_text(json.dumps(payload), encoding="utf-8")

    resolved, loaded = load_report_input_json(tmp_path)

    assert resolved == report.resolve()
    assert loaded == payload


def test_load_report_input_json_rejects_unreadable_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    report = tmp_path / "report.json"
    report.write_text("{}", encoding="utf-8")
    original_open = Path.open

    def _bad_open(self: Path, *args: object, **kwargs: object):
        if self == report:
            raise OSError("io fail")
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", _bad_open, raising=False)

    with pytest.raises(ReportInputError, match="not readable"):
        load_report_input_json(report)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="mkfifo not available")
def test_resolve_report_input_path_rejects_non_regular_fifo(tmp_path: Path) -> None:
    fifo = tmp_path / "report.pipe"
    os.mkfifo(fifo)

    with pytest.raises(ReportInputError, match="not a regular report file"):
        resolve_report_input_path(fifo)
