from __future__ import annotations

import json
from pathlib import Path

import pytest

from invarlock.cli.commands.report import _load_run_report


def test_load_run_report_from_file(tmp_path: Path):
    p = tmp_path / "report.json"
    p.write_text(json.dumps({"ok": True}), encoding="utf-8")
    out = _load_run_report(str(p))
    assert out == {"ok": True}


def test_load_run_report_from_dir_requires_canonical_report_name(tmp_path: Path):
    (tmp_path / "other.json").write_text("{}", encoding="utf-8")
    (tmp_path / "my_report.json").write_text(json.dumps({"hello": 1}), encoding="utf-8")

    with pytest.raises(ValueError, match="does not contain a canonical report file"):
        _load_run_report(str(tmp_path))


def test_load_run_report_from_dir_prefers_exact_canonical_report(tmp_path: Path):
    (tmp_path / "my_report.json").write_text(
        json.dumps({"hello": "fuzzy"}), encoding="utf-8"
    )
    (tmp_path / "report.json").write_text(
        json.dumps({"hello": "canonical"}), encoding="utf-8"
    )

    out = _load_run_report(str(tmp_path))

    assert out == {"hello": "canonical"}


def test_load_run_report_from_dir_raises_on_ambiguous_canonical_reports(tmp_path: Path):
    (tmp_path / "report.json").write_text(json.dumps({"kind": "run"}), encoding="utf-8")
    (tmp_path / "evaluation.report.json").write_text(
        json.dumps({"kind": "evaluation"}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Ambiguous report directory"):
        _load_run_report(str(tmp_path))


def test_load_run_report_dir_missing_raises(tmp_path: Path):
    with pytest.raises(ValueError, match="does not contain a canonical report file"):
        _load_run_report(str(tmp_path))
