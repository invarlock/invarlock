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


def test_load_run_report_from_dir_selects_report(tmp_path: Path):
    # Create two jsons, one with 'report' substring
    (tmp_path / "other.json").write_text("{}", encoding="utf-8")
    (tmp_path / "my_report.json").write_text(json.dumps({"hello": 1}), encoding="utf-8")
    out = _load_run_report(str(tmp_path))
    assert out == {"hello": 1}


def test_load_run_report_dir_missing_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        _load_run_report(str(tmp_path))
