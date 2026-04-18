from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest


def _load_script_module() -> ModuleType:
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / "docs_lint.py"
    spec = importlib.util.spec_from_file_location("tests_docs_lint", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_docs_tree(tmp_path: Path) -> None:
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "guide.md").write_text("# guide\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("# root\n", encoding="utf-8")


def test_main_skips_missing_tools_without_require_tools(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    module = _load_script_module()
    _write_docs_tree(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(module.shutil, "which", lambda name: None)
    monkeypatch.setattr(module, "_local_node_bin", lambda name: None)

    module.main(["--all"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out.strip().splitlines()[-1])
    assert payload == {"ok": True, "summary": {"markdown": False, "spell": False}}
    assert "skipping" in captured.err


def test_main_requires_tools_when_requested(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    module = _load_script_module()
    _write_docs_tree(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(module.shutil, "which", lambda name: None)
    monkeypatch.setattr(module, "_local_node_bin", lambda name: None)

    with pytest.raises(SystemExit) as excinfo:
        module.main(["--all", "--require-tools"])

    assert excinfo.value.code == 1
    captured = capsys.readouterr()
    payload = json.loads(captured.err.strip().splitlines()[-1])
    assert payload["ok"] is False
    assert payload["summary"] == {"markdown": None, "spell": None}
    assert payload["exit"] == 1
