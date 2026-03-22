from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace


def _load_script_module() -> ModuleType:
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / "verify_live_examples.py"
    spec = importlib.util.spec_from_file_location(
        "tests_verify_live_examples", script_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_resolve_paths_filters_requested_markdown_and_notebooks(tmp_path: Path) -> None:
    module = _load_script_module()
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "guide.md").write_text("# guide\n", encoding="utf-8")
    (tmp_path / "docs" / "ignore.txt").write_text("x\n", encoding="utf-8")
    (tmp_path / "notebooks").mkdir()
    (tmp_path / "notebooks" / "demo.ipynb").write_text("{}", encoding="utf-8")
    (tmp_path / "README.md").write_text("# root\n", encoding="utf-8")

    module.ROOT = tmp_path

    assert module._resolve_markdown_paths(["README.md", "docs"]) == [
        "README.md",
        "docs/guide.md",
    ]
    assert module._resolve_notebook_paths(["README.md", "notebooks"]) == [
        "notebooks/demo.ipynb"
    ]


def test_main_writes_summary_and_invokes_both_surfaces(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_script_module()
    (tmp_path / "README.md").write_text("# root\n", encoding="utf-8")
    (tmp_path / "notebooks").mkdir()
    (tmp_path / "notebooks" / "demo.ipynb").write_text("{}", encoding="utf-8")
    module.ROOT = tmp_path

    calls: list[list[str]] = []

    def _fake_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
        calls.append(list(cmd))
        return SimpleNamespace(returncode=0, stdout="ok\n", stderr="")

    monkeypatch.setattr(module.subprocess, "run", _fake_run)

    out_root = tmp_path / "artifacts"
    exit_code = module.main(
        ["--paths", "README.md", "notebooks", "--output-root", str(out_root)]
    )

    assert exit_code == 0
    assert len(calls) == 2
    assert calls[0][1].endswith("verify_markdown_bash_blocks.py")
    assert calls[1][1].endswith("verify_notebooks_smoke.py")

    summary = json.loads((out_root / "summary.json").read_text(encoding="utf-8"))
    assert summary["ok"] is True
    assert summary["failures"] == []
    assert summary["markdown"]["returncode"] == 0
    assert summary["notebooks"]["returncode"] == 0
