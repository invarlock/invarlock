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

    def _fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        return SimpleNamespace(returncode=0, stdout="ok\n", stderr="")

    monkeypatch.setattr(module.subprocess, "run", _fake_run)

    out_root = tmp_path / "artifacts"
    exit_code = module.main(
        [
            "--paths",
            "README.md",
            "notebooks",
            "--output-root",
            str(out_root),
            "--markdown-execution-mode",
            "trusted-local",
        ]
    )

    assert exit_code == 0
    assert len(calls) == 2
    assert calls[0][1].endswith("verify_markdown_bash_blocks.py")
    assert "--execution-mode" in calls[0]
    assert "trusted-local" in calls[0]
    assert calls[1][1].endswith("verify_notebooks_smoke.py")

    summary = json.loads((out_root / "summary.json").read_text(encoding="utf-8"))
    assert summary["ok"] is True
    assert summary["failures"] == []
    assert summary["markdown"]["returncode"] == 0
    assert summary["notebooks"]["returncode"] == 0


def test_main_limits_paths_to_curated_subset(tmp_path: Path, monkeypatch) -> None:
    module = _load_script_module()
    (tmp_path / "README.md").write_text("# root\n", encoding="utf-8")
    notebooks = tmp_path / "notebooks"
    notebooks.mkdir()
    (notebooks / "demo.ipynb").write_text("{}", encoding="utf-8")
    docs = tmp_path / "docs" / "user-guide"
    docs.mkdir(parents=True)
    (docs / "quickstart.md").write_text("# quickstart\n", encoding="utf-8")
    module.ROOT = tmp_path

    calls: list[list[str]] = []

    def _fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        return SimpleNamespace(returncode=0, stdout="ok\n", stderr="")

    monkeypatch.setattr(module.subprocess, "run", _fake_run)

    exit_code = module.main(
        [
            "--paths",
            "README.md",
            "docs/user-guide/quickstart.md",
            "notebooks/demo.ipynb",
            "--skip-markdown-model-loading",
            "--skip-notebook-model-loading",
            "--output-root",
            str(tmp_path / "artifacts"),
        ]
    )

    assert exit_code == 0
    assert "--paths" in calls[0]
    assert "--skip-model-loading" in calls[0]
    assert "README.md" in calls[0]
    assert "docs/user-guide/quickstart.md" in calls[0]
    assert "--skip-model-loading" in calls[1]
    assert "notebooks/demo.ipynb" in calls[1]


def test_default_notebook_inventory_requires_explicit_classification() -> None:
    module = _load_script_module()
    repo_root = Path(__file__).resolve().parents[2]
    discovered = [
        str(path.relative_to(repo_root))
        for path in sorted((repo_root / "notebooks").glob("*.ipynb"))
    ]

    assert list(module.DEFAULT_NOTEBOOK_PATHS) == discovered
    assert module._resolve_notebook_paths(None) == discovered
