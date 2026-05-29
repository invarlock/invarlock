from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType


def _load_script_module() -> ModuleType:
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / "docs" / "verify_live_examples.py"
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

    def _fake_run_subprocess(cmd, **kwargs):
        calls.append(list(cmd))
        return {
            "command": list(cmd),
            "returncode": 0,
            "log_path": "artifacts/run.log",
        }

    monkeypatch.setattr(module, "_run_subprocess", _fake_run_subprocess)

    out_root = tmp_path / "artifacts"
    exit_code = module.main(
        [
            "--paths",
            "README.md",
            "notebooks",
            "--output-root",
            str(out_root),
            "--markdown-execution-mode",
            "host",
        ]
    )

    assert exit_code == 0
    assert len(calls) == 2
    assert calls[0][1].endswith("verify_markdown_bash_blocks.py")
    assert "--execution-mode" in calls[0]
    assert "host" in calls[0]
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

    def _fake_run_subprocess(cmd, **kwargs):
        calls.append(list(cmd))
        return {
            "command": list(cmd),
            "returncode": 0,
            "log_path": "artifacts/run.log",
        }

    monkeypatch.setattr(module, "_run_subprocess", _fake_run_subprocess)

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


def test_default_env_reuses_configured_hf_cache_root(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_script_module()
    preferred_hf_home = tmp_path / "shared-hf"
    monkeypatch.setenv("HF_HOME", str(preferred_hf_home))
    monkeypatch.setattr(module, "_cache_root_is_writable", lambda root: True)

    env = module._default_env(output_root=tmp_path / "artifacts")

    assert env["HF_HOME"] == str(preferred_hf_home)
    assert env["HF_HUB_CACHE"] == str(preferred_hf_home / "hub")
    assert env["HF_DATASETS_CACHE"] == str(preferred_hf_home / "datasets")
    assert env["DISABLE_SAFETENSORS_CONVERSION"] == "1"


def test_default_env_falls_back_to_output_cache_when_preferred_root_is_not_writable(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_script_module()
    monkeypatch.setenv("HF_HOME", str(tmp_path / "read-only-hf"))
    monkeypatch.setattr(module, "_cache_root_is_writable", lambda root: False)

    env = module._default_env(output_root=tmp_path / "artifacts")

    assert env["HF_HOME"] == str(tmp_path / "artifacts" / ".hf")
    assert env["HF_HUB_CACHE"] == str(tmp_path / "artifacts" / ".hf" / "hub")
    assert env["HF_DATASETS_CACHE"] == str(tmp_path / "artifacts" / ".hf" / "datasets")
    assert env["DISABLE_SAFETENSORS_CONVERSION"] == "1"


def test_run_subprocess_streams_output_to_log_and_console(
    tmp_path: Path, capsys
) -> None:
    module = _load_script_module()
    module.ROOT = tmp_path

    script = tmp_path / "emit.py"
    script.write_text(
        "print('alpha')\nprint('beta')\n",
        encoding="utf-8",
    )
    log_path = tmp_path / "artifacts" / "run.log"

    result = module._run_subprocess(
        [sys.executable, str(script)],
        env=module._default_env(output_root=tmp_path / "artifacts"),
        log_path=log_path,
    )

    assert result["returncode"] == 0
    assert log_path.read_text(encoding="utf-8") == "alpha\nbeta\n"
    captured = capsys.readouterr()
    assert "[live] Running:" in captured.out
    assert "alpha" in captured.out
    assert "beta" in captured.out
    assert "[live] Finished rc=0:" in captured.out
