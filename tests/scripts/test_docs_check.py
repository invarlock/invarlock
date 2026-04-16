from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_script_module():
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / "docs_check.py"
    spec = importlib.util.spec_from_file_location("tests_docs_check", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_main_all_runs_curated_live_checks_but_not_full_live(
    monkeypatch, capsys
) -> None:
    module = _load_script_module()
    calls: list[str] = []

    monkeypatch.setattr(module, "check_build", lambda: calls.append("build"))
    monkeypatch.setattr(module, "check_links", lambda: calls.append("links"))
    monkeypatch.setattr(module, "check_references", lambda: calls.append("refs"))
    monkeypatch.setattr(module, "check_examples", lambda: calls.append("examples"))
    monkeypatch.setattr(
        module, "check_consistency", lambda: calls.append("consistency")
    )
    monkeypatch.setattr(module, "check_live_fast", lambda: calls.append("live_fast"))
    monkeypatch.setattr(module, "check_live", lambda: calls.append("live"))

    module.main(["--all"])
    capsys.readouterr()

    assert calls == [
        "build",
        "links",
        "refs",
        "examples",
        "consistency",
        "live_fast",
    ]


def test_main_live_fast_only_runs_curated_live_checks(monkeypatch, capsys) -> None:
    module = _load_script_module()
    calls: list[str] = []

    monkeypatch.setattr(module, "check_live_fast", lambda: calls.append("live_fast"))
    monkeypatch.setattr(module, "check_live", lambda: calls.append("live"))

    module.main(["--live-fast"])
    capsys.readouterr()

    assert calls == ["live_fast"]


def test_check_live_fast_uses_trusted_local_demo_mode(monkeypatch, capsys) -> None:
    module = _load_script_module()
    commands: list[list[str]] = []

    def _fake_run(cmd: list[str]) -> tuple[int, str]:
        commands.append(cmd)
        return 0, "ok\n"

    monkeypatch.setattr(module, "run", _fake_run)

    module.check_live_fast()
    capsys.readouterr()

    assert commands == [
        [
            sys.executable,
            "scripts/verify_live_examples.py",
            "--markdown-execution-mode",
            "host",
            "--skip-markdown-model-loading",
            "--skip-notebook-model-loading",
            "--paths",
            *module.CURATED_LIVE_EXAMPLE_PATHS,
        ]
    ]
