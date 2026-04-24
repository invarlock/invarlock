from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tests.integration.packaging import _support_installed_wheel as support


def test_select_python_prefers_workspace_python_when_build_capable(
    monkeypatch,
    tmp_path: Path,
) -> None:
    repo_root = tmp_path
    workspace_python = repo_root / ".venv" / "bin" / "python"
    workspace_python.parent.mkdir(parents=True)
    workspace_python.write_text("", encoding="utf-8")

    monkeypatch.setattr(support, "_python_can_build_wheel", lambda python_exe: True)

    selected = support._select_python(repo_root)

    assert selected == workspace_python


def test_select_python_skips_workspace_python_without_build_support(
    monkeypatch,
    tmp_path: Path,
) -> None:
    repo_root = tmp_path
    workspace_python = repo_root / ".venv" / "bin" / "python"
    workspace_python.parent.mkdir(parents=True)
    workspace_python.write_text("", encoding="utf-8")
    fallback_python = tmp_path / "python-ok"
    fallback_python.write_text("", encoding="utf-8")

    def fake_can_build(python_exe: Path) -> bool:
        return python_exe == fallback_python

    monkeypatch.setattr(support, "_python_can_build_wheel", fake_can_build)

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=args[0],
            returncode=0,
            stdout=f"{fallback_python}\n",
            stderr="",
        )

    monkeypatch.setattr(support.subprocess, "run", fake_run)

    selected = support._select_python(repo_root)

    assert selected == fallback_python


def test_select_python_requests_build_capable_selector(
    monkeypatch, tmp_path: Path
) -> None:
    repo_root = tmp_path
    workspace_python = repo_root / ".venv" / "bin" / "python"
    workspace_python.parent.mkdir(parents=True)
    workspace_python.write_text("", encoding="utf-8")
    fallback_python = tmp_path / "python-ok"
    fallback_python.write_text("", encoding="utf-8")

    monkeypatch.setattr(support, "_python_can_build_wheel", lambda python_exe: False)

    def fake_run(*args, **kwargs):
        assert kwargs["env"]["INVARLOCK_SELECT_PYTHON_REQUIRE_MODULES"] == "build"
        return subprocess.CompletedProcess(
            args=args[0],
            returncode=0,
            stdout=f"{fallback_python}\n",
            stderr="",
        )

    monkeypatch.setattr(support.subprocess, "run", fake_run)

    selected = support._select_python(repo_root)

    assert selected == fallback_python


def test_create_venv_uses_system_site_packages(
    monkeypatch,
    tmp_path: Path,
) -> None:
    calls: list[list[str]] = []

    def fake_run(args, **kwargs):
        calls.append(args)
        return subprocess.CompletedProcess(args=args, returncode=0)

    monkeypatch.setattr(support.subprocess, "run", fake_run)

    support._create_venv(tmp_path, tmp_path / "python3.12")

    assert calls
    assert "--system-site-packages" in calls[0]


def test_install_core_dependencies_noops_when_core_deps_are_already_available(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        support, "_core_dependencies_available", lambda python_exe: True
    )
    monkeypatch.setattr(
        support,
        "_install_requirements_file",
        lambda repo_root, python_exe, requirements: (_ for _ in ()).throw(
            AssertionError("requirements install should not run")
        ),
    )

    support._install_core_dependencies(tmp_path, tmp_path / "python3.12")


def test_install_core_dependencies_skips_when_offline_fallback_install_fails(
    monkeypatch,
    tmp_path: Path,
) -> None:
    requirements = tmp_path / "requirements" / "workflows" / "core-py312.txt"
    requirements.parent.mkdir(parents=True)
    requirements.write_text("placeholder", encoding="utf-8")

    monkeypatch.setattr(
        support, "_core_dependencies_available", lambda python_exe: False
    )
    monkeypatch.setattr(support, "_python_minor_version", lambda python_exe: (3, 12))
    monkeypatch.setattr(
        support,
        "_install_requirements_file",
        lambda repo_root, python_exe, requirement_path: subprocess.CompletedProcess(
            args=[str(python_exe), "-m", "pip"],
            returncode=1,
            stdout="",
            stderr="Failed to establish a new connection",
        ),
    )

    with pytest.raises(pytest.skip.Exception):
        support._install_core_dependencies(tmp_path, tmp_path / "python3.12")


def test_install_core_dependencies_raises_for_non_network_failures(
    monkeypatch,
    tmp_path: Path,
) -> None:
    requirements = tmp_path / "requirements" / "workflows" / "core-py312.txt"
    requirements.parent.mkdir(parents=True)
    requirements.write_text("placeholder", encoding="utf-8")

    monkeypatch.setattr(
        support, "_core_dependencies_available", lambda python_exe: False
    )
    monkeypatch.setattr(support, "_python_minor_version", lambda python_exe: (3, 12))
    monkeypatch.setattr(
        support,
        "_install_requirements_file",
        lambda repo_root, python_exe, requirement_path: subprocess.CompletedProcess(
            args=[str(python_exe), "-m", "pip"],
            returncode=1,
            stdout="",
            stderr="hash mismatch",
        ),
    )

    with pytest.raises(
        AssertionError, match="failed to install pinned core wheel-smoke dependencies"
    ):
        support._install_core_dependencies(tmp_path, tmp_path / "python3.12")


def test_ensure_hf_smoke_dependencies_skips_on_offline_extra_install(
    monkeypatch,
    tmp_path: Path,
) -> None:
    env = support.InstalledWheelEnv(
        repo_root=tmp_path,
        env_dir=tmp_path / "venv",
        wheel_path=tmp_path / "dist" / "invarlock.whl",
        python_exe=tmp_path / "venv" / "bin" / "python",
        cli_exe=tmp_path / "venv" / "bin" / "invarlock",
    )
    calls = iter(
        [
            subprocess.CompletedProcess(
                args=["python", "-c", "import torch, transformers"],
                returncode=1,
                stdout="",
                stderr="missing torch",
            ),
            subprocess.CompletedProcess(
                args=["python", "-m", "pip", "install"],
                returncode=1,
                stdout="",
                stderr="Failed to establish a new connection",
            ),
        ]
    )
    monkeypatch.setattr(support, "_run", lambda *args, **kwargs: next(calls))

    with pytest.raises(pytest.skip.Exception):
        support._ensure_hf_smoke_dependencies(env)
