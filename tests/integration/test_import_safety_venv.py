from __future__ import annotations

import os
import shutil
import subprocess
import sys
import venv
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration


def _create_venv(tmp_path: Path) -> tuple[Path, Path]:
    env_dir = tmp_path / "venv"

    preferred_python = shutil.which("python3.12")
    current_python = Path(sys.executable)
    candidate_pythons: list[Path] = []
    if preferred_python:
        candidate_pythons.append(Path(preferred_python))
    if current_python not in candidate_pythons:
        candidate_pythons.append(current_python)

    creation_errors: list[str] = []
    for python_exe in candidate_pythons:
        result = subprocess.run(
            [str(python_exe), "-m", "venv", str(env_dir)],
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode == 0:
            break
        creation_errors.append(
            f"{python_exe} -> {result.returncode}\n{result.stdout}{result.stderr}"
        )
        shutil.rmtree(env_dir, ignore_errors=True)
    else:
        builder = venv.EnvBuilder(with_pip=True)
        try:
            builder.create(env_dir)
        except subprocess.CalledProcessError as exc:
            combined_errors = "\n\n".join(creation_errors)
            raise AssertionError(
                "failed to create isolated venv for import-safety test\n"
                f"{combined_errors}\n\n"
                f"fallback builder failed: {exc}"
            ) from exc

    if os.name == "nt":
        python_exe = env_dir / "Scripts" / "python.exe"
    else:
        python_exe = env_dir / "bin" / "python"
    return env_dir, python_exe


def _run(python: Path, args: list[str]) -> subprocess.CompletedProcess[str]:
    cmd = [str(python), *args]
    # Use text mode for easier assertions.
    return subprocess.run(
        cmd,
        text=True,
        capture_output=True,
        check=False,
    )


def test_import_and_cli_help_without_torch(tmp_path: Path):
    # Create an isolated virtual environment and install the project.
    env_dir, python_exe = _create_venv(tmp_path)
    project_root = Path(__file__).resolve().parents[2]
    source_root = tmp_path / "src-copy"
    shutil.copytree(
        project_root,
        source_root,
        ignore=shutil.ignore_patterns(
            ".git",
            ".mypy_cache",
            ".ruff_cache",
            ".venv",
            ".worktrees",
            ".coverage",
            ".coverage.*",
            ".pytest_cache",
            "__pycache__",
            ".hf",
            "artifacts",
            "build",
            "custom-runs",
            "dist",
            "evidence_pack_runs",
            "reports",
            "runs",
            "site",
            "out",
            "tmp",
            "node_modules",
            "target",
        ),
    )

    install = _run(python_exe, ["-m", "pip", "install", str(source_root)])
    if install.returncode != 0:
        combined = f"{install.stdout}{install.stderr}"
        if "requires a different Python" in combined or "not in '>=3.12'" in combined:
            pytest.skip("Requires Python 3.12+ to install invarlock in a venv.")
        if any(
            marker in combined
            for marker in (
                "Failed to establish a new connection",
                "NewConnectionError",
                "Temporary failure in name resolution",
                "Name or service not known",
                "nodename nor servname provided",
            )
        ):
            pytest.skip(
                "Network unavailable to install runtime dependencies into an isolated venv."
            )
        assert install.returncode == 0, combined

    # Ensure torch/transformers are not present in the venv.
    _run(python_exe, ["-m", "pip", "uninstall", "-y", "torch", "transformers"])

    # Plain import of the package root must succeed and expose __version__.
    res_import = _run(
        python_exe,
        ["-c", "import invarlock; print(invarlock.__version__)"],
    )
    assert res_import.returncode == 0, res_import.stderr
    assert res_import.stdout.strip()

    # CLI help via `python -m invarlock --help` must be torch-free.
    res_help = _run(python_exe, ["-m", "invarlock", "--help"])
    assert res_help.returncode == 0, res_help.stderr
    assert "Usage:" in res_help.stdout

    # Version flag must also work without torch installed.
    res_version = _run(python_exe, ["-m", "invarlock", "--version"])
    assert res_version.returncode == 0, res_version.stderr
    assert "InvarLock" in res_version.stdout
