from __future__ import annotations

import os
import subprocess
import venv
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration


def _create_venv(tmp_path: Path) -> tuple[Path, Path]:
    env_dir = tmp_path / "venv"
    builder = venv.EnvBuilder(with_pip=True)
    builder.create(env_dir)
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

    install = _run(python_exe, ["-m", "pip", "install", str(project_root)])
    assert install.returncode == 0, install.stderr

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

    # Version command must also work without torch installed.
    res_version = _run(python_exe, ["-m", "invarlock", "version"])
    assert res_version.returncode == 0, res_version.stderr
    assert "InvarLock" in res_version.stdout
