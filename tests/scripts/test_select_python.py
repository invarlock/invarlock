from __future__ import annotations

import stat
import subprocess
from pathlib import Path


def _write_fake_python(path: Path, version: str) -> None:
    major, minor, patch = version.split(".")
    script = f"""#!/bin/bash
set -euo pipefail

major={major}
minor={minor}
patch={patch}

if [[ "${{1:-}}" == "--version" ]]; then
  printf 'Python %s.%s.%s\\n' "$major" "$minor" "$patch"
  exit 0
fi

if [[ "${{1:-}}" == "-c" ]]; then
  code="${{2:-}}"
  if [[ "$code" == *"sys.version_info[:2] == (3, 12)"* ]]; then
    [[ "$major" -eq 3 && "$minor" -eq 12 ]]
    exit $?
  fi
  if [[ "$code" == *"sys.version_info >= (3, 12)"* ]]; then
    [[ "$major" -gt 3 || ( "$major" -eq 3 && "$minor" -ge 12 ) ]]
    exit $?
  fi
fi

exit 0
"""
    path.write_text(script, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _run_selector(
    tmp_path: Path,
    *,
    python_version: str,
    python312_version: str | None = None,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    repo_root = Path(__file__).resolve().parents[2]
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()

    _write_fake_python(bin_dir / "python", python_version)
    if python312_version is not None:
        _write_fake_python(bin_dir / "python3.12", python312_version)

    env = {
        "PATH": str(bin_dir),
        "HOME": str(tmp_path),
    }
    if extra_env:
        env.update(extra_env)

    return subprocess.run(
        ["/bin/bash", str(repo_root / "scripts" / "select_python.sh")],
        capture_output=True,
        text=True,
        check=False,
        cwd=repo_root,
        env=env,
    )


def test_select_python_prefers_active_supported_python_on_github_actions(
    tmp_path: Path,
) -> None:
    proc = _run_selector(
        tmp_path,
        python_version="3.13.2",
        python312_version="3.12.9",
        extra_env={"GITHUB_ACTIONS": "true"},
    )

    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == str(tmp_path / "bin" / "python")


def test_select_python_prefers_active_supported_python_in_virtualenv(
    tmp_path: Path,
) -> None:
    proc = _run_selector(
        tmp_path,
        python_version="3.13.2",
        python312_version="3.12.9",
        extra_env={"VIRTUAL_ENV": str(tmp_path / ".venv")},
    )

    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == str(tmp_path / "bin" / "python")


def test_select_python_keeps_default_python312_preference_without_active_env(
    tmp_path: Path,
) -> None:
    proc = _run_selector(
        tmp_path,
        python_version="3.13.2",
        python312_version="3.12.9",
    )

    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == str(tmp_path / "bin" / "python3.12")


def test_select_python_finds_named_home_conda_env_when_off_path(
    tmp_path: Path,
) -> None:
    conda_python = (
        tmp_path / "anaconda3" / "envs" / "invarlock-py312" / "bin" / "python"
    )
    conda_python.parent.mkdir(parents=True)
    _write_fake_python(conda_python, "3.12.8")

    proc = _run_selector(
        tmp_path,
        python_version="3.11.5",
        python312_version=None,
    )

    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == str(conda_python)
