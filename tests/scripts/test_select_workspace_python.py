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

if [[ "${{1:-}}" == "-c" ]]; then
  code="${{2:-}}"
  if [[ "$code" == *"sys.version_info >= (3, 12)"* ]]; then
    [[ "$major" -gt 3 || ( "$major" -eq 3 && "$minor" -ge 12 ) ]]
    exit $?
  fi
fi

exit 0
"""
    path.write_text(script, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _copy_executable(src: Path, dst: Path) -> None:
    dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    dst.chmod(src.stat().st_mode | stat.S_IXUSR)


def test_select_workspace_python_prefers_repo_venv(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    fake_repo = tmp_path / "repo"
    scripts_dir = fake_repo / "scripts"
    venv_bin = fake_repo / ".venv" / "bin"
    scripts_dir.mkdir(parents=True)
    venv_bin.mkdir(parents=True)

    _copy_executable(
        repo_root / "scripts" / "select_workspace_python.sh",
        scripts_dir / "select_workspace_python.sh",
    )
    _write_fake_python(venv_bin / "python", "3.12.7")

    fallback = scripts_dir / "select_python.sh"
    fallback.write_text("#!/bin/bash\nprintf '/fallback/python\\n'\n", encoding="utf-8")
    fallback.chmod(fallback.stat().st_mode | stat.S_IXUSR)

    proc = subprocess.run(
        ["/bin/bash", str(scripts_dir / "select_workspace_python.sh")],
        capture_output=True,
        text=True,
        check=False,
        cwd=fake_repo,
    )

    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == str(venv_bin / "python")


def test_select_workspace_python_falls_back_to_selector_when_repo_venv_missing(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    fake_repo = tmp_path / "repo"
    scripts_dir = fake_repo / "scripts"
    scripts_dir.mkdir(parents=True)

    _copy_executable(
        repo_root / "scripts" / "select_workspace_python.sh",
        scripts_dir / "select_workspace_python.sh",
    )

    fallback_python = fake_repo / "fallback" / "python"
    fallback_python.parent.mkdir(parents=True)
    _write_fake_python(fallback_python, "3.12.9")

    fallback = scripts_dir / "select_python.sh"
    fallback.write_text(
        f"#!/bin/bash\nprintf '%s\\n' {str(fallback_python)!r}\n",
        encoding="utf-8",
    )
    fallback.chmod(fallback.stat().st_mode | stat.S_IXUSR)

    proc = subprocess.run(
        ["/bin/bash", str(scripts_dir / "select_workspace_python.sh")],
        capture_output=True,
        text=True,
        check=False,
        cwd=fake_repo,
    )

    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == str(fallback_python)
