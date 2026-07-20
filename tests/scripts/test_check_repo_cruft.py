from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "checks" / "check_repo_cruft.py"


def _run(root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(root)],
        capture_output=True,
        text=True,
        check=False,
    )


def test_check_repo_cruft_accepts_clean_tree(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "module.py").write_text("print('ok')\n", encoding="utf-8")

    result = _run(tmp_path)

    assert result.returncode == 0
    assert "Repo hygiene OK" in result.stdout


def test_check_repo_cruft_rejects_appledouble_and_macos_archive_dir(
    tmp_path: Path,
) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "._module.py").write_text("junk\n", encoding="utf-8")
    (tmp_path / "src" / ".DS_Store").write_text("junk\n", encoding="utf-8")
    (tmp_path / "__MACOSX").mkdir()

    result = _run(tmp_path)

    assert result.returncode == 1
    assert "src/._module.py" in result.stderr
    assert "src/.DS_Store" in result.stderr
    assert "__MACOSX" in result.stderr


def test_check_repo_cruft_ignores_tmp_and_virtualenv_paths(tmp_path: Path) -> None:
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "._pip").write_text("junk\n", encoding="utf-8")
    (tmp_path / "tmp").mkdir()
    (tmp_path / "tmp" / ".DS_Store").write_text("junk\n", encoding="utf-8")

    result = _run(tmp_path)

    assert result.returncode == 0


def test_check_repo_cruft_allows_untracked_gitignored_ds_store(
    tmp_path: Path,
) -> None:
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    (tmp_path / ".gitignore").write_text(".DS_Store\n", encoding="utf-8")
    (tmp_path / ".DS_Store").write_text("local finder state\n", encoding="utf-8")

    result = _run(tmp_path)

    assert result.returncode == 0


def test_check_repo_cruft_json_reports_sorted_matches(tmp_path: Path) -> None:
    (tmp_path / "z").mkdir()
    (tmp_path / "z" / "._second").write_text("junk\n", encoding="utf-8")
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "._first").write_text("junk\n", encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(tmp_path), "--json"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload == {
        "ok": False,
        "root": str(tmp_path.resolve()),
        "matches": ["a/._first", "z/._second"],
    }


def test_check_repo_cruft_ignores_generated_prefix_directories(tmp_path: Path) -> None:
    for dirname in ("tmp_session", "reports_local", "runs_experiment"):
        directory = tmp_path / dirname
        directory.mkdir()
        (directory / "._ignored").write_text("junk\n", encoding="utf-8")

    result = _run(tmp_path)

    assert result.returncode == 0
