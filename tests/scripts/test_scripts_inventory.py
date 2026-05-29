from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "check_scripts_inventory.py"


def _run(root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(root)],
        capture_output=True,
        text=True,
        check=False,
    )


def test_scripts_inventory_accepts_current_tree() -> None:
    result = _run(REPO_ROOT)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "[check_scripts_inventory] OK" in result.stdout
    assert "evidence-packs" in result.stdout


def test_scripts_inventory_json_audit_includes_per_file_metadata() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(REPO_ROOT), "--json"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    assert payload["format_version"] == "scripts-audit-v1"
    rows = {row["path"]: row for row in payload["files"]}
    row = rows["scripts/check_scripts_inventory.py"]
    assert row["owner"] == "maintainers"
    assert row["expected_runtime"] == "fast"
    assert row["network"] == "never"
    assert row["gpu"] == "never"
    assert isinstance(payload["unreferenced"], list)


def test_scripts_inventory_rejects_unclassified_files(tmp_path: Path) -> None:
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    (scripts_dir / "known.py").write_text("print('known')\n", encoding="utf-8")
    (scripts_dir / "loose.py").write_text("print('loose')\n", encoding="utf-8")
    (scripts_dir / "scripts_inventory.toml").write_text(
        """
version = 1

[[families]]
name = "known"
paths = ["scripts/known.py", "scripts/scripts_inventory.toml"]
""".lstrip(),
        encoding="utf-8",
    )

    result = _run(tmp_path)

    assert result.returncode == 1
    assert "unclassified script file: scripts/loose.py" in result.stderr


def test_scripts_inventory_rejects_overlapping_families(tmp_path: Path) -> None:
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    (scripts_dir / "known.py").write_text("print('known')\n", encoding="utf-8")
    (scripts_dir / "scripts_inventory.toml").write_text(
        """
version = 1

[[families]]
name = "first"
paths = ["scripts/*.py"]

[[families]]
name = "second"
paths = ["scripts/known.py", "scripts/scripts_inventory.toml"]
""".lstrip(),
        encoding="utf-8",
    )

    result = _run(tmp_path)

    assert result.returncode == 1
    assert "matches multiple families: scripts/known.py" in result.stderr
