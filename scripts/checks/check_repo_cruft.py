from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

EXCLUDED_DIRS = {
    ".git",
    ".claude",
    ".venv",
    ".venv-invarlock",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".hypothesis",
    ".nox",
    ".tox",
    "build",
    "dist",
    "htmlcov",
    "node_modules",
    "site",
    "tmp",
    "runs",
    "runs_cfg",
    "reports",
    "artifacts",
    "__pycache__",
}
BANNED_FILENAMES: set[str] = set()
BANNED_TRACKED_FILENAMES = {".DS_Store"}
BANNED_DIRNAMES = {"__MACOSX"}
BANNED_PREFIX = "._"


def _git_tracked_paths(root: Path) -> set[str] | None:
    try:
        result = subprocess.run(
            ["git", "ls-files"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return {line for line in result.stdout.splitlines() if line}


def _is_banned_file(
    rel_path: str,
    filename: str,
    tracked_paths: set[str] | None,
) -> bool:
    if filename in BANNED_FILENAMES or filename.startswith(BANNED_PREFIX):
        return True
    if filename not in BANNED_TRACKED_FILENAMES:
        return False
    return tracked_paths is None or rel_path in tracked_paths


def _iter_cruft_paths(root: Path) -> list[str]:
    matches: list[str] = []
    tracked_paths = _git_tracked_paths(root)
    for dirpath, dirnames, filenames in os.walk(root):
        current = Path(dirpath)
        rel_current = current.relative_to(root)
        dirnames[:] = [
            name
            for name in dirnames
            if name not in EXCLUDED_DIRS
            and not name.startswith("tmp_")
            and not name.startswith("reports_")
            and not name.startswith("runs_")
        ]

        for dirname in list(dirnames):
            if dirname in BANNED_DIRNAMES or dirname.startswith(BANNED_PREFIX):
                matches.append((rel_current / dirname).as_posix())

        for filename in filenames:
            rel_path = (rel_current / filename).as_posix()
            if _is_banned_file(rel_path, filename, tracked_paths):
                matches.append(rel_path)
    return sorted(matches)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fail when macOS transport artifacts leak into repo source paths."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Repository root to scan.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit a machine-readable result payload.",
    )
    args = parser.parse_args(argv)

    root = args.root.resolve()
    matches = _iter_cruft_paths(root)
    if args.json:
        print(
            json.dumps(
                {"ok": not matches, "root": str(root), "matches": matches},
                indent=2,
            )
        )
    elif matches:
        print(
            "ERROR: macOS transport artifacts found in repo source paths:",
            file=sys.stderr,
        )
        for path in matches:
            print(f"  - {path}", file=sys.stderr)
    else:
        print("Repo hygiene OK: no macOS transport artifacts found.")

    return 0 if not matches else 1


if __name__ == "__main__":
    raise SystemExit(main())
