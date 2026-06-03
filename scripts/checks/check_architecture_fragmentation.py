#!/usr/bin/env python3
"""Report module-fragmentation metrics without forcing file splitting."""

from __future__ import annotations

import argparse
import fnmatch
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

SCRIPT_GENERATED_CRUFT_PATTERNS = (
    "scripts/.DS_Store",
    "scripts/**/.DS_Store",
    "scripts/**/.coverage/**",
    "scripts/**/.mypy_cache/**",
    "scripts/**/.pytest_cache/**",
    "scripts/**/.ruff_cache/**",
    "scripts/**/__pycache__/**",
    "scripts/**/*.pyc",
)
SCRIPT_GENERATED_CRUFT_SAMPLE_LIMIT = 50
SCRIPT_LARGE_FILE_LINES = 1000


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _read_lines(path: Path) -> list[str]:
    return _read_text(path).splitlines()


def _python_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*.py") if path.is_file())


def _repo_rel(path: Path, repo_root: Path) -> str:
    return path.relative_to(repo_root).as_posix()


def _path_matches_any(rel_path: str, patterns: tuple[str, ...]) -> bool:
    return any(fnmatch.fnmatch(rel_path, pattern) for pattern in patterns)


def _tracked_files(repo_root: Path, prefix: str) -> list[Path]:
    try:
        result = subprocess.run(
            ["git", "ls-files", prefix],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        root = repo_root / prefix
        if not root.is_dir():
            return []
        return sorted(
            path
            for path in root.rglob("*")
            if path.is_file()
            and not _path_matches_any(
                _repo_rel(path, repo_root), SCRIPT_GENERATED_CRUFT_PATTERNS
            )
        )
    return sorted(
        repo_root / rel_path
        for rel_path in result.stdout.splitlines()
        if rel_path and (repo_root / rel_path).is_file()
    )


def _line_count(path: Path) -> int:
    try:
        return len(_read_lines(path))
    except UnicodeDecodeError:
        return 0


def _is_reexport_shim(path: Path) -> bool:
    text = _read_text(path)
    meaningful = [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    star_imports = [line for line in meaningful if " import *" in line]
    return bool(star_imports) and len(meaningful) <= 30


def _generated_script_cruft(repo_root: Path) -> list[str]:
    scripts_root = repo_root / "scripts"
    if not scripts_root.is_dir():
        return []
    return sorted(
        _repo_rel(path, repo_root)
        for path in scripts_root.rglob("*")
        if path.is_file()
        and _path_matches_any(
            _repo_rel(path, repo_root), SCRIPT_GENERATED_CRUFT_PATTERNS
        )
    )


def _script_family(rel_path: str) -> str:
    rel_parts = Path(rel_path).parts
    if len(rel_parts) <= 2:
        return "__root__"
    return rel_parts[1]


def collect_metrics(repo_root: Path) -> dict[str, object]:
    src_root = repo_root / "src" / "invarlock"
    files = _python_files(src_root)
    line_counts = {path: _line_count(path) for path in files}
    package_counts: Counter[str] = Counter()
    for path in files:
        rel = path.relative_to(src_root)
        package = rel.parts[0] if len(rel.parts) > 1 else "__root__"
        package_counts[package] += 1

    reexport_shims = [
        path.relative_to(repo_root).as_posix()
        for path in files
        if _is_reexport_shim(path)
    ]
    run_orchestrator_files = [
        path.relative_to(repo_root).as_posix()
        for path in files
        if path.name.startswith("run_orchestrator")
    ]
    reporting_files = [
        path.relative_to(repo_root).as_posix()
        for path in files
        if path.relative_to(src_root).parts[0] == "reporting"
    ]
    script_files = _tracked_files(repo_root, "scripts")
    script_rel_paths = [_repo_rel(path, repo_root) for path in script_files]
    script_line_counts = {path: _line_count(path) for path in script_files}
    script_family_counts = Counter(_script_family(path) for path in script_rel_paths)
    large_script_files = [
        _repo_rel(path, repo_root)
        for path, count in script_line_counts.items()
        if count > SCRIPT_LARGE_FILE_LINES
    ]
    large_shell_files = [
        _repo_rel(path, repo_root)
        for path, count in script_line_counts.items()
        if path.suffix == ".sh" and count > SCRIPT_LARGE_FILE_LINES
    ]
    generated_cruft = _generated_script_cruft(repo_root)

    return {
        "format_version": "architecture-fragmentation-v1",
        "source_python_files": len(files),
        "small_files_under_50_lines": sum(
            1 for count in line_counts.values() if count < 50
        ),
        "tiny_files_under_20_lines": sum(
            1 for count in line_counts.values() if count < 20
        ),
        "reexport_shim_count": len(reexport_shims),
        "reexport_shims": reexport_shims,
        "run_orchestrator_file_count": len(run_orchestrator_files),
        "run_orchestrator_files": run_orchestrator_files,
        "reporting_file_count": len(reporting_files),
        "largest_packages_by_file_count": [
            {"package": package, "files": count}
            for package, count in package_counts.most_common(10)
        ],
        "script_tracked_files": len(script_files),
        "script_python_files": sum(1 for path in script_files if path.suffix == ".py"),
        "script_shell_files": sum(1 for path in script_files if path.suffix == ".sh"),
        "script_small_files_under_80_lines": sum(
            1 for count in script_line_counts.values() if count < 80
        ),
        "script_tiny_files_under_20_lines": sum(
            1 for count in script_line_counts.values() if count < 20
        ),
        "script_large_files_over_1000_lines": len(large_script_files),
        "script_large_file_paths_over_1000_lines": large_script_files,
        "script_large_shell_files_over_1000_lines": len(large_shell_files),
        "script_large_shell_file_paths_over_1000_lines": large_shell_files,
        "script_evidence_pack_files": sum(
            1 for path in script_rel_paths if path.startswith("scripts/evidence_packs/")
        ),
        "script_generated_cruft_files": len(generated_cruft),
        "script_generated_cruft_sample_limit": SCRIPT_GENERATED_CRUFT_SAMPLE_LIMIT,
        "script_generated_cruft_sample_paths": generated_cruft[
            :SCRIPT_GENERATED_CRUFT_SAMPLE_LIMIT
        ],
        "largest_script_families_by_file_count": [
            {"family": family, "files": count}
            for family, count in script_family_counts.most_common(10)
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable metrics only.",
    )
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Repository root. Defaults to the current directory.",
    )
    args = parser.parse_args(argv)
    metrics = collect_metrics(Path(args.repo_root).resolve())
    if args.json:
        print(json.dumps(metrics, sort_keys=True))
        return 0
    print("[check_architecture_fragmentation] OK")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
