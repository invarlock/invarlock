#!/usr/bin/env python3
"""Report module-fragmentation metrics without forcing file splitting."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from common_io import read_lines, read_text


def _python_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*.py") if path.is_file())


def _line_count(path: Path) -> int:
    try:
        return len(read_lines(path))
    except UnicodeDecodeError:
        return 0


def _is_reexport_shim(path: Path) -> bool:
    text = read_text(path)
    meaningful = [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    star_imports = [line for line in meaningful if " import *" in line]
    return bool(star_imports) and len(meaningful) <= 30


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
