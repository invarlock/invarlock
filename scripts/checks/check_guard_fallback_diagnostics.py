#!/usr/bin/env python3
"""Audit guard numeric fallbacks for diagnostics or explicit rationale."""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

GUARD_ROOT = Path("src/invarlock/guards")
DIAGNOSTIC_MARKERS = (
    "guard-fallback-ok",
    "diagnostic",
    "diagnostics",
    "GuardDiagnostic",
    "logger.",
    "_log_event",
    ".warning(",
    ".error(",
    "warnings",
    "errors",
    "raise ",
)


def _repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[2]


def _is_neutral_numeric(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return float(node.value) in {0.0, 1.0}
    if isinstance(node, ast.Tuple):
        return any(_is_neutral_numeric(element) for element in node.elts)
    return False


def _handler_has_risky_return(handler: ast.ExceptHandler) -> bool:
    for child in ast.walk(handler):
        if isinstance(child, ast.Return) and child.value is not None:
            if _is_neutral_numeric(child.value):
                return True
    return False


def _source_segment(source: str, node: ast.AST) -> str:
    segment = ast.get_source_segment(source, node)
    if segment is not None:
        return segment
    return ""


def _audit_file(path: Path, root: Path) -> list[str]:
    source = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        rel = path.relative_to(root).as_posix()
        return [f"{rel}:{exc.lineno}: syntax error while auditing guard fallbacks"]

    errors: list[str] = []
    rel = path.relative_to(root).as_posix()
    for node in ast.walk(tree):
        if not isinstance(node, ast.ExceptHandler):
            continue
        if not _handler_has_risky_return(node):
            continue
        segment = _source_segment(source, node)
        if any(marker in segment for marker in DIAGNOSTIC_MARKERS):
            continue
        errors.append(
            f"{rel}:{node.lineno}: risky numeric fallback in except block must "
            "emit a diagnostic/log entry, raise, or include guard-fallback-ok rationale"
        )
    return errors


def check_guard_fallbacks(*, root: Path) -> tuple[bool, list[str]]:
    guard_root = root / GUARD_ROOT
    errors: list[str] = []
    for path in sorted(guard_root.glob("*.py")):
        errors.extend(_audit_file(path, root))
    return not errors, errors


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fail if guard except-block numeric fallbacks lack diagnostics."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=_repo_root_from_script(),
        help="Repository root. Defaults to the current checkout.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    ok, errors = check_guard_fallbacks(root=args.root.resolve())
    if ok:
        print("[check_guard_fallback_diagnostics] OK")
        return 0
    print("[check_guard_fallback_diagnostics] FAIL", file=sys.stderr)
    for error in errors:
        print(f"  - {error}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
