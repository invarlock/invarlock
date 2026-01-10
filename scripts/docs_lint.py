#!/usr/bin/env python3
"""Docs lint wrapper: markdownlint + cspell via local or npx.

Usage examples:
  python scripts/docs_lint.py --all
  python scripts/docs_lint.py --markdown
  python scripts/docs_lint.py --spell
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path


def run(cmd: list[str], *, timeout_s: float | None = None) -> tuple[int, str]:
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    try:
        out, _ = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        proc.kill()
        out, _ = proc.communicate()
        return 124, (out or "") + f"\n[docs_lint] Timed out after {timeout_s}s\n"
    return proc.returncode, out


def collect_markdown_files() -> list[str]:
    files: list[str] = []
    for p in [Path("README.md"), Path("CONTRIBUTING.md")]:
        if p.exists():
            files.append(str(p))
    for md in sorted(Path("docs").rglob("*.md")):
        files.append(str(md))
    return files


def _local_node_bin(name: str) -> str | None:
    candidates = [Path("node_modules") / ".bin" / name]
    if sys.platform.startswith("win"):
        candidates.insert(0, Path("node_modules") / ".bin" / f"{name}.cmd")
    for p in candidates:
        if p.exists():
            return str(p)
    return None


def lint_markdown(files: Iterable[str]) -> bool:
    files = list(files)
    if not files:
        print("[docs_lint] No markdown files found; skipping markdownlint")
        return False

    # Prefer markdownlint (CLI v1), then markdownlint-cli2, falling back to local node_modules,
    # finally using npx (which may require network).
    ml_cmd: list[str] | None = None
    timeout_s: float | None = None
    if shutil.which("markdownlint"):
        ml_cmd = ["markdownlint", *files]
    elif shutil.which("markdownlint-cli2"):
        ml_cmd = ["markdownlint-cli2", *files]
    else:
        local_cli2 = _local_node_bin("markdownlint-cli2")
        if local_cli2:
            ml_cmd = [local_cli2, *files]
        elif shutil.which("npx"):
            allow_install = os.environ.get(
                "DOCS_LINT_ALLOW_NPX_INSTALL"
            ) == "1" or bool(os.environ.get("CI"))
            if not allow_install:
                message = (
                    "[docs_lint] markdownlint not installed (install via `npm ci`, "
                    "or set DOCS_LINT_ALLOW_NPX_INSTALL=1 to allow npx fetching)"
                )
                if os.environ.get("CI"):
                    print(message, file=sys.stderr)
                    raise SystemExit(1)
                print(f"{message}; skipping", file=sys.stderr)
                return False

            timeout_s = float(os.environ.get("DOCS_LINT_NPX_TIMEOUT_SECONDS", "120"))
            ml_cmd = [
                "npx",
                "--yes",
                "--package",
                "markdownlint-cli2",
                "--",
                "markdownlint-cli2",
                *files,
            ]

    if not ml_cmd:
        message = "[docs_lint] markdownlint not available and npx missing"
        if os.environ.get("CI"):
            print(message, file=sys.stderr)
            raise SystemExit(1)
        print(f"{message}; skipping", file=sys.stderr)
        return False

    code, out = run(ml_cmd, timeout_s=timeout_s)
    print(out, end="")
    if code == 124 and timeout_s is not None:
        print(
            "[docs_lint] markdownlint via npx timed out (run `npm ci` for offline use)",
            file=sys.stderr,
        )
        raise SystemExit(code)
    if code != 0:
        raise SystemExit(code)
    return True


def lint_spell(files: Iterable[str]) -> bool:
    files = list(files)
    if not files:
        print("[docs_lint] No markdown files found; skipping cspell")
        return False

    sp_cmd: list[str] | None = None
    timeout_s: float | None = None
    if shutil.which("cspell"):
        sp_cmd = ["cspell", *files]
    else:
        local_cspell = _local_node_bin("cspell")
        if local_cspell:
            sp_cmd = [local_cspell, *files]
        elif shutil.which("npx"):
            allow_install = os.environ.get(
                "DOCS_LINT_ALLOW_NPX_INSTALL"
            ) == "1" or bool(os.environ.get("CI"))
            if not allow_install:
                message = (
                    "[docs_lint] cspell not installed (set DOCS_LINT_ALLOW_NPX_INSTALL=1 "
                    "to allow npx fetching)"
                )
                if os.environ.get("CI"):
                    print(message, file=sys.stderr)
                    raise SystemExit(1)
                print(f"{message}; skipping", file=sys.stderr)
                return False
            timeout_s = float(os.environ.get("DOCS_LINT_NPX_TIMEOUT_SECONDS", "120"))
            sp_cmd = ["npx", "--yes", "--package", "cspell", "--", "cspell", *files]

    if not sp_cmd:
        message = "[docs_lint] cspell not available and npx missing"
        if os.environ.get("CI"):
            print(message, file=sys.stderr)
            raise SystemExit(1)
        print(f"{message}; skipping", file=sys.stderr)
        return False

    code, out = run(sp_cmd, timeout_s=timeout_s)
    print(out, end="")
    if code == 124 and timeout_s is not None:
        print("[docs_lint] cspell via npx timed out", file=sys.stderr)
        raise SystemExit(code)
    # Some cspell outputs non-zero for suggestions; allow override via env if needed
    if code != 0:
        raise SystemExit(code)
    return True


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Docs lint wrapper")
    p.add_argument("--all", action="store_true", help="Run markdownlint and cspell")
    p.add_argument("--markdown", action="store_true", help="Run markdownlint only")
    p.add_argument("--spell", action="store_true", help="Run cspell only")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not any([args.all, args.markdown, args.spell]):
        print("No lints selected. Use --all or --markdown/--spell.", file=sys.stderr)
        raise SystemExit(2)

    files = collect_markdown_files()
    summary: dict[str, bool | None] = {"markdown": None, "spell": None}
    try:
        if args.all or args.markdown:
            summary["markdown"] = lint_markdown(files)
        if args.all or args.spell:
            summary["spell"] = lint_spell(files)
    except SystemExit as e:
        print(
            json.dumps({"ok": False, "summary": summary, "exit": e.code}),
            file=sys.stderr,
        )
        raise

    print(json.dumps({"ok": True, "summary": summary}))


if __name__ == "__main__":
    main()
