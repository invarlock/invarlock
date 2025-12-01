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
import shutil
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path


def run(cmd: list[str]) -> tuple[int, str]:
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    out, _ = proc.communicate()
    return proc.returncode, out


def collect_markdown_files() -> list[str]:
    files: list[str] = []
    for p in [Path("README.md"), Path("CONTRIBUTING.md")]:
        if p.exists():
            files.append(str(p))
    for md in sorted(Path("docs").rglob("*.md")):
        files.append(str(md))
    return files


def lint_markdown(files: Iterable[str]) -> None:
    files = list(files)
    if not files:
        print("[docs_lint] No markdown files found; skipping markdownlint")
        return

    # Prefer markdownlint (CLI v1), then markdownlint-cli2, falling back to npx
    ml_cmd: list[str] | None = None
    if shutil.which("markdownlint"):
        ml_cmd = ["markdownlint", *files]
    elif shutil.which("markdownlint-cli2"):
        ml_cmd = ["markdownlint-cli2", *files]
    elif shutil.which("npx"):
        # Try cli v1 first, then cli2
        code, _ = run(["npx", "--yes", "markdownlint-cli", "--version"])
        if code == 0:
            ml_cmd = ["npx", "--yes", "markdownlint-cli", *files]
        else:
            ml_cmd = ["npx", "--yes", "markdownlint-cli2", *files]

    if not ml_cmd:
        print("[docs_lint] markdownlint not available and npx missing", file=sys.stderr)
        raise SystemExit(1)

    code, out = run(ml_cmd)
    print(out, end="")
    if code != 0:
        raise SystemExit(code)


def lint_spell(files: Iterable[str]) -> None:
    files = list(files)
    if not files:
        print("[docs_lint] No markdown files found; skipping cspell")
        return

    sp_cmd: list[str] | None = None
    if shutil.which("cspell"):
        sp_cmd = ["cspell", *files]
    elif shutil.which("npx"):
        sp_cmd = ["npx", "--yes", "cspell", *files]

    if not sp_cmd:
        print("[docs_lint] cspell not available and npx missing", file=sys.stderr)
        raise SystemExit(1)

    code, out = run(sp_cmd)
    print(out, end="")
    # Some cspell outputs non-zero for suggestions; allow override via env if needed
    if code != 0:
        raise SystemExit(code)


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
    summary = {"markdown": None, "spell": None}
    try:
        if args.all or args.markdown:
            lint_markdown(files)
            summary["markdown"] = True
        if args.all or args.spell:
            lint_spell(files)
            summary["spell"] = True
    except SystemExit as e:
        print(
            json.dumps({"ok": False, "summary": summary, "exit": e.code}),
            file=sys.stderr,
        )
        raise

    print(json.dumps({"ok": True, "summary": summary}))


if __name__ == "__main__":
    main()
