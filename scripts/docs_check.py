#!/usr/bin/env python3
"""Consolidated documentation checks.

Runs the common docs validations in one place so CI and local developers can
invoke a single entry point. Use flags to select subsets or --all for the full
suite.

Examples:
  python scripts/docs_check.py --all
  python scripts/docs_check.py --build --links
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> tuple[int, str]:
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    out, _ = proc.communicate()
    return proc.returncode, out


def check_build() -> None:
    code, out = run(["mkdocs", "build", "--strict"])
    print(out, end="")
    if code != 0:
        raise SystemExit(code)


def check_links() -> None:
    # Global link check (fast)
    code, out = run([sys.executable, "scripts/check_docs_links.py"])
    print(out, end="")
    if code != 0:
        raise SystemExit(code)

    # Per-file internal link lint
    for md in Path("docs").rglob("*.md"):
        code, out = run([sys.executable, "scripts/check_internal_links.py", str(md)])
        # keep going but report failures
        if code != 0:
            print(out, end="")
            raise SystemExit(code)


def check_references() -> None:
    code, out = run([sys.executable, "scripts/validate_doc_references.py"])
    print(out, end="")
    if code != 0:
        raise SystemExit(code)


def check_examples() -> None:
    code, out = run([sys.executable, "scripts/validate_yaml_examples.py"])
    print(out, end="")
    if code != 0:
        raise SystemExit(code)

    code, out = run([sys.executable, "scripts/validate_python_examples.py"])
    print(out, end="")
    if code != 0:
        raise SystemExit(code)

    # CLI examples extractor may be absent in minimal OSS; skip gracefully.
    cli_tester = Path("scripts/test_cli_examples.py")
    if cli_tester.exists():
        code, out = run([sys.executable, str(cli_tester)])
        print(out, end="")
        if code != 0:
            raise SystemExit(code)
    else:
        print(
            "[docs_check] Skipping CLI examples test (scripts/test_cli_examples.py not found)"
        )


def check_consistency() -> None:
    checks = [
        "scripts/check_version_consistency.py",
        "scripts/check_config_schema_sync.py",
        "scripts/check_guard_completeness.py",
    ]
    for path in checks:
        code, out = run([sys.executable, path])
        print(out, end="")
        if code != 0:
            raise SystemExit(code)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Consolidated docs checks")
    p.add_argument("--all", action="store_true", help="Run all checks")
    p.add_argument("--build", action="store_true", help="Build MkDocs strictly")
    p.add_argument(
        "--links", action="store_true", help="Run link checks (global + internal)"
    )
    p.add_argument("--refs", action="store_true", help="Validate doc references")
    p.add_argument(
        "--examples",
        action="store_true",
        help="Validate YAML/Python examples and CLI snippets if available",
    )
    p.add_argument(
        "--consistency",
        action="store_true",
        help="Run version/CLI/schema/guards consistency checks",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not any(
        [args.all, args.build, args.links, args.refs, args.examples, args.consistency]
    ):
        print("No checks selected. Use --all or individual flags.", file=sys.stderr)
        raise SystemExit(2)

    summary = {
        "build": None,
        "links": None,
        "refs": None,
        "examples": None,
        "consistency": None,
    }

    try:
        if args.all or args.build:
            check_build()
            summary["build"] = True
        if args.all or args.links:
            check_links()
            summary["links"] = True
        if args.all or args.refs:
            check_references()
            summary["refs"] = True
        if args.all or args.examples:
            check_examples()
            summary["examples"] = True
        if args.all or args.consistency:
            check_consistency()
            summary["consistency"] = True
    except SystemExit as e:
        # On failure, print machine-readable summary and bubble up exit code
        print(
            json.dumps({"ok": False, "summary": summary, "exit": e.code}),
            file=sys.stderr,
        )
        raise

    print(json.dumps({"ok": True, "summary": summary}))


if __name__ == "__main__":
    main()
