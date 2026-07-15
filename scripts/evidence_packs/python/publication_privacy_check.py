#!/usr/bin/env python3
"""Run the repository's public-text privacy gate before pack publication."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from scripts.checks.public_evidence_checks.common import (  # noqa: E402
    _check_public_evidence_privacy,
)


def publication_privacy_errors(pack_dir: Path) -> list[str]:
    errors: list[str] = []
    _check_public_evidence_privacy(errors, pack_dir)
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("pack_dir", type=Path)
    args = parser.parse_args()
    if args.pack_dir.is_symlink() or not args.pack_dir.is_dir():
        parser.error("pack directory must be a regular directory")
    errors = publication_privacy_errors(args.pack_dir)
    for error in errors:
        print(f"ERROR: {error}", file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
