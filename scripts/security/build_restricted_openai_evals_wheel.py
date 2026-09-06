#!/usr/bin/env python3
"""Derive the OpenAI Evals basic.Match image profile without unused NLTK."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from build_cache_free_lm_eval_wheel import (
    DerivationError,
    build_derived_wheel,
    patch_metadata,
    validate_wheel_record,
)
from build_cache_free_lm_eval_wheel import (
    filter_lock as _filter_lock,
)

UPSTREAM_VERSION = "3.0.1.post1"
DERIVED_VERSION = "3.0.1.post1+invarlock.match.1"
UPSTREAM_WHEEL_SHA256 = (
    "0abcb2051303500784b1641a6e4f6b813ed43ad64f879a37d344a6774eb8eb78"
)
UPSTREAM_DIST_INFO = f"evals-{UPSTREAM_VERSION}.dist-info"
DERIVED_DIST_INFO = f"evals-{DERIVED_VERSION}.dist-info"
DERIVED_WHEEL_NAME = f"evals-{DERIVED_VERSION}-py3-none-any.whl"
REMOVED_REQUIREMENTS = frozenset(("evals", "nltk"))

__all__ = ["build_wheel", "filter_lock", "validate_wheel_record"]


def _patch_metadata(payload: bytes) -> bytes:
    return patch_metadata(
        payload,
        UPSTREAM_VERSION,
        DERIVED_VERSION,
        (b"Requires-Dist: nltk\n",),
    )


def build_wheel(source: Path, output_directory: Path) -> Path:
    """Keep every upstream code byte; alter only metadata and rebuild RECORD."""

    return build_derived_wheel(
        source,
        output_directory,
        upstream_sha256=UPSTREAM_WHEEL_SHA256,
        upstream_dist_info=UPSTREAM_DIST_INFO,
        derived_dist_info=DERIVED_DIST_INFO,
        derived_wheel_name=DERIVED_WHEEL_NAME,
        patches={f"{UPSTREAM_DIST_INFO}/METADATA": _patch_metadata},
    )


def filter_lock(source: Path, destination: Path) -> None:
    _filter_lock(source, destination, removed_requirements=REMOVED_REQUIREMENTS)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    wheel = commands.add_parser("build-wheel")
    wheel.add_argument("--input", type=Path, required=True)
    wheel.add_argument("--output-directory", type=Path, required=True)
    lock = commands.add_parser("filter-lock")
    lock.add_argument("--input", type=Path, required=True)
    lock.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "build-wheel":
            print(build_wheel(args.input, args.output_directory))
        else:
            filter_lock(args.input, args.output)
    except (DerivationError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
