#!/usr/bin/env python3
"""Derive the local Match profile without NLTK or Snowflake recording."""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import tempfile
import urllib.request
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
DERIVED_VERSION = "3.0.1.post1+invarlock.match.2"
UPSTREAM_WHEEL_SHA256 = (
    "0abcb2051303500784b1641a6e4f6b813ed43ad64f879a37d344a6774eb8eb78"
)
UPSTREAM_DIST_INFO = f"evals-{UPSTREAM_VERSION}.dist-info"
DERIVED_DIST_INFO = f"evals-{DERIVED_VERSION}.dist-info"
DERIVED_WHEEL_NAME = f"evals-{DERIVED_VERSION}-py3-none-any.whl"
REMOVED_REQUIREMENTS = frozenset(("evals",))
UPSTREAM_WHEEL_NAME = f"evals-{UPSTREAM_VERSION}-py3-none-any.whl"
UPSTREAM_WHEEL_URL = (
    "https://files.pythonhosted.org/packages/7f/e4/"
    "b54a8285cd6bece2722fb2091570b70df061a9c7821aa460f48717bb6bda/"
    + UPSTREAM_WHEEL_NAME
)
DEFAULT_WHEELHOUSE = Path(__file__).resolve().parents[2] / "runtime/wheels"

__all__ = ["build_wheel", "filter_lock", "validate_wheel_record"]


def _patch_metadata(payload: bytes) -> bytes:
    return patch_metadata(
        payload,
        UPSTREAM_VERSION,
        DERIVED_VERSION,
        (
            b"Requires-Dist: nltk\n",
            b"Requires-Dist: snowflake-connector-python[pandas]\n",
        ),
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
    """Remove the separately installed derived wheel from its resolved closure."""
    for line in source.read_text().splitlines():
        if line.split("==", 1)[0] in {"nltk", "snowflake-connector-python"}:
            raise DerivationError("unsupported dependency in restricted Evals lock")
    _filter_lock(source, destination, removed_requirements=REMOVED_REQUIREMENTS)


def bootstrap(output_directory: Path, *, source: Path | None = None) -> Path:
    """Authenticate and rederive the resolver input before accepting its bytes."""
    output_directory.mkdir(parents=True, exist_ok=True)
    if source is None:
        cache = output_directory / ".upstream"
        cache.mkdir(exist_ok=True)
        source = cache / UPSTREAM_WHEEL_NAME
        if source.is_symlink():
            raise DerivationError("upstream wheel cache must not be a symlink")
        if not source.exists():
            with urllib.request.urlopen(UPSTREAM_WHEEL_URL, timeout=60) as response:
                payload = response.read(64 * 1024 * 1024 + 1)
            if hashlib.sha256(payload).hexdigest() != UPSTREAM_WHEEL_SHA256:
                raise DerivationError("downloaded upstream Evals wheel changed")
            with source.open("xb") as stream:
                stream.write(payload)
    destination = output_directory / DERIVED_WHEEL_NAME
    with tempfile.TemporaryDirectory(
        dir=output_directory, prefix=".evals-build-"
    ) as tmp:
        derived = build_wheel(source, Path(tmp))
        if destination.exists() or destination.is_symlink():
            if destination.is_symlink() or not destination.is_file():
                raise DerivationError("derived wheel must be a regular file")
            if destination.read_bytes() != derived.read_bytes():
                raise DerivationError("derived Evals wheel bytes changed")
        else:
            os.replace(derived, destination)
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    wheel = commands.add_parser("build-wheel")
    wheel.add_argument("--input", type=Path, required=True)
    wheel.add_argument("--output-directory", type=Path, required=True)
    prepare = commands.add_parser("bootstrap")
    prepare.add_argument("--input", type=Path)
    prepare.add_argument("--output-directory", type=Path, default=DEFAULT_WHEELHOUSE)
    lock = commands.add_parser("filter-lock")
    lock.add_argument("--input", type=Path, required=True)
    lock.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "build-wheel":
            print(build_wheel(args.input, args.output_directory))
        elif args.command == "bootstrap":
            print(bootstrap(args.output_directory, source=args.input))
        else:
            filter_lock(args.input, args.output)
    except (DerivationError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
