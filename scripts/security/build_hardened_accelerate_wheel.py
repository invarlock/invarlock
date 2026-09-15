#!/usr/bin/env python3
"""Derive and authenticate the maintained Accelerate checkpoint-loading fix."""

from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import sys
import textwrap
import urllib.request
import zipfile
from pathlib import Path

if __package__:
    from .build_cache_free_lm_eval_wheel import (
        DerivationError,
        build_derived_wheel,
        patch_metadata,
        validate_wheel_record,
    )
else:
    from build_cache_free_lm_eval_wheel import (
        DerivationError,
        build_derived_wheel,
        patch_metadata,
        validate_wheel_record,
    )

UPSTREAM_VERSION = "1.14.0"
DERIVED_VERSION = "1.14.0+invarlock.1"
HARDENED_VERSION = DERIVED_VERSION
UPSTREAM_WHEEL_SHA256 = (
    "e94390c2863b873be18f623f9df48a0d8fe5eff13ea7f1a00092b0a7904888c6"
)
# Update only after reviewing the complete derivation and its installed tests.
DERIVED_WHEEL_SHA256 = (
    "5406d926d81114b891df4d2abf27ba08d4b11c2c085ec1261d0bfc2d7e4225a6"
)
UPSTREAM_WHEEL_NAME = f"accelerate-{UPSTREAM_VERSION}-py3-none-any.whl"
DERIVED_WHEEL_NAME = f"accelerate-{DERIVED_VERSION}-py3-none-any.whl"
UPSTREAM_DIST_INFO = f"accelerate-{UPSTREAM_VERSION}.dist-info"
DERIVED_DIST_INFO = f"accelerate-{DERIVED_VERSION}.dist-info"
UPSTREAM_WHEEL_URL = (
    "https://files.pythonhosted.org/packages/a8/db/"
    "253133d7e7cb40d3af384bb2f5c0b4a2b7fdcffbc95c688cc67a20a3c103/"
    + UPSTREAM_WHEEL_NAME
)
REMEDIATED_ADVISORIES = frozenset(("GHSA-4j2p-28q2-5m79", "PYSEC-2026-3804"))
DEFAULT_WHEELHOUSE = Path(__file__).resolve().parents[2] / "runtime" / "wheels"
_HELPER = Path(__file__).with_name("accelerate_checkpoint_files.py")


def _patch_version(payload: bytes) -> bytes:
    original = f'__version__ = "{UPSTREAM_VERSION}"'.encode()
    if payload.count(original) != 1:
        raise DerivationError("upstream Accelerate runtime version changed")
    return payload.replace(original, f'__version__ = "{DERIVED_VERSION}"'.encode())


def _patch_metadata(payload: bytes) -> bytes:
    return patch_metadata(payload, UPSTREAM_VERSION, DERIVED_VERSION, ())


def _patch_modeling(payload: bytes) -> bytes:
    source = payload.decode("utf-8")
    signature = "def load_state_dict(checkpoint_file, device_map=None):\n"
    predicate = '    if checkpoint_file.endswith(".safetensors"):\n'
    if source.count(signature) != 1 or source.count(predicate) != 1:
        raise DerivationError("upstream Accelerate state loader changed")
    wrapper = (
        "def load_state_dict(checkpoint_file, device_map=None):\n"
        '    """Load a regular checkpoint through its pinned descriptor."""\n'
        "    with _checkpoint_file(checkpoint_file) as (path, is_safetensors):\n"
        "        return _load_state_dict_from_file(\n"
        "            path, device_map=device_map, is_safetensors=is_safetensors\n"
        "        )\n\n\n"
        "def _load_state_dict_from_file(\n"
        "    checkpoint_file, device_map=None, *, is_safetensors=False\n"
        "):\n"
    )
    source = source.replace(signature, _HELPER.read_text() + "\n\n" + wrapper, 1)
    source = source.replace(predicate, "    if is_safetensors:\n", 1)
    tree = ast.parse(source)
    functions = [
        item
        for item in tree.body
        if isinstance(item, ast.FunctionDef) and item.name == "load_checkpoint_in_model"
    ]
    if len(functions) != 1:
        raise DerivationError("upstream Accelerate checkpoint loader changed")
    function = functions[0]
    lines = source.splitlines(keepends=True)
    body = "".join(lines[function.lineno - 1 : function.end_lineno])
    start_marker = "    checkpoint_files = None\n    index_filename = None\n"
    end_marker = "    # Logic for missing/unexpected keys goes here.\n"
    if body.count(start_marker) != 1 or body.count(end_marker) != 1:
        raise DerivationError("upstream Accelerate checkpoint selection changed")
    start, end = body.index(start_marker), body.index(end_marker)
    if end <= start:
        raise DerivationError("upstream Accelerate checkpoint selection is reordered")
    body = (
        body[:start]
        + "    with _checkpoint_files(checkpoint) as checkpoint_files:\n"
        + textwrap.indent(body[end:], "    ")
    )
    lines[function.lineno - 1 : function.end_lineno] = [body]
    result = "".join(lines)
    ast.parse(result)
    return result.encode("utf-8")


def build_wheel(source: Path, output_directory: Path) -> Path:
    """Derive a wheel without modifying the original distribution or environment."""
    return build_derived_wheel(
        source,
        output_directory,
        upstream_sha256=UPSTREAM_WHEEL_SHA256,
        upstream_dist_info=UPSTREAM_DIST_INFO,
        derived_dist_info=DERIVED_DIST_INFO,
        derived_wheel_name=DERIVED_WHEEL_NAME,
        patches={
            f"{UPSTREAM_DIST_INFO}/METADATA": _patch_metadata,
            "accelerate/__init__.py": _patch_version,
            "accelerate/utils/modeling.py": _patch_modeling,
        },
        # Stored members make the reviewed artifact independent of zlib versions.
        compression=zipfile.ZIP_STORED,
    )


def verify_hardened_wheel(path: Path) -> dict[str, str]:
    """Authorize only the complete reviewed wheel, including all upstream files."""
    if path.is_symlink() or not path.is_file():
        raise DerivationError("maintained Accelerate wheel must be a regular file")
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != DERIVED_WHEEL_SHA256:
        raise DerivationError("maintained Accelerate wheel SHA-256 changed")
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        validate_wheel_record(archive)
        metadata = archive.read(f"{DERIVED_DIST_INFO}/METADATA")
        if (
            metadata.count(b"Name: accelerate\n") != 1
            or metadata.count(f"Version: {DERIVED_VERSION}\n".encode()) != 1
        ):
            raise DerivationError("maintained Accelerate wheel identity changed")
    return {
        "package": "accelerate",
        "upstream_version": UPSTREAM_VERSION,
        "upstream_wheel_sha256": UPSTREAM_WHEEL_SHA256,
        "version": DERIVED_VERSION,
        "wheel_sha256": digest,
    }


def bootstrap(output_directory: Path, *, source: Path | None = None) -> Path:
    """Prepare the authenticated runtime wheelhouse; never install a package."""
    output_directory.mkdir(parents=True, exist_ok=True)
    destination = output_directory / DERIVED_WHEEL_NAME
    if destination.exists() or destination.is_symlink():
        verify_hardened_wheel(destination)
        return destination
    if source is None:
        cache = output_directory / ".upstream"
        cache.mkdir(exist_ok=True)
        source = cache / UPSTREAM_WHEEL_NAME
        if not source.exists() or source.is_symlink():
            if source.is_symlink():
                raise DerivationError("upstream wheel cache must not be a symlink")
            with urllib.request.urlopen(UPSTREAM_WHEEL_URL, timeout=30) as response:
                payload = response.read(2 * 1024 * 1024 + 1)
            if hashlib.sha256(payload).hexdigest() != UPSTREAM_WHEEL_SHA256:
                raise DerivationError("downloaded upstream Accelerate wheel changed")
            with source.open("xb") as stream:
                stream.write(payload)
    destination = build_wheel(source, output_directory)
    verify_hardened_wheel(destination)
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build-wheel")
    build.add_argument("--input", type=Path, required=True)
    build.add_argument("--output-directory", type=Path, required=True)
    prepare = commands.add_parser("bootstrap")
    prepare.add_argument("--input", type=Path)
    prepare.add_argument("--output-directory", type=Path, default=DEFAULT_WHEELHOUSE)
    verify = commands.add_parser("verify-wheel")
    verify.add_argument("--input", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "verify-wheel":
            print(json.dumps(verify_hardened_wheel(args.input), sort_keys=True))
        elif args.command == "bootstrap":
            print(bootstrap(args.output_directory, source=args.input))
        else:
            print(build_wheel(args.input, args.output_directory))
    except (DerivationError, OSError, ValueError, zipfile.BadZipFile) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
