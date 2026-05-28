#!/usr/bin/env python3
"""Synchronize public contract JSON files into the packaged wheel data directory."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = REPO_ROOT / "contracts"
PACKAGED_DIR = REPO_ROOT / "src" / "invarlock" / "_data" / "contracts"


def _contract_map(directory: Path, *, allow_missing: bool = False) -> dict[str, Path]:
    if not directory.is_dir():
        if allow_missing:
            return {}
        raise FileNotFoundError(f"contract directory not found: {directory}")
    return {
        path.name: path
        for path in sorted(directory.glob("*.json"))
        if path.is_file() and not path.name.startswith(".")
    }


def collect_contract_drift(
    source_dir: Path = SOURCE_DIR, packaged_dir: Path = PACKAGED_DIR
) -> dict[str, list[str]]:
    source = _contract_map(source_dir)
    packaged = _contract_map(packaged_dir)

    source_names = set(source)
    packaged_names = set(packaged)

    missing = sorted(source_names - packaged_names)
    extra = sorted(packaged_names - source_names)
    changed = sorted(
        name
        for name in source_names & packaged_names
        if source[name].read_bytes() != packaged[name].read_bytes()
    )
    return {
        "missing": missing,
        "extra": extra,
        "changed": changed,
    }


def check_contract_sync(
    source_dir: Path = SOURCE_DIR, packaged_dir: Path = PACKAGED_DIR
) -> list[str]:
    drift = collect_contract_drift(source_dir, packaged_dir)
    errors: list[str] = []
    if drift["missing"]:
        errors.append(
            "missing packaged contracts: " + ", ".join(sorted(drift["missing"]))
        )
    if drift["extra"]:
        errors.append("extra packaged contracts: " + ", ".join(sorted(drift["extra"])))
    if drift["changed"]:
        errors.append(
            "out-of-sync packaged contracts: " + ", ".join(sorted(drift["changed"]))
        )
    return errors


def sync_packaged_contracts(
    source_dir: Path = SOURCE_DIR, packaged_dir: Path = PACKAGED_DIR
) -> tuple[int, int]:
    source = _contract_map(source_dir)
    packaged = _contract_map(packaged_dir, allow_missing=True)
    packaged_dir.mkdir(parents=True, exist_ok=True)

    updated = 0
    removed = 0

    for name, source_path in source.items():
        dest_path = packaged_dir / name
        if (
            not dest_path.is_file()
            or source_path.read_bytes() != dest_path.read_bytes()
        ):
            shutil.copyfile(source_path, dest_path)
            updated += 1

    for name, packaged_path in packaged.items():
        if name in source:
            continue
        packaged_path.unlink()
        removed += 1

    return updated, removed


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check or synchronize the packaged contract JSON files under "
            "src/invarlock/_data/contracts."
        )
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Copy repo contracts into the packaged contract directory and remove extras.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check whether packaged contracts match repo contracts (default).",
    )
    parser.add_argument(
        "--source-dir",
        default=str(SOURCE_DIR),
        help="Override the source contracts directory (tests/debugging).",
    )
    parser.add_argument(
        "--packaged-dir",
        default=str(PACKAGED_DIR),
        help="Override the packaged contracts directory (tests/debugging).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    source_dir = Path(args.source_dir)
    packaged_dir = Path(args.packaged_dir)
    write_mode = args.write
    if not args.write and not args.check:
        check_mode = True
    else:
        check_mode = args.check or not args.write

    if write_mode:
        try:
            updated, removed = sync_packaged_contracts(source_dir, packaged_dir)
        except FileNotFoundError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        print(
            "Synchronized packaged contracts from "
            f"{source_dir} -> {packaged_dir} (updated={updated}, removed={removed})."
        )

    if check_mode:
        try:
            errors = check_contract_sync(source_dir, packaged_dir)
        except FileNotFoundError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        if errors:
            for error in errors:
                print(error, file=sys.stderr)
            return 1
        print(
            f"Packaged contracts are in sync ({len(_contract_map(source_dir))} files)."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
