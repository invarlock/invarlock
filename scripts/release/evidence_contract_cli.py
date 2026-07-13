"""Command-line interface for release evidence contract checks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import ModuleType


def _contracts() -> ModuleType:
    try:
        from scripts.release import evidence_contracts
    except ImportError:  # pragma: no cover - direct script execution path
        import evidence_contracts
    return evidence_contracts


def _add_release_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--root", default="artifacts/release")
    parser.add_argument("--dist", default="dist")
    parser.add_argument("--sbom", default="artifacts/supply-chain/sbom.json")
    parser.add_argument(
        "--guard-validation-json",
        default="artifacts/guard-validation/guard-validation-smoke.json",
    )
    parser.add_argument(
        "--guard-validation-md",
        default="artifacts/guard-validation/guard-validation-smoke.md",
    )
    parser.add_argument(
        "--offline-bundle-dir",
        default="artifacts/release/offline",
    )
    parser.add_argument("--json", action="store_true")


def _add_empirical_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--root",
        default="artifacts/guard-validation/empirical",
    )
    parser.add_argument("--json", action="store_true")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate release shape and diagnostic evidence inventories."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    release_parser = subparsers.add_parser(
        "release",
        help="Validate local release artifact shape only; not release approval.",
    )
    _add_release_args(release_parser)
    empirical_parser = subparsers.add_parser(
        "empirical-inventory",
        help=(
            "Validate a non-authoritative inventory of claimed empirical "
            "guard artifacts."
        ),
    )
    _add_empirical_args(empirical_parser)
    return parser.parse_args(argv)


def _run_release(args: argparse.Namespace) -> int:
    release_root = Path(args.root)
    dist_root = Path(args.dist)
    sbom_path = Path(args.sbom)
    guard_validation_json = Path(args.guard_validation_json)
    guard_validation_markdown = Path(args.guard_validation_md)
    offline_bundle_dir = Path(args.offline_bundle_dir)
    contracts = _contracts()
    failures = contracts.check_release_evidence(
        release_root=release_root,
        dist_root=dist_root,
        sbom_path=sbom_path,
        guard_validation_json=guard_validation_json,
        guard_validation_markdown=guard_validation_markdown,
        offline_bundle_dir=offline_bundle_dir,
    )
    summary = contracts._build_release_summary(
        release_root=release_root,
        dist_root=dist_root,
        sbom_path=sbom_path,
        guard_validation_json=guard_validation_json,
        guard_validation_markdown=guard_validation_markdown,
        offline_bundle_dir=offline_bundle_dir,
        failures=failures,
    )
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    elif failures:
        for failure in failures:
            print(f"ERROR: {failure}", file=sys.stderr)
    else:
        print("Release artifact-shape check passed; this is not release approval.")
    return 1 if failures else 0


def _run_empirical(args: argparse.Namespace) -> int:
    root = Path(args.root)
    contracts = _contracts()
    failures = contracts.check_empirical_guard_evidence(root=root)
    summary = contracts._build_empirical_summary(root=root, failures=failures)
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    elif failures:
        for failure in failures:
            print(f"ERROR: {failure}", file=sys.stderr)
    else:
        print(
            "Empirical guard artifact inventory is structurally valid; "
            "this is diagnostic only and cannot authorize release or "
            "calibration claims."
        )
    return 1 if failures else 0


def run_cli(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.command == "release":
        return _run_release(args)
    if args.command == "empirical-inventory":
        return _run_empirical(args)
    raise AssertionError(f"Unhandled evidence command: {args.command}")
