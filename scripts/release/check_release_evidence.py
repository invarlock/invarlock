#!/usr/bin/env python3
"""Validate local release-evidence artifacts before cutting a release."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:  # pragma: no branch
    sys.path.insert(0, str(SCRIPT_DIR))

from evidence_contracts import (  # noqa: E402
    DistHashManifest,
    GuardValidationSmokeManifest,
    ReleaseEvidenceManifest,
    StrictReportEvidence,
    StrictVerifyEvidence,
    existing_globs,
    load_json,
    require_any,
    require_file,
    sha256,
)


def _existing_globs(root: Path, patterns: tuple[str, ...]) -> list[Path]:
    return existing_globs(root, patterns)


def _require_file(path: Path, label: str, failures: list[str]) -> None:
    require_file(path, label, failures)


def _require_any(
    root: Path, patterns: tuple[str, ...], label: str, failures: list[str]
) -> None:
    require_any(root, patterns, label, failures)


def _dist_artifacts(dist_root: Path) -> list[Path]:
    return existing_globs(dist_root, ("*.whl", "*.tar.gz"))


def _sha256(path: Path) -> str:
    return sha256(path)


def _load_json(path: Path, label: str, failures: list[str]) -> object | None:
    return load_json(path, label, failures)


def _parse_hash_entries(path: Path, failures: list[str]) -> dict[str, str]:
    return DistHashManifest.load(path, failures).entries


def _validate_dist_hashes(
    *, dist_root: Path, hash_path: Path, failures: list[str]
) -> None:
    if not _dist_artifacts(dist_root) or not hash_path.is_file():
        return
    manifest = DistHashManifest.load(hash_path, failures)
    if not manifest.entries:
        failures.append(f"wheel/sdist hashes file has no valid entries: {hash_path}")
        return
    manifest.validate_artifacts(dist_root=dist_root, failures=failures)


def _validate_runtime_digest(path: Path, failures: list[str]) -> None:
    ReleaseEvidenceManifest._validate_runtime_digest(path, failures)


def _validate_strict_report(path: Path, failures: list[str]) -> None:
    StrictReportEvidence.load(path, failures).validate(failures)


def _validate_strict_verify(path: Path, report_path: Path, failures: list[str]) -> None:
    StrictVerifyEvidence.load(path, failures).validate(
        report_path=report_path,
        failures=failures,
    )


def _validate_guard_validation(
    *, json_path: Path, markdown_path: Path, failures: list[str]
) -> None:
    GuardValidationSmokeManifest.load(
        json_path=json_path,
        markdown_path=markdown_path,
        failures=failures,
    ).validate(failures)


def _validate_sbom(path: Path, failures: list[str]) -> None:
    payload = load_json(path, "SBOM", failures)
    if payload is not None and not isinstance(payload, dict):
        failures.append("SBOM must be a JSON object.")


def _validate_offline_bundle(offline_bundle_dir: Path, failures: list[str]) -> None:
    manifest = ReleaseEvidenceManifest(
        release_root=Path(),
        dist_root=Path(),
        sbom_path=Path(),
        guard_validation_json=Path(),
        guard_validation_markdown=Path(),
        offline_bundle_dir=offline_bundle_dir,
    )
    manifest._validate_offline_bundles(failures)


def check_release_evidence(
    *,
    release_root: Path,
    dist_root: Path,
    sbom_path: Path,
    guard_validation_json: Path,
    guard_validation_markdown: Path,
    offline_bundle_dir: Path,
) -> list[str]:
    manifest = ReleaseEvidenceManifest(
        release_root=release_root,
        dist_root=dist_root,
        sbom_path=sbom_path,
        guard_validation_json=guard_validation_json,
        guard_validation_markdown=guard_validation_markdown,
        offline_bundle_dir=offline_bundle_dir,
    )
    return manifest.validate()


def _build_summary(
    *,
    release_root: Path,
    dist_root: Path,
    sbom_path: Path,
    guard_validation_json: Path,
    guard_validation_markdown: Path,
    offline_bundle_dir: Path,
    failures: list[str],
) -> dict[str, object]:
    manifest = ReleaseEvidenceManifest(
        release_root=release_root,
        dist_root=dist_root,
        sbom_path=sbom_path,
        guard_validation_json=guard_validation_json,
        guard_validation_markdown=guard_validation_markdown,
        offline_bundle_dir=offline_bundle_dir,
    )
    return manifest.summary(failures)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the local release-evidence bundle."
    )
    parser.add_argument(
        "--root", default="artifacts/release", help="Release evidence root."
    )
    parser.add_argument(
        "--dist", default="dist", help="Distribution artifact directory."
    )
    parser.add_argument(
        "--sbom",
        default="artifacts/supply-chain/sbom.json",
        help="Install/tool surface SBOM JSON path.",
    )
    parser.add_argument(
        "--guard-validation-json",
        default="artifacts/guard-validation/guard-validation-smoke.json",
        help="Guard-validation smoke JSON artifact path.",
    )
    parser.add_argument(
        "--guard-validation-md",
        default="artifacts/guard-validation/guard-validation-smoke.md",
        help="Guard-validation smoke Markdown artifact path.",
    )
    parser.add_argument(
        "--offline-bundle-dir",
        default="artifacts/release/offline",
        help="Directory containing offline release bundle tarballs.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit a machine-readable summary.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    release_root = Path(args.root)
    dist_root = Path(args.dist)
    sbom_path = Path(args.sbom)
    guard_validation_json = Path(args.guard_validation_json)
    guard_validation_markdown = Path(args.guard_validation_md)
    offline_bundle_dir = Path(args.offline_bundle_dir)
    failures = check_release_evidence(
        release_root=release_root,
        dist_root=dist_root,
        sbom_path=sbom_path,
        guard_validation_json=guard_validation_json,
        guard_validation_markdown=guard_validation_markdown,
        offline_bundle_dir=offline_bundle_dir,
    )
    summary = _build_summary(
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
        print("Release evidence check passed.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
