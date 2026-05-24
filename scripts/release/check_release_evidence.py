#!/usr/bin/env python3
"""Validate local release-evidence artifacts before cutting a release."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _existing_globs(root: Path, patterns: tuple[str, ...]) -> list[Path]:
    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(path for path in root.glob(pattern) if path.is_file())
    return sorted(matches)


def _require_file(path: Path, label: str, failures: list[str]) -> None:
    if not path.is_file():
        failures.append(f"{label} missing: {path}")


def _require_any(
    root: Path, patterns: tuple[str, ...], label: str, failures: list[str]
) -> None:
    if not _existing_globs(root, patterns):
        joined = ", ".join(patterns)
        failures.append(f"{label} missing under {root}: {joined}")


def check_release_evidence(
    *,
    release_root: Path,
    dist_root: Path,
    sbom_path: Path,
) -> list[str]:
    failures: list[str] = []
    _require_any(dist_root, ("*.whl",), "wheel artifact", failures)
    _require_any(dist_root, ("*.tar.gz",), "sdist artifact", failures)
    _require_file(sbom_path, "SBOM", failures)
    _require_file(
        release_root / "wheel-sdist-hashes.txt", "wheel/sdist hashes", failures
    )
    _require_file(
        release_root / "runtime-image-digest.txt", "runtime image digest", failures
    )
    _require_file(
        release_root / "strict" / "evaluation.report.json",
        "strict example report",
        failures,
    )
    _require_file(
        release_root / "strict" / "verify.json",
        "strict verifier output",
        failures,
    )
    return failures


def _build_summary(
    *,
    release_root: Path,
    dist_root: Path,
    sbom_path: Path,
    failures: list[str],
) -> dict[str, object]:
    return {
        "schema": "invarlock/release-evidence-check-v1",
        "release_root": str(release_root),
        "dist_root": str(dist_root),
        "sbom_path": str(sbom_path),
        "ok": not failures,
        "failures": failures,
    }


def main(argv: list[str] | None = None) -> int:
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
        "--json",
        action="store_true",
        help="Emit a machine-readable summary.",
    )
    args = parser.parse_args(argv)

    release_root = Path(args.root)
    dist_root = Path(args.dist)
    sbom_path = Path(args.sbom)
    failures = check_release_evidence(
        release_root=release_root,
        dist_root=dist_root,
        sbom_path=sbom_path,
    )
    summary = _build_summary(
        release_root=release_root,
        dist_root=dist_root,
        sbom_path=sbom_path,
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
