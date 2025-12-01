#!/usr/bin/env python3
"""Validate that a certificate includes release-profile provenance, guard overhead,
and spectral observability fields."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def validate_cert(path: Path) -> list[str]:
    errors: list[str] = []
    if not path.exists():
        return [f"certificate not found: {path}"]

    try:
        data = json.loads(path.read_text())
    except Exception as exc:  # pragma: no cover - defensive
        return [f"failed to parse JSON: {exc}"]

    profile = data.get("provenance", {}).get("window_plan", {}).get("profile")
    if profile != "release":
        errors.append("window_plan.profile must be 'release'")

    guard = data.get("guard_overhead")
    if not isinstance(guard, dict) or "overhead_percent" not in guard:
        errors.append("guard_overhead.overhead_percent missing")
    if not isinstance(guard, dict) or "source" not in guard:
        errors.append("guard_overhead.source missing")

    fq = data.get("spectral", {}).get("family_z_quantiles")
    if not (isinstance(fq, dict) and fq):
        errors.append("spectral.family_z_quantiles missing or empty")

    return errors


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: check_cert_release_fields.py <cert.json>", file=sys.stderr)
        return 1

    errors = validate_cert(Path(argv[1]))
    if errors:
        for err in errors:
            print(f"[cert-check] {err}", file=sys.stderr)
        return 1

    print("[cert-check] release cert fields present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
