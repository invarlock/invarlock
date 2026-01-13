#!/usr/bin/env python3
"""Compare two certificates and fail if the PM ratio drift exceeds tolerance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_cert(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive guard
        raise SystemExit(f"Failed to parse certificate {path}: {exc}") from exc


def _extract_ratio(payload: dict) -> float:
    pm = payload.get("primary_metric")
    if isinstance(pm, dict):
        ratio = pm.get("ratio_vs_baseline")
        if isinstance(ratio, int | float):
            return float(ratio)
    raise SystemExit(
        "Certificate missing ratio (expected 'primary_metric.ratio_vs_baseline')"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "reference", type=Path, help="Reference certificate (e.g., CPU)"
    )
    parser.add_argument(
        "candidate", type=Path, help="Comparator certificate (e.g., GPU)"
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.01,
        help="Maximum allowed absolute ratio drift (default: 0.01).",
    )
    args = parser.parse_args()

    ref = _load_cert(args.reference)
    cand = _load_cert(args.candidate)

    ratio_ref = _extract_ratio(ref)
    ratio_cand = _extract_ratio(cand)
    drift = abs(ratio_ref - ratio_cand)

    if drift > args.tolerance:
        raise SystemExit(
            f"Device drift exceeded tolerance: |{ratio_ref:.6f} - {ratio_cand:.6f}| = "
            f"{drift:.6f} > {args.tolerance:.6f}"
        )

    print(
        f"Device drift OK: |{ratio_ref:.6f} - {ratio_cand:.6f}| = {drift:.6f} "
        f"≤ {args.tolerance:.6f}"
    )


if __name__ == "__main__":
    main()
