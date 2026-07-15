#!/usr/bin/env python3
"""Measure deterministic local-checkpoint hashing throughput and strict-run cost."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from invarlock.core.checkpoint_identity import (  # noqa: E402
    _EXCLUDED_DIRECTORY_NAMES,
    _checkpoint_file,
    checkpoint_tree_sha256,
)

STRICT_LOCAL_FULL_READS = 3
PROJECTIONS_GIB = {
    "7b_bf16_13.04_gib": 13.04,
    "32b_bf16_59.60_gib": 59.60,
}


def _checkpoint_bytes(root: Path) -> int:
    total = 0
    for item in root.rglob("*"):
        relative = item.relative_to(root)
        if any(part in _EXCLUDED_DIRECTORY_NAMES for part in relative.parts[:-1]):
            continue
        if not item.is_symlink() and item.is_file() and _checkpoint_file(item):
            total += item.stat(follow_symlinks=False).st_size
    return total


def benchmark_checkpoint(root: Path, *, repeat: int) -> dict[str, object]:
    checkpoint_bytes = _checkpoint_bytes(root)
    timings: list[float] = []
    digests: set[str] = set()
    for _ in range(repeat):
        started = time.perf_counter()
        digests.add(checkpoint_tree_sha256(root))
        timings.append(time.perf_counter() - started)
    if len(digests) != 1:
        raise RuntimeError("checkpoint identity changed between benchmark repetitions")
    seconds_per_hash = sum(timings) / len(timings)
    gib = checkpoint_bytes / (1024**3)
    gib_per_second = gib / seconds_per_hash
    projected = {
        label: (size_gib / gib_per_second) * STRICT_LOCAL_FULL_READS
        for label, size_gib in PROJECTIONS_GIB.items()
    }
    return {
        "schema": "invarlock.checkpoint_identity_benchmark.v1",
        "checkpoint": str(root.resolve()),
        "checkpoint_bytes": checkpoint_bytes,
        "repeat": repeat,
        "seconds_per_hash": seconds_per_hash,
        "gib_per_second": gib_per_second,
        "strict_local_full_reads": STRICT_LOCAL_FULL_READS,
        "projected_seconds": projected,
        "digest": next(iter(digests)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be at least 1")
    payload = benchmark_checkpoint(args.checkpoint, repeat=args.repeat)
    if args.json:
        print(json.dumps(payload, sort_keys=True))
    else:
        print(f"Checkpoint bytes: {payload['checkpoint_bytes']}")
        print(f"Seconds per hash: {payload['seconds_per_hash']:.6f}")
        print(f"Throughput GiB/s: {payload['gib_per_second']:.3f}")
        for label, seconds in payload["projected_seconds"].items():
            print(f"Projected strict-local {label}: {seconds:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
