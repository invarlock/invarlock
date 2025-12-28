from __future__ import annotations

# Policy-change regression baseline identifiers.
#
# When the benchmark golden outputs are intentionally updated, bump
# `BENCH_GOLDEN_ID` and update `BENCH_GOLDEN_SHA256` accordingly, then add a
# matching entry to `CHANGELOG.md`.

BENCH_GOLDEN_ID = "bench-golden-2025-12-13"
BENCH_GOLDEN_SHA256 = "2627b8872cd6bfc37bda31fbc11b78ed814751cbf2a9ad1396e173f1f4e5383a"

__all__ = ["BENCH_GOLDEN_ID", "BENCH_GOLDEN_SHA256"]
