from __future__ import annotations

# Policy-change regression baseline identifiers.
#
# When the benchmark golden outputs are intentionally updated, bump
# `BENCH_GOLDEN_ID` and update `BENCH_GOLDEN_SHA256` accordingly, then add a
# matching entry to `CHANGELOG.md`.

BENCH_GOLDEN_ID = "bench-golden-2025-12-13"
BENCH_GOLDEN_SHA256 = "0d9ff3274d29dad16ad580b4a0cf37b4f89e4f7c2e4345ce3d30a39f146ff5a7"

__all__ = ["BENCH_GOLDEN_ID", "BENCH_GOLDEN_SHA256"]
