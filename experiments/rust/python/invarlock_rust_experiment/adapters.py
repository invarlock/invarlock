"""Reversible, explicit candidate adapters for the installed baseline package."""

from __future__ import annotations

import hashlib
import re
from contextlib import contextmanager
from dataclasses import dataclass


def optimized_interval(*, metric, baseline_scores, subject_scores, schedule_sha256):
    """Only hoist invariant prefix/counter encodings; preserve hashes and arithmetic."""
    from invarlock import evidence_pack_contract as contract

    if re.fullmatch(r"[a-f0-9]{64}", schedule_sha256) is None:
        raise contract.EvidencePackError("paired records schedule_sha256 is invalid")
    count = len(baseline_scores)
    if count == 1:
        value = contract._comparison_value(metric, baseline_scores, subject_scores)
        return value, value
    seed = bytes.fromhex(schedule_sha256)
    draw_bytes = [draw.to_bytes(8, "big") for draw in range(count)]
    distribution = []
    for replicate in range(contract.PAIRED_INTERVAL_REPLICATES):
        prefix = seed + replicate.to_bytes(8, "big")
        indexes = []
        for encoded in draw_bytes:
            digest = hashlib.sha256(prefix + encoded).digest()
            indexes.append(int.from_bytes(digest[:8], "big") % count)
        distribution.append(
            contract._comparison_value(
                metric,
                [baseline_scores[index] for index in indexes],
                [subject_scores[index] for index in indexes],
            )
        )
    alpha = (1.0 - contract.PAIRED_INTERVAL_CONFIDENCE) / 2.0
    return (
        contract._percentile(distribution, alpha),
        contract._percentile(distribution, 1.0 - alpha),
    )


@dataclass
class InvocationCounter:
    calls: int = 0
    native_calls: int = 0


@contextmanager
def candidate_adapter(candidate):
    """Patch both aliases and restore each original object on every exit.

    There is no automatic fallback. A missing/broken native extension fails.
    This serial experiment adapter is intentionally not thread safe.
    """
    from invarlock import evidence_pack_contract as contract
    from invarlock.evaluation_comparison import likelihood

    originals = (
        contract._paired_resampling_interval,
        likelihood._paired_resampling_interval,
    )
    if candidate == "reference":
        selected = originals[0]
    elif candidate == "optimized":
        selected = optimized_interval
    elif candidate == "rust":
        from invarlock_native import paired_resampling_interval

        selected = paired_resampling_interval
    else:
        raise ValueError(f"unknown candidate: {candidate}")
    counter = InvocationCounter()

    def counted(**kwargs):
        counter.calls += 1
        if candidate == "rust":
            counter.native_calls += 1
        return selected(**kwargs)

    contract._paired_resampling_interval = counted
    likelihood._paired_resampling_interval = counted
    try:
        yield counter
    finally:
        contract._paired_resampling_interval, likelihood._paired_resampling_interval = (
            originals
        )
