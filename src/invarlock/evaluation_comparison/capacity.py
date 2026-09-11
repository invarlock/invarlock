"""Local capacity checks that leave complete comparison artifacts unchanged."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from invarlock.evaluation_record_contracts.contracts import EvaluationRecordsError
from invarlock.evidence_pack_contract import canonical_json_bytes

# One overall scalar over the supported 50,000 pairs uses 2,048 repetitions.
# This measured work allowance is not a time or memory guarantee.
DEFAULT_MAX_BOOTSTRAP_DRAWS = 102_400_000


def missing_pair(
    left: dict[str, Any], right: dict[str, Any], metric: dict[str, Any]
) -> bool:
    """Apply the shared missingness rule to already validated paired records."""
    return (
        left["error"] is not None
        or right["error"] is not None
        or (
            metric["kind"] == "recorded"
            and any(metric["score_key"] not in row["scores"] for row in (left, right))
        )
    )


def check_missing_id_capacity(
    scopes: Sequence[tuple[str, Sequence[tuple[dict[str, Any], dict[str, Any]]]]],
    metrics: Sequence[dict[str, Any]],
    *,
    byte_limit: int,
) -> int:
    """Reject only when actual missing-ID arrays alone exceed the supplied limit.

    Count the existing canonical UTF-8 representation, including each array's
    brackets and commas, without materializing any arrays. The complete output
    is larger, so acceptance here does not replace final artifact validation.
    The caller supplies the current comparison limit after validating inputs.
    """
    if type(byte_limit) is not int or byte_limit < 0:
        raise EvaluationRecordsError(
            "comparison byte limit must be a non-negative integer"
        )
    encoded_lengths: dict[str, int] = {}
    total = 0

    def check() -> None:
        if total > byte_limit:
            raise EvaluationRecordsError(
                f"comparison missing-ID arrays already require at least {total} bytes; "
                f"the byte limit is {byte_limit}. "
                "The complete policy has not been evaluated."
            )

    for _, selected in scopes:
        for metric in metrics:
            total += 2
            check()
            first = True
            for left, right in selected:
                if not missing_pair(left, right, metric):
                    continue
                identifier = left["id"]
                if identifier not in encoded_lengths:
                    encoded_lengths[identifier] = len(
                        canonical_json_bytes(identifier, newline=False)
                    )
                total += encoded_lengths[identifier] + (0 if first else 1)
                first = False
                check()
    return total
