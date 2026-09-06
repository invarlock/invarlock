"""The literal-agreement domain for Inspect's pinned single-target match scorer."""

from __future__ import annotations

import string
from typing import Any

PROFILE_ID = "inspect-ai-literal-pairs-v1"
SCORER_CONFIGURATION = {"location": "exact", "ignore_case": False, "numeric": False}


def validate_cases(cases: list[dict[str, Any]]) -> None:
    """Reject ambiguous pairs, while retaining equal boundary whitespace."""

    identifiers = [case.get("record_id") for case in cases]
    if any(not isinstance(value, str) or not value for value in identifiers) or len(
        set(identifiers)
    ) != len(identifiers):
        raise ValueError("Inspect cases require unique nonempty record IDs")
    for case in cases:
        output, reference = case.get("output"), case.get("reference")
        if not isinstance(output, str) or not isinstance(reference, str):
            raise ValueError(
                "unsupported Inspect literal pair: one string target required"
            )
        # Inspect 0.3.254 applies Unicode strip(), then strips ASCII boundary
        # whitespace and punctuation. This is a domain check, never the metric.
        boundary = string.whitespace + string.punctuation
        native_equal = output.strip().strip(boundary) == reference.strip().strip(
            boundary
        )
        if native_equal != (output == reference):
            raise ValueError(
                "unsupported Inspect literal pair: boundary normalization collision"
            )


def project_result(case: dict[str, Any], result: Any) -> tuple[float, dict[str, Any]]:
    """Preserve raw output semantics and reject unexpected native score details."""

    validate_cases([case])
    if result.value not in ("C", "I"):
        raise ValueError("Inspect native score must be C or I")
    score = float(result.value == "C")
    if score != float(case["output"] == case["reference"]):
        raise ValueError("Inspect native score contradicts the supported literal pair")
    if result.answer != case["output"].strip() or result.explanation != case["output"]:
        raise ValueError("Inspect native answer or explanation contradicts the output")
    return score, {"answer": result.answer, "score_value": result.value}
