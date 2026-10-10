"""Explain additive score changes without modifying an evidence report.

This standalone recipe does not verify evidence, infer independence, compute an
interval, or change a decision. Supply complete paired case means and declared
units after the corresponding workflow has validated them. Average repetitions
within each case first. For ordinary equal-case means, give each case its own
unit. Convert captured scores with Fraction.from_float(float(score)); convert
judge decimal score strings directly with Fraction(score).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Any


@dataclass(frozen=True)
class PairedCase:
    case_id: str
    unit_id: str
    baseline: Fraction
    subject: Fraction


def mean_contributions(
    cases: Sequence[PairedCase], *, reported_delta: Fraction, limit: int = 20
) -> dict[str, Any]:
    """Reconcile exact contributions, the omitted remainder and display rounding.

    All fractions serialize as strings, so JSON conversion cannot silently round
    a value or overflow a binary64 number. Ranking is descriptive within this one
    additive metric; it does not rank policy violations or explain uncertainty.
    """
    if not cases:
        raise ValueError("complete paired cases are required")
    if type(limit) is not int or not 0 <= limit <= 200:
        raise ValueError("limit must be an integer between zero and 200")
    if not isinstance(reported_delta, Fraction):
        raise ValueError("reported_delta must be an exact Fraction")
    identifiers = set()
    for case in cases:
        if not isinstance(case, PairedCase):
            raise ValueError("cases must be PairedCase values")
        if not all(
            isinstance(v, str) and v.strip() for v in (case.case_id, case.unit_id)
        ):
            raise ValueError("case and unit IDs must be nonempty strings")
        if case.case_id in identifiers:
            raise ValueError("case IDs must be unique")
        identifiers.add(case.case_id)
        if not all(isinstance(v, Fraction) for v in (case.baseline, case.subject)):
            raise ValueError("case means must be exact Fractions")
    counts = Counter(case.unit_id for case in cases)
    values = []
    for case in cases:
        weight = Fraction(1, len(counts) * counts[case.unit_id])
        contribution = weight * (case.subject - case.baseline)
        values.append((case, weight, contribution))
    total = sum((value for _, _, value in values), Fraction())
    ranked = sorted(values, key=lambda item: (-abs(item[2]), item[0].case_id))[:limit]
    visible = sum((value for _, _, value in ranked), Fraction())
    return {
        "authority": "none",
        "scope": "Descriptive arithmetic; not evidence verification or causal attribution",
        "weighting": "Equal units, equal case means within each unit",
        "case_count": len(cases),
        "unit_count": len(counts),
        "exact_delta": str(total),
        "reported_delta": str(reported_delta),
        "rounding_residual": str(reported_delta - total),
        "visible_contribution": str(visible),
        "omitted_contribution": str(total - visible),
        "omitted_case_count": len(cases) - len(ranked),
        "cases": [
            {
                "case_id": case.case_id,
                "unit_id": case.unit_id,
                "weight": str(weight),
                "baseline": str(case.baseline),
                "subject": str(case.subject),
                "contribution": str(value),
            }
            for case, weight, value in ranked
        ],
    }


if __name__ == "__main__":
    import json

    print(
        json.dumps(
            mean_contributions(
                [
                    PairedCase("large", "large", Fraction(2**54), Fraction(2**54)),
                    PairedCase("small", "small", Fraction(0), Fraction(1)),
                ],
                reported_delta=Fraction(0),
            ),
            indent=2,
        )
    )
