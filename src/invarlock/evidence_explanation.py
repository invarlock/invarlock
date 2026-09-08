"""Human policy checks derived from an already validated canonical report.

These views are presentation only. Callers must authenticate and validate the
report first; this module never supplies verification authority or a verdict.
"""

from __future__ import annotations

from typing import Any, TypedDict


class PolicyCheck(TypedDict):
    name: str
    observed: str
    required: str
    passed: bool
    explanation: str


def _number(value: int | float) -> str:
    return f"{value:.6g}"


def core_policy_checks(report: dict[str, Any]) -> list[PolicyCheck]:
    """Present every configured check without changing the recorded decision."""
    comparison = report["comparison"]
    uncertainty = report["uncertainty"]
    kind = comparison["kind"]
    if kind in {"exact_match_delta_pp", "scorer_extension_delta_pp"}:
        observed, required = uncertainty["lower"], comparison["minimum"]
        label = (
            "Finite-schedule lower bound"
            if uncertainty["scope"] == "authenticated_schedule"
            else "Paired lower bound"
        )
        checks: list[PolicyCheck] = [
            {
                "name": label,
                "observed": f"{_number(observed)} pp",
                "required": f">= {_number(required)} pp",
                "passed": observed >= required,
                "explanation": "The lower bound must meet the allowed change, not just the point estimate.",
            }
        ]
    elif kind == "normalized_nll_ratio":
        observed, required = uncertainty["upper"], comparison["maximum"]
        checks = [
            {
                "name": "Finite-schedule upper ratio bound",
                "observed": _number(observed),
                "required": f"<= {_number(required)}",
                "passed": observed <= required,
                "explanation": "The upper ratio bound must stay within the allowed loss increase.",
            }
        ]
    else:
        raise ValueError("unsupported canonical comparison kind")
    sample = report.get("sample_qualification")
    if sample is not None:
        count, width = sample["record_count"], sample["interval_width"]
        unit = " pp" if width["unit"] == "percentage_points" else " ratio"
        checks.extend(
            [
                {
                    "name": "Record count",
                    "observed": str(count["observed"]),
                    "required": f">= {count['minimum']}",
                    "passed": count["passed"],
                    "explanation": "The comparison must include the configured minimum number of paired records.",
                },
                {
                    "name": "Interval width",
                    "observed": f"{_number(width['observed'])}{unit}",
                    "required": f"<= {_number(width['maximum'])}{unit}",
                    "passed": width["passed"],
                    "explanation": "The interval must be narrow enough for the configured precision requirement.",
                },
            ]
        )
    accuracy = report.get("side_accuracy")
    if accuracy is not None:
        for side in ("baseline", "subject"):
            checks.append(
                {
                    "name": f"{'Baseline' if side == 'baseline' else 'Candidate'} accuracy",
                    "observed": f"{_number(accuracy[side]['observed'] * 100)}%",
                    "required": f">= {_number(accuracy['minimum'] * 100)}%",
                    "passed": accuracy[side]["passed"],
                    "explanation": "Each side must meet the configured absolute accuracy floor.",
                }
            )
    return checks
