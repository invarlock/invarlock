"""Common critical facts, derived only from validated presentation records."""

from __future__ import annotations

from typing import Any

from invarlock.report_presentation import ReportView, decision_label


def owner_context(
    context: dict[str, Any], view: ReportView
) -> tuple[tuple[str, str], ...]:
    """Preserve legacy prose without inferring an approval from arbitrary text."""
    attribution = (
        ("Authored review date", context["review_date"]),
        ("Review attribution", context["byline"]),
    )
    if context["format"].endswith("-v1"):
        return attribution + (
            (
                "Owner decision state",
                "Legacy unstructured commentary; no approval state inferred. A byline does not authenticate authority.",
            ),
        )
    state = context["decision_state"]
    if state in {"accepted", "conditional"} and view.decision != "pass":
        if not context.get("exception_rationale"):
            raise ValueError(
                "owner acceptance with unmet or unavailable policy requires exception_rationale"
            )
        state = "exception (owner " + state + "; recorded policy remains unchanged)"
    rows = [
        (
            "Owner decision state",
            state + ". Authored statement; byline does not authenticate authority.",
        ),
        ("Intended use", context["intended_use"]),
        ("Excluded uses", context["excluded_uses"]),
        ("Decision conditions", context["decision_conditions"]),
        ("Owner next action", context["next_action"]),
    ]
    if context.get("exception_rationale"):
        rows.append(("Owner exception rationale", context["exception_rationale"]))
    return attribution + tuple(rows)


def generated_facts(view: ReportView) -> tuple[tuple[str, str], ...]:
    rows = [
        (
            "Recorded policy result",
            decision_label(view.decision) + ". This is not release approval.",
        ),
        ("Comparison", view.summary),
        *view.subjects,
        (
            "Evidence authentication",
            " | ".join(
                f"{k}: {v}" for k, v in view.assurance if "recipient" not in k.lower()
            )
            or "Unknown; not supplied",
        ),
        (
            "Recipient verification",
            "Not performed for this recipient. An attachment is not an authenticated check.",
        ),
        (
            "Observation window",
            "Only observation windows retained with the model/service identities apply. No report date establishes freshness.",
        ),
        (
            "Review date",
            view.review.byline
            if view.review
            else "See the separately attributed review context; not the observation date.",
        ),
        ("Verification time", "Unavailable; no authenticated checking time supplied."),
    ]
    for i, m in sorted(
        enumerate(view.metrics, 1),
        key=lambda item: (not item[1].required, item[1].decision == "pass"),
    ):
        floor = {
            "absent": "No absolute minimum score is required by this policy.",
            "unknown": "Absolute score requirement is unknown in this view.",
            "configured": "An absolute score requirement is configured; see its individual checks.",
        }.get(m.quality_floor, "Absolute score requirement is unavailable.")
        rows.append(
            (
                f"Result {i}: {m.display_name} / {m.display_scope}",
                f"{decision_label(m.decision)}; {'required' if m.required else 'advisory'}. Baseline {m.baseline}; candidate {m.candidate}. {floor}",
            )
        )
        for c in m.checks:
            if c.passed is not True:
                rows.append(
                    (
                        f"Result {i} unmet check: {c.name}",
                        f"{c.observed}; required {c.required}. {c.explanation}",
                    )
                )
    rows.extend((k, v) for k, v in view.context if "observation window" in k.lower())
    rows.append(
        (
            "Coverage",
            "Each result retains its own paired or independent-unit count. Attempted, excluded and missing totals beyond the retained records are unknown; overlapping groups and repeated ratings cannot be added.",
        )
    )
    rows.append(
        (
            "Operational tradeoffs",
            "No operational benefit is inferred from a quality comparison. Bound supplementary observations, when present, appear in the technical details and do not become acceptance requirements.",
        )
    )
    rows.extend(("Evidence limit", text) for text in view.limitations)
    return tuple(rows)
