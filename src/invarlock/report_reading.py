"""Plain-language readings of display facts; no new acceptance decisions."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from invarlock.report_presentation import MetricView


def score_scale(metric: MetricView) -> float | None:
    """Use an honest zero origin, fixed bounds where known, and no guessed units."""
    scores = metric.scores
    if scores is None or scores.kind not in {"accuracy", "rubric", "loss"}:
        return None
    values = (scores.baseline, scores.candidate)
    if any(v is None or not math.isfinite(v) or v < 0 for v in values):
        return None
    assert scores.baseline is not None and scores.candidate is not None
    maximum = max(scores.baseline, scores.candidate)
    if scores.kind in {"accuracy", "rubric"}:
        return 1.0 if maximum <= 1 else None
    return maximum if maximum > 0 else None


def score_caption(metric: MetricView) -> str:
    scores = metric.scores
    if scores is None:
        return "Scores use the recorded units; consult the scoring rule before interpreting the change."
    direction = {"higher": "Higher is better.", "lower": "Lower is better."}.get(
        scores.direction, "The preferred direction was not supplied."
    )
    description = {
        "accuracy": "Share of results that met the scoring rule.",
        "rubric": "Judge rating on a 0–1 rubric; not a percentage of correct answers.",
        "loss": "Text-prediction loss (NLL): how poorly the model predicted the reference text.",
    }.get(scores.kind, "Scores use the recorded units.")
    return f"{description} {direction}"


def range_reading(metric: MetricView) -> str:
    """Explain retained endpoints without assigning a population interpretation."""
    interval = metric.interval
    if interval is None:
        return "No change range is available. The scores alone cannot establish that the requirement was met."
    values = (interval.lower, interval.upper, interval.estimate)
    if not all(math.isfinite(v) for v in values) or interval.lower > interval.upper:
        return (
            "The change range is unavailable in this view; consult the recorded checks."
        )
    loss = metric.scores is not None and metric.scores.kind == "loss"
    # Percent changes are valid only for a retained ratio, never a guessed unit.
    relative = loss and interval.unit == "ratio" and interval.neutral == 1
    scale = 100 if relative else 1
    neutral = 1 if relative else 0

    def value(raw: float) -> str:
        transformed = (raw - neutral) * scale
        if not math.isfinite(transformed):
            return "unavailable"
        if transformed and abs(transformed) < 0.001:
            return f"{transformed:+.3g}"
        return f"{transformed:+.3f}".rstrip("0").rstrip(".")

    unit = (
        "%"
        if relative
        else " percentage points"
        if interval.unit == "pp"
        else f" {interval.unit}"
    )
    reading = f"{'Loss change' if relative else 'Change'} range: {value(interval.lower)} to {value(interval.upper)}{unit}."
    threshold = interval.threshold
    if threshold is not None and math.isfinite(threshold):
        if interval.threshold_direction == "minimum":
            reading += f" Required: lower end at least {value(threshold)}{unit}."
        elif interval.threshold_direction == "maximum":
            reading += f" Required: upper end at most {value(threshold)}{unit}."
        bound = (
            interval.lower
            if interval.threshold_direction == "minimum"
            else interval.upper
        )
        failed = (
            bound < threshold
            if interval.threshold_direction == "minimum"
            else bound > threshold
            if interval.threshold_direction == "maximum"
            else False
        )
        if failed:
            reading += f" Failure margin: {abs(bound - threshold)!r} {interval.unit} (unrounded bound {bound!r}; requirement {threshold!r})."
    if (
        interval.neutral is not None
        and interval.lower <= interval.neutral <= interval.upper
    ):
        reading += (
            " The range includes no change; it does not establish an improvement."
        )
    return reading


def observed_change(metric: MetricView) -> str:
    interval = metric.interval
    if (
        metric.scores is not None
        and metric.scores.kind == "loss"
        and interval is not None
        and interval.unit == "ratio"
        and interval.neutral == 1
        and math.isfinite(interval.estimate)
    ):
        change = (interval.estimate - 1) * 100
        if math.isfinite(change):
            return f"{change:+.3g}% loss"
    if interval is not None and interval.unit == "pp" and metric.change.endswith(" pp"):
        return metric.change.removesuffix(" pp") + " percentage points"
    return metric.change


def check_reason(name: str) -> str:
    """Translate legacy display labels once, before deriving guidance."""
    return {
        "Complete paired results": "missing",
        "Schedule completeness": "missing",
        "Included pair count": "count",
        "Record count": "count",
        "Independent units": "count",
        "Interval width": "precision",
        "Paired score change interval width": "precision",
        "Subject score bound interval width": "precision",
        "Subject minimum": "quality",
        "Subject maximum": "quality",
        "Subject score bound": "quality",
        "Candidate accuracy": "quality",
        "Baseline accuracy": "baseline_quality",
        "Allowed change": "change",
        "Maximum NLL ratio": "change",
        "Paired score change": "change",
        "Paired lower bound": "change",
        "Finite-schedule lower bound": "change",
        "Finite-schedule upper ratio bound": "change",
    }.get(name, "unresolved")


def suggested_action(metric: MetricView) -> str:
    """List every unresolved check; no first-match cause can hide another."""
    if metric.decision == "pass" and all(c.passed is True for c in metric.checks):
        return "Confirm that these tests and any minimum quality requirement cover the intended use before making a release decision."
    actions = {
        "missing": "Resolve the missing results and retain them in the record. Review the complete comparison before deciding.",
        "count": "Review the sampling plan and the required amount of independent evidence. Repeated ratings of the same cases do not replace independent evidence.",
        "precision": "The range is too wide or unavailable. Review the evaluation design and sources of uncertainty before collecting more evidence.",
        "quality": "Investigate the candidate's task quality against the recorded score requirement before proposing a replacement.",
        "baseline_quality": "Review whether the reference version is an acceptable quality baseline.",
        "unresolved": "Ask the evaluation owner to resolve the unavailable or unmet checks before using this result for a decision.",
    }
    interval = metric.interval
    outside = (
        interval is not None
        and interval.threshold is not None
        and (
            interval.estimate < interval.threshold
            if interval.threshold_direction == "minimum"
            else interval.estimate > interval.threshold
            if interval.threshold_direction == "maximum"
            else False
        )
    )
    actions["change"] = (
        "The observed change is outside the allowance. Investigate the affected cases and candidate behavior before proposing a replacement."
        if outside
        else "The observed score alone is not enough: the range does not establish that the allowance was met. Review the affected cases and uncertainty with the evaluation owner."
    )
    unresolved = [c for c in metric.checks if c.passed is not True]
    return (
        " ".join(
            f"{c.name}: {c.observed}; required {c.required}. {actions.get(c.reason_id, actions['unresolved'])}"
            for c in unresolved
        )
        or actions["unresolved"]
    )


def exact_number(value: float) -> str:
    """Round-trip representation with no redundant decimal zero."""
    text = str(value)
    return text[:-2] if text.endswith(".0") else text
