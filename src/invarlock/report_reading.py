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


def suggested_action(metric: MetricView) -> str:
    """Link a bounded follow-up to the actual unmet checks, without changing policy."""
    if metric.decision == "pass":
        return "Confirm that these tests and any minimum quality requirement cover the intended use before making a release decision."
    unresolved = [check for check in metric.checks if check.passed is not True]
    names = {check.name for check in unresolved}
    if names & {"Complete paired results", "Schedule completeness"}:
        return "Resolve the missing results and retain them in the record. Review the complete comparison before deciding."
    if names & {"Included pair count", "Record count", "Independent units"}:
        return "Review the sampling plan and the required amount of independent evidence. Repeated ratings of the same cases do not replace independent evidence."
    if any("interval width" in name.lower() for name in names):
        return "The range is too wide or unavailable. Review the evaluation design and sources of uncertainty before collecting more evidence."
    if names & {
        "Subject minimum",
        "Subject maximum",
        "Subject score bound",
        "Candidate accuracy",
    }:
        return "Investigate the candidate's task quality against the recorded score requirement before proposing a replacement."
    if "Baseline accuracy" in names:
        return "Review whether the reference version is an acceptable quality baseline."
    if names & {
        "Allowed change",
        "Maximum NLL ratio",
        "Paired score change",
        "Paired lower bound",
        "Finite-schedule lower bound",
        "Finite-schedule upper ratio bound",
    }:
        interval = metric.interval
        if interval is not None and interval.threshold is not None:
            outside = (
                interval.estimate < interval.threshold
                if interval.threshold_direction == "minimum"
                else interval.estimate > interval.threshold
                if interval.threshold_direction == "maximum"
                else False
            )
            if outside:
                return "The observed change is outside the allowance. Investigate the affected cases and candidate behavior before proposing a replacement."
        return "The observed score alone is not enough: the range does not establish that the allowance was met. Review the affected cases and uncertainty with the evaluation owner."
    return "Ask the evaluation owner to resolve the unavailable or unmet checks before using this result for a decision."
