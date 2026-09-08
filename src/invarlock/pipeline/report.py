"""Human reports derived from explicit pipeline evidence and policy checks."""

from __future__ import annotations

import json
import math
from typing import Any, cast
from xml.etree.ElementTree import Element, SubElement, tostring

from invarlock.evidence_pack_contract import canonical_json_bytes
from invarlock.pipeline.contracts import PipelineError, digest, validate
from invarlock.report_presentation import (
    CheckView,
    IntervalView,
    MetricView,
    ReportView,
    number,
)
from invarlock.report_presentation import (
    render_html as render_report_html,
)
from invarlock.report_presentation import (
    render_markdown as render_report_markdown,
)


def _configuration_preview(value: Any) -> dict[str, Any]:
    """Bound advanced configuration display without copying its full subtrees."""
    remaining = 128

    def text(item: Any, depth: int) -> str:
        nonlocal remaining
        if remaining == 0:
            return "… node budget reached"
        remaining -= 1
        if isinstance(item, str):
            shortened = item[:256]
            if len(item) > 256:
                shortened += f"… ({len(item)} characters; preview)"
            return json.dumps(shortened, ensure_ascii=False)
        if isinstance(item, (dict, list)):
            if depth >= 6:
                return "… depth limit reached"
            parts = []
            children = item.items() if isinstance(item, dict) else enumerate(item)
            for key, child in children:
                if remaining == 0:
                    parts.append("… node budget reached")
                    break
                prefix = text(key, depth + 1) + ": " if isinstance(item, dict) else ""
                parts.append(prefix + text(child, depth + 1))
            return (
                ("{" if isinstance(item, dict) else "[")
                + ", ".join(parts)
                + ("}" if isinstance(item, dict) else "]")
            )
        return json.dumps(item, allow_nan=False, ensure_ascii=False)

    return {
        "preview_only": True,
        "limits": {"depth": 6, "nodes": 128, "string_characters": 256},
        "text": text(value, 0),
        "note": "Display preview, not a replacement configuration. Full configuration remains in evidence under the original policy binding.",
    }


def _policy_preview(policy: dict[str, Any]) -> dict[str, Any]:
    return {
        **policy,
        "metrics": [
            {**metric, "configuration": _configuration_preview(metric["configuration"])}
            for metric in policy["metrics"]
        ],
    }


def _view(comparison: dict[str, Any], evidence: dict[str, Any] | None) -> ReportView:
    validate(comparison, "comparison")
    policy_metrics: dict[str, dict[str, Any]] = {}
    identity: list[tuple[str, str]] = []
    if evidence is not None:
        validate(evidence, "evidence")
        if canonical_json_bytes(evidence["comparison"]) != canonical_json_bytes(
            comparison
        ):
            raise PipelineError("report comparison differs from supplied evidence")
        if comparison["bindings"] != {
            k: digest(evidence[k]) for k in ("baseline", "candidate", "policy")
        }:
            raise PipelineError("report inputs differ from the comparison bindings")
        policy_metrics = {m["name"]: m for m in evidence["policy"]["metrics"]}
        for role in ("baseline", "candidate"):
            run = evidence[role]
            identity.extend(
                (
                    (role.title() + " run", run["run_id"]),
                    (role.title() + " artifact", run["artifact_digest"]),
                )
            )
        signing = (
            "Unsigned local comparison. No signature is available for independent authentication."
            if evidence["signature"] is None
            else "Signature present. This rendering has not authenticated it against a recipient-owned key."
        )
    else:
        signing = "Signing state unavailable: this view was created from comparison data only."
    identity.extend(
        (name.title() + " binding", value)
        for name, value in comparison["bindings"].items()
    )
    metrics: list[MetricView] = []
    for m in comparison["metrics"]:
        policy = policy_metrics.get(m["name"])
        binary = m["kind"] in {
            "exact_match",
            "normalized_match",
            "numeric_tolerance",
            "json_exact",
        }
        display_values = [m["baseline_mean"], m["candidate_mean"], m["delta"]]
        if m["interval"]:
            display_values.extend((m["interval"]["lower"], m["interval"]["upper"]))
        if policy:
            display_values.extend(
                policy.get(key)
                for key in (
                    "maximum_regression",
                    "maximum_interval_width",
                    "candidate_minimum",
                    "candidate_maximum",
                )
            )
        percentage = binary and all(
            math.isfinite(value * 100) for value in display_values if value is not None
        )
        scale = 100 if percentage else 1
        unit = "pp" if percentage else m["unit"]
        suffix = "%" if percentage else " " + m["unit"]
        missing = len(m["missing_ids"])
        complete = m["count"] - missing
        checks: list[CheckView] = [
            CheckView(
                "Complete paired results",
                f"{complete:,} of {m['count']:,}",
                "All included pairs",
                missing == 0,
                "Missing results are retained; they cannot be dropped to obtain a pass.",
            )
        ]
        if policy:
            checks.append(
                CheckView(
                    "Included pair count",
                    f"{m['count']:,}",
                    f">= {policy['minimum_count']:,}",
                    m["count"] >= policy["minimum_count"],
                )
            )
        else:
            checks.append(
                CheckView(
                    "Policy thresholds",
                    "Not supplied to renderer",
                    "Bound policy to explain requirements",
                    None,
                )
            )
        interval = m["interval"]
        visual = None
        if interval:
            higher = m["direction"] == "higher"
            threshold = (
                (
                    -policy["maximum_regression"]
                    if higher
                    else policy["maximum_regression"]
                )
                if policy
                else None
            )
            label = (
                "Paired 95% confidence interval"
                if binary
                else "95% paired-schedule resampling interval"
            )
            visual = IntervalView(
                interval["lower"] * scale,
                interval["upper"] * scale,
                m["delta"] * scale,
                threshold * scale if threshold is not None else None,
                label,
                unit,
            )
            if policy:
                bound = interval["lower" if higher else "upper"]
                checks.append(
                    CheckView(
                        "Allowed change",
                        number(bound * scale) + " " + unit,
                        (">= " if higher else "<= ")
                        + number(cast(float, threshold) * scale)
                        + " "
                        + unit,
                        bound >= threshold if higher else bound <= threshold,
                        "Uses the interval bound, not just the observed change.",
                    )
                )
                width = interval["upper"] - interval["lower"]
                checks.append(
                    CheckView(
                        "Interval width",
                        number(width * scale) + " " + unit,
                        "<= "
                        + number(policy["maximum_interval_width"] * scale)
                        + " "
                        + unit,
                        width <= policy["maximum_interval_width"],
                    )
                )
                for key, label, minimum in (
                    ("candidate_minimum", "Candidate minimum", True),
                    ("candidate_maximum", "Candidate maximum", False),
                ):
                    if key in policy:
                        value = m["candidate_mean"]
                        checks.append(
                            CheckView(
                                label,
                                number(value * scale) + suffix,
                                (">= " if minimum else "<= ")
                                + number(policy[key] * scale)
                                + suffix,
                                value >= policy[key]
                                if minimum
                                else value <= policy[key],
                            )
                        )
        elif policy:
            checks.append(
                CheckView(
                    "Score and uncertainty",
                    "Not evaluated",
                    "Complete results and sufficient count",
                    None,
                )
            )
        unmet = [c.name for c in checks if c.passed is False]
        explanation = (
            "All configured checks passed."
            if m["decision"] == "pass"
            else (
                "More evidence is needed: "
                + (", ".join(unmet) if unmet else "; ".join(m["reasons"]))
                + "."
                if m["decision"] == "insufficient_evidence"
                else "The policy was not met: "
                + (", ".join(unmet) if unmet else "; ".join(m["reasons"]))
                + "."
            )
        )
        notes = [
            "Higher values are better."
            if m["direction"] == "higher"
            else "Lower values are better.",
            "Scoring: recomputed from recorded expected and output values."
            if m["scoring_assurance"] == "recomputed"
            else "Scoring: recorded external measurements or judgments; aggregation is recomputed.",
            f"{m['count']:,} included pairs; {missing:,} missing paired results. Counts in overlapping slices must not be added together.",
        ]
        if policy is None:
            notes.append(
                "Requirements are unavailable in this comparison-only view. The original decision is displayed without independent replay."
            )
        metrics.append(
            MetricView(
                name=m["name"],
                scope=m["slice"],
                decision=m["decision"],
                baseline="Unavailable"
                if m["baseline_mean"] is None
                else number(m["baseline_mean"] * scale) + suffix,
                candidate="Unavailable"
                if m["candidate_mean"] is None
                else number(m["candidate_mean"] * scale) + suffix,
                change="Unavailable"
                if m["delta"] is None
                else number(m["delta"] * scale, signed=True) + " " + unit,
                count=f"{complete:,}",
                explanation=explanation,
                checks=tuple(checks),
                interval=visual,
                notes=tuple(notes),
            )
        )
    # Bring actionable findings first; original order and exact values remain in evidence.
    metrics.sort(
        key=lambda m: {"regression": 0, "insufficient_evidence": 1, "pass": 2}[
            m.decision
        ]
    )
    failed = sum(m.decision == "regression" for m in metrics)
    insufficient = sum(m.decision == "insufficient_evidence" for m in metrics)
    summary = (
        "All configured metric and slice checks passed for these recorded runs."
        if comparison["decision"] == "pass"
        else (
            " ".join(
                (
                    [f"{failed} metric / scope results did not meet policy."]
                    if failed
                    else []
                )
                + (
                    [f"{insufficient} metric / scope results need more evidence."]
                    if insufficient
                    else []
                )
                + ["Review the affected checks before accepting this change."]
            )
        )
    )
    # Human detail remains compact even when evidence lists many missing case IDs.
    technical = {
        **comparison,
        "metrics": [
            {k: v for k, v in m.items() if k != "missing_ids"}
            | {
                "missing_count": len(m["missing_ids"]),
                "missing_ids_preview": m["missing_ids"][:20],
                "missing_ids_preview_is_complete": len(m["missing_ids"]) <= 20,
            }
            for m in comparison["metrics"]
        ],
    }
    return ReportView(
        title="InvarLock pipeline comparison",
        family="Existing evaluation results",
        decision=comparison["decision"],
        summary=summary,
        metrics=tuple(metrics),
        assurance=(
            (
                "Input checks",
                "Evidence structure and comparison bindings checked."
                if evidence
                else "Comparison structure checked; evidence inputs were not supplied.",
            ),
            ("Signing", signing),
            (
                "Independent recipient verification",
                "Not performed by report. Use pipeline verify with independently supplied inputs.",
            ),
        ),
        identity=tuple(identity),
        subjects=tuple(
            (role.title() + " run", evidence[role]["run_id"])
            for role in ("baseline", "candidate")
        )
        if evidence
        else (),
        next_steps=(
            "Review failed or insufficient checks and their approved requirements.",
            "For a signed handoff, run invarlock pipeline verify with the authorized public key, policy and expected complete-run digests.",
            "Use the comparison JSON and JUnit outputs in CI; preserve missing results and the original policy.",
        ),
        limitations=tuple(comparison["limitations"]),
        details=(
            (
                "Bound policy (configuration preview)",
                _policy_preview(evidence["policy"]),
            ),
        )
        if evidence
        else (),
        technical=technical,
    )


def render_markdown(
    comparison: dict[str, Any], *, evidence: dict[str, Any] | None = None
) -> str:
    return render_report_markdown(_view(comparison, evidence))


def render_html(
    comparison: dict[str, Any], *, evidence: dict[str, Any] | None = None
) -> str:
    return render_report_html(_view(comparison, evidence))


def render_reports(
    comparison: dict[str, Any], *, evidence: dict[str, Any] | None = None
) -> tuple[str, str]:
    """Build one validated view for HTML and Markdown publication."""
    view = _view(comparison, evidence)
    return render_report_html(view), render_report_markdown(view)


def render_junit(comparison: dict[str, Any]) -> bytes:
    validate(comparison, "comparison")
    metrics = comparison["metrics"]
    root = Element(
        "testsuite",
        name="InvarLock release comparison",
        tests=str(len(metrics)),
        failures=str(sum(m["decision"] == "regression" for m in metrics)),
        errors=str(sum(m["decision"] == "insufficient_evidence" for m in metrics)),
    )
    for metric in metrics:
        case = SubElement(
            root, "testcase", name=metric["name"], classname=metric["slice"]
        )
        if metric["decision"] != "pass":
            SubElement(
                case,
                "failure" if metric["decision"] == "regression" else "error",
                message="; ".join(metric["reasons"]),
            )
    return cast(bytes, tostring(root, encoding="utf-8", xml_declaration=True))
