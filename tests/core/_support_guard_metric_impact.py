from __future__ import annotations

from typing import Any

from invarlock.eval.guard_metric_impact import (
    REQUIRED_GUARD_METRIC_IMPACT_CHECKS,
    build_guard_metric_bare_report,
    compute_guard_metric_impact,
    extract_guard_metric_arm_facts,
    guard_metric_schedule_digest,
)


def bind_guard_metric_impact_evidence(report: dict[str, Any]) -> None:
    """Bind a fixture's retained impact evidence to its final evaluation arm."""

    metric_impact = report.get("guard_metric_impact")
    if not isinstance(metric_impact, dict):
        return
    metric_kind = metric_impact.get("metric_kind")
    guarded_facts = extract_guard_metric_arm_facts(report, metric_kind)
    bare_report = build_guard_metric_bare_report(report, metric_kind)
    bare_facts = extract_guard_metric_arm_facts(bare_report, metric_kind)
    measurement = compute_guard_metric_impact(
        metric_kind,
        metric_impact.get("bare_value"),
        metric_impact.get("guarded_value"),
    )
    schedule_digest = guard_metric_schedule_digest(report, metric_kind)
    assert measurement is not None
    assert bare_report is not None
    bare_report["status"] = "success"
    assert bare_facts is not None
    assert guarded_facts is not None
    assert schedule_digest is not None
    metric_impact.update(measurement.to_metrics())
    metric_impact.update(
        {
            "bare_facts": bare_facts,
            "guarded_facts": guarded_facts,
            "bare_report": bare_report,
            "schedule_digest": schedule_digest,
            "checks": dict.fromkeys(REQUIRED_GUARD_METRIC_IMPACT_CHECKS, True),
        }
    )
