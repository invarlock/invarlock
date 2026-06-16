from __future__ import annotations

from invarlock.core.assurance_contract import REPORT_BUILD_EVENT_CATEGORIES


def ensure_report_build_evidence(report: dict[str, object]) -> dict[str, object]:
    section = report.setdefault("report_build", {})
    if not isinstance(section, dict):
        section = {}
        report["report_build"] = section
    for category in REPORT_BUILD_EVENT_CATEGORIES:
        events = section.get(category)
        if not isinstance(events, list):
            section[category] = []
    return section


def record_report_build_event(
    report: dict[str, object],
    *,
    category: str,
    field: str,
    reason: str,
    source: str,
) -> None:
    if category not in REPORT_BUILD_EVENT_CATEGORIES:
        raise ValueError(f"Unknown report-build event category: {category}")
    section = ensure_report_build_evidence(report)
    events = section[category]
    events.append(
        {
            "field": str(field),
            "reason": str(reason),
            "source": str(source),
        }
    )


def report_build_has_evidence_events(report: dict[str, object]) -> bool:
    section = report.get("report_build")
    if not isinstance(section, dict):
        return False
    for category in REPORT_BUILD_EVENT_CATEGORIES:
        events = section.get(category)
        if isinstance(events, list) and bool(events):
            return True
    return False


__all__ = [
    "REPORT_BUILD_EVENT_CATEGORIES",
    "ensure_report_build_evidence",
    "record_report_build_event",
    "report_build_has_evidence_events",
]
