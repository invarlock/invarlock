from __future__ import annotations

import pytest

from invarlock.reporting.report_builder_support import (
    ensure_report_build_evidence,
    record_report_build_event,
    report_build_has_evidence_events,
)
from tests._repo_root import REPO_ROOT


def test_ensure_report_build_evidence_replaces_non_dict_section() -> None:
    report: dict[str, object] = {"report_build": "bad"}

    section = ensure_report_build_evidence(report)

    assert report["report_build"] is section
    assert section == {
        "synthesized_fields": [],
        "repaired_fields": [],
        "fallback_fields": [],
    }


def test_ensure_report_build_evidence_preserves_existing_event_lists() -> None:
    report: dict[str, object] = {
        "report_build": {
            "synthesized_fields": [{"field": "x"}],
            "repaired_fields": [],
            "fallback_fields": [],
        }
    }

    section = ensure_report_build_evidence(report)

    assert section["synthesized_fields"] == [{"field": "x"}]


def test_record_report_build_event_rejects_unknown_category() -> None:
    with pytest.raises(ValueError, match="Unknown report-build event category"):
        record_report_build_event(
            {},
            category="estimated_fields",
            field="primary_metric.display_ci",
            reason="test",
            source="test",
        )


def test_report_build_has_evidence_events_handles_missing_or_non_list_events() -> None:
    assert report_build_has_evidence_events({}) is False
    assert report_build_has_evidence_events({"report_build": "bad"}) is False
    assert (
        report_build_has_evidence_events(
            {"report_build": {"synthesized_fields": "bad"}}
        )
        is False
    )


def test_report_build_has_evidence_events_detects_events() -> None:
    report: dict[str, object] = {}

    record_report_build_event(
        report,
        category="synthesized_fields",
        field="primary_metric.display_ci",
        reason="missing_report_field",
        source="test",
    )

    assert report_build_has_evidence_events(report) is True


def test_primary_metric_repair_paths_record_structured_evidence() -> None:
    primary_metric_utils = (
        REPO_ROOT / "src" / "invarlock" / "reporting" / "primary_metric_utils.py"
    ).read_text(encoding="utf-8")
    report_enrichment = (
        REPO_ROOT / "src" / "invarlock" / "reporting" / "report_enrichment.py"
    ).read_text(encoding="utf-8")

    for required in (
        'field="primary_metric.ratio_vs_baseline"',
        'field="primary_metric.display_ci"',
        'field="primary_metric"',
        'category="fallback_fields"',
        'category="synthesized_fields"',
        'category="repaired_fields"',
    ):
        assert required in primary_metric_utils

    assert "record_report_build_event(" in report_enrichment
    assert 'field="primary_metric.display_ci"' in report_enrichment
