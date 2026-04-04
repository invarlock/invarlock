from __future__ import annotations

from invarlock.reporting import report_types


def test_validate_report_rejects_blank_primary_metric_kind() -> None:
    report = report_types.create_empty_report()
    report["metrics"]["primary_metric"] = {"kind": "   ", "final": 1.0}

    assert report_types.validate_report(report) is False


def test_validate_report_rejects_bool_primary_metric_final() -> None:
    report = report_types.create_empty_report()
    report["metrics"]["primary_metric"] = {"kind": "accuracy", "final": True}

    assert report_types.validate_report(report) is False


def test_validate_report_rejects_bool_meta_seed() -> None:
    report = report_types.create_empty_report()
    report["meta"]["seed"] = True
    report["metrics"]["primary_metric"] = {"kind": "accuracy", "final": 1.0}

    assert report_types.validate_report(report) is False
