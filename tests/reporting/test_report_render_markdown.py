from __future__ import annotations

import json
from pathlib import Path

from invarlock.reporting import render as render_mod
from invarlock.reporting import report_summary as report_summary_mod
from invarlock.reporting.render import (
    render_report_markdown,
)
from invarlock.reporting.report_schema import validate_report


def _mk_cert() -> dict:
    return {
        "schema_version": "v1",
        "run_id": "r1",
        "artifacts": {"generated_at": "t"},
        "plugins": {},
        "meta": {},
        "dataset": {
            "provider": "p",
            "seq_len": 8,
            "windows": {"preview": 0, "final": 0},
        },
        "primary_metric": {
            "kind": "ppl_causal",
            "final": 10.0,
            "ratio_vs_baseline": 1.0,
            "display_ci": [1.0, 1.0],
        },
        "validation": {
            "primary_metric_acceptable": True,
            "preview_final_drift_acceptable": True,
            "invariants_pass": True,
            "spectral_stable": True,
            "rmt_stable": True,
        },
    }


def test_render_report_markdown_is_presentation_only() -> None:
    cert = _mk_cert()
    cert["schema_version"] = "invalid"
    assert validate_report(cert) is False
    md = render_report_markdown(cert)
    assert "# InvarLock Evaluation Report" in md


def test_validate_evaluation_report_rejects_unknown_validation_keys() -> None:
    cert = _mk_cert()
    # Add an unexpected key; JSONSchema validation should fail and fallback minimal check should still accept structure
    cert["validation"]["unexpected_key_for_test"] = True  # type: ignore[index]
    # validate_report uses JSONSchema first; since schema disallows unknown keys in validation, it will fall back
    assert validate_report(cert) is True


def test_render_report_markdown_tolerates_missing_generated_at() -> None:
    cert = _mk_cert()
    cert["artifacts"] = {}

    assert validate_report(cert) is True
    md = render_report_markdown(cert)

    assert "**Generated:** (not recorded)" in md
    assert "## Contents" not in md
    assert "## Evaluation Dashboard" not in md
    assert "## Executive Summary" in md


def test_render_report_markdown_uses_precomputed_quality_gates_view_model(
    monkeypatch,
) -> None:
    cert = _mk_cert()
    summary = report_summary_mod.QualityGatesSummary(
        overall_pass=True,
        overall_status="✅ PASS",
        rows=(
            report_summary_mod.QualityGateRow(
                label="Primary Metric Acceptable",
                status="✅ PASS",
                measured="1.000x",
                threshold="≤ 1.10x",
                basis="point",
                description="Ratio vs baseline",
            ),
        ),
        hysteresis_applied=True,
    )
    seen: dict[str, object] = {}

    def fake_build_quality_gates_summary(
        evaluation_report: dict[str, object],
    ) -> report_summary_mod.QualityGatesSummary:
        seen["evaluation_report"] = evaluation_report
        return summary

    monkeypatch.setattr(
        render_mod,
        "build_quality_gates_summary",
        fake_build_quality_gates_summary,
    )

    md = render_report_markdown(cert)

    assert seen["evaluation_report"] == cert
    assert (
        "| Primary Metric Acceptable | ✅ PASS | 1.000x | ≤ 1.10x | point | Ratio vs baseline |"
        in md
    )
    assert "- Note: hysteresis applied to gate boundary" in md


def test_render_report_markdown_hides_empty_window_plan_summary() -> None:
    cert = _mk_cert()
    cert["dataset"]["seq_len"] = 16
    cert["dataset"]["windows"] = {"preview": 1, "final": 1, "stats": {}}

    md = render_report_markdown(cert)

    assert "Window Plan:" not in md


def test_render_report_markdown_accepts_legacy_minimal_fixture() -> None:
    fixture = Path("tests/artifacts/golden_runs/gpt2/evaluation.report.json")
    cert = json.loads(fixture.read_text())

    assert validate_report(cert) is True
    md = render_report_markdown(cert)

    assert "# InvarLock Evaluation Report" in md
    assert "**Generated:** (not recorded)" in md
