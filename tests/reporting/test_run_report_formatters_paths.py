from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import pytest

from invarlock.reporting import run_report_formatters as formatters
from invarlock.reporting.report_types import create_empty_report


class _Stringable:
    def __str__(self) -> str:
        return "stringified"


def _report(
    *,
    pm_kind: str = "ppl_causal",
    pm_preview: Any = None,
    pm_final: Any = 1.0,
    pm_ratio: Any = None,
) -> dict[str, Any]:
    report = create_empty_report()
    report["meta"]["model_id"] = "demo-model"
    report["meta"]["adapter"] = "hf"
    report["meta"]["commit"] = "1234567890abcdef"
    report["meta"]["seed"] = 7
    report["meta"]["device"] = "cpu"
    report["meta"]["ts"] = "2026-03-30T00:00:00Z"
    report["data"]["dataset"] = "unit"
    report["data"]["split"] = "validation"
    report["data"]["seq_len"] = 8
    report["data"]["stride"] = 8
    report["data"]["preview_n"] = 4
    report["data"]["final_n"] = 8
    report["edit"]["name"] = "quant_rtn"
    report["edit"]["plan_digest"] = "abcdef1234567890feedface"
    report["edit"]["deltas"]["params_changed"] = 12
    report["edit"]["deltas"]["layers_modified"] = 3
    report["metrics"]["latency_ms_per_tok"] = 1.25
    report["metrics"]["memory_mb_peak"] = 32.0
    report["metrics"]["primary_metric"] = {"kind": pm_kind, "final": pm_final}
    if pm_preview is not None:
        report["metrics"]["primary_metric"]["preview"] = pm_preview
    if pm_ratio is not None:
        report["metrics"]["primary_metric"]["ratio_vs_baseline"] = pm_ratio
    return report


def test_to_json_and_sanitize_cover_invalid_and_non_json_values() -> None:
    with pytest.raises(ValueError, match="Invalid RunReport structure"):
        formatters.to_json({})

    report = _report(pm_preview=0.9, pm_final=1.0, pm_ratio=1.02)
    report["provenance"] = {
        "captured_at": datetime(2026, 3, 30, tzinfo=UTC),
        "owner": _Stringable(),
        "nested": [_Stringable()],
    }

    payload = json.loads(formatters.to_json(report))
    sanitized = formatters._sanitize_for_json(
        {
            "captured_at": datetime(2026, 3, 30, tzinfo=UTC),
            "fallback": _Stringable(),
        }
    )

    assert payload["provenance"]["captured_at"] == "2026-03-30T00:00:00+00:00"
    assert payload["provenance"]["owner"] == "stringified"
    assert payload["provenance"]["nested"] == ["stringified"]
    assert sanitized == {
        "captured_at": "2026-03-30T00:00:00+00:00",
        "fallback": "stringified",
    }


def test_coerce_run_reports_and_markdown_title_cover_validation_paths() -> None:
    report = _report(pm_preview=0.9, pm_final=1.0, pm_ratio=1.02)

    rendered = formatters.to_markdown(dict(report), title="Custom Title")

    assert rendered.startswith("# Custom Title")
    with pytest.raises(ValueError, match="Invalid primary RunReport structure"):
        formatters._coerce_run_reports({})
    with pytest.raises(ValueError, match="Invalid comparison RunReport structure"):
        formatters._coerce_run_reports(report, {})


@pytest.mark.parametrize(
    ("ratio", "expected"),
    [
        (1.04, "Minimal"),
        (1.07, "Moderate"),
        (1.12, "Significant"),
    ],
)
def test_generate_single_markdown_summary_assessment_variants(
    ratio: float, expected: str
) -> None:
    report = _report(pm_preview=0.95, pm_final=1.05, pm_ratio=ratio)

    markdown = "\n".join(formatters._generate_single_markdown(report))

    assert f"PM ratio {ratio:.3f}" in markdown
    assert (
        f"{expected} model changes with {expected.lower()} performance impact"
        in markdown
    )


def test_generate_single_markdown_guard_and_status_variants() -> None:
    report = _report(pm_preview=0.91, pm_final=1.05, pm_ratio=1.12)
    report["edit"]["deltas"]["sparsity"] = 0.25
    report["guards"] = [
        {
            "name": "variance",
            "passed": False,
            "decision": "rollback",
            "policy": {},
            "metrics": {"score": 0.5},
            "diagnostics": [
                {"severity": "warning", "message": "watch this"},
                "plain note",
            ],
            "violations": [{"message": "dict violation"}, "raw violation"],
        }
    ]
    report["flags"]["rollback_reason"] = "policy block"

    markdown = "\n".join(formatters._generate_single_markdown(report))

    assert "preview=0.910" in markdown
    assert "ratio_vs_baseline=1.120" in markdown
    assert "| Overall Sparsity | 0.250 |" in markdown
    assert "**Decision:** rollback" in markdown
    assert "**Metrics:**" in markdown
    assert "- score: 0.5" in markdown
    assert "**Diagnostics:**" in markdown
    assert "- [WARNING] watch this" in markdown
    assert "- plain note" in markdown
    assert "- ⚠️ dict violation" in markdown
    assert "- ⚠️ raw violation" in markdown
    assert "🔄 **ROLLBACK**: policy block" in markdown
    assert "Pipeline did not complete successfully" in markdown


def test_generate_single_markdown_handles_missing_pm_and_guard_violations() -> None:
    report = _report()
    report["metrics"]["primary_metric"] = "not-a-dict"
    report["guards"] = [
        {
            "name": "variance",
            "passed": False,
            "decision": "",
            "policy": {},
            "metrics": {},
            "diagnostics": [],
            "violations": ["trip"],
        }
    ]

    markdown = "\n".join(formatters._generate_single_markdown(report))

    assert "- **Primary Metric**: unavailable" in markdown
    assert "Some guards reported violations" in markdown
    assert "Review guard reports above for details" in markdown
    assert "Performance Impact" not in markdown


def test_generate_single_markdown_covers_guard_recovery_and_partial_pm_fields() -> None:
    preview_missing = _report(pm_final=1.01)
    preview_missing["metrics"]["primary_metric"].pop("preview", None)
    preview_missing["metrics"]["primary_metric"]["ratio_vs_baseline"] = "bad"
    preview_missing["flags"]["guard_recovered"] = True

    final_missing = _report(pm_preview=0.9, pm_final=None)
    final_missing["metrics"]["primary_metric"].pop("ratio_vs_baseline", None)

    missing_preview_markdown = "\n".join(
        formatters._generate_single_markdown(preview_missing)
    )
    missing_final_markdown = "\n".join(
        formatters._generate_single_markdown(final_missing)
    )

    assert "Guard recovery was triggered" in missing_preview_markdown
    assert "Some guards detected issues but were resolved" in missing_preview_markdown
    assert "preview=" not in missing_preview_markdown
    assert "Performance Impact" not in missing_preview_markdown
    assert "preview=0.900" in missing_final_markdown
    assert "final=" not in missing_final_markdown
    assert "ratio_vs_baseline=" not in missing_final_markdown


def test_generate_comparison_markdown_covers_primary_metric_and_guard_paths() -> None:
    report1 = _report(pm_kind="accuracy", pm_final=0.81)
    report2 = _report(pm_kind="ppl_causal", pm_final=0.83)
    report1["edit"]["deltas"]["params_changed"] = object()
    report1["edit"]["deltas"]["layers_modified"] = object()
    report2["edit"]["deltas"]["layers_modified"] = object()
    report1["guards"] = [
        {
            "name": "variance",
            "passed": True,
            "decision": "allow",
            "policy": {},
            "metrics": {},
            "diagnostics": [],
            "violations": [],
        }
    ]
    report2["guards"] = [
        {
            "name": "variance",
            "passed": True,
            "decision": "allow",
            "policy": {},
            "metrics": {},
            "diagnostics": [],
            "violations": [],
        }
    ]

    comparison = "\n".join(formatters._generate_comparison_markdown(report1, report2))

    assert "| Primary Metric | 0.810 | 0.830 | 📈 +0.020 |" in comparison
    assert "| Params Changed | 0 | 12 | +12 |" in comparison
    assert "| Layers Modified | 0 | 0 | +0 |" in comparison
    assert "#### variance" in comparison
    assert comparison.count("**Violations:**") == 0

    report1["metrics"]["primary_metric"] = "missing"
    comparison_without_pm = "\n".join(
        formatters._generate_comparison_markdown(report1, report2)
    )
    assert "| Primary Metric |" not in comparison_without_pm


def test_generate_comparison_markdown_handles_non_numeric_metrics_and_parse_failures() -> (
    None
):
    report1 = _report(pm_final="not-numeric")
    report2 = _report(pm_final=1.0)
    report1["edit"]["deltas"]["params_changed"] = "bad-int"
    report2["edit"]["deltas"]["layers_modified"] = "bad-int"

    comparison = "\n".join(formatters._generate_comparison_markdown(report1, report2))

    assert "| Primary Metric (" not in comparison
    assert "| Params Changed | 0 | 12 | +12 |" in comparison
    assert "| Layers Modified | 3 | 0 | -3 |" in comparison
    assert "Guard Reports" not in comparison


def test_to_markdown_and_to_html_cover_top_level_single_and_comparison_paths() -> None:
    report1 = _report(pm_preview=0.9, pm_final=1.0, pm_ratio=1.03)
    report2 = _report(pm_kind="accuracy", pm_final=0.95, pm_ratio=1.01)
    report1["guards"] = [
        {
            "name": "variance",
            "passed": False,
            "decision": "monitor",
            "policy": {},
            "metrics": {},
            "diagnostics": [],
            "violations": ["cap"],
        }
    ]
    report2["guards"] = [
        {
            "name": "variance",
            "passed": False,
            "decision": "monitor",
            "policy": {},
            "metrics": {},
            "diagnostics": [],
            "violations": ["cap-2"],
        }
    ]

    markdown = formatters.to_markdown(report1, compare=report2)
    single_html = formatters.to_html(report1, title="HTML Title", include_css=True)
    comparison_html = formatters.to_html(report1, compare=report2, include_css=False)

    assert markdown.startswith("# InvarLock Evaluation Report Comparison")
    assert "| Primary Metric | 1.000 | 0.950 | 📉 -0.050 |" in markdown
    assert "**Violations:**" in markdown
    assert "<style>" in single_html
    assert "<h1>HTML Title</h1>" in single_html
    assert "<style>" not in comparison_html
    assert "InvarLock Report Comparison" in comparison_html


def test_html_generation_covers_heading_and_table_closure_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    single_markdown = [
        "# Top Level",
        "| Metric | Value |",
        "|--------|-------|",
        "| Latency | 1 |",
        "- item after table",
        "| Metric | Value |",
        "|--------|-------|",
        "| Memory | 2 |",
        "Paragraph after table",
        "| Metric | Value |",
        "|--------|-------|",
        "| Tail | 3 |",
    ]
    comparison_markdown = [
        "## Comparison Summary",
        "| Metric | Report 1 | Report 2 | Delta |",
        "|--------|----------|----------|-------|",
        "| Latency | 1 | 2 | +1 |",
        "Paragraph after table",
        "| Aspect | Report 1 | Report 2 |",
        "|--------|----------|----------|",
        "| Model | a | b |",
    ]

    monkeypatch.setattr(
        formatters, "_generate_single_markdown", lambda _report: single_markdown
    )
    monkeypatch.setattr(
        formatters,
        "_generate_comparison_markdown",
        lambda _report1, _report2: comparison_markdown,
    )

    single_html = "\n".join(formatters._generate_single_html(_report()))
    comparison_html = "\n".join(
        formatters._generate_comparison_html(_report(), _report())
    )

    assert "<h1>Top Level</h1>" in single_html
    assert "<li>item after table</li>" in single_html
    assert "<p>Paragraph after table</p>" in single_html
    assert single_html.count("</tbody></table>") >= 3
    assert "<h2>Comparison Summary</h2>" in comparison_html
    assert "<p>Paragraph after table</p>" in comparison_html
    assert comparison_html.count("</tbody></table>") >= 2


def test_html_helpers_cover_blank_line_and_h3_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    single_markdown = [
        "## Section",
        "### Details",
        "| Metric | Value |",
        "|--------|-------|",
        "| Latency | 1 |",
        "",
        "- trailing list item",
    ]
    comparison_markdown = [
        "### Guard Reports",
        "| Metric | Report 1 | Report 2 | Delta |",
        "|--------|----------|----------|-------|",
        "| Latency | 1 | 2 | +1 |",
        "",
        "plain comparison paragraph",
    ]

    monkeypatch.setattr(
        formatters, "_generate_single_markdown", lambda _report: single_markdown
    )
    monkeypatch.setattr(
        formatters,
        "_generate_comparison_markdown",
        lambda _report1, _report2: comparison_markdown,
    )

    single_html = "\n".join(formatters._generate_single_html(_report()))
    comparison_html = "\n".join(
        formatters._generate_comparison_html(_report(), _report())
    )

    assert "<h2>Section</h2>" in single_html
    assert "<h3>Details</h3>" in single_html
    assert "<li>trailing list item</li>" in single_html
    assert "<h3>Guard Reports</h3>" in comparison_html
    assert "<p>plain comparison paragraph</p>" in comparison_html
