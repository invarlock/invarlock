from __future__ import annotations

from pathlib import Path

import pytest

from invarlock.reporting import html, report_contract
from invarlock.reporting.rendering import guard_sections, markdown, tables
from invarlock.reporting.report_outline import (
    EvaluationReportOutline,
    ReportFact,
    ReportSection,
)
from invarlock.reporting.report_types import create_empty_report, validate_report


def _section(
    key: str,
    *,
    facts: tuple[ReportFact, ...] = (),
    sources: tuple[str, ...] = (),
    priority: str = "summary",
) -> ReportSection:
    return ReportSection(
        key=key,
        title=key.title(),
        summary="summary",
        facts=facts,
        source_blocks=sources,
        priority=priority,
    )


def _outline(*sections: ReportSection) -> EvaluationReportOutline:
    return EvaluationReportOutline(
        title="Report",
        report_kind="evaluation",
        overall_status="pass",
        sections=sections,
    )


def test_html_summary_requires_decision_and_primary_metric() -> None:
    assert html._render_summary_strip(_outline()) == ""
    decision = _section("decision")
    assert html._render_summary_strip(_outline(decision)) == ""


def test_html_summary_and_fact_tables_escape_untrusted_values() -> None:
    decision = _section(
        "decision",
        facts=(
            ReportFact("Model", "<model>", "info", "meta.model", "detail"),
            ReportFact("Baseline", "base", "info", "baseline"),
            ReportFact("Guard Warnings", "2", "warn", "guard_warnings"),
        ),
        sources=("meta",),
    )
    primary = _section(
        "primary_metric",
        facts=(ReportFact("Metric", "ppl", "pass", "primary_metric"),),
    )
    outline = _outline(decision, primary)

    rendered = html._render_summary_strip(outline)
    assert "&lt;model&gt;" in rendered
    assert "tone-warn" in rendered
    assert 'aria-current="true"' in html._render_nav(outline)
    assert "<code>meta</code>" in html._render_source_chips(decision)
    assert "<th>Detail</th>" in html._render_fact_table(decision)
    assert html._fact_value(decision, "missing") == "N/A"
    assert html._tone("unknown") == "tone-info"


def test_html_guard_warnings_ignore_noise_and_render_structured_values() -> None:
    assert html._render_guard_warnings({}) == ""
    assert html._render_guard_warnings({"guard_warnings": {"warnings": []}}) == ""
    report = {
        "guard_warnings": {
            "warnings": [
                "noise",
                {
                    "guard": "spectral<script>",
                    "kind": "movement",
                    "family": "attention",
                    "baseline": {"z": 1},
                    "subject": [2],
                    "policy_gate": "warning",
                    "message": "review <this>",
                },
            ]
        }
    }

    rendered = html._render_guard_warnings(report)

    assert "spectral&lt;script&gt;" in rendered
    assert "review &lt;this&gt;" in rendered
    assert "{&quot;z&quot;: 1}" in rendered


def test_html_appendix_preview_is_bounded_and_skips_empty_blocks() -> None:
    ordinary = html._preview_json({"small": True})
    assert ordinary.startswith("{")
    long = html._preview_json({"payload": "x" * 2000})
    assert "truncated in HTML preview" in long

    non_appendix = _section("decision", sources=("metrics",))
    assert html._render_appendix_previews({"metrics": {"x": 1}}, non_appendix) == ""
    appendix = _section(
        "technical_appendix", sources=("empty", "metrics"), priority="audit"
    )
    rendered = html._render_appendix_previews(
        {"empty": {}, "metrics": {"x": 1}}, appendix
    )
    assert "<summary>metrics</summary>" in rendered


def test_html_section_extras_are_key_specific() -> None:
    warning_section = _section("guard_signals")
    rendered = html._render_section(
        {"guard_warnings": {"warnings": [{"message": "warning"}]}},
        warning_section,
    )
    assert "Guard Warnings" in rendered
    appendix = _section("technical_appendix", sources=("metrics",))
    assert "Appendix previews" in html._render_section({"metrics": {"x": 1}}, appendix)


def test_report_payload_loader_rejects_non_object(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        report_contract, "load_report_input_json", lambda _path: ({}, [])
    )
    with pytest.raises(ValueError, match="Invalid RunReport structure"):
        report_contract.load_report_payload(Path("report.json"))


def test_report_contract_rejects_accuracy_ratio_fields() -> None:
    report = {
        "metrics": {"primary_metric": {"kind": "accuracy", "ratio_vs_baseline": 1}}
    }
    message = report_contract._describe_run_report_health_error(report, role="subject")
    assert "use delta_vs_baseline_pp" in str(message)

    with pytest.raises(ValueError, match="PPL-only"):
        report_contract._assert_evaluation_report_is_finite(
            {"primary_metric": {"kind": "accuracy", "ratio_vs_baseline": 1}}
        )


def test_report_type_validation_rejects_metric_domain_violations() -> None:
    report = create_empty_report()
    report["metrics"]["primary_metric"] = {
        "kind": "accuracy",
        "preview": 0.5,
        "final": 0.5,
        "ratio_vs_baseline": 1.0,
    }
    assert validate_report(report) is False
    report["metrics"]["primary_metric"] = {
        "kind": "accuracy",
        "preview": 2.0,
        "final": 0.5,
    }
    assert validate_report(report) is False


def test_markdown_tables_handle_empty_and_malformed_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lines: list[str] = []
    tables.append_system_overhead_section(lines, {})
    tables.append_accuracy_subgroups(lines, {})
    assert lines == []

    tables.append_system_overhead_section(
        lines,
        {
            "latency_ms_p50": {"baseline": "bad", "edited": 1.0},
            "latency_ms_p95": None,
            "throughput_sps": {
                "baseline": 2.0,
                "edited": 3.0,
                "delta": "bad",
                "ratio": None,
            },
        },
    )
    assert "| Latency p50 (ms) | - | 1 | - | - |" in lines
    assert "| Throughput (samples/s) | 2.0 | 3.0 | - | - |" in lines

    tables.append_accuracy_subgroups(
        lines,
        {
            "group|one": {
                "n_preview": object(),
                "n_final": object(),
                "preview": 0.5,
                "final": 0.6,
                "delta_pp": object(),
            }
        },
    )
    assert any("| group|one | 0 | 0 |" in line and "N/A" in line for line in lines)

    monkeypatch.setattr(
        tables, "build_evaluation_report_outline", lambda _report: _outline()
    )
    untouched = list(lines)
    tables.append_outline_fact_summary_section(lines, {})
    assert lines == untouched
    assert tables._markdown_table_cell("a|b\nc") == "a\\|b c"


class _ExplodingGet(dict):
    def get(self, *_args, **_kwargs):
        raise TypeError("cannot read")


def test_guard_pairing_details_render_malformed_observations_fail_safely() -> None:
    lines: list[str] = []
    guard_sections._append_pairing_details(
        lines,
        {
            "dataset": {
                "windows": {
                    "stats": {
                        "paired_windows": object(),
                        "window_match_fraction": "bad",
                        "window_overlap_fraction": float("nan"),
                        "bootstrap": {"replicates": object(), "seed": "seed"},
                    }
                }
            }
        },
    )
    rendered = "\n".join(lines)
    assert "windows=" in rendered
    assert "match=bad" in rendered
    assert "overlap=nan" in rendered
    assert "replicates" in rendered and "seed=seed" in rendered

    untouched: list[str] = []
    guard_sections._append_pairing_details(untouched, _ExplodingGet())
    assert untouched == []


def test_guard_family_details_skip_malformed_rows_but_keep_valid_evidence() -> None:
    lines: list[str] = []
    guard_sections._append_spectral_family_details(
        lines,
        {
            "caps_applied_by_family": {"mlp": 1},
            "family_z_quantiles": {"mlp": {"q95": 1.2, "max": 2.0}},
            "family_caps": {"mlp": {"kappa": 1.5}},
        },
        {
            "bad": "not-a-list",
            "empty": [],
            "mlp": ["bad-entry", {"module": "up", "z": float("nan")}],
        },
    )
    rendered = "\n".join(lines)
    assert "| mlp | 1.500 | 1.200" in rendered
    assert "up (|z|=n/a)" in rendered
    assert "bad-entry" not in rendered


def test_guard_observability_handles_mixed_top_scores_and_contracts() -> None:
    lines: list[str] = []
    guard_sections.append_guard_observability_sections(
        lines,
        {
            "validation": {"spectral_stable": False},
            "spectral": {
                "caps_applied": 2,
                "max_caps": 1,
                "summary": {"caps_exceeded": True},
                "top_z_scores": {
                    "bad": "not-a-list",
                    "mlp": ["bad", {"z": 2.5}, {"module": "up", "z": -3.0}],
                },
                "family_caps": {"mlp": object()},
                "multiple_testing": {"method": "holm", "alpha": 0.05, "m": 3},
            },
            "rmt": {
                "stable": None,
                "mode": "weights",
                "delta_total": 1,
                "measurement_contract": {
                    "estimator": {"kind": "svd"},
                    "activation_sampling": {"windows": 2},
                },
            },
        },
    )
    rendered = "\n".join(lines)
    assert "Max |z|" in rendered
    assert "No κ" in rendered
    assert "method=holm" in rendered
    assert "Measurement Contract" in rendered
    assert "NOT EVALUATED" in rendered


def test_guard_warnings_skip_non_mapping_entries() -> None:
    lines: list[str] = []
    guard_sections.append_guard_warnings_section(
        lines,
        {"guard_warnings": {"warnings": ["noise", {"guard": "rmt"}]}},
    )
    rendered = "\n".join(lines)
    assert "| rmt | warning |" in rendered
    assert "noise" not in rendered


def test_guard_details_cover_malformed_and_nonfatal_invariant_branches() -> None:
    note_lines: list[str] = []
    guard_sections._append_invariant_notes(
        note_lines,
        ["noise", {"check": "finite", "detail": {"count": 1}}],
    )
    assert "finite" in "\n".join(note_lines)
    assert "noise" not in "\n".join(note_lines)

    malformed: list[str] = []
    guard_sections.append_guard_check_details_section(
        malformed,
        {"invariants": {"summary": "bad"}, "validation": {}},
    )
    assert "## Guard Check Details" in malformed

    warning: list[str] = []
    guard_sections.append_guard_check_details_section(
        warning,
        {
            "invariants": {
                "summary": {"warning_violations": 1},
                "failures": [{}],
            },
            "validation": {"invariants_pass": True},
        },
    )
    assert "- Non-fatal: Non-fatal invariant warnings present." in warning


def test_markdown_helpers_fail_closed_on_malformed_metric_context() -> None:
    assert markdown._primary_metric_is_pseudo_accuracy({"primary_metric": []}) is False
    assert markdown._format_secondary_metric_ratio({}, "ppl_causal") == "N/A"
    assert (
        markdown._format_secondary_metric_ratio({"ratio_vs_baseline": 1.2}, "ppl")
        == "1.200"
    )
    assert markdown._dataset_hash_source_label("unknown") is None

    lines: list[str] = []
    markdown._append_primary_metric_section(
        lines,
        {
            "primary_metric": {
                "kind": "accuracy",
                "preview": 0.5,
                "final": 0.6,
                "paired": "unknown",
                "delta_vs_baseline_pp": None,
            },
            "secondary_metrics": ["noise", {"kind": "ppl", "ci": [0.9, 1.1]}],
        },
    )
    rendered = "\n".join(lines)
    assert "Δ vs Baseline | N/A" in rendered
    assert "| ppl |" in rendered


def test_markdown_dataset_policy_and_window_helpers_cover_optional_fields() -> None:
    lines: list[str] = []
    markdown._append_dataset_details(
        lines,
        {
            "provider": "fixture",
            "seq_len": "variable",
            "windows": {"preview": 2, "final": 3, "seed": 7},
            "hash": {
                "preview_tokens": 4,
                "final_tokens": 5,
                "total_tokens": 9,
                "dataset": "sha256:data",
                "source": "explicit_token_ids",
            },
            "tokenizer": {
                "name": "tok",
                "hash": "sha256:tok",
                "pad_token": "<pad>",
                "add_prefix_space": False,
            },
        },
    )
    assert any("PAD" in line for line in lines)
    assert any("add_prefix_space" in line for line in lines)

    policy_lines: list[str] = []
    markdown._append_policy_configuration_section(
        policy_lines,
        {
            "auto": {"tier": "balanced"},
            "policy_digest": {"thresholds_hash": "abcdef"},
            "resolved_policy": {"spectral": {"max_caps": 2}},
        },
    )
    assert "**Tier:** balanced" in "\n".join(policy_lines)
    assert markdown._get_window_plan_summary({"window_plan": "bad"}) is None
    assert (
        markdown._get_window_plan_summary(
            {
                "dataset": {
                    "windows": {"profile": "ci", "actual_preview": 2, "actual_final": 4}
                }
            }
        )
        == "Window Plan: ci, 2/4"
    )
