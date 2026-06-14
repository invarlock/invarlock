from __future__ import annotations

from invarlock import __version__
from invarlock.reporting.report_make import make_report


def _mk_report() -> dict:
    return {
        "meta": {
            "model_id": "gpt2",
            "adapter": "hf_causal",
            "device": "cpu",
            "seed": 42,
            "ts": "now",
            "auto": {"tier": "balanced"},
        },
        "data": {
            "dataset": "dummy",
            "split": "validation",
            "seq_len": 8,
            "stride": 4,
            "preview_n": 1,
            "final_n": 1,
        },
        "edit": {
            "name": "noop",
            "plan_digest": "noop",
            "deltas": {
                "params_changed": 0,
                "heads_pruned": 0,
                "neurons_pruned": 0,
                "layers_modified": 0,
                "sparsity": None,
                "bitwidth_map": None,
            },
        },
        "guards": [],
        "metrics": {
            "primary_metric": {
                "kind": "ppl_causal",
                "preview": 50.0,
                "final": 49.0,
                "ratio_vs_baseline": 0.98,
                "display_ci": (0.97, 0.99),
            }
        },
        "artifacts": {"events_path": "", "logs_path": "", "checkpoint_path": None},
        "flags": {"guard_recovered": False, "rollback_reason": None},
    }


def test_html_exporter_renders_report_outline_sections():
    from invarlock.reporting.html import render_report_html

    cert = make_report(_mk_report(), _mk_report())
    html = render_report_html(cert)

    assert '<section id="decision"' in html
    assert '<section id="primary_metric"' in html
    assert '<section id="policy_gates"' in html
    assert '<section id="guard_signals"' in html
    assert '<section id="evidence_provenance"' in html
    assert '<section id="technical_appendix"' in html
    assert "Benchmark Comparison" not in html
    assert "<table" in html
    assert "report-outline" in html
    assert "summary-strip" in html
    assert "summary-table" in html
    assert "<th>Baseline</th>" in html
    assert "baseline_ref" in html
    assert "data-theme-toggle" in html
    assert "invarlock-report-theme" in html
    assert "aria-current" in html
    assert "stickyOffset()" in html
    assert "--sticky-offset" in html
    assert "box-shadow" not in html
    assert "summary-chip" not in html
    assert "Linked Run Reports" not in html
    assert "Workflow" not in html
    assert "brand-lockup" in html
    assert "brand-mark-svg" in html
    assert "--bg:#fcfbf7" in html
    assert "--accent:#1f3a7a" in html
    assert "--signal:#8d2433" in html
    assert "--bg:#11130f" in html
    assert "--accent:#9fb7ff" in html
    assert "#236b67" not in html
    assert "#7fd3c9" not in html
    assert ">IL<" not in html
    assert "✅" not in html
    assert "❌" not in html
    assert "Auditable verification for edited model checkpoints." in html
    assert f"InvarLock {__version__}" in html


def test_html_summary_uses_computed_validation_status():
    from invarlock.reporting.html import render_report_html

    cert = make_report(_mk_report(), _mk_report())
    cert.get("validation", {}).pop("overall_pass", None)

    html = render_report_html(cert)

    assert '<td><strong class="tone-pass">PASS</strong></td>' in html
    assert '<td><strong class="tone-fail">FAIL</strong></td>' not in html


def test_html_exporter_renders_benchmark_comparison_section():
    from invarlock.reporting.html import render_report_html

    cert = make_report(_mk_report(), _mk_report())
    cert["benchmark_comparison"] = {
        "profile": "ci",
        "scenarios": [
            {
                "edit": "quant_rtn",
                "skip": False,
                "primary_metric_overhead": 0.009,
                "guard_overhead_time": 0.13,
                "guard_overhead_mem": 0.09,
                "rmt_outliers_bare": 2,
                "rmt_outliers_guarded": 3,
                "pass": {"quality": True, "time": True, "mem": True},
            }
        ],
    }

    html = render_report_html(cert)

    assert '<section id="benchmark_comparison"' in html
    assert "Benchmark Comparison" in html
    assert "1 total, 1 passed, 0 skipped" in html
    assert "0.9%" in html


def test_html_exporter_renders_accuracy_without_perplexity_language():
    from invarlock.reporting.html import render_report_html

    cert = make_report(_mk_report(), _mk_report())
    cert["meta"]["adapter"] = "hf_multimodal"
    cert["primary_metric"] = {
        "kind": "accuracy",
        "unit": "accuracy",
        "preview": 0.86,
        "final": 0.855,
        "ratio_vs_baseline": 0.0,
        "display_ci": [-0.01, 0.01],
    }

    html = render_report_html(cert)

    assert "+0.00 pp" in html
    assert "-0.01 to +0.01 pp" in html
    assert "Perplexity" not in html


def test_html_exporter_escapes_report_controlled_html_payloads():
    from invarlock.reporting.html import render_report_html

    cert = make_report(_mk_report(), _mk_report())
    cert["plugins"] = {
        "adapter": {
            "name": '<script>alert("adapter")</script>',
            "module": "safe.module",
            "version": "1.0",
        },
        "guards": [
            {
                "name": '<img src=x onerror="alert(1)">',
                "module": "safe.guard",
            }
        ],
    }

    html = render_report_html(cert)
    lowered = html.lower()

    assert lowered.count("<script") == 2
    assert '<script>alert("adapter")</script>' not in lowered
    assert "<img" not in lowered
    assert "&lt;script&gt;" in lowered
    assert "&lt;img" in lowered
