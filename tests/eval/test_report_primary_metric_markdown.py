from invarlock.reporting.report import to_markdown
from invarlock.reporting.report_types import create_empty_report


def _report_with_pm():
    rep = create_empty_report()
    rep["meta"].update(
        {
            "model_id": "gpt2",
            "adapter": "hf_causal",
            "device": "cpu",
            "commit": "deadbeefcafebabe",
            "ts": "2025-01-01T00:00:00",
        }
    )
    rep["data"].update(
        {
            "dataset": "dummy",
            "split": "validation",
            "seq_len": 16,
            "stride": 8,
            "preview_n": 2,
            "final_n": 2,
        }
    )
    rep["edit"]["name"] = "noop"
    rep["metrics"].update(
        {
            "latency_ms_per_tok": 1.23,
            "memory_mb_peak": 42.0,
            # ppl_* fields may exist but the renderer should ignore them when primary_metric is present
            "ppl_preview": 10.0,
            "ppl_final": 10.0,
            "ppl_ratio": 1.0,
            "primary_metric": {
                "kind": "ppl_causal",
                "preview": 12.3,
                "final": 12.1,
                "ratio_vs_baseline": 1.01,
            },
        }
    )
    return rep


def test_to_markdown_prefers_primary_metric_over_ppl_table():
    rep = _report_with_pm()
    md = to_markdown(rep)
    assert "## Primary Metric" in md
    assert "ppl_causal" in md
    # Should not render the PPL metrics table
    assert "Preview PPL" not in md
    assert "Final PPL" not in md
    assert "PPL Ratio" not in md
