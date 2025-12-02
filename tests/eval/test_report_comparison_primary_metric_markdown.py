from copy import deepcopy

from invarlock.reporting.report import to_markdown
from invarlock.reporting.report_types import create_empty_report


def _report_with_pm(final_value: float):
    rep = create_empty_report()
    rep["meta"].update(
        {
            "model_id": "gpt2",
            "adapter": "hf_gpt2",
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
            "primary_metric": {
                "kind": "ppl_causal",
                "preview": 12.3,
                "final": final_value,
                "ratio_vs_baseline": 1.01,
            },
        }
    )
    return rep


def test_comparison_markdown_uses_primary_metric_when_present():
    r1 = _report_with_pm(12.0)
    r2 = _report_with_pm(13.0)
    md = to_markdown(r1, compare=deepcopy(r2), title="Compare")
    # Has a comparison table and mentions Primary Metric rather than PPL Ratio
    assert "## Comparison Summary" in md
    assert "Primary Metric" in md or "primary metric" in md.lower()
    assert "PPL Ratio" not in md
