from invarlock.reporting.report import to_markdown
from invarlock.reporting.report_types import create_empty_report


def _base_report():
    rep = create_empty_report()
    rep["meta"].update(
        {
            "model_id": "gpt2",
            "adapter": "hf_gpt2",
            "device": "cpu",
            "commit": "deadbeef",
        }
    )
    rep["data"].update(
        {
            "dataset": "dummy",
            "split": "validation",
            "seq_len": 8,
            "stride": 4,
            "preview_n": 1,
            "final_n": 1,
        }
    )
    rep["edit"]["name"] = "structured"
    rep["edit"]["plan_digest"] = "abcd"
    rep["edit"]["deltas"].update(
        {
            "params_changed": 0,
            "heads_pruned": 0,
            "neurons_pruned": 0,
            "layers_modified": 0,
            "sparsity": None,
        }
    )
    rep["metrics"].update(
        {
            "ppl_preview": 10.0,
            "ppl_final": 10.0,
            "ppl_ratio": 1.0,
            "latency_ms_per_tok": 1.0,
            "memory_mb_peak": 100.0,
            "primary_metric": {
                "kind": "ppl_causal",
                "preview": 10.0,
                "final": 10.0,
                "ratio_vs_baseline": 1.0,
                "display_ci": [10.0, 10.0],
            },
        }
    )
    rep["guards"] = []
    return rep


def test_report_flags_guard_recovery():
    rep = _base_report()
    rep["flags"].update({"guard_recovered": True, "rollback_reason": None})
    md = to_markdown(rep)
    assert "Guard recovery" in md


def test_report_flags_rollback():
    rep = _base_report()
    rep["flags"].update({"guard_recovered": False, "rollback_reason": "capacity"})
    md = to_markdown(rep)
    assert "ROLLBACK" in md
