from __future__ import annotations

from invarlock.reporting.normalizer import normalize_run_report


def test_normalize_run_report_accuracy_fallback_and_retention():
    # classification aggregate fallback → accuracy
    raw = {
        "meta": {"model_id": "cls-model", "adapter": "hf", "seed": 1, "device": "cpu"},
        "data": {"dataset": "ds", "split": "val"},
        "metrics": {"classification": {"final": {"correct_total": 8, "total": 10}}},
        "evaluation_windows": {"preview": {}, "final": {}},
        "guard_overhead": {"evaluated": True},
        "provenance": {"provider_digest": {"ids_sha256": "x"}},
    }
    rep = normalize_run_report(raw)
    pm = rep["metrics"]["primary_metric"]
    assert pm["kind"] == "accuracy"
    assert 0.79 < pm["final"] < 0.81
    assert (
        "evaluation_windows" in rep and "guard_overhead" in rep and "provenance" in rep
    )


def test_normalize_run_report_vqa_kind_inference():
    raw = {
        "meta": {
            "model_id": "my-vqa-model",
            "adapter": "hf",
            "seed": 0,
            "device": "cpu",
        },
        "data": {"dataset": "ds"},
        "metrics": {"classification": {"final": 0.9}},
    }
    rep = normalize_run_report(raw)
    assert rep["metrics"]["primary_metric"]["kind"] == "vqa_accuracy"
