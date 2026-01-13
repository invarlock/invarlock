from __future__ import annotations

from invarlock.core.api import RunConfig, RunReport
from invarlock.core.runner import CoreRunner


def test_eval_phase_computes_primary_metric_tail_from_paired_baseline(
    monkeypatch,
) -> None:
    runner = CoreRunner()

    def fake_compute(*_args, **_kwargs):
        metrics = {
            "primary_metric": {"kind": "ppl_causal", "preview": 1.0, "final": 1.0},
        }
        eval_windows = {
            "preview": {},
            "final": {
                "window_ids": [0, 0, 99, "bad_id", 0],
                "logloss": [2.0, float("inf"), 3.0, 4.0, 2.5],
                "token_counts": [10, "oops"],
            },
        }
        return metrics, eval_windows

    monkeypatch.setattr(CoreRunner, "_compute_real_metrics", staticmethod(fake_compute))

    report = RunReport()
    report.meta["tier_policies"] = {"metrics": {"pm_tail": {"mode": "warn"}}}

    cfg = RunConfig(
        context={
            "baseline_eval_windows": {
                "final": {"window_ids": [0, "bad"], "logloss": [1.0, 2.0]}
            },
            "run": {"strict_eval": False},
        }
    )

    metrics = runner._eval_phase(
        model=object(),
        adapter=object(),
        calibration_data=[{"input_ids": [1, 2, 3]}],
        report=report,
        preview_n=1,
        final_n=1,
        config=cfg,
    )

    tail = metrics.get("primary_metric_tail")
    assert isinstance(tail, dict)
    assert tail.get("source") == "paired_baseline.final"
