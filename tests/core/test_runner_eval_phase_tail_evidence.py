from __future__ import annotations

from invarlock.core.api import RunConfig, RunReport
from invarlock.core.runner import CoreRunner
from invarlock.core.runner_eval_phase import eval_phase


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


def test_eval_phase_passes_empty_pairs_when_final_windows_are_not_lists() -> None:
    captured: dict[str, object] = {}

    class _Runner:
        def _log_event(self, *_args, **_kwargs) -> None:
            return None

        def _compute_real_metrics(self, *_args, **_kwargs):
            metrics = {
                "primary_metric": {"kind": "ppl_causal", "preview": 1.0, "final": 1.0},
            }
            eval_windows = {
                "preview": {},
                "final": {
                    "window_ids": ("not", "a", "list"),
                    "logloss": [0.7],
                    "token_counts": [5],
                },
            }
            return metrics, eval_windows

        def _resolve_policy_flags(self, _config) -> dict[str, bool]:
            return {"strict_eval": False}

    def _fake_tail(
        *, deltas: list[float], weights: list[float] | None, policy: dict[str, object] | None
    ) -> dict[str, object]:
        captured["deltas"] = list(deltas)
        captured["weights"] = weights
        captured["policy"] = policy
        return {"mode": "warn", "evaluated": True, "passed": True}

    report = RunReport()
    report.meta["tier_policies"] = {"metrics": {"pm_tail": {"mode": "warn"}}}
    config = RunConfig(
        context={
            "baseline_eval_windows": {
                "final": {"window_ids": [0], "logloss": [0.5]},
            },
            "run": {"strict_eval": False},
        }
    )

    metrics = eval_phase(
        _Runner(),
        model=object(),
        adapter=object(),
        calibration_data=[{"input_ids": [1, 2, 3]}],
        report=report,
        preview_n=1,
        final_n=1,
        config=config,
        evaluate_metric_tail_fn=_fake_tail,
    )

    assert captured == {
        "deltas": [],
        "weights": None,
        "policy": {"mode": "warn"},
    }
    assert metrics["primary_metric_tail"]["source"] == "paired_baseline.final"


def test_eval_phase_strips_primary_metric_kind_before_tail_detection(
    monkeypatch,
) -> None:
    runner = CoreRunner()

    def fake_compute(*_args, **_kwargs):
        metrics = {
            "primary_metric": {"kind": "  PPL_Causal  ", "preview": 1.0, "final": 1.0},
        }
        eval_windows = {
            "preview": {},
            "final": {
                "window_ids": [0],
                "logloss": [2.0],
                "token_counts": [10],
            },
        }
        return metrics, eval_windows

    monkeypatch.setattr(CoreRunner, "_compute_real_metrics", staticmethod(fake_compute))

    report = RunReport()
    report.meta["tier_policies"] = {"metrics": {"pm_tail": {"mode": "warn"}}}

    cfg = RunConfig(
        context={
            "baseline_eval_windows": {"final": {"window_ids": [0], "logloss": [1.0]}},
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


def test_eval_phase_ignores_boolean_window_ids_values_and_weights_in_tail_input(
    monkeypatch,
) -> None:
    runner = CoreRunner()
    captured: dict[str, object] = {}

    def fake_compute(*_args, **_kwargs):
        metrics = {
            "primary_metric": {"kind": "ppl_causal", "preview": 1.0, "final": 1.0},
        }
        eval_windows = {
            "preview": {},
            "final": {
                "window_ids": [True, 7],
                "logloss": [0.1, 0.4],
                "token_counts": [True, 9],
            },
        }
        return metrics, eval_windows

    def fake_tail(*, deltas, weights=None, policy=None):
        captured["deltas"] = deltas
        captured["weights"] = weights
        captured["policy"] = policy
        return {"evaluated": True, "passed": True}

    monkeypatch.setattr(CoreRunner, "_compute_real_metrics", staticmethod(fake_compute))

    report = RunReport()
    report.meta["tier_policies"] = {"metrics": {"pm_tail": {"mode": "warn"}}}
    cfg = RunConfig(
        context={
            "baseline_eval_windows": {
                "final": {"window_ids": [True, 7], "logloss": [0.0, 0.1]}
            },
            "run": {"strict_eval": False},
        }
    )

    eval_phase(
        runner,
        model=object(),
        adapter=object(),
        calibration_data=[{"input_ids": [1, 2, 3]}],
        report=report,
        preview_n=1,
        final_n=1,
        config=cfg,
        evaluate_metric_tail_fn=fake_tail,
    )

    assert captured["deltas"] == [0.30000000000000004]
    assert captured["weights"] == [9.0]


def test_eval_phase_skips_primary_metric_tail_for_non_ppl_metrics(
    monkeypatch,
) -> None:
    runner = CoreRunner()

    def fake_compute(*_args, **_kwargs):
        metrics = {
            "primary_metric": {"kind": "accuracy", "preview": 0.9, "final": 0.8},
        }
        eval_windows = {
            "preview": {},
            "final": {"window_ids": [0], "logloss": [0.2], "token_counts": [10]},
        }
        return metrics, eval_windows

    monkeypatch.setattr(CoreRunner, "_compute_real_metrics", staticmethod(fake_compute))

    report = RunReport()
    cfg = RunConfig(
        context={
            "baseline_eval_windows": {
                "final": {"window_ids": [0], "logloss": [0.1], "token_counts": [10]}
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

    assert "primary_metric_tail" not in metrics
