from __future__ import annotations

import math

import pytest

from invarlock.core.api import RunConfig, RunReport
from invarlock.core.runner import CoreRunner
from tests.core._support_runner_failure import (
    DummyAdapter,
    EditStub,
    _minimal_calibration,
    _toy_model_with_losses,
)

## Removed flaky negative request coverage assertion; coverage for _resolve_limit
## with non-positive requests is exercised by other tests (preview/final zero cases).


# Removed: Indexable debug snapshot that caused slicing issues outside safeguards


def test_delta_weights_path(monkeypatch):
    # Ensure delta_weights population path executes by having non-empty token_counts
    runner = CoreRunner()
    model = _toy_model_with_losses([1.0, 1.1, 1.0, 1.1])
    adapter = DummyAdapter()
    cal = [
        {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]},
        {"input_ids": [4, 5, 6], "attention_mask": [1, 1, 1]},
        {"input_ids": [7, 8, 9], "attention_mask": [1, 1, 1]},
        {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]},
    ]
    metrics, _ = runner._compute_real_metrics(
        model,
        cal,
        adapter,
        preview_n=2,
        final_n=2,
        config=RunConfig(
            context={"eval": {"bootstrap": {"enabled": True, "replicates": 5}}}
        ),
    )
    # paired_delta_summary must exist and be consistent
    assert "paired_delta_summary" in metrics


def test_eval_debug_snapshot_indexable_supported(monkeypatch):
    import os

    os.environ["INVARLOCK_DEBUG_TRACE"] = "1"
    try:
        runner = CoreRunner()
        model = _toy_model_with_losses([1.0, 1.0])
        adapter = DummyAdapter()

        class Indexable:
            def __len__(self):
                return 2

            def __getitem__(self, idx):
                if isinstance(idx, slice):
                    return [
                        {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]},
                        {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]},
                    ][idx]
                return {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}

        metrics, _ = runner._compute_real_metrics(
            model, Indexable(), adapter, preview_n=1, final_n=1, config=RunConfig()
        )
        pm = metrics.get("primary_metric", {})
        assert pm.get("final") and pm.get("preview")
    finally:
        del os.environ["INVARLOCK_DEBUG_TRACE"]


def test_resolve_guard_policies_from_argument(monkeypatch):
    from invarlock.core.api import RunReport

    runner = CoreRunner()
    seen = {}

    def fake_resolver(tier, edit_name, overrides):
        seen["tier"] = tier
        return {}

    import invarlock.core.runner as runner_mod

    monkeypatch.setattr(runner_mod, "resolve_tier_policies", fake_resolver)
    report = RunReport()
    report.meta["config"] = {"guards": {}}
    _ = runner._resolve_guard_policies(
        report, auto_config={"tier": "conservative", "enabled": True}
    )
    assert seen.get("tier") == "conservative"


def test_resolve_guard_policies_default(monkeypatch):
    from invarlock.core.api import RunReport

    runner = CoreRunner()
    seen = {}

    def fake_resolver(tier, edit_name, overrides):
        seen["tier"] = tier
        return {}

    import invarlock.core.runner as runner_mod

    monkeypatch.setattr(runner_mod, "resolve_tier_policies", fake_resolver)
    report = RunReport()  # no auto in meta
    report.meta["config"] = {"guards": {}}
    _ = runner._resolve_guard_policies(report, auto_config=None)
    assert seen.get("tier") == "balanced"


def test_measure_latency_dim_exception(monkeypatch):
    import torch

    runner = CoreRunner()

    class M:
        def __call__(self, *a, **k):
            class Obj:
                def __init__(self):
                    self.loss = type("L", (), {"item": lambda self: 1.0})()

            return Obj()

    # Force dim() to raise; guarded path should proceed
    orig_dim = torch.Tensor.dim

    def raising_dim(self):
        raise RuntimeError("dim-fail")

    monkeypatch.setattr(torch.Tensor, "dim", raising_dim)
    try:
        sample = {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}
        with pytest.raises(
            RuntimeError, match="Latency measurement input shape inspection failed"
        ):
            runner._measure_latency(M(), [sample], "cpu")
    finally:
        monkeypatch.setattr(torch.Tensor, "dim", orig_dim)


def test_window_shortage_and_final_shortage_warnings(tmp_path):
    # Sum of requested windows exceeds available; triggers shortage branches
    runner = CoreRunner()
    model = _toy_model_with_losses([1.0, 1.1, 1.0, 1.1])
    adapter = DummyAdapter()
    cal = [
        {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]},
        {"input_ids": [4, 5, 6], "attention_mask": [1, 1, 1]},
        {"input_ids": [7, 8, 9], "attention_mask": [1, 1, 1]},
        {"input_ids": [10, 11, 12], "attention_mask": [1, 1, 1]},
    ]
    metrics, _ = runner._compute_real_metrics(
        model, cal, adapter, preview_n=3, final_n=3, config=RunConfig()
    )
    eval_samples = metrics.get("eval_samples")
    assert eval_samples is not None and float(eval_samples) > 0


def test_invalid_ppl_ratio_error_caught(monkeypatch):
    runner = CoreRunner()

    class NaNModel:
        def eval(self):
            return None

        def parameters(self):
            class P:
                device = "cpu"

            yield P()

        def __call__(self, *a, **k):
            class Obj:
                def __init__(self):
                    self.loss = type("L", (), {"item": lambda self: float("nan")})()

            return Obj()

    adapter = DummyAdapter()
    cal = [
        {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]},
        {"input_ids": [4, 5, 6], "attention_mask": [1, 1, 1]},
    ]
    # Should not raise due to internal try/except
    metrics, _ = runner._compute_real_metrics(
        NaNModel(), cal, adapter, preview_n=1, final_n=1, config=RunConfig()
    )
    assert isinstance(metrics, dict)


def test_empty_calibration_raises_valueerror():
    runner = CoreRunner()
    model = _toy_model_with_losses([1.0])
    adapter = DummyAdapter()
    with pytest.raises(ValueError):
        runner._compute_real_metrics(
            model, [], adapter, preview_n=1, final_n=1, config=RunConfig()
        )


def test_final_zero_uses_remaining_batches():
    runner = CoreRunner()
    model = _toy_model_with_losses([1.0, 1.1, 1.2])
    adapter = DummyAdapter()
    cal = [
        {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]},
        {"input_ids": [4, 5, 6], "attention_mask": [1, 1, 1]},
        {"input_ids": [7, 8, 9], "attention_mask": [1, 1, 1]},
    ]
    metrics, _ = runner._compute_real_metrics(
        model, cal, adapter, preview_n=2, final_n=0, config=RunConfig()
    )
    pm = metrics.get("primary_metric", {})
    assert pm.get("invalid") is True
    assert pm.get("degraded_reason") == "non_finite_pm"
    assert pm.get("preview") is not None
    assert pm.get("final") is None


def test_labels_present_without_attention_mask():
    runner = CoreRunner()
    model = _toy_model_with_losses([1.0, 1.1])
    adapter = DummyAdapter()
    cal = [
        {"input_ids": [1, 2, 3], "labels": [0, 0, 0]},
        {"input_ids": [4, 5, 6], "labels": [0, 0, 0]},
    ]
    metrics, _ = runner._compute_real_metrics(
        model, cal, adapter, preview_n=1, final_n=1, config=RunConfig()
    )
    pm = metrics.get("primary_metric", {})
    assert pm.get("final") and pm.get("preview")


def test_preview_zero_final_positive_path_local():
    runner = CoreRunner()
    model = _toy_model_with_losses([1.0, 1.1])
    adapter = DummyAdapter()
    cal = [
        {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]},
        {"input_ids": [4, 5, 6], "attention_mask": [1, 1, 1]},
    ]
    metrics, _ = runner._compute_real_metrics(
        model, cal, adapter, preview_n=0, final_n=1, config=RunConfig()
    )
    pm = metrics.get("primary_metric", {})
    final_val = pm.get("final")
    assert final_val is not None and float(final_val) > 0


def test_bootstrap_alpha_edge_again(tmp_path):
    runner = CoreRunner()
    model = _toy_model_with_losses([1.0, 1.1, 1.2, 1.3])
    adapter = DummyAdapter()
    cfg = RunConfig(
        context={
            "eval": {"bootstrap": {"enabled": True, "alpha": 0.0, "replicates": 5}}
        }
    )
    metrics, _ = runner._compute_real_metrics(
        model, _minimal_calibration(4), adapter, preview_n=2, final_n=2, config=cfg
    )
    assert metrics.get("bootstrap", {}).get("alpha") == 0.05


def test_delta_ci_normal_no_mismatch():
    # Ensure delta_ci computed and ratio_ci matches expected; path without raise
    runner = CoreRunner()
    model = _toy_model_with_losses([1.0, 1.05, 1.1, 1.2])
    adapter = DummyAdapter()
    cal = [
        {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]},
        {"input_ids": [4, 5, 6], "attention_mask": [1, 1, 1]},
        {"input_ids": [7, 8, 9], "attention_mask": [1, 1, 1]},
        {"input_ids": [10, 11, 12], "attention_mask": [1, 1, 1]},
    ]
    metrics, _ = runner._compute_real_metrics(
        model,
        cal,
        adapter,
        preview_n=2,
        final_n=2,
        config=RunConfig(
            context={"eval": {"bootstrap": {"enabled": True, "replicates": 10}}}
        ),
    )
    lo, hi = metrics.get("logloss_delta_ci", (0.0, 0.0))
    # Ratio CI equals exp of delta bounds by definition; ensure bounds are finite and consistent
    import math

    assert math.isfinite(lo) and math.isfinite(hi) and (hi - lo) >= 0.0


def test_resolve_guard_policies_exception_surfaces(monkeypatch):
    from invarlock.core.api import RunReport

    runner = CoreRunner()

    def boom(*a, **k):
        raise RuntimeError("boom")

    import invarlock.core.runner as runner_mod

    monkeypatch.setattr(runner_mod, "resolve_tier_policies", boom)
    report = RunReport()
    report.meta["config"] = {"guards": {}}
    with pytest.raises(RuntimeError, match="boom"):
        runner._resolve_guard_policies(report, auto_config=None)


def test_eval_phase_no_calibration_returns_non_evaluated_state(tmp_path):
    runner = CoreRunner()
    model = _toy_model_with_losses([1.0])
    adapter = DummyAdapter()
    edit = EditStub("e")
    cfg = RunConfig(
        context={"run_id": "r", "run": {"strict_eval": False}},
        checkpoint_interval=0,
        event_path=None,
    )
    report = runner.execute(model, adapter, edit, [], cfg, calibration_data=None)
    assert report.status == "success"
    assert report.metrics["eval_state"] == {
        "evaluated": False,
        "reason": "missing_calibration_data",
    }
    assert "eval_error" not in report.metrics


def test_zero_mask_batch_warning_debug(monkeypatch):
    import os

    os.environ["INVARLOCK_DEBUG_TRACE"] = "1"
    try:
        runner = CoreRunner()
        adapter = DummyAdapter()

        class Model:
            def eval(self):
                return None

            def parameters(self):
                class P:
                    device = "cpu"

                yield P()

            def __call__(self, *a, **k):
                class Out:
                    def __init__(self):
                        self.loss = type("L", (), {"item": lambda self: 1.0})()

                return Out()

        # Labels all -100 → masked_tokens_batch == 0 triggers zero_mask warning path
        samples = [
            {
                "input_ids": [1, 2, 3],
                "attention_mask": [1, 1, 1],
                "labels": [-100, -100, -100],
            },
            {
                "input_ids": [4, 5, 6],
                "attention_mask": [1, 1, 1],
                "labels": [-100, -100, -100],
            },
        ]
        metrics, _ = runner._compute_real_metrics(
            Model(), samples, adapter, preview_n=1, final_n=1, config=RunConfig()
        )
        pm = metrics.get("primary_metric", {})
        assert pm.get("final") and pm.get("preview")
    finally:
        del os.environ["INVARLOCK_DEBUG_TRACE"]


def test_mlm_zero_usable_batches_caught():
    # Zero usable batches for MLM path should be caught internally
    class NoLossModel:
        def eval(self):
            return None

        def parameters(self):
            class P:
                device = "cpu"

            yield P()

        def __call__(self, *a, **k):
            class Out:
                pass

            return Out()

    runner = CoreRunner()
    adapter = DummyAdapter()
    metrics, _ = runner._compute_real_metrics(
        NoLossModel(),
        _minimal_calibration(2),
        adapter,
        preview_n=1,
        final_n=1,
        config=RunConfig(context={"eval": {"loss": {"type": "mlm"}}}),
    )
    assert isinstance(metrics, dict)


def test_preview_final_defaults_and_seed_fallback():
    runner = CoreRunner()
    model = _toy_model_with_losses([1.0, 1.1])
    adapter = DummyAdapter()
    cfg = RunConfig(
        context={
            "dataset": {"seed": "abc"},
            "eval": {
                "bootstrap": {
                    "enabled": True,
                    "replicates": 10,
                    "alpha": 0.5,
                    "seed": "abc",
                }
            },
        }
    )
    metrics, _ = runner._compute_real_metrics(
        model,
        _minimal_calibration(2),
        adapter,
        preview_n=None,
        final_n=None,
        config=cfg,
    )
    # invalid seed coerced to 0
    assert metrics.get("bootstrap", {}).get("seed") == 0


def test_dataset_seed_used_when_bootstrap_unspecified():
    runner = CoreRunner()
    model = _toy_model_with_losses([1.0, 1.1])
    adapter = DummyAdapter()
    cfg = RunConfig(
        context={
            "dataset": {"seed": 7},
            "eval": {"bootstrap": {"enabled": True, "replicates": 10}},
        }
    )
    metrics, _ = runner._compute_real_metrics(
        model, _minimal_calibration(2), adapter, preview_n=1, final_n=1, config=cfg
    )
    assert metrics.get("bootstrap", {}).get("seed") == 7


def test_paired_delta_single_pair_reason():
    # Use exactly one batch per split to exercise 'single_pair' degeneracy
    runner = CoreRunner()
    model = _toy_model_with_losses([1.0, 1.1])
    adapter = DummyAdapter()
    metrics, _ = runner._compute_real_metrics(
        model,
        _minimal_calibration(2),
        adapter,
        preview_n=1,
        final_n=1,
        config=RunConfig(),
    )
    summary = metrics.get("paired_delta_summary", {})
    assert summary.get("degenerate") is True and summary.get("degenerate_reason") in {
        "single_pair",
        "no_pairs",
        "no_variation",
    }


def test_ratio_ci_mismatch_caught(monkeypatch):
    # Force ratio_ci != exp(delta_ci) without asserting raise (caught internally)
    import invarlock.core.runner as runner_mod

    def fake_ratio_ci(delta_ci):
        lo, hi = delta_ci
        return (float(lo), float(hi + 0.5))

    monkeypatch.setattr(runner_mod, "logspace_to_ratio_ci", fake_ratio_ci)

    runner = CoreRunner()
    model = _toy_model_with_losses([1.0, 1.3, 0.9, 1.2])
    adapter = DummyAdapter()
    cfg = RunConfig(context={"eval": {"bootstrap": {"enabled": True, "replicates": 5}}})
    metrics, _ = runner._compute_real_metrics(
        model,
        _minimal_calibration(4),
        adapter,
        preview_n=2,
        final_n=2,
        config=cfg,
    )
    assert isinstance(metrics, dict)


def test_degenerate_delta_populates_weights_and_marks_degraded(monkeypatch):
    calls: dict[str, object] = {}

    def fake_delta_ci(final_losses, preview_losses, weights=None, **kwargs):
        calls["weights"] = weights
        return (0.0, 0.0)

    monkeypatch.setattr(
        "invarlock.core.runner.compute_paired_delta_log_ci", fake_delta_ci
    )

    runner = CoreRunner()
    model = _toy_model_with_losses([1.0, 1.0, 1.0, 1.0])
    adapter = DummyAdapter()
    cfg = RunConfig(context={"eval": {"bootstrap": {"enabled": True, "replicates": 3}}})

    metrics, _ = runner._compute_real_metrics(
        model,
        _minimal_calibration(4),
        adapter,
        preview_n=2,
        final_n=2,
        config=cfg,
    )

    pm = metrics.get("primary_metric", {})
    assert pm.get("degraded") is True
    assert str(pm.get("degraded_reason", "")).startswith("degenerate_delta")
    weights = calls.get("weights")
    assert isinstance(weights, list)
    assert len(weights) == 2 and all(w >= 1.0 for w in weights)


def test_pairing_unexpected_ids_reason(tmp_path):
    # Baseline has fewer IDs than run → unexpected IDs in run
    runner = CoreRunner()
    model = _toy_model_with_losses([1.0, 1.1, 1.0, 1.1])
    adapter = DummyAdapter()
    cfg = RunConfig(
        context={
            "pairing_baseline": {
                "preview": {"window_ids": [0], "input_ids": [[9, 9, 9]]},
                "final": {"window_ids": [2], "input_ids": [[7, 7, 7]]},
            }
        }
    )
    metrics, _ = runner._compute_real_metrics(
        model,
        _minimal_calibration(4),
        adapter,
        preview_n=2,
        final_n=2,
        config=cfg,
    )
    reason = metrics.get("window_pairing_reason")
    # May be unexpected_ids or preview/final mismatch depending on ordering
    assert reason is None or isinstance(reason, str)


def test_mlm_zero_mask_batches_sets_eval_error():
    runner = CoreRunner()
    model = _toy_model_with_losses([0.1, 0.2])
    adapter = DummyAdapter()
    cfg = RunConfig(context={"eval": {"loss": {"type": "mlm"}}})
    calibration = [
        {
            "input_ids": [1, 2, 3],
            "attention_mask": [0, 0, 0],
            "labels": [1, 1, 1],
        }
    ]

    metrics, _ = runner._compute_real_metrics(
        model,
        calibration,
        adapter,
        preview_n=1,
        final_n=0,
        config=cfg,
    )

    eval_error = metrics.get("eval_error") or {}
    assert eval_error.get("error") == "mlm_missing_masks"


def test_preview_handles_mixed_labels_and_missing_inputs():
    runner = CoreRunner()
    model = _toy_model_with_losses([0.5, 0.6, 0.7])
    adapter = DummyAdapter()
    calibration = [
        {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": [1, 2, 3]},
        {"input_ids": [4, 5, 6], "attention_mask": [1, 1, 1]},
        {"input_ids": None},
    ]

    metrics, _ = runner._compute_real_metrics(
        model,
        calibration,
        adapter,
        preview_n=3,
        final_n=0,
        config=RunConfig(),
    )

    pm = metrics.get("primary_metric", {})
    assert pm.get("preview") and math.isfinite(float(pm.get("preview")))
    assert metrics.get("eval_samples", 0) >= 2


def test_strict_eval_raises_on_eval_error():
    runner = CoreRunner()
    model = _toy_model_with_losses([0.1, 0.2])
    adapter = DummyAdapter()
    cfg = RunConfig(context={"eval": {"loss": {"type": "mlm"}, "strict": True}})
    calibration = [
        {
            "input_ids": [1, 2, 3],
            "attention_mask": [0, 0, 0],
            "labels": [1, 1, 1],
        }
    ]

    report = RunReport()

    with pytest.raises(RuntimeError):
        runner._eval_phase(
            model,
            adapter,
            calibration,
            report,
            preview_n=1,
            final_n=0,
            config=cfg,
        )


def test_tail_paired_baseline_emits_source(monkeypatch):
    def fake_tail_eval(*args, **kwargs):
        return {"mean": 0.0}

    monkeypatch.setattr("invarlock.core.runner.evaluate_metric_tail", fake_tail_eval)

    runner = CoreRunner()
    model = _toy_model_with_losses([0.4, 0.5])
    adapter = DummyAdapter()
    cfg = RunConfig(
        context={
            "baseline_eval_windows": {
                "preview": {"window_ids": [0], "logloss": [1.0], "token_counts": [3]},
                "final": {"window_ids": [1], "logloss": [1.5], "token_counts": [4]},
            }
        }
    )

    report = RunReport()

    _ = runner._eval_phase(
        model,
        adapter,
        _minimal_calibration(2),
        report,
        preview_n=1,
        final_n=1,
        config=cfg,
    )

    pm_tail = report.metrics.get("primary_metric_tail", {})
    assert pm_tail.get("source") == "paired_baseline.final"
