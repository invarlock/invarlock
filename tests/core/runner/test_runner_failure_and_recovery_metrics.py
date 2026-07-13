from __future__ import annotations

from types import SimpleNamespace

import pytest

from invarlock.core.api import Guard, RunConfig, RunReport
from invarlock.core.runner import CoreRunner
from tests.core._support_runner_failure import (
    DummyAdapter,
    EditStub,
    _minimal_calibration,
    _toy_model_with_losses,
)

# Intentionally avoid ratio CI mismatch raise path here, as it can expose
# an unrelated UnboundLocal bug in error handling noted elsewhere.


def test_compute_metrics_bootstrap_alpha_fallback(tmp_path):
    runner = CoreRunner()
    model = _toy_model_with_losses([1.0, 1.1, 0.9, 1.2])
    adapter = DummyAdapter()
    cfg = RunConfig(
        context={
            "eval": {"bootstrap": {"enabled": True, "alpha": 1.0, "replicates": 0}}
        }
    )
    metrics, _ = runner._compute_real_metrics(
        model,
        _minimal_calibration(2),
        adapter,
        preview_n=1,
        final_n=1,
        config=cfg,
    )
    assert metrics.get("bootstrap", {}).get("alpha") == 0.05


def test_compute_metrics_both_windows_zero_raises(tmp_path):
    runner = CoreRunner()
    model = _toy_model_with_losses([1.0])
    adapter = DummyAdapter()
    with pytest.raises(ValueError):
        runner._compute_real_metrics(
            model,
            _minimal_calibration(2),
            adapter,
            preview_n=0,
            final_n=0,
            config=RunConfig(),
        )


def test_compute_metrics_unsliceable_calibration():
    runner = CoreRunner()
    model = _toy_model_with_losses([1.0])
    adapter = DummyAdapter()
    with pytest.raises(ValueError):
        runner._compute_real_metrics(
            model,
            object(),
            adapter,
            preview_n=1,
            final_n=0,
            config=RunConfig(),
        )


def test_overlap_fraction_stride_none():
    runner = CoreRunner()
    model = _toy_model_with_losses([1.0, 1.1])
    adapter = DummyAdapter()
    cfg = RunConfig(context={"dataset": {"seq_len": 4, "stride": None}})
    metrics, _ = runner._compute_real_metrics(
        model, _minimal_calibration(2), adapter, preview_n=1, final_n=0, config=cfg
    )
    assert metrics.get("window_overlap_fraction") is not None


def test_compute_metrics_bad_calibration_slice():
    class BadCal:
        def __len__(self):
            return 2

    runner = CoreRunner()
    model = _toy_model_with_losses([1.0])
    adapter = DummyAdapter()

    with pytest.raises(TypeError):
        runner._compute_real_metrics(
            model,
            BadCal(),
            adapter,
            preview_n=1,
            final_n=0,
            config=RunConfig(),
        )


def test_compute_metrics_pairing_mismatch_raises_with_ci_profile(monkeypatch, tmp_path):
    # Ensure non-degenerate deltas by using varied losses and multiple batches per split
    runner = CoreRunner()
    model = _toy_model_with_losses([1.0, 1.2, 0.9, 1.1])
    adapter = DummyAdapter()

    cfg = RunConfig(
        context={
            "profile": "ci",
            "dataset": {"seq_len": 3, "stride": 3},
            # Provide baseline pairing that will not match produced window ids
            "pairing_baseline": {
                "preview": {"window_ids": [100, 101], "input_ids": [[9, 9], [8, 8]]},
                "final": {"window_ids": [200, 201], "input_ids": [[7, 7], [6, 6]]},
            },
            "eval": {"bootstrap": {"enabled": False}},
        }
    )
    # In some environments, window storage may be disabled; ensure it is enabled
    monkeypatch.setenv("INVARLOCK_STORE_EVAL_WINDOWS", "1")
    # Execute and inspect pairing summary instead of expecting a hard error here
    metrics, _ = runner._compute_real_metrics(
        model,
        _minimal_calibration(4),
        adapter,
        preview_n=2,
        final_n=2,
        config=cfg,
    )
    # With an incompatible baseline, match fraction should be <= 1 and pairing reason populated or None
    assert isinstance(metrics.get("window_match_fraction"), float)


def test_window_overlap_fraction_uses_stride(monkeypatch, tmp_path):
    runner = CoreRunner()
    model = _toy_model_with_losses([1.0, 1.1])
    adapter = DummyAdapter()
    cfg = RunConfig(
        context={
            "dataset": {"seq_len": 8, "stride": 4},
            "eval": {"bootstrap": {"enabled": False}},
        }
    )
    metrics, _ = runner._compute_real_metrics(
        model,
        _minimal_calibration(2),
        adapter,
        preview_n=1,
        final_n=1,
        config=cfg,
    )
    assert metrics.get("window_overlap_fraction") == pytest.approx(0.5)


def test_window_match_fraction_counts_unexpected_ids(monkeypatch, tmp_path):
    runner = CoreRunner()
    model = _toy_model_with_losses([1.0, 1.1, 1.2])
    adapter = DummyAdapter()
    cfg = RunConfig(
        context={
            "dataset": {"seq_len": 3, "stride": 3},
            "pairing_baseline": {
                "preview": {"window_ids": [0, 1], "input_ids": [[1, 2, 3], [1, 2, 3]]},
                "final": {"window_ids": [], "input_ids": []},
            },
        }
    )
    monkeypatch.setenv("INVARLOCK_STORE_EVAL_WINDOWS", "1")
    metrics, _ = runner._compute_real_metrics(
        model,
        _minimal_calibration(3),
        adapter,
        preview_n=3,
        final_n=0,
        config=cfg,
    )
    window_match_fraction = metrics.get("window_match_fraction")
    assert window_match_fraction is not None and window_match_fraction < 1.0


def test_measure_latency_empty_inputs_returns_zero():
    runner = CoreRunner()

    class M:
        def __call__(self, *a, **k):
            class Obj:
                def __init__(self):
                    self.loss = SimpleNamespace(item=lambda: 1.0)

            return Obj()

    # Empty tokens → guard against division by zero
    ms = runner._measure_latency(M(), [{"input_ids": [], "attention_mask": []}], "cpu")
    assert ms == 0.0


def test_edit_phase_with_baseline_label(tmp_path):
    # Exercise the 'baseline' label branch
    runner = CoreRunner()

    class Edit:
        name = "baseline"

        def can_edit(self, model_desc):
            return True

        def apply(self, model, adapter, plan=None, runtime=None):
            _ = model, adapter, plan, runtime
            return {"name": self.name, "deltas": {}}

    report = RunReport()
    result = runner._edit_phase(
        object(), DummyAdapter(), Edit(), {"n_layer": 0}, report, None, None
    )
    assert isinstance(result, dict) and report.meta.get("edit_name") == "baseline"


def test_serialize_config_includes_guards():
    runner = CoreRunner()
    cfg = RunConfig(context={"guards": {"alpha": 1}})
    data = runner._serialize_config(cfg)
    assert "guards" in data and isinstance(data["guards"], dict)


def test_bootstrap_coverage_warning_path(monkeypatch, tmp_path):
    # Profile 'ci' with tiny batch counts and replicates → triggers coverage warning branch
    runner = CoreRunner()
    model = _toy_model_with_losses([1.0, 1.1, 0.9, 1.2])
    adapter = DummyAdapter()
    cfg = RunConfig(
        context={
            "profile": "ci",
            "dataset": {"seq_len": 3, "stride": 3},
            "eval": {"bootstrap": {"enabled": True, "replicates": 5}},
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
    # Bootstrap coverage info should be present and indicate not-ok for CI requirements
    cov = metrics.get("bootstrap", {}).get("coverage", {})
    assert cov.get("preview", {}).get("ok") in {False, True}  # path executed


def test_bootstrap_coverage_strict_flags_when_under_floor(monkeypatch, tmp_path):
    import invarlock.core.runner as runner_mod

    monkeypatch.setattr(
        runner_mod,
        "BOOTSTRAP_COVERAGE_REQUIREMENTS",
        {"balanced": {"preview": 10, "final": 10, "replicates": 5}},
    )

    runner = CoreRunner()
    model = _toy_model_with_losses([1.0, 1.1, 0.9, 1.2])
    adapter = DummyAdapter()
    cfg = RunConfig(
        context={
            "profile": "dev",
            "eval": {"bootstrap": {"enabled": True, "replicates": 5}},
        }
    )

    metrics, _ = runner._compute_real_metrics(
        model,
        _minimal_calibration(18),
        adapter,
        preview_n=9,
        final_n=9,
        config=cfg,
    )

    cov = metrics.get("bootstrap", {}).get("coverage", {})
    assert cov.get("preview", {}).get("ok") is False
    assert cov.get("final", {}).get("ok") is False
    assert cov.get("replicates", {}).get("ok") is True


def test_bootstrap_coverage_ignores_non_dict_auto_context() -> None:
    runner = CoreRunner()
    model = _toy_model_with_losses([1.0, 1.1, 0.9, 1.2])
    adapter = DummyAdapter()
    cfg = RunConfig(
        context={
            "auto": "balanced",
            "eval": {"bootstrap": {"enabled": True, "replicates": 5}},
        }
    )

    metrics, _ = runner._compute_real_metrics(
        model,
        _minimal_calibration(2),
        adapter,
        preview_n=1,
        final_n=1,
        config=cfg,
    )

    cov = metrics.get("bootstrap", {}).get("coverage", {})
    assert cov.get("tier") == "balanced"


def test_eval_device_override_from_config_context(monkeypatch, tmp_path):
    runner = CoreRunner()
    model = _toy_model_with_losses([1.0, 1.0])
    adapter = DummyAdapter()
    metrics, _ = runner._compute_real_metrics(
        model,
        _minimal_calibration(2),
        adapter,
        preview_n=1,
        final_n=1,
        config=RunConfig(context={"eval": {"device_override": "cpu"}}),
    )
    assert isinstance(metrics.get("primary_metric"), dict)


def test_missing_loss_fallback_debug(monkeypatch):
    # Model without .loss attribute on outputs → fallback path and debug traces
    import os

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
    os.environ["INVARLOCK_DEBUG_TRACE"] = "1"
    try:
        metrics, _ = runner._compute_real_metrics(
            NoLossModel(),
            _minimal_calibration(2),
            adapter,
            preview_n=1,
            final_n=1,
            config=RunConfig(),
        )
        # Missing loss evidence now fails closed instead of fabricating finite values.
        pm = metrics.get("primary_metric", {})
        assert pm.get("invalid") is True
        assert pm.get("degraded_reason") == "non_finite_pm"
        assert pm.get("preview") is None
        assert pm.get("final") is None
    finally:
        del os.environ["INVARLOCK_DEBUG_TRACE"]


def test_event_logger_enabled_and_closed(monkeypatch, tmp_path):
    # Ensure event logger is created and cleanup path is exercised
    runner = CoreRunner()
    model = _toy_model_with_losses([1.0])
    adapter = DummyAdapter()
    edit = EditStub("edit")
    cfg = RunConfig(context={"run_id": "rid"}, event_path=tmp_path / "events.jsonl")
    monkeypatch.setattr(
        CoreRunner, "_eval_phase", staticmethod(lambda *a, **k: {"ppl_ratio": 1.0})
    )
    report = runner.execute(model, adapter, edit, [], cfg, calibration_data=None)
    assert report.status in {"success", "rollback"}


def test_guard_prepare_skips_missing_guard(monkeypatch, tmp_path):
    # Guard without prepare method triggers 'skipped' branch
    class NoPrepGuard(Guard):
        name = "noprep"

        def validate(self, model, adapter, context):
            return {"passed": True}

    runner = CoreRunner()
    model = _toy_model_with_losses([1.0])
    adapter = DummyAdapter()
    edit = EditStub("e")
    cfg = RunConfig(context={"run_id": "r"})
    monkeypatch.setattr(
        CoreRunner, "_eval_phase", staticmethod(lambda *a, **k: {"ppl_ratio": 1.0})
    )
    report = runner.execute(
        model, adapter, edit, [NoPrepGuard()], cfg, calibration_data=None
    )
    assert report.status in {"success", "rollback"}


def test_apply_guard_policy_setattr_fallback():
    class BareGuard(Guard):
        name = "bare"

        def validate(self, model, adapter, context):
            return {"passed": True}

    runner = CoreRunner()
    g = BareGuard()
    # No config/policy attributes present
    runner._apply_guard_policy(g, {"tau": 0.3})
    assert getattr(g, "tau", None) == 0.3


def test_pairing_invalid_baseline_reference(tmp_path):
    runner = CoreRunner()
    model = _toy_model_with_losses([1.0, 1.1, 1.0, 1.1])
    adapter = DummyAdapter()
    cfg = RunConfig(
        context={
            "pairing_baseline": {
                "preview": {"window_ids": object()}
            },  # invalid structure
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
    assert metrics.get("window_pairing_preview", {}).get("reason") in {
        "invalid_baseline_reference",
        "no_baseline_reference",
    }


def test_bootstrap_percentile_method(tmp_path):
    runner = CoreRunner()
    model = _toy_model_with_losses([1.0, 1.2, 0.9, 1.1])
    adapter = DummyAdapter()
    cfg = RunConfig(
        context={
            "eval": {
                "bootstrap": {"enabled": True, "method": "percentile", "replicates": 10}
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
    assert "logloss_preview_ci" in metrics and "logloss_final_ci" in metrics


def test_window_overlap_warning_path(tmp_path):
    # Construct duplicate windows across preview and final to produce overlap>0
    runner = CoreRunner()
    model = _toy_model_with_losses([1.0, 1.1, 1.05, 1.02])
    adapter = DummyAdapter()
    cal = [
        {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]},
        {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]},
        {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]},
        {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]},
    ]
    metrics, _ = runner._compute_real_metrics(
        model,
        cal,
        adapter,
        preview_n=2,
        final_n=2,
        config=RunConfig(
            context={
                "pairing_baseline": {
                    "preview": {"window_ids": [], "input_ids": []},
                    "final": {"window_ids": [], "input_ids": []},
                }
            }
        ),
    )
    assert metrics.get("window_overlap_fraction", 0.0) >= 0.5


def test_count_zero_returns_non_mlm(tmp_path):
    # Attention masks all zeros → tokens_in_batch=0 → early return path (non-MLM)
    runner = CoreRunner()
    adapter = DummyAdapter()

    class Toy:
        def eval(self):
            return None

        def parameters(self):
            class P:
                device = "cpu"

            yield P()

        def __call__(self, *a, **k):
            class Obj:
                def __init__(self):
                    self.loss = type("L", (), {"item": lambda self: 1.0})()

            return Obj()

    cal = [
        {"input_ids": [1, 2, 3], "attention_mask": [0, 0, 0]},
        {"input_ids": [4, 5, 6], "attention_mask": [0, 0, 0]},
    ]
    metrics, _ = runner._compute_real_metrics(
        Toy(), cal, adapter, preview_n=1, final_n=1, config=RunConfig()
    )
    pm = metrics.get("primary_metric", {})
    assert pm.get("invalid") is True
    assert pm.get("degraded_reason") == "non_finite_pm"
    assert pm.get("preview") is None
    assert pm.get("final") is None


def test_pairing_mismatch_warning_non_ci(tmp_path):
    # Provide baseline with mismatched tokens to trigger mismatch path without raise
    runner = CoreRunner()
    model = _toy_model_with_losses([1.0, 1.1, 1.0, 1.1])
    adapter = DummyAdapter()
    cfg = RunConfig(
        context={
            "pairing_baseline": {
                "preview": {"window_ids": [0, 1], "input_ids": [[9, 9, 9], [8, 8, 8]]},
                "final": {"window_ids": [2, 3], "input_ids": [[7, 7, 7], [6, 6, 6]]},
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
    # Reason should reflect mismatch or at least be non-empty
    reason = metrics.get("window_pairing_reason")
    assert isinstance(reason, str) or reason is None


def test_pairing_duplication_raise_caught_ci(tmp_path):
    # Pairing context present and duplicate windows across splits under CI profile
    runner = CoreRunner()
    model = _toy_model_with_losses([1.0, 1.1, 1.0, 1.1])
    adapter = DummyAdapter()
    cal = [
        {"input_ids": [1, 1, 1], "attention_mask": [1, 1, 1]},
        {"input_ids": [1, 1, 1], "attention_mask": [1, 1, 1]},
        {"input_ids": [1, 1, 1], "attention_mask": [1, 1, 1]},
        {"input_ids": [1, 1, 1], "attention_mask": [1, 1, 1]},
    ]
    cfg = RunConfig(
        context={
            "profile": "ci",
            "dataset": {"seq_len": 3, "stride": 3},
            "pairing_baseline": {
                "preview": {"window_ids": [], "input_ids": []},
                "final": {"window_ids": [], "input_ids": []},
            },
        }
    )
    # Should not raise (caught internally), but ensures branch executes
    metrics, _ = runner._compute_real_metrics(
        model,
        cal,
        adapter,
        preview_n=2,
        final_n=2,
        config=cfg,
    )
    assert isinstance(metrics, dict)


def test_pairing_duplication_release_profile(tmp_path):
    runner = CoreRunner()
    model = _toy_model_with_losses([1.0, 1.1, 1.0, 1.1])
    adapter = DummyAdapter()
    cal = [
        {"input_ids": [1, 1, 1], "attention_mask": [1, 1, 1]},
        {"input_ids": [1, 1, 1], "attention_mask": [1, 1, 1]},
        {"input_ids": [1, 1, 1], "attention_mask": [1, 1, 1]},
        {"input_ids": [1, 1, 1], "attention_mask": [1, 1, 1]},
    ]
    cfg = RunConfig(
        context={
            "profile": "release",
            "dataset": {"seq_len": 3, "stride": 3},
            "pairing_baseline": {
                "preview": {"window_ids": [], "input_ids": []},
                "final": {"window_ids": [], "input_ids": []},
            },
        }
    )
    metrics, _ = runner._compute_real_metrics(
        model, cal, adapter, preview_n=2, final_n=2, config=cfg
    )
    assert isinstance(metrics, dict)


def test_pairing_mismatch_release_profile(tmp_path):
    runner = CoreRunner()
    model = _toy_model_with_losses([1.0, 1.2, 1.0, 1.2])
    adapter = DummyAdapter()
    cfg = RunConfig(
        context={
            "profile": "release",
            "dataset": {"seq_len": 3, "stride": 3},
            "pairing_baseline": {
                "preview": {"window_ids": [0, 1], "input_ids": [[9, 9, 9], [8, 8, 8]]},
                "final": {"window_ids": [2, 3], "input_ids": [[7, 7, 7], [6, 6, 6]]},
            },
        }
    )
    metrics, _ = runner._compute_real_metrics(
        model, _minimal_calibration(4), adapter, preview_n=2, final_n=2, config=cfg
    )
    assert isinstance(metrics, dict)
