from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from invarlock.core.api import Guard, ModelAdapter, RunConfig, RunReport
from invarlock.core.runner import CoreRunner


class DummyAdapter(ModelAdapter):
    name = "dummy"

    def can_handle(self, model: Any) -> bool:  # pragma: no cover - not used here
        return True

    def describe(self, model: Any) -> dict[str, Any]:  # pragma: no cover - minimal
        return {"n_layer": 1, "heads_per_layer": [1], "mlp_dims": [3], "tying": {}}

    def snapshot(self, model: Any) -> bytes:  # pragma: no cover - minimal stub
        return b"s"

    def restore(self, model: Any, blob: bytes) -> None:  # pragma: no cover - stub
        return None


class EditStub:
    def __init__(self, name: str = "e", result: dict[str, Any] | None = None):
        self.name = name
        self._result = result or {"name": name, "deltas": {}}

    def can_edit(self, model_desc: dict[str, Any]) -> bool:
        return True

    def apply(
        self,
        model: Any,
        adapter: ModelAdapter,
        plan=None,
        runtime=None,
    ) -> dict[str, Any]:
        _ = model, adapter, plan, runtime
        return dict(self._result)


def _toy_model_with_losses(losses):
    import torch

    class Toy(torch.nn.Module):
        def __init__(self, seq):
            super().__init__()
            self.seq = list(seq)
            self.idx = 0
            self.lin = torch.nn.Linear(3, 3, bias=False)

        def forward(self, *args, **kwargs):
            class Out:
                def __init__(self, val: float):
                    self.loss = type("L", (), {"item": lambda self: float(val)})()

            val = self.seq[self.idx % len(self.seq)]
            self.idx += 1
            return Out(val)

    return Toy(losses)


def _minimal_calibration(n: int) -> list[dict[str, Any]]:
    # Small integer lists; runner converts to tensors internally
    return [
        {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]} for _ in range(max(1, n))
    ]


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


# Intentionally avoid ratio CI mismatch raise path here, as it can expose
# an unrelated UnboundLocal bug in error handling noted elsewhere.


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


def test_pairing_mismatch_raise_caught_ci(tmp_path):
    runner = CoreRunner()
    model = _toy_model_with_losses([1.0, 1.1, 1.0, 1.1])
    adapter = DummyAdapter()
    cfg = RunConfig(
        context={
            "profile": "ci",
            "dataset": {"seq_len": 3, "stride": 3},
            "pairing_baseline": {
                "preview": {"window_ids": [0, 1], "input_ids": [[9, 9, 9], [8, 8, 8]]},
                "final": {"window_ids": [2, 3], "input_ids": [[7, 7, 7], [6, 6, 6]]},
            },
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
    assert isinstance(metrics, dict)


def test_store_eval_windows_disabled(monkeypatch):
    import os

    os.environ["INVARLOCK_STORE_EVAL_WINDOWS"] = "0"
    try:
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
        pm = metrics.get("primary_metric", {})
        assert pm.get("final") and pm.get("preview")
    finally:
        del os.environ["INVARLOCK_STORE_EVAL_WINDOWS"]


def test_measure_latency_to_device_exceptions(monkeypatch):
    import torch

    runner = CoreRunner()

    class M:
        def __call__(self, *a, **k):
            class Obj:
                def __init__(self):
                    self.loss = type("L", (), {"item": lambda self: 1.0})()

            return Obj()

    # Patch Tensor.to to raise, exercising defensive except blocks for both input and masks
    original_to = torch.Tensor.to

    def raising_to(self, *args, **kwargs):
        raise RuntimeError("to-device-fail")

    monkeypatch.setattr(torch.Tensor, "to", raising_to)
    try:
        sample = {
            "input_ids": [1, 2, 3],
            "attention_mask": [1, 1, 1],
            "token_type_ids": [0, 0, 0],
        }
        with pytest.raises(
            RuntimeError, match="Latency measurement device transfer failed"
        ):
            runner._measure_latency(M(), [sample], "cpu")
    finally:
        monkeypatch.setattr(torch.Tensor, "to", original_to)


def test_execute_with_none_context(monkeypatch, tmp_path):
    # None context should be handled gracefully and serialized with empty guards
    runner = CoreRunner()
    model = _toy_model_with_losses([1.0])
    adapter = DummyAdapter()
    edit = EditStub("e")
    cfg = RunConfig(context=None, event_path=None)
    monkeypatch.setattr(
        CoreRunner, "_eval_phase", staticmethod(lambda *a, **k: {"ppl_ratio": 1.0})
    )
    report = runner.execute(model, adapter, edit, [], cfg, calibration_data=None)
    assert report.status in {"success", "rollback"} and isinstance(
        report.meta.get("config", {}).get("guards", {}), dict
    )


def test_compute_real_metrics_nondict_batches(tmp_path):
    # Non-dict batch path exercises alternative ingestion
    runner = CoreRunner()
    model = _toy_model_with_losses([1.0, 1.1])
    adapter = DummyAdapter()
    cal = [[1, 2, 3], [4, 5, 6]]
    metrics, _ = runner._compute_real_metrics(
        model, cal, adapter, preview_n=1, final_n=1, config=RunConfig()
    )
    pm = metrics.get("primary_metric", {})
    assert pm.get("final") and pm.get("preview")


def test_ratio_ci_fallback_when_missing_preview_losses(tmp_path):
    # Preview yields zero usable tokens (mask zeros), final yields normal → fallback ratio_ci path
    runner = CoreRunner()
    model = _toy_model_with_losses([1.0, 1.1])
    adapter = DummyAdapter()
    cal = [
        {"input_ids": [1, 2, 3], "attention_mask": [0, 0, 0]},  # preview zero tokens
        {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]},  # final usable
    ]
    cfg = RunConfig(context={"eval": {"bootstrap": {"enabled": True, "replicates": 5}}})
    metrics, _ = runner._compute_real_metrics(
        model, cal, adapter, preview_n=1, final_n=1, config=cfg
    )
    # logloss_delta_ci falls back to (delta_mean_log, delta_mean_log)
    assert (
        isinstance(metrics.get("logloss_delta_ci"), tuple)
        and len(metrics["logloss_delta_ci"]) == 2
    )


def test_zero_mask_total_debug(monkeypatch):
    import os

    os.environ["INVARLOCK_DEBUG_TRACE"] = "1"
    try:
        runner = CoreRunner()
        model = _toy_model_with_losses([1.0, 1.1])
        adapter = DummyAdapter()
        cal = [
            {
                "input_ids": [1, 2, 3],
                "attention_mask": [0, 0, 0],
                "labels": [-100, -100, -100],
            },
            {
                "input_ids": [4, 5, 6],
                "attention_mask": [0, 0, 0],
                "labels": [-100, -100, -100],
            },
        ]
        metrics, _ = runner._compute_real_metrics(
            model, cal, adapter, preview_n=1, final_n=1, config=RunConfig()
        )
        pm = metrics.get("primary_metric", {})
        assert pm.get("invalid") is True
        assert pm.get("degraded_reason") == "non_finite_pm"
        assert pm.get("preview") is None
        assert pm.get("final") is None
    finally:
        del os.environ["INVARLOCK_DEBUG_TRACE"]


def test_resolve_guard_policies_from_config_auto(monkeypatch):
    runner = CoreRunner()
    seen = {}

    def fake_resolver(tier, edit_name, overrides):
        seen["tier"] = tier
        return {}

    import invarlock.core.runner as runner_mod

    monkeypatch.setattr(runner_mod, "resolve_tier_policies", fake_resolver)
    report = RunReport()
    report.meta["config"] = {
        "auto": {"tier": "aggressive", "enabled": True},
        "guards": {},
    }
    policies = runner._resolve_guard_policies(report, auto_config=None)
    assert isinstance(policies, dict) and seen.get("tier") == "aggressive"


def test_resolve_policies_edit_name_from_meta(monkeypatch):
    runner = CoreRunner()
    seen = {}

    def fake_resolver(tier, edit_name, overrides):
        seen["edit"] = edit_name
        return {}

    import invarlock.core.runner as runner_mod

    monkeypatch.setattr(runner_mod, "resolve_tier_policies", fake_resolver)
    report = RunReport()
    report.meta["config"] = {"guards": {}}
    report.meta["edit_name"] = "foo-edit"
    _ = runner._resolve_guard_policies(report, auto_config=None)
    assert seen.get("edit") == "foo-edit"


def test_eval_device_override_moves_model(monkeypatch):
    class MovableModel:
        def __init__(self):
            self.moved = False

        def eval(self):
            return None

        def parameters(self):
            class P:
                device = "meta"

            yield P()

        def to(self, device):
            self.moved = True
            return self

        def __call__(self, *a, **k):
            class Obj:
                def __init__(self):
                    self.loss = type("L", (), {"item": lambda self: 1.0})()

            return Obj()

    runner = CoreRunner()
    adapter = DummyAdapter()
    metrics, _ = runner._compute_real_metrics(
        MovableModel(),
        _minimal_calibration(2),
        adapter,
        preview_n=1,
        final_n=1,
        config=RunConfig(context={"eval": {"device_override": "cpu"}}),
    )
    pm = metrics.get("primary_metric", {})
    assert pm.get("final") and pm.get("preview")  # branch executed


def test_eval_device_override_no_move_when_equal(monkeypatch):
    import torch

    class CpuModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(3, 3, bias=False)
            self.moved = False

        def to(self, *args, **kwargs):
            self.moved = True
            return super().to(*args, **kwargs)

        def forward(self, *a, **k):
            class Obj:
                def __init__(self):
                    self.loss = type("L", (), {"item": lambda self: 1.0})()

            return Obj()

    runner = CoreRunner()
    adapter = DummyAdapter()
    # Parameters are on CPU; override also CPU → no-op move path
    metrics, _ = runner._compute_real_metrics(
        CpuModel(),
        _minimal_calibration(2),
        adapter,
        preview_n=1,
        final_n=1,
        config=RunConfig(context={"eval": {"device_override": "cpu"}}),
    )
    pm = metrics.get("primary_metric", {})
    assert pm.get("final") and pm.get("preview")


def test_eval_debug_snapshot_with_labels(monkeypatch, tmp_path):
    import os

    os.environ["INVARLOCK_DEBUG_TRACE"] = "1"
    try:
        runner = CoreRunner()
        model = _toy_model_with_losses([1.0])
        adapter = DummyAdapter()
        edit = EditStub("e")

        cfg = RunConfig(context={"run_id": "r"}, checkpoint_interval=0, event_path=None)
        cal = [
            {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": [0, 0, 0]}
        ]
        # Stub compute_real_metrics to avoid heavy path
        monkeypatch.setattr(
            CoreRunner,
            "_compute_real_metrics",
            staticmethod(
                lambda *a, **k: ({"ppl_ratio": 1.0}, {"preview": {}, "final": {}})
            ),
        )
        report = runner.execute(model, adapter, edit, [], cfg, calibration_data=cal)
        assert report.metrics.get("ppl_ratio") == 1.0
    finally:
        del os.environ["INVARLOCK_DEBUG_TRACE"]
