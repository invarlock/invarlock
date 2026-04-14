from __future__ import annotations

from typing import Any

import pytest

from invarlock.core.api import Guard
from invarlock.core.runner import CoreRunner
from tests.core.test_runner_execution_metrics_and_memory import (
    DummyAdapter,
    DummyEdit,
    DummyModel,
    GoodGuard,
    MissingDeltasEdit,
    NonDictDeltasEdit,
    NonDictEdit,
    make_config,
)

# Fallback model layers metric via .get("n_layer", 0) covered by execution


def test_edit_phase_with_non_dict_report_context():
    # Call _edit_phase directly with report.context not a dict to hit fallback
    runner = CoreRunner()
    model = DummyModel()
    adapter = DummyAdapter()
    edit = DummyEdit()
    from invarlock.core.api import RunReport

    report = RunReport()
    report.context = []
    res = runner._edit_phase(
        model,
        adapter,
        edit,
        {},
        report,
        edit_config=None,
        edit_runtime=None,
    )
    assert isinstance(res, dict) and isinstance(report.context, dict)


def test_initialize_services_without_event_logger(tmp_path):
    runner = CoreRunner()
    model = DummyModel()
    adapter = DummyAdapter()
    edit = DummyEdit()
    guards = [GoodGuard()]
    cfg = make_config(tmp_path)
    cfg.event_path = None
    # Patch eval to avoid heavy compute
    from invarlock.core.runner import CoreRunner as CR

    old = CR._eval_phase
    try:
        CR._eval_phase = staticmethod(
            lambda *a, **k: {
                "primary_metric": {"kind": "ppl_causal", "preview": 1.0, "final": 1.0}
            }
        )
        report = runner.execute(
            model, adapter, edit, guards, cfg, calibration_data=None
        )
        assert report.status in {"success", "rollback"}
    finally:
        CR._eval_phase = old


def test_resolve_guard_policies_from_meta(monkeypatch):
    from invarlock.core.api import RunReport

    runner = CoreRunner()

    # Fake resolver to observe tier and edit name propagation
    seen = {}

    def fake_resolver(tier, edit_name, overrides):
        seen["tier"] = tier
        seen["edit"] = edit_name
        return {"good": {"alpha": 0.1}}

    import invarlock.core.runner as runner_mod

    monkeypatch.setattr(runner_mod, "resolve_tier_policies", fake_resolver)

    report = RunReport()
    report.edit = {"name": "editX"}
    report.meta["config"] = {"guards": {}}
    policies = runner._resolve_guard_policies(report, auto_config=None)
    assert policies and seen.get("tier") == "balanced" and seen.get("edit") == "editX"


def test_apply_guard_policy_exception_path(monkeypatch):
    class WeirdGuard(GoodGuard):
        name = "weird"

        def __init__(self):
            super().__init__()
            object.__setattr__(self, "oops", 0)

        def __setattr__(self, name, value):
            if name == "oops":
                raise RuntimeError("nope")
            return super().__setattr__(name, value)

    runner = CoreRunner()
    g = WeirdGuard()
    g.config = {}
    g.policy = {}
    with pytest.raises(RuntimeError, match="nope"):
        runner._apply_guard_policy(g, {"oops": 1, "cfg": 2})


def test_prepare_phase_no_checkpoint_and_missing_nlayer(tmp_path):
    # Adapter without n_layer in describe
    class BareAdapter(DummyAdapter):
        def describe(self, model: Any) -> dict[str, Any]:
            return {"heads_per_layer": [], "mlp_dims": [], "tying": {}}

    runner = CoreRunner()
    model = DummyModel()
    adapter = BareAdapter()
    edit = DummyEdit()
    guards = [GoodGuard()]
    cfg = make_config(tmp_path, checkpoint_interval=0)

    report = runner.execute(model, adapter, edit, guards, cfg, calibration_data=None)
    # No checkpoint captured
    assert "initial_checkpoint" not in report.meta


def test_edit_phase_with_config_and_nondict_result(monkeypatch, tmp_path):
    runner = CoreRunner()
    model = DummyModel()
    adapter = DummyAdapter()
    edit = NonDictEdit()
    guards = [GoodGuard()]
    cfg = make_config(tmp_path)
    # Patch eval to avoid heavy compute; return acceptable metrics
    monkeypatch.setattr(
        CoreRunner,
        "_eval_phase",
        staticmethod(
            lambda *a, **k: {
                "primary_metric": {"kind": "ppl_causal", "preview": 1.0, "final": 1.0}
            }
        ),
    )

    report = runner.execute(
        model,
        adapter,
        edit,
        guards,
        cfg,
        calibration_data=None,
        edit_config={"scale": 0.9},
    )
    assert report.status in {"success", "rollback"}


def test_edit_result_missing_deltas(monkeypatch, tmp_path):
    runner = CoreRunner()
    model = DummyModel()
    adapter = DummyAdapter()
    edit = MissingDeltasEdit()
    guards = [GoodGuard()]
    cfg = make_config(tmp_path)
    monkeypatch.setattr(
        CoreRunner,
        "_eval_phase",
        staticmethod(
            lambda *a, **k: {
                "primary_metric": {"kind": "ppl_causal", "preview": 1.0, "final": 1.0}
            }
        ),
    )
    report = runner.execute(model, adapter, edit, guards, cfg, calibration_data=None)
    assert report.status in {"success", "rollback"}


def test_edit_result_non_dict_deltas(monkeypatch, tmp_path):
    runner = CoreRunner()
    model = DummyModel()
    adapter = DummyAdapter()
    edit = NonDictDeltasEdit()
    guards = [GoodGuard()]
    cfg = make_config(tmp_path)
    monkeypatch.setattr(
        CoreRunner,
        "_eval_phase",
        staticmethod(
            lambda *a, **k: {
                "primary_metric": {"kind": "ppl_causal", "preview": 1.0, "final": 1.0}
            }
        ),
    )
    report = runner.execute(model, adapter, edit, guards, cfg, calibration_data=None)
    assert report.status in {"success", "rollback"}


def test_prepare_guard_success_complete(monkeypatch, tmp_path):
    class PrepGuard(Guard):
        name = "prep_ok"

        def prepare(self, model, adapter, calib, policy):
            return {"ready": True}

        def validate(self, model, adapter, context):
            return {"passed": True}

    def fake_resolver(tier, edit_name, overrides):
        return {"prep_ok": {}}

    import invarlock.core.runner as runner_mod

    monkeypatch.setattr(runner_mod, "resolve_tier_policies", fake_resolver)

    runner = CoreRunner()
    model = DummyModel()
    adapter = DummyAdapter()
    edit = DummyEdit()
    cfg = make_config(tmp_path)
    # Patch eval to be trivial
    monkeypatch.setattr(
        CoreRunner,
        "_eval_phase",
        staticmethod(
            lambda *a, **k: {
                "primary_metric": {"kind": "ppl_causal", "preview": 1.0, "final": 1.0}
            }
        ),
    )
    report = runner.execute(
        model, adapter, edit, [PrepGuard()], cfg, calibration_data=None
    )
    assert report.status in {"success", "rollback"}


def test_execute_with_empty_context(monkeypatch, tmp_path):
    runner = CoreRunner()
    model = DummyModel()
    adapter = DummyAdapter()
    edit = DummyEdit()
    cfg = make_config(tmp_path)
    cfg.context = {}
    monkeypatch.setattr(
        CoreRunner, "_eval_phase", staticmethod(lambda *a, **k: {"ppl_ratio": 1.0})
    )
    report = runner.execute(model, adapter, edit, [], cfg, calibration_data=None)
    assert report.status in {"success", "rollback"}


def test_compute_metrics_data_scaled_and_shortages(tmp_path):
    # Few samples; ask for more to trigger data_scaled, window_shortage, and final_window_shortage
    import torch

    class FakeLoss:
        def __init__(self, v: float):
            self._v = float(v)

        def item(self) -> float:
            return self._v

    class FakeOutputs:
        def __init__(self, loss: float):
            self.loss = FakeLoss(loss)

    class ToyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(3, 3, bias=False)

        def forward(
            self, input_ids=None, attention_mask=None, labels=None, token_type_ids=None
        ):
            return FakeOutputs(1.0)

    runner = CoreRunner()
    model = ToyModel()
    adapter = DummyAdapter()
    cfg = make_config(tmp_path)
    cfg.context.setdefault("eval", {}).update(
        {
            "bootstrap": {
                "enabled": True,
                "method": "percentile",
                "replicates": 1,
                "alpha": 1.5,
            }
        }
    )

    calibration = [
        {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]},
        {"input_ids": [4, 5, 6], "attention_mask": [1, 1, 1]},
        {"input_ids": [7, 8, 9], "attention_mask": [1, 1, 1]},
    ]

    metrics, windows = runner._compute_real_metrics(
        model, calibration, adapter, preview_n=5, final_n=5, config=cfg
    )
    pm = metrics.get("primary_metric", {})
    assert pm.get("preview") > 0 and pm.get("final") > 0
    assert set(windows.keys()) == {"preview", "final"}


def test_degenerate_single_pair_and_no_variation(tmp_path):
    import torch

    class FakeLoss:
        def __init__(self, v: float):
            self._v = float(v)

        def item(self) -> float:
            return self._v

    class FakeOutputs:
        def __init__(self, loss: float):
            self.loss = FakeLoss(loss)

    class ToyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(3, 3, bias=False)

        def forward(self, *_, **__):
            return FakeOutputs(1.0)

    runner = CoreRunner()
    model = ToyModel()
    adapter = DummyAdapter()
    cfg = make_config(tmp_path)
    cfg.context.setdefault("eval", {}).update(
        {"bootstrap": {"enabled": True, "replicates": 2}}
    )

    # Single pair case
    calibration = [
        {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]},
        {"input_ids": [4, 5, 6], "attention_mask": [1, 1, 1]},
    ]
    metrics1, _ = runner._compute_real_metrics(
        model, calibration, adapter, preview_n=1, final_n=1, config=cfg
    )
    assert metrics1["paired_windows"] == 1

    # No variation case: two pairs with identical deltas
    calibration2 = [
        {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]},
        {"input_ids": [4, 5, 6], "attention_mask": [1, 1, 1]},
        {"input_ids": [7, 8, 9], "attention_mask": [1, 1, 1]},
        {"input_ids": [10, 11, 12], "attention_mask": [1, 1, 1]},
    ]
    metrics2, _ = runner._compute_real_metrics(
        model, calibration2, adapter, preview_n=2, final_n=2, config=cfg
    )
    assert metrics2["paired_windows"] == 2


def test_store_eval_windows_disabled(tmp_path, monkeypatch):
    import torch

    class FakeLoss:
        def __init__(self, v: float):
            self._v = float(v)

        def item(self) -> float:
            return self._v

    class FakeOutputs:
        def __init__(self, loss: float):
            self.loss = FakeLoss(loss)

    class ToyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(3, 3, bias=False)

        def forward(self, *_, **__):
            return FakeOutputs(1.0)

    runner = CoreRunner()
    model = ToyModel()
    adapter = DummyAdapter()
    cfg = make_config(tmp_path)
    monkeypatch.setenv("INVARLOCK_STORE_EVAL_WINDOWS", "0")
    metrics, windows = runner._compute_real_metrics(
        model,
        [
            {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]},
            {"input_ids": [4, 5, 6], "attention_mask": [1, 1, 1]},
        ],
        adapter,
        preview_n=1,
        final_n=1,
        config=cfg,
    )
    pm = metrics.get("primary_metric", {})
    assert (pm.get("final") and pm.get("preview")) and windows["preview"].get(
        "input_ids"
    ) == []


def test_zero_mask_batch_warning(tmp_path, monkeypatch):
    import os

    import torch

    class FakeLoss:
        def __init__(self, v: float):
            self._v = float(v)

        def item(self) -> float:
            return self._v

    class FakeOutputs:
        def __init__(self, loss: float):
            self.loss = FakeLoss(loss)

    class ToyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(3, 3, bias=False)

        def forward(self, *_, **__):
            return FakeOutputs(1.0)

    runner = CoreRunner()
    model = ToyModel()
    adapter = DummyAdapter()
    cfg = make_config(tmp_path)
    cfg.context.setdefault("eval", {}).update({"loss": {"type": "mlm"}})
    os.environ["INVARLOCK_DEBUG_TRACE"] = "1"
    try:
        # Provide labels with all -100 to trigger zero_mask_batch path
        calibration = [
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
            model, calibration, adapter, preview_n=1, final_n=1, config=cfg
        )
        pm = metrics.get("primary_metric", {})
        assert pm.get("final") and pm.get("preview")
    finally:
        del os.environ["INVARLOCK_DEBUG_TRACE"]


def test_eval_device_override_no_move(tmp_path, monkeypatch):
    # Setting eval.device_override equal to current device should not move model

    import torch

    class T(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.p = torch.nn.Parameter(torch.zeros(1))

        def forward(self, *a, **k):
            class Obj:
                def __init__(self):
                    self.loss = type("L", (), {"item": lambda self: 1.0})()

            return Obj()

    runner = CoreRunner()
    model = T()
    adapter = DummyAdapter()
    cfg = make_config(tmp_path)
    cfg.context.setdefault("eval", {})["device_override"] = "cpu"
    metrics, _ = runner._compute_real_metrics(
        model,
        [
            {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]},
            {"input_ids": [4, 5, 6], "attention_mask": [1, 1, 1]},
        ],
        adapter,
        preview_n=1,
        final_n=1,
        config=cfg,
    )
    pm = metrics.get("primary_metric", {})
    assert pm.get("final") and pm.get("preview")


def test_compute_metrics_preview_final_zero_and_empty(tmp_path):
    runner = CoreRunner()
    model = DummyModel()
    adapter = DummyAdapter()
    cfg = make_config(tmp_path)

    # Empty calibration raises
    with pytest.raises(ValueError):
        runner._compute_real_metrics(
            model, [], adapter, preview_n=None, final_n=None, config=cfg
        )

    # Both zero raises
    with pytest.raises(ValueError):
        runner._compute_real_metrics(
            model,
            [{"input_ids": [1, 2], "attention_mask": [1, 1]}],
            adapter,
            preview_n=0,
            final_n=0,
            config=cfg,
        )


def test_compute_slice_missing_loss_debug(monkeypatch, tmp_path):
    import os

    import torch

    class NoLossOutputs:
        pass

    class NoLossModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(3, 3, bias=False)

        def forward(
            self, input_ids=None, attention_mask=None, labels=None, token_type_ids=None
        ):
            return NoLossOutputs()

    runner = CoreRunner()
    model = NoLossModel()
    adapter = DummyAdapter()
    cfg = make_config(tmp_path)

    os.environ["INVARLOCK_DEBUG_TRACE"] = "1"
    try:
        metrics, _ = runner._compute_real_metrics(
            model,
            [{"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}],
            adapter,
            preview_n=1,
            final_n=1,
            config=cfg,
        )
        pm = metrics.get("primary_metric", {})
        assert pm.get("invalid") is True
        assert pm.get("degraded_reason") == "non_finite_pm"
        assert pm.get("preview") is None
        assert pm.get("final") is None
    finally:
        del os.environ["INVARLOCK_DEBUG_TRACE"]


def test_mlm_path_no_crash_on_zero_masks(tmp_path):
    import torch

    class FakeLoss:
        def __init__(self, v: float):
            self._v = float(v)

        def item(self) -> float:
            return self._v

    class FakeOutputs:
        def __init__(self, loss: float):
            self.loss = FakeLoss(loss)

    class ToyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(3, 3, bias=False)

        def forward(
            self, input_ids=None, attention_mask=None, labels=None, token_type_ids=None
        ):
            return FakeOutputs(1.0)

    runner = CoreRunner()
    model = ToyModel()
    adapter = DummyAdapter()
    cfg = make_config(tmp_path)
    cfg.context.setdefault("eval", {}).update({"loss": {"type": "mlm"}})

    # Use non-zero masks to exercise MLM path without raising
    calibration = [
        {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": [1, 2, 3]},
        {"input_ids": [4, 5, 6], "attention_mask": [1, 1, 1], "labels": [4, 5, 6]},
    ]
    metrics, _ = runner._compute_real_metrics(
        model, calibration, adapter, preview_n=1, final_n=1, config=cfg
    )
    pm = metrics.get("primary_metric", {})
    assert pm.get("final") and pm.get("preview")


def test_pairing_mismatch_and_overlap_logging(tmp_path):
    import torch

    class FakeLoss:
        def __init__(self, v: float):
            self._v = float(v)

        def item(self) -> float:
            return self._v

    class FakeOutputs:
        def __init__(self, loss: float):
            self.loss = FakeLoss(loss)

    class ToyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(3, 3, bias=False)

        def forward(
            self, input_ids=None, attention_mask=None, labels=None, token_type_ids=None
        ):
            return FakeOutputs(1.0)

    runner = CoreRunner()
    model = ToyModel()
    adapter = DummyAdapter()
    cfg = make_config(tmp_path)
    # Provide baseline with expected IDs and tokens that do not match run tokens
    cfg.context.update(
        {
            "pairing_baseline": {
                "preview": {"window_ids": [0, 1], "input_ids": [[9, 9, 9], [8, 8, 8]]},
                "final": {"window_ids": [1, 2], "input_ids": [[7, 7, 7], [6, 6, 6]]},
            }
        }
    )

    # Duplicate tokens across preview and final to produce overlap warning as well
    calibration = [
        {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]},
        {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]},
        {"input_ids": [4, 5, 6], "attention_mask": [1, 1, 1]},
        {"input_ids": [4, 5, 6], "attention_mask": [1, 1, 1]},
    ]
    metrics, windows = runner._compute_real_metrics(
        model, calibration, adapter, preview_n=2, final_n=2, config=cfg
    )
    assert metrics["window_overlap_fraction"] >= 0.25


def test_measure_latency_paths(tmp_path):
    runner = CoreRunner()
    # Empty sample_data
    assert runner._measure_latency(object(), [], "cpu") == 0.0
    # None sample
    assert runner._measure_latency(object(), [None], "cpu") == 0.0
    # Missing input_ids
    assert runner._measure_latency(object(), [{"attention_mask": [1, 1]}], "cpu") == 0.0

    # Model that raises to exercise try/except
    class RaisingModel:
        def __call__(self, *a, **k):
            raise RuntimeError("boom")

    sample = {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}
    with pytest.raises(RuntimeError, match="boom"):
        runner._measure_latency(RaisingModel(), [sample], "cpu")


def test_finalize_metrics_unacceptable_no_checkpoint(monkeypatch, tmp_path):
    runner = CoreRunner()
    model = DummyModel()
    adapter = DummyAdapter()
    edit = DummyEdit()
    guards = [GoodGuard()]
    cfg = make_config(tmp_path, checkpoint_interval=0)

    # Return unacceptable metrics (above max ratio but below spike threshold)
    monkeypatch.setattr(
        CoreRunner,
        "_eval_phase",
        staticmethod(
            lambda *a, **k: {
                "primary_metric": {"kind": "ppl_causal", "preview": 1.0, "final": 2.0}
            }
        ),
    )
    report = runner.execute(model, adapter, edit, guards, cfg, calibration_data=None)
    assert report.status == "rollback"
    assert "rollback_checkpoint" not in report.meta
