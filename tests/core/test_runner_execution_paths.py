from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from invarlock.core.api import Guard, ModelAdapter, ModelEdit, RunConfig
from invarlock.core.runner import CoreRunner
from invarlock.core.runner_context import collect_cuda_flags


class DummyModel:
    def __init__(self):
        self._restored = False

    def parameters(self):  # minimal iterator with a .device
        class P:
            device = "cpu"

        yield P()

    def eval(self):  # pragma: no cover - trivial
        return None


class DummyAdapter(ModelAdapter):
    name = "dummy"

    def can_handle(self, model: Any) -> bool:  # pragma: no cover - not used
        return True

    def describe(self, model: Any) -> dict[str, Any]:
        return {"n_layer": 1, "heads_per_layer": [1], "mlp_dims": [1], "tying": {}}

    def snapshot(self, model: Any) -> bytes:
        return b"blob"

    def restore(self, model: Any, blob: bytes) -> None:
        model._restored = True


class DummyEdit(ModelEdit):
    def __init__(self, name: str = "test", can: bool = True):
        self.name = name
        self._can = can

    def can_edit(self, model_desc: dict[str, Any]) -> bool:
        return self._can

    def apply(
        self,
        model: Any,
        adapter: ModelAdapter,
        plan=None,
        runtime=None,
    ) -> dict[str, Any]:
        _ = model, adapter, plan, runtime
        return {
            "name": self.name,
            "deltas": {"params_changed": 1, "layers_modified": 0},
        }


class NonDictEdit(ModelEdit):
    name = "non_dict_edit"

    def can_edit(self, model_desc: dict[str, Any]) -> bool:
        return True

    def apply(
        self,
        model: Any,
        adapter: ModelAdapter,
        plan=None,
        runtime=None,
    ) -> Any:
        _ = model, adapter, plan, runtime
        return "ok"  # Non-dict result to exercise fallback context updates


class MissingDeltasEdit(ModelEdit):
    name = "missing_deltas"

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
        return {"name": self.name}  # No 'deltas' key


class NonDictDeltasEdit(ModelEdit):
    name = "non_dict_deltas"

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
        return {"name": self.name, "deltas": 0}


class GoodGuard(Guard):
    name = "good"

    def __init__(self):
        self.config = {}
        self.policy = {}

    def set_run_context(self, report):  # noqa: D401 - simple stub
        # record that context was set
        self.config["context"] = True

    def validate(
        self, model: Any, adapter: ModelAdapter, context: dict[str, Any]
    ) -> dict[str, Any]:
        return {"passed": True}


class BadGuard(Guard):
    name = "bad"

    def __init__(self):
        self.policy = {}

    def set_run_context(self, report):
        raise RuntimeError("context boom")

    def validate(
        self, model: Any, adapter: ModelAdapter, context: dict[str, Any]
    ) -> dict[str, Any]:
        raise RuntimeError("validate boom")


class ErrPrepareGuard(Guard):
    name = "err"

    def set_run_context(self, report):
        return None

    def prepare(self, model, adapter, calib, policy):
        raise RuntimeError("prepare boom")

    def validate(
        self, model: Any, adapter: ModelAdapter, context: dict[str, Any]
    ) -> dict[str, Any]:
        return {"passed": True}


def make_config(tmp_path: Path, **overrides: Any) -> RunConfig:
    ctx = {
        "run_id": "run-xyz",
        "plugins": ["demo"],
        "guards": {"bad": {"threshold": 0.5}},
        "eval": {"loss": {"type": "ce"}},
    }
    cfg = RunConfig(
        device="cpu",
        max_pm_ratio=1.5,
        spike_threshold=2.0,
        event_path=tmp_path / "events.jsonl",
        checkpoint_interval=1,
        context=ctx,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def test_runner_rollback_on_guard_failure(monkeypatch, tmp_path):
    runner = CoreRunner()

    # Apply guard policy via resolver
    def fake_resolver(tier: str, edit_name: str | None, overrides: dict[str, Any]):
        assert tier in {"balanced", "aggressive", "conservative"}
        return {
            "good": {
                "alpha": 0.1
            },  # attribute set on guard.config via _apply_guard_policy
            "bad": {
                "beta": 0.2
            },  # attribute set on guard.policy via _apply_guard_policy
        }

    import invarlock.core.runner as runner_mod

    monkeypatch.setattr(runner_mod, "resolve_tier_policies", fake_resolver)

    class ValidateBoomGuard(Guard):
        name = "bad"

        def __init__(self):
            self.policy = {}

        def set_run_context(self, report):
            self.policy["context"] = True

        def validate(self, model: Any, adapter: ModelAdapter, context: dict[str, Any]):
            raise RuntimeError("validate boom")

    model = DummyModel()
    adapter = DummyAdapter()
    edit = DummyEdit(name="edit-name")
    guards = [ValidateBoomGuard(), GoodGuard()]
    cfg = make_config(tmp_path)

    report = runner.execute(model, adapter, edit, guards, cfg, calibration_data=None)

    assert report.status == "failed"
    assert isinstance(report.error, str) and "validate boom" in report.error
    assert model._restored is True


def test_runner_success_and_catastrophic_spike(monkeypatch, tmp_path):
    runner = CoreRunner()
    model = DummyModel()
    adapter = DummyAdapter()
    edit = DummyEdit()
    guards = [GoodGuard()]
    cfg = make_config(tmp_path)

    # Force eval metrics to a successful ratio
    def fake_eval_success(model, adapter, calib, report, preview_n, final_n, config):
        # CoreRunner._eval_phase normally returns only metrics dict; keep same shape here
        return {"primary_metric": {"kind": "ppl_causal", "preview": 1.0, "final": 1.0}}

    # And a catastrophic spike case
    def fake_eval_spike(model, adapter, calib, report, preview_n, final_n, config):
        return {"primary_metric": {"kind": "ppl_causal", "preview": 1.0, "final": 5.0}}

    monkeypatch.setattr(CoreRunner, "_eval_phase", staticmethod(fake_eval_success))
    report_ok = runner.execute(model, adapter, edit, guards, cfg, calibration_data=None)
    assert report_ok.status == "success"

    # Swap in the spike behavior
    monkeypatch.setattr(CoreRunner, "_eval_phase", staticmethod(fake_eval_spike))
    report_spike = runner.execute(
        model, adapter, edit, guards, cfg, calibration_data=None
    )
    assert report_spike.status == "rollback"
    assert "catastrophic" in report_spike.meta.get("rollback_reason", "")


def test_runner_edit_cannot_apply_sets_failed(tmp_path):
    runner = CoreRunner()
    model = DummyModel()
    adapter = DummyAdapter()
    edit = DummyEdit(can=False)
    guards = []
    cfg = make_config(tmp_path)

    report = runner.execute(model, adapter, edit, guards, cfg, calibration_data=None)
    assert report.status == "failed"
    assert isinstance(report.error, str) and "cannot be applied" in report.error


def test_runner_records_timing_and_guard_timings(tmp_path):
    runner = CoreRunner()
    model = DummyModel()
    adapter = DummyAdapter()
    edit = DummyEdit()
    guards = [GoodGuard()]
    cfg = make_config(tmp_path, checkpoint_interval=0)

    report = runner.execute(model, adapter, edit, guards, cfg, calibration_data=None)
    timings = report.metrics.get("timings", {})
    assert isinstance(timings, dict)
    for key in (
        "prepare",
        "prepare_guards",
        "edit",
        "guards",
        "eval",
        "finalize",
        "total",
    ):
        assert key in timings

    guard_timings = report.metrics.get("guard_timings", {})
    assert isinstance(guard_timings, dict)
    assert guard_timings.get("good") is not None


def test_runner_merges_memory_snapshots_into_metrics(monkeypatch, tmp_path):
    import invarlock.core.runner as runner_mod

    runner = CoreRunner()
    model = DummyModel()
    adapter = DummyAdapter()
    edit = DummyEdit()
    cfg = make_config(tmp_path, checkpoint_interval=0)

    monkeypatch.setattr(
        runner_mod,
        "capture_memory_snapshot",
        lambda phase: {"phase": phase, "rss_mb": 1.0},
    )
    monkeypatch.setattr(
        runner_mod,
        "summarize_memory_snapshots",
        lambda _snaps: {"memory_mb_peak": 0.5},
    )

    report = runner.execute(model, adapter, edit, [], cfg, calibration_data=None)
    assert report.metrics.get("memory_snapshots")
    assert report.metrics.get("memory_mb_peak") is not None


def test_runner_skips_empty_memory_snapshots(monkeypatch, tmp_path):
    import invarlock.core.runner as runner_mod

    runner = CoreRunner()
    model = DummyModel()
    adapter = DummyAdapter()
    edit = DummyEdit()
    cfg = make_config(tmp_path, checkpoint_interval=0)

    monkeypatch.setattr(runner_mod, "capture_memory_snapshot", lambda phase: {})
    monkeypatch.setattr(runner_mod, "summarize_memory_snapshots", lambda _snaps: {})

    report = runner.execute(model, adapter, edit, [], cfg, calibration_data=None)
    assert "memory_snapshots" not in report.metrics


def test_runner_memory_snapshot_summary_empty(monkeypatch, tmp_path):
    import invarlock.core.runner as runner_mod

    called = {"summary": 0}

    runner = CoreRunner()
    model = DummyModel()
    adapter = DummyAdapter()
    edit = DummyEdit()
    cfg = make_config(tmp_path, checkpoint_interval=0)

    monkeypatch.setattr(
        runner_mod,
        "capture_memory_snapshot",
        lambda phase: {"phase": phase, "rss_mb": 1.0},
    )
    monkeypatch.setattr(
        runner_mod,
        "summarize_memory_snapshots",
        lambda _snaps: called.__setitem__("summary", called["summary"] + 1) or {},
    )

    report = runner.execute(model, adapter, edit, [], cfg, calibration_data=None)
    assert report.metrics.get("memory_snapshots")
    assert called["summary"] == 1


def test_runner_eval_phase_with_calibration_uses_compute(monkeypatch, tmp_path):
    runner = CoreRunner()
    model = DummyModel()
    adapter = DummyAdapter()
    edit = DummyEdit()
    guards = [GoodGuard()]
    cfg = make_config(tmp_path)

    # Patch compute_real_metrics to avoid heavy torch usage and still exercise branch
    def fake_compute(model, calibration_data, adapter, preview_n, final_n, config):
        return (
            {
                "primary_metric": {"kind": "ppl_causal", "preview": 1.0, "final": 1.1},
                "latency_ms_per_tok": 0.0,
                "memory_mb_peak": 0.0,
            },
            {"preview": {"window_ids": [0]}, "final": {"window_ids": [0]}},
        )

    monkeypatch.setattr(CoreRunner, "_compute_real_metrics", staticmethod(fake_compute))

    # Minimal calibration data sample
    calibration = [
        {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": [1, 2, 3]},
        {"input_ids": [4, 5, 6], "attention_mask": [1, 1, 1], "labels": [4, 5, 6]},
    ]

    report = runner.execute(
        model, adapter, edit, guards, cfg, calibration_data=calibration
    )
    assert report.status in {"success", "rollback"}  # depends on ppl thresholds
    assert "evaluation_windows" in report.__dict__
    assert isinstance(report.evaluation_windows.get("preview"), dict)


def test_eval_phase_debug_snapshot(monkeypatch, tmp_path):
    import os

    runner = CoreRunner()
    model = DummyModel()
    adapter = DummyAdapter()
    edit = DummyEdit()
    guards = [GoodGuard()]
    cfg = make_config(tmp_path)
    # Provide small calibration data and patch compute to avoid heavy logic
    cal = [{"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}]
    os.environ["INVARLOCK_DEBUG_TRACE"] = "1"
    try:
        monkeypatch.setattr(
            CoreRunner,
            "_compute_real_metrics",
            staticmethod(
                lambda *a, **k: (
                    {
                        "primary_metric": {
                            "kind": "ppl_causal",
                            "preview": 1.0,
                            "final": 1.0,
                        }
                    },
                    {"preview": {}, "final": {}},
                )
            ),
        )
        report = runner.execute(model, adapter, edit, guards, cfg, calibration_data=cal)
        pm = report.metrics.get("primary_metric", {})
        assert pm.get("final") == 1.0 and pm.get("preview") == 1.0
    finally:
        del os.environ["INVARLOCK_DEBUG_TRACE"]


def test_compute_real_metrics_config_none(tmp_path):
    import torch

    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(3, 3, bias=False)

        def forward(self, *a, **k):
            class Obj:
                def __init__(self):
                    self.loss = type("L", (), {"item": lambda self: 1.0})()

            return Obj()

    runner = CoreRunner()
    model = Toy()
    adapter = DummyAdapter()
    metrics, _ = runner._compute_real_metrics(
        model,
        [
            {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]},
            {"input_ids": [4, 5, 6], "attention_mask": [1, 1, 1]},
        ],
        adapter,
        preview_n=1,
        final_n=1,
        config=None,
    )
    pm = metrics.get("primary_metric", {})
    assert isinstance(pm, dict) and pm.get("final") and pm.get("preview")


def test_bootstrap_replicates_disable(tmp_path):
    import torch

    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(3, 3, bias=False)

        def forward(self, *a, **k):
            class Obj:
                def __init__(self):
                    self.loss = type("L", (), {"item": lambda self: 1.0})()

            return Obj()

    runner = CoreRunner()
    model = Toy()
    adapter = DummyAdapter()
    cfg = make_config(tmp_path)
    cfg.context.setdefault("eval", {}).update(
        {"bootstrap": {"enabled": True, "replicates": 0}}
    )
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
    assert isinstance(pm, dict) and pm.get("final") and pm.get("preview")


def test_bootstrap_unknown_method_disabled(tmp_path):
    import torch

    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(3, 3, bias=False)

        def forward(self, *a, **k):
            class Obj:
                def __init__(self):
                    self.loss = type("L", (), {"item": lambda self: 1.0})()

            return Obj()

    runner = CoreRunner()
    model = Toy()
    adapter = DummyAdapter()
    cfg = make_config(tmp_path)
    cfg.context.setdefault("eval", {}).update(
        {"bootstrap": {"enabled": True, "method": "weird", "replicates": 0}}
    )
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
    assert isinstance(pm, dict) and pm.get("final") and pm.get("preview")


def test_preview_zero_final_positive_path(tmp_path):
    import torch

    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(3, 3, bias=False)

        def forward(self, *a, **k):
            class Obj:
                def __init__(self):
                    self.loss = type("L", (), {"item": lambda self: 1.0})()

            return Obj()

    runner = CoreRunner()
    model = Toy()
    adapter = DummyAdapter()
    cfg = make_config(tmp_path)
    metrics, _ = runner._compute_real_metrics(
        model,
        [
            {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]},
            {"input_ids": [4, 5, 6], "attention_mask": [1, 1, 1]},
        ],
        adapter,
        preview_n=0,
        final_n=1,
        config=cfg,
    )
    pm = metrics.get("primary_metric", {})
    assert pm.get("final") > 0


def test_bootstrap_dataset_seed_str(tmp_path):
    import torch

    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(3, 3, bias=False)

        def forward(self, *a, **k):
            class Obj:
                def __init__(self):
                    self.loss = type("L", (), {"item": lambda self: 1.0})()

            return Obj()

    runner = CoreRunner()
    model = Toy()
    adapter = DummyAdapter()
    cfg = make_config(tmp_path)
    cfg.context.setdefault("eval", {}).update(
        {"bootstrap": {"enabled": True, "method": "percentile", "replicates": 2}}
    )
    cfg.context.setdefault("dataset", {})["seed"] = (
        "abc"  # triggers fallback seed parsing
    )
    metrics, _ = runner._compute_real_metrics(
        model,
        [{"input_ids": [1, 2, 3]}, {"input_ids": [4, 5, 6]}],
        adapter,
        preview_n=1,
        final_n=1,
        config=cfg,
    )
    pm = metrics.get("primary_metric", {})
    assert pm.get("final") and pm.get("preview")


def test_measure_latency_tensor_inputs_variants():
    import torch

    runner = CoreRunner()

    class M:
        def __call__(self, *a, **k):
            class Obj:
                def __init__(self):
                    self.loss = type("L", (), {"item": lambda self: 1.0})()

            return Obj()

    # 1-D tensor → unsqueeze path
    s1 = {"input_ids": torch.tensor([1, 2, 3])}
    # 2-D tensor → no unsqueeze
    s2 = {"input_ids": torch.tensor([[1, 2, 3]])}
    assert (
        runner._measure_latency(M(), [s1], "cpu") == 0.0
        or runner._measure_latency(M(), [s1], "cpu") > 0.0
    )
    assert (
        runner._measure_latency(M(), [s2], "cpu") == 0.0
        or runner._measure_latency(M(), [s2], "cpu") > 0.0
    )


def test_measure_latency_device_to_exception():
    runner = CoreRunner()

    class M:
        def __call__(self, *a, **k):
            class Obj:
                def __init__(self):
                    self.loss = type("L", (), {"item": lambda self: 1.0})()

            return Obj()

    # Nonstandard device object forces .to(device) to raise
    bad_device = object()
    sample = {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}
    with pytest.raises(
        RuntimeError, match="Latency measurement device transfer failed"
    ):
        runner._measure_latency(M(), [sample], bad_device)


def test_runner_resolve_policies_error(monkeypatch, tmp_path):
    # Resolver failures should now surface instead of being normalized away.
    from invarlock.core import runner as runner_mod

    monkeypatch.setattr(
        runner_mod,
        "resolve_tier_policies",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("oops")),
    )
    from invarlock.core.api import RunReport

    r = CoreRunner()
    model = DummyModel()
    adapter = DummyAdapter()
    guards = [GoodGuard()]
    cfg = make_config(tmp_path)
    # Call prepare guards phase directly to isolate the branch
    r._initialize_services(cfg)
    with pytest.raises(RuntimeError, match="oops"):
        r._prepare_guards_phase(
            model,
            adapter,
            guards,
            calibration_data=None,
            report=RunReport(),
            auto_config=None,
            config=cfg,
        )
    r._cleanup_services()


def test_runner_compute_real_metrics_smoke(tmp_path):
    # Execute the real metrics path with a tiny fake model and samples
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
            self.lin = torch.nn.Linear(4, 4, bias=False)

        def forward(
            self, input_ids=None, attention_mask=None, labels=None, token_type_ids=None
        ):
            # Return a constant-ish loss regardless of inputs to keep CI light
            return FakeOutputs(1.0)

    runner = CoreRunner()
    model = ToyModel()
    adapter = DummyAdapter()
    cfg = make_config(tmp_path)
    # Disable bootstrap to keep execution fast
    cfg.context.setdefault("eval", {}).update({"bootstrap": {"enabled": False}})

    # Two tiny samples → preview_n=1, final_n=1
    calibration = [
        {"input_ids": [1, 2, 3, 4], "attention_mask": [1, 1, 1, 1]},
        {"input_ids": [5, 6, 7, 8], "attention_mask": [1, 1, 1, 1]},
    ]

    metrics, windows = runner._compute_real_metrics(
        model, calibration, adapter, preview_n=None, final_n=None, config=cfg
    )
    pm = metrics.get("primary_metric", {})
    assert (pm.get("final") / pm.get("preview")) == pytest.approx(1.0, rel=1e-3)
    assert set(windows.keys()) == {"preview", "final"}


def test_runner_eval_fallback_no_calibration(tmp_path):
    runner = CoreRunner()
    model = DummyModel()
    adapter = DummyAdapter()
    edit = DummyEdit()
    guards = [GoodGuard()]
    cfg = make_config(tmp_path)

    # No calibration_data should produce an explicit non-evaluated state.
    report = runner.execute(model, adapter, edit, guards, cfg, calibration_data=None)
    assert report.status == "success"
    assert report.metrics["eval_state"] == {
        "evaluated": False,
        "reason": "missing_calibration_data",
    }
    assert "primary_metric" not in report.metrics


def test_execute_with_auto_config_passthrough(tmp_path, monkeypatch):
    runner = CoreRunner()
    model = DummyModel()
    adapter = DummyAdapter()
    edit = DummyEdit()
    guards = [GoodGuard()]
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
        model,
        adapter,
        edit,
        guards,
        cfg,
        calibration_data=None,
        auto_config={"tier": "aggressive", "enabled": True},
    )
    assert isinstance(report.meta.get("auto"), dict)


def test_collect_cuda_flags_env_toggle(monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    flags = collect_cuda_flags()
    assert "CUBLAS_WORKSPACE_CONFIG" in flags


def test_prepare_guards_prepare_error(monkeypatch, tmp_path):
    # ErrPrepareGuard.prepare raises; ensure no crash in prepare phase
    runner = CoreRunner()
    model = DummyModel()
    adapter = DummyAdapter()
    cfg = make_config(tmp_path)
    # Patch eval to avoid heavy compute
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
        model, adapter, DummyEdit(), [ErrPrepareGuard()], cfg, calibration_data=None
    )
    assert report.status == "failed"


def test_latency_with_token_type_ids_and_success(tmp_path):
    runner = CoreRunner()
    sample = {
        "input_ids": [1, 2, 3],
        "attention_mask": [1, 1, 1],
        "token_type_ids": [0, 0, 0],
    }

    class SimpleModel:
        def __call__(
            self, input_ids=None, attention_mask=None, labels=None, token_type_ids=None
        ):
            class Out:
                def __init__(self):
                    self.loss = type("L", (), {"item": lambda self: 1.0})()

            return Out()

    ms = runner._measure_latency(SimpleModel(), [sample], "cpu")
    assert ms == 0.0 or ms > 0.0


def test_measure_latency_non_dict_sample():
    runner = CoreRunner()

    class M:
        def __call__(self, *a, **k):
            class Obj:
                def __init__(self):
                    self.loss = type("L", (), {"item": lambda self: 1.0})()

            return Obj()

    # Non-dict sample path
    ms = runner._measure_latency(M(), [[1, 2, 3]], "cpu")
    assert ms == 0.0 or ms > 0.0


def test_samples_to_dataloader_paths():
    runner = CoreRunner()
    samples = [
        {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]},  # labels None path
        {
            "input_ids": [4, 5, 6],
            "attention_mask": [1, 1, 1],
            "labels": [4, 5, 6],
            "token_type_ids": [0, 0, 0],
        },
        {"input_ids": [7, 8, 9]},  # no attention mask branch
        {"input_ids": None},  # skip branch
    ]
    dl = runner._samples_to_dataloader(samples)
    batches = list(iter(dl))
    assert len(batches) == 3
    assert set(batches[0].keys()) >= {"input_ids", "labels"}
    assert set(batches[1].keys()) >= {"input_ids", "labels", "token_type_ids"}
    assert set(batches[2].keys()) >= {"input_ids", "labels"}
