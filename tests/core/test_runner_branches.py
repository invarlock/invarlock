from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from invarlock.core.api import Guard, ModelAdapter, ModelEdit, RunConfig
from invarlock.core.runner import CoreRunner, _collect_cuda_flags


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

    def apply(self, model: Any, adapter: ModelAdapter, **kwargs) -> dict[str, Any]:
        return {
            "name": self.name,
            "deltas": {"params_changed": 1, "layers_modified": 0},
        }


class NonDictEdit(ModelEdit):
    name = "non_dict_edit"

    def can_edit(self, model_desc: dict[str, Any]) -> bool:
        return True

    def apply(self, model: Any, adapter: ModelAdapter, **kwargs) -> Any:
        return "ok"  # Non-dict result to exercise fallback context updates


class MissingDeltasEdit(ModelEdit):
    name = "missing_deltas"

    def can_edit(self, model_desc: dict[str, Any]) -> bool:
        return True

    def apply(self, model: Any, adapter: ModelAdapter, **kwargs) -> dict[str, Any]:
        return {"name": self.name}  # No 'deltas' key


class NonDictDeltasEdit(ModelEdit):
    name = "non_dict_deltas"

    def can_edit(self, model_desc: dict[str, Any]) -> bool:
        return True

    def apply(self, model: Any, adapter: ModelAdapter, **kwargs) -> dict[str, Any]:
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

    model = DummyModel()
    adapter = DummyAdapter()
    edit = DummyEdit(name="edit-name")
    guards = [BadGuard(), GoodGuard()]
    cfg = make_config(tmp_path)

    report = runner.execute(model, adapter, edit, guards, cfg, calibration_data=None)

    assert report.status == "rollback"
    # Check that rollback metadata was set
    assert "rollback_checkpoint" in report.meta
    assert "rollback_reason" in report.meta
    # Adapter.restore should have been invoked by rollback path
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
    ms = runner._measure_latency(M(), [sample], bad_device)
    assert ms == 0.0 or ms > 0.0


def test_runner_resolve_policies_error(monkeypatch, tmp_path):
    # Ensure the error path in _resolve_guard_policies returns {} and does not crash
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
    try:
        r._prepare_guards_phase(
            model,
            adapter,
            guards,
            calibration_data=None,
            report=RunReport(),
            auto_config=None,
        )
    except (
        Exception
    ):  # The private method should not raise; but guard against unexpected behavior
        pytest.fail("_prepare_guards_phase unexpectedly raised")
    finally:
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

    # No calibration_data triggers fallback metrics path
    report = runner.execute(model, adapter, edit, guards, cfg, calibration_data=None)
    pm = report.metrics.get("primary_metric", {})
    assert pm.get("preview") == 25.0
    assert pm.get("final") == 26.0
    assert report.metrics.get("latency_ms_per_tok") == 15.0


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
    flags = _collect_cuda_flags()
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
    assert report.status in {"success", "rollback"}


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


def test_edit_phase_with_non_dict_report_context():
    # Call _edit_phase directly with report.context not a dict to hit fallback
    runner = CoreRunner()
    model = DummyModel()
    adapter = DummyAdapter()
    edit = DummyEdit()
    from invarlock.core.api import RunReport

    report = RunReport()
    report.context = []  # type: ignore[assignment]
    res = runner._edit_phase(model, adapter, edit, {}, report, edit_config=None)
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

        def __setattr__(self, name, value):
            if name == "oops":
                raise RuntimeError("nope")
            return super().__setattr__(name, value)

    runner = CoreRunner()
    g = WeirdGuard()
    g.config = {}
    g.policy = {}
    # Should not raise despite exception setting attribute
    runner._apply_guard_policy(g, {"oops": 1, "cfg": 2})
    assert g.config.get("cfg") == 2


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
    # Fallback model layers metric via .get("n_layer", 0) covered by execution


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
    # Setting INVARLOCK_EVAL_DEVICE equal to current device should not move model
    import os

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
    os.environ["INVARLOCK_EVAL_DEVICE"] = "cpu"
    try:
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
    finally:
        del os.environ["INVARLOCK_EVAL_DEVICE"]


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
        # Should fallback from NaN ppl to default 50.0 due to invalid/empty summaries
        pm = metrics.get("primary_metric", {})
        assert pm.get("preview") > 0 and pm.get("final") > 0
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
    assert runner._measure_latency(RaisingModel(), [sample], "cpu") == 0.0


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


def test_policy_application_paths(monkeypatch, tmp_path):
    # Exercise _apply_guard_policy paths: direct attr, config, policy, fallback
    class PolyGuard(GoodGuard):
        name = "poly"

        def __init__(self):
            super().__init__()
            self.alpha = 0.0  # direct attribute target

    def fake_resolver(tier, edit_name, overrides):
        return {
            "poly": {
                "alpha": 0.5,  # direct attribute
                "cfg_only": 1,  # into config dict
                "pol_only": 2,  # into policy dict
                "new_attr": 3,  # setattr fallback
            }
        }

    import invarlock.core.runner as runner_mod

    monkeypatch.setattr(runner_mod, "resolve_tier_policies", fake_resolver)

    runner = CoreRunner()
    model = DummyModel()
    adapter = DummyAdapter()
    edit = DummyEdit()
    g = PolyGuard()
    g.config = {}
    g.policy = {}
    cfg = make_config(tmp_path)

    # Use calibration None to avoid heavy work
    report = runner.execute(model, adapter, edit, [g], cfg, calibration_data=None)
    assert report.status in {"success", "rollback"}
    # By implementation, policy params are applied to guard.config first when present
    assert (
        g.alpha == 0.5
        and g.config.get("cfg_only") == 1
        and g.config.get("pol_only") == 2
        and g.config.get("new_attr") == 3
    )


def test_policy_only_guard_application(monkeypatch, tmp_path):
    class PolicyOnlyGuard(GoodGuard):
        name = "policy_only"

        def __init__(self):
            super().__init__()
            self.config = None  # not a dict; should use policy dict path
            self.policy = {}

    def fake_resolver(tier, edit_name, overrides):
        return {"policy_only": {"theta": 7}}

    import invarlock.core.runner as runner_mod

    monkeypatch.setattr(runner_mod, "resolve_tier_policies", fake_resolver)

    runner = CoreRunner()
    model = DummyModel()
    adapter = DummyAdapter()
    edit = DummyEdit()
    g = PolicyOnlyGuard()
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
    _ = runner.execute(model, adapter, edit, [g], cfg, calibration_data=None)
    assert g.policy.get("theta") == 7


def test_guard_missing_passed_default(monkeypatch, tmp_path):
    class NoPassGuard(Guard):
        name = "nopass"

        def validate(self, model, adapter, context):
            return {}  # no 'passed' key → defaults to False

    runner = CoreRunner()
    model = DummyModel()
    adapter = DummyAdapter()
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
    report = runner.execute(
        model, adapter, DummyEdit(), [NoPassGuard()], cfg, calibration_data=None
    )
    assert report.status in {"rollback", "success"}


def test_handle_error_paths(monkeypatch):
    from invarlock.core.api import RunReport

    r = CoreRunner()
    report = RunReport()
    # No start_time: duration calculation branch skip
    r._handle_error(RuntimeError("x"), report)
    assert report.status == "failed" and isinstance(report.error, str)

    # With checkpoint set: exercise emergency_rollback and rollback_failed
    report.meta["initial_checkpoint"] = "checkpoint_1"

    calls = []

    def patched_log(component, operation, level, data=None):
        calls.append((component, operation))
        if operation == "emergency_rollback":
            raise RuntimeError("fail-log")

    # Ensure checkpoint_manager is present to enter emergency rollback branch
    from invarlock.core.checkpoint import CheckpointManager

    r.checkpoint_manager = CheckpointManager()
    monkeypatch.setattr(CoreRunner, "_log_event", staticmethod(patched_log))
    r._handle_error(RuntimeError("y"), report)  # Should not raise
    assert any(op == "rollback_failed" for _, op in calls)
