from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from invarlock.core.api import Guard, ModelAdapter, ModelEdit, RunConfig
from invarlock.core.runner import CoreRunner, collect_cuda_flags


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
    assert report.status == "rollback"
    assert report.metrics["eval_state"] == {
        "evaluated": False,
        "reason": "missing_calibration_data",
    }
    assert "primary_metric" not in report.metrics
    assert report.meta["rollback_reason"] == "primary_metric_invalid"


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
