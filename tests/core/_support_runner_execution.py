from __future__ import annotations

from pathlib import Path
from typing import Any

from invarlock.core.api import Guard, ModelAdapter, ModelEdit, RunConfig


class DummyModel:
    def __init__(self):
        self._restored = False

    def parameters(self):
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
        return "ok"


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
        return {"name": self.name}


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
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg
