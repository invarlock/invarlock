from pathlib import Path
from typing import Any

import pytest

from invarlock.core.runner import CoreRunner
from invarlock.core.types import LogLevel


class DummyAdapter:
    def describe(self, model: Any) -> dict[str, Any]:
        return {"n_layer": 2, "heads_per_layer": 2}

    def snapshot(self, model: Any) -> bytes:  # for checkpoint manager if enabled
        return b"snapshot"


class DummyEdit:
    name = "baseline"

    def can_edit(self, model_desc: dict[str, Any]) -> bool:
        return True

    def apply(self, model: Any, adapter: Any, **kwargs) -> dict[str, Any]:
        return {
            "name": self.name,
            "deltas": {"params_changed": 0, "layers_modified": 0},
        }


class DummyGuard:
    def __init__(self, name: str) -> None:
        self.name = name
        self.config = {}
        self.policy = {}

    def validate(self, model: Any, adapter: Any, context: dict) -> dict[str, Any]:
        return {"passed": True, "action": "none"}


def test_serialize_config_and_services_init_cleanup(tmp_path: Path):
    r = CoreRunner()
    # Serialize config without guards in context
    from invarlock.core.api import RunConfig

    cfg = RunConfig(
        event_path=tmp_path / "events.jsonl", checkpoint_interval=1, context={}
    )
    ser = r._serialize_config(cfg)
    assert ser["device"] == "auto"
    assert ser["checkpoint_interval"] == 1
    assert ser["guards"] == {}

    # Initialize services, then cleanup
    r._initialize_services(cfg)
    assert r.event_logger is not None
    assert r.checkpoint_manager is not None
    # _log_event should not raise
    r._log_event("runner", "ping", LogLevel.INFO, {"ok": True})
    r._cleanup_services()
    assert r.event_logger is None or r.event_logger._file is None


def test_prepare_and_eval_fallback(tmp_path: Path):
    r = CoreRunner()
    from invarlock.core.api import RunConfig, RunReport

    cfg = RunConfig()
    rep = RunReport()
    adapter = DummyAdapter()
    model = object()

    _ = r._prepare_phase(model, adapter, rep)
    assert rep.meta.get("model", {}).get("n_layer") == 2

    # No checkpoint manager configured; no checkpoint id in meta
    assert "initial_checkpoint" not in rep.meta

    # Eval phase without calibration data uses fallback metrics
    metrics = r._eval_phase(
        model, adapter, calibration_data=None, report=rep, config=cfg
    )
    pm = metrics.get("primary_metric", {})
    assert isinstance(pm, dict) and pm
    assert pm.get("final") and pm.get("preview")


def test_resolve_guard_policies_and_apply(monkeypatch: pytest.MonkeyPatch):
    r = CoreRunner()
    from invarlock.core.api import RunReport

    rep = RunReport()
    rep.meta["config"] = {"guards": {}, "auto": {"tier": "balanced", "enabled": True}}
    rep.meta["edit_name"] = "baseline"

    # Happy path
    called = {}

    def fake_resolve(
        tier: str, edit_name: str | None, overrides: dict
    ) -> dict[str, dict[str, Any]]:
        called["args"] = (tier, edit_name)
        return {"g": {"sigma_quantile": 0.9, "deadband": 0.1}}

    monkeypatch.setattr("invarlock.core.runner.resolve_tier_policies", fake_resolve)
    policies = r._resolve_guard_policies(rep)
    assert policies["g"]["sigma_quantile"] == 0.9
    assert called["args"] == ("balanced", "baseline")

    # Apply to guard: attribute/config/policy application
    g = DummyGuard("g")
    r._apply_guard_policy(g, policies["g"])  # should set in config dict
    assert g.config["sigma_quantile"] == 0.9
    assert g.config["deadband"] == 0.1

    # Error path: resolver raises -> empty dict returned
    def bad_resolve(*_a, **_k):
        raise RuntimeError("boom")

    monkeypatch.setattr("invarlock.core.runner.resolve_tier_policies", bad_resolve)
    empty = r._resolve_guard_policies(rep)
    assert empty == {}


def test_samples_to_dataloader_and_latency():
    r = CoreRunner()

    # One sample with ids and attention mask triggers token masking path too
    sample = {
        "input_ids": [1, 2, 3, 4],
        "attention_mask": [1, 1, 1, 0],
    }
    dl = r._samples_to_dataloader([sample])
    batches = list(iter(dl))
    assert len(batches) == 1
    b = batches[0]
    assert set(b.keys()) >= {"input_ids", "labels"}
    # labels masked at padding positions
    assert int(b["labels"][0, -1].item()) == -100

    class TinyModel:
        def __call__(
            self, input_ids, attention_mask=None, labels=None, token_type_ids=None
        ):
            class Out:  # noqa: D401
                def __init__(self):
                    self.loss = 0.0

            return Out()

    latency = r._measure_latency(TinyModel(), [sample], device="cpu")
    assert isinstance(latency, float)
    assert latency >= 0.0
