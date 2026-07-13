from __future__ import annotations

from typing import Any

import pytest

from invarlock.core.api import ModelAdapter, RunConfig, RunReport
from invarlock.core.runner import CoreRunner, _profile_from_context


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


@pytest.mark.parametrize(
    ("context", "expected"),
    [
        (None, None),
        ({}, None),
        ({"profile": " CI "}, "ci"),
        ({"runtime": {"profile": " Release "}}, "release"),
        ({"profile": " ", "runtime": {"profile": ""}}, None),
    ],
)
def test_profile_from_context(context, expected):
    assert _profile_from_context(context) == expected


# Intentionally avoid ratio CI mismatch raise path here, as it can expose
# an unrelated UnboundLocal bug in error handling noted elsewhere.


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

    def fake_resolver(tier, edit_name, overrides, *, profile):
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

    def fake_resolver(tier, edit_name, overrides, *, profile):
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
