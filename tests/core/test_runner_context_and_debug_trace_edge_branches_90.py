from __future__ import annotations

from typing import Any

import invarlock.core.runner as runner_mod
from invarlock.core.api import Guard, ModelAdapter, ModelEdit, RunConfig, RunReport
from invarlock.core.runner import CoreRunner


class _DummyModel:
    def parameters(self):  # minimal iterator with a .device
        class _P:
            device = "cpu"

        yield _P()

    def eval(self):  # pragma: no cover - trivial
        return None


class _DummyAdapter(ModelAdapter):
    name = "dummy"

    def can_handle(self, model: Any) -> bool:  # pragma: no cover - not used
        return True

    def describe(self, model: Any) -> dict[str, Any]:
        return {"n_layer": 0}

    def snapshot(self, model: Any) -> bytes:
        return b""

    def restore(self, model: Any, blob: bytes) -> None:  # pragma: no cover - not used
        _ = blob


class _DummyEdit(ModelEdit):
    name = "noop"

    def can_edit(self, model_desc: dict[str, Any]) -> bool:  # pragma: no cover - not used
        return True

    def apply(self, model: Any, adapter: ModelAdapter, **kwargs) -> dict[str, Any]:
        _ = model, adapter, kwargs
        return {"name": self.name, "deltas": {"params_changed": 0}}


class _DummyGuard(Guard):
    name = "noop_guard"

    def validate(
        self, model: Any, adapter: ModelAdapter, context: dict[str, Any]
    ) -> dict[str, Any]:
        _ = model, adapter, context
        return {"passed": True}


def _patch_minimal_pipeline(monkeypatch) -> None:
    monkeypatch.setattr(CoreRunner, "_prepare_phase", lambda *_a, **_k: {})
    monkeypatch.setattr(CoreRunner, "_prepare_guards_phase", lambda *_a, **_k: None)
    monkeypatch.setattr(CoreRunner, "_edit_phase", lambda *_a, **_k: {"name": "noop"})
    monkeypatch.setattr(CoreRunner, "_guard_phase", lambda *_a, **_k: [])
    monkeypatch.setattr(
        CoreRunner,
        "_eval_phase",
        lambda *_a, **_k: {
            "primary_metric": {"kind": "ppl_causal", "preview": 1.0, "final": 1.0}
        },
    )
    monkeypatch.setattr(CoreRunner, "_finalize_phase", lambda *_a, **_k: "success")
    monkeypatch.setattr(CoreRunner, "_log_event", lambda *_a, **_k: None)


def test_execute_falls_back_when_report_context_update_raises(monkeypatch) -> None:
    _patch_minimal_pipeline(monkeypatch)

    update_called = {"called": False}

    class _BadContext(dict):
        def update(self, *_a, **_k):  # noqa: ANN001
            update_called["called"] = True
            raise RuntimeError("boom")

    class _BadReport(RunReport):
        def __init__(self) -> None:
            super().__init__()
            self.context = _BadContext()

    monkeypatch.setattr(runner_mod, "RunReport", _BadReport)

    runner = CoreRunner()
    cfg = RunConfig(context={"run_id": "x"})
    report = runner.execute(
        _DummyModel(),
        _DummyAdapter(),
        _DummyEdit(),
        [_DummyGuard()],
        cfg,
        calibration_data=None,
    )
    assert update_called["called"] is True
    assert report.status == "success"
    assert isinstance(report.context, dict)


def test_execute_merges_auto_config_when_context_already_has_auto(
    monkeypatch,
) -> None:
    _patch_minimal_pipeline(monkeypatch)

    runner = CoreRunner()
    cfg = RunConfig(context={"auto": {"tier": "balanced", "keep": True}})
    report = runner.execute(
        _DummyModel(),
        _DummyAdapter(),
        _DummyEdit(),
        [_DummyGuard()],
        cfg,
        calibration_data=None,
        auto_config={"tier": "aggressive"},
    )
    assert report.status == "success"
    assert cfg.context["auto"]["tier"] == "aggressive"
    assert cfg.context["auto"]["keep"] is True


def test_eval_phase_debug_trace_handles_empty_list(monkeypatch) -> None:
    runner = CoreRunner()
    model = _DummyModel()
    adapter = _DummyAdapter()
    report = RunReport()
    cfg = RunConfig(context={})

    monkeypatch.setenv("INVARLOCK_DEBUG_TRACE", "1")
    monkeypatch.setattr(
        CoreRunner,
        "_compute_real_metrics",
        staticmethod(
            lambda *_a, **_k: (
                {"primary_metric": {"kind": "ppl_causal", "preview": 1.0, "final": 1.0}},
                {"preview": {}, "final": {}},
            )
        ),
    )

    metrics = runner._eval_phase(
        model, adapter, calibration_data=[], report=report, preview_n=None, final_n=None, config=cfg
    )
    assert metrics["primary_metric"]["final"] == 1.0


def test_eval_phase_debug_trace_handles_custom_indexable(monkeypatch) -> None:
    runner = CoreRunner()
    model = _DummyModel()
    adapter = _DummyAdapter()
    report = RunReport()
    cfg = RunConfig(context={})

    class _Indexable:
        def __len__(self) -> int:
            return 1

        def __getitem__(self, _idx: int) -> dict[str, Any]:
            return {"input_ids": [1, 2], "labels": [-100, 5]}

    monkeypatch.setenv("INVARLOCK_DEBUG_TRACE", "1")
    monkeypatch.setattr(
        CoreRunner,
        "_compute_real_metrics",
        staticmethod(
            lambda *_a, **_k: (
                {"primary_metric": {"kind": "ppl_causal", "preview": 1.0, "final": 1.0}},
                {"preview": {}, "final": {}},
            )
        ),
    )

    metrics = runner._eval_phase(
        model, adapter, calibration_data=_Indexable(), report=report, preview_n=None, final_n=None, config=cfg
    )
    assert metrics["primary_metric"]["preview"] == 1.0
