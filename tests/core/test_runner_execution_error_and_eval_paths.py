from __future__ import annotations

from types import SimpleNamespace

import pytest

from invarlock.core.runner import CoreRunner, coerce_bool, env_flag
from tests.core.test_runner_execution_metrics_and_memory import (
    BadGuard,
    DummyAdapter,
    DummyModel,
    ErrPrepareGuard,
    make_config,
)


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


def test_coerce_bool_and_env_flag(monkeypatch):
    assert coerce_bool(True) is True
    assert coerce_bool(False) is False
    assert coerce_bool(1) is True
    assert coerce_bool(0) is False
    assert coerce_bool(" yes ") is True
    assert coerce_bool("off") is False
    assert coerce_bool("maybe") is None

    monkeypatch.delenv("INVARLOCK_TEST_FLAG", raising=False)
    assert env_flag("INVARLOCK_TEST_FLAG") is None
    monkeypatch.setenv("INVARLOCK_TEST_FLAG", "0")
    assert env_flag("INVARLOCK_TEST_FLAG") is False


def test_resolve_policy_flags_precedence(monkeypatch, tmp_path):
    runner = CoreRunner()
    cfg = make_config(tmp_path)
    cfg.context.setdefault("run", {})["strict_eval"] = None
    cfg.context.setdefault("eval", {})["strict"] = "false"
    cfg.context.setdefault("run", {})["strict_guard_prepare"] = "0"
    cfg.context.setdefault("run", {})["allow_calibration_materialize"] = "1"

    flags = runner._resolve_policy_flags(cfg)

    assert flags["strict_eval"] is False
    assert flags["strict_guard_prepare"] is False
    assert flags["allow_calibration_materialize"] is True


def test_prepare_guards_phase_non_strict_still_raises_programming_errors(tmp_path):
    from invarlock.core.api import RunReport

    runner = CoreRunner()
    cfg = make_config(tmp_path)
    cfg.context.setdefault("run", {})["strict_guard_prepare"] = False
    report = RunReport()

    with pytest.raises(RuntimeError, match="Guard 'err' prepare failed"):
        runner._prepare_guards_phase(
            DummyModel(),
            DummyAdapter(),
            [ErrPrepareGuard()],
            calibration_data=None,
            report=report,
            auto_config=None,
            config=cfg,
        )

    failures = report.meta.get("guard_prepare_failures", [])
    assert failures and failures[0]["guard"] == ErrPrepareGuard.name


def test_prepare_guards_phase_context_errors_raise(tmp_path):
    from invarlock.core.api import RunReport

    runner = CoreRunner()
    cfg = make_config(tmp_path)
    report = RunReport()

    with pytest.raises(RuntimeError, match="run context setup failed"):
        runner._prepare_guards_phase(
            DummyModel(),
            DummyAdapter(),
            [BadGuard()],
            calibration_data=None,
            report=report,
            auto_config=None,
            config=cfg,
        )


def test_eval_phase_strict_eval_raises(monkeypatch, tmp_path):
    runner = CoreRunner()
    cfg = make_config(tmp_path)
    cfg.context.setdefault("run", {})["strict_eval"] = True

    def fake_compute(*_args, **_kwargs):
        return (
            {
                "eval_error": {"message": "boom", "type": "fail"},
                "primary_metric": {"kind": "ppl_causal", "preview": 1.0, "final": 1.0},
            },
            {"preview": {}, "final": {}},
        )

    monkeypatch.setattr(CoreRunner, "_compute_real_metrics", staticmethod(fake_compute))
    with pytest.raises(RuntimeError, match="Evaluation failed"):
        runner._eval_phase(
            DummyModel(),
            DummyAdapter(),
            calibration_data=[{"input_ids": [1, 2, 3]}],
            report=SimpleNamespace(),
            preview_n=1,
            final_n=1,
            config=cfg,
        )


def test_eval_phase_soft_fail_sets_metrics_on_plain_report(monkeypatch, tmp_path):
    runner = CoreRunner()
    cfg = make_config(tmp_path)
    cfg.context.setdefault("run", {})["strict_eval"] = False

    def fake_compute(*_args, **_kwargs):
        return (
            {
                "eval_error": {"message": "soft", "type": "fail"},
                "primary_metric": {"kind": "ppl_causal", "preview": 1.0, "final": 1.0},
            },
            {"preview": {}, "final": {}},
        )

    monkeypatch.setattr(CoreRunner, "_compute_real_metrics", staticmethod(fake_compute))
    report = SimpleNamespace()
    metrics = runner._eval_phase(
        DummyModel(),
        DummyAdapter(),
        calibration_data=[{"input_ids": [1, 2, 3]}],
        report=report,
        preview_n=1,
        final_n=1,
        config=cfg,
    )

    assert metrics["eval_error"]["message"] == "soft"
    assert report.metrics["primary_metric"]["kind"] == "ppl_causal"


def test_compute_real_metrics_materialize_iterable(tmp_path):
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

    class IterableCalib:
        def __init__(self, items):
            self._items = list(items)

        def __iter__(self):
            return iter(self._items)

    runner = CoreRunner()
    cfg = make_config(tmp_path)
    cfg.context.setdefault("run", {})["allow_calibration_materialize"] = True
    cfg.context.setdefault("eval", {}).update({"bootstrap": {"enabled": False}})
    calibration = IterableCalib(
        [
            {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]},
            {"input_ids": [4, 5, 6], "attention_mask": [1, 1, 1]},
        ]
    )

    metrics, _ = runner._compute_real_metrics(
        Toy(), calibration, DummyAdapter(), preview_n=1, final_n=1, config=cfg
    )
    pm = metrics.get("primary_metric", {})
    assert pm.get("final") and pm.get("preview")


def test_compute_real_metrics_slice_fallback_materializes(tmp_path):
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

    class BadIndexCalib:
        def __init__(self, items):
            self._items = list(items)

        def __len__(self):
            return len(self._items)

        def __iter__(self):
            return iter(self._items)

        def __getitem__(self, idx):
            raise TypeError("indexing disabled")

    runner = CoreRunner()
    cfg = make_config(tmp_path)
    cfg.context.setdefault("run", {})["allow_calibration_materialize"] = True
    cfg.context.setdefault("eval", {}).update({"bootstrap": {"enabled": False}})
    calibration = BadIndexCalib(
        [
            {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]},
            {"input_ids": [4, 5, 6], "attention_mask": [1, 1, 1]},
        ]
    )

    metrics, _ = runner._compute_real_metrics(
        Toy(), calibration, DummyAdapter(), preview_n=1, final_n=1, config=cfg
    )
    pm = metrics.get("primary_metric", {})
    assert pm.get("final") and pm.get("preview")
