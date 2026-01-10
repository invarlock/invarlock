from __future__ import annotations

from typing import Any

import pytest
import torch

import invarlock.core.runner as runner_mod
from invarlock.core.api import RunConfig, RunReport
from invarlock.core.runner import CoreRunner


class _FakeLoss:
    def __init__(self, value: float):
        self._value = float(value)

    def item(self) -> float:
        return float(self._value)


class _FakeOutputs:
    def __init__(self, loss: float):
        self.loss = _FakeLoss(loss)


class _ToyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = torch.nn.Linear(3, 3, bias=False)

    def forward(
        self, input_ids=None, attention_mask=None, labels=None, token_type_ids=None
    ):
        _ = (attention_mask, labels, token_type_ids)
        if input_ids is None:
            loss = 1.0
        elif isinstance(input_ids, torch.Tensor):
            loss = float(input_ids.detach().float().mean().item())
        else:
            loss = float(torch.as_tensor(input_ids).detach().float().mean().item())
        return _FakeOutputs(loss)


def _batch(seq: list[int]) -> dict[str, object]:
    return {"input_ids": list(seq), "attention_mask": [1] * len(seq)}


def test_initialize_services_event_logger_run_id_skips_non_dict_context(
    tmp_path, monkeypatch
) -> None:
    captured: dict[str, Any] = {}

    class DummyLogger:
        def __init__(self, _path, *, run_id=None):  # noqa: ANN001
            captured["run_id"] = run_id

        def close(self) -> None:  # pragma: no cover - not exercised
            return None

    monkeypatch.setattr(runner_mod, "EventLogger", DummyLogger)
    runner = CoreRunner()
    cfg = RunConfig(event_path=tmp_path / "events.jsonl", context=[("run_id", "x")])
    runner._initialize_services(cfg)
    assert captured.get("run_id") is None


def test_edit_phase_sets_defaults_when_deltas_not_dict() -> None:
    runner = CoreRunner()
    runner._log_event = lambda *_a, **_k: None  # type: ignore[method-assign]

    class Edit:
        name = "noop"

        def can_edit(self, _desc):  # noqa: ANN001
            return True

        def apply(self, _model, _adapter, **_kwargs):  # noqa: ANN001
            return {"name": self.name, "deltas": ["bad"]}

    report = RunReport()
    runner._edit_phase(
        _ToyModel(),
        adapter=object(),
        edit=Edit(),
        model_desc={},
        report=report,
        edit_config=None,
    )
    assert report.context.get("edit", {}).get("params_changed") == 0


def test_compute_real_metrics_raises_without_len_when_materialize_not_allowed(
    monkeypatch,
) -> None:
    monkeypatch.delenv("INVARLOCK_ALLOW_CALIBRATION_MATERIALIZE", raising=False)

    runner = CoreRunner()
    cfg = RunConfig(
        device="cpu",
        context={"eval": {"loss": {"type": "causal"}}},
    )

    def _gen():
        yield _batch([1, 1, 1])

    with pytest.raises(ValueError, match=r"must define __len__"):
        runner._compute_real_metrics(
            _ToyModel(),
            _gen(),
            adapter=object(),
            preview_n=1,
            final_n=1,
            config=cfg,
        )


def test_compute_real_metrics_dataset_seed_skips_non_dict_dataset_cfg() -> None:
    runner = CoreRunner()
    cfg = RunConfig(
        device="cpu",
        context={
            "dataset": "not-a-dict",
            "eval": {"loss": {"type": "causal"}},
        },
    )
    calibration = [
        _batch([1, 1, 1]),
        _batch([2, 2, 2]),
    ]

    metrics, _ = runner._compute_real_metrics(
        _ToyModel(),
        calibration,
        adapter=object(),
        preview_n=1,
        final_n=1,
        config=cfg,
    )
    assert metrics.get("eval_error") is None
