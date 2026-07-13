from __future__ import annotations

import torch

from invarlock.core.api import RunConfig
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


def _cfg(
    *, profile: str, pairing_baseline: dict, seq_len: int, stride: int
) -> RunConfig:
    return RunConfig(
        device="cpu",
        event_path=None,
        context={
            "profile": profile,
            "pairing_baseline": pairing_baseline,
            "dataset": {"seq_len": int(seq_len), "stride": int(stride), "seed": 0},
            "eval": {"loss": {"type": "causal"}},
        },
    )


def _batch(seq: list[int]) -> dict[str, object]:
    return {"input_ids": list(seq), "attention_mask": [1] * len(seq)}


def test_ci_pairing_no_pairing_context_skips_ci_enforcement() -> None:
    runner = CoreRunner()
    model = _ToyModel()
    adapter = object()

    cfg = _cfg(
        profile="ci",
        pairing_baseline={},
        seq_len=8,
        stride=8,
    )
    calibration = [
        _batch([1, 1, 1]),
        _batch([2, 2, 2]),
        _batch([10, 10, 10]),
        _batch([20, 20, 20]),
    ]

    metrics, _ = runner._compute_real_metrics(
        model, calibration, adapter, preview_n=2, final_n=2, config=cfg
    )
    assert metrics.get("eval_error") is None


def test_ci_pairing_insufficient_sample_sets_eval_error() -> None:
    runner = CoreRunner()
    model = _ToyModel()
    adapter = object()

    cfg = _cfg(
        profile="ci",
        pairing_baseline={"preview": {}, "final": {}},
        seq_len=8,
        stride=8,
    )
    calibration = [
        _batch([1, 1, 1]),
        _batch([2, 2, 2]),
        _batch([10, 10, 10]),
        _batch([20, 20, 20]),
    ]

    metrics, _ = runner._compute_real_metrics(
        model, calibration, adapter, preview_n=2, final_n=2, config=cfg
    )
    assert metrics["eval_error"]["type"] == "InvarlockError"
    assert "INSUFFICIENT-SAMPLE" in metrics["eval_error"]["message"]


def test_ci_pairing_count_mismatch_sets_eval_error() -> None:
    runner = CoreRunner()
    model = _ToyModel()
    adapter = object()

    cfg = _cfg(
        profile="ci",
        pairing_baseline={"preview": {}, "final": {}},
        seq_len=8,
        stride=8,
    )
    calibration = [
        _batch([1, 1, 1]),
        _batch([2, 2, 2]),
        _batch([10, 10, 10]),
        _batch([20, 20, 20]),
        _batch([30, 30, 30]),
    ]

    metrics, _ = runner._compute_real_metrics(
        model, calibration, adapter, preview_n=2, final_n=3, config=cfg
    )
    assert "Window count mismatch detected" in metrics["eval_error"]["message"]


def test_ci_pairing_overlap_sets_eval_error() -> None:
    runner = CoreRunner()
    model = _ToyModel()
    adapter = object()

    cfg = _cfg(
        profile="ci",
        pairing_baseline={"preview": {}, "final": {}},
        seq_len=8,
        stride=4,
    )
    calibration = [
        _batch([1, 1, 1]),
        _batch([2, 2, 2]),
        _batch([10, 10, 10]),
        _batch([20, 20, 20]),
    ]

    metrics, _ = runner._compute_real_metrics(
        model, calibration, adapter, preview_n=2, final_n=2, config=cfg
    )
    assert "Window overlap detected" in metrics["eval_error"]["message"]


def test_ci_pairing_token_mismatch_sets_eval_error() -> None:
    runner = CoreRunner()
    model = _ToyModel()
    adapter = object()

    pairing_baseline = {
        "preview": {"window_ids": [0, 1], "input_ids": [[9, 9, 9], [9, 9, 9]]},
        "final": {"window_ids": [2, 3], "input_ids": [[8, 8, 8], [8, 8, 8]]},
    }
    cfg = _cfg(
        profile="ci",
        pairing_baseline=pairing_baseline,
        seq_len=8,
        stride=8,
    )
    calibration = [
        _batch([1, 1, 1]),
        _batch([2, 2, 2]),
        _batch([10, 10, 10]),
        _batch([20, 20, 20]),
    ]

    metrics, _ = runner._compute_real_metrics(
        model, calibration, adapter, preview_n=2, final_n=2, config=cfg
    )
    assert "Window pairing mismatch detected" in metrics["eval_error"]["message"]
