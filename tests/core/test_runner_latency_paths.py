from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from invarlock.core.runner_latency import measure_latency, samples_to_dataloader


class _FakeTensor:
    def dim(self) -> int:
        return 1

    def unsqueeze(self, _axis: int):
        raise RuntimeError("unsqueeze disabled")

    def to(self, _device):
        raise RuntimeError("device transfer disabled")

    def numel(self) -> int:
        raise RuntimeError("numel disabled")


class _LatencyModel:
    def __call__(
        self, input_ids, attention_mask=None, labels=None, token_type_ids=None
    ):
        return SimpleNamespace(loss=torch.tensor(0.1))


class _CudaDevice:
    type = "cuda"


def test_measure_latency_handles_sync_unsqueeze_to_and_numel_failures(
    monkeypatch,
) -> None:
    model = _LatencyModel()

    monkeypatch.setattr(torch, "tensor", lambda *_args, **_kwargs: _FakeTensor())
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def _boom_sync() -> None:
        raise RuntimeError("sync failed")

    monkeypatch.setattr(torch.cuda, "synchronize", _boom_sync)

    with pytest.raises(RuntimeError, match="Latency measurement batch shaping failed"):
        measure_latency(model, [[1, 2, 3]], _CudaDevice())


def test_measure_latency_tolerates_attention_and_token_type_conversion_failures(
    monkeypatch,
) -> None:
    model = _LatencyModel()
    real_tensor = torch.tensor

    def fake_tensor(value, *args, **kwargs):
        if isinstance(value, str) and value in {"bad-attn", "bad-token"}:
            raise RuntimeError("bad auxiliary tensor")
        return real_tensor(value, *args, **kwargs)

    monkeypatch.setattr(torch, "tensor", fake_tensor)

    with pytest.raises(
        RuntimeError,
        match="Latency measurement attention-mask preparation failed",
    ):
        measure_latency(
            model,
            [
                {
                    "input_ids": torch.tensor([1, 2, 3], dtype=torch.long),
                    "attention_mask": "bad-attn",
                    "token_type_ids": "bad-token",
                }
            ],
            "cpu",
        )


def test_samples_to_dataloader_skips_missing_inputs_and_keeps_2d_tensor_fields() -> (
    None
):
    samples = [
        {"attention_mask": [1, 1, 1]},
        {
            "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
            "token_type_ids": torch.tensor([[0, 0, 1]], dtype=torch.long),
            "labels": torch.tensor([[1, -100, 3]], dtype=torch.long),
        },
    ]

    dataloader = samples_to_dataloader(samples)
    batches = list(iter(dataloader))

    assert len(dataloader) == 2
    assert len(batches) == 1
    batch = batches[0]
    assert batch["input_ids"].shape == (1, 3)
    assert batch["attention_mask"].shape == (1, 3)
    assert batch["token_type_ids"].shape == (1, 3)
    assert batch["labels"].shape == (1, 3)
