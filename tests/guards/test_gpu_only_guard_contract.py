import torch
import torch.nn as nn

from invarlock.guards.rmt import RMTGuard
from invarlock.guards.spectral import SpectralGuard


class _DummyAttn(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.c_attn = nn.Linear(hidden, hidden, bias=False)
        self.c_proj = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj(self.c_attn(x))


class _DummyMLP(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.c_fc = nn.Linear(hidden, hidden * 2, bias=False)
        self.c_proj = nn.Linear(hidden * 2, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj(torch.relu(self.c_fc(x)))


class _DummyModel(nn.Module):
    def __init__(self, *, vocab: int = 128, hidden: int = 8):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.ln = nn.LayerNorm(hidden)
        self.attn = _DummyAttn(hidden)
        self.mlp = _DummyMLP(hidden)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        _ = attention_mask
        x = self.embed(input_ids)
        x = self.ln(x)
        x = x.reshape(-1, x.shape[-1])
        x = self.attn(x)
        x = self.mlp(x)
        return x


def _raise_cpu(*_args, **_kwargs):  # pragma: no cover - raised when violated
    # Use BaseException so broad `except Exception` blocks do not swallow it.
    raise BaseException("cpu() called in guard path")


def _raise_svd(*_args, **_kwargs):  # pragma: no cover - raised when violated
    # Use BaseException so broad `except Exception` blocks do not swallow it.
    raise BaseException("SVD called in guard path")


def test_spectral_is_gpu_only_contract(monkeypatch):
    monkeypatch.setattr(torch.Tensor, "cpu", _raise_cpu, raising=True)
    monkeypatch.setattr(torch.linalg, "svdvals", _raise_svd, raising=True)
    monkeypatch.setattr(torch, "svd", _raise_svd, raising=True)

    model = _DummyModel()
    guard = SpectralGuard(correction_enabled=False, ignore_preview_inflation=False)
    out = guard.prepare(model, adapter=None, calib=None, policy={})
    assert out["ready"] is True
    guard.after_edit(model)
    result = guard.validate(model, adapter=None, context={})
    assert isinstance(result, dict)


def test_rmt_is_gpu_only_contract(monkeypatch):
    monkeypatch.setattr(torch.Tensor, "cpu", _raise_cpu, raising=True)
    monkeypatch.setattr(torch.linalg, "svdvals", _raise_svd, raising=True)
    monkeypatch.setattr(torch, "svd", _raise_svd, raising=True)

    model = _DummyModel()
    calib = [
        {
            "input_ids": torch.randint(0, 128, (2, 8)),
            "attention_mask": torch.ones((2, 8), dtype=torch.long),
        }
        for _ in range(2)
    ]
    guard = RMTGuard(correct=False)
    out = guard.prepare(
        model, adapter=None, calib=calib, policy={"activation_required": True}
    )
    assert out["ready"] is True
    guard.after_edit(model)
    outcome = guard.finalize(model)
    assert outcome.passed in {True, False}
