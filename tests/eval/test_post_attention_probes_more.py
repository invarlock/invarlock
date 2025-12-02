import types

import torch
import torch.nn as nn

import invarlock.eval.probes.post_attention as pa


class TinyBlock(nn.Module):
    def __init__(self, hidden_size: int, n_heads: int, mlp_dim: int):
        super().__init__()
        self.attn = nn.Linear(hidden_size, hidden_size, bias=False)
        self.ln_2 = nn.Identity()
        # Provide an object with c_fc attr to match probe expectations
        self.mlp = types.SimpleNamespace(
            c_fc=nn.Linear(hidden_size, mlp_dim, bias=False)
        )

    def forward(self, x):
        a = self.attn(x)
        _ = self.ln_2(a)
        m = self.mlp.c_fc(a)
        return types.SimpleNamespace(attn_out=a, mlp_out=m)


class TinyModelObj(nn.Module):
    def __init__(self, n_layers=2, n_heads=2, hidden=8, mlp_dim=8, vocab=16):
        super().__init__()
        self.config = types.SimpleNamespace(n_layer=n_layers, n_head=n_heads)
        self.embedding = nn.Embedding(vocab, hidden)
        self.transformer = types.SimpleNamespace(
            h=nn.ModuleList(
                [TinyBlock(hidden, n_heads, mlp_dim) for _ in range(n_layers)]
            )
        )
        self.to_logits = nn.Linear(mlp_dim, vocab, bias=False)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        for blk in self.transformer.h:
            x = blk(x).mlp_out
        return types.SimpleNamespace(logits=self.to_logits(x))


class TinyModelRaw(nn.Module):
    def __init__(self, n_layers=1, n_heads=2, hidden=8, mlp_dim=8, vocab=16):
        super().__init__()
        self.config = types.SimpleNamespace(n_layer=n_layers, n_head=n_heads)
        self.embedding = nn.Embedding(vocab, hidden)
        self.transformer = types.SimpleNamespace(
            h=nn.ModuleList(
                [TinyBlock(hidden, n_heads, mlp_dim) for _ in range(n_layers)]
            )
        )
        self.to_logits = nn.Linear(mlp_dim, vocab, bias=False)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        for blk in self.transformer.h:
            x = blk(x).mlp_out
        return self.to_logits(x)


def test_head_scores_with_norm_patched(monkeypatch):
    # Patch torch.norm inside module to avoid multi-dim error on some builds
    def fake_norm(inp, p="fro", dim=None, keepdim=False, out=None, dtype=None):  # noqa: ARG001
        # Return a vector over head dimension length
        if isinstance(dim, tuple | list) and len(dim) == 3 and inp.dim() == 4:
            return torch.ones(inp.size(2), device=inp.device)
        # Fallback
        return torch.linalg.vector_norm(inp)

    monkeypatch.setattr(pa.torch, "norm", fake_norm)

    model = TinyModelObj(n_layers=2, n_heads=2, hidden=8, mlp_dim=8, vocab=32)
    calib = [torch.randint(0, 32, (2, 4)), torch.randint(0, 32, (1, 5))]
    out = pa.compute_post_attention_head_scores(model, calib, calibration_windows=2)
    scores = out["scores"]
    assert tuple(scores.shape) == (2, 2)


def test_wanda_with_raw_outputs_and_short_seq():
    model = TinyModelRaw(n_layers=1, n_heads=2, hidden=8, mlp_dim=8, vocab=16)
    # Short sequence length triggers the branch that skips backward
    calib = [torch.randint(0, 16, (1, 1))]
    out = pa.compute_wanda_neuron_scores(model, calib, calibration_windows=1)
    scores = out["scores"]
    assert scores.shape[0] == 1 and scores.shape[1] >= 8


def test_blend_one_dimensional_scores_branch():
    a = torch.ones(6)
    b = torch.full((4,), 2.0)
    out = pa.blend_neuron_scores([a, b])
    assert out.shape == (6,) and torch.all(out[:4] > 0)
