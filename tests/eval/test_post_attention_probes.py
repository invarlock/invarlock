import types

import pytest
import torch
import torch.nn as nn

from invarlock.eval.probes.post_attention import (
    blend_neuron_scores,
    compute_post_attention_head_scores,
    compute_wanda_neuron_scores,
)


class TinyBlock(nn.Module):
    def __init__(self, hidden_size: int, n_heads: int, mlp_dim: int):
        super().__init__()
        self.hidden = hidden_size
        self.n_heads = n_heads

        # Attention stub that returns a tensor with shape [B, S, hidden]
        class AttnStub(nn.Module):
            def __init__(self, hidden: int):
                super().__init__()
                self.proj = nn.Linear(hidden, hidden, bias=False)

            def forward(self, x):
                # x: [B, S, hidden]
                return (self.proj(x),)

        class MLP(nn.Module):
            def __init__(self, hidden: int, mlp_dim: int):
                super().__init__()
                self.c_fc = nn.Linear(hidden, mlp_dim, bias=False)

            def forward(self, x):
                return self.c_fc(x)

        self.attn = AttnStub(hidden_size)
        self.ln_2 = nn.Identity()
        self.mlp = MLP(hidden_size, mlp_dim)

    def forward(self, x):
        out = self.attn(x)[0]
        _ = self.ln_2(out)
        out = self.mlp(out)
        return out


class TinyTransformer(nn.Module):
    def __init__(self, n_layers=2, n_heads=2, hidden=8, mlp_dim=None, vocab=16):
        super().__init__()
        self.config = types.SimpleNamespace(n_layer=n_layers, n_head=n_heads)
        self.embedding = nn.Embedding(vocab, hidden)
        mlp_dim = hidden if mlp_dim is None else mlp_dim
        self.transformer = types.SimpleNamespace(
            h=nn.ModuleList(
                [TinyBlock(hidden, n_heads, mlp_dim) for _ in range(n_layers)]
            )
        )
        self.to_logits = nn.Linear(mlp_dim, vocab, bias=False)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        for blk in self.transformer.h:
            x = blk(x)
        logits = self.to_logits(x)
        return types.SimpleNamespace(logits=logits)


class TinyTransformerDirect(nn.Module):
    """Variant without transformer namespace to hit fallback branch."""

    def __init__(self, n_layers=1, n_heads=2, hidden=8, vocab=32):
        super().__init__()
        self.config = types.SimpleNamespace(n_layer=n_layers, n_head=n_heads)
        self.embedding = nn.Embedding(vocab, hidden)
        self.h = nn.ModuleList(
            [TinyBlock(hidden, n_heads, hidden) for _ in range(n_layers)]
        )
        self.to_logits = nn.Linear(hidden, vocab, bias=False)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        for blk in self.h:
            x = blk(x)
        logits = self.to_logits(x)
        return types.SimpleNamespace(logits=logits)


def test_post_attention_head_scores_and_wanda_paths():
    torch.manual_seed(0)
    model = TinyTransformer(n_layers=2, n_heads=2, hidden=8, vocab=32)
    # Calibration data: two small batches
    calib = [torch.randint(0, 32, (2, 4)), torch.randint(0, 32, (1, 5))]

    # Head scores shape [n_layers, n_heads]
    try:
        out_heads = compute_post_attention_head_scores(
            model, calib, calibration_windows=2
        )
        scores_h = out_heads["scores"]
        assert tuple(scores_h.shape) == (2, 2)
    except RuntimeError as e:
        # Some torch builds error on torch.norm with 3 dims and p='fro'; skip gracefully
        if "dim must be a 2-tuple" in str(e):
            pytest.skip("torch.linalg.matrix_norm 2-dim requirement on this build")
        raise

    # WANDA neuron scores shape [n_layers, mlp_dim]
    out_wanda = compute_wanda_neuron_scores(model, calib, calibration_windows=1)
    scores_w = out_wanda["scores"]
    assert scores_w.shape[0] == 2 and scores_w.shape[1] >= 8


def test_blend_neuron_scores_variants():
    a = torch.ones(2, 6)
    b = torch.full((2, 4), 2.0)
    # Different shapes trigger pad/align path
    blended = blend_neuron_scores([a, b], weights=[0.6, 0.4])
    assert blended.shape == (2, 6)
    # Equal shape blend
    c = torch.zeros(2, 6)
    blended2 = blend_neuron_scores([a, c])
    assert torch.allclose(blended2, a * 0.5)

    # Error branches
    with pytest.raises(ValueError):
        blend_neuron_scores([])
    with pytest.raises(ValueError):
        blend_neuron_scores([a], weights=[0.3, 0.7])


def test_post_attention_head_scores_with_dict_batches():
    torch.manual_seed(0)
    model = TinyTransformer(n_layers=1, n_heads=2, hidden=4, vocab=16)
    calib = [{"input_ids": None}, {"inputs": torch.randint(0, 16, (1, 3))}]
    try:
        out = compute_post_attention_head_scores(model, calib, calibration_windows=2)
    except RuntimeError as e:
        if "dim must be a 2-tuple" in str(e):
            pytest.skip("torch.linalg.matrix_norm 2-dim requirement on this build")
        raise
    assert torch.count_nonzero(out["scores"]) > 0


def test_wanda_scores_tensor_outputs_and_dict_inputs():
    class TinyTransformerTensor(TinyTransformer):
        def forward(self, input_ids):
            return super().forward(input_ids).logits

    torch.manual_seed(0)
    model = TinyTransformerTensor(n_layers=1, n_heads=2, hidden=4, vocab=8)
    calib = [
        {"input_ids": None},
        {"input_ids": torch.randint(0, 8, (1, 3))},
    ]
    out = compute_wanda_neuron_scores(model, calib, calibration_windows=2)
    assert torch.count_nonzero(out["scores"]) > 0


def test_post_attention_zero_windows_uses_direct_h_blocks():
    torch.manual_seed(0)
    model = TinyTransformerDirect(n_layers=1, n_heads=2, hidden=4, vocab=16)
    calib = [torch.randint(0, 16, (1, 4))]
    out = compute_post_attention_head_scores(
        model, calib, calibration_windows=0, device="cpu"
    )
    # No samples processed → zero scores but branch executed
    assert torch.count_nonzero(out["scores"]) == 0


def test_wanda_zero_windows_skips_forward_pass():
    torch.manual_seed(0)
    model = TinyTransformer(n_layers=1, n_heads=2, hidden=4, vocab=8)
    calib = [torch.randint(0, 8, (1, 4))]
    out = compute_wanda_neuron_scores(model, calib, calibration_windows=0, device="cpu")
    assert torch.count_nonzero(out["scores"]) == 0


def test_wanda_prefers_logits_namespace_outputs():
    torch.manual_seed(0)
    model = TinyTransformer(n_layers=1, n_heads=2, hidden=4, vocab=8)
    calib = [torch.randint(0, 8, (1, 4))]
    out = compute_wanda_neuron_scores(model, calib, calibration_windows=1, device="cpu")
    assert torch.isfinite(out["scores"]).all()
