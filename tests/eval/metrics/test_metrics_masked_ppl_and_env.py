import math
from types import SimpleNamespace

import torch
import torch.nn as nn

from invarlock.eval.metrics import (
    MetricsConfig,
    compute_ppl,
    get_metrics_info,
    validate_metrics_environment,
)
from invarlock.eval.metrics_activation import _mi_gini_optimized_cpu_path


class DummyLM(nn.Module):
    def __init__(self, vocab=16):
        super().__init__()
        self.vocab = vocab
        self.emb = nn.Embedding(vocab, 8)
        self.lm = nn.Linear(8, vocab)

    def forward(self, input_ids=None, attention_mask=None, return_dict=True):
        x = self.emb(input_ids)
        logits = self.lm(x)
        if return_dict:
            return SimpleNamespace(logits=logits)
        return (logits,)


def test_compute_ppl_with_partial_attention_mask():
    model = DummyLM()
    # Two samples, with some tokens masked out
    input_ids = [
        [1, 2, 3, 4, 5, 6],
        [7, 8, 9, 10, 11, 12],
    ]
    attention_masks = [
        [1, 1, 1, 0, 0, 0],  # partially masked
        [1, 1, 1, 1, 0, 0],
    ]
    window = SimpleNamespace(input_ids=input_ids, attention_masks=attention_masks)
    ppl = compute_ppl(model, window=window, device="cpu")
    assert isinstance(ppl, float) and math.isfinite(ppl) and ppl >= 1.0


def test_metrics_env_info_and_validation():
    info = get_metrics_info()
    assert isinstance(info, dict) and "available_metrics" in info
    report = validate_metrics_environment()
    assert report.ok is True


def test_mi_gini_cpu_chunk_warnings():
    def fail_score(_x, _y):
        raise RuntimeError("layer-wise failure")

    # Build shapes consistent with CPU path: [L, N, D] and targets [N]
    L, N, D = 3, 5, 4
    feats_cpu = torch.randn(L, N, D)
    targ_cpu = torch.randint(0, 10, (N,))
    cfg = MetricsConfig()
    val = _mi_gini_optimized_cpu_path(
        feats_cpu,
        targ_cpu,
        max_per_layer=10,
        config=cfg,
        mi_scores_fn=fail_score,
    )
    # Function should not crash and should produce a float (likely NaN)
    assert isinstance(val, float) and (math.isnan(val) or math.isfinite(val))
