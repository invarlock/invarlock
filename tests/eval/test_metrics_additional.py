import math
from types import SimpleNamespace

import pytest
import torch

from invarlock.eval import metrics as M


def test_get_metrics_info_and_validate_env():
    info = M.get_metrics_info()
    assert {"sigma_max", "head_energy", "mi_gini"}.issubset(info["available_metrics"])
    ok = M.validate_metrics_environment()
    assert ok is True


def test_compute_ppl_with_partial_attention_mask():
    class DummyLM:
        def __init__(self, vocab=8):
            self.config = SimpleNamespace(model_type="gpt2")
            self.emb = torch.nn.Embedding(vocab, 4)
            self.lm_head = torch.nn.Linear(4, vocab)

        def parameters(self):  # pragma: no cover - helper for device detection
            yield from self.lm_head.parameters()

        def eval(self):  # pragma: no cover - no-op for interface
            return self

        def __call__(self, input_ids=None, attention_mask=None, return_dict=True):
            B, T = input_ids.shape
            hidden = torch.zeros(B, T, 4)
            logits = self.lm_head(hidden)
            return SimpleNamespace(logits=logits)

    model = DummyLM()
    # Batch with some masked tokens
    batch = {
        "input_ids": torch.ones(1, 6, dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 1, 0, 0, 0]], dtype=torch.long),
    }
    ppl = M.compute_perplexity(model, [batch])
    assert isinstance(ppl, float)
    assert ppl >= 1.0 and math.isfinite(ppl)


def test_mi_gini_cpu_chunk_warning(monkeypatch):
    # Create small fake activations: L x N x D
    L, N, D = 3, 2, 4
    feats = torch.randn(L, N, D)
    targ = torch.randint(0, 5, (N,))

    class StubDep:
        def __init__(self):
            self.available_modules = {"mi_scores": self._fn}

        def is_available(self, name):  # noqa: D401
            return name in self.available_modules

        def get_module(self, name):  # noqa: D401
            return self.available_modules[name]

        @staticmethod
        def _fn(x, y):  # Always raise to trigger warning path
            raise RuntimeError("simulated failure")

    monkeypatch.setattr(M, "DependencyManager", lambda: StubDep())

    cfg = M.MetricsConfig(progress_bars=False)
    out = M._mi_gini_optimized_cpu_path(feats, targ, max_per_layer=10, config=cfg)
    assert isinstance(out, float) and (math.isnan(out) or out >= 0.0)


def test_compute_perplexity_strict_errors_on_no_valid_tokens():
    class DummyLM:
        def __init__(self, vocab=8):
            self.out = torch.nn.Linear(4, vocab)

        def parameters(self):  # pragma: no cover
            yield from self.out.parameters()

        def eval(self):  # pragma: no cover
            return self

        def __call__(self, input_ids=None, attention_mask=None, return_dict=True):
            B, T = input_ids.shape
            logits = self.out(torch.zeros(B, T, 4))
            return SimpleNamespace(logits=logits)

    model = DummyLM()
    # Sequence too short or fully masked -> no valid tokens
    bad_batches = [
        {"input_ids": torch.ones(1, 1, dtype=torch.long)},
        {
            "input_ids": torch.ones(1, 4, dtype=torch.long),
            "attention_mask": torch.zeros(1, 4, dtype=torch.long),
        },
    ]
    for batch in bad_batches:
        with pytest.raises(M.ValidationError):
            _ = M.compute_perplexity(model, [batch])


def test_latency_and_memory_smoke():
    class DummyLM:
        def __init__(self, vocab=8):
            self.out = torch.nn.Linear(4, vocab)

        def parameters(self):
            yield from self.out.parameters()

        def eval(self):  # pragma: no cover
            return self

        def __call__(self, input_ids=None, attention_mask=None, return_dict=True):
            B, T = input_ids.shape
            logits = self.out(torch.zeros(B, T, 4))
            return SimpleNamespace(logits=logits)

    # Create a window with at least one sequence > 10 tokens
    window = SimpleNamespace(
        input_ids=[[1] * 12, [1] * 8], attention_masks=[[1] * 12, [1] * 8]
    )
    model = DummyLM()
    lat = M.measure_latency(model, window)
    mem = M.measure_memory(model, window)
    assert isinstance(lat, float) and lat >= 0.0
    assert isinstance(mem, float) and mem >= 0.0
