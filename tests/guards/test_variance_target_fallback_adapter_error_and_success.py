from types import SimpleNamespace

import torch.nn as nn

from invarlock.guards.variance import VarianceGuard


class TinyNoProj:
    def __init__(self, layers=2):
        # Provide transformer.h without attn/mlp projections so primary pass yields no targets
        self.transformer = SimpleNamespace(h=[SimpleNamespace() for _ in range(layers)])

    def named_modules(self):  # minimal generator
        yield ("root", nn.ReLU())


class AdapterFallback:
    def __init__(self):
        self.calls = 0

    def get_layer_modules(self, model, i):
        self.calls += 1
        if i == 0:
            raise RuntimeError("boom")
        # Provide both attn and mlp projections for fallback
        return {
            "attn.c_proj": nn.Linear(2, 2, bias=False),
            "mlp.c_proj": nn.Linear(2, 2, bias=False),
        }


def test_target_resolution_adapter_error_then_success():
    g = VarianceGuard(
        policy={
            "scope": "both",
            "tap": ["transformer.h.*.attn.c_proj", "transformer.h.*.mlp.c_proj"],
        }
    )
    m = TinyNoProj(layers=2)
    g._prepared = True

    _ = g._resolve_target_modules(m, adapter=AdapterFallback())
    stats = g._stats.get("target_resolution", {})
    assert stats.get("fallback_used") is True
    # Adapter error recorded for first layer
    rejected = stats.get("rejected", {})
    assert any(key.startswith("adapter_error:") for key in rejected.keys()) or any(
        key.startswith("adapter_error") for key in rejected.keys()
    )
    # Fallback matched at least one module
    assert len(stats.get("matched", [])) >= 1
