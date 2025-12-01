import torch.nn as nn

from invarlock.guards.variance import VarianceGuard


class TinyBlockNoProj(nn.Module):
    def __init__(self):
        super().__init__()
        # attn and mlp exist but without c_proj to force fallback
        self.attn = nn.Module()
        self.mlp = nn.Module()


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = nn.Module()
        self.transformer.h = nn.ModuleList([TinyBlockNoProj()])


def test_adapter_fallback_resolves_targets():
    model = TinyModel()
    g = VarianceGuard(
        policy={
            "scope": "both",
            "tap": ["transformer.h.*.attn.c_proj", "transformer.h.*.mlp.c_proj"],
            "min_gain": 0.0,
            "max_calib": 0,
        }
    )

    class GoodAdapter:
        def get_layer_modules(self, model, idx):
            return {
                "attn.c_proj": nn.Linear(4, 4, bias=False),
                "mlp.c_proj": nn.Linear(4, 4, bias=False),
            }

    targets = g._resolve_target_modules(model, adapter=GoodAdapter())
    assert targets
    # Stats should record fallback used and matched names
    tr = g._stats.get("target_resolution", {})
    assert tr.get("fallback_used") is True
    matched = set(tr.get("matched") or [])
    assert any(name.endswith("attn.c_proj") for name in matched)
    assert any(name.endswith("mlp.c_proj") for name in matched)
