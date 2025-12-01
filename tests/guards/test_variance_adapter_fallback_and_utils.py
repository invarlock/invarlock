import torch
import torch.nn as nn

from invarlock.guards.variance import VarianceGuard


def test_resolve_target_modules_adapter_fallback_success():
    class Proj(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(2, 2))

    class Block(nn.Module):
        def __init__(self):
            super().__init__()

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer = nn.Module()
            self.transformer.h = nn.ModuleList([Block()])

    class Adapter:
        def get_layer_modules(self, model, idx):
            return {
                "attn.c_proj": Proj(),
                "mlp.c_proj": Proj(),
            }

    g = VarianceGuard(policy={"scope": "both", "tap": ["transformer.h.*.*.c_proj"]})
    targets = g._resolve_target_modules(Model(), adapter=Adapter())
    # Should have matched via adapter fallback
    assert any(k.endswith("attn.c_proj") for k in targets.keys())
    assert any(k.endswith("mlp.c_proj") for k in targets.keys())
    stats = g._stats.get("target_resolution", {})
    assert stats.get("fallback_used") is True


def test_normalize_module_name_and_pairing_ids():
    g = VarianceGuard()
    # block.* path normalization
    assert g._normalize_module_name("block3.attn") == "transformer.h.3.attn.c_proj"
    assert g._normalize_module_name("block2.mlp") == "transformer.h.2.mlp.c_proj"
    # transformer.h.* already normalized
    assert g._normalize_module_name("transformer.h.1.mlp.c_proj").endswith("mlp.c_proj")
    # pairing ids normalization preserves prefixes
    ids = g._normalize_pairing_ids("preview", [1, "2", "preview::x"])  # type: ignore[list-item]
    assert ids[0].startswith("preview::") and ids[-1] == "preview::x"
