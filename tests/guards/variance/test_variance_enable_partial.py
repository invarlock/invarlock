import torch
import torch.nn as nn

from invarlock.guards.variance import VarianceGuard


class TinyBlock(nn.Module):
    def __init__(self, d=4):
        super().__init__()
        self.attn = nn.Module()
        self.attn.c_proj = nn.Linear(d, d, bias=False)
        self.mlp = nn.Module()
        self.mlp.c_proj = nn.Linear(d, d, bias=False)


class TinyModel(nn.Module):
    def __init__(self, d=4):
        super().__init__()
        self.transformer = nn.Module()
        self.transformer.h = nn.ModuleList([TinyBlock(d)])


def test_enable_rejects_partial_target_coverage_without_mutation():
    model = TinyModel()
    g = VarianceGuard(policy={"scope": "both", "min_gain": 0.0, "max_calib": 0})

    targets = g._resolve_target_modules(model, adapter=None)
    assert targets
    g._prepared = True
    g._target_modules = targets

    # Use one valid name and one bogus to force a partial success path
    good_name = next(iter(targets.keys()))
    bad_name = good_name + ".bogus"
    g._scales = {good_name: 1.1, bad_name: 0.9}

    # Keep a copy of original weights
    before = {}
    for name, module in targets.items():
        before[name] = module.weight.detach().clone()

    assert g.enable(model) is False
    assert g._enabled is False
    assert g.disable(model) is True

    # The valid target was never partially mutated.
    restored = targets[good_name].weight.detach()
    assert torch.equal(restored, before[good_name])
