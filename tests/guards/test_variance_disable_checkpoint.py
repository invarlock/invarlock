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


def test_disable_uses_checkpoint_restoration_when_available():
    model = TinyModel()
    g = VarianceGuard(policy={"scope": "both"})

    targets = g._resolve_target_modules(model, adapter=None)
    g._target_modules = targets

    # Capture original and push a checkpoint
    before = {name: mod.weight.detach().clone() for name, mod in targets.items()}
    g._push_checkpoint(model)
    g._enabled = True

    # Mutate weights so that disable must restore
    for mod in targets.values():
        mod.weight.data.add_(1.0)

    assert g.disable(model) is True
    # After checkpoint restoration, weights equal original
    for name, mod in targets.items():
        assert (mod.weight.detach() == before[name]).all()
