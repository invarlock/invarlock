import torch.nn as nn

from invarlock.eval.metrics import _locate_transformer_blocks_enhanced


class M1(nn.Module):
    def __init__(self):
        super().__init__()
        self.h = nn.ModuleList([nn.Linear(2, 2, bias=False)])


class M2(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = nn.Module()
        self.base_model.h = nn.ModuleList([nn.Linear(2, 2, bias=False)])


class M3(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.h = nn.ModuleList([nn.Linear(2, 2, bias=False)])


def test_locate_transformer_blocks_patterns():
    for m in (M1(), M2(), M3()):
        blocks = _locate_transformer_blocks_enhanced(m)
        assert blocks is not None and len(blocks) >= 1
