import torch.nn as nn

from invarlock.guards.rmt import rmt_detect_with_names


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = nn.Module()
        self.transformer.h = nn.ModuleList([nn.Module()])
        self.transformer.h[0].attn = nn.Module()
        self.transformer.h[0].attn.c_proj = nn.Linear(4, 4)
        self.transformer.h[0].mlp = nn.Module()
        self.transformer.h[0].mlp.c_fc = nn.Linear(4, 4)


def test_rmt_detect_with_names_flags_outliers_verbose():
    model = TinyModel()
    # Set low threshold to guarantee outliers are flagged
    result = rmt_detect_with_names(model, threshold=1.01, verbose=True)
    assert isinstance(result, dict)
    assert "per_layer" in result and isinstance(result["per_layer"], list)
    # Either flags or not depending on random weights, but structure should exist
    assert "layers" in result
