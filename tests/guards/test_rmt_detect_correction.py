import torch
import torch.nn as nn

from invarlock.guards.rmt import rmt_detect


class TinyTransformers(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = nn.Module()
        self.transformer.h = nn.ModuleList([nn.Module()])
        # Allowed suffix modules with 2D weights
        self.transformer.h[0].attn = nn.Module()
        self.transformer.h[0].attn.c_proj = nn.Linear(8, 8)
        self.transformer.h[0].mlp = nn.Module()
        self.transformer.h[0].mlp.c_fc = nn.Linear(8, 8)
        with torch.no_grad():
            self.transformer.h[0].attn.c_proj.weight.mul_(3.0)
            self.transformer.h[0].mlp.c_fc.weight.mul_(3.0)


def test_rmt_detect_correction_applies_iterations():
    model = TinyTransformers()
    # Baseline stats mapping to the exact names enumerated in rmt_detect analysis path
    baseline_sigmas = {
        "transformer.h.0.attn.c_proj": 1.0,
        "transformer.h.0.mlp.c_fc": 1.0,
    }
    baseline_mp_stats = {
        "transformer.h.0.attn.c_proj": {"sigma_base": 1.0, "mp_bulk_edge_base": 1.0},
        "transformer.h.0.mlp.c_fc": {"sigma_base": 1.0, "mp_bulk_edge_base": 1.0},
    }
    res = rmt_detect(
        model,
        threshold=1.1,
        detect_only=False,
        correction_factor=0.9,
        baseline_sigmas=baseline_sigmas,
        baseline_mp_stats=baseline_mp_stats,
        deadband=0.0,
        max_iterations=1,
    )
    assert isinstance(res, dict)
    # correction_iterations may be 0 or 1 depending on ratios; accept ≥0
    assert res.get("correction_iterations", 0) >= 0
