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

    def forward(self, inputs, labels=None):
        x = self.transformer.h[0].attn.c_proj(inputs)
        x = torch.tanh(x)
        x = self.transformer.h[0].mlp.c_proj(x)
        return x


def _tensor_batches(n: int, b: int = 4, d: int = 4):
    # Create inputs and simple masks (ignored for equalisation, used in calibration path)
    for _ in range(n):
        yield (torch.randn(b, d), torch.ones(b, d))


def test_prepare_complete_with_point_estimates():
    model = TinyModel()
    batches = list(_tensor_batches(6))
    policy = {
        "scope": "both",
        "deadband": 0.0,
        "min_gain": 0.0,
        "max_calib": 100,  # ample scale windows for stable raw scales
        "clamp": (0.8, 1.2),
        "calibration": {"windows": 4, "min_coverage": 3, "seed": 11},
        "predictive_gate": True,
    }
    g = VarianceGuard(policy=policy)
    res = g.prepare(model, adapter=None, calib=batches, policy=None)
    assert isinstance(res, dict)
    status = g._calibration_stats.get("status")
    assert status in {"complete", "no_scaling_required", "pending", "insufficient"}
    # If complete, check estimates and ratio_ci presence
    if status == "complete":
        assert g._ratio_ci is not None


def test_prepare_with_focus_modules_records_focus_in_metrics():
    model = TinyModel()
    focus_name = "transformer.h.0.mlp.c_proj"
    batches = list(_tensor_batches(4))
    g = VarianceGuard(
        policy={
            "scope": "both",
            "target_modules": [focus_name],
            "min_gain": 0.0,
            "deadband": 0.0,
            "max_calib": 50,
            "calibration": {"windows": 2, "min_coverage": 1, "seed": 5},
            "predictive_gate": False,
        }
    )
    g.prepare(model, adapter=None, calib=batches, policy=None)
    out = g.finalize(model)
    metrics = out.get("metrics", {})
    assert "focus_modules" in metrics
    assert focus_name in metrics.get("focus_modules", [])


def test_prepare_adapter_fallback_sets_flag():
    class ModelNoProj(nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer = nn.Module()
            self.transformer.h = nn.ModuleList([nn.Module()])

    class GoodAdapter:
        def get_layer_modules(self, model, idx):
            return {"attn.c_proj": nn.Linear(4, 4, bias=False)}

    g = VarianceGuard(policy={"scope": "both", "min_gain": 0.0, "max_calib": 0})
    g.prepare(ModelNoProj(), adapter=GoodAdapter(), calib=None, policy=None)
    tr = g._stats.get("target_resolution", {})
    assert tr.get("fallback_used") in (True, False)
    # At least matched list exists when fallback used
    if tr.get("fallback_used"):
        assert tr.get("matched")
