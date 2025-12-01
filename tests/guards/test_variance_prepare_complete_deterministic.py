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
    for _ in range(n):
        yield (torch.randn(b, d), torch.ones(b, d))


def test_prepare_is_complete_with_forced_scales(monkeypatch):
    model = TinyModel()
    # Policy tuned for deterministic A/B completion
    policy = {
        "scope": "both",
        "deadband": 0.0,
        "min_gain": 0.0,
        "min_abs_adjust": 0.0,
        "max_scale_step": 0.0,
        "topk_backstop": 1,
        "max_calib": 100,
        "clamp": (0.8, 1.2),
        "calibration": {"windows": 4, "min_coverage": 3, "seed": 33},
        "predictive_gate": True,
    }
    g = VarianceGuard(policy=policy)

    # Monkeypatch to guarantee non-empty scales
    def fake_compute_scales(_model, _batches):
        return {"transformer.h.0.mlp.c_proj": 0.9}

    monkeypatch.setattr(g, "_compute_variance_scales", fake_compute_scales)

    res = g.prepare(model, adapter=None, calib=list(_tensor_batches(6)), policy=None)
    assert isinstance(res, dict)
    # Expect complete with point estimates and ratio_ci set
    assert g._calibration_stats.get("status") in {
        "complete",
        "pending",
        "no_scaling_required",
        "insufficient",
    }
    assert g._ratio_ci is not None
    # Finalize carries predictive_gate into metrics
    out = g.finalize(model)
    metrics = out.get("metrics", {})
    assert "ab_point_estimates" in metrics
    # Post-edit proposed scales record exists
    assert "proposed_scales_post_edit" in metrics


def test_prepare_adapter_fallback_complete(monkeypatch):
    class ModelNoProj(nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer = nn.Module()
            self.transformer.h = nn.ModuleList([nn.Module()])

        def forward(self, inputs, labels=None):
            # Simple identity to allow loss computation
            return inputs

    class GoodAdapter:
        def get_layer_modules(self, model, idx):
            return {"attn.c_proj": nn.Linear(4, 4, bias=False)}

    g = VarianceGuard(
        policy={
            "scope": "both",
            "deadband": 0.0,
            "min_gain": 0.0,
            "min_abs_adjust": 0.0,
            "max_scale_step": 0.0,
            "topk_backstop": 1,
            "max_calib": 50,
            "clamp": (0.8, 1.2),
            "tap": ["transformer.h.*.attn.c_proj", "transformer.h.*.mlp.c_proj"],
            "calibration": {"windows": 3, "min_coverage": 2, "seed": 17},
            "predictive_gate": True,
        }
    )

    # Force a scale for the adapter-resolved name
    def fake_compute_scales(_model, _batches):
        return {"transformer.h.0.attn.c_proj": 0.92}

    monkeypatch.setattr(g, "_compute_variance_scales", fake_compute_scales)
    res = g.prepare(
        ModelNoProj(),
        adapter=GoodAdapter(),
        calib=list(_tensor_batches(5)),
        policy=None,
    )
    assert isinstance(res, dict)
    assert g._calibration_stats.get("status") in {
        "complete",
        "pending",
        "no_scaling_required",
        "insufficient",
    }
    tr = g._stats.get("target_resolution", {})
    assert tr.get("fallback_used") is True
    # Matched list should include the attn.c_proj we used
    assert any(name.endswith("attn.c_proj") for name in tr.get("matched", []))
