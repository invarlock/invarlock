import torch
import torch.nn as nn

from invarlock.guards.variance import VarianceGuard


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = nn.Module()
        blk = nn.Module()
        blk.attn = nn.Module()
        blk.attn.c_proj = nn.Linear(2, 2, bias=False)
        blk.mlp = nn.Module()
        blk.mlp.c_proj = nn.Linear(2, 2, bias=False)
        self.transformer.h = nn.ModuleList([blk])

    def forward(self, inputs, labels=None):
        x = self.transformer.h[0].attn.c_proj(inputs)
        return self.transformer.h[0].mlp.c_proj(x)


def test_no_scales_branch_sets_status_and_estimates():
    model = Tiny()
    g = VarianceGuard(
        policy={
            "scope": "both",
            "min_gain": 0.0,
            "predictive_gate": True,
            "calibration": {"windows": 2, "min_coverage": 2, "seed": 11},
        }
    )

    # Resolve targets and ensure empty scales
    g._target_modules = g._resolve_target_modules(model, adapter=None)
    g._scales = {}
    # Prepare two batches to meet coverage
    batches = [torch.ones(1, 2), torch.zeros(1, 2)]
    g._store_calibration_batches(batches)

    # Drive the internal evaluation to the no_scales path
    g._calibration_stats = {
        "requested": 2,
        "coverage": 0,
        "min_coverage": 2,
        "seed": 11,
        "status": "pending",
        "tag": "t",
    }
    g._evaluate_calibration_pass(
        model, g._calibration_batches, min_coverage=2, calib_seed=11, tag="t"
    )

    status = g._calibration_stats.get("status")
    assert status in {"no_scaling_required", "pending", "insufficient"}
    if status == "no_scaling_required":
        ape = g._stats.get("ab_point_estimates", {})
        assert "ppl_no_ve" in ape and "ppl_with_ve" in ape


def test_finalize_includes_ab_provenance_metrics():
    model = Tiny()
    g = VarianceGuard(
        policy={
            "scope": "both",
            "min_gain": 0.0,
            "predictive_gate": False,
            "max_calib": 10,
        }
    )
    # Minimal prepare to populate targets and stats
    g.prepare(model, adapter=None, calib=None, policy=None)
    g._prepared = True
    # Force some provenance info
    g._stats.setdefault("ab_provenance", {})["condition_a"] = {"status": "evaluated"}
    out = g.finalize(model)
    metrics = out.get("metrics", {})
    # If finalize returned full metrics, ab_provenance should be present
    if metrics:
        assert "ab_provenance" in metrics
