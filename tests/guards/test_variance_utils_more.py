import numpy as np
import torch
import torch.nn as nn

from invarlock.guards.variance import VarianceGuard


def test_normalize_module_name_variants():
    g = VarianceGuard()
    assert g._normalize_module_name("block3.attn").endswith(
        "transformer.h.3.attn.c_proj"
    )
    assert g._normalize_module_name("transformer.h.2.mlp").endswith(
        "transformer.h.2.mlp.c_proj"
    )
    assert g._normalize_module_name(123) == ""


def test_normalize_pairing_and_expected_window_ids():
    g = VarianceGuard()
    ids = g._normalize_pairing_ids("preview", [1, "x"])
    assert ids == ["preview::1", "preview::x"]
    g._pairing_reference = ids
    assert g._expected_window_ids() == ids


def test_extract_window_ids_list_variant():
    g = VarianceGuard()
    batch = {"window_ids": ["w1", "w2"], "inputs": np.ones((1, 2))}
    out = g._extract_window_ids([batch])
    assert out == ["w1", "w2"]


def test_disable_idempotent_returns_true():
    g = VarianceGuard()
    g._enabled = False
    assert g.disable(nn.Linear(1, 1, bias=False)) is True


def test_predictive_gate_ci_unavailable_reason(monkeypatch):
    # Build guard with scales and batches meeting min_coverage, but force enable() to fail
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer = nn.Module()
            blk = nn.Module()
            blk.attn = nn.Module()
            blk.attn.c_proj = nn.Linear(2, 2, bias=False)
            blk.mlp = nn.Module()
            blk.mlp.c_proj = nn.Linear(2, 2, bias=False)
            self.transformer.h = nn.ModuleList([blk])

        def forward(self, x):
            return self.transformer.h[0].mlp.c_proj(
                self.transformer.h[0].attn.c_proj(x)
            )

    g = VarianceGuard(
        policy={
            "scope": "both",
            "min_gain": 0.0,
            "predictive_gate": True,
            "calibration": {"windows": 2, "min_coverage": 1, "seed": 7},
        }
    )

    # Force non-empty scales and make enable() return False so ppl_with_ve_samples is empty
    def fake_scales(_model, _batches):
        return {"transformer.h.0.mlp.c_proj": 0.95}

    monkeypatch.setattr(g, "_compute_variance_scales", fake_scales)
    monkeypatch.setattr(g, "enable", lambda model: False)

    batches = [torch.ones(1, 2), torch.ones(1, 2)]
    res = g.prepare(M(), adapter=None, calib=batches, policy=None)
    assert isinstance(res, dict)
    pg = getattr(g, "_predictive_gate_state", {})
    # Should record an explanatory reason; CI may be unavailable or computed
    assert pg.get("reason") in {
        "ci_unavailable",
        "ve_enable_failed",
        "disabled",
        "no_scales",
        "insufficient_coverage",
        "not_evaluated",
    }
    if "delta_ci" in pg and pg["delta_ci"] is not None:
        dc = pg.get("delta_ci")
        assert isinstance(dc, tuple) and len(dc) == 2
