"""
Tests for variance guard scale-to-target module filtering.

When scope limits which target modules are resolved (e.g., scope=ffn only includes MLP modules),
the scales computed by equalise_residual_variance must be filtered to only include those
that have corresponding target modules. Otherwise, enable() will fail because it can't
find modules to scale.
"""

import torch
import torch.nn as nn

import invarlock.guards.variance as var_mod
from invarlock.guards.variance import VarianceGuard


class TinyBlock(nn.Module):
    def __init__(self, d=4):
        super().__init__()
        self.attn = nn.Module()
        self.attn.c_proj = nn.Linear(d, d, bias=False)
        self.mlp = nn.Module()
        self.mlp.c_proj = nn.Linear(d, d, bias=False)


class TinyModel(nn.Module):
    def __init__(self, n_layers=2, d=4):
        super().__init__()
        self.transformer = nn.Module()
        self.transformer.h = nn.ModuleList([TinyBlock(d) for _ in range(n_layers)])

    def forward(self, x):
        out = x
        for blk in self.transformer.h:
            out = blk.mlp.c_proj(blk.attn.c_proj(out))
        return out


class TestScaleTargetFiltering:
    """Test that scales are properly filtered to match target modules."""

    def test_scope_ffn_filters_attn_scales(self, monkeypatch):
        """With scope=ffn, scales for attn modules should be filtered out."""
        model = TinyModel(n_layers=2)

        # Create guard with scope=ffn (only MLP modules targeted)
        g = VarianceGuard(
            policy={
                "scope": "ffn",
                "min_gain": 0.0,
                "max_calib": 100,
                "deadband": 0.0,
                "min_abs_adjust": 0.0,
                "tap": "transformer.h.*.mlp.c_proj",
            }
        )

        # Resolve target modules - should only have mlp modules
        targets = g._resolve_target_modules(model)
        assert all("mlp" in name for name in targets.keys())
        assert not any("attn" in name for name in targets.keys())
        g._target_modules = targets

        # Mock equalise_residual_variance to return scales for both attn and mlp
        def fake_eq(*_, **__):
            return {
                "block0.attn": 0.95,
                "block0.mlp": 1.05,
                "block1.attn": 0.97,
                "block1.mlp": 1.03,
            }

        monkeypatch.setattr(var_mod, "equalise_residual_variance", fake_eq)

        # Compute scales - should filter out attn scales
        batches = [{"input_ids": torch.randint(0, 100, (1, 8))}]
        scales = g._compute_variance_scales(model, batches)

        # Only mlp scales should remain
        assert len(scales) == 2
        assert all(
            "mlp" in key or key.startswith("block") and "mlp" in key
            for key in scales.keys()
        )
        assert not any("attn" in key for key in scales.keys())

    def test_scope_attn_filters_mlp_scales(self, monkeypatch):
        """With scope=attn, scales for mlp modules should be filtered out."""
        model = TinyModel(n_layers=2)

        # Create guard with scope=attn (only attn modules targeted)
        g = VarianceGuard(
            policy={
                "scope": "attn",
                "min_gain": 0.0,
                "max_calib": 100,
                "deadband": 0.0,
                "min_abs_adjust": 0.0,
                "tap": "transformer.h.*.attn.c_proj",
            }
        )

        # Resolve target modules - should only have attn modules
        targets = g._resolve_target_modules(model)
        assert all("attn" in name for name in targets.keys())
        assert not any("mlp" in name for name in targets.keys())
        g._target_modules = targets

        # Mock equalise_residual_variance to return scales for both attn and mlp
        def fake_eq(*_, **__):
            return {
                "block0.attn": 0.95,
                "block0.mlp": 1.05,
                "block1.attn": 0.97,
                "block1.mlp": 1.03,
            }

        monkeypatch.setattr(var_mod, "equalise_residual_variance", fake_eq)

        # Compute scales - should filter out mlp scales
        batches = [{"input_ids": torch.randint(0, 100, (1, 8))}]
        scales = g._compute_variance_scales(model, batches)

        # Only attn scales should remain
        assert len(scales) == 2
        assert all(
            "attn" in key or key.startswith("block") and "attn" in key
            for key in scales.keys()
        )
        assert not any("mlp" in key for key in scales.keys())

    def test_scope_both_keeps_all_scales(self, monkeypatch):
        """With scope=both, all scales should be kept."""
        model = TinyModel(n_layers=2)

        # Create guard with scope=both
        g = VarianceGuard(
            policy={
                "scope": "both",
                "min_gain": 0.0,
                "max_calib": 100,
                "deadband": 0.0,
                "min_abs_adjust": 0.0,
                "tap": "transformer.h.*.*.*",  # Match all projections
            }
        )

        # Resolve target modules - should have both attn and mlp
        targets = g._resolve_target_modules(model)
        assert any("attn" in name for name in targets.keys())
        assert any("mlp" in name for name in targets.keys())
        g._target_modules = targets

        # Mock equalise_residual_variance to return scales for both
        def fake_eq(*_, **__):
            return {
                "block0.attn": 0.95,
                "block0.mlp": 1.05,
                "block1.attn": 0.97,
                "block1.mlp": 1.03,
            }

        monkeypatch.setattr(var_mod, "equalise_residual_variance", fake_eq)

        # Compute scales - should keep all
        batches = [{"input_ids": torch.randint(0, 100, (1, 8))}]
        scales = g._compute_variance_scales(model, batches)

        # All scales should remain
        assert len(scales) == 4

    def test_scale_matches_target_helper(self):
        """Test the _scale_matches_target helper method."""
        g = VarianceGuard()

        # Direct match via normalization
        assert g._scale_matches_target("block0.mlp", "transformer.h.0.mlp.c_proj")
        assert g._scale_matches_target("block1.attn", "transformer.h.1.attn.c_proj")

        # Non-match: different layer
        assert not g._scale_matches_target("block0.mlp", "transformer.h.1.mlp.c_proj")

        # Non-match: different component
        assert not g._scale_matches_target("block0.mlp", "transformer.h.0.attn.c_proj")

        # Non-match: invalid format
        assert not g._scale_matches_target("invalid_name", "transformer.h.0.mlp.c_proj")

    def test_enable_succeeds_with_filtered_scales(self, monkeypatch):
        """After filtering, enable() should successfully apply only matching scales."""
        model = TinyModel(n_layers=1)

        # Create guard with scope=ffn
        g = VarianceGuard(
            policy={
                "scope": "ffn",
                "min_gain": 0.0,
                "max_calib": 100,
                "deadband": 0.0,
                "min_abs_adjust": 0.0,
                "tap": "transformer.h.*.mlp.c_proj",
            }
        )

        # Resolve target modules
        g._target_modules = g._resolve_target_modules(model)

        # Mock equalise_residual_variance to return scales for both (simulating pipeline behavior)
        def fake_eq(*_, **__):
            return {"block0.attn": 0.95, "block0.mlp": 1.05}

        monkeypatch.setattr(var_mod, "equalise_residual_variance", fake_eq)

        # Prepare - this should filter scales to only mlp
        batches = [{"input_ids": torch.randint(0, 100, (1, 8))}]
        g._scales = g._compute_variance_scales(model, batches)

        # Only mlp scale should remain
        assert len(g._scales) == 1
        assert "mlp" in list(g._scales.keys())[0] or "block0.mlp" in g._scales

        # Now enable should succeed
        g._prepared = True
        result = g.enable(model)

        # Should succeed since there's a matching target module for the scale
        assert result is True
        assert g._enabled is True

        # Cleanup
        g.disable(model)
