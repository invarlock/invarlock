from __future__ import annotations

import torch
import torch.nn as nn

from invarlock.guards import rmt as R


class TinyBlock(nn.Module):
    def __init__(self, in_f=4, out_f=4):
        super().__init__()
        self.attn = nn.Module()
        self.attn.c_attn = nn.Linear(in_f, out_f, bias=False)
        self.attn.c_proj = nn.Linear(out_f, out_f, bias=False)
        self.mlp = nn.Module()
        self.mlp.c_fc = nn.Linear(out_f, out_f, bias=False)
        self.mlp.c_proj = nn.Linear(out_f, out_f, bias=False)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Build hierarchy to yield names like transformer.h.0.attn.c_attn
        self.transformer = nn.Module()
        self.transformer.h = nn.ModuleList([TinyBlock()])


def test_capture_baseline_mp_stats_importerror_branch_and_stats_present():
    model = TinyModel()
    stats = R.capture_baseline_mp_stats(model)
    # Should collect stats for linear layers with allowed suffixes
    assert isinstance(stats, dict)
    # Expect at least one entry (e.g., first block attn.c_attn)
    assert any(
        k.endswith((".attn.c_attn", ".attn.c_proj", ".mlp.c_fc", ".mlp.c_proj"))
        for k in stats
    )


def test_rmt_detect_report_smoke():
    model = TinyModel()
    summary, per_layer = R.rmt_detect_report(
        model, threshold=10.0
    )  # high threshold → likely no outliers
    assert isinstance(summary, dict) and isinstance(per_layer, list)
    assert "has_outliers" in summary and "max_ratio" in summary


def test_rmt_detect_verbose_flags_outliers():
    model = TinyModel()
    # Set a very low threshold so any deviation is flagged
    res = R.rmt_detect(model, threshold=1.0, verbose=True)
    assert isinstance(res, dict)
    assert "has_outliers" in res


def test_analyze_weight_distribution_paths():
    model = TinyModel()
    stats = R.analyze_weight_distribution(model)
    assert (
        isinstance(stats, dict) and "histogram" in stats and "singular_values" in stats
    )

    # Empty-model path returns {}
    empty = nn.Module()
    assert R.analyze_weight_distribution(empty) == {}


def test_rmt_detect_with_names_encoder_branch():
    # Build model exposing encoder.layer path
    class EncoderModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Module()
            self.encoder.layer = nn.ModuleList([TinyBlock()])

    em = EncoderModel()
    out = R.rmt_detect_with_names(em, threshold=10.0, verbose=False)
    assert isinstance(out, dict) and "per_layer" in out


def test_rmt_detect_with_names_fallback_heuristic():
    # Model not matching transformer/encoder patterns but with attn/mlp attrs
    class OddLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = nn.Module()
            self.attn.c_proj = nn.Linear(2, 2, bias=False)
            self.mlp = nn.Module()
            self.mlp.c_fc = nn.Linear(2, 2, bias=False)

    class OddModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.block = OddLayer()

        def named_modules(self, memo=None, prefix=""):
            # Yield the odd layer to simulate fallback discovery
            yield ("odd", self.block)

    m = OddModel()
    out = R.rmt_detect_with_names(m, threshold=10.0, verbose=False)
    assert isinstance(out, dict)


def test_clip_full_svd_nonfinite_short_circuits():
    W = torch.tensor([[float("nan"), 0.0], [0.0, 1.0]])
    # Returns original (non-components)
    out = R.clip_full_svd(W, clip_val=1.0, return_components=False)
    assert out is W
    # Returns (None,None,None) when components requested
    U, S, Vt = R.clip_full_svd(W, clip_val=1.0, return_components=True)
    assert U is None and S is None and Vt is None


def test_small_helpers_deadband_and_growth_ratio():
    # within_deadband true/false branches
    assert R.within_deadband(1.0, 1.0, 0.0) is True
    assert R.within_deadband(1.2, 1.0, 0.1) is False
    # growth ratio numeric path
    assert R.rmt_growth_ratio(2.0, 1.0, 1.0, 1.0) == 2.0


def test_clip_full_svd_components_return():
    W = torch.randn(3, 2)
    U, S, Vt = R.clip_full_svd(W, clip_val=10.0, return_components=True)
    assert U is not None and S is not None and Vt is not None


def test_layer_svd_stats_exception_path(monkeypatch):
    # Force svdvals to raise to trigger exception handling branch in layer_svd_stats
    import torch.linalg as tla

    original = tla.svdvals

    def _boom(x):
        raise RuntimeError("svd fail")

    monkeypatch.setattr(tla, "svdvals", _boom)
    try:
        layer = nn.Linear(2, 2, bias=False)
        out = R.layer_svd_stats(layer)
        assert isinstance(out, dict) and "worst_ratio" in out
    finally:
        monkeypatch.setattr(tla, "svdvals", original)
