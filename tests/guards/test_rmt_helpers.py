import torch
import torch.nn as nn

from invarlock.guards.rmt import (
    _iter_transformer_layers,
    capture_baseline_mp_stats,
    layer_svd_stats,
    mp_bulk_edge,
    mp_bulk_edges,
    rmt_detect_report,
    rmt_growth_ratio,
    within_deadband,
)


def test_mp_bulk_edges_and_edge_basic():
    lo, hi = mp_bulk_edges(16, 32, whitened=True)
    assert hi > 0 and lo >= 0
    edge = mp_bulk_edge(16, 32, whitened=False)
    assert edge > 0


class TinyLayer(nn.Module):
    def __init__(self, d=4):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(d, d))


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Transformer-like container structure
        self.transformer = nn.Module()
        self.transformer.h = nn.ModuleList([nn.Module(), nn.Module()])
        # Populate each with a linear weight param
        self.transformer.h[0].attn = nn.Module()
        self.transformer.h[0].attn.c_proj = nn.Linear(4, 4)
        self.transformer.h[0].mlp = nn.Module()
        self.transformer.h[0].mlp.c_fc = nn.Linear(4, 4)
        self.transformer.h[1].attn = nn.Module()
        self.transformer.h[1].attn.c_proj = nn.Linear(4, 4)
        self.transformer.h[1].mlp = nn.Module()
        self.transformer.h[1].mlp.c_fc = nn.Linear(4, 4)


def test_layer_svd_stats_fallback_98th_percentile():
    layer = nn.Linear(8, 8)
    stats = layer_svd_stats(layer)
    assert "sigma_max" in stats and "worst_ratio" in stats
    assert stats["sigma_max"] >= 0.0


def test_rmt_detect_report_smoke():
    model = TinyModel()
    summary, per_layer = rmt_detect_report(model, threshold=10.0)
    assert "has_outliers" in summary
    assert isinstance(per_layer, list)


def test_capture_baseline_and_helpers():
    model = TinyModel()
    stats = capture_baseline_mp_stats(model)
    assert isinstance(stats, dict)
    # growth ratio and deadband helpers
    assert rmt_growth_ratio(2.0, 1.0, 1.0, 1.0) >= 2.0
    assert within_deadband(1.0, 1.0, 0.1)


def test_iter_layers_fallback():
    class Fallback(nn.Module):
        def __init__(self):
            super().__init__()
            self.block = nn.Module()
            self.block.attn = nn.Module()
            self.block.attn.c_proj = nn.Linear(4, 4)
            self.block.mlp = nn.Module()
            self.block.mlp.c_fc = nn.Linear(4, 4)

        def modules(self):  # pragma: no cover - satisfy hasattr check
            return super().modules()

    f = Fallback()
    layers = list(_iter_transformer_layers(f))
    # Fallback yields modules having attn/mlp
    assert layers


def test_iter_transformer_layers_handles_iteration_error() -> None:
    class _BadIterable:
        def __len__(self) -> int:
            return 1

        def __iter__(self):
            raise TypeError("boom")

    class _Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.transformer = nn.Module()
            self.transformer.h = _BadIterable()

    layers = list(_iter_transformer_layers(_Model()))
    assert layers == []


def test_iter_transformer_layers_transformer_h_without_iter_len() -> None:
    class _Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.transformer = nn.Module()
            self.transformer.h = object()

    layers = list(_iter_transformer_layers(_Model()))
    assert layers == []
