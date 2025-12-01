from __future__ import annotations

import torch.nn as nn

from invarlock.guards.rmt import (
    analyze_weight_distribution,
    capture_baseline_mp_stats,
    mp_bulk_edges,
    rmt_detect_report,
    rmt_detect_with_names,
    rmt_growth_ratio,
    within_deadband,
)


def test_mp_bulk_edges_whitened_and_unwhitened() -> None:
    # Two branches: whitened and not
    lo_w, hi_w = mp_bulk_edges(10, 20, whitened=True)
    lo_u, hi_u = mp_bulk_edges(10, 20, whitened=False)
    assert hi_w > 0 and hi_u > hi_w  # unwhitened scales by sqrt(m)
    assert lo_w >= 0 and lo_u >= 0


def test_growth_and_deadband_boundaries() -> None:
    # Growth ratio stable at 1.0 when identical
    r = rmt_growth_ratio(2.0, 1.0, 2.0, 1.0)
    assert abs(r - 1.0) < 1e-12
    # Deadband boundary inclusive
    assert within_deadband(1.10, 1.0, 0.10)
    assert not within_deadband(1.1000001, 1.0, 0.10)


def test_capture_baseline_mp_stats_importerror_branch() -> None:
    # Simple model without transformers Conv1D available should exercise ImportError path
    model = nn.Sequential(nn.Linear(8, 4), nn.ReLU(), nn.Linear(4, 2))
    stats = capture_baseline_mp_stats(model)
    assert isinstance(stats, dict)


def test_analyze_weight_distribution_paths() -> None:
    model = nn.Sequential(nn.Linear(6, 3), nn.ReLU(), nn.Linear(3, 2))
    out = analyze_weight_distribution(model)
    # Either empty (if no 2D weights detected due to system) or populated with core keys
    if out:
        assert {"mean", "std", "min", "max", "sparsity"}.issubset(out.keys())


def test_rmt_detect_wrappers_smoke() -> None:
    model = nn.Sequential(nn.Linear(6, 3), nn.ReLU(), nn.Linear(3, 2))
    summary, per_layer = rmt_detect_report(model, threshold=10.0)
    assert isinstance(summary, dict)
    assert isinstance(per_layer, list)
    named = rmt_detect_with_names(model, threshold=10.0)
    assert isinstance(named, dict)
