import pytest
import torch.nn as nn

from invarlock.guards.rmt import (
    create_custom_rmt_policy,
    get_rmt_policy,
    layer_svd_stats,
    mp_bulk_edge,
    mp_bulk_edges,
    rmt_growth_ratio,
    within_deadband,
)


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        # simple linear to ensure 2D weight exists
        self.lin = nn.Linear(4, 3, bias=False)

    def forward(self, x):
        return self.lin(x)


def test_mp_edges_and_edge():
    lo, hi = mp_bulk_edges(4, 3, whitened=True)
    assert 0.0 <= lo <= hi
    lo2, hi2 = mp_bulk_edges(4, 3, whitened=False)
    assert hi2 > hi

    # q > 1 branch (sigma_min = 0 for whitened)
    lo3, hi3 = mp_bulk_edges(3, 4, whitened=True)
    assert lo3 == 0.0 and hi3 > 1.0

    e1 = mp_bulk_edge(4, 3, whitened=True)
    e2 = mp_bulk_edge(4, 3, whitened=False)
    assert e2 > e1
    assert mp_bulk_edge(0, 0) == 0.0


def test_growth_ratio_and_deadband():
    r = rmt_growth_ratio(2.0, 1.0, 1.0, 1.0)
    assert r == 2.0
    assert within_deadband(1.05, 1.0, 0.1) is True
    assert within_deadband(1.2, 1.0, 0.1) is False


def test_layer_svd_stats_paths():
    m = Tiny()
    # Baseline-aware path
    base_sigmas = {"lin.weight": 1.0}
    stats = layer_svd_stats(m, baseline_sigmas=base_sigmas, module_name="lin.weight")
    assert "sigma_max" in stats and "worst_ratio" in stats

    # Fallback 98th percentile path (no baseline)
    stats2 = layer_svd_stats(m)
    assert "sigma_max" in stats2 and "worst_ratio" in stats2


def test_rmt_policy_utils():
    from invarlock.core.exceptions import GuardError, ValidationError

    p = get_rmt_policy("balanced")
    assert p["margin"] >= 1.0
    with pytest.raises(GuardError):
        get_rmt_policy("unknown-policy")

    # Valid custom
    custom = create_custom_rmt_policy(
        q=0.5, deadband=0.2, margin=1.1, correct=False, epsilon_by_family={"attn": 0.1}
    )
    assert custom["q"] == 0.5 and custom["correct"] is False
    assert custom["epsilon_by_family"] == {"attn": 0.1}
    # epsilon_default as float allowed
    custom2 = create_custom_rmt_policy(q=0.5, deadband=0.2, margin=1.1, epsilon_default=0.1)
    assert custom2["epsilon_default"] == 0.1
    # Invalid args
    with pytest.raises(ValidationError):
        create_custom_rmt_policy(q=100.0)
    with pytest.raises(ValidationError):
        create_custom_rmt_policy(deadband=0.9)
    with pytest.raises(ValidationError):
        create_custom_rmt_policy(margin=0.9)
