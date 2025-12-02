import pytest
import torch

from invarlock.core import contracts as C
from invarlock.core.auto_tuning import TIER_POLICIES, resolve_tier_policies
from invarlock.core.bootstrap import (
    compute_logloss_ci,
    compute_paired_delta_log_ci,
    logspace_to_ratio_ci,
)


def test_bootstrap_helpers_smoke():
    lo, hi = compute_logloss_ci([1.0, 1.1, 0.9], replicates=100, seed=0)
    assert lo <= hi
    lo2, hi2 = compute_paired_delta_log_ci(
        [1.0, 1.1], [1.0, 1.0], replicates=100, seed=0
    )
    rlo, rhi = logspace_to_ratio_ci((lo2, hi2))
    assert rlo <= rhi


def test_auto_tuning_resolve_tier_policies():
    for tier in ("balanced", "conservative", "aggressive"):
        cfg = resolve_tier_policies(tier)
        assert set(cfg.keys()).issuperset({"spectral", "rmt", "variance"})
    # Unknown tier raises
    with pytest.raises(ValueError):
        resolve_tier_policies("unknown")
    # TIER_POLICIES contains expected keys
    assert "balanced" in TIER_POLICIES


def test_contracts_monotone_and_caps():
    W = torch.randn(8, 8)
    capped = C.enforce_relative_spectral_cap(
        W.clone(), baseline_sigma=2.0, cap_ratio=1.1
    )
    assert isinstance(capped, torch.Tensor)

    approx = torch.ones(4)
    exact = torch.zeros(4)
    out = C.enforce_weight_energy_bound(approx, exact, max_relative_error=0.5)
    # approx too far from exact -> returns exact
    assert torch.allclose(out, exact)

    assert C.rmt_correction_is_monotone(
        1.0, baseline_sigma=2.0, max_ratio=5.0, deadband=0.1
    )
    assert not C.rmt_correction_is_monotone(
        -1.0, baseline_sigma=2.0, max_ratio=5.0, deadband=0.1
    )
