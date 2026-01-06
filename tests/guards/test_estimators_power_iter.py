import math

import torch

from invarlock.guards._estimators import power_iter_sigma_max


def test_power_iter_sigma_max_is_reasonable_on_small_matrix() -> None:
    torch.manual_seed(0)
    W = torch.randn(32, 16)
    exact = float(torch.linalg.svdvals(W)[0].item())
    est = power_iter_sigma_max(W, iters=25, init="ones")
    assert math.isfinite(est) and est > 0
    rel_err = abs(est - exact) / max(exact, 1e-12)
    assert rel_err < 0.15


def test_power_iter_sigma_max_is_deterministic() -> None:
    torch.manual_seed(0)
    W = torch.randn(32, 16)
    a = power_iter_sigma_max(W, iters=7, init="ones")
    b = power_iter_sigma_max(W, iters=7, init="ones")
    assert a == b
