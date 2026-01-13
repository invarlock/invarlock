from __future__ import annotations

import math

import torch

from invarlock.guards import _estimators as E


def test_power_iter_sigma_max_edge_cases() -> None:
    assert E.power_iter_sigma_max(123, iters=3) == 0.0
    assert E.power_iter_sigma_max(torch.empty((0, 0)), iters=3) == 0.0
    assert E.power_iter_sigma_max(torch.ones((2, 2), dtype=torch.int8), iters=3) == 0.0

    W = torch.eye(4)
    assert E.power_iter_sigma_max(W, iters="bad") > 0.0
    assert E.power_iter_sigma_max(W, iters=0) > 0.0
    assert E.power_iter_sigma_max(W, iters=3, init="e0") > 0.0

    W_nan = torch.tensor([[float("nan"), 0.0], [0.0, 1.0]])
    assert E.power_iter_sigma_max(W_nan, iters=3) == 0.0


def test_frobenius_norm_sq_and_as_matrix_paths() -> None:
    assert E.frobenius_norm_sq(torch.empty((0, 3))) == 0.0
    assert E.frobenius_norm_sq(torch.tensor([[float("nan")]])) == 0.0

    W3 = torch.arange(8, dtype=torch.float32).reshape(2, 2, 2)
    out = E.frobenius_norm_sq(W3)
    expected = float((W3 * W3).sum().item())
    assert math.isfinite(out)
    assert math.isclose(out, expected, rel_tol=1e-6, abs_tol=1e-4)


def test_row_col_norm_extrema_and_stable_rank_estimate_branches() -> None:
    assert E.row_col_norm_extrema(torch.empty((0, 3))) == {
        "row_min": 0.0,
        "row_median": 0.0,
        "row_max": 0.0,
        "col_min": 0.0,
        "col_median": 0.0,
        "col_max": 0.0,
    }

    W = torch.tensor([[1.0, 0.0], [0.0, 2.0], [0.0, 3.0], [0.0, 4.0]])
    extrema = E.row_col_norm_extrema(W)
    assert extrema["row_min"] <= extrema["row_median"] <= extrema["row_max"]
    assert extrema["col_min"] <= extrema["col_median"] <= extrema["col_max"]

    assert E.stable_rank_estimate(W, sigma_max=object()) == 0.0
    assert E.stable_rank_estimate(W, sigma_max=0.0) == 0.0

    sigma = float(torch.linalg.svdvals(W)[0].item())
    est = E.stable_rank_estimate(W, sigma_max=sigma)
    assert math.isfinite(est) and est >= 0.0
