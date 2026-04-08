from __future__ import annotations

import math

import pytest
import torch

import invarlock.guards._estimators as E


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


def test_power_iter_sigma_max_returns_zero_on_non_finite_iteration_norm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_vector_norm = torch.linalg.vector_norm
    call_count = {"count": 0}

    def _vector_norm(*args: object, **kwargs: object) -> torch.Tensor:
        call_count["count"] += 1
        if call_count["count"] == 2:
            return torch.tensor(float("nan"))
        return original_vector_norm(*args, **kwargs)

    monkeypatch.setattr(torch.linalg, "vector_norm", _vector_norm)
    assert E.power_iter_sigma_max(torch.eye(2), iters=2) == 0.0


def test_frobenius_norm_sq_and_as_matrix_paths() -> None:
    assert E.frobenius_norm_sq(torch.empty((0, 3))) == 0.0
    assert E.frobenius_norm_sq(torch.tensor([[float("nan")]])) == 0.0

    W1 = torch.tensor([[3.0, 4.0]])
    assert math.isclose(E.frobenius_norm_sq(W1), 25.0, rel_tol=1e-6, abs_tol=1e-6)

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

    odd_extrema = E.row_col_norm_extrema(
        torch.tensor([[3.0, 4.0], [0.0, 5.0], [8.0, 15.0]])
    )
    assert math.isclose(odd_extrema["row_median"], 5.0, rel_tol=1e-6, abs_tol=1e-6)
    assert math.isclose(
        odd_extrema["col_median"],
        (math.sqrt(73.0) + math.sqrt(266.0)) / 2.0,
        rel_tol=1e-6,
        abs_tol=1e-6,
    )

    even_extrema = E.row_col_norm_extrema(
        torch.tensor(
            [
                [3.0, 4.0, 0.0],
                [0.0, 5.0, 12.0],
                [8.0, 15.0, 0.0],
                [0.0, 0.0, 7.0],
            ]
        )
    )
    assert math.isclose(even_extrema["row_median"], 10.0, rel_tol=1e-6, abs_tol=1e-6)
    assert math.isclose(
        even_extrema["col_median"],
        math.sqrt(193.0),
        rel_tol=1e-6,
        abs_tol=1e-6,
    )

    assert E.stable_rank_estimate(W, sigma_max=object()) == 0.0
    assert E.stable_rank_estimate(W, sigma_max=0.0) == 0.0
    assert E.stable_rank_estimate(W, sigma_max=float("nan")) == 0.0
    assert E.stable_rank_estimate(W, sigma_max=float("inf")) == 0.0

    sigma = float(torch.linalg.svdvals(W)[0].item())
    est = E.stable_rank_estimate(W, sigma_max=sigma)
    assert math.isfinite(est) and est >= 0.0
