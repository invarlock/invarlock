from __future__ import annotations

import numpy as np
import pytest

from invarlock.diagnostics import DiagnosticInputError, rmt_observation


def test_wide_rank_one_preserves_full_feature_denominator_and_zero_eigenvalues():
    x = np.tile([[-1.0], [1.0]], (8, 64))
    result = rmt_observation(x, method="smaller_gram")
    assert result["method"] == "column_standardized_smaller_gram_eigh"
    assert result["gram_dimension"] == 16
    assert result["gram_matrix_bytes"] == 16 * 16 * 8
    assert result["empirical_eigenvalue_min"] == 0
    assert result["empirical_eigenvalue_max"] == pytest.approx(64)
    assert result["eigenvalues_above_upper_edge"] == 1
    assert result["fraction_above_upper_edge"] == 1 / 64
    assert result["minimum_upper_edge_distance"] == pytest.approx(9)


def test_tall_identity_and_constants_use_varying_feature_count():
    x = np.array([[-1, -1, 7], [1, 1, 7], [-1, 1, 7], [1, -1, 7]])
    original = rmt_observation(x)
    assert "gram_dimension" not in original
    candidate = rmt_observation(x, method="smaller_gram")
    assert candidate["gram_dimension"] == 2
    assert candidate["constant_feature_count"] == 1
    for key in original.keys() - {"method"}:
        assert candidate[key] == original[key]


def test_gram_allocation_limit_rejects_before_eigensolver(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("eigensolver must not run")

    monkeypatch.setattr(np.linalg, "eigvalsh", forbidden)
    with pytest.raises(DiagnosticInputError, match="Gram matrix requires 800 bytes"):
        rmt_observation(np.tile([[-1.0], [1.0]], (1, 10)), max_gram_bytes=799)


def test_dual_path_fits_limit_that_rejects_feature_covariance():
    x = np.tile([[-1.0], [1.0]], (1, 10))
    result = rmt_observation(x, method="smaller_gram", max_gram_bytes=32)
    assert result["gram_matrix_bytes"] == 32


@pytest.mark.parametrize("value", [True, 0, -1, 1.5, None])
def test_invalid_gram_budget(value):
    with pytest.raises(DiagnosticInputError, match="positive integer"):
        rmt_observation([[-1], [1]], max_gram_bytes=value)


@pytest.mark.parametrize("method", ["approximate", None, [], {}])
def test_unknown_method_rejected(method):
    with pytest.raises(DiagnosticInputError, match="method"):
        rmt_observation([[-1], [1]], method=method)


@pytest.mark.parametrize("kind", ["nonfinite", "convergence"])
def test_smaller_gram_eigensolver_failure(kind, monkeypatch):
    def broken(*args, **kwargs):
        if kind == "convergence":
            raise np.linalg.LinAlgError("fixture")
        return np.array([float("nan")])

    monkeypatch.setattr(np.linalg, "eigvalsh", broken)
    with pytest.raises(DiagnosticInputError):
        rmt_observation([[-1, -1, -1], [1, 1, 1]], method="smaller_gram")


def test_edge_distance_does_not_change_strict_count(monkeypatch):
    x = np.tile([[-1.0], [1.0]], (1, 8))
    edge = 9.0
    monkeypatch.setattr(np.linalg, "eigvalsh", lambda _: np.array([0, edge]))
    result = rmt_observation(x, method="smaller_gram")
    assert result["minimum_upper_edge_distance"] == 0
    assert result["eigenvalues_above_upper_edge"] == 0
    monkeypatch.setattr(
        np.linalg, "eigvalsh", lambda _: np.array([0, np.nextafter(edge, np.inf)])
    )
    result = rmt_observation(x, method="smaller_gram")
    assert result["eigenvalues_above_upper_edge"] == 1
    assert result["minimum_upper_edge_distance"] == np.spacing(edge)
