from __future__ import annotations

import json
import math
from collections.abc import Mapping

import invarlock_addins.diagnostics.observations as observations
import numpy as np
import numpy.typing as npt
import pytest
from invarlock_addins.diagnostics import (
    DiagnosticInputError,
    canonical_observation_bytes,
    rmt_observation,
    spectral_observation,
    variance_observation,
)


def _assert_observation_only(payload: Mapping[str, object], *, kind: str) -> None:
    assert payload["format"] == "invarlock/diagnostic-observation-v1"
    assert payload["kind"] == kind
    assert payload["status"] == "observation"
    assert len(str(payload["input_sha256"])) == 64
    forbidden = {"pass", "fail", "verdict", "threshold", "calibrated"}
    assert forbidden.isdisjoint(payload)
    json.dumps(payload, allow_nan=False, sort_keys=True)


def test_spectral_observation_matches_known_diagonal_matrix() -> None:
    result = spectral_observation(np.diag([3.0, 1.0]))

    _assert_observation_only(result, kind="spectral")
    assert result["method"] == "exact_svd"
    assert (result["rows"], result["columns"], result["rank"]) == (2, 2, 2)
    assert result["singular_value_max"] == pytest.approx(3.0)
    assert result["singular_value_min"] == pytest.approx(1.0)
    assert result["frobenius_norm"] == pytest.approx(math.sqrt(10.0))
    assert result["stable_rank"] == pytest.approx(10.0 / 9.0)
    assert result["condition_number"] == pytest.approx(3.0)


def test_spectral_observation_reports_rank_deficiency_without_infinite_json() -> None:
    result = spectral_observation([[1.0, 2.0], [2.0, 4.0]])

    assert result["rank"] == 1
    assert result["condition_number"] is None
    json.dumps(result, allow_nan=False)


def test_rmt_observation_matches_identity_covariance_and_excludes_constant() -> None:
    samples = np.array(
        [
            [-1.0, -1.0, 7.0],
            [1.0, 1.0, 7.0],
            [-1.0, 1.0, 7.0],
            [1.0, -1.0, 7.0],
        ]
    )

    result = rmt_observation(samples)

    _assert_observation_only(result, kind="rmt")
    assert result["sample_count"] == 4
    assert result["feature_count"] == 3
    assert result["varying_feature_count"] == 2
    assert result["constant_feature_count"] == 1
    assert result["aspect_ratio"] == pytest.approx(0.5)
    assert result["marchenko_pastur_lower_edge"] == pytest.approx(
        (1.0 - math.sqrt(0.5)) ** 2
    )
    assert result["marchenko_pastur_upper_edge"] == pytest.approx(
        (1.0 + math.sqrt(0.5)) ** 2
    )
    assert result["empirical_eigenvalue_min"] == pytest.approx(1.0)
    assert result["empirical_eigenvalue_max"] == pytest.approx(1.0)
    assert result["eigenvalues_above_upper_edge"] == 0
    assert result["fraction_above_upper_edge"] == 0.0


def test_rmt_observation_describes_correlated_spike_without_a_verdict() -> None:
    column = np.tile(np.array([-1.0, 1.0]), 50)
    samples = np.column_stack([column, column])

    result = rmt_observation(samples)

    _assert_observation_only(result, kind="rmt")
    assert result["empirical_eigenvalue_max"] == pytest.approx(2.0)
    assert result["eigenvalues_above_upper_edge"] == 1
    assert result["fraction_above_upper_edge"] == pytest.approx(0.5)


def test_variance_observation_matches_population_and_sample_definitions() -> None:
    result = variance_observation(np.array([[1.0, 2.0], [3.0, 4.0]]))

    _assert_observation_only(result, kind="variance")
    assert result["method"] == "float64_two_pass"
    assert result["shape"] == [2, 2]
    assert result["count"] == 4
    assert result["mean"] == pytest.approx(2.5)
    assert result["minimum"] == 1.0
    assert result["maximum"] == 4.0
    assert result["population_variance"] == pytest.approx(1.25)
    assert result["population_standard_deviation"] == pytest.approx(math.sqrt(1.25))
    assert result["sample_variance"] == pytest.approx(5.0 / 3.0)


def test_variance_singleton_has_no_sample_variance() -> None:
    result = variance_observation([12.0])

    assert result["population_variance"] == 0.0
    assert result["sample_variance"] is None


class _TensorLike:
    def __init__(self, values: list[list[float]]) -> None:
        self._values = np.asarray(values)
        self.calls: list[str] = []

    def detach(self) -> _TensorLike:
        self.calls.append("detach")
        return self

    def cpu(self) -> _TensorLike:
        self.calls.append("cpu")
        return self

    def numpy(self) -> npt.NDArray[np.float64]:
        self.calls.append("numpy")
        return self._values


def test_tensor_protocol_is_supported_without_a_torch_dependency() -> None:
    tensor = _TensorLike([[2.0, 0.0], [0.0, 1.0]])

    result = spectral_observation(tensor)

    assert tensor.calls == ["detach", "cpu", "numpy"]
    assert result["singular_value_max"] == pytest.approx(2.0)


@pytest.mark.parametrize(
    ("function", "values", "message"),
    [
        (spectral_observation, [], "must not be empty"),
        (spectral_observation, [1.0, 2.0], "two-dimensional"),
        (spectral_observation, [[True, False]], "real numbers"),
        (spectral_observation, [[1.0 + 2.0j]], "real numbers"),
        (variance_observation, [1.0, float("nan")], "finite"),
        (variance_observation, 1.0, "at least one dimension"),
        (rmt_observation, [[1.0, 2.0]], "at least two sample rows"),
        (rmt_observation, [[1.0], [1.0]], "no varying feature"),
    ],
)
def test_invalid_or_mathematically_undefined_inputs_fail_closed(
    function: object, values: object, message: str
) -> None:
    with pytest.raises(DiagnosticInputError, match=message):
        function(values)  # type: ignore[operator]


def test_observations_are_repeatable_and_do_not_mutate_inputs() -> None:
    values = np.arange(12, dtype=np.float64).reshape(4, 3)
    original = values.copy()

    first = (
        spectral_observation(values),
        rmt_observation(values),
        variance_observation(values),
    )
    second = (
        spectral_observation(values),
        rmt_observation(values),
        variance_observation(values),
    )

    assert first == second
    assert first[0]["input_sha256"] != spectral_observation(values + 1)["input_sha256"]
    np.testing.assert_array_equal(values, original)


def test_canonical_observation_bytes_are_ready_for_evidence_input() -> None:
    observation = variance_observation([1.0, 2.0, 3.0])

    encoded = canonical_observation_bytes(observation)

    assert encoded.endswith(b"\n")
    assert json.loads(encoded) == observation
    assert b" " not in encoded


def test_unrepresentable_variance_fails_instead_of_emitting_infinity() -> None:
    with pytest.raises(DiagnosticInputError, match="not representable"):
        variance_observation([1e308, -1e308])


@pytest.mark.parametrize("method_name", ["detach", "cpu", "numpy"])
def test_tensor_protocol_failures_are_reported(method_name: str) -> None:
    class BrokenTensor:
        def detach(self) -> BrokenTensor:
            if method_name == "detach":
                raise RuntimeError("broken")
            return self

        def cpu(self) -> BrokenTensor:
            if method_name == "cpu":
                raise RuntimeError("broken")
            return self

        def numpy(self) -> object:
            if method_name == "numpy":
                raise RuntimeError("broken")
            return [[1.0]]

    with pytest.raises(DiagnosticInputError, match=rf"{method_name}\(\) failed"):
        variance_observation(BrokenTensor())


def test_numeric_conversion_and_size_limits_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class InvalidArray:
        def __array__(self) -> object:
            raise TypeError("invalid")

    with pytest.raises(DiagnosticInputError, match="numeric array"):
        variance_observation(InvalidArray())

    monkeypatch.setattr(observations, "_MAX_ELEMENTS", 1)
    with pytest.raises(DiagnosticInputError, match="diagnostic limit"):
        variance_observation([1.0, 2.0])


def test_canonical_and_decomposition_failures_are_reported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(DiagnosticInputError, match="canonical JSON"):
        canonical_observation_bytes({"invalid": {1, 2}})

    monkeypatch.setattr(
        observations.np.linalg,
        "svd",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            np.linalg.LinAlgError("no convergence")
        ),
    )
    with pytest.raises(DiagnosticInputError, match="SVD did not converge"):
        spectral_observation([[1.0]])


def test_rmt_decomposition_failures_are_reported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        observations.np.linalg,
        "eigvalsh",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            np.linalg.LinAlgError("no convergence")
        ),
    )
    with pytest.raises(DiagnosticInputError, match="did not converge"):
        rmt_observation([[0.0], [1.0]])


def test_decompositions_reject_nonfinite_library_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        observations.np.linalg,
        "svd",
        lambda *_args, **_kwargs: np.asarray([math.nan]),
    )
    with pytest.raises(DiagnosticInputError, match="non-finite singular values"):
        spectral_observation([[1.0]])

    monkeypatch.setattr(
        observations.np.linalg,
        "eigvalsh",
        lambda *_args, **_kwargs: np.asarray([math.nan]),
    )
    with pytest.raises(DiagnosticInputError, match="non-finite eigenvalues"):
        rmt_observation([[0.0], [1.0]])


def test_rmt_rejects_nonrepresentable_normalization() -> None:
    with pytest.raises(
        DiagnosticInputError, match="normalization is not representable"
    ):
        rmt_observation([[1e308], [-1e308]])
