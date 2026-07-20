"""Deterministic descriptive summaries with no decision authority."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from typing import Any, Literal, TypedDict

import numpy as np
import numpy.typing as npt

_FORMAT: Literal["invarlock/diagnostic-observation-v1"] = (
    "invarlock/diagnostic-observation-v1"
)
_MAX_ELEMENTS = 5_000_000


class DiagnosticInputError(ValueError):
    """Raised when a caller-supplied numeric input cannot be summarized."""


class SpectralObservation(TypedDict):
    format: Literal["invarlock/diagnostic-observation-v1"]
    kind: Literal["spectral"]
    status: Literal["observation"]
    method: Literal["exact_svd"]
    input_sha256: str
    rows: int
    columns: int
    rank: int
    singular_value_max: float
    singular_value_min: float
    frobenius_norm: float
    stable_rank: float
    condition_number: float | None


class RmtObservation(TypedDict):
    format: Literal["invarlock/diagnostic-observation-v1"]
    kind: Literal["rmt"]
    status: Literal["observation"]
    method: Literal["column_standardized_covariance_eigh"]
    input_sha256: str
    sample_count: int
    feature_count: int
    varying_feature_count: int
    constant_feature_count: int
    aspect_ratio: float
    marchenko_pastur_lower_edge: float
    marchenko_pastur_upper_edge: float
    empirical_eigenvalue_min: float
    empirical_eigenvalue_max: float
    eigenvalues_above_upper_edge: int
    fraction_above_upper_edge: float


class VarianceObservation(TypedDict):
    format: Literal["invarlock/diagnostic-observation-v1"]
    kind: Literal["variance"]
    status: Literal["observation"]
    method: Literal["float64_two_pass"]
    input_sha256: str
    shape: list[int]
    count: int
    mean: float
    minimum: float
    maximum: float
    population_variance: float
    population_standard_deviation: float
    sample_variance: float | None


def _tensor_to_array_candidate(values: object) -> object:
    """Convert torch-like values without importing or depending on torch."""

    candidate = values
    for method_name in ("detach", "cpu"):
        method = getattr(candidate, method_name, None)
        if callable(method):
            try:
                candidate = method()
            except (RuntimeError, TypeError, ValueError) as exc:
                raise DiagnosticInputError(
                    f"tensor-like input {method_name}() failed: {exc}"
                ) from exc
    to_numpy = getattr(candidate, "numpy", None)
    if callable(to_numpy):
        try:
            candidate = to_numpy()
        except (RuntimeError, TypeError, ValueError) as exc:
            raise DiagnosticInputError(
                f"tensor-like input numpy() failed: {exc}"
            ) from exc
    return candidate


def _numeric_array(values: object, *, label: str) -> npt.NDArray[np.float64]:
    try:
        source = np.asarray(_tensor_to_array_candidate(values))
    except DiagnosticInputError:
        raise
    except (TypeError, ValueError) as exc:
        raise DiagnosticInputError(
            f"{label} must be a numeric array or tensor"
        ) from exc
    if source.dtype.kind not in {"i", "u", "f"}:
        raise DiagnosticInputError(
            f"{label} must contain real numbers, not {source.dtype}"
        )
    if source.size == 0:
        raise DiagnosticInputError(f"{label} must not be empty")
    if source.size > _MAX_ELEMENTS:
        raise DiagnosticInputError(
            f"{label} exceeds the {_MAX_ELEMENTS}-element diagnostic limit"
        )
    try:
        array = np.array(source, dtype=np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise DiagnosticInputError(f"{label} cannot be represented as float64") from exc
    if not bool(np.isfinite(array).all()):
        raise DiagnosticInputError(f"{label} must contain only finite values")
    return array


def _matrix(values: object, *, label: str) -> npt.NDArray[np.float64]:
    matrix = _numeric_array(values, label=label)
    if matrix.ndim != 2 or min(matrix.shape) < 1:
        raise DiagnosticInputError(
            f"{label} must be a non-empty two-dimensional matrix"
        )
    return matrix


def _finite_float(value: float | np.floating[Any], *, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise DiagnosticInputError(f"{label} is not representable as finite float64")
    return result


def _input_sha256(array: npt.NDArray[np.float64]) -> str:
    """Bind the exact normalized float64 material summarized by a diagnostic."""

    little_endian = np.ascontiguousarray(array, dtype="<f8")
    header = json.dumps(
        {"dtype": "float64-le", "shape": list(little_endian.shape)},
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(len(header).to_bytes(8, "big"))
    digest.update(header)
    digest.update(memoryview(little_endian).cast("B"))
    return digest.hexdigest()


def canonical_observation_bytes(observation: Mapping[str, object]) -> bytes:
    """Serialize one diagnostic result for an evidence-request observation input."""

    try:
        return (
            json.dumps(
                dict(observation),
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise DiagnosticInputError(
            f"diagnostic observation is not canonical JSON: {exc}"
        ) from exc


def spectral_observation(values: object) -> SpectralObservation:
    """Return exact SVD statistics for one caller-supplied matrix."""

    matrix = _matrix(values, label="spectral input")
    try:
        singular_values = np.linalg.svd(matrix, compute_uv=False, hermitian=False)
    except np.linalg.LinAlgError as exc:
        raise DiagnosticInputError(f"exact SVD did not converge: {exc}") from exc
    if not bool(np.isfinite(singular_values).all()):
        raise DiagnosticInputError("exact SVD produced non-finite singular values")
    sigma_max = _finite_float(singular_values[0], label="maximum singular value")
    sigma_min = _finite_float(singular_values[-1], label="minimum singular value")
    scale = float(np.max(np.abs(matrix)))
    frobenius = (
        0.0
        if scale == 0.0
        else _finite_float(
            scale * np.linalg.norm(matrix / scale), label="Frobenius norm"
        )
    )
    stable_rank = (
        0.0
        if sigma_max == 0.0
        else _finite_float((frobenius / sigma_max) ** 2, label="stable rank")
    )
    tolerance = max(matrix.shape) * np.finfo(np.float64).eps * sigma_max
    rank = int(np.count_nonzero(singular_values > tolerance))
    condition_number = (
        _finite_float(sigma_max / sigma_min, label="condition number")
        if sigma_min > tolerance
        else None
    )
    return {
        "format": _FORMAT,
        "kind": "spectral",
        "status": "observation",
        "method": "exact_svd",
        "input_sha256": _input_sha256(matrix),
        "rows": int(matrix.shape[0]),
        "columns": int(matrix.shape[1]),
        "rank": rank,
        "singular_value_max": sigma_max,
        "singular_value_min": sigma_min,
        "frobenius_norm": frobenius,
        "stable_rank": stable_rank,
        "condition_number": condition_number,
    }


def rmt_observation(values: object) -> RmtObservation:
    """Return standardized covariance eigenvalues relative to MP edges.

    Rows are samples and columns are features. Constant columns are reported and
    excluded before column standardization. The MP edges are theoretical
    references only; the function does not classify the observation.
    """

    matrix = _matrix(values, label="RMT input")
    sample_count, feature_count = (int(value) for value in matrix.shape)
    if sample_count < 2:
        raise DiagnosticInputError("RMT input requires at least two sample rows")
    with np.errstate(over="ignore", invalid="ignore"):
        centered = matrix - np.mean(matrix, axis=0, dtype=np.float64)
        scales = np.sqrt(np.mean(centered * centered, axis=0, dtype=np.float64))
    if not bool(np.isfinite(centered).all()) or not bool(np.isfinite(scales).all()):
        raise DiagnosticInputError("RMT normalization is not representable as float64")
    varying = scales > 0.0
    varying_count = int(np.count_nonzero(varying))
    if varying_count == 0:
        raise DiagnosticInputError("RMT input has no varying feature columns")
    standardized = centered[:, varying] / scales[varying]
    covariance = (standardized.T @ standardized) / float(sample_count)
    try:
        eigenvalues = np.linalg.eigvalsh(covariance)
    except np.linalg.LinAlgError as exc:
        raise DiagnosticInputError(
            f"standardized covariance eigendecomposition did not converge: {exc}"
        ) from exc
    if not bool(np.isfinite(eigenvalues).all()):
        raise DiagnosticInputError(
            "standardized covariance produced non-finite eigenvalues"
        )
    # Numerical roundoff can make positive-semidefinite eigenvalues minutely
    # negative. Clamp only that representational artifact for the summary.
    eigenvalues = np.maximum(eigenvalues, 0.0)
    aspect_ratio = varying_count / float(sample_count)
    root_ratio = math.sqrt(aspect_ratio)
    lower_edge = (1.0 - root_ratio) ** 2
    upper_edge = (1.0 + root_ratio) ** 2
    above = int(np.count_nonzero(eigenvalues > upper_edge))
    return {
        "format": _FORMAT,
        "kind": "rmt",
        "status": "observation",
        "method": "column_standardized_covariance_eigh",
        "input_sha256": _input_sha256(matrix),
        "sample_count": sample_count,
        "feature_count": feature_count,
        "varying_feature_count": varying_count,
        "constant_feature_count": feature_count - varying_count,
        "aspect_ratio": _finite_float(aspect_ratio, label="aspect ratio"),
        "marchenko_pastur_lower_edge": _finite_float(
            lower_edge, label="Marchenko-Pastur lower edge"
        ),
        "marchenko_pastur_upper_edge": _finite_float(
            upper_edge, label="Marchenko-Pastur upper edge"
        ),
        "empirical_eigenvalue_min": _finite_float(
            eigenvalues[0], label="minimum empirical eigenvalue"
        ),
        "empirical_eigenvalue_max": _finite_float(
            eigenvalues[-1], label="maximum empirical eigenvalue"
        ),
        "eigenvalues_above_upper_edge": above,
        "fraction_above_upper_edge": _finite_float(
            above / float(varying_count), label="fraction above upper edge"
        ),
    }


def variance_observation(values: object) -> VarianceObservation:
    """Return a two-pass scalar variance summary over all input elements."""

    array = _numeric_array(values, label="variance input")
    if array.ndim < 1:
        raise DiagnosticInputError("variance input must have at least one dimension")
    flattened = array.reshape(-1)
    count = int(flattened.size)
    with np.errstate(over="ignore", invalid="ignore"):
        mean_value = np.mean(flattened, dtype=np.float64)
    mean = _finite_float(mean_value, label="mean")
    with np.errstate(over="ignore", invalid="ignore"):
        centered = flattened - mean
        squared_sum = np.dot(centered, centered)
    population_variance = _finite_float(
        squared_sum / float(count), label="population variance"
    )
    sample_variance = (
        _finite_float(squared_sum / float(count - 1), label="sample variance")
        if count > 1
        else None
    )
    return {
        "format": _FORMAT,
        "kind": "variance",
        "status": "observation",
        "method": "float64_two_pass",
        "input_sha256": _input_sha256(array),
        "shape": [int(value) for value in array.shape],
        "count": count,
        "mean": mean,
        "minimum": _finite_float(np.min(flattened), label="minimum"),
        "maximum": _finite_float(np.max(flattened), label="maximum"),
        "population_variance": population_variance,
        "population_standard_deviation": _finite_float(
            math.sqrt(population_variance), label="population standard deviation"
        ),
        "sample_variance": sample_variance,
    }


__all__ = [
    "canonical_observation_bytes",
    "DiagnosticInputError",
    "RmtObservation",
    "SpectralObservation",
    "VarianceObservation",
    "rmt_observation",
    "spectral_observation",
    "variance_observation",
]
