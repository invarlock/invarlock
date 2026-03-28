"""Canonical eval-metrics entrypoint with split owner modules."""

from __future__ import annotations

from invarlock.core.exceptions import MetricsError, ValidationError

from .metrics_aggregation import (
    AccuracyMetric,
    Metric,
    PerplexityMetric,
    analyze_spectral_changes,
    bootstrap_confidence_interval,
    compute_parameter_deltas,
)
from .metrics_lens import (
    calculate_lens_metrics_for_model,
)
from .metrics_runtime import (
    PerplexityStatus,
    compute_perplexity,
    compute_perplexity_strict,
    compute_ppl,
    measure_latency,
    measure_memory,
    validate_perplexity,
)
from .metrics_support import (
    DependencyError,
    DependencyManager,
    InputValidator,
    MetricsConfig,
    ResourceError,
    ResourceManager,
)

__all__ = [
    "AccuracyMetric",
    "DependencyError",
    "DependencyManager",
    "InputValidator",
    "Metric",
    "MetricsConfig",
    "MetricsError",
    "PerplexityMetric",
    "PerplexityStatus",
    "ResourceError",
    "ResourceManager",
    "ValidationError",
    "analyze_spectral_changes",
    "bootstrap_confidence_interval",
    "calculate_lens_metrics_for_model",
    "compute_parameter_deltas",
    "compute_perplexity",
    "compute_perplexity_strict",
    "compute_ppl",
    "measure_latency",
    "measure_memory",
    "validate_perplexity",
]
