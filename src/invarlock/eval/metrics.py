"""Canonical eval-metrics entrypoint."""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from typing import Any, Protocol

import numpy as np
import torch
import torch.nn as nn

from invarlock.core.error_utils import wrap_errors
from invarlock.core.exceptions import MetricsError, ValidationError
from invarlock.utils import (
    bootstrap_statistics,
    percentile_interval_from_statistics,
)

from .metrics_activation import (
    ResultCache,
    _calculate_head_energy,
    _calculate_mi_gini,
    _calculate_sigma_max,
    _collect_activations,
    _perform_pre_eval_checks,
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

logger = logging.getLogger(__name__)
_COERCE_ERRORS = (TypeError, ValueError, OverflowError)
_MODEL_ANALYSIS_ERRORS = (AttributeError, RuntimeError, TypeError, ValueError, OSError)
_ENVIRONMENT_ERRORS = (AttributeError, RuntimeError, TypeError, ValueError)
_LENS_METRIC_ERRORS = (
    AttributeError,
    KeyError,
    MetricsError,
    RuntimeError,
    TypeError,
    ValueError,
)


@dataclass(frozen=True)
class MetricsEnvironmentReport:
    ok: bool
    available_dependencies: tuple[str, ...]
    missing_dependencies: tuple[tuple[str, str], ...]
    messages: tuple[str, ...] = ()


def get_metrics_info() -> dict[str, Any]:
    dep_manager = DependencyManager()
    return {
        "available_metrics": ["sigma_max", "head_energy", "mi_gini"],
        "available_dependencies": list(dep_manager.available_modules.keys()),
        "missing_dependencies": dep_manager.get_missing_dependencies(),
        "default_config": asdict(MetricsConfig(use_cache=False)),
    }


def validate_metrics_environment() -> MetricsEnvironmentReport:
    messages: list[str] = []
    try:
        dep_manager = DependencyManager()
        MetricsConfig(use_cache=False)

        messages.append("Basic dependencies available")
        logger.info("Basic dependencies available")
        available_count = len(dep_manager.available_modules)
        total_count = available_count + len(dep_manager.missing_modules)
        messages.append(
            f"{available_count}/{total_count} optional dependencies available"
        )
        logger.info(
            "%s/%s optional dependencies available",
            available_count,
            total_count,
        )

        if dep_manager.missing_modules:
            missing_messages: list[str] = []
            logger.warning("Some optional dependencies are missing:")
            for name, error in dep_manager.missing_modules:
                line = f"{name}: {error}"
                missing_messages.append(line)
                logger.warning("  - %s", line)

            messages.extend(missing_messages)

        return MetricsEnvironmentReport(
            ok=True,
            available_dependencies=tuple(dep_manager.available_modules.keys()),
            missing_dependencies=tuple(dep_manager.missing_modules),
            messages=tuple(messages),
        )
    except _ENVIRONMENT_ERRORS as e:
        logger.error("Environment validation failed: %s", e)
        messages.append(str(e))
        return MetricsEnvironmentReport(
            ok=False,
            available_dependencies=(),
            missing_dependencies=(),
            messages=tuple(messages),
        )


def bootstrap_confidence_interval(
    samples: list[float] | np.ndarray,
    n_bootstrap: int = 500,
    alpha: float = 0.05,
    statistic: Callable[[np.ndarray], Any] = np.mean,
    random_state: np.random.Generator | None = None,
) -> tuple[float, float]:
    data = np.asarray(samples, dtype=float)
    if data.ndim != 1:
        raise ValidationError(
            code="E402",
            message="METRICS-VALIDATION-FAILED",
            details={"reason": "samples must be 1-dimensional"},
        )
    if data.size == 0:
        raise ValidationError(
            code="E402",
            message="METRICS-VALIDATION-FAILED",
            details={"reason": "samples cannot be empty"},
        )
    if not 0.0 < alpha < 1.0:
        raise ValidationError(
            code="E402",
            message="METRICS-VALIDATION-FAILED",
            details={"reason": "alpha must be between 0 and 1", "alpha": alpha},
        )
    if n_bootstrap <= 0:
        raise ValidationError(
            code="E402",
            message="METRICS-VALIDATION-FAILED",
            details={
                "reason": "n_bootstrap must be positive",
                "n_bootstrap": n_bootstrap,
            },
        )

    with wrap_errors(MetricsError, "E401", "METRICS-COMPUTE-FAILED"):
        rng = random_state or np.random.default_rng()
        stats = bootstrap_statistics(
            data,
            n_bootstrap=int(n_bootstrap),
            random_state=rng,
            statistic=statistic,
        )
        lower, upper = percentile_interval_from_statistics(stats, alpha=alpha)
        return lower, upper


def compute_parameter_deltas(
    model_before: nn.Module, model_after: nn.Module
) -> dict[str, Any]:
    deltas: dict[str, Any] = {
        "params_changed": 0,
        "layers_modified": 0,
        "sparsity": None,
    }

    try:
        before_params = dict(model_before.named_parameters())
        after_params = dict(model_after.named_parameters())
        modified_layers = set()
        total_changed = 0

        for name, before_param in before_params.items():
            if name not in after_params:
                continue

            after_param = after_params[name]
            if not torch.equal(before_param.data, after_param.data):
                total_changed += before_param.numel()
                if ".h." in name or ".layers." in name:
                    import re

                    match = re.search(r"\.(?:h|layers)\.(\d+)\.", name)
                    if match:
                        modified_layers.add(int(match.group(1)))

        deltas["params_changed"] = total_changed
        deltas["layers_modified"] = len(modified_layers)

        total_params_before = sum(p.numel() for p in model_before.parameters())
        total_params_after = sum(p.numel() for p in model_after.parameters())
        if total_params_after < total_params_before:
            deltas["sparsity"] = 1.0 - (total_params_after / total_params_before)
    except _MODEL_ANALYSIS_ERRORS as e:
        logger.warning(f"Parameter delta computation failed: {e}")

    return deltas


def analyze_spectral_changes(
    model_before: nn.Module, model_after: nn.Module, scope: str = "ffn"
) -> dict[str, Any]:
    try:
        from invarlock.guards.spectral_measurement import compute_spectral_norms

        before_norms = compute_spectral_norms(model_before, scope=scope)
        after_norms = compute_spectral_norms(model_after, scope=scope)

        changes = {}
        for layer_name in before_norms:
            if layer_name in after_norms:
                before_norm = before_norms[layer_name]
                after_norm = after_norms[layer_name]
                change_ratio = after_norm / before_norm if before_norm > 0 else 1.0
                changes[layer_name] = {
                    "before": before_norm,
                    "after": after_norm,
                    "ratio": change_ratio,
                }

        ratios = [change["ratio"] for change in changes.values()]
        return {
            "layer_changes": changes,
            "mean_ratio": float(np.mean(ratios)) if ratios else 1.0,
            "max_ratio": float(np.max(ratios)) if ratios else 1.0,
            "min_ratio": float(np.min(ratios)) if ratios else 1.0,
            "layers_analyzed": len(changes),
        }
    except ImportError:
        logger.debug("Spectral analysis not available")
        return {"error": "spectral_analysis_unavailable"}
    except _MODEL_ANALYSIS_ERRORS as e:
        logger.warning(f"Spectral analysis failed: {e}")
        return {"error": str(e)}


class Metric(Protocol):
    name: str
    kind: str

    def compute(self, model: Any, dataset: Iterable[Any]) -> float: ...


class PerplexityMetric:
    """Lightweight perplexity metric from per-record logloss + token counts."""

    name = "perplexity"
    kind = "ppl"

    def compute(self, model: Any, dataset: Iterable[Any]) -> float:  # noqa: ARG002
        total_loss = 0.0
        total_tokens = 0.0
        for raw_record in dataset:
            if not isinstance(raw_record, dict):
                continue
            record: dict[str, Any] = raw_record
            loss: Any = record.get("logloss", record.get("loss"))
            tokens: Any = record.get("token_count", record.get("tokens", 1))
            try:
                loss_val = float(loss)
                tok_val = float(tokens)
            except _COERCE_ERRORS:
                continue
            if (
                not math.isfinite(loss_val)
                or not math.isfinite(tok_val)
                or tok_val <= 0
            ):
                continue
            total_loss += loss_val * tok_val
            total_tokens += tok_val
        if total_tokens <= 0:
            return float("nan")
        return float(math.exp(total_loss / total_tokens))


class AccuracyMetric:
    """Classification accuracy metric from label/prediction records."""

    name = "accuracy"
    kind = "accuracy"

    def compute(self, model: Any, dataset: Iterable[Any]) -> float:  # noqa: ARG002
        from invarlock.eval.tasks import accuracy_from_records

        return accuracy_from_records(dataset)


def _finalize_results(
    results: dict[str, Any],
    skipped_metrics: list[str],
    cache: ResultCache,
    cache_key: str,
    start_time: float,
) -> dict[str, float]:
    for key, value in results.items():
        if not isinstance(value, int | float):
            logger.warning(
                f"Metric {key} has invalid type {type(value)}, setting to NaN"
            )
            results[key] = float("nan")
        elif not (math.isnan(value) or math.isfinite(value)):
            logger.warning(f"Metric {key} is infinite, setting to NaN")
            results[key] = float("nan")

    if skipped_metrics:
        logger.info(f"Skipped metrics: {', '.join(skipped_metrics)}")

    cache.set(cache_key, results)
    elapsed = time.time() - start_time
    logger.info(f"Metrics calculation completed in {elapsed:.2f}s: {results}")
    return results


@torch.no_grad()
def calculate_lens_metrics_for_model(
    model: nn.Module,
    dataloader: Any,
    *,
    config: MetricsConfig,
) -> dict[str, float]:
    dep_manager = DependencyManager()
    resource_manager = ResourceManager(config)
    validator = InputValidator()
    cache = ResultCache(config)

    validator.validate_model(model, config)
    validator.validate_dataloader(dataloader, config)

    cache_key = cache._get_cache_key(model, dataloader, config)
    cached_result = cache.get(cache_key)
    if cached_result is not None:
        logger.info("Using cached metrics result")
        return cached_result

    start_time = time.time()
    logger.info(
        f"Starting metrics calculation with config: oracle_windows={config.oracle_windows}, "
        f"max_tokens={config.max_tokens}, device={resource_manager.device}"
    )

    try:
        _perform_pre_eval_checks(model, dataloader, resource_manager.device, config)
    except _LENS_METRIC_ERRORS as e:
        logger.warning(f"Pre-evaluation checks failed: {e}")

    base_model = getattr(model, "base_model", None)
    if isinstance(base_model, nn.Module):
        model = base_model

    model.eval()
    device = resource_manager.device
    results = {
        "sigma_max": float("nan"),
        "head_energy": float("nan"),
        "mi_gini": float("nan"),
    }
    skipped_metrics: list[str] = []

    try:
        logger.info("Collecting model activations...")
        activation_data = _collect_activations(model, dataloader, config, device)

        if not activation_data["hidden_states"]:
            logger.warning("No activations collected - returning default values")
            return _finalize_results(
                results, skipped_metrics, cache, cache_key, start_time
            )

        results["sigma_max"] = _calculate_sigma_max(
            model, activation_data["first_batch"], dep_manager, config, device
        )
        results["head_energy"] = _calculate_head_energy(
            activation_data["hidden_states"], config
        )
        results["mi_gini"] = _calculate_mi_gini(
            model, activation_data, dep_manager, config, device
        )
    except _LENS_METRIC_ERRORS as e:
        logger.error(f"Metrics calculation failed: {e}")
        if config.strict_validation:
            raise MetricsError(
                code="E401",
                message=f"METRICS-COMPUTE-FAILED: {e}",
            ) from e
    finally:
        resource_manager.cleanup()

    return _finalize_results(results, skipped_metrics, cache, cache_key, start_time)


__all__ = [
    "AccuracyMetric",
    "DependencyError",
    "DependencyManager",
    "InputValidator",
    "Metric",
    "MetricsConfig",
    "MetricsEnvironmentReport",
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
    "get_metrics_info",
    "measure_latency",
    "measure_memory",
    "validate_perplexity",
    "validate_metrics_environment",
]
