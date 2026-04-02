"""Lens-service orchestration for eval metrics."""

from __future__ import annotations

import logging
import math
import time
from typing import Any

import torch
import torch.nn as nn

from invarlock.core.exceptions import MetricsError

from .metrics_activation import (
    ResultCache,
    _calculate_head_energy,
    _calculate_mi_gini,
    _calculate_sigma_max,
    _collect_activations,
    _perform_pre_eval_checks,
)
from .metrics_support import (
    DependencyManager,
    InputValidator,
    MetricsConfig,
    ResourceManager,
)

logger = logging.getLogger(__name__)


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
    dataloader,
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
    except Exception as e:
        logger.warning(f"Pre-evaluation checks failed: {e}")

    if hasattr(model, "base_model"):
        try:
            model = model.base_model
        except Exception:
            pass

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
    except Exception as e:
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
    "calculate_lens_metrics_for_model",
]
