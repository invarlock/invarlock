"""Aggregation and post-compute helpers for eval metrics."""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Iterable
from typing import Any, Protocol

import numpy as np
import torch
import torch.nn as nn

from invarlock.core.error_utils import wrap_errors
from invarlock.core.exceptions import MetricsError, ValidationError
from invarlock.utils.bootstrap import (
    bootstrap_statistics,
    percentile_interval_from_statistics,
)

logger = logging.getLogger(__name__)


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
    deltas = {
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
    except Exception as e:
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
    except Exception as e:
        logger.warning(f"Spectral analysis failed: {e}")
        return {"error": str(e)}


class Metric(Protocol):
    name: str
    kind: str

    def compute(self, model: Any, dataset: Iterable[dict[str, Any]]) -> float: ...


class PerplexityMetric:
    """Lightweight perplexity metric from per-record logloss + token counts."""

    name = "perplexity"
    kind = "ppl"

    def compute(self, model: Any, dataset: Iterable[dict[str, Any]]) -> float:  # noqa: ARG002
        total_loss = 0.0
        total_tokens = 0.0
        for record in dataset:
            if not isinstance(record, dict):
                continue
            loss = record.get("logloss", record.get("loss"))
            tokens = record.get("token_count", record.get("tokens", 1))
            try:
                loss_val = float(loss)
                tok_val = float(tokens)
            except Exception:
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

    def compute(self, model: Any, dataset: Iterable[dict[str, Any]]) -> float:  # noqa: ARG002
        from invarlock.eval.tasks.classification import accuracy_from_records

        return accuracy_from_records(dataset)


__all__ = [
    "AccuracyMetric",
    "Metric",
    "PerplexityMetric",
    "analyze_spectral_changes",
    "bootstrap_confidence_interval",
    "compute_parameter_deltas",
]
