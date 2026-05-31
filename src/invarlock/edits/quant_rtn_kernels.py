from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import torch
import torch.nn as nn

from invarlock.edits.quant_rtn_plan import TargetModule

_WeightRestorer = Callable[[torch.Tensor], torch.Tensor]


def population_std(tensor: torch.Tensor) -> float:
    if tensor.numel() == 0:
        return 0.0
    return float(tensor.float().std(unbiased=False))


def compute_quantization_stats(target_modules: list[TargetModule]) -> dict[str, Any]:
    """Compute statistics about what will be quantized."""
    stats: dict[str, Any] = {
        "total_modules": len(target_modules),
        "total_params": 0,
        "module_stats": [],
    }

    for target in target_modules:
        weight = cast(Any, target.module).weight.detach()
        module_stat: dict[str, Any] = {
            "name": target.name,
            "shape": list(weight.shape),
            "params": int(weight.numel()),
            "weight_range": [float(weight.min()), float(weight.max())],
            "weight_mean": float(weight.mean()),
            "weight_std": population_std(weight),
            "selection_reason": target.selection_reason,
            "matched_pattern": target.matched_pattern,
            "module_type": target.module_type,
        }

        if len(weight.shape) >= 2:
            channel_stats = []
            for channel in range(weight.shape[0]):
                channel_weight = weight[channel]
                channel_stats.append(
                    {
                        "channel": channel,
                        "absmax": float(channel_weight.abs().max()),
                        "mean": float(channel_weight.mean()),
                        "std": population_std(channel_weight),
                    }
                )
            module_stat["channel_stats"] = channel_stats[:10]

        stats["module_stats"].append(module_stat)
        stats["total_params"] += module_stat["params"]

    return stats


def is_transformers_conv1d(module: nn.Module) -> bool:
    return (
        module.__class__.__name__ == "Conv1D"
        and module.__class__.__module__ == "transformers.pytorch_utils"
    )


def weight_to_channel_matrix(
    module: nn.Module, weight: torch.Tensor
) -> tuple[torch.Tensor, _WeightRestorer]:
    if is_transformers_conv1d(module):
        matrix = weight.detach().transpose(0, 1).contiguous()

        def restore_conv1d(value: torch.Tensor) -> torch.Tensor:
            return value.transpose(0, 1).contiguous()

        return matrix, restore_conv1d

    if len(weight.shape) == 1:
        matrix = weight.detach().unsqueeze(0)

        def restore_vector(value: torch.Tensor) -> torch.Tensor:
            return value.squeeze(0)

        return matrix, restore_vector

    original_shape = weight.shape
    matrix = weight.detach().reshape(weight.shape[0], -1)

    def restore_matrix(value: torch.Tensor) -> torch.Tensor:
        return value.reshape(original_shape)

    return matrix, restore_matrix


def apply_outlier_clipping(weight: torch.Tensor, clamp_ratio: float) -> torch.Tensor:
    """Apply outlier clipping based on quantile thresholds."""
    if clamp_ratio <= 0.0:
        return weight

    lower = clamp_ratio / 2
    upper = 1 - lower
    weight_f32 = weight.float()
    quantiles = torch.quantile(
        weight_f32,
        torch.tensor([lower, upper], device=weight.device, dtype=torch.float32),
        dim=1,
        keepdim=True,
    ).to(weight.dtype)

    q_low = quantiles[0]
    q_high = quantiles[1]
    return torch.clamp(weight, q_low, q_high)


def quantize_per_channel(
    weight: torch.Tensor, qmin: int, qmax: int
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Apply per-channel symmetric quantization."""
    channel_absmax = weight.abs().max(dim=1, keepdim=True)[0]
    eps = 1e-8
    channel_absmax = torch.clamp(channel_absmax, min=eps)
    scales = channel_absmax / qmax
    weight_scaled = weight / scales
    weight_quantized = torch.clamp(torch.round(weight_scaled), qmin, qmax)
    quant_code_edge_fraction = float(
        ((weight_quantized <= qmin) | (weight_quantized >= qmax)).float().mean()
    )
    weight_dequantized = weight_quantized * scales
    scale_stats = {
        "channel_count": int(scales.numel()),
        "scale_mean": float(scales.mean()),
        "scale_std": population_std(scales),
        "scale_min": float(scales.min()),
        "scale_max": float(scales.max()),
        "zero_scales": int((scales <= eps).sum()),
        "quant_code_edge_fraction": quant_code_edge_fraction,
        "saturation_fraction": quant_code_edge_fraction,
    }

    return weight_dequantized, scales.squeeze(), scale_stats


def quantization_error_metrics(
    original: torch.Tensor,
    edited: torch.Tensor,
    *,
    clipped_fraction: float,
    quant_code_edge_fraction: float,
) -> dict[str, float]:
    original_f32 = original.detach().float().reshape(-1)
    edited_f32 = edited.detach().float().reshape(-1)
    diff = edited_f32 - original_f32
    abs_diff = diff.abs()
    rmse = torch.sqrt(torch.mean(diff * diff)) if diff.numel() else torch.tensor(0.0)
    original_rms = (
        torch.sqrt(torch.mean(original_f32 * original_f32))
        if original_f32.numel()
        else torch.tensor(0.0)
    )
    denom = torch.clamp(original_rms, min=1e-12)
    original_norm = torch.linalg.vector_norm(original_f32)
    edited_norm = torch.linalg.vector_norm(edited_f32)
    if float(original_norm) <= 1e-12 and float(edited_norm) <= 1e-12:
        cosine_similarity = 1.0
    elif float(original_norm) <= 1e-12 or float(edited_norm) <= 1e-12:
        cosine_similarity = 0.0
    else:
        cosine_similarity = float(
            torch.dot(original_f32, edited_f32) / (original_norm * edited_norm)
        )

    return {
        "mean_abs_error": float(abs_diff.mean()) if abs_diff.numel() else 0.0,
        "max_abs_error": float(abs_diff.max()) if abs_diff.numel() else 0.0,
        "rmse": float(rmse),
        "relative_rmse": float(rmse / denom),
        "cosine_similarity": cosine_similarity,
        "quant_code_edge_fraction": float(quant_code_edge_fraction),
        "saturation_fraction": float(quant_code_edge_fraction),
        "clipped_fraction": float(clipped_fraction),
    }


def apply_rtn_quantization(
    module: nn.Module,
    bitwidth: int,
    clamp_ratio: float,
) -> dict[str, Any]:
    """Apply RTN quantize/dequantize simulation to a single module."""
    weight = cast(Any, module).weight
    original_weight = weight.detach().clone()
    original_shape = weight.shape
    params_quantized = weight.numel()
    weight_2d, restore_weight = weight_to_channel_matrix(module, weight)
    pre_clip_weight = weight_2d

    if clamp_ratio > 0.0:
        weight_2d = apply_outlier_clipping(weight_2d, clamp_ratio)
        clipped_fraction = float((weight_2d != pre_clip_weight).float().mean())
    else:
        clipped_fraction = 0.0

    qmin = -(2 ** (bitwidth - 1))
    qmax = 2 ** (bitwidth - 1) - 1
    quantized_weight_2d, _scales, scale_stats = quantize_per_channel(
        weight_2d, qmin, qmax
    )
    quantized_weight = restore_weight(quantized_weight_2d).reshape(original_shape)
    quantized_weight = quantized_weight.to(dtype=weight.dtype, device=weight.device)

    with torch.no_grad():
        cast(Any, module).weight.copy_(quantized_weight)

    error_metrics = quantization_error_metrics(
        original_weight,
        quantized_weight,
        clipped_fraction=clipped_fraction,
        quant_code_edge_fraction=float(
            scale_stats.get(
                "quant_code_edge_fraction",
                scale_stats.get("saturation_fraction", 0.0),
            )
        ),
    )

    return {
        "params_quantized": params_quantized,
        "original_shape": original_shape,
        "bitwidth": bitwidth,
        "scale_stats": scale_stats,
        "clamp_applied": clamp_ratio > 0.0,
        "error_metrics": error_metrics,
        "actual_storage_dtype": str(module.weight.dtype).replace("torch.", ""),
        "actual_storage_format": "float_dequantized",
        "packed_quantized_storage": False,
        "runtime_memory_reduction": False,
    }


def aggregate_error_metrics(results: list[dict[str, Any]]) -> dict[str, float]:
    metric_pairs = [
        (
            item.get("error_metrics", {}),
            max(int(item.get("params_quantized", 0)), 0),
        )
        for item in results
        if isinstance(item.get("error_metrics"), dict)
    ]
    if not metric_pairs:
        return {}
    metrics = [pair[0] for pair in metric_pairs]
    weighted_params = [pair[1] for pair in metric_pairs]
    total_params = sum(weighted_params)
    aggregate: dict[str, float] = {}
    keys = (
        "mean_abs_error",
        "max_abs_error",
        "rmse",
        "relative_rmse",
        "cosine_similarity",
        "quant_code_edge_fraction",
        "saturation_fraction",
        "clipped_fraction",
    )
    for key in keys:
        values = [float(metric.get(key, 0.0)) for metric in metrics]
        if key == "max_abs_error":
            aggregate[key] = max(values)
        elif total_params > 0:
            aggregate[key] = (
                sum(
                    value * weight
                    for value, weight in zip(values, weighted_params, strict=True)
                )
                / total_params
            )
        else:
            aggregate[key] = sum(values) / len(values)
    return aggregate
