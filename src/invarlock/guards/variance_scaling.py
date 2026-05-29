from __future__ import annotations

import itertools
import math
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal

import torch
import torch.nn as nn

from .variance_types import ScaleComputationResult

ProgressPhase = Literal["calibration"]
_VARIANCE_SCALING_ERRORS = (AttributeError, RuntimeError, TypeError, ValueError)


@dataclass(frozen=True)
class VarianceScalingProgress:
    phase: ProgressPhase
    completed: int
    total: int | None


ProgressCallback = Callable[[VarianceScalingProgress], None]


def _emit_progress(
    callback: ProgressCallback | None, *, completed: int, total: int | None
) -> None:
    if callback is None:
        return
    callback(
        VarianceScalingProgress(
            phase="calibration",
            completed=int(completed),
            total=None if total is None else int(total),
        )
    )


def unwrap_model(model: nn.Module) -> nn.Module:
    """Unwrap DataParallel/DDP wrappers to get the underlying model."""
    unwrapped = model
    while hasattr(unwrapped, "module"):
        unwrapped = unwrapped.module
    return unwrapped


def iter_transformer_layers(model: nn.Module):
    """Iterate over transformer layers across supported architectures."""
    model = unwrap_model(model)
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        yield from model.transformer.h
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        yield from model.model.layers
    elif (
        hasattr(model, "model")
        and hasattr(model.model, "model")
        and hasattr(model.model.model, "layers")
    ):
        yield from model.model.model.layers
    elif hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
        yield from model.encoder.layer
    elif hasattr(model, "decoder") and hasattr(model.decoder, "layers"):
        yield from model.decoder.layers
    elif hasattr(model, "layers"):
        yield from model.layers
    else:
        for module in model.modules():
            if (hasattr(module, "attn") and hasattr(module, "mlp")) or (
                hasattr(module, "self_attn")
                and (hasattr(module, "mlp") or hasattr(module, "block_sparse_moe"))
            ):
                yield module


@torch.no_grad()
def equalise_residual_variance(
    model: nn.Module,
    dataloader,
    *,
    windows: int = 32,
    tol: float = 0.02,
    scale_bias: bool = True,
    seed: int = 42,
    device: str | None = None,
    allow_empty: bool = False,
    clamp_range: tuple | None = (0.9, 1.1),
    apply: bool = True,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, float]:
    """Apply data-driven variance equalization to transformer branches."""
    torch.manual_seed(seed)

    if device is None:
        device = next(model.parameters()).device
    else:
        device = torch.device(device)

    model.eval()
    hooks: dict[str, Any] = {}
    sample_values: dict[str, list[float]] = defaultdict(list)

    def branch_hook(name: str):
        def fn(_, __, out):
            y = out[0] if isinstance(out, tuple) else out
            y = y.detach().float()
            if y.numel() == 0:
                return
            sample_values[name].append(float(y.pow(2).mean().item()))

        return fn

    def moe_expert_out_modules(block: Any) -> list[Any]:
        experts = getattr(block, "experts", None)
        if experts is None:
            return []
        out: list[Any] = []
        iterable: Iterable[Any]
        modules_map = getattr(experts, "_modules", None)
        if isinstance(experts, nn.Module) and isinstance(modules_map, dict):
            iterable = modules_map.values()
        else:
            try:
                iterable = list(experts)
            except TypeError:
                iterable = []
        for expert in iterable:
            for attr in ("w2", "down_proj", "c_proj", "fc2"):
                proj = getattr(expert, attr, None)
                if proj is not None and hasattr(proj, "weight"):
                    out.append(proj)
                    break
        if out:
            return out
        for attr in ("w2", "down_proj", "c_proj", "fc2"):
            proj = getattr(experts, attr, None)
            if proj is None:
                continue
            weight = getattr(proj, "weight", None)
            candidate = weight if isinstance(weight, torch.Tensor) else None
            if candidate is None and isinstance(proj, torch.Tensor):
                candidate = proj
            if candidate is None:
                continue
            try:
                dim = candidate.dim()
            except _VARIANCE_SCALING_ERRORS:
                dim = getattr(candidate, "ndim", None)
            if dim in (2, 3):
                return [proj]
        return out

    for index, block in enumerate(iter_transformer_layers(model)):
        if hasattr(block, "attn"):
            attn_proj = getattr(block.attn, "c_proj", None) or getattr(
                block.attn, "out_proj", None
            )
            if attn_proj is not None:
                name = f"block{index}.attn"
                hooks[name] = attn_proj.register_forward_hook(branch_hook(name))

        mlp_container = getattr(block, "mlp", None)
        if mlp_container is None:
            mlp_container = getattr(block, "block_sparse_moe", None)

        if mlp_container is not None:
            mlp_proj = (
                getattr(mlp_container, "c_proj", None)
                or getattr(mlp_container, "down_proj", None)
                or getattr(mlp_container, "fc2", None)
            )
            if mlp_proj is not None:
                name = f"block{index}.mlp"
                hooks[name] = mlp_proj.register_forward_hook(branch_hook(name))
            else:
                if moe_expert_out_modules(mlp_container):
                    name = f"block{index}.mlp"
                    hooks[name] = mlp_container.register_forward_hook(branch_hook(name))

    try:
        batches = list(itertools.islice(iter(dataloader), windows))
    except (StopIteration, TypeError):
        batches = []

    if not batches and not allow_empty:
        raise ValueError("Empty dataloader provided and allow_empty=False")

    total_batches = len(batches)
    for idx, batch in enumerate(batches, start=1):
        if isinstance(batch, dict):
            input_ids = batch.get("input_ids", batch.get("inputs", None))
        elif isinstance(batch, tuple | list):
            input_ids = batch[0] if len(batch) > 0 else None
        else:
            input_ids = batch

        if input_ids is not None:
            if not isinstance(input_ids, torch.Tensor):
                input_ids = torch.as_tensor(input_ids)
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
            with torch.no_grad():
                model(input_ids.to(device))
        _emit_progress(progress_callback, completed=idx, total=total_batches)

    for hook in hooks.values():
        hook.remove()

    applied_scales: dict[str, float] = {}
    for index, block in enumerate(iter_transformer_layers(model)):
        if hasattr(block, "attn"):
            attn_proj = getattr(block.attn, "c_proj", None) or getattr(
                block.attn, "out_proj", None
            )
            if attn_proj is not None:
                name = f"block{index}.attn"
                values = sample_values.get(name, [])
                if values:
                    tensor_vals = torch.tensor(values, dtype=torch.float64)
                    if tensor_vals.numel() >= 10:
                        lower = torch.quantile(tensor_vals, 0.02)
                        upper = torch.quantile(tensor_vals, 0.98)
                        tensor_vals = torch.clamp(
                            tensor_vals, lower.item(), upper.item()
                        )
                    group_count = 8 if tensor_vals.numel() >= 8 else tensor_vals.numel()
                    if group_count > 1:
                        chunks = torch.chunk(tensor_vals, group_count)
                        group_means = torch.stack([chunk.mean() for chunk in chunks])
                        var_f = group_means.median().item()
                    else:
                        var_f = tensor_vals.mean().item()
                    alpha = (1.0 / max(var_f, 1e-9)) ** 0.5
                    if clamp_range is not None:
                        alpha = max(clamp_range[0], min(alpha, clamp_range[1]))
                    if abs(alpha - 1.0) >= tol:
                        if apply:
                            with torch.no_grad():
                                attn_proj.weight.mul_(alpha)
                                if scale_bias and attn_proj.bias is not None:
                                    attn_proj.bias.mul_(alpha)
                        applied_scales[name] = alpha

        mlp_container = getattr(block, "mlp", None)
        if mlp_container is None:
            mlp_container = getattr(block, "block_sparse_moe", None)

        if mlp_container is None:
            continue

        mlp_proj = (
            getattr(mlp_container, "c_proj", None)
            or getattr(mlp_container, "down_proj", None)
            or getattr(mlp_container, "fc2", None)
        )
        name = f"block{index}.mlp"
        values = sample_values.get(name, [])
        if not values:
            continue

        tensor_vals = torch.tensor(values, dtype=torch.float64)
        if tensor_vals.numel() >= 10:
            lower = torch.quantile(tensor_vals, 0.02)
            upper = torch.quantile(tensor_vals, 0.98)
            tensor_vals = torch.clamp(tensor_vals, lower.item(), upper.item())
        group_count = 8 if tensor_vals.numel() >= 8 else tensor_vals.numel()
        if group_count > 1:
            chunks = torch.chunk(tensor_vals, group_count)
            group_means = torch.stack([chunk.mean() for chunk in chunks])
            var_f = group_means.median().item()
        else:
            var_f = tensor_vals.mean().item()

        alpha = (1.0 / max(var_f, 1e-9)) ** 0.5
        if clamp_range is not None:
            alpha = max(clamp_range[0], min(alpha, clamp_range[1]))
        if abs(alpha - 1.0) < tol:
            continue

        if mlp_proj is not None:
            if apply:
                with torch.no_grad():
                    mlp_proj.weight.mul_(alpha)
                    if scale_bias and mlp_proj.bias is not None:
                        mlp_proj.bias.mul_(alpha)
            applied_scales[name] = alpha
            continue

        moe_out = moe_expert_out_modules(mlp_container)
        if moe_out:
            if apply:
                with torch.no_grad():
                    for proj in moe_out:
                        weight = getattr(proj, "weight", None)
                        if isinstance(weight, torch.Tensor):
                            weight.mul_(alpha)
                            bias = getattr(proj, "bias", None)
                            if scale_bias and isinstance(bias, torch.Tensor):
                                bias.mul_(alpha)
                        elif isinstance(proj, torch.Tensor):
                            proj.mul_(alpha)
            applied_scales[name] = alpha

    return applied_scales


def compute_variance_scales(
    guard: Any,
    model: nn.Module,
    dataloader,
) -> ScaleComputationResult:
    """Compute filtered VE scales for the guard state."""
    if guard._monitor_only:
        guard._log_event(
            "monitor_only",
            message="Skipping variance scale computation in monitor-only mode",
        )
        guard._raw_scales = {}
        return ScaleComputationResult({}, {}, False, False)

    tensor_ready_batches = guard._tensorize_calibration_batches(dataloader)
    proposed_scales = equalise_residual_variance(
        model=model,
        dataloader=tensor_ready_batches,
        windows=min(guard._policy["max_calib"] // 10, 50),
        tol=guard._policy["deadband"],
        scale_bias=False,
        seed=guard._policy["seed"],
        clamp_range=guard._policy["clamp"],
        allow_empty=True,
        apply=False,
    )

    if not proposed_scales and guard._policy.get("deadband", 0.0) > 0.0:
        relaxed_tol = max(guard._policy["deadband"] * 0.5, 1e-4)
        tensor_ready_batches = guard._tensorize_calibration_batches(dataloader)
        proposed_scales = equalise_residual_variance(
            model=model,
            dataloader=tensor_ready_batches,
            windows=min(guard._policy["max_calib"] // 10, 50),
            tol=relaxed_tol,
            scale_bias=False,
            seed=guard._policy["seed"] + 7,
            clamp_range=guard._policy["clamp"],
            allow_empty=True,
            apply=False,
        )

    raw_scales = dict(proposed_scales)
    if guard._target_modules:
        filtered_raw_scales: dict[str, float] = {}
        for scale_name, scale_value in raw_scales.items():
            target_name = guard._normalize_scale_name(scale_name)
            if target_name in guard._target_modules:
                filtered_raw_scales[scale_name] = scale_value
            elif guard._is_focus_match(scale_name):
                for target_module_name in guard._target_modules:
                    if guard._scale_matches_target(scale_name, target_module_name):
                        filtered_raw_scales[scale_name] = scale_value
                        break
        raw_scales = filtered_raw_scales

    focus_raw_scales = {
        guard._normalize_scale_name(name): scale
        for name, scale in raw_scales.items()
        if guard._is_focus_match(name)
    }
    if focus_raw_scales:
        guard._log_event(
            "variance_raw_scales",
            message="Captured raw VE scales",
            count=len(focus_raw_scales),
            min_scale=min(focus_raw_scales.values()),
            max_scale=max(focus_raw_scales.values()),
        )
    guard._stats.setdefault("raw_scales_observations", []).append(
        {
            "timestamp": datetime.now(UTC).isoformat(),
            "count": len(focus_raw_scales),
            "scales": focus_raw_scales,
        }
    )

    filtered_scales: dict[str, float] = {}
    raw_delta_map: dict[str, float] = {}
    min_abs = float(max(guard._policy.get("min_abs_adjust", 0.0), 0.0))
    max_step = float(max(guard._policy.get("max_scale_step", 0.0), 0.0))
    topk = int(max(guard._policy.get("topk_backstop", 0) or 0, 0))
    best_candidate: tuple[str, float] | None = None
    best_delta = 0.0

    for name, scale in raw_scales.items():
        normalized_name = guard._normalize_scale_name(name)
        if not guard._is_focus_match(normalized_name):
            continue
        raw_delta = abs(scale - 1.0)
        raw_delta_map[name] = raw_delta
        if raw_delta > best_delta:
            best_candidate = (name, scale)
            best_delta = raw_delta
        if raw_delta < min_abs:
            continue
        if max_step > 0.0:
            limited_delta = min(raw_delta, max_step)
            scale = 1.0 + math.copysign(limited_delta, scale - 1.0)
        filtered_scales[name] = scale

    backstop_used = False
    if not filtered_scales and topk > 0 and best_candidate:
        name, scale = best_candidate
        deadband = float(guard._policy.get("deadband", 0.0) or 0.0)
        threshold = max(deadband * 0.5, min_abs * 0.5)
        if min_abs > 0 and threshold >= min_abs:
            threshold = min_abs * 0.5
        if best_delta >= threshold:
            if max_step > 0.0:
                limited_delta = min(best_delta, max_step)
                scale = 1.0 + math.copysign(limited_delta, scale - 1.0)
            filtered_scales[name] = scale
            raw_delta_map.setdefault(name, best_delta)
            backstop_used = True

    trimmed_to_limit = False
    max_adjusted = int(max(guard._policy.get("max_adjusted_modules", 0) or 0, 0))
    if max_adjusted > 0 and len(filtered_scales) > max_adjusted:
        sorted_candidates = sorted(
            filtered_scales.items(),
            key=lambda item: (
                raw_delta_map.get(item[0], abs(item[1] - 1.0))
                + (2.0 if item[1] >= 1.0 else 0.0),
                raw_delta_map.get(item[0], abs(item[1] - 1.0)),
                item[1],
            ),
            reverse=True,
        )
        filtered_scales = dict(sorted_candidates[:max_adjusted])
        trimmed_to_limit = True

    if backstop_used:
        guard._log_event(
            "scale_backstop",
            message=f"Top-{topk} backstop injected {len(filtered_scales)} scale",
            count=len(filtered_scales),
            candidate=best_candidate[0] if best_candidate else None,
            candidate_normalized=guard._normalize_scale_name(best_candidate[0])
            if best_candidate
            else None,
            delta=best_delta,
        )
    if trimmed_to_limit:
        guard._log_event(
            "scale_limit",
            message="Trimmed VE scales to max_adjusted_modules",
            limit=max_adjusted,
            count=len(filtered_scales),
        )

    filtered_normalized = {
        guard._normalize_scale_name(name): scale
        for name, scale in filtered_scales.items()
    }
    guard._stats.setdefault("filtered_scales_observations", []).append(
        {
            "timestamp": datetime.now(UTC).isoformat(),
            "count": len(filtered_normalized),
            "scales": filtered_normalized,
            "backstop_used": backstop_used,
        }
    )

    return ScaleComputationResult(
        raw_scales=raw_scales,
        filtered_scales=filtered_scales,
        backstop_used=backstop_used,
        trimmed_to_limit=trimmed_to_limit,
    )


__all__ = [
    "compute_variance_scales",
    "equalise_residual_variance",
    "iter_transformer_layers",
    "unwrap_model",
    "VarianceScalingProgress",
]
