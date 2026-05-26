from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

try:
    from edit_targeting import matches_edit_scope
except ImportError:  # pragma: no cover - direct module load under pytest
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from edit_targeting import matches_edit_scope


@dataclass
class EditStats:
    edited_tensors: int = 0
    edited_params: int = 0
    total_params: int = 0
    details: dict[str, object] = field(default_factory=dict)

    @property
    def coverage_ratio(self) -> float:
        if self.total_params <= 0:
            return 0.0
        return self.edited_params / self.total_params

    def coverage_payload(self) -> dict[str, object]:
        return {
            "edited_tensors": self.edited_tensors,
            "edited_params": self.edited_params,
            "total_params": self.total_params,
            "coverage_ratio": self.coverage_ratio,
        }


def total_model_params(model: Any) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def _matches_scope(name: str, scope: str) -> bool:
    return matches_edit_scope(name, scope)


@torch.no_grad()
def round_to_nearest_dequantized(
    tensor: torch.Tensor,
    *,
    bits: int,
    group_size: int,
) -> torch.Tensor:
    qmin = -(2 ** (bits - 1))
    qmax = max((2 ** (bits - 1)) - 1, 1)
    orig_shape = tensor.shape
    flat = tensor.reshape(orig_shape[0], -1)
    in_features = flat.shape[1]
    eff_group_size = group_size if group_size > 0 else in_features
    if eff_group_size >= in_features:
        eff_group_size = in_features
    num_groups = (in_features + eff_group_size - 1) // eff_group_size
    pad = (num_groups * eff_group_size) - in_features
    if pad > 0:
        flat = torch.nn.functional.pad(flat, (0, pad))
    grouped = flat.reshape(orig_shape[0], num_groups, eff_group_size)
    max_abs = grouped.abs().amax(dim=-1, keepdim=True)
    scale = torch.clamp(max_abs / qmax, min=1e-10)
    quantized = torch.round(grouped / scale).clamp(qmin, qmax) * scale
    quantized = quantized.reshape(orig_shape[0], num_groups * eff_group_size)
    if pad > 0:
        quantized = quantized[:, :in_features]
    return quantized.reshape(orig_shape).to(tensor.dtype)


@torch.no_grad()
def apply_rtn_dequantized_simulation(
    model: Any,
    *,
    bits: int,
    group_size: int,
    scope: str,
) -> EditStats:
    stats = EditStats(total_params=total_model_params(model))
    for name, param in model.named_parameters():
        if _matches_scope(name, scope) and param.dim() >= 2:
            param.data = round_to_nearest_dequantized(
                param.data,
                bits=bits,
                group_size=group_size,
            )
            stats.edited_tensors += 1
            stats.edited_params += param.numel()
            if stats.edited_tensors <= 3:
                print(f"  Quantized: {name} ({tuple(param.shape)})")
    stats.details.update({"bits": bits, "group_size": group_size})
    return stats


def fp8_dtype(format_type: str) -> torch.dtype | None:
    if format_type in {"e4m3", "e4m3fn", "e4m3fnuz"}:
        return getattr(torch, "float8_e4m3fn", None)
    if format_type in {"e5m2", "e5m2fn", "e5m2fnuz"}:
        return getattr(torch, "float8_e5m2", None)
    return None


@torch.no_grad()
def apply_fp8_dequantized_simulation(
    model: Any,
    *,
    format_type: str,
    scope: str,
) -> EditStats:
    dtype = fp8_dtype(format_type)
    stats = EditStats(total_params=total_model_params(model))
    rel_error_total = 0.0

    for name, param in model.named_parameters():
        if not _matches_scope(name, scope) or param.dim() < 2:
            continue
        original = param.data.clone()
        if dtype is None:
            param.data = param.data.to(torch.float16).to(param.dtype)
        else:
            param.data = param.data.to(dtype).to(param.dtype)
        stats.edited_tensors += 1
        stats.edited_params += param.numel()
        denom = original.abs().mean() + 1e-10
        rel_error_total += float((param.data - original).abs().mean() / denom)
        if stats.edited_tensors <= 3:
            print(f"  FP8: {name}")

    avg_error = rel_error_total / max(stats.edited_tensors, 1)
    stats.details.update(
        {
            "format": format_type,
            "avg_relative_error": avg_error,
            "torch_fp8_dtype_available": dtype is not None,
        }
    )
    return stats


@torch.no_grad()
def magnitude_prune_tensor(weight: torch.Tensor, sparsity: float) -> torch.Tensor:
    flat = weight.abs().flatten()
    k = int(flat.numel() * sparsity)
    if k == 0:
        return weight
    threshold = torch.kthvalue(flat, k).values
    mask = weight.abs() >= threshold
    return weight * mask.to(weight.dtype)


@torch.no_grad()
def apply_dense_magnitude_prune(
    model: Any,
    *,
    sparsity: float,
    scope: str,
) -> EditStats:
    stats = EditStats(total_params=total_model_params(model))
    total_zeros = 0

    for name, param in model.named_parameters():
        if _matches_scope(name, scope) and param.dim() >= 2:
            original_zeros = int((param == 0).sum().item())
            param.data = magnitude_prune_tensor(param.data, sparsity)
            new_zeros = int((param == 0).sum().item())
            stats.edited_tensors += 1
            stats.edited_params += param.numel()
            total_zeros += new_zeros
            if stats.edited_tensors <= 3:
                print(f"  Pruned: {name} ({original_zeros} -> {new_zeros} zeros)")

    actual_sparsity = total_zeros / stats.edited_params if stats.edited_params else 0.0
    stats.details.update(
        {
            "target_sparsity": sparsity,
            "actual_sparsity": actual_sparsity,
        }
    )
    return stats


def parse_scope_layers(raw_scope: str) -> tuple[str, int | None, int | None]:
    base = (raw_scope or "").strip()
    layer_limit: int | None = None
    layer_exact: int | None = None
    if "@" in base:
        base, rest = base.split("@", 1)
        base = base.strip()
        for item in (s.strip() for s in rest.split(",") if s.strip()):
            if item.startswith("layers="):
                try:
                    layer_limit = int(item.split("=", 1)[1])
                except (TypeError, ValueError):
                    layer_limit = None
            elif item.startswith("layer="):
                try:
                    layer_exact = int(item.split("=", 1)[1])
                except (TypeError, ValueError):
                    layer_exact = None
    return base, layer_limit, layer_exact


def extract_layer_index(name: str) -> int | None:
    marker = ".layers."
    pos = name.find(marker)
    if pos < 0:
        return None
    start = pos + len(marker)
    end = start
    while end < len(name) and name[end].isdigit():
        end += 1
    if end == start:
        return None
    try:
        return int(name[start:end])
    except (TypeError, ValueError):
        return None


def _layer_selected(
    name: str,
    *,
    layer_limit: int | None,
    layer_exact: int | None,
) -> bool:
    if layer_limit is None and layer_exact is None:
        return True
    idx = extract_layer_index(name)
    if idx is None:
        return False
    if layer_exact is not None and idx != layer_exact:
        return False
    if layer_limit is not None and idx >= layer_limit:
        return False
    return True


@torch.no_grad()
def truncated_svd(weight: torch.Tensor, rank: int) -> torch.Tensor:
    if weight.dim() < 2:
        return weight

    original_shape = weight.shape
    weight_2d = weight.view(weight.shape[0], -1).float()
    max_rank = min(weight_2d.shape)
    effective_rank = min(rank, max_rank)
    u, s, v = torch.svd_lowrank(weight_2d, q=effective_rank, niter=2)
    lowrank = (u * s) @ v.T
    return lowrank.to(weight.dtype).view(original_shape)


@torch.no_grad()
def apply_dense_lowrank_approximation(
    model: Any,
    *,
    rank: int,
    scope: str,
) -> EditStats:
    base_scope, layer_limit, layer_exact = parse_scope_layers(scope)
    if base_scope != scope:
        print(
            "Parsed scope="
            f"{scope} -> base_scope={base_scope}, "
            f"layer_limit={layer_limit}, layer={layer_exact}"
        )

    stats = EditStats(total_params=total_model_params(model))
    total_energy_retained = 0.0

    for name, param in model.named_parameters():
        if not _layer_selected(
            name,
            layer_limit=layer_limit,
            layer_exact=layer_exact,
        ):
            continue
        if _matches_scope(name, base_scope) and param.dim() >= 2:
            original_norm = param.data.norm()
            param.data = truncated_svd(param.data, rank)
            new_norm = param.data.norm()
            energy_retained = (
                (new_norm / original_norm).item() if original_norm > 0 else 1.0
            )
            stats.edited_tensors += 1
            stats.edited_params += param.numel()
            total_energy_retained += energy_retained
            if stats.edited_tensors <= 3:
                print(f"  Low-rank: {name}, energy retained: {energy_retained:.4f}")

    avg_energy = (
        total_energy_retained / stats.edited_tensors if stats.edited_tensors else 1.0
    )
    stats.details.update(
        {
            "rank": rank,
            "avg_energy_retained": avg_energy,
            "base_scope": base_scope,
            "layer_limit": layer_limit,
            "layer": layer_exact,
        }
    )
    return stats


__all__ = [
    "EditStats",
    "apply_dense_lowrank_approximation",
    "apply_dense_magnitude_prune",
    "apply_fp8_dequantized_simulation",
    "apply_rtn_dequantized_simulation",
    "fp8_dtype",
    "magnitude_prune_tensor",
    "parse_scope_layers",
    "round_to_nearest_dequantized",
    "total_model_params",
    "truncated_svd",
]
