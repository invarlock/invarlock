"""RMT weight-analysis helpers for baseline capture and SVD inspection."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import torch
import torch.linalg as tla
import torch.nn as nn

__all__ = [
    "mp_bulk_edges",
    "mp_bulk_edge",
    "rmt_growth_ratio",
    "rmt_correction_is_monotone",
    "within_deadband",
    "clip_full_svd",
    "capture_baseline_mp_stats",
    "collect_linear_rmt_modules",
    "layer_svd_stats",
    "analyze_weight_distribution",
    "_iter_transformer_layers",
]


def mp_bulk_edges(m: int, n: int, whitened: bool = True) -> tuple[float, float]:
    """Compute Marchenko-Pastur bulk edges for an ``m x n`` matrix."""
    if m == 0 or n == 0:
        return 0.0, 0.0

    q = n / m
    if whitened:
        sigma_max = 1.0 + np.sqrt(q)
        sigma_min = abs(1.0 - np.sqrt(q)) if q <= 1 else 0.0
    else:
        sigma_max = np.sqrt(m) * (1.0 + np.sqrt(q))
        sigma_min = np.sqrt(m) * abs(1.0 - np.sqrt(q)) if q <= 1 else 0.0

    return float(sigma_min), float(sigma_max)


def mp_bulk_edge(m: int, n: int, whitened: bool = False) -> float:
    """Compute the upper Marchenko-Pastur bulk edge for an ``m x n`` matrix."""
    return mp_bulk_edges(m, n, whitened=whitened)[1]


def rmt_growth_ratio(
    sigma_cur: float,
    mp_cur: float,
    sigma_base: float,
    mp_base: float,
) -> float:
    """Compute the baseline-aware growth ratio used by RMT checks."""
    r_base = sigma_base / max(mp_base, 1e-12)
    r_cur = sigma_cur / max(mp_cur, 1e-12)
    return r_cur / max(r_base, 1e-12)


def within_deadband(sigma_cur: float, sigma_base: float, deadband: float) -> bool:
    """Check whether the current sigma stays within the allowed deadband."""
    return sigma_cur <= (1.0 + deadband) * sigma_base


def rmt_correction_is_monotone(
    corrected_sigma: float,
    baseline_sigma: float,
    max_ratio: float,
    deadband: float,
) -> bool:
    """
    Validate monotonicity for RMT correction.

    ``corrected_sigma`` should not exceed ``baseline_sigma * (1 + deadband)``
    and must remain <= ``max_ratio``.
    """
    if corrected_sigma < 0 or baseline_sigma <= 0 or max_ratio <= 0:
        return False
    if corrected_sigma > max_ratio:
        return False
    return corrected_sigma <= baseline_sigma * (1.0 + deadband)


def clip_full_svd(
    W: torch.Tensor,
    clip_val: float,
    return_components: bool = False,
) -> (
    torch.Tensor | tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]
):
    """Clip singular values of a matrix using full SVD."""
    if not torch.isfinite(W).all():
        if return_components:
            return None, None, None
        return W

    try:
        U, S, Vt = torch.linalg.svd(W.float(), full_matrices=False)
        S_clipped = torch.clamp(S, max=clip_val)
        if return_components:
            return U, S_clipped, Vt
        clipped = (U @ torch.diag(S_clipped) @ Vt).to(W.dtype)
        return cast(torch.Tensor, clipped)
    except (RuntimeError, torch.linalg.LinAlgError):
        if return_components:
            return None, None, None
        return W


def collect_linear_rmt_modules(
    model: nn.Module,
    allowed_suffixes: list[str] | None = None,
    *,
    allowed_module_names: list[str] | None = None,
) -> list[tuple[str, nn.Module]]:
    """Collect canonical linear modules in scope for RMT analysis."""
    if allowed_suffixes is None:
        allowed_suffixes = [
            ".attn.c_attn",
            ".attn.c_proj",
            ".mlp.c_fc",
            ".mlp.c_proj",
        ]
    try:
        from transformers.pytorch_utils import Conv1D

        module_types_with_conv1d: tuple[
            type[nn.Linear], type[nn.Conv1d], type[Conv1D]
        ] = (nn.Linear, nn.Conv1d, Conv1D)
        module_types = module_types_with_conv1d
    except ImportError:
        module_types_without_conv1d: tuple[type[nn.Linear], type[nn.Conv1d]] = (
            nn.Linear,
            nn.Conv1d,
        )
        module_types = module_types_without_conv1d

    allowed_set = None
    if isinstance(allowed_module_names, list) and allowed_module_names:
        allowed_set = {str(name).strip() for name in allowed_module_names if name}

    candidates: list[tuple[str, nn.Module]] = []
    for name, module in model.named_modules():
        if not (isinstance(module, module_types) and hasattr(module, "weight")):
            continue
        if allowed_set is not None and name not in allowed_set:
            continue
        if any(name.endswith(suffix) for suffix in allowed_suffixes):
            candidates.append((name, module))
    candidates.sort(key=lambda item: item[0])
    return candidates


def _iter_weight_matrices(layer: nn.Module):
    """Iterate over 2D weight matrices in a layer."""
    for name, param in layer.named_parameters():
        if param.ndim == 2 and "weight" in name:
            yield name, param.detach()


def layer_svd_stats(
    layer: nn.Module,
    baseline_sigmas: dict[str, float] | None = None,
    baseline_mp_stats: dict[str, dict[str, float]] | None = None,
    module_name: str | None = None,
) -> dict[str, float]:
    """Compute SVD statistics for a single layer with optional baseline-aware normalization."""
    sigma_min_global = float("inf")
    sigma_max_global = 0.0
    worst_ratio = 0.0
    worst_details = None

    for name, W in _iter_weight_matrices(layer):
        if W.numel() == 0:
            continue
        if not torch.isfinite(W).all():
            continue

        m, n = W.shape
        try:
            s_actual = tla.svdvals(W.float().cpu())
            s_min = s_actual[-1].item()
            s_max = s_actual[0].item()
        except (RuntimeError, torch.linalg.LinAlgError):
            continue

        sigma_min_global = min(sigma_min_global, s_min)
        sigma_max_global = max(sigma_max_global, s_max)

        if baseline_sigmas and module_name and module_name in baseline_sigmas:
            baseline_sigma = baseline_sigmas[module_name]
            if baseline_sigma > 0:
                mp_edge_current = mp_bulk_edge(m, n, whitened=False)
                if baseline_mp_stats and module_name in baseline_mp_stats:
                    mp_edge_baseline = baseline_mp_stats[module_name].get(
                        "mp_bulk_edge_base", mp_edge_current
                    )
                else:
                    mp_edge_baseline = mp_edge_current
                ratio = rmt_growth_ratio(
                    s_max, mp_edge_current, baseline_sigma, mp_edge_baseline
                )
            else:
                ratio = 1.0
        else:
            if len(s_actual) > 1:
                s_sorted = s_actual.sort()[0]
                idx_98 = int(0.98 * len(s_sorted))
                s_98 = s_sorted[idx_98].item()
                ratio = s_max / s_98 if s_98 > 0 else 1.0
            else:
                ratio = 1.0

        if ratio > worst_ratio:
            worst_ratio = ratio
            worst_details = {
                "name": name,
                "shape": (m, n),
                "s_max": s_max,
                "s_min": s_min,
                "s_median": s_actual.median().item() if len(s_actual) > 1 else s_max,
                "s_98": s_actual.sort()[0][int(0.98 * len(s_actual))].item()
                if len(s_actual) > 1
                else s_max,
                "ratio": ratio,
                "mp_edge": mp_bulk_edge(m, n, whitened=False),
                "normalization": "baseline_aware"
                if baseline_sigmas and module_name and module_name in baseline_sigmas
                else "98th_percentile",
            }

    result = {
        "sigma_min": sigma_min_global,
        "sigma_max": sigma_max_global,
        "worst_ratio": worst_ratio,
    }
    if worst_details:
        result["worst_details"] = worst_details
    return result


def capture_baseline_mp_stats(
    model: nn.Module, *, allowed_module_names: list[str] | None = None
) -> dict[str, dict[str, float]]:
    """Capture baseline MP statistics for the canonical linear-layer allowlist."""
    mp_stats: dict[str, dict[str, float]] = {}

    for name, module in collect_linear_rmt_modules(
        model,
        allowed_module_names=allowed_module_names,
    ):
        for param_name, param in module.named_parameters(recurse=False):
            if param.ndim == 2 and "weight" in param_name:
                W = param.detach()
                try:
                    from transformers.pytorch_utils import Conv1D

                    if isinstance(module, Conv1D):
                        W = W.T
                except ImportError:
                    pass

                m, n = W.shape
                if not torch.isfinite(W).all():
                    continue
                try:
                    s_actual = torch.linalg.svdvals(W.float().cpu())
                    sigma_base = s_actual[0].item()
                    mp_edge_base = mp_bulk_edge(m, n, whitened=False)
                    r_mp_base = sigma_base / max(mp_edge_base, 1e-12)
                    mp_stats[name] = {
                        "mp_bulk_edge_base": mp_edge_base,
                        "r_mp_base": r_mp_base,
                        "sigma_base": sigma_base,
                    }
                except (RuntimeError, torch.linalg.LinAlgError):
                    continue
                break

    return mp_stats


def _iter_transformer_layers(model: nn.Module):
    """Iterate over transformer layers in a model."""
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        h_layers = model.transformer.h
        if hasattr(h_layers, "__iter__") and hasattr(h_layers, "__len__"):
            try:
                for layer in h_layers:
                    yield layer
            except (TypeError, AttributeError):
                pass
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
        if hasattr(layers, "__iter__") and hasattr(layers, "__len__"):
            try:
                for layer in layers:
                    yield layer
            except (TypeError, AttributeError):
                pass
    elif hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
        layer_attr = model.encoder.layer
        if hasattr(layer_attr, "__iter__") and hasattr(layer_attr, "__len__"):
            try:
                for layer in layer_attr:
                    yield layer
            except (TypeError, AttributeError):
                pass
    else:
        for module in model.modules():
            if hasattr(module, "attn") and hasattr(module, "mlp"):
                yield module


def analyze_weight_distribution(model: nn.Module, n_bins: int = 50) -> dict[str, Any]:
    """Analyze global weight-distribution statistics for model 2D weight matrices."""
    all_weights = []
    all_singular_values = []

    for name, param in model.named_parameters():
        if param.ndim == 2 and "weight" in name:
            param_cpu = param.detach().cpu()
            if not torch.isfinite(param_cpu).all():
                continue
            all_weights.append(param_cpu.flatten())
            try:
                s = torch.linalg.svdvals(param_cpu.float())
                all_singular_values.append(s)
            except (RuntimeError, torch.linalg.LinAlgError):
                continue

    if not all_weights:
        return {}

    weights = torch.cat(all_weights)
    stats = {
        "mean": weights.mean().item(),
        "std": weights.std().item(),
        "min": weights.min().item(),
        "max": weights.max().item(),
        "sparsity": (weights.abs() < 1e-6).float().mean().item(),
    }

    hist, edges = torch.histogram(weights, bins=n_bins)
    stats["histogram"] = hist.tolist()
    stats["bin_edges"] = edges.tolist()

    if all_singular_values:
        s_all = torch.cat(all_singular_values)
        stats["singular_values"] = {
            "mean": s_all.mean().item(),
            "std": s_all.std().item(),
            "min": s_all.min().item(),
            "max": s_all.max().item(),
            "condition_number": (s_all.max() / (s_all.min() + 1e-8)).item(),
        }

        n_samples: float = sum(s.shape[0] for s in all_singular_values)
        n_features: float = sum(s.shape[0] for s in all_singular_values) / float(
            len(all_singular_values)
        )
        mp_min, mp_max = mp_bulk_edges(int(n_samples), int(n_features))
        stats["mp_edges"] = {"min": mp_min, "max": mp_max}
        stats["eigenvalue_stats"] = stats["singular_values"]

    return stats
