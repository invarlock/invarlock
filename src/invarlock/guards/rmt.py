"""
InvarLock – Safety: Random Matrix Theory (RMT) Health Check
=======================================================

Detect-only mode for v0: identifies singular value outliers that
deviate from the Marchenko-Pastur bulk distribution.

Based on insights from Słowik et al., 2025 linking MP outliers
to training instability.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal, TypedDict

import numpy as np
import torch
import torch.linalg as tla
import torch.nn as nn

from invarlock.cli._evidence import maybe_dump_guard_evidence
from invarlock.core.api import Guard

from ._contracts import guard_assert

__all__ = [
    # Utility functions
    "mp_bulk_edges",
    "mp_bulk_edge",
    "layer_svd_stats",
    "rmt_detect",
    "rmt_detect_report",
    "rmt_detect_with_names",
    "clip_full_svd",
    "analyze_weight_distribution",
    "rmt_growth_ratio",
    "within_deadband",
    "capture_baseline_mp_stats",
    # Guard classes and types
    "RMTGuard",
    "RMTPolicy",
    "RMTPolicyDict",
    # Policy utilities
    "get_rmt_policy",
    "create_custom_rmt_policy",
]


def mp_bulk_edges(m: int, n: int, whitened: bool = True) -> tuple[float, float]:
    """
    Compute Marchenko-Pastur bulk edges for an m×n matrix.

    For a weight matrix W ∈ ℝ^{m×n}, the MP distribution describes
    the eigenvalues of (W^T W)/m when entries are i.i.d. with variance 1/m.

    Args:
        m: Number of rows (input features for Conv1D)
        n: Number of columns (output features for Conv1D)
        whitened: If True, assumes W is already whitened by √m

    Returns:
        (σ_min, σ_max) theoretical bulk edges for singular values
    """
    if m == 0 or n == 0:
        return 0.0, 0.0

    # q = n/m (aspect ratio)
    q = n / m

    if whitened:
        # For whitened matrix W/√m, singular values follow MP with:
        sigma_max = 1.0 + np.sqrt(q)
        sigma_min = abs(1.0 - np.sqrt(q)) if q <= 1 else 0.0
    else:
        # For unwhitened matrix, scale by √m
        sigma_max = np.sqrt(m) * (1.0 + np.sqrt(q))
        sigma_min = np.sqrt(m) * abs(1.0 - np.sqrt(q)) if q <= 1 else 0.0

    return sigma_min, sigma_max


def mp_bulk_edge(m: int, n: int, whitened: bool = False) -> float:
    """
    Compute Marchenko-Pastur bulk edge for an m×n matrix.

    This function computes the upper edge (maximum singular value) of the
    Marchenko-Pastur distribution, which represents the theoretical maximum
    singular value for a random matrix with i.i.d. entries.

    Args:
        m: Number of rows (input features for Conv1D)
        n: Number of columns (output features for Conv1D)
        whitened: If True, assumes W is already whitened by √m

    Returns:
        σ_max theoretical bulk edge for singular values
    """
    if m == 0 or n == 0:
        return 0.0

    # q = n/m (aspect ratio)
    q = n / m

    if whitened:
        # For whitened matrix W/√m, singular values follow MP with:
        sigma_max = 1.0 + np.sqrt(q)
    else:
        # For unwhitened matrix, scale by √m
        sigma_max = np.sqrt(m) * (1.0 + np.sqrt(q))

    return float(sigma_max)


def _iter_weight_matrices(layer: nn.Module):
    """Iterate over 2D weight matrices in a layer."""
    for name, param in layer.named_parameters():
        if param.ndim == 2 and "weight" in name:
            yield name, param.detach()


def rmt_growth_ratio(
    sigma_cur: float, mp_cur: float, sigma_base: float, mp_base: float
) -> float:
    """
    Compute baseline-aware growth ratio for RMT outlier detection.

    Compares the growth of σ/mp_edge ratio relative to baseline.

    Args:
        sigma_cur: Current maximum singular value
        mp_cur: Current MP bulk edge
        sigma_base: Baseline maximum singular value
        mp_base: Baseline MP bulk edge

    Returns:
        Growth ratio: (σ_cur / mp_cur) / (σ_base / mp_base)
    """
    r_base = sigma_base / max(mp_base, 1e-12)
    r_cur = sigma_cur / max(mp_cur, 1e-12)
    return r_cur / max(r_base, 1e-12)


def within_deadband(sigma_cur: float, sigma_base: float, deadband: float) -> bool:
    """
    Check if current sigma is within deadband of baseline.

    Args:
        sigma_cur: Current spectral norm
        sigma_base: Baseline spectral norm
        deadband: Deadband threshold (e.g., 0.1 for 10%)

    Returns:
        True if within deadband threshold
    """
    return sigma_cur <= (1.0 + deadband) * sigma_base


def layer_svd_stats(
    layer: nn.Module,
    baseline_sigmas: dict[str, float] | None = None,
    baseline_mp_stats: dict[str, dict[str, float]] | None = None,
    module_name: str | None = None,
) -> dict[str, float]:
    """
    Compute SVD statistics for a single layer with baseline-aware normalization.

    For HuggingFace Conv1D layers:
    - Weight shape is (in_features, out_features)
    - m = in_features, n = out_features

    Args:
        layer: Transformer layer to analyze
        baseline_sigmas: Optional baseline singular values for baseline-aware comparison
        baseline_mp_stats: Optional baseline MP statistics (mp_bulk_edge, r_mp_base) for each weight matrix
        module_name: Optional module name for baseline lookups

    Returns:
        Dict with sigma_min, sigma_max, worst_ratio
    """
    sigma_min_global = float("inf")
    sigma_max_global = 0.0
    worst_ratio = 0.0
    worst_details = None

    for name, W in _iter_weight_matrices(layer):
        if W.numel() == 0:
            continue
        if not torch.isfinite(W).all():
            continue

        # For Conv1D: W.shape = (in_features, out_features)
        m, n = W.shape  # m = in_features, n = out_features

        # Compute singular values of the actual matrix
        try:
            s_actual = tla.svdvals(W.float().cpu())
            s_min = s_actual[-1].item()
            s_max = s_actual[0].item()
        except (RuntimeError, torch.linalg.LinAlgError):
            continue

        # Track global min/max
        sigma_min_global = min(sigma_min_global, s_min)
        sigma_max_global = max(sigma_max_global, s_max)

        # Baseline-aware ratio computation for better outlier detection
        if baseline_sigmas and module_name and module_name in baseline_sigmas:
            # Use baseline-aware growth ratio (preferred method)
            baseline_sigma = baseline_sigmas[module_name]
            if baseline_sigma > 0:
                # Compute current MP edge
                mp_edge_current = mp_bulk_edge(m, n, whitened=False)

                # Get baseline MP edge from stored stats, or fallback to current
                if baseline_mp_stats and module_name in baseline_mp_stats:
                    mp_edge_baseline = baseline_mp_stats[module_name].get(
                        "mp_bulk_edge_base", mp_edge_current
                    )
                else:
                    # Fallback: assume same shape so use same MP edge
                    mp_edge_baseline = mp_edge_current

                # Use new helper function for consistent growth ratio calculation
                ratio = rmt_growth_ratio(
                    s_max, mp_edge_current, baseline_sigma, mp_edge_baseline
                )
            else:
                ratio = 1.0
        else:
            # Fallback: Use quantile-based normalization when no baseline available
            if len(s_actual) > 1:
                # Use 98th percentile as robust baseline (less sensitive to outliers)
                s_sorted = s_actual.sort()[0]
                idx_98 = int(0.98 * len(s_sorted))
                s_98 = s_sorted[idx_98].item()

                if s_98 > 0:
                    # Ratio relative to 98th percentile
                    ratio = s_max / s_98
                else:
                    ratio = 1.0
            else:
                # Single singular value
                ratio = 1.0

        # Track worst deviation
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


def capture_baseline_mp_stats(model: nn.Module) -> dict[str, dict[str, float]]:
    """
    Capture baseline MP statistics for linear layers only.

    CRITICAL: Only includes layers where MP analysis makes sense:
    - attn.c_attn, attn.c_proj, mlp.c_fc, mlp.c_proj
    - EXCLUDES: wte, wpe, lm_head, layer norms, biases

    Stores mp_bulk_edge and r_mp_base (sigma/mp_edge ratio) for each weight matrix.
    This enables true baseline-aware RMT detection.

    Args:
        model: Model to analyze

    Returns:
        Dict mapping module names to their MP statistics:
        {
            'module_name': {
                'mp_bulk_edge_base': float,
                'r_mp_base': float,
                'sigma_base': float
            }
        }
    """
    mp_stats = {}

    # Get all modules with 2D weight matrices
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

    # Define allowlist for RMT analysis - only linear layers where MP makes sense
    allowed_suffixes = [".attn.c_attn", ".attn.c_proj", ".mlp.c_fc", ".mlp.c_proj"]

    for name, module in model.named_modules():
        if isinstance(module, module_types) and hasattr(module, "weight"):
            # CRITICAL: Restrict to only linear layers where MP analysis is meaningful
            # Skip embeddings, LM head, layer norms - MP heuristics don't apply there
            if any(name.endswith(suffix) for suffix in allowed_suffixes):
                # Get 2D weight matrix
                for param_name, param in module.named_parameters(recurse=False):
                    if param.ndim == 2 and "weight" in param_name:
                        W = param.detach()

                        # Handle Conv1D transposition
                        try:
                            from transformers.pytorch_utils import Conv1D

                            if isinstance(module, Conv1D):
                                W = W.T
                        except ImportError:
                            pass

                        if W.ndim == 2:
                            m, n = W.shape

                            # Compute current sigma and MP edge
                            if not torch.isfinite(W).all():
                                continue
                            try:
                                s_actual = torch.linalg.svdvals(W.float().cpu())
                                sigma_base = s_actual[0].item()
                                mp_edge_base = mp_bulk_edge(m, n, whitened=False)

                                # Compute baseline r_mp ratio
                                r_mp_base = sigma_base / max(mp_edge_base, 1e-12)

                                # Store statistics with consistent naming
                                mp_stats[name] = {
                                    "mp_bulk_edge_base": mp_edge_base,
                                    "r_mp_base": r_mp_base,
                                    "sigma_base": sigma_base,
                                }
                            except (RuntimeError, torch.linalg.LinAlgError):
                                # Skip if SVD fails
                                continue
                        break  # Only process first weight parameter

    return mp_stats


def _iter_transformer_layers(model: nn.Module):
    """Iterate over transformer layers in a model."""
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        # GPT-2 style
        h_layers = model.transformer.h
        if hasattr(h_layers, "__iter__") and hasattr(h_layers, "__len__"):
            try:
                for layer in h_layers:
                    yield layer
            except (TypeError, AttributeError):
                pass
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        # LLaMA style
        layers = model.model.layers
        if hasattr(layers, "__iter__") and hasattr(layers, "__len__"):
            try:
                for layer in layers:
                    yield layer
            except (TypeError, AttributeError):
                pass
    elif hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
        # BERT style
        layer_attr = model.encoder.layer
        if hasattr(layer_attr, "__iter__") and hasattr(layer_attr, "__len__"):
            try:
                for layer in layer_attr:
                    yield layer
            except (TypeError, AttributeError):
                pass
    else:
        # Fallback
        for module in model.modules():
            if hasattr(module, "attn") and hasattr(module, "mlp"):
                yield module


def rmt_detect(
    model: nn.Module,
    threshold: float = 1.5,
    detect_only: bool = True,
    correction_factor: float | None = None,
    layer_indices: list[int] | None = None,
    target_layers: list[str] | None = None,  # Alternative layer specification
    verbose: bool = False,
    max_iterations: int = 2,  # Add iteration guard
    baseline_sigmas: dict[str, float]
    | None = None,  # Add baseline sigmas for baseline-aware checking
    baseline_mp_stats: dict[str, dict[str, float]]
    | None = None,  # Store baseline MP statistics
    deadband: float = 0.0,  # Add deadband parameter to align with spectral control
    use_quantile_mp: bool = False,  # Use quantile-based MP edge for heavy-tailed spectra
) -> dict[str, Any]:
    """
    Detect RMT outliers in model with baseline-aware checking and iteration guard.

    Args:
        model: Model to analyze
        threshold: Ratio threshold for flagging outliers (default 1.5)
        detect_only: If True, only detect outliers without correction
        correction_factor: Factor to apply for correction (if not detect_only)
        layer_indices: Specific layers to analyze by index (None = all)
        target_layers: Specific layers to analyze by name (None = all)
        verbose: Whether to print warnings and details
        max_iterations: Maximum iterations for correction (default 2)
        baseline_sigmas: Baseline sigmas for baseline-aware checking
        baseline_mp_stats: Baseline MP statistics (mp_bulk_edge, r_mp_base) for each weight matrix
        deadband: Deadband threshold to align with spectral control
        use_quantile_mp: Use quantile-based MP edge for heavy-tailed spectra

    Returns:
        Dict with detection results including per-layer details
    """
    per_layer: list[dict[str, Any]] = []
    flagged_layers: list[int] = []

    # Analyze only linear layers where MP analysis is meaningful
    modules_to_analyze = []

    # Define allowlist for RMT analysis - same as in capture_baseline_mp_stats
    allowed_suffixes = [".attn.c_attn", ".attn.c_proj", ".mlp.c_fc", ".mlp.c_proj"]

    if layer_indices is not None or target_layers is not None:
        # If specific layers requested, only analyze transformer layers
        for idx, layer in enumerate(_iter_transformer_layers(model)):
            # Skip if not in specified layers (by index)
            if layer_indices is not None and idx not in layer_indices:
                continue

            # Skip if not in specified layers (by name)
            if target_layers is not None:
                layer_name = None
                for name, module in model.named_modules():
                    if module is layer:
                        layer_name = name
                        break
                if layer_name is None or not any(
                    target in layer_name for target in target_layers
                ):
                    continue

            modules_to_analyze.append((f"transformer_layer_{idx}", layer))
    else:
        # CRITICAL: Only analyze modules where MP analysis makes sense
        # Exclude embeddings, LM head, layer norms - they have different spectral properties
        for name, module in model.named_modules():
            # Check if this is an allowed module type with 2D weights
            if any(name.endswith(suffix) for suffix in allowed_suffixes):
                has_2d_weights = any(
                    param.ndim == 2 and "weight" in param_name
                    for param_name, param in module.named_parameters(recurse=False)
                )
                if has_2d_weights:
                    modules_to_analyze.append((name, module))

    # Iteration guard for correction
    prev_outlier_count = float("inf")
    correction_iterations = 0

    while correction_iterations < max_iterations:
        current_outliers = 0
        per_layer = []  # Reset per iteration
        flagged_layers = []

        for idx, (module_name, module) in enumerate(modules_to_analyze):
            # Use baseline-aware stats if available
            stats = layer_svd_stats(
                module, baseline_sigmas, baseline_mp_stats, module_name
            )

            # Apply baseline-aware RMT detection with deadband support
            has_outlier = False
            skip_reason = None

            if (
                baseline_sigmas
                and baseline_mp_stats
                and module_name in baseline_sigmas
                and module_name in baseline_mp_stats
            ):
                # Step 5 spec: ratio = σ_max_post / bulk_edge_base, flag if ratio > (1+deadband)*margin
                sigma_post = stats["sigma_max"]
                mp_stats = baseline_mp_stats[module_name]
                bulk_edge_base = mp_stats.get("mp_bulk_edge_base", 1.0)

                # Exact Step 5 detection rule
                ratio = sigma_post / max(bulk_edge_base, 1e-12)
                detection_threshold = (1.0 + deadband) * threshold

                if ratio > detection_threshold:
                    has_outlier = True
                    skip_reason = None
                else:
                    # Determine skip reason for clear logging
                    skip_reason = (
                        f"≤ threshold (ratio={ratio:.2f} ≤ {detection_threshold:.2f})"
                    )
            elif deadband > 0.0 and baseline_sigmas and module_name in baseline_sigmas:
                # Partial baseline-aware: deadband check only (fallback when no MP stats)
                baseline_sigma = baseline_sigmas[module_name]
                sigma_post = stats["sigma_max"]
                ratio = sigma_post / max(baseline_sigma, 1e-12)
                detection_threshold = (1.0 + deadband) * threshold

                if ratio > detection_threshold:
                    has_outlier = True
                    skip_reason = None
                else:
                    skip_reason = (
                        f"≤ threshold (ratio={ratio:.2f} ≤ {detection_threshold:.2f})"
                    )
            else:
                # Standard check without baseline awareness (fallback)
                ratio = stats["worst_ratio"]
                if ratio > threshold:
                    has_outlier = True
                    skip_reason = None
                else:
                    skip_reason = f"≤ threshold (ratio={ratio:.2f} ≤ {threshold:.2f})"

            layer_info = {
                "layer": idx,
                "module_name": module_name,
                "sigma_min": stats["sigma_min"],
                "sigma_max": stats["sigma_max"],
                "worst_ratio": stats["worst_ratio"],
                "has_outlier": has_outlier,
            }

            # Add detailed info if available
            if "worst_details" in stats:
                layer_info["details"] = stats["worst_details"]

            per_layer.append(layer_info)

            # Store skip reason in layer info for better logging
            layer_info["skip_reason"] = skip_reason

            if has_outlier:
                flagged_layers.append(idx)
                current_outliers += 1
                if verbose:
                    normalization = stats.get("worst_details", {}).get(
                        "normalization", "unknown"
                    )
                    print(
                        f"      Module {module_name}: ratio={stats['worst_ratio']:.2f} "
                        f"(σ_max={stats['sigma_max']:.2f}, norm={normalization})"
                    )
            elif verbose and skip_reason:
                print(f"      Module {module_name}: SKIP: {skip_reason}")

        # Apply correction if requested and not detect-only
        if not detect_only and current_outliers > 0 and correction_factor is not None:
            if correction_iterations == 0:
                if verbose:
                    print(
                        f"    Applying RMT correction (iteration {correction_iterations + 1})..."
                    )
                # Apply correction to flagged modules
                for idx in flagged_layers:
                    module_name, module = modules_to_analyze[idx]
                    _apply_rmt_correction(
                        module,
                        correction_factor,
                        baseline_sigmas,
                        baseline_mp_stats,
                        module_name,
                        deadband,
                        verbose,
                        adapter=None,
                    )
            else:
                # Check if improvement occurred
                if current_outliers >= prev_outlier_count:
                    if verbose:
                        print(
                            f"    RMT correction stalled ({current_outliers} outliers unchanged), "
                            f"downgrading to warning"
                        )
                    break
                elif verbose:
                    print(
                        f"    RMT correction improving ({prev_outlier_count} → {current_outliers} outliers)"
                    )
        else:
            # No correction requested, exit after first iteration
            break

        prev_outlier_count = current_outliers
        correction_iterations += 1

        # Exit if no outliers remain
        if current_outliers == 0:
            break

    # Aggregate results
    n_outliers = len(flagged_layers)
    max_ratio = max((item["worst_ratio"] for item in per_layer), default=0.0)
    has_outliers = n_outliers > 0

    if verbose and has_outliers:
        baseline_note = (
            " (baseline-aware)"
            if baseline_sigmas and baseline_mp_stats
            else " (absolute)"
        )
        deadband_note = f" with {deadband:.0%} deadband" if deadband > 0.0 else ""

        # Count detected vs will-be-capped
        n_detected = n_outliers
        n_will_be_capped = n_outliers if not detect_only else 0

        print(f"    ⚠️ RMT outliers detected{baseline_note}{deadband_note}:")
        print(f"      Detected: {n_detected}, will correct: {n_will_be_capped}")
        print(f"      Max ratio: {max_ratio:.2f}")
        print("      Top offenders (σ_post / σ_ref):")

        # Show top 3 offenders with detailed information
        top_offenders = sorted(
            [
                (item["worst_ratio"], item["module_name"], item.get("details", {}))
                for item in per_layer
                if item["has_outlier"]
            ],
            reverse=True,
        )[:3]

        for ratio, module_name, details in top_offenders:
            sigma_max = details.get("s_max", 0.0)
            ref_type = "mp_bulk_edge" if not baseline_sigmas else "baseline-aware"
            print(
                f"        - {module_name}: {ratio:.2f} (σ_post={sigma_max:.2f}, ref={ref_type})"
            )

        if len(top_offenders) < n_outliers:
            print(
                f"      ... and {n_outliers - len(top_offenders)} more layers flagged"
            )

    return {
        "has_outliers": has_outliers,
        "n_layers_flagged": n_outliers,
        "outlier_count": n_outliers,  # Alias for compatibility
        "max_ratio": max_ratio,
        "threshold": threshold,
        "correction_iterations": correction_iterations,
        "per_layer": per_layer,
        "flagged_layers": flagged_layers,
        "layers": {
            f"layer_{item['layer']}": item for item in per_layer
        },  # Alternative format
    }


def rmt_detect_report(
    model: nn.Module, threshold: float = 1.5
) -> tuple[dict, list[dict]]:
    """
    Generate an RMT health report.

    Args:
        model: Model to analyze
        threshold: Ratio threshold for outliers

    Returns:
        (summary_dict, per_layer_list)
    """
    result = rmt_detect(model, threshold, verbose=False)

    summary = {
        "has_outliers": result["has_outliers"],
        "n_layers_flagged": result["n_layers_flagged"],
        "max_ratio": result["max_ratio"],
        "rmt_max_ratio": result["max_ratio"],  # Alias for compatibility
        "rmt_has_outliers": result["has_outliers"],  # Alias
    }

    return summary, result["per_layer"]


def rmt_detect_with_names(
    model: nn.Module, threshold: float = 1.5, verbose: bool = False
) -> dict[str, Any]:
    """
    Detect RMT outliers in model and return detailed information with module names.

    Args:
        model: Model to analyze
        threshold: Ratio threshold for flagging outliers (default 1.5)
        verbose: Whether to print warnings and details

    Returns:
        Dict with detection results including per-layer details and module names
    """
    outliers = []
    per_layer = []
    flagged_layers = []

    # Get all transformer layers with their names
    layer_modules = []
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        # GPT-2 style
        h_layers = model.transformer.h
        if hasattr(h_layers, "__iter__"):
            for idx, layer in enumerate(h_layers):
                layer_modules.append((f"transformer.h.{idx}", layer))
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        # LLaMA style
        layers = model.model.layers
        if hasattr(layers, "__iter__"):
            for idx, layer in enumerate(layers):
                layer_modules.append((f"model.layers.{idx}", layer))
    elif hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
        # BERT style
        layer_attr = model.encoder.layer
        if hasattr(layer_attr, "__iter__"):
            for idx, layer in enumerate(layer_attr):
                layer_modules.append((f"encoder.layer.{idx}", layer))
    else:
        # Fallback - try to find transformer layers by attributes
        for name, module in model.named_modules():
            if hasattr(module, "attn") and hasattr(module, "mlp"):
                layer_modules.append((name, module))

    for layer_name, layer in layer_modules:
        stats = layer_svd_stats(layer, module_name=layer_name)

        # Check if layer has outliers
        has_outlier = stats["worst_ratio"] > threshold

        # Add detailed info if available
        if "worst_details" in stats:
            layer_info = {
                "layer_name": layer_name,
                "sigma_min": stats["sigma_min"],
                "sigma_max": stats["sigma_max"],
                "worst_ratio": stats["worst_ratio"],
                "has_outlier": has_outlier,
                "details": stats["worst_details"],
            }

            # Add module name to outlier details
            if has_outlier:
                outlier_info = {
                    "layer_name": layer_name,
                    "module_name": f"{layer_name}.{stats['worst_details']['name']}",
                    "sigma_max": stats["sigma_max"],
                    "ratio": stats["worst_ratio"],
                    "details": stats["worst_details"],
                }
                outliers.append(outlier_info)
                flagged_layers.append(layer_name)
        else:
            layer_info = {
                "layer_name": layer_name,
                "sigma_min": stats["sigma_min"],
                "sigma_max": stats["sigma_max"],
                "worst_ratio": stats["worst_ratio"],
                "has_outlier": has_outlier,
            }

        per_layer.append(layer_info)

    # Aggregate results
    n_outliers = len(flagged_layers)
    max_ratio = 0.0
    if per_layer:
        try:
            max_ratio = max(float(item.get("worst_ratio", 0.0)) for item in per_layer)
        except (TypeError, ValueError):
            max_ratio = 0.0
    has_outliers = n_outliers > 0

    if verbose and has_outliers:
        print("    ⚠️ RMT outliers detected:")
        print(f"      Layers flagged: {n_outliers}")
        print(f"      Max ratio: {max_ratio:.2f}")
        print(f"      Threshold: {threshold:.2f}")
        print("      Top offenders (σ_post / σ_ref):")
        # Show top offenders with full module names and consistent formatting
        for outlier in outliers[:3]:
            print(
                f"        - {outlier['module_name']}: {outlier['ratio']:.2f} (σ_post={outlier['sigma_max']:.2f}, ref=mp_bulk_edge)"
            )
        if len(outliers) > 3:
            print(f"      ... and {len(outliers) - 3} more layers flagged")

    return {
        "has_outliers": has_outliers,
        "n_layers_flagged": n_outliers,
        "outlier_count": n_outliers,
        "max_ratio": max_ratio,
        "threshold": threshold,
        "per_layer": per_layer,
        "flagged_layers": flagged_layers,
        "outliers": outliers,  # Add the outliers list with full module names
        "layers": {item["layer_name"]: item for item in per_layer},
    }


def _apply_rmt_correction(
    layer: nn.Module,
    factor: float,
    baseline_sigmas: dict[str, float] | None = None,
    baseline_mp_stats: dict[str, dict[str, float]] | None = None,
    layer_name: str = "",
    deadband: float = 0.0,
    verbose: bool = False,
    adapter=None,
):
    """
    Apply RMT-based correction to layer weights with proper cap application.

    Enhanced for Step 5 with:
    - Step 5 detection rule: target = bulk_edge_base * margin * (1 - deadband)
    - Adapter tying map support for preserving weight tying relationships
    - IN-PLACE scaling (param.mul_) to preserve weight tying
    - Never rewraps Parameters to avoid breaking lm_head ↔ wte aliasing
    """
    for name, param in layer.named_parameters():
        if param.ndim == 2 and "weight" in name:
            with torch.no_grad():
                # Get current spectral norm
                try:
                    W = param.detach()
                    # Handle Conv1D transposition
                    Conv1D = None
                    try:
                        from transformers.pytorch_utils import Conv1D as _Conv1D

                        Conv1D = _Conv1D

                        if isinstance(layer, Conv1D):
                            W = W.T
                    except ImportError:
                        pass

                    if not torch.isfinite(W).all():
                        continue
                    s_vals = torch.linalg.svdvals(W.float().cpu())
                    sigma_pre = s_vals[0].item()

                    # Step 5 correction logic: target based on MP bulk edge
                    target_sigma = None

                    if (
                        baseline_sigmas
                        and baseline_mp_stats
                        and layer_name in baseline_mp_stats
                    ):
                        # CORRECTED Step 5: Use baseline sigma for target calculation
                        mp_stats = baseline_mp_stats[layer_name]
                        sigma_base = mp_stats.get("sigma_base", 1.0)

                        # Step 5 target: baseline * margin * (1 - deadband) for conservative correction
                        margin = (
                            1.5  # Default from policy, could be passed as parameter
                        )
                        target_sigma = sigma_base * margin * (1.0 - deadband)
                    else:
                        # Fallback: Use current MP edge
                        m, n = W.shape
                        mp_edge = mp_bulk_edge(m, n, whitened=False)
                        target_sigma = mp_edge * 1.0  # Conservative cap at edge

                    # Apply correction only if needed
                    if sigma_pre > target_sigma:
                        # Compute proper scale: target/σ_pre
                        scale = target_sigma / sigma_pre
                        scale = max(
                            scale, 0.1
                        )  # Floor at 10% to avoid extreme shrinkage

                        # Check for tied parameters using adapter's tying map
                        tied_params = []
                        if adapter and hasattr(adapter, "get_tying_map"):
                            try:
                                tying_map = adapter.get_tying_map()
                                full_param_name = f"{layer_name}.{name}"
                                tied_params = tying_map.get(full_param_name, [])
                            except Exception:
                                # Fallback if adapter doesn't support tying map
                                tied_params = []

                        # CRITICAL: Apply IN-PLACE scaling to preserve weight tying
                        param.mul_(scale)  # PRESERVES TYING - same data pointer

                        # Apply same scaling to tied parameters if any
                        if tied_params and adapter:
                            for tied_name in tied_params:
                                try:
                                    # Get tied parameter and apply same scale
                                    tied_param = adapter.get_parameter_by_name(
                                        tied_name
                                    )
                                    if tied_param is not None:
                                        tied_param.mul_(scale)
                                except Exception:
                                    # Continue if tied parameter access fails
                                    pass

                        # Recompute sigma after scaling for accurate logging
                        W_after = param.detach()
                        if Conv1D is not None and isinstance(layer, Conv1D):
                            W_after = W_after.T
                        s_vals_after = torch.linalg.svdvals(W_after.float().cpu())
                        sigma_post = s_vals_after[0].item()

                        # Log the correction with proper values
                        if verbose:
                            tied_info = (
                                f", tied to {len(tied_params)} params"
                                if tied_params
                                else ""
                            )
                            print(
                                f"      {layer_name}.{name}: σ={sigma_pre:.2f}→{sigma_post:.2f} "
                                f"(scale={scale:.3f}, target={target_sigma:.2f}{tied_info})"
                            )
                    else:
                        # No correction needed - log skip reason
                        if verbose:
                            print(
                                f"      {layer_name}.{name}: SKIP: ≤ target (σ={sigma_pre:.2f} ≤ {target_sigma:.2f})"
                            )

                except (RuntimeError, torch.linalg.LinAlgError):
                    # CRITICAL: Even fallback must use in-place scaling
                    param.mul_(factor)
                    if verbose:
                        print(
                            f"      {layer_name}.{name}: fallback scaling (SVD failed)"
                        )


def clip_full_svd(
    W: torch.Tensor, clip_val: float, return_components: bool = False
) -> torch.Tensor:
    """
    Clip singular values of a matrix using full SVD.

    Args:
        W: Weight matrix
        clip_val: Maximum singular value
        return_components: If True, return (U, S_clipped, Vt)

    Returns:
        Clipped weight matrix or components
    """
    if not torch.isfinite(W).all():
        if return_components:
            return None, None, None
        return W

    try:
        U, S, Vt = torch.linalg.svd(W.float(), full_matrices=False)
        S_clipped = torch.clamp(S, max=clip_val)

        if return_components:
            return U, S_clipped, Vt
        else:
            return (U @ torch.diag(S_clipped) @ Vt).to(W.dtype)
    except (RuntimeError, torch.linalg.LinAlgError):
        # Return original on error
        if return_components:
            return None, None, None
        return W


def analyze_weight_distribution(model: nn.Module, n_bins: int = 50) -> dict[str, Any]:
    """
    Analyze weight distribution statistics for RMT analysis.

    Args:
        model: Model to analyze
        n_bins: Number of histogram bins

    Returns:
        Dict with distribution statistics
    """
    all_weights = []
    all_singular_values = []

    for name, param in model.named_parameters():
        if param.ndim == 2 and "weight" in name:
            param_cpu = param.detach().cpu()
            if not torch.isfinite(param_cpu).all():
                continue

            # Collect weights
            all_weights.append(param_cpu.flatten())

            # Collect singular values
            try:
                s = torch.linalg.svdvals(param_cpu.float())
                all_singular_values.append(s)
            except (RuntimeError, torch.linalg.LinAlgError):
                continue

    if not all_weights:
        return {}

    # Concatenate all weights
    weights = torch.cat(all_weights)

    # Compute statistics
    stats = {
        "mean": weights.mean().item(),
        "std": weights.std().item(),
        "min": weights.min().item(),
        "max": weights.max().item(),
        "sparsity": (weights.abs() < 1e-6).float().mean().item(),
    }

    # Compute histogram
    hist, edges = torch.histogram(weights, bins=n_bins)
    stats["histogram"] = hist.tolist()
    stats["bin_edges"] = edges.tolist()

    # Singular value statistics
    if all_singular_values:
        s_all = torch.cat(all_singular_values)
        singular_values_dict: dict[str, float] = {
            "mean": s_all.mean().item(),
            "std": s_all.std().item(),
            "min": s_all.min().item(),
            "max": s_all.max().item(),
            "condition_number": (s_all.max() / (s_all.min() + 1e-8)).item(),
        }
        stats["singular_values"] = singular_values_dict

    # Add MP edge information
    if all_singular_values:
        # Estimate MP edges from data
        n_samples: float = sum(s.shape[0] for s in all_singular_values)
        n_features: float = np.mean([s.shape[0] for s in all_singular_values])
        mp_min, mp_max = mp_bulk_edges(int(n_samples), int(n_features))
        mp_edges_dict: dict[str, float] = {"min": mp_min, "max": mp_max}
        stats["mp_edges"] = mp_edges_dict

        # Add eigenvalue stats (alias for singular values)
        stats["eigenvalue_stats"] = stats["singular_values"]

    return stats


# === Guard Implementation ===

# Import GuardOutcome types if available
try:
    from invarlock.core.types import GuardOutcome

    HAS_GUARD_OUTCOME = True
except ImportError:
    # Fallback for standalone usage or when types not available
    HAS_GUARD_OUTCOME = False
    GuardOutcome = dict


@dataclass
class RMTPolicy:
    """
    RMT Guard Policy Configuration.

    Defines parameters for baseline-aware RMT outlier detection and correction.
    """

    q: float | Literal["auto"] = (
        "auto"  # MP aspect ratio m/n (auto-derived from weights)
    )
    deadband: float = 0.10  # Tolerance margin (10%)
    margin: float = 1.5  # RMT threshold ratio
    correct: bool = True  # Enable automatic correction


class RMTPolicyDict(TypedDict):
    """TypedDict version of RMTPolicy for compatibility."""

    q: float | Literal["auto"]
    deadband: float
    margin: float
    correct: bool
    epsilon: float | dict[str, float] | None


class RMTGuard(Guard):
    """
    Standalone RMT Guard for baseline-aware outlier detection and correction.

    Implements Marchenko-Pastur theory-based outlier tracking with:
    - Activation-based outlier counts from calibration batches
    - Baseline capture of MP bulk edges for linear layers (correction/fallback)
    - Conservative outlier detection with deadband support
    - Optional in-place correction preserving weight tying
    - Comprehensive event logging and metrics

    Policy Structure:
    - q: MP aspect ratio (auto-derived or manual)
    - deadband: Tolerance margin before flagging (default 0.10 = 10%)
    - margin: RMT threshold ratio (default 1.5)
    - correct: Enable automatic correction (default True)

    Linear Layer Scope (correction/fallback):
    - attn.c_attn, attn.c_proj, mlp.c_fc, mlp.c_proj
    - Excludes: embeddings, LM head, layer norms, biases
    """

    name = "rmt"

    def __init__(
        self,
        q: float | Literal["auto"] = "auto",
        deadband: float = 0.10,
        margin: float = 1.5,
        correct: bool = True,
        epsilon: float | dict[str, float] | None = None,
    ):
        """
        Initialize RMT Guard.

        Args:
            q: MP aspect ratio (auto-derived from weight shapes if "auto")
            deadband: Tolerance margin before flagging outliers (0.10 = 10%)
            margin: RMT threshold ratio for outlier detection (1.5)
            correct: Enable automatic correction when outliers detected
        """
        self.q = q
        self.deadband = deadband
        self.margin = margin
        self.correct = correct
        self.epsilon_default = 0.10
        self.epsilon_by_family: dict[str, float] = {}
        self._set_epsilon(epsilon)
        for family_key in ("attn", "ffn", "embed", "other"):
            self.epsilon_by_family.setdefault(family_key, self.epsilon_default)

        # Internal state
        self._calibration_batches: list[Any] = []
        self._activation_ready = False
        self._require_activation = False
        self._activation_required_failed = False
        self._activation_required_reason: str | None = None
        self._run_profile: str | None = None
        self._run_tier: str | None = None
        self.baseline_mp_stats: dict[str, dict[str, float]] | None = None
        self.baseline_sigmas: dict[str, float] | None = None
        self.prepared = False
        self.events: list[dict[str, Any]] = []
        self._last_result: dict[str, Any] | None = None
        self.adapter = None  # Store adapter for tying map access

        # Linear layer scope enforcement - same as existing RMT
        self.allowed_suffixes = [
            ".attn.c_attn",
            ".attn.c_proj",
            ".mlp.c_fc",
            ".mlp.c_proj",
        ]
        self.baseline_outliers_per_family: dict[str, int] = {}
        self.baseline_total_outliers: int = 0
        self.outliers_per_family: dict[str, int] = {}
        self.outliers_total: int = 0
        self.epsilon_violations: list[dict[str, Any]] = []

    def _log_event(
        self, operation: str, level: str = "INFO", message: str = "", **data
    ):
        """Log an event with timestamp."""
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "component": "rmt_guard",
            "operation": operation,
            "level": level,
            "message": message,
            "data": data,
        }
        self.events.append(event)

    def set_run_context(self, report: Any) -> None:
        """Capture tier/profile context for activation requirements."""
        ctx = getattr(report, "context", {}) or {}
        profile = ""
        tier = "balanced"
        if isinstance(ctx, dict):
            profile = str(ctx.get("profile", "") or "").strip().lower()
            auto = ctx.get("auto")
            if isinstance(auto, dict):
                tier = str(auto.get("tier", tier) or tier).strip().lower()
        self._run_profile = profile or None
        self._run_tier = tier or None
        self._require_activation = bool(
            profile in {"ci", "release"} and tier in {"balanced", "conservative"}
        )

    def _set_epsilon(self, epsilon: float | dict[str, float] | None) -> None:
        """Configure epsilon defaults and per-family overrides."""
        if isinstance(epsilon, dict):
            mapped: dict[str, float] = {}
            for family, value in epsilon.items():
                try:
                    mapped[str(family)] = float(value)
                except (TypeError, ValueError):
                    continue
            if mapped:
                self.epsilon_by_family.update(mapped)
                self.epsilon_default = max(mapped.values())
        elif isinstance(epsilon, int | float):
            self.epsilon_default = float(epsilon)
            if self.epsilon_by_family:
                for family in list(self.epsilon_by_family):
                    self.epsilon_by_family[family] = self.epsilon_default

    @staticmethod
    def _classify_family(module_name: str) -> str:
        """Classify module name into a guard family."""
        lower = module_name.lower()
        # MoE
        if any(
            tok in lower
            for tok in ("router", "routing", "gate", "gating", "dispatch", "switch")
        ):
            return "router"
        if any(
            tok in lower for tok in ("experts", "expert", "moe", "mixture_of_experts")
        ):
            return "expert_ffn"
        if ".attn." in lower or "attention" in lower:
            return "attn"
        if ".mlp." in lower or "ffn" in lower or ".c_fc" in lower:
            return "ffn"
        if "embed" in lower or "wte" in lower or "wpe" in lower:
            return "embed"
        return "other"

    def _count_outliers_per_family(
        self, per_layer: list[dict[str, Any]]
    ) -> dict[str, int]:
        """Count outliers grouped by family."""
        counts: dict[str, int] = {}
        for layer_info in per_layer:
            outlier_count = layer_info.get("outlier_count")
            if outlier_count is None:
                if not layer_info.get("has_outlier"):
                    continue
                increment = 1
            else:
                try:
                    increment = int(outlier_count)
                except (TypeError, ValueError):
                    continue
                if increment <= 0:
                    continue
            module_name = layer_info.get("module_name", "")
            family = self._classify_family(module_name)
            counts[family] = counts.get(family, 0) + increment
        return counts

    def _compute_epsilon_violations(self) -> list[dict[str, Any]]:
        """Compute epsilon-rule violations per family."""
        violations: list[dict[str, Any]] = []
        families = set(self.outliers_per_family) | set(
            self.baseline_outliers_per_family
        )
        for family in families:
            bare = int(self.baseline_outliers_per_family.get(family, 0) or 0)
            guarded = int(self.outliers_per_family.get(family, 0) or 0)
            epsilon_val = float(
                self.epsilon_by_family.get(family, self.epsilon_default)
            )
            allowed = math.ceil(bare * (1 + epsilon_val))
            if guarded > allowed:
                violations.append(
                    {
                        "family": family,
                        "bare": bare,
                        "guarded": guarded,
                        "allowed": allowed,
                        "epsilon": epsilon_val,
                    }
                )
        return violations

    def _get_linear_modules(self, model: nn.Module) -> list[tuple[str, nn.Module]]:
        """
        Get linear modules that are in scope for RMT analysis.

        Args:
            model: Model to analyze

        Returns:
            List of (name, module) tuples for linear layers in scope
        """
        modules = []

        # Get module types
        try:
            from transformers.pytorch_utils import Conv1D

            module_types_with_conv1d_2: tuple[
                type[nn.Linear], type[nn.Conv1d], type[Conv1D]
            ] = (nn.Linear, nn.Conv1d, Conv1D)
            module_types = module_types_with_conv1d_2
        except ImportError:
            module_types_without_conv1d_2: tuple[type[nn.Linear], type[nn.Conv1d]] = (
                nn.Linear,
                nn.Conv1d,
            )
            module_types = module_types_without_conv1d_2

        modules: list[tuple[str, nn.Module]] = []
        for name, module in model.named_modules():
            if isinstance(module, module_types) and hasattr(module, "weight"):
                # Strict scope enforcement - only allowed linear layers
                if any(name.endswith(suffix) for suffix in self.allowed_suffixes):
                    modules.append((name, module))

        return modules

    def _collect_calibration_batches(self, calib: Any, max_windows: int) -> list[Any]:
        """Collect a deterministic slice of calibration batches."""
        if calib is None or max_windows <= 0:
            return []
        source = getattr(calib, "dataloader", None) or calib
        try:
            iterator = iter(source)
        except TypeError:
            return []
        return list(itertools.islice(iterator, max_windows))

    def _prepare_activation_inputs(
        self, batch: Any, device: torch.device
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Normalize batch inputs to tensors on the target device."""
        if isinstance(batch, dict):
            input_ids = batch.get("input_ids", batch.get("inputs"))
            attention_mask = batch.get("attention_mask")
        elif isinstance(batch, tuple | list) and batch:
            input_ids = batch[0]
            attention_mask = batch[1] if len(batch) > 1 else None
        else:
            input_ids = batch
            attention_mask = None

        if input_ids is None:
            return None, None

        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.as_tensor(input_ids)
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        try:
            input_ids = input_ids.to(device)
        except Exception:
            input_ids = input_ids.clone()

        if attention_mask is not None:
            if not isinstance(attention_mask, torch.Tensor):
                attention_mask = torch.as_tensor(attention_mask)
            if attention_mask.dim() == 1:
                attention_mask = attention_mask.unsqueeze(0)
            try:
                attention_mask = attention_mask.to(device)
            except Exception:
                attention_mask = attention_mask.clone()

        return input_ids, attention_mask

    @staticmethod
    def _batch_token_weight(
        input_ids: torch.Tensor | None, attention_mask: torch.Tensor | None
    ) -> int:
        """Compute token-weight for a batch (used for activation outlier weighting)."""
        weight = 0
        if isinstance(attention_mask, torch.Tensor):
            try:
                weight = int(attention_mask.sum().item())
            except Exception:
                weight = 0
        if weight <= 0 and isinstance(input_ids, torch.Tensor):
            try:
                weight = int(input_ids.numel())
            except Exception:
                weight = 0
        return max(weight, 1)

    def _get_activation_modules(self, model: nn.Module) -> list[tuple[str, nn.Module]]:
        """Return modules to analyze for activation-based RMT."""
        modules: list[tuple[str, nn.Module]] = []
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

        for name, module in model.named_modules():
            if isinstance(module, nn.Embedding):
                modules.append((name, module))
                continue
            if isinstance(module, nn.LayerNorm):
                modules.append((name, module))
                continue
            if isinstance(module, module_types) and hasattr(module, "weight"):
                name_lower = name.lower()
                if any(
                    name.endswith(suffix) for suffix in self.allowed_suffixes
                ) or any(
                    tok in name_lower
                    for tok in (
                        "attn",
                        "attention",
                        "mlp",
                        "ffn",
                        "router",
                        "expert",
                        "moe",
                        "gate",
                        "gating",
                        "switch",
                    )
                ):
                    modules.append((name, module))

        return modules

    def _activation_svd_outliers(
        self, activations: Any, margin: float, deadband: float
    ) -> tuple[int, float, float]:
        """Count activation singular values beyond the MP edge."""
        if isinstance(activations, tuple | list):
            activations = activations[0] if activations else None
        if not isinstance(activations, torch.Tensor):
            return 0, 0.0, 0.0

        if activations.dim() < 2:
            return 0, 0.0, 0.0

        if activations.dim() > 2:
            activations = activations.reshape(-1, activations.shape[-1])

        if activations.numel() == 0:
            return 0, 0.0, 0.0

        try:
            mat = activations.detach().float().cpu()
        except Exception:
            return 0, 0.0, 0.0

        if not torch.isfinite(mat).all():
            return 0, 0.0, 0.0

        mat = mat - mat.mean()
        std = float(mat.std().item())
        if not math.isfinite(std) or std <= 0.0:
            return 0, 0.0, 0.0

        mat = mat / std
        m, n = mat.shape
        mp_edge_val = mp_bulk_edge(m, n, whitened=False)
        threshold = mp_edge_val * (1.0 + deadband) * margin

        try:
            s_vals = torch.linalg.svdvals(mat)
        except (RuntimeError, torch.linalg.LinAlgError):
            return 0, 0.0, 0.0

        if s_vals.numel() == 0:
            return 0, 0.0, 0.0

        sigma_max = float(s_vals.max().item())
        max_ratio = sigma_max / max(mp_edge_val, 1e-12)
        outlier_count = int((s_vals > threshold).sum().item())
        return outlier_count, float(max_ratio), sigma_max

    def _compute_activation_outliers(
        self, model: nn.Module, batches: list[Any]
    ) -> dict[str, Any] | None:
        """Compute activation-based RMT outlier counts."""
        if not batches:
            return None

        modules = self._get_activation_modules(model)
        if not modules:
            return None

        per_layer_map: dict[str, dict[str, Any]] = {}
        batch_weight_holder = {"weight": 1}
        for idx, (module_name, _module) in enumerate(modules):
            per_layer_map[module_name] = {
                "layer": idx,
                "module_name": module_name,
                "sigma_max": 0.0,
                "worst_ratio": 0.0,
                "outlier_count": 0,
                "has_outlier": False,
            }

        handles: list[Any] = []

        def _make_hook(name: str):
            def _hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any):
                try:
                    outliers, max_ratio, sigma_max = self._activation_svd_outliers(
                        output, self.margin, self.deadband
                    )
                except Exception:
                    return
                stats = per_layer_map.get(name)
                if stats is None:
                    return
                weight = int(batch_weight_holder.get("weight", 1) or 1)
                if outliers > 0:
                    increment = int(outliers) * weight
                    stats["outlier_count"] = (
                        int(stats.get("outlier_count", 0)) + increment
                    )
                    stats["has_outlier"] = True
                stats["worst_ratio"] = max(
                    float(stats.get("worst_ratio", 0.0)), float(max_ratio)
                )
                stats["sigma_max"] = max(
                    float(stats.get("sigma_max", 0.0)), float(sigma_max)
                )

            return _hook

        for name, module in modules:
            try:
                handles.append(module.register_forward_hook(_make_hook(name)))
            except Exception:
                continue

        model_was_training = model.training
        model.eval()
        device = next(model.parameters()).device
        batches_used = 0
        token_weight_total = 0

        try:
            with torch.no_grad():
                for batch in batches:
                    inputs, attention_mask = self._prepare_activation_inputs(
                        batch, device
                    )
                    if inputs is None:
                        continue
                    batch_weight = self._batch_token_weight(inputs, attention_mask)
                    batch_weight_holder["weight"] = batch_weight
                    try:
                        if attention_mask is not None:
                            model(inputs, attention_mask=attention_mask)
                        else:
                            model(inputs)
                        batches_used += 1
                        token_weight_total += batch_weight
                    except TypeError:
                        try:
                            model(inputs)
                            batches_used += 1
                            token_weight_total += batch_weight
                        except Exception:
                            continue
                    except Exception:
                        continue
        finally:
            for handle in handles:
                try:
                    handle.remove()
                except Exception:
                    pass
            if model_was_training:
                model.train()

        if batches_used == 0:
            return None

        per_layer = [per_layer_map[name] for name, _module in modules]
        flagged_layers = [
            info["layer"] for info in per_layer if info.get("has_outlier")
        ]
        outlier_total = sum(
            int(info.get("outlier_count", 0) or 0) for info in per_layer
        )
        max_ratio = max(
            (float(info.get("worst_ratio", 0.0)) for info in per_layer), default=0.0
        )

        return {
            "has_outliers": bool(flagged_layers),
            "n_layers_flagged": len(flagged_layers),
            "outlier_count": outlier_total,
            "max_ratio": max_ratio,
            "threshold": (1.0 + self.deadband) * self.margin,
            "per_layer": per_layer,
            "flagged_layers": flagged_layers,
            "analysis_source": "activations",
            "token_weight_total": int(token_weight_total),
            "token_weighted": True,
        }

    def _apply_rmt_detection_and_correction(self, model: nn.Module) -> dict[str, Any]:
        """
        Apply Step 5 RMT detection and correction with adapter support.

        Uses exact Step 5 detection rule: ratio = σ_max_post / bulk_edge_base
        Flag if ratio > (1+deadband)*margin
        """
        per_layer = []
        flagged_layers = []
        corrected_layers = 0

        # Get linear modules in scope
        modules_to_analyze = self._get_linear_modules(model)

        self._log_event(
            "rmt_correction",
            message=f"Applying Step 5 detection and correction to {len(modules_to_analyze)} modules",
        )

        for idx, (module_name, module) in enumerate(modules_to_analyze):
            # Get current stats
            stats = layer_svd_stats(
                module, self.baseline_sigmas, self.baseline_mp_stats, module_name
            )

            # Step 5 detection rule
            has_outlier = False
            skip_reason = None

            if self.baseline_mp_stats and module_name in self.baseline_mp_stats:
                sigma_post = stats["sigma_max"]
                mp_stats = self.baseline_mp_stats[module_name]
                sigma_base = mp_stats.get("sigma_base", 1.0)

                # CORRECTED Step 5 detection rule: baseline-aware growth ratio
                # Compare current σ_max to baseline σ_max, normalized for stability
                ratio = sigma_post / max(sigma_base, 1e-12)
                detection_threshold = (1.0 + self.deadband) * self.margin

                if ratio > detection_threshold:
                    has_outlier = True

                    # Apply correction using enhanced logic with adapter support
                    if self.correct:
                        try:
                            _apply_rmt_correction(
                                module,
                                0.95,  # Conservative factor (not used in Step 5 logic)
                                self.baseline_sigmas,
                                self.baseline_mp_stats,
                                module_name,
                                self.deadband,
                                verbose=False,
                                adapter=self.adapter,
                            )
                            corrected_layers += 1

                            self._log_event(
                                "rmt_correct",
                                message=f"Applied correction to {module_name}",
                                module_name=module_name,
                                pre_ratio=ratio,
                                threshold=detection_threshold,
                            )

                            # Re-compute stats after correction
                            stats_post = layer_svd_stats(
                                module,
                                self.baseline_sigmas,
                                self.baseline_mp_stats,
                                module_name,
                            )
                            mp_stats = self.baseline_mp_stats[module_name]
                            bulk_edge_base = mp_stats.get("mp_bulk_edge_base", 1.0)
                            ratio_post = stats_post["sigma_max"] / max(
                                bulk_edge_base, 1e-12
                            )

                            # Update has_outlier based on post-correction ratio
                            has_outlier = ratio_post > detection_threshold

                        except Exception as e:
                            self._log_event(
                                "rmt_correct_failed",
                                level="ERROR",
                                message=f"Correction failed for {module_name}: {str(e)}",
                                module_name=module_name,
                                error=str(e),
                            )
                else:
                    skip_reason = (
                        f"≤ threshold (ratio={ratio:.2f} ≤ {detection_threshold:.2f})"
                    )
            else:
                # Fallback when no baseline MP stats
                ratio = stats["worst_ratio"]
                if ratio > self.margin:
                    has_outlier = True
                else:
                    skip_reason = f"≤ margin (ratio={ratio:.2f} ≤ {self.margin:.2f})"

            layer_info = {
                "layer": idx,
                "module_name": module_name,
                "sigma_min": stats["sigma_min"],
                "sigma_max": stats["sigma_max"],
                "worst_ratio": stats["worst_ratio"],
                "has_outlier": has_outlier,
                "skip_reason": skip_reason,
            }

            if "worst_details" in stats:
                layer_info["details"] = stats["worst_details"]

            per_layer.append(layer_info)

            if has_outlier:
                flagged_layers.append(idx)

        # Aggregate results
        n_outliers = len(flagged_layers)
        max_ratio = max((float(item["worst_ratio"]) for item in per_layer), default=0.0)
        has_outliers = n_outliers > 0

        return {
            "has_outliers": has_outliers,
            "n_layers_flagged": n_outliers,
            "outlier_count": n_outliers,
            "max_ratio": max_ratio,
            "threshold": self.margin,
            "correction_iterations": 1 if corrected_layers > 0 else 0,
            "corrected_layers": corrected_layers,
            "per_layer": per_layer,
            "flagged_layers": flagged_layers,
            "layers": {f"layer_{item['layer']}": item for item in per_layer},
        }

    def prepare(
        self,
        model: nn.Module,
        adapter=None,
        calib=None,
        policy: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Prepare RMT guard by capturing baseline MP statistics.

        Args:
            model: The model that will be edited
            adapter: ModelAdapter (optional, for tying map access)
            calib: Calibration data for activation-based outlier counting
            policy: Guard policy parameters (optional)

        Returns:
            Dictionary with preparation results and baseline metrics
        """
        import time

        start_time = time.time()
        self._activation_required_failed = False
        self._activation_required_reason = None

        # Store adapter for tying map access during correction
        self.adapter = adapter

        # Update parameters from policy if provided
        if policy:
            self.q = policy.get("q", self.q)
            self.deadband = policy.get("deadband", self.deadband)
            self.margin = policy.get("margin", self.margin)
            self.correct = policy.get("correct", self.correct)
            if "epsilon" in policy:
                self._set_epsilon(policy["epsilon"])
            if "epsilon_by_family" in policy:
                self._set_epsilon(policy["epsilon_by_family"])
            if "activation_required" in policy:
                self._require_activation = bool(policy.get("activation_required"))

        self._log_event(
            "prepare",
            message=f"Preparing RMT guard with q={self.q}, deadband={self.deadband}, margin={self.margin}, correct={self.correct}",
        )

        try:
            # Capture baseline MP statistics for linear layers
            self.baseline_mp_stats = capture_baseline_mp_stats(model)

            # Extract baseline sigmas for compatibility with existing detection
            self.baseline_sigmas = {}
            for name, stats in self.baseline_mp_stats.items():
                self.baseline_sigmas[name] = stats.get("sigma_base", 0.0)

            # Get linear modules in scope
            linear_modules = self._get_linear_modules(model)
            self._activation_ready = False
            self._calibration_batches = []
            max_windows = 0
            if calib is not None:
                source = getattr(calib, "dataloader", None) or calib
                if hasattr(source, "__len__"):
                    try:
                        max_windows = int(len(source))
                    except Exception:
                        max_windows = 0
                self._calibration_batches = self._collect_calibration_batches(
                    calib, max_windows
                )

            activation_baseline = None
            if self._calibration_batches:
                activation_baseline = self._compute_activation_outliers(
                    model, self._calibration_batches
                )

            if activation_baseline:
                self._activation_ready = True
                self.baseline_total_outliers = int(
                    activation_baseline.get("outlier_count", 0) or 0
                )
                self.baseline_outliers_per_family = self._count_outliers_per_family(
                    activation_baseline.get("per_layer", [])
                )
            else:
                if self._require_activation:
                    self._activation_required_failed = True
                    self._activation_required_reason = "activation_baseline_unavailable"
                    self._log_event(
                        "activation_required_missing",
                        level="ERROR",
                        message="Activation baseline unavailable for RMT",
                        profile=self._run_profile,
                        tier=self._run_tier,
                    )
                baseline_detection = rmt_detect(
                    model=model,
                    threshold=self.margin,
                    detect_only=True,
                    baseline_sigmas=self.baseline_sigmas,
                    baseline_mp_stats=self.baseline_mp_stats,
                    deadband=self.deadband,
                )
                self.baseline_total_outliers = baseline_detection.get(
                    "n_layers_flagged", 0
                )
                self.baseline_outliers_per_family = self._count_outliers_per_family(
                    baseline_detection.get("per_layer", [])
                )
            for family_key in ("attn", "ffn", "embed", "other"):
                self.baseline_outliers_per_family.setdefault(family_key, 0)
            self.outliers_per_family = {}
            self.outliers_total = 0
            self.epsilon_violations = []

            self.prepared = True
            preparation_time = time.time() - start_time

            self._log_event(
                "prepare_success",
                message=f"Captured {len(self.baseline_mp_stats)} baseline MP statistics",
                baseline_count=len(self.baseline_mp_stats),
                linear_modules_count=len(linear_modules),
                preparation_time=preparation_time,
            )

            return {
                "baseline_metrics": {
                    "mp_stats_sample": dict(list(self.baseline_mp_stats.items())[:3]),
                    "total_layers": len(self.baseline_mp_stats),
                    "linear_modules_in_scope": len(linear_modules),
                    "scope_suffixes": self.allowed_suffixes,
                    "average_baseline_sigma": np.mean(
                        list(self.baseline_sigmas.values())
                    ),
                    "max_baseline_sigma": max(self.baseline_sigmas.values())
                    if self.baseline_sigmas
                    else 0.0,
                    "min_baseline_sigma": min(self.baseline_sigmas.values())
                    if self.baseline_sigmas
                    else 0.0,
                },
                "policy_applied": {
                    "q": self.q,
                    "deadband": self.deadband,
                    "margin": self.margin,
                    "correct": self.correct,
                },
                "preparation_time": preparation_time,
                "ready": True,
            }

        except Exception as e:
            self.prepared = False
            self._log_event(
                "prepare_failed",
                level="ERROR",
                message=f"Failed to prepare RMT guard: {str(e)}",
                error=str(e),
            )

            return {
                "baseline_metrics": {},
                "policy_applied": policy or {},
                "preparation_time": time.time() - start_time,
                "ready": False,
                "error": str(e),
            }

    def before_edit(self, model: nn.Module) -> None:
        """
        Execute before edit (no action needed for RMT).

        Args:
            model: The model about to be edited
        """
        if self.prepared:
            self._log_event(
                "before_edit",
                message="RMT guard ready for post-edit detection and correction",
            )

    def after_edit(self, model: nn.Module) -> None:
        """
        Execute after edit - perform RMT detection and optional correction.

        Args:
            model: The model that was just edited
        """
        if not self.prepared:
            self._log_event(
                "after_edit_skipped",
                level="WARN",
                message="RMT guard not prepared, skipping post-edit detection",
            )
            return

        self._log_event("after_edit", message="Applying RMT detection and correction")

        try:
            detection_result: dict[str, Any] | None = None
            corrected_layers = 0
            correction_iterations = 0
            use_activation = bool(self._activation_ready and self._calibration_batches)

            if self._require_activation and not use_activation:
                self._activation_required_failed = True
                self._activation_required_reason = "activation_unavailable"
                self._last_result = {
                    "has_outliers": False,
                    "n_layers_flagged": 0,
                    "per_layer": [],
                    "max_ratio": 0.0,
                    "analysis_source": "activations",
                }
                self._log_event(
                    "activation_required_missing",
                    level="ERROR",
                    message="Activation outlier analysis required but unavailable",
                    profile=self._run_profile,
                    tier=self._run_tier,
                )
                return

            if use_activation:
                if self.correct:
                    correction_result = self._apply_rmt_detection_and_correction(model)
                    correction_iterations = int(
                        correction_result.get("correction_iterations", 0) or 0
                    )
                    corrected_layers = int(
                        correction_result.get("corrected_layers", 0) or 0
                    )
                detection_result = self._compute_activation_outliers(
                    model, self._calibration_batches
                )
                if not detection_result:
                    if self._require_activation:
                        self._activation_required_failed = True
                        self._activation_required_reason = (
                            "activation_outliers_unavailable"
                        )
                        self._last_result = {
                            "has_outliers": False,
                            "n_layers_flagged": 0,
                            "per_layer": [],
                            "max_ratio": 0.0,
                            "analysis_source": "activations",
                        }
                        self._log_event(
                            "activation_required_missing",
                            level="ERROR",
                            message="Activation outlier analysis failed",
                            profile=self._run_profile,
                            tier=self._run_tier,
                        )
                        return
                    use_activation = False

            if not use_activation:
                if self.correct:
                    # Apply correction using enhanced logic with adapter support
                    detection_result = self._apply_rmt_detection_and_correction(model)
                else:
                    # Detection only
                    detection_result = rmt_detect(
                        model=model,
                        threshold=self.margin,  # Use margin as threshold
                        detect_only=True,
                        verbose=False,
                        baseline_sigmas=self.baseline_sigmas,
                        baseline_mp_stats=self.baseline_mp_stats,
                        deadband=self.deadband,
                    )

            if detection_result is None:
                raise RuntimeError("RMT detection failed to produce results")

            if use_activation:
                detection_result["correction_iterations"] = correction_iterations
                detection_result["corrected_layers"] = corrected_layers

            # Store results
            self._last_result = detection_result
            self.outliers_per_family = self._count_outliers_per_family(
                detection_result.get("per_layer", [])
            )
            for family_key in ("attn", "ffn", "embed", "other"):
                self.outliers_per_family.setdefault(family_key, 0)
            outlier_total = detection_result.get("outlier_count")
            if outlier_total is None:
                outlier_total = detection_result.get(
                    "n_layers_flagged", len(self.outliers_per_family)
                )
            self.outliers_total = int(outlier_total or 0)
            self.epsilon_violations = self._compute_epsilon_violations()

            flagged_layers = detection_result.get("n_layers_flagged", 0)
            corrected_layers = detection_result.get("correction_iterations", 0)

            self._log_event(
                "rmt_detection_complete",
                message=f"Detected {flagged_layers} outlier layers, correction enabled: {self.correct}",
                layers_flagged=flagged_layers,
                correction_iterations=corrected_layers,
                has_outliers=detection_result.get("has_outliers", False),
                max_ratio=detection_result.get("max_ratio", 0.0),
            )

            # Log individual layer results
            for layer_info in detection_result.get("per_layer", []):
                if layer_info.get("has_outlier", False):
                    self._log_event(
                        "outlier_detected",
                        message=f"Outlier detected in {layer_info.get('module_name', 'unknown')}",
                        layer_name=layer_info.get("module_name"),
                        ratio=layer_info.get("worst_ratio", 0.0),
                        sigma_max=layer_info.get("sigma_max", 0.0),
                        corrected=self.correct,
                    )
                elif layer_info.get("skip_reason"):
                    self._log_event(
                        "layer_skipped",
                        message=f"Layer {layer_info.get('module_name', 'unknown')} skipped: {layer_info.get('skip_reason')}",
                        layer_name=layer_info.get("module_name"),
                        skip_reason=layer_info.get("skip_reason"),
                    )

        except Exception as e:
            self._log_event(
                "after_edit_failed",
                level="ERROR",
                message=f"RMT detection failed: {str(e)}",
                error=str(e),
            )
            # Store empty result for finalize
            self._last_result = {
                "has_outliers": False,
                "n_layers_flagged": 0,
                "per_layer": [],
                "max_ratio": 0.0,
            }
            self.outliers_per_family = {}
            self.outliers_total = 0
            self.epsilon_violations = []

    def validate(
        self, model: Any, adapter: Any, context: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Validate model state (Guard ABC interface).

        Args:
            model: Model to validate
            adapter: ModelAdapter instance
            context: Validation context

        Returns:
            Dictionary with validation results
        """
        # Use finalize to get comprehensive results
        result = self.finalize(model, adapter)

        # Convert to simple dict format if GuardOutcome
        if (
            hasattr(result, "passed")
            and hasattr(result, "action")
            and hasattr(result, "metrics")
        ):
            violations_list: list[str] = []
            if hasattr(result, "violations") and result.violations:
                violations_list = [str(v) for v in result.violations]
            return {
                "passed": bool(result.passed),
                "action": str(result.action),
                "metrics": dict(result.metrics),
                "violations": violations_list,
                "message": "RMT guard validation completed",
            }
        else:
            return {
                "passed": result.get("passed", False),
                "action": "continue" if result.get("passed", False) else "warn",
                "metrics": result.get("metrics", {}),
                "violations": result.get("errors", []),
                "message": "RMT guard validation completed",
            }

    def finalize(self, model: nn.Module, adapter=None) -> GuardOutcome | dict[str, Any]:
        """
        Finalize RMT guard and return comprehensive results.

        Args:
            model: The final edited model
            adapter: Optional adapter for tying map access

        Returns:
            GuardOutcome or dict with RMT detection and correction results
        """
        import time

        start_time = time.time()

        if not self.prepared:
            self._log_event(
                "finalize_failed",
                level="ERROR",
                message="RMT guard not properly prepared",
            )

            if HAS_GUARD_OUTCOME:
                return GuardOutcome(
                    name=self.name,
                    passed=False,
                    action="abort",
                    violations=[
                        {
                            "type": "preparation",
                            "severity": "error",
                            "message": "RMT guard not properly prepared",
                            "module_name": None,
                        }
                    ],
                    metrics={
                        "prepared": False,
                        "finalize_time": time.time() - start_time,
                    },
                )
            else:
                return {
                    "passed": False,
                    "metrics": {
                        "prepared": False,
                        "finalize_time": time.time() - start_time,
                    },
                    "warnings": ["RMT guard not properly prepared"],
                    "errors": ["Preparation failed or baseline MP stats not captured"],
                    "events": self.events,
                }

        # Get results from after_edit
        if self._require_activation and (
            self._activation_required_failed or not self._activation_ready
        ):
            reason = self._activation_required_reason or "activation_required"
            message = "Activation outlier analysis required but unavailable"
            finalize_time = time.time() - start_time
            if HAS_GUARD_OUTCOME:
                return GuardOutcome(
                    name=self.name,
                    passed=False,
                    action="rollback",
                    violations=[
                        {
                            "type": "activation_required",
                            "severity": "error",
                            "message": message,
                            "module_name": None,
                            "reason": reason,
                        }
                    ],
                    metrics={
                        "prepared": True,
                        "activation_required": True,
                        "activation_ready": False,
                        "activation_reason": reason,
                        "finalize_time": finalize_time,
                    },
                )
            return {
                "passed": False,
                "metrics": {
                    "prepared": True,
                    "activation_required": True,
                    "activation_ready": False,
                    "activation_reason": reason,
                    "finalize_time": finalize_time,
                },
                "warnings": [],
                "errors": [message],
                "violations": [],
                "events": self.events,
            }

        result = self._last_result or {
            "has_outliers": False,
            "n_layers_flagged": 0,
            "per_layer": [],
            "max_ratio": 0.0,
        }

        if result and not self.outliers_per_family:
            self.outliers_per_family = self._count_outliers_per_family(
                result.get("per_layer", [])
            )
        for family_key in ("attn", "ffn", "embed", "other"):
            self.outliers_per_family.setdefault(family_key, 0)
            self.baseline_outliers_per_family.setdefault(family_key, 0)
        outlier_total = result.get("outlier_count")
        if outlier_total is None:
            outlier_total = result.get("n_layers_flagged", self.outliers_total or 0)
        self.outliers_total = int(outlier_total or 0)
        self.epsilon_violations = self._compute_epsilon_violations()
        # Contracts: epsilon non-negative, counts non-negative
        for fam, eps in self.epsilon_by_family.items():
            guard_assert(eps >= 0.0, f"rmt.epsilon[{fam}] must be >= 0")
        for fam in set(self.outliers_per_family) | set(
            self.baseline_outliers_per_family
        ):
            guard_assert(
                self.outliers_per_family.get(fam, 0) >= 0,
                "rmt.outliers_per_family negative",
            )
            guard_assert(
                self.baseline_outliers_per_family.get(fam, 0) >= 0,
                "rmt.baseline_outliers negative",
            )

        # Calculate metrics
        flagged_layers = result.get("n_layers_flagged", 0)
        total_layers = (
            len(result.get("per_layer", []))
            if result.get("per_layer")
            else len(self.baseline_mp_stats)
            if self.baseline_mp_stats
            else 0
        )
        flagged_rate = flagged_layers / total_layers if total_layers > 0 else 0.0

        # Acceptance gate: pass when epsilon-rule holds per family.
        passed = not bool(self.epsilon_violations)

        # Generate violations for GuardOutcome
        violations = []
        warnings = []
        errors = []

        # Create violations for each flagged layer
        for layer_info in result.get("per_layer", []):
            if layer_info.get("has_outlier", False):
                violations.append(
                    {
                        "type": "rmt_outlier",
                        "severity": "warning" if self.correct else "error",
                        "message": f"RMT outlier detected: ratio={layer_info.get('worst_ratio', 0.0):.2f}",
                        "module_name": layer_info.get("module_name"),
                        "ratio": layer_info.get("worst_ratio", 0.0),
                        "threshold": (1.0 + self.deadband) * self.margin,
                        "corrected": self.correct,
                    }
                )

        if flagged_rate > 0.3:  # Warning threshold at 30%
            warnings.append(
                f"High RMT outlier rate: {flagged_layers}/{total_layers} layers flagged ({flagged_rate:.1%})"
            )

        if flagged_rate > 0.7:  # Escalate to warning for unusually high rates
            warnings.append(
                f"Excessive RMT outliers: {flagged_layers}/{total_layers} layers flagged"
            )

        if self.epsilon_violations:
            passed = False
            for failure in self.epsilon_violations:
                errors.append(
                    "RMT ε-rule violation: "
                    f"{failure['family']} bare={failure['bare']} "
                    f"guarded={failure['guarded']} allowed={failure['allowed']} "
                    f"(ε={failure['epsilon']:.3f})"
                )

        finalize_time = time.time() - start_time

        # Final metrics
        final_metrics = {
            "layers_flagged": flagged_layers,
            "total_layers": total_layers,
            "flagged_rate": flagged_rate,
            "rmt_outliers": flagged_layers,
            "max_ratio": result.get("max_ratio", 0.0),
            "correction_enabled": self.correct,
            "correction_iterations": result.get("correction_iterations", 0),
            "q_used": self.q,
            "deadband_used": self.deadband,
            "margin_used": self.margin,
            "detection_threshold": (1.0 + self.deadband) * self.margin,
            "baseline_layers_captured": len(self.baseline_mp_stats)
            if self.baseline_mp_stats
            else 0,
            "finalize_time": finalize_time,
            "activation_required": bool(self._require_activation),
            "activation_ready": bool(self._activation_ready),
            "analysis_source": result.get("analysis_source"),
            "token_weight_total": result.get("token_weight_total"),
            "baseline_outliers_per_family": {
                k: int(v) for k, v in self.baseline_outliers_per_family.items()
            },
            "outliers_per_family": {
                k: int(v) for k, v in self.outliers_per_family.items()
            },
            "baseline_outliers_total": int(self.baseline_total_outliers),
            "outliers_total": int(self.outliers_total),
            "epsilon_by_family": {
                k: float(v) for k, v in self.epsilon_by_family.items()
            },
            "epsilon_default": float(self.epsilon_default),
            "epsilon_violations": self.epsilon_violations,
        }

        self._log_event(
            "finalize_complete",
            message=f"RMT guard finalized - {'PASSED' if passed else 'FAILED'}",
            passed=passed,
            flagged_rate=flagged_rate,
            finalize_time=finalize_time,
        )

        # Return GuardOutcome if available, otherwise legacy dict
        # Env-gated tiny evidence dump for auditors
        try:
            payload = {
                "rmt": {
                    "epsilon_by_family": {
                        k: float(v) for k, v in self.epsilon_by_family.items()
                    },
                    "deadband": float(self.deadband),
                    "margin": float(self.margin),
                    "evaluated": True,
                }
            }
            maybe_dump_guard_evidence(".", payload)
        except Exception:
            pass

        if HAS_GUARD_OUTCOME:
            # Add details to metrics since GuardOutcome doesn't have a details field
            final_metrics.update(
                {
                    "guard_type": "rmt",
                    "baseline_captured": self.baseline_mp_stats is not None,
                    "baseline_count": len(self.baseline_mp_stats)
                    if self.baseline_mp_stats
                    else 0,
                    "flagged_layer_names": [v["module_name"] for v in violations],
                    "per_layer_results": result.get("per_layer", []),
                    "policy": {
                        "q": self.q,
                        "deadband": self.deadband,
                        "margin": self.margin,
                        "correct": self.correct,
                        "epsilon": self.epsilon_by_family.copy(),
                    },
                    "scope_suffixes": self.allowed_suffixes,
                }
            )

            return GuardOutcome(
                name=self.name,
                passed=passed,
                action="none" if passed else "rollback",
                violations=violations,
                metrics=final_metrics,
            )
        else:
            return {
                "passed": passed,
                "metrics": final_metrics,
                "warnings": warnings,
                "errors": errors,
                "violations": violations,
                "events": self.events,
                "details": {
                    "guard_type": "rmt",
                    "baseline_captured": self.baseline_mp_stats is not None,
                    "baseline_count": len(self.baseline_mp_stats)
                    if self.baseline_mp_stats
                    else 0,
                    "flagged_layer_names": [v["module_name"] for v in violations],
                    "per_layer_results": result.get("per_layer", []),
                    "policy": {
                        "q": self.q,
                        "deadband": self.deadband,
                        "margin": self.margin,
                        "correct": self.correct,
                        "epsilon": self.epsilon_by_family.copy(),
                    },
                    "scope_suffixes": self.allowed_suffixes,
                },
            }

    def policy(self) -> RMTPolicyDict:
        """
        Get default policy for RMT guard.

        Returns:
            RMTPolicyDict with current configuration
        """
        return RMTPolicyDict(
            q=self.q,
            deadband=self.deadband,
            margin=self.margin,
            correct=self.correct,
            epsilon=self.epsilon_by_family.copy(),
        )


# === Policy Utilities ===


def get_rmt_policy(name: str = "balanced") -> RMTPolicyDict:
    """
    Get a RMT policy by name.

    Args:
        name: Policy name ("conservative", "balanced", "aggressive")

    Returns:
        RMTPolicyDict configuration
    """
    # Per-family ε values match tiers.yaml (November 2025 calibration)
    policies = {
        "conservative": RMTPolicyDict(
            q="auto",
            deadband=0.05,
            margin=1.3,
            correct=True,
            epsilon={"ffn": 0.06, "attn": 0.05, "embed": 0.07, "other": 0.07},
        ),
        "balanced": RMTPolicyDict(
            q="auto",
            deadband=0.10,
            margin=1.5,
            correct=True,
            epsilon={"ffn": 0.10, "attn": 0.08, "embed": 0.12, "other": 0.12},
        ),
        "aggressive": RMTPolicyDict(
            q="auto",
            deadband=0.15,
            margin=1.8,
            correct=True,
            epsilon={"ffn": 0.14, "attn": 0.12, "embed": 0.18, "other": 0.18},
        ),
    }

    if name not in policies:
        from invarlock.core.exceptions import GuardError

        available = list(policies.keys())
        raise GuardError(
            code="E502",
            message="POLICY-NOT-FOUND",
            details={"name": name, "available": available},
        )

    return policies[name]


def create_custom_rmt_policy(
    q: float | Literal["auto"] = "auto",
    deadband: float = 0.10,
    margin: float = 1.5,
    correct: bool = True,
    epsilon: float | dict[str, float] | None = None,
) -> RMTPolicyDict:
    """
    Create a custom RMT policy.

    Args:
        q: MP aspect ratio (auto-derived or manual)
        deadband: Tolerance margin (0.0-0.5)
        margin: RMT threshold ratio (> 1.0)
        correct: Enable automatic correction

    Returns:
        Custom RMTPolicyDict configuration
    """
    if isinstance(q, float) and not 0.1 <= q <= 10.0:
        from invarlock.core.exceptions import ValidationError

        raise ValidationError(
            code="E501",
            message="POLICY-PARAM-INVALID",
            details={"param": "q", "value": q},
        )

    if not 0.0 <= deadband <= 0.5:
        from invarlock.core.exceptions import ValidationError

        raise ValidationError(
            code="E501",
            message="POLICY-PARAM-INVALID",
            details={"param": "deadband", "value": deadband},
        )

    if not margin >= 1.0:
        from invarlock.core.exceptions import ValidationError

        raise ValidationError(
            code="E501",
            message="POLICY-PARAM-INVALID",
            details={"param": "margin", "value": margin},
        )

    return RMTPolicyDict(
        q=q,
        deadband=deadband,
        margin=margin,
        correct=correct,
        epsilon=epsilon,
    )
