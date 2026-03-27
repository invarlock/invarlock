"""Legacy RMT analysis helpers.

This module contains the weight/SVD/MP analysis utilities that are no longer
part of the public runtime guard contract in `invarlock.guards.rmt`.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
import torch.linalg as tla
import torch.nn as nn

logger = logging.getLogger(__name__)

__all__ = [
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


def _emit_verbose(verbose: bool, message: str) -> None:
    if verbose:
        logger.info(message)


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


def capture_baseline_mp_stats(
    model: nn.Module, *, allowed_module_names: list[str] | None = None
) -> dict[str, dict[str, float]]:
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
    allowed_set = None
    if isinstance(allowed_module_names, list) and allowed_module_names:
        allowed_set = {str(name).strip() for name in allowed_module_names if name}

    for name, module in model.named_modules():
        if isinstance(module, module_types) and hasattr(module, "weight"):
            if allowed_set is not None and name not in allowed_set:
                continue
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
        # RoPE decoder style
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
    allowed_module_names: list[str] | None = None,  # Exact module allowlist
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
        allowed_module_names: Exact module names to analyze (None = derived default scope)
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
        allowed_set = None
        if isinstance(allowed_module_names, list) and allowed_module_names:
            allowed_set = {str(name).strip() for name in allowed_module_names if name}
        for name, module in model.named_modules():
            # Check if this is an allowed module type with 2D weights
            if any(name.endswith(suffix) for suffix in allowed_suffixes):
                if allowed_set is not None and name not in allowed_set:
                    continue
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
                    _emit_verbose(
                        verbose,
                        f"      Module {module_name}: ratio={stats['worst_ratio']:.2f} "
                        f"(σ_max={stats['sigma_max']:.2f}, norm={normalization})",
                    )
            elif verbose and skip_reason:
                _emit_verbose(verbose, f"      Module {module_name}: SKIP: {skip_reason}")

        # Apply correction if requested and not detect-only
        if not detect_only and current_outliers > 0 and correction_factor is not None:
            if correction_iterations == 0:
                if verbose:
                    _emit_verbose(
                        verbose,
                        f"    Applying RMT correction (iteration {correction_iterations + 1})...",
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
                        _emit_verbose(
                            verbose,
                            f"    RMT correction stalled ({current_outliers} outliers unchanged), "
                            f"downgrading to warning",
                        )
                    break
                elif verbose:
                    _emit_verbose(
                        verbose,
                        f"    RMT correction improving ({prev_outlier_count} → {current_outliers} outliers)",
                    )
        else:
            # No correction requested, exit after first iteration
            break

        prev_outlier_count = current_outliers
        correction_iterations += 1

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

        _emit_verbose(verbose, f"    ⚠️ RMT outliers detected{baseline_note}{deadband_note}:")
        _emit_verbose(
            verbose, f"      Detected: {n_detected}, will correct: {n_will_be_capped}"
        )
        _emit_verbose(verbose, f"      Max ratio: {max_ratio:.2f}")
        _emit_verbose(verbose, "      Top offenders (σ_post / σ_ref):")

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
            _emit_verbose(
                verbose,
                f"        - {module_name}: {ratio:.2f} (σ_post={sigma_max:.2f}, ref={ref_type})",
            )

        if len(top_offenders) < n_outliers:
            _emit_verbose(
                verbose,
                f"      ... and {n_outliers - len(top_offenders)} more layers flagged",
            )

    return {
        "has_outliers": has_outliers,
        "n_layers_flagged": n_outliers,
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
        # RoPE decoder style
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
        _emit_verbose(verbose, "    ⚠️ RMT outliers detected:")
        _emit_verbose(verbose, f"      Layers flagged: {n_outliers}")
        _emit_verbose(verbose, f"      Max ratio: {max_ratio:.2f}")
        _emit_verbose(verbose, f"      Threshold: {threshold:.2f}")
        _emit_verbose(verbose, "      Top offenders (σ_post / σ_ref):")
        # Show top offenders with full module names and consistent formatting
        for outlier in outliers[:3]:
            _emit_verbose(
                verbose,
                f"        - {outlier['module_name']}: {outlier['ratio']:.2f} (σ_post={outlier['sigma_max']:.2f}, ref=mp_bulk_edge)",
            )
        if len(outliers) > 3:
            _emit_verbose(
                verbose, f"      ... and {len(outliers) - 3} more layers flagged"
            )

    return {
        "has_outliers": has_outliers,
        "n_layers_flagged": n_outliers,
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
                            _emit_verbose(
                                verbose,
                                f"      {layer_name}.{name}: σ={sigma_pre:.2f}→{sigma_post:.2f} "
                                f"(scale={scale:.3f}, target={target_sigma:.2f}{tied_info})",
                            )
                    else:
                        # No correction needed - log skip reason
                        if verbose:
                            _emit_verbose(
                                verbose,
                                f"      {layer_name}.{name}: SKIP: ≤ target (σ={sigma_pre:.2f} ≤ {target_sigma:.2f})",
                            )

                except (RuntimeError, torch.linalg.LinAlgError):
                    # CRITICAL: Even fallback must use in-place scaling
                    param.mul_(factor)
                    if verbose:
                        _emit_verbose(
                            verbose,
                            f"      {layer_name}.{name}: fallback scaling (SVD failed)",
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
