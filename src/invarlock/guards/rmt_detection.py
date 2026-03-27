"""RMT detection and correction helpers built on top of analysis and math owners."""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

from . import rmt_analysis, rmt_math

logger = logging.getLogger(__name__)

__all__ = [
    "rmt_detect",
    "rmt_detect_report",
    "rmt_detect_with_names",
    "_apply_rmt_correction",
]


def _emit_verbose(verbose: bool, message: str) -> None:
    if verbose:
        logger.info(message)


def rmt_detect(
    model: nn.Module,
    threshold: float = 1.5,
    detect_only: bool = True,
    correction_factor: float | None = None,
    layer_indices: list[int] | None = None,
    target_layers: list[str] | None = None,
    allowed_module_names: list[str] | None = None,
    verbose: bool = False,
    max_iterations: int = 2,
    baseline_sigmas: dict[str, float] | None = None,
    baseline_mp_stats: dict[str, dict[str, float]] | None = None,
    deadband: float = 0.0,
    use_quantile_mp: bool = False,
) -> dict[str, Any]:
    """Detect RMT outliers in model weights with optional in-place correction."""
    del use_quantile_mp

    per_layer: list[dict[str, Any]] = []
    flagged_layers: list[int] = []
    modules_to_analyze = []
    allowed_suffixes = [".attn.c_attn", ".attn.c_proj", ".mlp.c_fc", ".mlp.c_proj"]

    if layer_indices is not None or target_layers is not None:
        for idx, layer in enumerate(rmt_analysis._iter_transformer_layers(model)):
            if layer_indices is not None and idx not in layer_indices:
                continue
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
        allowed_set = None
        if isinstance(allowed_module_names, list) and allowed_module_names:
            allowed_set = {str(name).strip() for name in allowed_module_names if name}
        for name, module in model.named_modules():
            if any(name.endswith(suffix) for suffix in allowed_suffixes):
                if allowed_set is not None and name not in allowed_set:
                    continue
                has_2d_weights = any(
                    param.ndim == 2 and "weight" in param_name
                    for param_name, param in module.named_parameters(recurse=False)
                )
                if has_2d_weights:
                    modules_to_analyze.append((name, module))

    prev_outlier_count = float("inf")
    correction_iterations = 0

    while correction_iterations < max_iterations:
        current_outliers = 0
        per_layer = []
        flagged_layers = []

        for idx, (module_name, module) in enumerate(modules_to_analyze):
            stats = rmt_analysis.layer_svd_stats(
                module,
                baseline_sigmas,
                baseline_mp_stats,
                module_name,
            )

            has_outlier = False
            skip_reason = None

            if (
                baseline_sigmas
                and baseline_mp_stats
                and module_name in baseline_sigmas
                and module_name in baseline_mp_stats
            ):
                sigma_post = stats["sigma_max"]
                mp_stats = baseline_mp_stats[module_name]
                bulk_edge_base = mp_stats.get("mp_bulk_edge_base", 1.0)
                ratio = sigma_post / max(bulk_edge_base, 1e-12)
                detection_threshold = (1.0 + deadband) * threshold
                if ratio > detection_threshold:
                    has_outlier = True
                else:
                    skip_reason = (
                        f"≤ threshold (ratio={ratio:.2f} ≤ {detection_threshold:.2f})"
                    )
            elif deadband > 0.0 and baseline_sigmas and module_name in baseline_sigmas:
                baseline_sigma = baseline_sigmas[module_name]
                sigma_post = stats["sigma_max"]
                ratio = sigma_post / max(baseline_sigma, 1e-12)
                detection_threshold = (1.0 + deadband) * threshold
                if ratio > detection_threshold:
                    has_outlier = True
                else:
                    skip_reason = (
                        f"≤ threshold (ratio={ratio:.2f} ≤ {detection_threshold:.2f})"
                    )
            else:
                ratio = stats["worst_ratio"]
                if ratio > threshold:
                    has_outlier = True
                else:
                    skip_reason = f"≤ threshold (ratio={ratio:.2f} ≤ {threshold:.2f})"

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

        if not detect_only and current_outliers > 0 and correction_factor is not None:
            if correction_iterations == 0:
                if verbose:
                    _emit_verbose(
                        verbose,
                        f"    Applying RMT correction (iteration {correction_iterations + 1})...",
                    )
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
                if current_outliers >= prev_outlier_count:
                    if verbose:
                        _emit_verbose(
                            verbose,
                            f"    RMT correction stalled ({current_outliers} outliers unchanged), downgrading to warning",
                        )
                    break
                elif verbose:
                    _emit_verbose(
                        verbose,
                        f"    RMT correction improving ({prev_outlier_count} → {current_outliers} outliers)",
                    )
        else:
            break

        prev_outlier_count = current_outliers
        correction_iterations += 1

    n_outliers = len(flagged_layers)
    max_ratio = max((item["worst_ratio"] for item in per_layer), default=0.0)
    has_outliers = n_outliers > 0

    if verbose and has_outliers:
        baseline_note = (
            " (baseline-aware)" if baseline_sigmas and baseline_mp_stats else " (absolute)"
        )
        deadband_note = f" with {deadband:.0%} deadband" if deadband > 0.0 else ""
        n_detected = n_outliers
        n_will_be_capped = n_outliers if not detect_only else 0
        _emit_verbose(verbose, f"    ⚠️ RMT outliers detected{baseline_note}{deadband_note}:")
        _emit_verbose(
            verbose,
            f"      Detected: {n_detected}, will correct: {n_will_be_capped}",
        )
        _emit_verbose(verbose, f"      Max ratio: {max_ratio:.2f}")
        _emit_verbose(verbose, "      Top offenders (σ_post / σ_ref):")
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
        "layers": {f"layer_{item['layer']}": item for item in per_layer},
    }


def rmt_detect_report(
    model: nn.Module, threshold: float = 1.5
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Generate an RMT health report."""
    result = rmt_detect(model, threshold, verbose=False)
    summary = {
        "has_outliers": result["has_outliers"],
        "n_layers_flagged": result["n_layers_flagged"],
        "max_ratio": result["max_ratio"],
    }
    return summary, result["per_layer"]


def rmt_detect_with_names(
    model: nn.Module,
    threshold: float = 1.5,
    verbose: bool = False,
) -> dict[str, Any]:
    """Detect RMT outliers and return detailed per-layer information with module names."""
    outliers = []
    per_layer = []
    flagged_layers = []
    layer_modules = []

    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        h_layers = model.transformer.h
        if hasattr(h_layers, "__iter__"):
            for idx, layer in enumerate(h_layers):
                layer_modules.append((f"transformer.h.{idx}", layer))
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
        if hasattr(layers, "__iter__"):
            for idx, layer in enumerate(layers):
                layer_modules.append((f"model.layers.{idx}", layer))
    elif hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
        layer_attr = model.encoder.layer
        if hasattr(layer_attr, "__iter__"):
            for idx, layer in enumerate(layer_attr):
                layer_modules.append((f"encoder.layer.{idx}", layer))
    else:
        for name, module in model.named_modules():
            if hasattr(module, "attn") and hasattr(module, "mlp"):
                layer_modules.append((name, module))

    for layer_name, layer in layer_modules:
        stats = rmt_analysis.layer_svd_stats(layer, module_name=layer_name)
        has_outlier = stats["worst_ratio"] > threshold
        if "worst_details" in stats:
            layer_info = {
                "layer_name": layer_name,
                "sigma_min": stats["sigma_min"],
                "sigma_max": stats["sigma_max"],
                "worst_ratio": stats["worst_ratio"],
                "has_outlier": has_outlier,
                "details": stats["worst_details"],
            }
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
        for outlier in outliers[:3]:
            _emit_verbose(
                verbose,
                f"        - {outlier['module_name']}: {outlier['ratio']:.2f} (σ_post={outlier['sigma_max']:.2f}, ref=mp_bulk_edge)",
            )
        if len(outliers) > 3:
            _emit_verbose(
                verbose,
                f"      ... and {len(outliers) - 3} more layers flagged",
            )

    return {
        "has_outliers": has_outliers,
        "n_layers_flagged": n_outliers,
        "max_ratio": max_ratio,
        "threshold": threshold,
        "per_layer": per_layer,
        "flagged_layers": flagged_layers,
        "outliers": outliers,
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
    """Apply in-place RMT correction to a layer's canonical 2D weights."""
    for name, param in layer.named_parameters():
        if param.ndim == 2 and "weight" in name:
            with torch.no_grad():
                try:
                    W = param.detach()
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

                    if baseline_sigmas and baseline_mp_stats and layer_name in baseline_mp_stats:
                        mp_stats = baseline_mp_stats[layer_name]
                        sigma_base = mp_stats.get("sigma_base", 1.0)
                        margin = 1.5
                        target_sigma = sigma_base * margin * (1.0 - deadband)
                    else:
                        m, n = W.shape
                        mp_edge = rmt_math.mp_bulk_edge(m, n, whitened=False)
                        target_sigma = mp_edge * 1.0

                    if sigma_pre > target_sigma:
                        scale = target_sigma / sigma_pre
                        scale = max(scale, 0.1)

                        tied_params = []
                        if adapter and hasattr(adapter, "get_tying_map"):
                            try:
                                tying_map = adapter.get_tying_map()
                                full_param_name = f"{layer_name}.{name}"
                                tied_params = tying_map.get(full_param_name, [])
                            except Exception:
                                tied_params = []

                        param.mul_(scale)
                        if tied_params and adapter:
                            for tied_name in tied_params:
                                try:
                                    tied_param = adapter.get_parameter_by_name(tied_name)
                                    if tied_param is not None:
                                        tied_param.mul_(scale)
                                except Exception:
                                    pass

                        W_after = param.detach()
                        if Conv1D is not None and isinstance(layer, Conv1D):
                            W_after = W_after.T
                        s_vals_after = torch.linalg.svdvals(W_after.float().cpu())
                        sigma_post = s_vals_after[0].item()

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
                    elif verbose:
                        _emit_verbose(
                            verbose,
                            f"      {layer_name}.{name}: SKIP: ≤ target (σ={sigma_pre:.2f} ≤ {target_sigma:.2f})",
                        )
                except (RuntimeError, torch.linalg.LinAlgError):
                    param.mul_(factor)
                    if verbose:
                        _emit_verbose(
                            verbose,
                            f"      {layer_name}.{name}: fallback scaling (SVD failed)",
                        )
