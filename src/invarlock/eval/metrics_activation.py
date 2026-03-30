"""Activation collection and lens-metric helpers."""

from __future__ import annotations

import logging
from typing import Any, cast

import torch
import torch.nn as nn

from .metrics_support import (
    DependencyManager,
    InputValidator,
    MetricsConfig,
    MetricsProgressPhase,
    MetricsProgressUpdate,
)

logger = logging.getLogger(__name__)


def _call_model(model: nn.Module, /, *args: Any, **kwargs: Any) -> Any:
    return cast(Any, model)(*args, **kwargs)


validator = InputValidator()


def _emit_progress(
    config: MetricsConfig,
    *,
    phase: MetricsProgressPhase,
    completed: int,
    total: int | None,
) -> None:
    observer = config.progress_observer
    if observer is None:
        return
    observer(
        MetricsProgressUpdate(
            phase=phase,
            completed=int(completed),
            total=None if total is None else int(total),
        )
    )


def _gini_vectorized(vec: torch.Tensor) -> float:
    flat = vec.flatten().abs().float()
    if flat.numel() == 0 or torch.sum(flat) == 0:
        return float("nan")

    sorted_vals = torch.sort(flat)[0]
    n = sorted_vals.numel()
    indices = torch.arange(1, n + 1, dtype=torch.float32, device=flat.device)
    gini = (2 * torch.sum(indices * sorted_vals) / torch.sum(sorted_vals) - (n + 1)) / n
    return gini.item()


def _mi_gini_optimized_cpu_path(
    feats_cpu: torch.Tensor,
    targ_cpu: torch.Tensor,
    max_per_layer: int,
    config: MetricsConfig,
) -> float:
    l_count, sample_count, _ = feats_cpu.shape
    if sample_count > max_per_layer:
        sel = torch.randperm(sample_count)[:max_per_layer]
        feats_cpu = feats_cpu[:, sel, :]
        targ_cpu = targ_cpu[sel]

    dep_manager = DependencyManager()
    if not dep_manager.is_available("mi_scores"):
        return float("nan")

    mi_scores_fn = dep_manager.get_module("mi_scores")
    chunk_size = min(8, l_count)
    mi_scores_all: list[torch.Tensor] = []

    for i in range(0, l_count, chunk_size):
        end_idx = min(i + chunk_size, l_count)
        chunk_feats = feats_cpu[i:end_idx]
        chunk_scores: list[torch.Tensor] = []
        for j in range(chunk_feats.shape[0]):
            try:
                score = mi_scores_fn(chunk_feats[j], targ_cpu)
                chunk_scores.append(score)
            except Exception as e:
                logger.warning(f"MI calculation failed for layer {i + j}: {e}")
                chunk_scores.append(torch.zeros_like(chunk_feats[j, 0, :]))

        mi_scores_all.extend(chunk_scores)
        _emit_progress(
            config,
            phase="mi_gini_cpu",
            completed=end_idx,
            total=l_count,
        )

    if not mi_scores_all:
        return float("nan")

    try:
        mi_mat = torch.stack(mi_scores_all)
        return _gini_vectorized(mi_mat)
    except Exception as e:
        logger.warning(f"Failed to stack MI scores: {e}")
        return float("nan")


def _locate_transformer_blocks_enhanced(model: nn.Module) -> list[nn.Module] | None:
    def safe_getattr_chain(obj, *attrs):
        for attr in attrs:
            if obj is None:
                return None
            obj = getattr(obj, attr, None)
        return obj

    patterns = [
        lambda m: safe_getattr_chain(m, "transformer", "h"),
        lambda m: safe_getattr_chain(m, "h"),
        lambda m: safe_getattr_chain(m, "base_model", "h"),
        lambda m: safe_getattr_chain(m, "model", "h"),
        lambda m: safe_getattr_chain(m, "transformer", "layers"),
    ]

    for pattern in patterns:
        try:
            blocks = pattern(model)
            if blocks is not None and hasattr(blocks, "__len__") and len(blocks) > 0:
                logger.debug(f"Found {len(blocks)} transformer blocks using pattern")
                return list(blocks)
        except (AttributeError, TypeError):
            continue

    transformer_modules = []
    for name, module in model.named_modules():
        if any(attr in name.lower() for attr in ["block", "layer", "transformer"]):
            if hasattr(module, "attn") and hasattr(module, "mlp"):
                transformer_modules.append(module)

    if transformer_modules:
        logger.debug(
            f"Found {len(transformer_modules)} transformer blocks via fallback search"
        )
        return transformer_modules

    logger.warning("Could not locate transformer blocks in model")
    return None


class ResultCache:
    """Simple result caching for expensive lens operations."""

    def __init__(self, config: MetricsConfig):
        self.config = config
        self.cache: dict[str, dict[str, float]] = {}
        self.enabled = config.use_cache

    def _get_cache_key(
        self, model: nn.Module, dataloader, config: MetricsConfig
    ) -> str:
        model_hash = hash(tuple(p.data_ptr() for p in model.parameters()))
        config_hash = hash(
            (config.oracle_windows, config.max_tokens, config.max_samples_per_layer)
        )
        return f"{model_hash}_{config_hash}"

    def get(self, key: str) -> dict[str, float] | None:
        if not self.enabled:
            return None
        return self.cache.get(key)

    def set(self, key: str, result: dict[str, float]) -> None:
        if self.enabled:
            self.cache[key] = result.copy()

    def clear(self) -> None:
        self.cache.clear()


def _perform_pre_eval_checks(
    model: nn.Module, dataloader, device: torch.device, config: MetricsConfig
) -> None:
    try:
        tok_len_attr = getattr(model.config, "n_positions", None) or getattr(
            model.config, "max_position_embeddings", None
        )
        if tok_len_attr:
            sample_batch = next(iter(dataloader))
            sample_ids = sample_batch["input_ids"]
            if sample_ids.shape[1] > tok_len_attr:
                logger.warning(
                    f"Input sequence length {sample_ids.shape[1]} exceeds "
                    f"model limit {tok_len_attr}"
                )
    except Exception as e:
        logger.debug(f"Context length check failed: {e}")

    try:
        dry_batch = next(iter(dataloader))
        model_input = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in dry_batch.items()
        }
        _ = _call_model(model, **model_input)
        logger.debug("Pre-evaluation dry run successful")
    except Exception as e:
        logger.warning(f"Pre-evaluation dry run failed: {e}")


def _collect_activations(
    model: nn.Module, dataloader, config: MetricsConfig, device: torch.device
) -> dict[str, Any]:
    hidden_states_list: list[torch.Tensor] = []
    fc1_activations_list: list[torch.Tensor] = []
    targets_list: list[torch.Tensor] = []
    first_batch = None

    total_batches = (
        min(config.oracle_windows, len(dataloader))
        if hasattr(dataloader, "__len__")
        else config.oracle_windows
    )

    for i, batch in enumerate(dataloader):
        if i >= config.oracle_windows:
            break

        try:
            if first_batch is None:
                first_batch = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }

            input_ids = batch["input_ids"].to(device)
            if input_ids.shape[1] > config.max_tokens:
                input_ids = input_ids[:, : config.max_tokens]

            output = _call_model(model, input_ids, output_hidden_states=True)

            if hasattr(output, "hidden_states") and len(output.hidden_states) > 2:
                hidden_states = torch.stack(output.hidden_states[1:-1])
                hidden_states = validator.validate_tensor(
                    hidden_states, f"hidden_states_batch_{i}", config
                )
                hidden_states_list.append(hidden_states)

            fc1_acts = _extract_fc1_activations(model, output, config)
            if fc1_acts is not None:
                fc1_activations_list.append(fc1_acts)
                targets_list.append(input_ids[:, 1:])
        except Exception as e:
            logger.warning(f"Failed to process batch {i}: {e}")
            continue
        finally:
            _emit_progress(
                config,
                phase="activation_collection",
                completed=min(i + 1, total_batches),
                total=total_batches,
            )

    return {
        "hidden_states": hidden_states_list,
        "fc1_activations": fc1_activations_list,
        "targets": targets_list,
        "first_batch": first_batch,
    }


def _extract_fc1_activations(
    model: nn.Module, output, config: MetricsConfig
) -> torch.Tensor | None:
    blocks = _locate_transformer_blocks_enhanced(model)
    if blocks is None:
        return None

    try:
        valid_activations: list[torch.Tensor] = []
        for idx, block in enumerate(blocks):
            if hasattr(block, "mlp") and hasattr(block.mlp, "c_fc"):
                try:
                    if (
                        hasattr(output, "hidden_states")
                        and len(output.hidden_states) > idx + 1
                    ):
                        hidden_state = output.hidden_states[idx + 1]
                        activation = block.mlp.c_fc(hidden_state)
                        activation = validator.validate_tensor(
                            activation, f"fc1_activation_{idx}", config
                        )
                        valid_activations.append(activation)
                except Exception as e:
                    logger.debug(
                        f"Failed to extract FC1 activation for block {idx}: {e}"
                    )
                    continue

        if valid_activations:
            shapes = [act.shape for act in valid_activations]
            if len(set(shapes)) > 1:
                logger.warning(f"Inconsistent FC1 activation shapes: {set(shapes)}")
                from collections import Counter

                most_common_shape = Counter(shapes).most_common(1)[0][0]
                valid_activations = [
                    act for act in valid_activations if act.shape == most_common_shape
                ]

            return torch.stack(valid_activations)
    except Exception as e:
        logger.warning(f"FC1 activation extraction failed: {e}")

    return None


def _calculate_sigma_max(
    model: nn.Module,
    first_batch: dict | None,
    dep_manager: DependencyManager,
    config: MetricsConfig,
    device: torch.device,
) -> float:
    if not dep_manager.is_available("scan_model_gains"):
        logger.info("Skipping σ_max: scan_model_gains not available")
        return float("nan")

    if first_batch is None:
        logger.info("Skipping σ_max: no data batch available")
        return float("nan")

    try:
        scan_model_gains = dep_manager.get_module("scan_model_gains")
        gains_df = scan_model_gains(model, first_batch)
        if gains_df is None:
            logger.warning("scan_model_gains returned None")
            return float("nan")

        if hasattr(gains_df, "columns") and "name" in gains_df.columns:
            mask = ~gains_df["name"].str.contains(
                "embed|lm_head", case=False, regex=True
            )
            filtered_gains = gains_df[mask]
        else:
            logger.info("Could not filter layers by name for σ_max")
            filtered_gains = gains_df

        if len(filtered_gains) == 0:
            logger.warning("No valid layers found for σ_max computation")
            return float("nan")

        gains_values = getattr(
            filtered_gains, "gain", getattr(filtered_gains, "values", [])
        )
        gains_tensor = torch.as_tensor(gains_values, dtype=torch.float32, device=device)
        if gains_tensor.numel() == 0:
            logger.warning("No gain values found")
            return float("nan")

        gains_tensor = validator.validate_tensor(
            gains_tensor, "sigma_max_gains", config
        )
        finite_mask = torch.isfinite(gains_tensor)
        if not finite_mask.any():
            logger.warning("All σ_max gains are NaN/Inf")
            return float("nan")

        sigma_max = torch.max(gains_tensor[finite_mask]).item()
        logger.debug(f"Calculated σ_max: {sigma_max:.4f}")
        return sigma_max
    except Exception as e:
        logger.warning(f"σ_max calculation failed: {e}")
        return float("nan")


def _calculate_head_energy(
    hidden_states_list: list[torch.Tensor], config: MetricsConfig
) -> float:
    if not hidden_states_list:
        logger.info("Skipping head energy: no hidden states available")
        return float("nan")

    try:
        hidden_stack = torch.cat(hidden_states_list, dim=1)
        hidden_crop = hidden_stack[:, :, : config.max_tokens, :]
        hidden_crop = validator.validate_tensor(
            hidden_crop, "head_energy_hidden_states", config
        )
        squared_activations = hidden_crop.float().pow(2).mean(dim=-1)
        per_layer_energy = squared_activations.mean(dim=(1, 2))
        finite_mask = torch.isfinite(per_layer_energy)
        if not finite_mask.any():
            logger.warning("All head energies are NaN/Inf")
            return float("nan")

        head_energy = per_layer_energy[finite_mask].mean().item()
        logger.debug(f"Calculated head energy: {head_energy:.6f}")
        return head_energy
    except Exception as e:
        logger.warning(f"Head energy calculation failed: {e}")
        return float("nan")


def _calculate_mi_gini(
    model: nn.Module,
    activation_data: dict[str, Any],
    dep_manager: DependencyManager,
    config: MetricsConfig,
    device: torch.device,
) -> float:
    del model
    if not dep_manager.is_available("mi_scores"):
        logger.info("Skipping MI-Gini: mi_scores not available")
        return float("nan")

    if not activation_data["fc1_activations"] or not activation_data["targets"]:
        logger.info("Skipping MI-Gini: no FC1 activations available")
        return float("nan")

    try:
        fc1_all = torch.cat(activation_data["fc1_activations"], dim=1)
        targ_all = torch.cat(activation_data["targets"], dim=0)

        fc1_trim = fc1_all[:, :, :-1, :]
        fc1_trim = fc1_trim[:, :, : config.max_tokens, :]
        targ_trim = targ_all[:, : config.max_tokens]

        l_count, batch_count, token_count, width = fc1_trim.shape
        fc1_flat = fc1_trim.permute(0, 2, 1, 3).reshape(
            l_count, batch_count * token_count, width
        )
        targ_flat = targ_trim.flatten()

        fc1_flat = InputValidator.validate_tensor(fc1_flat, "mi_gini_features", config)
        targ_flat = InputValidator.validate_tensor(targ_flat, "mi_gini_targets", config)

        mi_scores_fn = dep_manager.get_module("mi_scores")

        try:
            logger.debug("Attempting MI-Gini calculation on GPU")
            mi_scores_result = mi_scores_fn(fc1_flat, targ_flat)
            mi_gini = _gini_vectorized(mi_scores_result)
            logger.debug(f"Calculated MI-Gini (GPU): {mi_gini:.6f}")
            return mi_gini
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise

            logger.warning("GPU OOM for MI-Gini, falling back to CPU")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            mi_gini = _mi_gini_optimized_cpu_path(
                fc1_flat.cpu().float(),
                targ_flat.cpu(),
                config.max_samples_per_layer,
                config,
            )
            logger.debug(f"Calculated MI-Gini (CPU): {mi_gini:.6f}")
            return mi_gini
    except Exception as e:
        logger.warning(f"MI-Gini calculation failed: {e}")
        return float("nan")


__all__ = [
    "ResultCache",
    "_calculate_head_energy",
    "_calculate_mi_gini",
    "_calculate_sigma_max",
    "_collect_activations",
    "_extract_fc1_activations",
    "_gini_vectorized",
    "_locate_transformer_blocks_enhanced",
    "_mi_gini_optimized_cpu_path",
    "_perform_pre_eval_checks",
]
