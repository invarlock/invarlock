"""
Mutual Information based Neuron Scoring
=======================================

Mutual information analysis for neuron importance scoring.
"""

from collections.abc import Iterable
from typing import Any

import numpy as np
import torch
import torch.nn as nn

_MI_SCORE_ERRORS = (FloatingPointError, RuntimeError, TypeError, ValueError)
_MI_EXTRA_HINT = (
    "Mutual information probes require the optional 'scikit-learn' dependency. "
    "Install it with `pip install 'invarlock[eval]'` or "
    "`pip install 'invarlock[advanced]'`."
)


def mutual_info_regression(*args: Any, **kwargs: Any) -> Any:
    """Load scikit-learn lazily and delegate to mutual_info_regression."""
    try:
        from sklearn.feature_selection import mutual_info_regression as _impl
    except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover
        raise ModuleNotFoundError(_MI_EXTRA_HINT) from exc
    return _impl(*args, **kwargs)


def _call_model(model: nn.Module, /, *args: Any, **kwargs: Any) -> Any:
    return model(*args, **kwargs)


def _tensor_to_cpu_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def compute_neuron_mi_scores(
    model: Any,
    calib_data: Iterable[Any],
    oracle_windows: int = 100,
    device: str | torch.device | None = None,
) -> list[torch.Tensor]:
    """
    Compute neuron importance scores using mutual information.

    Args:
        model: Model to analyze
        calib_data: Calibration dataset
        oracle_windows: Number of calibration windows to use
        device: Device for computation

    Returns:
        List of tensors with MI scores for each layer
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    config = model.config
    n_layers = int(config.n_layer)

    # Hook storage for MLP activations
    mlp_activations: dict[int, torch.Tensor] = {}

    def make_mlp_hook(layer_idx: int) -> Any:
        def hook(module: Any, input: Any, output: Any) -> None:
            # Store MLP intermediate activations
            if isinstance(output, torch.Tensor):
                mlp_activations[layer_idx] = output.detach().cpu()

        return hook

    # Register hooks on MLP layers
    hooks = []
    blocks: list[Any] = (
        list(model.transformer.h) if hasattr(model, "transformer") else list(model.h)
    )
    for layer_idx, block in enumerate(blocks):
        # Hook on c_fc output (after activation)
        hook = block.mlp.c_fc.register_forward_hook(make_mlp_hook(layer_idx))
        hooks.append(hook)

    try:
        # Collect activations and targets
        samples_processed = 0
        all_activations: dict[int, list[torch.Tensor]] = {
            i: [] for i in range(n_layers)
        }
        all_targets: list[Any] = []

        for batch in calib_data:
            if samples_processed >= oracle_windows:
                break

            with torch.no_grad():
                # Extract input_ids from batch
                if isinstance(batch, dict):
                    input_ids = batch.get("input_ids", batch.get("inputs"))
                else:
                    input_ids = batch

                if not isinstance(input_ids, torch.Tensor):
                    continue

                input_ids = input_ids.to(device)

                # Forward pass to collect activations
                _call_model(model, input_ids)

                # Get next token targets (shift by 1)
                if input_ids.size(1) > 1:
                    targets = input_ids[:, 1:]  # [batch, seq-1]

                    # Flatten for MI computation
                    flat_targets = targets.flatten().cpu().numpy()

                    # Store activations for each layer
                    for layer_idx, activations in mlp_activations.items():
                        if layer_idx < n_layers:
                            # activations: [batch, seq, mlp_dim]
                            flat_activations = activations[:, :-1, :].flatten(
                                0, 1
                            )  # [batch*(seq-1), mlp_dim]
                            all_activations[layer_idx].append(flat_activations)

                    all_targets.append(flat_targets)

                mlp_activations.clear()
                samples_processed += 1

        # Compute MI scores for each layer
        mi_scores = []
        if all_targets:
            all_targets_concat = np.concatenate(all_targets)

            for layer_idx in range(n_layers):
                if all_activations[layer_idx]:
                    # Concatenate all activations for this layer
                    layer_activations = torch.cat(
                        all_activations[layer_idx], dim=0
                    )  # [total_samples, mlp_dim]

                    # Compute MI for each neuron
                    mlp_dim = layer_activations.size(1)
                    neuron_mi_scores = torch.zeros(mlp_dim)

                    # Sample subset for efficiency
                    max_samples = 10000
                    if layer_activations.size(0) > max_samples:
                        indices = torch.randperm(layer_activations.size(0))[
                            :max_samples
                        ]
                        layer_activations = layer_activations[indices]
                        targets_subset = all_targets_concat[indices.cpu().numpy()]
                    else:
                        targets_subset = all_targets_concat[: layer_activations.size(0)]

                    # Compute MI for each neuron
                    activations_np = _tensor_to_cpu_numpy(layer_activations)

                    for neuron_idx in range(min(mlp_dim, 100)):  # Limit for efficiency
                        try:
                            neuron_activations = activations_np[:, neuron_idx]
                            mi_score = mutual_info_regression(
                                neuron_activations.reshape(-1, 1),
                                targets_subset[: len(neuron_activations)],
                                random_state=42,
                            )[0]
                            neuron_mi_scores[neuron_idx] = mi_score
                        except _MI_SCORE_ERRORS:
                            neuron_mi_scores[neuron_idx] = 0.0

                    mi_scores.append(neuron_mi_scores)
                else:
                    # No data for this layer
                    mlp_dim = blocks[layer_idx].mlp.c_fc.weight.size(0)
                    mi_scores.append(torch.zeros(mlp_dim))
        else:
            # No data collected
            for layer_idx in range(n_layers):
                mlp_dim = blocks[layer_idx].mlp.c_fc.weight.size(0)
                mi_scores.append(torch.zeros(mlp_dim))

    finally:
        # Remove hooks
        for hook in hooks:
            hook.remove()

    return mi_scores


def mi_neuron_scores(
    activations: torch.Tensor, targets: torch.Tensor, max_samples: int = 10000
) -> torch.Tensor:
    """
    Compute MI scores for a single layer.

    Args:
        activations: Neuron activations [samples, neurons]
        targets: Target values [samples]
        max_samples: Maximum samples to use for efficiency

    Returns:
        MI scores for each neuron
    """
    n_samples = int(activations.shape[0])
    n_neurons = int(activations.shape[1])
    activations = activations.detach()
    targets = targets.detach()

    # Sample subset for efficiency
    if n_samples > max_samples:
        indices = torch.randperm(n_samples, device=activations.device)[:max_samples]
        activations = activations.index_select(0, indices)
        targets = targets.index_select(0, indices)

    # Compute MI for each neuron
    mi_scores = torch.zeros(n_neurons)
    activations_np = _tensor_to_cpu_numpy(activations)
    targets_np = _tensor_to_cpu_numpy(targets)

    for neuron_idx in range(n_neurons):
        try:
            neuron_activations = activations_np[:, neuron_idx]
            mi_score = mutual_info_regression(
                neuron_activations.reshape(-1, 1), targets_np, random_state=42
            )[0]
            mi_scores[neuron_idx] = mi_score
        except _MI_SCORE_ERRORS:
            mi_scores[neuron_idx] = 0.0

    return mi_scores


__all__ = ["compute_neuron_mi_scores", "mi_neuron_scores"]
