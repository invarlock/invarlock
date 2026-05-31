"""
Eval probe importance scoring.

FFT head energy, mutual-information neuron scoring, post-attention head
scoring, and WANDA-style neuron scoring live together because they share the
same optional heavy runtime surface.
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


def compute_head_energy_scores(
    model: nn.Module,
    calib_data: Any,
    oracle_windows: int = 100,
    device: str | None = None,
) -> torch.Tensor:
    """
    Compute head energy scores using FFT analysis.

    Args:
        model: Model to analyze
        calib_data: Calibration dataset
        oracle_windows: Number of calibration windows to use
        device: Device for computation

    Returns:
        Tensor of shape [n_layers, n_heads] with energy scores
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    config = model.config
    n_layers = config.n_layer
    n_heads = config.n_head

    head_energies = torch.zeros(n_layers, n_heads, device=device)
    attention_outputs = {}

    def make_attention_hook(layer_idx):
        def hook(module, input, output):
            if isinstance(output, tuple) and len(output) >= 2:
                attn_weights = output[1]
                if attn_weights is not None:
                    attention_outputs[layer_idx] = attn_weights.detach()

        return hook

    hooks = []
    blocks = model.transformer.h if hasattr(model, "transformer") else model.h
    for layer_idx, block in enumerate(blocks):
        hook = block.attn.register_forward_hook(make_attention_hook(layer_idx))
        hooks.append(hook)

    try:
        samples_processed = 0
        for batch in calib_data:
            if samples_processed >= oracle_windows:
                break

            with torch.no_grad():
                if isinstance(batch, dict):
                    input_ids = batch.get("input_ids", batch.get("inputs"))
                else:
                    input_ids = batch

                if input_ids is None:
                    continue

                input_ids = input_ids.to(device)
                _ = _call_model(model, input_ids, output_attentions=True)

                for layer_idx, attn_weights in attention_outputs.items():
                    if layer_idx < n_layers:
                        _batch_size, heads, _seq_len, _ = attn_weights.shape

                        for head_idx in range(min(heads, n_heads)):
                            attn_matrix = attn_weights[0, head_idx, :, :]
                            fft_result = torch.fft.fft2(attn_matrix.float())
                            energy = torch.sum(torch.abs(fft_result) ** 2).item()
                            head_energies[layer_idx, head_idx] += energy

                attention_outputs.clear()
                samples_processed += 1

        if samples_processed > 0:
            head_energies /= samples_processed

    finally:
        for hook in hooks:
            hook.remove()

    return head_energies


def fft_head_energy(attention_matrix: torch.Tensor) -> float:
    """
    Compute FFT energy for a single attention matrix.

    Args:
        attention_matrix: Attention weights [seq_len, seq_len]

    Returns:
        Energy score
    """
    attn_float = attention_matrix.float()
    fft_result = torch.fft.fft2(attn_float)
    energy = torch.sum(torch.abs(fft_result) ** 2).item()
    return energy


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


def compute_post_attention_head_scores(
    model: nn.Module,
    calib_data: Any,
    calibration_windows: int = 100,
    global_pruning: bool = True,
    device: str | None = None,
) -> dict[str, torch.Tensor]:
    """
    Compute attention head importance scores based on post-attention analysis.

    Args:
        model: Model to analyze
        calib_data: Calibration dataset
        calibration_windows: Number of calibration windows
        global_pruning: Whether to use global importance ranking
        device: Device for computation

    Returns:
        Dictionary with 'scores' tensor of shape [n_layers, n_heads]
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    config = model.config
    n_layers = config.n_layer
    n_heads = config.n_head

    head_importance = torch.zeros(n_layers, n_heads, device=device)
    attention_outputs = {}
    residual_norms = {}

    def make_attention_hook(layer_idx):
        def hook(module, input, output):
            if isinstance(output, tuple):
                attention_out = output[0]
            else:
                attention_out = output
            attention_outputs[layer_idx] = attention_out.detach()

        return hook

    def make_residual_hook(layer_idx):
        def hook(module, input, output):
            residual_norms[layer_idx] = torch.norm(output.detach(), dim=-1)

        return hook

    hooks = []
    blocks = model.transformer.h if hasattr(model, "transformer") else model.h
    for layer_idx, block in enumerate(blocks):
        attn_hook = block.attn.register_forward_hook(make_attention_hook(layer_idx))
        hooks.append(attn_hook)

        if hasattr(block, "ln_2"):
            res_hook = block.ln_2.register_forward_hook(make_residual_hook(layer_idx))
            hooks.append(res_hook)

    try:
        samples_processed = 0
        for batch in calib_data:
            if samples_processed >= calibration_windows:
                break

            with torch.no_grad():
                if isinstance(batch, dict):
                    input_ids = batch.get("input_ids", batch.get("inputs"))
                else:
                    input_ids = batch

                if input_ids is None:
                    continue

                input_ids = input_ids.to(device)
                _ = _call_model(model, input_ids)

                for layer_idx, attn_output in attention_outputs.items():
                    if layer_idx < n_layers:
                        batch_size, seq_len, hidden_size = attn_output.shape
                        head_dim = hidden_size // n_heads
                        head_outputs = attn_output.view(
                            batch_size, seq_len, n_heads, head_dim
                        )
                        head_norms = torch.norm(head_outputs, dim=(0, 1, 3))
                        head_importance[layer_idx] += head_norms

                attention_outputs.clear()
                residual_norms.clear()
                samples_processed += 1

        if samples_processed > 0:
            head_importance /= samples_processed

    finally:
        for hook in hooks:
            hook.remove()

    return {"scores": head_importance}


def compute_wanda_neuron_scores(
    model: nn.Module,
    calib_data: Any,
    calibration_windows: int = 100,
    device: str | None = None,
) -> dict[str, torch.Tensor]:
    """
    Compute WANDA-style neuron importance scores.

    Args:
        model: Model to analyze
        calib_data: Calibration dataset
        calibration_windows: Number of calibration windows
        device: Device for computation

    Returns:
        Dictionary with 'scores' tensor of shape [n_layers, mlp_dim]
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    config = model.config
    n_layers = config.n_layer

    neuron_importance = []
    blocks = model.transformer.h if hasattr(model, "transformer") else model.h
    active_blocks = tuple(blocks[:n_layers])
    for _layer_idx, block in enumerate(active_blocks):
        mlp_dim = block.mlp.c_fc.weight.shape[0]
        neuron_importance.append(torch.zeros(mlp_dim, device=device))

    activations = {}

    def make_activation_hook(layer_idx):
        def hook(module, input, output):
            activations[layer_idx] = output.detach()

        return hook

    hooks = []
    for layer_idx, block in enumerate(active_blocks):
        hook = block.mlp.c_fc.register_forward_hook(make_activation_hook(layer_idx))
        hooks.append(hook)

    try:
        samples_processed = 0
        for batch in calib_data:
            if samples_processed >= calibration_windows:
                break

            if isinstance(batch, dict):
                input_ids = batch.get("input_ids", batch.get("inputs"))
            else:
                input_ids = batch

            if input_ids is None:
                continue

            input_ids = input_ids.to(device)
            model.zero_grad()
            outputs = _call_model(model, input_ids)

            if hasattr(outputs, "logits"):
                logits = outputs.logits
            else:
                logits = outputs

            if input_ids.size(1) > 1:
                targets = input_ids[:, 1:]
                shift_logits = logits[:, :-1, :].contiguous()
                loss = nn.functional.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    targets.view(-1),
                    reduction="mean",
                )
                loss.backward()

                for layer_idx, _layer_activations in activations.items():
                    mlp_layer = active_blocks[layer_idx].mlp.c_fc
                    if mlp_layer.weight.grad is not None:
                        weight_grad = mlp_layer.weight.grad
                        weight_magnitude = torch.abs(mlp_layer.weight.data)
                        wanda_scores = torch.mean(
                            weight_magnitude * torch.abs(weight_grad), dim=1
                        )
                        neuron_importance[layer_idx] += wanda_scores

            activations.clear()
            samples_processed += 1

        if samples_processed > 0:
            for layer_idx in range(n_layers):
                neuron_importance[layer_idx] /= samples_processed

    finally:
        for hook in hooks:
            hook.remove()

    max_mlp_dim = max(scores.size(0) for scores in neuron_importance)
    padded_scores = torch.zeros(n_layers, max_mlp_dim, device=device)
    for layer_idx, scores in enumerate(neuron_importance):
        padded_scores[layer_idx, : scores.size(0)] = scores

    return {"scores": padded_scores}


def blend_neuron_scores(
    scores_list: list[torch.Tensor], weights: list[float] | None = None
) -> torch.Tensor:
    """
    Blend multiple neuron importance scores.

    Args:
        scores_list: List of score tensors
        weights: Weights for blending (defaults to equal)

    Returns:
        Blended scores
    """
    if not scores_list:
        raise ValueError("Empty scores list")

    if weights is None:
        weights = [1.0 / len(scores_list)] * len(scores_list)

    if len(weights) != len(scores_list):
        raise ValueError("Weights and scores list must have same length")

    target_shape = scores_list[0].shape
    device = scores_list[0].device
    blended = torch.zeros(target_shape, device=device)

    for scores, weight in zip(scores_list, weights, strict=False):
        if scores.shape != target_shape:
            padded_scores = torch.zeros(target_shape, device=device)
            min_shape = tuple(
                min(a, b) for a, b in zip(scores.shape, target_shape, strict=False)
            )

            if len(min_shape) == 2:
                padded_scores[: min_shape[0], : min_shape[1]] = scores[
                    : min_shape[0], : min_shape[1]
                ]
            else:
                padded_scores[: min_shape[0]] = scores[: min_shape[0]]

            scores = padded_scores

        blended += weight * scores

    return blended


__all__ = [
    "blend_neuron_scores",
    "compute_head_energy_scores",
    "compute_neuron_mi_scores",
    "compute_post_attention_head_scores",
    "compute_wanda_neuron_scores",
    "fft_head_energy",
    "mi_neuron_scores",
]
