"""
Post-Attention Analysis and WANDA Scoring
=========================================

Attention-based importance scoring and WANDA neuron analysis.
"""

from typing import Any

import torch
import torch.nn as nn


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

    # Initialize importance accumulator
    head_importance = torch.zeros(n_layers, n_heads, device=device)

    # Hook storage
    attention_outputs = {}
    residual_norms = {}

    def make_attention_hook(layer_idx):
        def hook(module, input, output):
            # Store attention output for analysis
            if isinstance(output, tuple):
                attention_out = output[0]  # Attention output
            else:
                attention_out = output
            attention_outputs[layer_idx] = attention_out.detach()

        return hook

    def make_residual_hook(layer_idx):
        def hook(module, input, output):
            # Store residual connection output
            residual_norms[layer_idx] = torch.norm(output.detach(), dim=-1)

        return hook

    # Register hooks
    hooks = []
    blocks = model.transformer.h if hasattr(model, "transformer") else model.h
    for layer_idx, block in enumerate(blocks):
        # Hook attention output
        attn_hook = block.attn.register_forward_hook(make_attention_hook(layer_idx))
        hooks.append(attn_hook)

        # Hook residual output if available
        if hasattr(block, "ln_2"):  # Post-attention layer norm
            res_hook = block.ln_2.register_forward_hook(make_residual_hook(layer_idx))
            hooks.append(res_hook)

    try:
        # Process calibration data
        samples_processed = 0
        for batch in calib_data:
            if samples_processed >= calibration_windows:
                break

            with torch.no_grad():
                # Extract input_ids from batch
                if isinstance(batch, dict):
                    input_ids = batch.get("input_ids", batch.get("inputs"))
                else:
                    input_ids = batch

                if input_ids is None:
                    continue

                input_ids = input_ids.to(device)

                # Forward pass to collect attention outputs
                _ = model(input_ids)

                # Analyze head contributions
                for layer_idx, attn_output in attention_outputs.items():
                    if layer_idx < n_layers:
                        # Compute importance based on attention output magnitude
                        batch_size, seq_len, hidden_size = attn_output.shape
                        head_dim = hidden_size // n_heads

                        # Reshape to separate heads
                        head_outputs = attn_output.view(
                            batch_size, seq_len, n_heads, head_dim
                        )

                        # Compute importance as norm of each head's contribution
                        head_norms = torch.norm(
                            head_outputs, dim=(0, 1, 3)
                        )  # [n_heads]
                        head_importance[layer_idx] += head_norms

                attention_outputs.clear()
                residual_norms.clear()
                samples_processed += 1

        # Normalize by number of samples
        if samples_processed > 0:
            head_importance /= samples_processed

    finally:
        # Remove hooks
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

    # Hook storage for gradients and activations
    neuron_importance = []

    # Initialize storage for each layer
    blocks = model.transformer.h if hasattr(model, "transformer") else model.h
    for _layer_idx, block in enumerate(blocks):
        mlp_dim = block.mlp.c_fc.weight.shape[0]
        neuron_importance.append(torch.zeros(mlp_dim, device=device))

    # Hook storage
    activations = {}

    def make_activation_hook(layer_idx):
        def hook(module, input, output):
            # Store MLP activations for WANDA computation
            activations[layer_idx] = output.detach()

        return hook

    # Register hooks on MLP layers
    hooks = []
    for layer_idx, block in enumerate(blocks):
        hook = block.mlp.c_fc.register_forward_hook(make_activation_hook(layer_idx))
        hooks.append(hook)

    try:
        # Process calibration data
        samples_processed = 0
        for batch in calib_data:
            if samples_processed >= calibration_windows:
                break

            # Extract input_ids from batch
            if isinstance(batch, dict):
                input_ids = batch.get("input_ids", batch.get("inputs"))
            else:
                input_ids = batch

            if input_ids is None:
                continue

            input_ids = input_ids.to(device)

            # Enable gradients for WANDA computation
            model.zero_grad()

            # Forward pass
            outputs = model(input_ids)

            # Compute loss for gradient computation
            if hasattr(outputs, "logits"):
                logits = outputs.logits
            else:
                logits = outputs

            # Use next token prediction loss
            if input_ids.size(1) > 1:
                targets = input_ids[:, 1:]
                shift_logits = logits[:, :-1, :].contiguous()
                loss = nn.functional.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    targets.view(-1),
                    reduction="mean",
                )

                # Backward pass to get gradients
                loss.backward()

                # Compute WANDA scores (activation * gradient)
                for layer_idx, _layer_activations in activations.items():
                    if layer_idx < n_layers:
                        # Get gradients from the MLP layer
                        mlp_layer = blocks[layer_idx].mlp.c_fc
                        if mlp_layer.weight.grad is not None:
                            # Compute WANDA score
                            weight_grad = (
                                mlp_layer.weight.grad
                            )  # [mlp_dim, hidden_size]
                            weight_magnitude = torch.abs(mlp_layer.weight.data)

                            # WANDA score: weight magnitude * gradient magnitude
                            wanda_scores = torch.mean(
                                weight_magnitude * torch.abs(weight_grad), dim=1
                            )
                            neuron_importance[layer_idx] += wanda_scores

            activations.clear()
            samples_processed += 1

        # Normalize by number of samples
        if samples_processed > 0:
            for layer_idx in range(n_layers):
                neuron_importance[layer_idx] /= samples_processed

    finally:
        # Remove hooks
        for hook in hooks:
            hook.remove()

    # Stack into tensor
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

    # Ensure all tensors have same shape
    target_shape = scores_list[0].shape
    device = scores_list[0].device

    blended = torch.zeros(target_shape, device=device)

    for scores, weight in zip(scores_list, weights, strict=False):
        if scores.shape != target_shape:
            # Pad or truncate to match target shape
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
    "compute_post_attention_head_scores",
    "compute_wanda_neuron_scores",
    "blend_neuron_scores",
]
