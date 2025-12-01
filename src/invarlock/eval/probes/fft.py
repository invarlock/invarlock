"""
FFT-based Head Energy Scoring
=============================

Frequency domain analysis for attention head importance scoring.
"""

from typing import Any

import torch
import torch.nn as nn


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

    # Initialize energy accumulator
    head_energies = torch.zeros(n_layers, n_heads, device=device)

    # Hook storage
    attention_outputs = {}

    def make_attention_hook(layer_idx):
        def hook(module, input, output):
            # Extract attention weights from output
            if isinstance(output, tuple) and len(output) >= 2:
                attn_weights = output[1]  # Attention weights
                if attn_weights is not None:
                    # Store for FFT analysis
                    attention_outputs[layer_idx] = attn_weights.detach()

        return hook

    # Register hooks
    hooks = []
    blocks = model.transformer.h if hasattr(model, "transformer") else model.h
    for layer_idx, block in enumerate(blocks):
        hook = block.attn.register_forward_hook(make_attention_hook(layer_idx))
        hooks.append(hook)

    try:
        # Process calibration data
        samples_processed = 0
        for batch in calib_data:
            if samples_processed >= oracle_windows:
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

                # Forward pass to collect attention
                _ = model(input_ids, output_attentions=True)

                # Compute FFT energy for each head
                for layer_idx, attn_weights in attention_outputs.items():
                    if layer_idx < n_layers:
                        # attn_weights: [batch, heads, seq, seq]
                        batch_size, heads, seq_len, _ = attn_weights.shape

                        for head_idx in range(min(heads, n_heads)):
                            # Get attention matrix for this head
                            attn_matrix = attn_weights[0, head_idx, :, :]  # [seq, seq]

                            # Compute FFT and energy
                            fft_result = torch.fft.fft2(attn_matrix.float())
                            energy = torch.sum(torch.abs(fft_result) ** 2).item()

                            head_energies[layer_idx, head_idx] += energy

                attention_outputs.clear()
                samples_processed += 1

        # Normalize by number of samples
        if samples_processed > 0:
            head_energies /= samples_processed

    finally:
        # Remove hooks
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
    # Ensure float type for FFT
    attn_float = attention_matrix.float()

    # Compute 2D FFT
    fft_result = torch.fft.fft2(attn_float)

    # Compute energy as sum of squared magnitudes
    energy = torch.sum(torch.abs(fft_result) ** 2).item()

    return energy


__all__ = ["compute_head_energy_scores", "fft_head_energy"]
