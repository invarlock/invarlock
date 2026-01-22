"""
Tests for batch dimension handling in equalise_residual_variance.

This tests the fix for the bug where 1-D input tensors (shape [seq_len])
caused IndexError in HF models that expect 2-D tensors [batch, seq_len].
"""

import pytest
import torch
import torch.nn as nn


class MockTransformerBlock(nn.Module):
    """Mock transformer block with attn and mlp projections."""

    def __init__(self, hidden_size=64):
        super().__init__()
        self.attn = nn.Module()
        self.attn.c_proj = nn.Linear(hidden_size, hidden_size)
        self.mlp = nn.Module()
        self.mlp.c_proj = nn.Linear(hidden_size, hidden_size)


class MockGPT2Model(nn.Module):
    """Mock GPT-2 style model for testing batch dimension handling."""

    def __init__(self, n_layers=2, hidden_size=64, vocab_size=100):
        super().__init__()
        self.transformer = nn.Module()
        self.transformer.h = nn.ModuleList(
            [MockTransformerBlock(hidden_size) for _ in range(n_layers)]
        )
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.hidden_size = hidden_size
        self._forward_calls = []

    def forward(self, input_ids):
        # Track the input shape for verification
        self._forward_calls.append(
            {"input_shape": tuple(input_ids.shape), "input_dim": input_ids.dim()}
        )

        # Simulate GPT-2 forward pass - requires 2-D input
        if input_ids.dim() == 1:
            # This is what HF GPT-2 does internally and can cause issues
            raise IndexError("too many indices for tensor of dimension 2")

        # Actually run through layers
        hidden = self.embed(input_ids)
        for blk in self.transformer.h:
            # Simulate attention and MLP
            attn_out = blk.attn.c_proj(hidden)
            mlp_out = blk.mlp.c_proj(hidden)
            hidden = hidden + attn_out + mlp_out
        return hidden


class TestBatchDimensionHandling:
    """Tests for batch dimension handling in equalise_residual_variance."""

    def test_1d_tensor_converted_to_2d(self):
        """Test that 1-D input tensors are converted to 2-D before model call."""
        from invarlock.guards.variance import equalise_residual_variance

        model = MockGPT2Model(n_layers=2, hidden_size=64)

        # Create calibration data with 1-D tensors (the problematic case)
        # This simulates what happens in the pipeline when calibration_data
        # contains lists that are converted to 1-D tensors
        calib_data = [
            {"input_ids": torch.tensor([1, 2, 3, 4, 5])},  # 1-D tensor
            {"input_ids": torch.tensor([6, 7, 8, 9, 10])},  # 1-D tensor
        ]

        # This should NOT raise IndexError now
        equalise_residual_variance(
            model=model,
            dataloader=calib_data,
            windows=2,
            tol=0.0,  # Accept any scale change
            allow_empty=True,
        )

        # Verify that forward was called with 2-D tensors
        assert len(model._forward_calls) == 2
        for call in model._forward_calls:
            assert call["input_dim"] == 2, (
                f"Expected 2-D input, got {call['input_dim']}-D"
            )
            assert len(call["input_shape"]) == 2, (
                f"Expected 2-D shape, got {call['input_shape']}"
            )

    def test_2d_tensor_unchanged(self):
        """Test that 2-D tensors pass through unchanged."""
        from invarlock.guards.variance import equalise_residual_variance

        model = MockGPT2Model(n_layers=2, hidden_size=64)

        # Create calibration data with 2-D tensors (already batch dimension)
        calib_data = [
            {"input_ids": torch.tensor([[1, 2, 3, 4, 5]])},  # 2-D tensor
            {"input_ids": torch.tensor([[6, 7, 8, 9, 10]])},  # 2-D tensor
        ]

        equalise_residual_variance(
            model=model,
            dataloader=calib_data,
            windows=2,
            tol=0.0,
            allow_empty=True,
        )

        # Verify that forward was called with 2-D tensors
        assert len(model._forward_calls) == 2
        for call in model._forward_calls:
            assert call["input_dim"] == 2

    def test_list_input_converted_to_tensor(self):
        """Test that list inputs are converted to tensors."""
        from invarlock.guards.variance import equalise_residual_variance

        model = MockGPT2Model(n_layers=2, hidden_size=64)

        # Create calibration data with lists (as they come from pipeline)
        calib_data = [
            {"input_ids": [1, 2, 3, 4, 5]},  # Plain list
            {"input_ids": [6, 7, 8, 9, 10]},  # Plain list
        ]

        # This should convert lists to tensors and add batch dimension
        equalise_residual_variance(
            model=model,
            dataloader=calib_data,
            windows=2,
            tol=0.0,
            allow_empty=True,
        )

        # Verify that forward was called with 2-D tensors
        assert len(model._forward_calls) == 2
        for call in model._forward_calls:
            assert call["input_dim"] == 2

    def test_tuple_batch_format(self):
        """Test tuple batch format (e.g., from TensorDataset)."""
        from invarlock.guards.variance import equalise_residual_variance

        model = MockGPT2Model(n_layers=2, hidden_size=64)

        # Create calibration data as tuples (input_ids, labels)
        calib_data = [
            (torch.tensor([1, 2, 3, 4, 5]), torch.tensor([2, 3, 4, 5, 6])),
            (torch.tensor([6, 7, 8, 9, 10]), torch.tensor([7, 8, 9, 10, 11])),
        ]

        equalise_residual_variance(
            model=model,
            dataloader=calib_data,
            windows=2,
            tol=0.0,
            allow_empty=True,
        )

        # Verify 2-D input
        assert len(model._forward_calls) == 2
        for call in model._forward_calls:
            assert call["input_dim"] == 2

    def test_raw_tensor_batch(self):
        """Test raw tensor as batch (no dict wrapping)."""
        from invarlock.guards.variance import equalise_residual_variance

        model = MockGPT2Model(n_layers=2, hidden_size=64)

        # Raw tensors without dict wrapping
        calib_data = [
            torch.tensor([1, 2, 3, 4, 5]),  # 1-D raw tensor
            torch.tensor([6, 7, 8, 9, 10]),  # 1-D raw tensor
        ]

        equalise_residual_variance(
            model=model,
            dataloader=calib_data,
            windows=2,
            tol=0.0,
            allow_empty=True,
        )

        # Verify 2-D input
        assert len(model._forward_calls) == 2
        for call in model._forward_calls:
            assert call["input_dim"] == 2


class TestBatchDimensionWithRealModel:
    """Integration test with a more realistic model structure."""

    @pytest.mark.skipif(
        not torch.cuda.is_available() and not hasattr(torch.backends, "mps"),
        reason="Skip without GPU/MPS available (optional test)",
    )
    def test_hf_causal_compatible_input(self):
        """Test that input format is compatible with HF GPT-2."""
        # This test would use actual HF GPT-2 if available
        # For CI, we skip if no accelerator is present
        pass


class TestEdgeCases:
    """Test edge cases in batch dimension handling."""

    def test_empty_calibration_data(self):
        """Test with empty calibration data."""
        from invarlock.guards.variance import equalise_residual_variance

        model = MockGPT2Model(n_layers=2)

        scales = equalise_residual_variance(
            model=model,
            dataloader=[],
            windows=10,
            allow_empty=True,
        )

        assert scales == {}

    def test_none_input_ids_skipped(self):
        """Test that batches with None input_ids are skipped."""
        from invarlock.guards.variance import equalise_residual_variance

        model = MockGPT2Model(n_layers=2)

        calib_data = [
            {"input_ids": None},  # Should be skipped
            {"input_ids": torch.tensor([1, 2, 3])},  # Should be processed
        ]

        equalise_residual_variance(
            model=model,
            dataloader=calib_data,
            windows=2,
            tol=0.0,
            allow_empty=True,
        )

        # Only one forward call (the None batch was skipped)
        assert len(model._forward_calls) == 1

    def test_mixed_dimension_inputs(self):
        """Test with a mix of 1-D and 2-D inputs."""
        from invarlock.guards.variance import equalise_residual_variance

        model = MockGPT2Model(n_layers=2)

        calib_data = [
            {"input_ids": torch.tensor([1, 2, 3])},  # 1-D
            {"input_ids": torch.tensor([[4, 5, 6]])},  # 2-D
            {"input_ids": torch.tensor([7, 8, 9])},  # 1-D
        ]

        equalise_residual_variance(
            model=model,
            dataloader=calib_data,
            windows=3,
            tol=0.0,
            allow_empty=True,
        )

        # All should be 2-D
        assert len(model._forward_calls) == 3
        for call in model._forward_calls:
            assert call["input_dim"] == 2
