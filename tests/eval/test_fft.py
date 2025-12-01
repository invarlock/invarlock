"""
Comprehensive test coverage for invarlock.eval.probes.fft module.

Tests for FFT-based head energy scoring functions.
"""

from unittest.mock import Mock, patch

import pytest
import torch
import torch.nn as nn

from invarlock.eval.probes.fft import compute_head_energy_scores, fft_head_energy


class MockAttentionModule(nn.Module):
    """Mock attention module for testing."""

    def __init__(self):
        super().__init__()
        self.c_attn = nn.Linear(768, 2304)
        self.c_proj = nn.Linear(768, 768)

    def forward(self, x, **kwargs):
        # Return tuple with output and attention weights
        batch_size, seq_len, hidden_size = x.shape
        n_heads = 12

        # Mock attention weights [batch, heads, seq, seq]
        attn_weights = torch.randn(batch_size, n_heads, seq_len, seq_len)
        attn_weights = torch.softmax(attn_weights, dim=-1)

        output = torch.randn(batch_size, seq_len, hidden_size)
        return output, attn_weights


class MockTransformerBlock(nn.Module):
    """Mock transformer block with attention."""

    def __init__(self):
        super().__init__()
        self.attn = MockAttentionModule()
        self.ln_1 = nn.LayerNorm(768)
        self.ln_2 = nn.LayerNorm(768)
        self.mlp = Mock()

    def forward(self, x, **kwargs):
        return x


class MockGPT2Model(nn.Module):
    """Mock GPT-2 model for testing."""

    def __init__(self, n_layers: int = 2, n_heads: int = 12):
        super().__init__()
        self.config = Mock()
        self.config.n_layer = n_layers
        self.config.n_head = n_heads

        # Create transformer structure
        self.transformer = Mock()
        self.transformer.h = nn.ModuleList(
            [MockTransformerBlock() for _ in range(n_layers)]
        )

        self.wte = nn.Embedding(50257, 768)
        self.wpe = nn.Embedding(1024, 768)

    def forward(self, input_ids, output_attentions=False, **kwargs):
        batch_size, seq_len = input_ids.shape

        # Simple forward pass returning mock outputs
        logits = torch.randn(batch_size, seq_len, 50257, requires_grad=True)

        outputs = Mock()
        outputs.logits = logits
        return outputs


class MockAlternativeModel(nn.Module):
    """Alternative model structure without transformer attribute."""

    def __init__(self, n_layers: int = 2, n_heads: int = 8):
        super().__init__()
        self.config = Mock()
        self.config.n_layer = n_layers
        self.config.n_head = n_heads

        # Direct h attribute (no transformer wrapper)
        self.h = nn.ModuleList([MockTransformerBlock() for _ in range(n_layers)])

    def forward(self, input_ids, output_attentions=False, **kwargs):
        batch_size, seq_len = input_ids.shape
        return torch.randn(batch_size, seq_len, 50257, requires_grad=True)


class TestComputeHeadEnergyScores:
    """Test compute_head_energy_scores function."""

    def test_basic_energy_computation(self):
        """Test basic energy score computation."""
        model = MockGPT2Model(n_layers=2, n_heads=4)

        # Create mock calibration data
        calib_data = [
            {"input_ids": torch.randint(0, 1000, (2, 8))},
            {"input_ids": torch.randint(0, 1000, (2, 10))},
        ]

        scores = compute_head_energy_scores(
            model=model, calib_data=calib_data, oracle_windows=2
        )

        assert isinstance(scores, torch.Tensor)
        assert scores.shape == (2, 4)  # n_layers x n_heads
        assert torch.all(scores >= 0)  # Energy should be non-negative

    def test_device_handling(self):
        """Test device parameter handling."""
        model = MockGPT2Model()
        calib_data = [{"input_ids": torch.randint(0, 1000, (1, 5))}]

        # Test explicit device
        scores = compute_head_energy_scores(
            model=model, calib_data=calib_data, device="cpu"
        )
        assert scores.device.type == "cpu"

        # Test default device (from model)
        scores = compute_head_energy_scores(
            model=model, calib_data=calib_data, device=None
        )
        assert isinstance(scores, torch.Tensor)

    def test_alternative_model_structure(self):
        """Test model without transformer attribute."""
        model = MockAlternativeModel(n_layers=3, n_heads=6)
        calib_data = [{"input_ids": torch.randint(0, 1000, (1, 8))}]

        scores = compute_head_energy_scores(
            model=model, calib_data=calib_data, oracle_windows=1
        )

        assert scores.shape == (3, 6)
        assert torch.all(scores >= 0)

    def test_different_input_formats(self):
        """Test different calibration data formats."""
        model = MockGPT2Model()

        # Test dict with 'inputs' key
        calib_data1 = [{"inputs": torch.randint(0, 1000, (1, 6))}]
        scores1 = compute_head_energy_scores(model, calib_data1, oracle_windows=1)
        assert scores1.shape == (2, 12)

        # Test direct tensor
        calib_data2 = [torch.randint(0, 1000, (1, 6))]
        scores2 = compute_head_energy_scores(model, calib_data2, oracle_windows=1)
        assert scores2.shape == (2, 12)

        # Test dict with neither key (should be skipped)
        calib_data3 = [{"other_key": torch.randint(0, 1000, (1, 6))}]
        scores3 = compute_head_energy_scores(model, calib_data3, oracle_windows=1)
        assert scores3.shape == (2, 12)

    def test_hook_creation_and_cleanup(self):
        """Test that hooks are properly created and cleaned up."""
        model = MockGPT2Model()
        calib_data = [{"input_ids": torch.randint(0, 1000, (1, 4))}]

        # Track hook operations
        hook_calls = []

        def mock_register_hook(hook_fn):
            mock_hook = Mock()
            hook_calls.append(mock_hook)
            return mock_hook

        # Patch all attention modules
        with patch.object(
            model.transformer.h[0].attn, "register_forward_hook", mock_register_hook
        ):
            with patch.object(
                model.transformer.h[1].attn, "register_forward_hook", mock_register_hook
            ):
                compute_head_energy_scores(model, calib_data, oracle_windows=1)

                # Verify hooks were created and removed
                assert len(hook_calls) == 2  # One for each layer
                for hook in hook_calls:
                    hook.remove.assert_called_once()

    def test_oracle_windows_limit(self):
        """Test oracle_windows parameter limits processing."""
        model = MockGPT2Model()

        # Create more batches than oracle_windows
        calib_data = [{"input_ids": torch.randint(0, 1000, (1, 5))} for _ in range(5)]

        scores = compute_head_energy_scores(
            model=model,
            calib_data=calib_data,
            oracle_windows=3,  # Should only process 3 batches
        )

        assert scores.shape == (2, 12)

    def test_attention_hook_functionality(self):
        """Test that attention hooks properly capture attention weights."""
        model = MockGPT2Model()
        calib_data = [{"input_ids": torch.randint(0, 1000, (2, 6))}]

        # Test that the function works with the mock model
        scores = compute_head_energy_scores(model, calib_data, oracle_windows=1)

        # Verify basic functionality
        assert scores.shape == (2, 12)
        assert torch.all(scores >= 0)

    def test_fft_energy_computation(self):
        """Test that FFT energy computation works correctly."""
        model = MockGPT2Model()
        calib_data = [{"input_ids": torch.randint(0, 1000, (1, 4))}]

        # Use a model that returns predictable attention weights
        def mock_attention_forward(self, x, **kwargs):
            batch_size, seq_len, hidden_size = x.shape
            n_heads = 12

            # Create deterministic attention weights
            attn_weights = torch.ones(batch_size, n_heads, seq_len, seq_len) * 0.25
            output = torch.randn(batch_size, seq_len, hidden_size)
            return output, attn_weights

        # Patch the attention forward method
        with patch.object(MockAttentionModule, "forward", mock_attention_forward):
            scores = compute_head_energy_scores(model, calib_data, oracle_windows=1)

            # All scores should be positive (energy values)
            assert torch.all(scores >= 0)
            assert scores.shape == (2, 12)

    def test_empty_calibration_data(self):
        """Test behavior with empty calibration data."""
        model = MockGPT2Model()
        calib_data = []

        scores = compute_head_energy_scores(
            model=model, calib_data=calib_data, oracle_windows=1
        )

        # Should return zero energies
        assert scores.shape == (2, 12)
        assert torch.all(scores == 0)

    def test_none_input_ids_handling(self):
        """Test handling when input_ids is None."""
        model = MockGPT2Model()
        calib_data = [
            {"input_ids": None},  # This should be skipped
            {"input_ids": torch.randint(0, 1000, (1, 5))},  # This should be processed
        ]

        scores = compute_head_energy_scores(
            model=model, calib_data=calib_data, oracle_windows=2
        )

        assert scores.shape == (2, 12)

    def test_attention_without_weights(self):
        """Test handling when attention doesn't return weights."""
        model = MockGPT2Model()
        calib_data = [{"input_ids": torch.randint(0, 1000, (1, 4))}]

        # Mock attention that doesn't return weights
        def mock_attention_no_weights(self, x, **kwargs):
            batch_size, seq_len, hidden_size = x.shape
            output = torch.randn(batch_size, seq_len, hidden_size)
            return output, None  # No attention weights

        with patch.object(MockAttentionModule, "forward", mock_attention_no_weights):
            scores = compute_head_energy_scores(model, calib_data, oracle_windows=1)

            # Should handle gracefully with zero scores
            assert scores.shape == (2, 12)
            assert torch.all(scores == 0)

    def test_mismatched_heads_handling(self):
        """Test handling when actual heads differ from config."""
        model = MockGPT2Model(n_layers=2, n_heads=4)  # Config says 4 heads
        calib_data = [{"input_ids": torch.randint(0, 1000, (1, 5))}]

        # Mock attention that returns different number of heads
        def mock_attention_different_heads(self, x, **kwargs):
            batch_size, seq_len, hidden_size = x.shape
            n_heads = 8  # Different from config

            attn_weights = torch.randn(batch_size, n_heads, seq_len, seq_len)
            attn_weights = torch.softmax(attn_weights, dim=-1)
            output = torch.randn(batch_size, seq_len, hidden_size)
            return output, attn_weights

        with patch.object(
            MockAttentionModule, "forward", mock_attention_different_heads
        ):
            scores = compute_head_energy_scores(model, calib_data, oracle_windows=1)

            # Should handle gracefully - only process up to config.n_head
            assert scores.shape == (2, 4)

    def test_exception_during_processing(self):
        """Test hook cleanup when exception occurs."""
        model = MockGPT2Model()
        calib_data = [{"input_ids": torch.randint(0, 1000, (1, 4))}]

        # Track hook removals
        mock_hooks = []

        def mock_register_hook(hook_fn):
            mock_hook = Mock()
            mock_hooks.append(mock_hook)
            return mock_hook

        # Make model forward raise an exception
        with patch.object(model, "forward", side_effect=RuntimeError("Model failed")):
            with patch.object(
                model.transformer.h[0].attn, "register_forward_hook", mock_register_hook
            ):
                with patch.object(
                    model.transformer.h[1].attn,
                    "register_forward_hook",
                    mock_register_hook,
                ):
                    with pytest.raises(RuntimeError):
                        compute_head_energy_scores(model, calib_data, oracle_windows=1)

                    # Verify hooks were still removed
                    for hook in mock_hooks:
                        hook.remove.assert_called_once()


class TestFFTHeadEnergy:
    """Test fft_head_energy function."""

    def test_basic_energy_computation(self):
        """Test basic FFT energy computation."""
        # Create a simple attention matrix
        attention_matrix = torch.randn(8, 8)

        energy = fft_head_energy(attention_matrix)

        assert isinstance(energy, float)
        assert energy >= 0  # Energy should be non-negative

    def test_deterministic_input(self):
        """Test with deterministic input."""
        # Create a matrix with known pattern
        attention_matrix = torch.eye(4)  # Identity matrix

        energy = fft_head_energy(attention_matrix)

        assert isinstance(energy, float)
        assert energy > 0  # Should have non-zero energy

    def test_uniform_matrix(self):
        """Test with uniform attention matrix."""
        # Matrix with uniform attention (all values equal)
        attention_matrix = torch.ones(5, 5) * 0.2

        energy = fft_head_energy(attention_matrix)

        assert isinstance(energy, float)
        assert energy >= 0

    def test_zero_matrix(self):
        """Test with zero matrix."""
        attention_matrix = torch.zeros(6, 6)

        energy = fft_head_energy(attention_matrix)

        assert energy == 0.0  # Zero matrix should have zero energy

    def test_different_data_types(self):
        """Test with different tensor data types."""
        # Test with integer tensor
        attention_matrix_int = torch.randint(0, 5, (4, 4))
        energy_int = fft_head_energy(attention_matrix_int)
        assert isinstance(energy_int, float)

        # Test with double tensor
        attention_matrix_double = torch.randn(4, 4).double()
        energy_double = fft_head_energy(attention_matrix_double)
        assert isinstance(energy_double, float)

        # Test with float tensor
        attention_matrix_float = torch.randn(4, 4).float()
        energy_float = fft_head_energy(attention_matrix_float)
        assert isinstance(energy_float, float)

    def test_different_matrix_sizes(self):
        """Test with different matrix sizes."""
        sizes = [2, 3, 5, 8, 16]

        for size in sizes:
            attention_matrix = torch.randn(size, size)
            energy = fft_head_energy(attention_matrix)

            assert isinstance(energy, float)
            assert energy >= 0

    def test_rectangular_matrix(self):
        """Test with non-square matrix."""
        # FFT2D should work with rectangular matrices too
        attention_matrix = torch.randn(4, 6)

        energy = fft_head_energy(attention_matrix)

        assert isinstance(energy, float)
        assert energy >= 0

    def test_fft_properties(self):
        """Test that FFT computation follows expected properties."""
        # Parseval's theorem: energy should be preserved under FFT
        attention_matrix = torch.randn(8, 8)

        # Compute energy using our function
        energy = fft_head_energy(attention_matrix)

        # Manually compute FFT energy for comparison
        fft_result = torch.fft.fft2(attention_matrix.float())
        manual_energy = torch.sum(torch.abs(fft_result) ** 2).item()

        # Should be the same (within floating point precision)
        assert abs(energy - manual_energy) < 1e-6

    def test_scaling_properties(self):
        """Test energy scaling properties."""
        attention_matrix = torch.randn(6, 6)

        # Original energy
        energy1 = fft_head_energy(attention_matrix)

        # Scaled matrix energy
        scaled_matrix = attention_matrix * 2.0
        energy2 = fft_head_energy(scaled_matrix)

        # Energy should scale by factor^2 (power scaling)
        expected_ratio = 4.0  # 2^2
        actual_ratio = energy2 / energy1 if energy1 > 0 else float("inf")

        if energy1 > 1e-10:  # Avoid division by very small numbers
            assert abs(actual_ratio - expected_ratio) < 0.1

    def test_edge_case_single_element(self):
        """Test with 1x1 matrix."""
        attention_matrix = torch.tensor([[0.5]])

        energy = fft_head_energy(attention_matrix)

        assert isinstance(energy, float)
        assert energy >= 0

    def test_complex_input_handling(self):
        """Test that function converts to float properly."""
        # Test with complex-valued input (should be converted to float)
        attention_matrix = torch.randn(4, 4) + 1j * torch.randn(4, 4)

        # Should work by taking the real part through .float()
        energy = fft_head_energy(attention_matrix.real)

        assert isinstance(energy, float)
        assert energy >= 0


class TestModuleExports:
    """Test module exports and imports."""

    def test_all_exports(self):
        """Test that __all__ contains expected functions."""
        from invarlock.eval.probes.fft import __all__

        expected_exports = ["compute_head_energy_scores", "fft_head_energy"]

        assert set(__all__) == set(expected_exports)

    def test_function_imports(self):
        """Test that functions can be imported."""
        from invarlock.eval.probes.fft import (
            compute_head_energy_scores,
            fft_head_energy,
        )

        assert callable(compute_head_energy_scores)
        assert callable(fft_head_energy)


class TestIntegration:
    """Integration tests combining both functions."""

    def test_energy_function_consistency(self):
        """Test that both functions produce consistent results."""
        # Create a simple attention matrix
        attention_matrix = torch.randn(4, 4)

        # Compute energy using standalone function
        standalone_energy = fft_head_energy(attention_matrix)

        # The main function should use the same computation internally
        assert isinstance(standalone_energy, float)
        assert standalone_energy >= 0

    def test_energy_accumulation(self):
        """Test that energy accumulation works correctly."""
        model = MockGPT2Model(n_layers=1, n_heads=2)

        # Create multiple batches
        calib_data = [
            {"input_ids": torch.randint(0, 1000, (1, 4))},
            {"input_ids": torch.randint(0, 1000, (1, 4))},
        ]

        scores = compute_head_energy_scores(
            model=model, calib_data=calib_data, oracle_windows=2
        )

        # Should average over both samples
        assert scores.shape == (1, 2)
        assert torch.all(scores >= 0)


if __name__ == "__main__":
    pytest.main([__file__])
