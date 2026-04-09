"""Tests for mi_neuron_scores and module exports."""

from unittest.mock import patch

import numpy as np
import torch

from invarlock.eval.probes.mi import mi_neuron_scores


class TestMINeuronScores:
    """Test mi_neuron_scores function."""

    def test_basic_mi_computation(self):
        """Test basic MI score computation for single layer."""
        activations = torch.randn(100, 50)
        targets = torch.randint(0, 1000, (100,))

        with patch("invarlock.eval.probes.mi.mutual_info_regression") as mock_mi:
            mock_mi.return_value = [0.4]

            scores = mi_neuron_scores(activations, targets)

            assert isinstance(scores, torch.Tensor)
            assert scores.shape == (50,)
            assert mock_mi.call_count == 50

    def test_subsampling_large_dataset(self):
        """Test subsampling when dataset is too large."""
        activations = torch.randn(15000, 20)
        targets = torch.randint(0, 1000, (15000,))

        with patch("invarlock.eval.probes.mi.mutual_info_regression") as mock_mi:
            with patch("torch.randperm") as mock_randperm:
                mock_randperm.return_value = torch.arange(10000)
                mock_mi.return_value = [0.3]

                scores = mi_neuron_scores(activations, targets, max_samples=10000)

                assert scores.shape == (20,)
                mock_randperm.assert_called_once_with(15000)

    def test_custom_max_samples(self):
        """Test custom max_samples parameter."""
        activations = torch.randn(1000, 10)
        targets = torch.randint(0, 100, (1000,))

        with patch("invarlock.eval.probes.mi.mutual_info_regression") as mock_mi:
            with patch("torch.randperm") as mock_randperm:
                mock_randperm.return_value = torch.arange(500)
                mock_mi.return_value = [0.6]

                scores = mi_neuron_scores(activations, targets, max_samples=500)

                assert scores.shape == (10,)
                mock_randperm.assert_called_once_with(1000)

    def test_no_subsampling_needed(self):
        """Test when no subsampling is needed."""
        activations = torch.randn(50, 15)
        targets = torch.randint(0, 100, (50,))

        with patch("invarlock.eval.probes.mi.mutual_info_regression") as mock_mi:
            with patch("torch.randperm") as mock_randperm:
                mock_mi.return_value = [0.7]

                scores = mi_neuron_scores(activations, targets)

                assert scores.shape == (15,)
                mock_randperm.assert_not_called()

    def test_mi_computation_parameters(self):
        """Test MI computation is called with correct parameters."""
        activations = torch.randn(30, 5)
        targets = torch.randint(0, 50, (30,))

        with patch("invarlock.eval.probes.mi.mutual_info_regression") as mock_mi:
            mock_mi.return_value = [0.5]

            mi_neuron_scores(activations, targets)

            assert mock_mi.call_count == 5
            first_call = mock_mi.call_args_list[0]
            args, kwargs = first_call
            assert args[0].shape == (30, 1)
            assert len(args[1]) == 30
            assert kwargs.get("random_state") == 42

    def test_exception_handling_in_mi_computation(self):
        """Test handling of exceptions during MI computation."""
        activations = torch.randn(20, 8)
        targets = torch.randint(0, 10, (20,))

        with patch("invarlock.eval.probes.mi.mutual_info_regression") as mock_mi:

            def side_effect(*args, **kwargs):
                if len(mock_mi.call_args_list) % 3 == 1:
                    raise ValueError("MI computation failed")
                return [0.4]

            mock_mi.side_effect = side_effect

            scores = mi_neuron_scores(activations, targets)

            assert scores.shape == (8,)
            assert (scores == 0.0).sum() > 0

    def test_tensor_conversion(self):
        """Test proper tensor conversion during computation."""
        activations = torch.randn(40, 12)
        targets = torch.randint(0, 20, (40,))

        with patch("invarlock.eval.probes.mi.mutual_info_regression") as mock_mi:
            mock_mi.return_value = [0.8]

            mi_neuron_scores(activations, targets)

            for call in mock_mi.call_args_list:
                args, kwargs = call
                assert isinstance(args[0], np.ndarray)
                assert isinstance(args[1], np.ndarray)

    def test_edge_case_empty_tensors(self):
        """Test edge case with minimal tensor sizes."""
        activations = torch.randn(10, 1)
        targets = torch.randint(0, 5, (10,))

        with patch("invarlock.eval.probes.mi.mutual_info_regression") as mock_mi:
            mock_mi.return_value = [0.1]

            scores = mi_neuron_scores(activations, targets)

            assert scores.shape == (1,)
            assert mock_mi.call_count == 1

    def test_different_tensor_types(self):
        """Test with different tensor dtypes."""
        activations = torch.randint(0, 100, (25, 6)).float()
        targets = torch.randint(0, 10, (25,))

        with patch("invarlock.eval.probes.mi.mutual_info_regression") as mock_mi:
            mock_mi.return_value = [0.2]

            scores = mi_neuron_scores(activations, targets)

            assert scores.shape == (6,)
            assert scores.dtype == torch.float32


class TestModuleExports:
    """Test module exports and imports."""

    def test_all_exports(self):
        """Test that __all__ contains expected functions."""
        from invarlock.eval.probes.mi import __all__

        expected_exports = ["compute_neuron_mi_scores", "mi_neuron_scores"]
        assert set(__all__) == set(expected_exports)

    def test_function_imports(self):
        """Test that functions can be imported."""
        from invarlock.eval.probes.mi import compute_neuron_mi_scores, mi_neuron_scores

        assert callable(compute_neuron_mi_scores)
        assert callable(mi_neuron_scores)
