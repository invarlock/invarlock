"""Tests for compute_neuron_mi_scores."""

from unittest.mock import Mock, patch

import pytest
import torch

from invarlock.eval.probes.mi import compute_neuron_mi_scores
from tests.eval._support_mi import MockAlternativeModel, MockGPT2Model


class TestComputeNeuronMIScores:
    """Test compute_neuron_mi_scores function."""

    def test_basic_mi_computation(self):
        """Test basic MI score computation."""
        model = MockGPT2Model(n_layers=2)
        calib_data = [
            {"input_ids": torch.randint(0, 1000, (2, 10))},
            {"input_ids": torch.randint(0, 1000, (2, 10))},
        ]

        with patch("invarlock.eval.probes.mi.mutual_info_regression") as mock_mi:
            mock_mi.return_value = [0.5]

            scores = compute_neuron_mi_scores(
                model=model, calib_data=calib_data, oracle_windows=2
            )

            assert isinstance(scores, list)
            assert len(scores) == 2
            assert all(isinstance(score, torch.Tensor) for score in scores)

    def test_device_handling(self):
        """Test device parameter handling."""
        model = MockGPT2Model()
        calib_data = [{"input_ids": torch.randint(0, 1000, (1, 5))}]

        with patch("invarlock.eval.probes.mi.mutual_info_regression") as mock_mi:
            mock_mi.return_value = [0.3]

            scores = compute_neuron_mi_scores(
                model=model, calib_data=calib_data, device="cpu"
            )
            assert len(scores) == 2

            scores = compute_neuron_mi_scores(
                model=model, calib_data=calib_data, device=None
            )
            assert len(scores) == 2

    def test_alternative_model_structure(self):
        """Test model without transformer attribute."""
        model = MockAlternativeModel(n_layers=2)
        calib_data = [{"input_ids": torch.randint(0, 1000, (1, 8))}]

        with patch("invarlock.eval.probes.mi.mutual_info_regression") as mock_mi:
            mock_mi.return_value = [0.4]

            scores = compute_neuron_mi_scores(
                model=model, calib_data=calib_data, oracle_windows=1
            )

            assert len(scores) == 2
            assert all(isinstance(score, torch.Tensor) for score in scores)

    def test_different_input_formats(self):
        """Test different calibration data formats."""
        model = MockGPT2Model()

        with patch("invarlock.eval.probes.mi.mutual_info_regression") as mock_mi:
            mock_mi.return_value = [0.2]

            calib_data1 = [{"inputs": torch.randint(0, 1000, (1, 6))}]
            scores1 = compute_neuron_mi_scores(model, calib_data1, oracle_windows=1)
            assert len(scores1) == 2

            calib_data2 = [torch.randint(0, 1000, (1, 6))]
            scores2 = compute_neuron_mi_scores(model, calib_data2, oracle_windows=1)
            assert len(scores2) == 2

            calib_data3 = [{"other_key": torch.randint(0, 1000, (1, 6))}]
            scores3 = compute_neuron_mi_scores(model, calib_data3, oracle_windows=1)
            assert len(scores3) == 2

    def test_model_outputs_without_logits_attribute(self):
        """Test model that returns tensor directly."""
        model = MockAlternativeModel()
        calib_data = [torch.randint(0, 1000, (1, 7))]

        with patch("invarlock.eval.probes.mi.mutual_info_regression") as mock_mi:
            mock_mi.return_value = [0.1]

            scores = compute_neuron_mi_scores(
                model=model, calib_data=calib_data, oracle_windows=1
            )

            assert len(scores) == 2

    def test_sequence_length_handling(self):
        """Test handling of different sequence lengths."""
        model = MockGPT2Model()

        with patch("invarlock.eval.probes.mi.mutual_info_regression") as mock_mi:
            mock_mi.return_value = [0.6]

            calib_data1 = [{"input_ids": torch.randint(0, 1000, (2, 1))}]
            scores1 = compute_neuron_mi_scores(model, calib_data1, oracle_windows=1)
            assert len(scores1) == 2

            calib_data2 = [{"input_ids": torch.randint(0, 1000, (2, 5))}]
            scores2 = compute_neuron_mi_scores(model, calib_data2, oracle_windows=1)
            assert len(scores2) == 2

    def test_hook_cleanup(self):
        """Test that hooks are properly cleaned up."""
        model = MockGPT2Model()
        calib_data = [{"input_ids": torch.randint(0, 1000, (1, 4))}]
        mock_hooks = []
        _ = model.transformer.h[0].mlp.c_fc.register_forward_hook

        def mock_register_hook(hook_fn):
            mock_hook = Mock()
            mock_hooks.append(mock_hook)
            return mock_hook

        with patch.object(
            model.transformer.h[0].mlp.c_fc, "register_forward_hook", mock_register_hook
        ):
            with patch.object(
                model.transformer.h[1].mlp.c_fc,
                "register_forward_hook",
                mock_register_hook,
            ):
                with patch(
                    "invarlock.eval.probes.mi.mutual_info_regression"
                ) as mock_mi:
                    mock_mi.return_value = [0.7]

                    compute_neuron_mi_scores(model, calib_data, oracle_windows=1)

                    for hook in mock_hooks:
                        hook.remove.assert_called_once()

    def test_exception_during_processing(self):
        """Test hook cleanup even when exception occurs."""
        model = MockGPT2Model()
        calib_data = [{"input_ids": torch.randint(0, 1000, (1, 4))}]
        mock_hooks = []

        def mock_register_hook(hook_fn):
            mock_hook = Mock()
            mock_hooks.append(mock_hook)
            return mock_hook

        with patch.object(
            model.transformer.h[0].mlp.c_fc, "register_forward_hook", mock_register_hook
        ):
            with patch.object(
                model.transformer.h[1].mlp.c_fc,
                "register_forward_hook",
                mock_register_hook,
            ):
                with patch.object(
                    model, "forward", side_effect=RuntimeError("Model forward failed")
                ):
                    with pytest.raises(RuntimeError):
                        compute_neuron_mi_scores(model, calib_data, oracle_windows=1)

                    for hook in mock_hooks:
                        hook.remove.assert_called_once()

    def test_oracle_windows_limit(self):
        """Test oracle_windows parameter limits processing."""
        model = MockGPT2Model()
        calib_data = [{"input_ids": torch.randint(0, 1000, (1, 5))} for _ in range(5)]

        with patch("invarlock.eval.probes.mi.mutual_info_regression") as mock_mi:
            mock_mi.return_value = [0.8]

            scores = compute_neuron_mi_scores(
                model=model,
                calib_data=calib_data,
                oracle_windows=2,
            )

            assert len(scores) == 2

    def test_large_sample_handling(self):
        """Test handling of large batches without crashing."""
        model = MockGPT2Model()
        large_batch = {"input_ids": torch.randint(0, 1000, (50, 50))}
        calib_data = [large_batch]

        with patch("invarlock.eval.probes.mi.mutual_info_regression") as mock_mi:
            mock_mi.return_value = [0.9]

            scores = compute_neuron_mi_scores(
                model=model, calib_data=calib_data, oracle_windows=1
            )

            assert len(scores) == 2
            assert all(isinstance(score, torch.Tensor) for score in scores)

    def test_mi_computation_exception_handling(self):
        """Test handling of exceptions during MI computation."""
        model = MockGPT2Model()
        calib_data = [{"input_ids": torch.randint(0, 1000, (2, 6))}]

        with patch("invarlock.eval.probes.mi.mutual_info_regression") as mock_mi:
            mock_mi.side_effect = [0.5, Exception("MI failed"), 0.3]

            scores = compute_neuron_mi_scores(
                model=model, calib_data=calib_data, oracle_windows=1
            )

            assert len(scores) == 2
            assert all(isinstance(score, torch.Tensor) for score in scores)

    def test_no_data_collected(self):
        """Test behavior when no valid data is collected."""
        model = MockGPT2Model()
        calib_data = []

        scores = compute_neuron_mi_scores(
            model=model, calib_data=calib_data, oracle_windows=1
        )

        assert len(scores) == 2
        assert all(isinstance(score, torch.Tensor) for score in scores)
        assert all(torch.all(score == 0.0) for score in scores)

    def test_no_activations_for_layer(self):
        """Test handling when no activations collected for a layer."""
        model = MockGPT2Model()
        calib_data = [{"input_ids": torch.randint(0, 1000, (1, 4))}]

        def failing_hook(module, input, output):
            pass

        with patch("invarlock.eval.probes.mi.mutual_info_regression") as mock_mi:
            mock_mi.return_value = [0.2]

            scores = compute_neuron_mi_scores(
                model=model, calib_data=calib_data, oracle_windows=1
            )

            assert len(scores) == 2
