"""Higher-coverage MI probe execution tests."""

from unittest.mock import Mock, patch

import numpy as np
import torch
import torch.nn as nn

from invarlock.eval.probes.mi import compute_neuron_mi_scores, mi_neuron_scores
from tests.eval._support_mi import MockGPT2Model


class TestMIModuleCoverage:
    """Additional tests to improve coverage to 80%+."""

    def test_hook_activation_storage(self):
        """Test that hook function stores activations properly."""
        model = MockGPT2Model(n_layers=1)
        calib_data = [{"input_ids": torch.randint(0, 1000, (2, 8))}]

        with patch("invarlock.eval.probes.mi.mutual_info_regression") as mock_mi:
            mock_mi.return_value = [0.5]

            scores = compute_neuron_mi_scores(
                model=model, calib_data=calib_data, oracle_windows=1
            )

            assert len(scores) == 1
            assert isinstance(scores[0], torch.Tensor)
            assert scores[0].shape[0] > 0

    def test_full_mi_computation_path(self):
        """Test the full MI computation path including large dataset handling."""
        model = MockGPT2Model(n_layers=1, mlp_dim=50)
        calib_data = []
        for _ in range(3):
            calib_data.append({"input_ids": torch.randint(0, 1000, (4, 12))})

        with patch("invarlock.eval.probes.mi.mutual_info_regression") as mock_mi:
            mock_mi.return_value = [0.6]

            scores = compute_neuron_mi_scores(
                model=model,
                calib_data=calib_data,
                oracle_windows=3,
            )

            assert len(scores) == 1
            assert scores[0].shape == (50,)
            assert isinstance(scores[0], torch.Tensor)

    def test_activations_concatenation_and_sampling(self):
        """Test activation concatenation and subsampling for large datasets."""
        model = MockGPT2Model(n_layers=1, mlp_dim=20)
        calib_data = []
        for _ in range(5):
            calib_data.append({"input_ids": torch.randint(0, 1000, (8, 10))})

        with patch("invarlock.eval.probes.mi.mutual_info_regression") as mock_mi:
            with patch("torch.randperm") as mock_randperm:
                mock_randperm.return_value = torch.arange(100)
                mock_mi.return_value = [0.4]

                scores = compute_neuron_mi_scores(
                    model=model, calib_data=calib_data, oracle_windows=5
                )

                assert len(scores) == 1
                assert scores[0].shape == (20,)

    def test_neuron_limit_efficiency(self):
        """Test the neuron limit for efficiency (max 100 neurons processed)."""
        model = MockGPT2Model(n_layers=1, mlp_dim=150)
        calib_data = [{"input_ids": torch.randint(0, 1000, (2, 6))}]

        with patch("invarlock.eval.probes.mi.mutual_info_regression") as mock_mi:
            mock_mi.return_value = [0.7]

            scores = compute_neuron_mi_scores(
                model=model, calib_data=calib_data, oracle_windows=1
            )

            assert len(scores) == 1
            assert scores[0].shape == (150,)
            assert mock_mi.call_count <= 100

    def test_targets_subset_alignment(self):
        """Test proper alignment of targets subset with activations."""
        model = MockGPT2Model(n_layers=1, mlp_dim=10)
        calib_data = [{"input_ids": torch.randint(0, 1000, (3, 7))}]

        with patch("invarlock.eval.probes.mi.mutual_info_regression") as mock_mi:
            mock_mi.return_value = [0.8]

            scores = compute_neuron_mi_scores(
                model=model, calib_data=calib_data, oracle_windows=1
            )

            assert len(scores) == 1
            assert scores[0].shape == (10,)

            if mock_mi.call_count > 0:
                call_args = mock_mi.call_args_list[0]
                neuron_acts, targets = call_args[0]
                assert len(targets) >= len(neuron_acts)

    def test_targets_and_activations_stay_paired_when_subsampled(self):
        """Test that subsampled activations use the matching target rows."""

        class SubsamplePairingModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.config = Mock()
                self.config.n_layer = 1

                class Block(nn.Module):
                    def __init__(self) -> None:
                        super().__init__()

                        class MLP(nn.Module):
                            def __init__(self) -> None:
                                super().__init__()
                                self.c_fc = nn.Linear(1, 1, bias=False)

                        self.mlp = MLP()

                self.transformer = nn.Module()
                self.transformer.h = nn.ModuleList([Block()])
                self.dummy_param = nn.Parameter(torch.randn(1))

            def forward(self, input_ids):
                batch_size, seq_len = input_ids.shape
                x = torch.randn(batch_size, seq_len, 1)
                self.transformer.h[0].mlp.c_fc(x)
                result = Mock()
                result.logits = torch.randn(batch_size, seq_len, 8)
                return result

        model = SubsamplePairingModel()
        input_ids = torch.arange(10002, dtype=torch.long).unsqueeze(0)

        with patch("invarlock.eval.probes.mi.mutual_info_regression") as mock_mi:
            captured_targets = []

            def capture_targets(features, targets, random_state=42):
                captured_targets.append(np.asarray(targets))
                return np.asarray([0.25])

            mock_mi.side_effect = capture_targets

            with patch("torch.randperm") as mock_randperm:
                mock_randperm.return_value = torch.arange(10000, 0, -1)

                scores = compute_neuron_mi_scores(
                    model=model,
                    calib_data=[{"input_ids": input_ids}],
                    oracle_windows=1,
                )

        assert len(scores) == 1
        assert scores[0].shape == (1,)
        assert len(captured_targets) == 1
        expected_targets = np.arange(1, 10002)[torch.arange(10000, 0, -1).numpy()]
        assert np.array_equal(captured_targets[0], expected_targets)

    def test_exception_handling_in_neuron_mi(self):
        """Test exception handling during individual neuron MI computation."""
        model = MockGPT2Model(n_layers=1, mlp_dim=5)
        calib_data = [{"input_ids": torch.randint(0, 1000, (2, 5))}]

        with patch("invarlock.eval.probes.mi.mutual_info_regression") as mock_mi:

            def failing_mi(*args, **kwargs):
                if mock_mi.call_count % 2 == 0:
                    raise ValueError("MI computation failed")
                return [0.9]

            mock_mi.side_effect = failing_mi

            scores = compute_neuron_mi_scores(
                model=model, calib_data=calib_data, oracle_windows=1
            )

            assert len(scores) == 1
            assert scores[0].shape == (5,)
            assert (scores[0] == 0.0).sum() > 0


class TestMIRealExecutionCoverage:
    """Tests designed to achieve 80%+ coverage by exercising real execution paths."""

    def test_real_hook_execution_line_47(self):
        """Test real hook execution to cover line 47."""

        class RealHookModel(nn.Module):
            """Functional model that executes real hooks."""

            def __init__(self):
                super().__init__()
                self.config = Mock()
                self.config.n_layer = 1

                class RealBlock(nn.Module):
                    def __init__(self):
                        super().__init__()

                        class RealMLP(nn.Module):
                            def __init__(self):
                                super().__init__()
                                self.c_fc = nn.Linear(4, 6)

                        self.mlp = RealMLP()

                self.transformer = nn.Module()
                self.transformer.h = nn.ModuleList([RealBlock()])
                self.dummy_param = nn.Parameter(torch.randn(1))

            def forward(self, input_ids):
                batch_size, seq_len = input_ids.shape
                x = torch.randn(batch_size, seq_len, 4, requires_grad=True)
                self.transformer.h[0].mlp.c_fc(x)
                logits = torch.randn(batch_size, seq_len, 10, requires_grad=True)
                result = Mock()
                result.logits = logits
                return result

        model = RealHookModel()
        calib_data = [{"input_ids": torch.randint(0, 100, (2, 8))}]

        with patch("invarlock.eval.probes.mi.mutual_info_regression") as mock_mi:
            mock_mi.return_value = [0.5]

            scores = compute_neuron_mi_scores(
                model=model, calib_data=calib_data, oracle_windows=1
            )

            assert len(scores) == 1
            assert scores[0].shape == (6,)
            assert torch.any(scores[0] > 0)

    def test_activation_processing_lines_99_to_102(self):
        """Test activation flattening and processing logic."""

        class ProcessingModel(nn.Module):
            """Model designed to trigger activation processing."""

            def __init__(self):
                super().__init__()
                self.config = Mock()
                self.config.n_layer = 2

                class RealBlock(nn.Module):
                    def __init__(self):
                        super().__init__()

                        class RealMLP(nn.Module):
                            def __init__(self):
                                super().__init__()
                                self.c_fc = nn.Linear(3, 5)

                        self.mlp = RealMLP()

                self.transformer = nn.Module()
                self.transformer.h = nn.ModuleList([RealBlock() for _ in range(2)])
                self.dummy_param = nn.Parameter(torch.randn(1))

            def forward(self, input_ids):
                batch_size, seq_len = input_ids.shape
                x = torch.randn(batch_size, seq_len, 3, requires_grad=True)
                for block in self.transformer.h:
                    _ = block.mlp.c_fc(x)
                logits = torch.randn(batch_size, seq_len, 50, requires_grad=True)
                result = Mock()
                result.logits = logits
                return result

        model = ProcessingModel()
        calib_data = [
            {"input_ids": torch.randint(0, 100, (3, 10))},
            {"input_ids": torch.randint(0, 100, (2, 8))},
        ]

        with patch("invarlock.eval.probes.mi.mutual_info_regression") as mock_mi:
            mock_mi.return_value = [0.6]

            scores = compute_neuron_mi_scores(
                model=model,
                calib_data=calib_data,
                oracle_windows=2,
            )

            assert len(scores) == 2
            assert all(score.shape == (5,) for score in scores)
            assert mock_mi.call_count > 0

    def test_mi_computation_loop_lines_117_to_147(self):
        """Test MI computation loop for each neuron."""

        class MIComputationModel(nn.Module):
            """Model that generates data for MI computation."""

            def __init__(self):
                super().__init__()
                self.config = Mock()
                self.config.n_layer = 1

                class RealBlock(nn.Module):
                    def __init__(self):
                        super().__init__()

                        class RealMLP(nn.Module):
                            def __init__(self):
                                super().__init__()
                                self.c_fc = nn.Linear(5, 8)

                        self.mlp = RealMLP()

                self.transformer = nn.Module()
                self.transformer.h = nn.ModuleList([RealBlock()])
                self.dummy_param = nn.Parameter(torch.randn(1))

            def forward(self, input_ids):
                batch_size, seq_len = input_ids.shape
                x = torch.randn(batch_size, seq_len, 5, requires_grad=True)
                self.transformer.h[0].mlp.c_fc(x)
                logits = torch.randn(batch_size, seq_len, 20, requires_grad=True)
                result = Mock()
                result.logits = logits
                return result

        model = MIComputationModel()
        calib_data = []
        for _ in range(3):
            calib_data.append({"input_ids": torch.randint(0, 100, (4, 12))})

        mi_call_count = 0

        def counting_mi_regression(X, y, random_state=None):
            nonlocal mi_call_count
            mi_call_count += 1
            return [0.4 + 0.1 * mi_call_count]

        with patch(
            "invarlock.eval.probes.mi.mutual_info_regression",
            side_effect=counting_mi_regression,
        ):
            scores = compute_neuron_mi_scores(
                model=model, calib_data=calib_data, oracle_windows=3
            )

            assert len(scores) == 1
            assert scores[0].shape == (8,)
            assert mi_call_count > 0
            assert mi_call_count <= 8
            assert torch.any(scores[0] > 0)

    def test_mi_neuron_scores_direct_execution(self):
        """Test mi_neuron_scores function directly for full coverage."""
        n_samples, n_neurons = 150, 12
        activations = torch.randn(n_samples, n_neurons)
        targets = torch.randint(0, 50, (n_samples,))
        mi_calls = []

        def tracking_mi_regression(X, y, random_state=None):
            mi_calls.append((X.shape, len(y)))
            return [0.3 + len(mi_calls) * 0.05]

        with patch(
            "invarlock.eval.probes.mi.mutual_info_regression",
            side_effect=tracking_mi_regression,
        ):
            scores = mi_neuron_scores(activations, targets, max_samples=100)

            assert scores.shape == (n_neurons,)
            assert len(mi_calls) == n_neurons
            for call in mi_calls:
                X_shape, y_len = call
                assert X_shape == (100, 1)
                assert y_len == 100
            assert torch.all(scores > 0)
            assert torch.all(scores[1:] >= scores[:-1])

    def test_mi_computation_with_large_dataset_subsampling(self):
        """Test subsampling logic."""

        class LargeDataModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = Mock()
                self.config.n_layer = 1

                class RealBlock(nn.Module):
                    def __init__(self):
                        super().__init__()

                        class RealMLP(nn.Module):
                            def __init__(self):
                                super().__init__()
                                self.c_fc = nn.Linear(6, 4)

                        self.mlp = RealMLP()

                self.transformer = nn.Module()
                self.transformer.h = nn.ModuleList([RealBlock()])
                self.dummy_param = nn.Parameter(torch.randn(1))

            def forward(self, input_ids):
                batch_size, seq_len = input_ids.shape
                x = torch.randn(batch_size, seq_len, 6, requires_grad=True)
                self.transformer.h[0].mlp.c_fc(x)
                logits = torch.randn(batch_size, seq_len, 30, requires_grad=True)
                result = Mock()
                result.logits = logits
                return result

        model = LargeDataModel()
        calib_data = []
        for _ in range(10):
            calib_data.append({"input_ids": torch.randint(0, 100, (50, 20))})

        randperm_called = []

        def mock_randperm(n):
            randperm_called.append(n)
            return torch.arange(min(n, 10000))

        with patch("torch.randperm", side_effect=mock_randperm):
            with patch("invarlock.eval.probes.mi.mutual_info_regression") as mock_mi:
                mock_mi.return_value = [0.7]

                scores = compute_neuron_mi_scores(
                    model=model, calib_data=calib_data, oracle_windows=10
                )

                assert len(scores) == 1
                assert scores[0].shape == (4,)
                if len(randperm_called) > 0:
                    assert any(n > 10000 for n in randperm_called)

    def test_exception_handling_in_mi_loop(self):
        """Test exception handling in MI loop."""

        class ExceptionTestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = Mock()
                self.config.n_layer = 1

                class RealBlock(nn.Module):
                    def __init__(self):
                        super().__init__()

                        class RealMLP(nn.Module):
                            def __init__(self):
                                super().__init__()
                                self.c_fc = nn.Linear(3, 6)

                        self.mlp = RealMLP()

                self.transformer = nn.Module()
                self.transformer.h = nn.ModuleList([RealBlock()])
                self.dummy_param = nn.Parameter(torch.randn(1))

            def forward(self, input_ids):
                batch_size, seq_len = input_ids.shape
                x = torch.randn(batch_size, seq_len, 3)
                self.transformer.h[0].mlp.c_fc(x)
                logits = torch.randn(batch_size, seq_len, 15)
                result = Mock()
                result.logits = logits
                return result

        model = ExceptionTestModel()
        calib_data = [{"input_ids": torch.randint(0, 100, (3, 8))}]
        call_count = 0

        def failing_mi_regression(X, y, random_state=None):
            nonlocal call_count
            call_count += 1
            if call_count % 3 == 0:
                raise ValueError("MI computation failed")
            return [0.5]

        with patch(
            "invarlock.eval.probes.mi.mutual_info_regression",
            side_effect=failing_mi_regression,
        ):
            scores = compute_neuron_mi_scores(
                model=model, calib_data=calib_data, oracle_windows=1
            )

            assert len(scores) == 1
            assert scores[0].shape == (6,)
            assert torch.any(scores[0] == 0.0)
            assert torch.any(scores[0] > 0.0)
