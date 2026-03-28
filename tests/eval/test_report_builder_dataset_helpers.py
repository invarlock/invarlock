# ruff: noqa: F405
from __future__ import annotations

from tests.eval.report_builder_support import *  # noqa: F401,F403,F405


class TestNormalizationAndDataset:
    """Coverage for normalization helpers and dataset hashing."""

    def test_normalize_baseline_invalid_ppl_raises(self):
        baseline = {
            "run_id": "r1",
            "model_id": "m",
            "ppl_final": 0.0,
            "ppl_preview": 0.4,
        }
        with pytest.raises(ValueError, match="Invalid baseline"):
            _normalize_baseline(baseline)

    def test_normalize_baseline_schema_v1(self):
        baseline = {
            "schema_version": "baseline-v1",
            "meta": {"model_id": "m", "commit_sha": "abc"},
            "metrics": {
                "ppl_final": 9.5,
                "ppl_preview": 9.4,
                "spectral": {"sigma_ratios": [1.0]},
                "bootstrap": {"replicates": 1000},
            },
            "spectral_base": {"sigma_ratios": [1.0]},
            "rmt_base": {"outliers": 1},
            "invariants": {"weight_norm": {"passed": True}},
        }
        normalized = _normalize_baseline(baseline)
        assert normalized["ppl_final"] == 9.5
        assert "spectral" in normalized

    def test_normalize_baseline_invalid_type_raises(self):
        with pytest.raises(ValueError):
            _normalize_baseline("not-a-baseline")

    def test_extract_dataset_info_with_windows(self):
        report = create_mock_run_report(include_evaluation_windows=True)
        dataset_info = _extract_dataset_info(report)
        assert dataset_info["hash"]["preview"].startswith("sha256:")
        assert dataset_info["hash"]["final"].startswith("sha256:")

    def test_extract_dataset_info_config_fallback(self):
        report = create_mock_run_report(include_evaluation_windows=False)
        dataset_info = _extract_dataset_info(report)
        assert dataset_info["hash"]["dataset"] is None
        assert dataset_info["hash"]["total_tokens"] > 0


class TestComputeWindowHashes:
    """Test compute_window_hashes function."""

    def test_basic_hash_computation(self):
        """Test basic window hash computation."""
        # Mock EvaluationWindow objects
        preview_window = Mock()
        preview_window.input_ids = [[1, 2, 3], [4, 5, 6]]

        final_window = Mock()
        final_window.input_ids = [[7, 8, 9], [10, 11, 12]]

        with patch(
            "invarlock.reporting.dataset_hashing.compute_window_hash"
        ) as mock_hash:
            mock_hash.side_effect = ["preview_hash123", "final_hash456"]

            result = compute_window_hashes(preview_window, final_window)

            assert result["preview"] == "sha256:preview_hash123"
            assert result["final"] == "sha256:final_hash456"
            assert result["total_tokens"] == 12  # 6 tokens in each window

            # Verify compute_window_hash was called correctly
            assert mock_hash.call_count == 2
            mock_hash.assert_any_call(preview_window, include_data=True)
            mock_hash.assert_any_call(final_window, include_data=True)
