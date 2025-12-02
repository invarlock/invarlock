"""
Tests for RetryController and retry parameter adjustments.
"""

import time

from invarlock.core.retry import RetryController, adjust_edit_params


class TestRetryController:
    def test_initialization(self) -> None:
        controller = RetryController(max_attempts=3, timeout=None, verbose=False)
        assert controller.max_attempts == 3
        assert controller.timeout is None
        assert controller.attempt_history == []

    def test_should_retry_on_pass(self) -> None:
        controller = RetryController(max_attempts=3)
        assert controller.should_retry(certificate_passed=True) is False
        assert controller.attempt_history == []

    def test_attempt_budget_enforced(self) -> None:
        controller = RetryController(max_attempts=3)

        controller.record_attempt(1, {"passed": False}, {"energy_keep": 0.98})
        assert controller.should_retry(False) is True

        controller.record_attempt(2, {"passed": False}, {"energy_keep": 0.99})
        assert controller.should_retry(False) is True

        controller.record_attempt(3, {"passed": False}, {"energy_keep": 0.995})
        assert controller.should_retry(False) is False
        assert len(controller.attempt_history) == 3

    def test_timeout_budget_enforced(self) -> None:
        controller = RetryController(max_attempts=5, timeout=1, verbose=False)

        controller.record_attempt(1, {"passed": False}, {})
        assert controller.should_retry(False) is True

        time.sleep(1.1)
        assert controller.should_retry(False) is False

    def test_record_attempt_handles_none(self) -> None:
        controller = RetryController(max_attempts=2)
        controller.record_attempt(1, None, None)
        entry = controller.attempt_history[0]
        assert entry["certificate_passed"] is False
        assert entry["failures"] == []
        assert entry["edit_params"] == {}

    def test_get_attempt_summary(self) -> None:
        controller = RetryController(max_attempts=3, timeout=60)
        controller.record_attempt(1, {"passed": False}, {"param": "value1"})
        controller.record_attempt(2, {"passed": True}, {"param": "value2"})

        summary = controller.get_attempt_summary()
        assert summary["total_attempts"] == 2
        assert summary["max_attempts"] == 3
        assert summary["timeout"] == 60
        assert len(summary["attempts"]) == 2

    def test_zero_max_attempts(self) -> None:
        controller = RetryController(max_attempts=0)
        assert controller.should_retry(False) is False

    def test_negative_timeout(self) -> None:
        controller = RetryController(max_attempts=3, timeout=-1)
        controller.record_attempt(1, {"passed": False}, {})
        assert controller.should_retry(False) is False


class TestAdjustEditParams:
    def test_adjust_quant_adds_clamp_ratio(self) -> None:
        params = {"bits": 8}
        adjusted = adjust_edit_params("quant_rtn", params, attempt=1)
        assert adjusted["clamp_ratio"] == 0.01

    def test_adjust_unknown_edit_noop(self) -> None:
        params = {"some_param": "value"}
        adjusted = adjust_edit_params("unknown_edit", params, attempt=1)
        assert adjusted == params

    def test_adjust_preserves_other_fields(self) -> None:
        params = {"bits": 8, "scope": "ffn", "seed": 42}
        adjusted = adjust_edit_params("quant_rtn", params, attempt=1)
        assert adjusted["scope"] == "ffn"
        assert adjusted["seed"] == 42
