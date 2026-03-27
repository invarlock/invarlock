"""
InvarLock Retry Controller
=====================

Manages retry logic for automated evaluation workflows with:
- Attempt budgets (max 3 attempts default)
- Time budgets (optional timeout)
- Parameter adjustment strategies per edit type
- Gate-driven retry decisions
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

__all__ = ["EditAdjustmentResult", "RetryController", "adjust_edit_params"]


@dataclass(frozen=True)
class EditAdjustmentResult:
    params: dict[str, Any]
    notices: tuple[str, ...] = ()


class RetryController:
    """
    Controls retry logic for evaluation-report-driven automation.

    Features:
    - Attempt budget enforcement (default 3 max)
    - Optional timeout enforcement
    - Edit parameter adjustment between attempts
    - Logging of retry decisions
    """

    def __init__(
        self, max_attempts: int = 3, timeout: int | None = None, verbose: bool = False
    ):
        """
        Initialize retry controller.

        Args:
            max_attempts: Maximum retry attempts (default 3)
            timeout: Optional timeout in seconds
            verbose: Enable verbose logging
        """
        self.max_attempts = max_attempts
        self.timeout = timeout
        self.verbose = verbose
        self.start_time = time.time()
        self.attempt_history: list[dict[str, Any]] = []
        self._pending_notices: list[str] = []

    def should_retry(self, report_passed: bool) -> bool:
        """
        Determine if retry should be attempted.

        Args:
            report_passed: Whether evaluation report gates passed

        Returns:
            True if retry should be attempted, False otherwise
        """
        # If report passed, no retry needed
        if report_passed:
            return False

        # Check attempt budget (attempt count equals history length)
        if len(self.attempt_history) >= self.max_attempts:
            if self.verbose:
                self._pending_notices.append(
                    f"Exhausted {self.max_attempts} attempts, stopping retry"
                )
            return False

        # Check timeout budget
        if self.timeout is not None:
            elapsed = time.time() - self.start_time
            if elapsed > self.timeout:
                if self.verbose:
                    self._pending_notices.append(
                        f"Timeout {self.timeout}s exceeded ({elapsed:.1f}s), stopping retry"
                    )
                return False

        # Retry is allowed - increment attempt counter for next check
        return True

    def record_attempt(
        self,
        attempt_num: int,
        report_result: dict[str, Any],
        edit_params: dict[str, Any],
    ) -> None:
        """Record details of an attempt for tracking."""
        report_result = report_result or {}
        edit_params = edit_params or {}

        self.attempt_history.append(
            {
                "attempt": attempt_num,
                "timestamp": time.time(),
                "report_passed": report_result.get("passed", False),
                "edit_params": edit_params.copy(),
                "failures": report_result.get("failures", []),
                "validation": report_result.get("validation", {}),
            }
        )

    def drain_notices(self) -> tuple[str, ...]:
        notices = tuple(self._pending_notices)
        self._pending_notices.clear()
        return notices

    def get_attempt_summary(self) -> dict[str, Any]:
        """Get summary of all retry attempts."""
        return {
            "total_attempts": len(self.attempt_history),
            "max_attempts": self.max_attempts,
            "timeout": self.timeout,
            "elapsed_time": time.time() - self.start_time,
            "attempts": self.attempt_history,
        }


def adjust_edit_params(
    edit_name: str,
    edit_params: dict[str, Any],
    attempt: int,
    report_result: dict[str, Any] | None = None,
) -> EditAdjustmentResult:
    """
    Adjust edit parameters for retry attempt based on edit type and failure mode.

    Strategies:
    - Quant: Add/increase clamp_ratio for stability

    Args:
        edit_name: Name of the edit operation
        edit_params: Current edit parameters
        attempt: Attempt number (1-indexed)
        report_result: Optional evaluation report result for failure analysis

    Returns:
        Adjusted parameters plus caller-rendered notices for the next attempt
    """
    adjusted = edit_params.copy()
    notices: list[str] = []

    # Quantization adjustments
    if "quant" in edit_name.lower():
        # Add clamp_ratio for stability
        if "clamp_ratio" not in adjusted:
            adjusted["clamp_ratio"] = 0.01
            notices.append("Quant retry adjustment: added clamp_ratio=0.01")
        else:
            # Could increase existing clamp_ratio if needed
            pass

    return EditAdjustmentResult(params=adjusted, notices=tuple(notices))
