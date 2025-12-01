"""
InvarLock Retry Controller
=====================

Manages retry logic for automated certification workflows with:
- Attempt budgets (max 3 attempts default)
- Time budgets (optional timeout)
- Parameter adjustment strategies per edit type
- Certificate-driven retry decisions
"""

from __future__ import annotations

import time
from typing import Any

__all__ = ["RetryController", "adjust_edit_params"]


class RetryController:
    """
    Controls retry logic for certificate-driven automation.

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

    def should_retry(self, certificate_passed: bool) -> bool:
        """
        Determine if retry should be attempted.

        Args:
            certificate_passed: Whether certificate validation passed

        Returns:
            True if retry should be attempted, False otherwise
        """
        # If certificate passed, no retry needed
        if certificate_passed:
            return False

        # Check attempt budget (attempt count equals history length)
        if len(self.attempt_history) >= self.max_attempts:
            if self.verbose:
                print(f"Exhausted {self.max_attempts} attempts, stopping retry")
            return False

        # Check timeout budget
        if self.timeout is not None:
            elapsed = time.time() - self.start_time
            if elapsed > self.timeout:
                if self.verbose:
                    print(
                        f"Timeout {self.timeout}s exceeded ({elapsed:.1f}s), stopping retry"
                    )
                return False

        # Retry is allowed - increment attempt counter for next check
        return True

    def record_attempt(
        self,
        attempt_num: int,
        certificate_result: dict[str, Any],
        edit_params: dict[str, Any],
    ) -> None:
        """Record details of an attempt for tracking."""
        certificate_result = certificate_result or {}
        edit_params = edit_params or {}

        self.attempt_history.append(
            {
                "attempt": attempt_num,
                "timestamp": time.time(),
                "certificate_passed": certificate_result.get("passed", False),
                "edit_params": edit_params.copy(),
                "failures": certificate_result.get("failures", []),
                "validation": certificate_result.get("validation", {}),
            }
        )

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
    certificate_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Adjust edit parameters for retry attempt based on edit type and failure mode.

    Strategies:
    - Quant: Add/increase clamp_ratio for stability

    Args:
        edit_name: Name of the edit operation
        edit_params: Current edit parameters
        attempt: Attempt number (1-indexed)
        certificate_result: Optional certificate result for failure analysis

    Returns:
        Adjusted parameters for next attempt
    """
    adjusted = edit_params.copy()

    # Quantization adjustments
    if "quant" in edit_name.lower():
        # Add clamp_ratio for stability
        if "clamp_ratio" not in adjusted:
            adjusted["clamp_ratio"] = 0.01
            print("  Quant retry adjustment: added clamp_ratio=0.01")
        else:
            # Could increase existing clamp_ratio if needed
            pass

    return adjusted
