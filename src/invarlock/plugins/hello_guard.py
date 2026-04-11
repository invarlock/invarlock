"""Template guard plugin for entry point demonstrations."""

from __future__ import annotations

from typing import Any

from invarlock.core.abi import INVARLOCK_CORE_ABI as CORE_ABI
from invarlock.core.api import Guard, ModelAdapter
from invarlock.core.types import GuardValidationResult

INVARLOCK_CORE_ABI = CORE_ABI


class HelloGuard(Guard):
    """Simple guard that checks a score in the validation context."""

    name = "hello_guard"

    def __init__(self, threshold: float = 1.0):
        self.threshold = float(threshold)

    def validate(
        self,
        model: Any,
        adapter: ModelAdapter,
        context: dict[str, Any],
    ) -> GuardValidationResult:
        score = float(context.get("hello_score", 0.0))
        passed = score <= self.threshold
        return GuardValidationResult(
            passed=passed,
            decision="allow" if passed else "block",
            metrics={"score": score},
            extras={
                "message": (
                    f"Hello guard score {score:.3f} (threshold {self.threshold:.3f})"
                )
            },
        )
