"""Template guard plugin for entry point demonstrations."""

from __future__ import annotations

from typing import Any

from invarlock.core.api import Guard, ModelAdapter


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
    ) -> dict[str, Any]:
        score = float(context.get("hello_score", 0.0))
        passed = score <= self.threshold
        return {
            "passed": passed,
            "action": "warn" if passed else "abort",
            "message": f"Hello guard score {score:.3f} (threshold {self.threshold:.3f})",
            "metrics": {
                "score": score,
            },
        }
