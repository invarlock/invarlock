"""Offline wiring fixture; never makes provider calls or measures model quality."""

from __future__ import annotations

from typing import Any


def count_input_tokens(request: dict[str, Any]) -> int:
    # The toy tokenizer is part of this fixture, not a model tokenizer.
    return max(1, len(str(request["input"]).split()))


async def generate(request: dict[str, Any]) -> dict[str, Any]:
    return {
        "output": "offline fixture answer",
        "input_tokens": count_input_tokens(request),
        "output_tokens": 3,
    }
