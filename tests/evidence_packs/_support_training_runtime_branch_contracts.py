from __future__ import annotations

from typing import Any


class _FakeFunction:
    def __init__(self, result: int) -> None:
        self.result = result
        self.argtypes: Any = None
        self.restype: Any = None

    def __call__(self, *_args: Any) -> int:
        return self.result
