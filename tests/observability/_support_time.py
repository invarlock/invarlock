from __future__ import annotations

DEBOUNCE_SHORT_WAIT = 0.01
DEBOUNCE_SETTLE_WAIT = 0.15


class FakeClock:
    def __init__(self, start: float = 1000.0) -> None:
        self.now = float(start)

    def time(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += float(seconds)


class CallCounter:
    def __init__(self) -> None:
        self.count = 0

    def callback(self) -> None:
        self.count += 1
