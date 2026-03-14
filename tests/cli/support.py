from __future__ import annotations


class RecordingConsole:
    def __init__(self, *, fail_with_kwargs: bool = False) -> None:
        self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
        self._fail_with_kwargs = fail_with_kwargs

    def print(self, *args: object, **kwargs: object) -> None:
        if self._fail_with_kwargs and kwargs:
            self._fail_with_kwargs = False
            raise TypeError("kwargs unsupported")
        self.calls.append((args, kwargs))

    @property
    def lines(self) -> list[str]:
        return [" ".join(str(arg) for arg in args) for args, _ in self.calls]

    def joined(self) -> str:
        return "\n".join(self.lines)
