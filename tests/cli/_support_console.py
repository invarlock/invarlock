from __future__ import annotations

import io

from rich.console import Console as RichConsole


class RecordingConsole:
    def __init__(self, *, fail_with_kwargs: bool = False) -> None:
        self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
        self.rendered: list[str] = []
        self._fail_with_kwargs = fail_with_kwargs

    def print(self, *args: object, **kwargs: object) -> None:
        if self._fail_with_kwargs and kwargs:
            self._fail_with_kwargs = False
            raise TypeError("kwargs unsupported")
        self.calls.append((args, kwargs))
        if all(isinstance(arg, str) for arg in args):
            self.rendered.append(" ".join(str(arg) for arg in args))
            return
        buffer = io.StringIO()
        render_console = RichConsole(
            file=buffer,
            force_terminal=False,
            color_system=None,
            width=4096,
        )
        try:
            render_console.print(*args, **kwargs)
        except TypeError:
            render_console.print(*args)
        self.rendered.append(buffer.getvalue().rstrip("\n"))

    @property
    def lines(self) -> list[str]:
        return list(self.rendered)

    def joined(self) -> str:
        return "\n".join(self.lines)
