from __future__ import annotations

from typing import Any


class EntryPointStub:
    """Small stand-in for importlib.metadata.EntryPoint."""

    def __init__(
        self, name: str, value: str, dist: Any | None = None, loader: Any | None = None
    ) -> None:
        self.name = name
        self.value = value
        self.dist = dist
        self._loader = loader

    def load(self):  # pragma: no cover - exercised via get_* calls
        if self._loader is not None:
            return self._loader
        mod, _, attr = self.value.partition(":")
        module = __import__(mod, fromlist=[attr])
        return getattr(module, attr)


class DistStub:
    def __init__(self, name: str, version: str) -> None:
        self.name = name
        self.version = version
        self.metadata = {"Name": name}


class SelectEntryPoints:
    def __init__(self, **groups: list[EntryPointStub]) -> None:
        self._groups = {f"invarlock.{key}": value for key, value in groups.items()}

    def select(self, *, group: str) -> list[EntryPointStub]:
        return list(self._groups.get(group, []))
