from __future__ import annotations

import sys
import types
from typing import Any

import pytest

import invarlock.core.registry as reg


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


class MappingEntryPoints(dict):
    pass


def install_plain_module(
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
    **attrs: object,
) -> types.ModuleType:
    module = types.ModuleType(module_name)
    for name, value in attrs.items():
        if isinstance(value, type):
            value.__module__ = module_name
        setattr(module, name, value)
    monkeypatch.setitem(sys.modules, module_name, module)
    return module


def install_plugin_module(
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
    *,
    abi: str,
) -> tuple[type[reg.ModelAdapter], type[reg.ModelEdit], type[reg.Guard]]:
    class DummyAdapter(reg.ModelAdapter):
        name = "dummy_adapter"

        def can_handle(self, model: Any) -> bool:
            return True

        def describe(self, model: Any) -> dict[str, Any]:
            return {"n_layer": 1}

        def snapshot(self, model: Any) -> bytes:
            return b"snapshot"

        def restore(self, model: Any, blob: bytes) -> None:
            return None

    class DummyEdit(reg.ModelEdit):
        name = "dummy_edit"

        def can_edit(self, model_desc: dict[str, Any]) -> bool:
            return True

        def apply(
            self, model: Any, adapter: reg.ModelAdapter, **kwargs: Any
        ) -> dict[str, Any]:
            return {"ok": True}

    class DummyGuard(reg.Guard):
        name = "dummy_guard"

        def validate(
            self, model: Any, adapter: reg.ModelAdapter, context: dict[str, Any]
        ) -> dict[str, Any]:
            return {"passed": True}

    install_plain_module(
        monkeypatch,
        module_name,
        INVARLOCK_CORE_ABI=abi,
        DummyAdapter=DummyAdapter,
        DummyEdit=DummyEdit,
        DummyGuard=DummyGuard,
    )
    return DummyAdapter, DummyEdit, DummyGuard
