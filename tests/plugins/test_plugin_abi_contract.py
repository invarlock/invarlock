from __future__ import annotations

from types import SimpleNamespace

import pytest

from invarlock.core.api import Guard, ModelAdapter, ModelEdit
from invarlock.core.registry import CoreRegistry, PluginInfo


class _BadAdapter(ModelAdapter):
    name = "bad_adapter"

    def can_handle(self, model):
        return True

    def describe(self, model):
        return {}

    def snapshot(self, model):
        return b""

    def restore(self, model, blob):
        return None


class _BadEdit(ModelEdit):
    name = "bad_edit"

    def can_edit(self, model_desc):
        return True

    def apply(self, model, adapter, **kwargs):
        return {}


class _BadGuard(Guard):
    name = "bad_guard"

    def validate(self, model, adapter, context):
        return {"passed": True}


def _registry_with(plugin_type: str, name: str, module_name: str) -> CoreRegistry:
    registry = CoreRegistry()
    registry._initialized = True
    info = PluginInfo(
        name=name,
        module=module_name,
        class_name="ignored",
        available=True,
        status="Deferred load",
        entry_point=None,
    )
    if plugin_type == "adapters":
        registry._adapters[name] = info
    elif plugin_type == "edits":
        registry._edits[name] = info
    else:
        registry._guards[name] = info
    return registry


def test_registry_rejects_adapter_edit_and_guard_abi_mismatch(monkeypatch) -> None:
    _BadAdapter.__module__ = "bad.adapter"
    _BadEdit.__module__ = "bad.edit"
    _BadGuard.__module__ = "bad.guard"

    modules = {
        "bad.adapter": SimpleNamespace(
            INVARLOCK_CORE_ABI="9.9", BadAdapter=_BadAdapter
        ),
        "bad.edit": SimpleNamespace(INVARLOCK_CORE_ABI="9.9", BadEdit=_BadEdit),
        "bad.guard": SimpleNamespace(INVARLOCK_CORE_ABI="9.9", BadGuard=_BadGuard),
    }

    def fake_import(name: str):
        if name in modules:
            return modules[name]
        raise ImportError(name)

    monkeypatch.setattr("invarlock.core.registry.importlib.import_module", fake_import)

    adapter_registry = _registry_with("adapters", "bad", "bad.adapter")
    adapter_registry._adapters["bad"].class_name = "BadAdapter"
    with pytest.raises(ImportError, match="ABI mismatch"):
        adapter_registry.get_adapter("bad")

    edit_registry = _registry_with("edits", "bad", "bad.edit")
    edit_registry._edits["bad"].class_name = "BadEdit"
    with pytest.raises(ImportError, match="ABI mismatch"):
        edit_registry.get_edit("bad")

    guard_registry = _registry_with("guards", "bad", "bad.guard")
    guard_registry._guards["bad"].class_name = "BadGuard"
    with pytest.raises(ImportError, match="ABI mismatch"):
        guard_registry.get_guard("bad")
