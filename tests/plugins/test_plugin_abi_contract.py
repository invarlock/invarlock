from __future__ import annotations

import importlib
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest

from invarlock.core import INVARLOCK_CORE_ABI
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

    def apply(self, model, adapter, plan=None, runtime=None):
        _ = model, adapter, plan, runtime
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


def test_registry_rejects_plugins_without_declared_abi(monkeypatch) -> None:
    _BadAdapter.__module__ = "missing.adapter"
    _BadEdit.__module__ = "missing.edit"
    _BadGuard.__module__ = "missing.guard"

    modules = {
        "missing.adapter": SimpleNamespace(BadAdapter=_BadAdapter),
        "missing.edit": SimpleNamespace(BadEdit=_BadEdit),
        "missing.guard": SimpleNamespace(BadGuard=_BadGuard),
    }

    def fake_import(name: str):
        if name in modules:
            return modules[name]
        raise ImportError(name)

    monkeypatch.setattr("invarlock.core.registry.importlib.import_module", fake_import)

    adapter_registry = _registry_with("adapters", "missing", "missing.adapter")
    adapter_registry._adapters["missing"].class_name = "BadAdapter"
    with pytest.raises(ImportError, match="ABI missing"):
        adapter_registry.get_adapter("missing")

    edit_registry = _registry_with("edits", "missing", "missing.edit")
    edit_registry._edits["missing"].class_name = "BadEdit"
    with pytest.raises(ImportError, match="ABI missing"):
        edit_registry.get_edit("missing")

    guard_registry = _registry_with("guards", "missing", "missing.guard")
    guard_registry._guards["missing"].class_name = "BadGuard"
    with pytest.raises(ImportError, match="ABI missing"):
        guard_registry.get_guard("missing")


def test_builtin_provider_modules_declare_core_abi() -> None:
    provider_modules = (
        "invarlock.adapters.auto",
        "invarlock.adapters.hf_causal",
        "invarlock.adapters.hf_mlm",
        "invarlock.adapters.hf_multimodal",
        "invarlock.adapters.hf_seq2seq",
        "invarlock.edits",
        "invarlock.edits.quant_rtn",
        "invarlock.guards.invariants",
        "invarlock.guards.rmt",
        "invarlock.guards.spectral",
        "invarlock.guards.variance",
        "invarlock.plugins",
    )

    missing = []
    for module_name in provider_modules:
        module = importlib.import_module(module_name)
        if getattr(module, "INVARLOCK_CORE_ABI", None) != INVARLOCK_CORE_ABI:
            missing.append(module_name)

    assert not missing, "\n".join(missing)


def test_declared_plugin_entry_points_are_loadable() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    project = tomllib.loads((repo_root / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]
    failures: list[str] = []

    for group_name, entries in sorted(project["entry-points"].items()):
        for entry_name, value in sorted(entries.items()):
            module_name, separator, attribute_name = value.partition(":")
            if not separator:
                failures.append(f"{group_name}.{entry_name}: invalid target {value!r}")
                continue
            try:
                module = importlib.import_module(module_name)
                getattr(module, attribute_name)
            except (AttributeError, ImportError) as exc:
                failures.append(f"{group_name}.{entry_name}: {value}: {exc}")

    assert failures == []
