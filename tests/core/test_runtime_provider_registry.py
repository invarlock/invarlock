from __future__ import annotations

import sys
from importlib.metadata import EntryPoint
from types import ModuleType

import pytest

import invarlock.core.registry as registry_mod
from invarlock.core.builtin_plugin_catalog import builtin_plugin_specs
from invarlock.core.plugins_inventory import (
    gather_runtime_provider_inventory_rows,
    runtime_provider_inventory_json_items,
)


def test_builtin_runtime_provider_catalog_declares_only_hf_foundation() -> None:
    specs = builtin_plugin_specs("runtime_providers")

    assert [(spec.name, spec.module, spec.class_name) for spec in specs] == [
        (
            "hf_transformers",
            "invarlock.runtime_providers.hf_transformers",
            "HFTransformersProvider",
        )
    ]


def test_registry_lists_builtin_runtime_provider_without_importing_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    imported: list[str] = []
    real_import = registry_mod.importlib.import_module

    def guarded_import(name: str, *args, **kwargs):
        imported.append(name)
        if name.startswith("invarlock.runtime_providers"):
            raise AssertionError("runtime provider imported during discovery")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(registry_mod.importlib, "import_module", guarded_import)
    registry = registry_mod.CoreRegistry()

    assert registry.list_runtime_providers() == ["hf_transformers"]
    assert registry.get_plugin_info("hf_transformers", "runtime_providers") == {
        "available": True,
        "status": "Built-in",
        "module": "invarlock.runtime_providers.hf_transformers",
        "package": "invarlock",
        "version": registry_mod.INVARLOCK_VERSION,
        "entry_point": None,
        "entry_point_group": None,
        "support_tier": "core_supported",
        "strict_assurance_allowed": True,
        "maintained_catalog": False,
        "deployment_claim": False,
    }
    assert not any(name.startswith("invarlock.runtime_providers") for name in imported)


def test_registry_loads_hf_reference_provider_only_on_request() -> None:
    registry = registry_mod.CoreRegistry()

    provider = registry.get_runtime_provider("hf_transformers")

    assert provider.name == "hf_transformers"
    assert provider.abi_version == "1"


def test_runtime_provider_entry_point_discovery_is_lazy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class SelectEntryPoints:
        def __init__(self, provider: EntryPoint):
            self.provider = provider

        def select(self, *, group: str):
            if group == "invarlock.runtime_providers":
                return [self.provider]
            return []

    provider = EntryPoint(
        name="third_party_runtime",
        value="third_party.runtime:Provider",
        group="invarlock.runtime_providers",
    )
    monkeypatch.setattr(registry_mod, "third_party_plugins_allowed", lambda: True)
    monkeypatch.setattr(
        registry_mod, "entry_points", lambda: SelectEntryPoints(provider)
    )
    registry = registry_mod.CoreRegistry()

    assert registry.list_runtime_providers() == [
        "hf_transformers",
        "third_party_runtime",
    ]
    assert (
        registry.get_plugin_info("third_party_runtime", "runtime_providers")["module"]
        == "third_party.runtime"
    )


def test_runtime_provider_entry_point_name_must_match_public_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class SelectEntryPoints:
        def select(self, *, group: str):
            if group == "invarlock.runtime_providers":
                return [
                    EntryPoint(
                        name="runtime-provider",
                        value="third_party.runtime:Provider",
                        group=group,
                    )
                ]
            return []

    monkeypatch.setattr(registry_mod, "third_party_plugins_allowed", lambda: True)
    monkeypatch.setattr(registry_mod, "entry_points", SelectEntryPoints)

    with pytest.raises(RuntimeError, match="Invalid runtime provider plugin name"):
        registry_mod.CoreRegistry().list_runtime_providers()


def test_runtime_provider_metadata_rejects_unknown_category_and_name() -> None:
    registry = registry_mod.CoreRegistry()

    assert (
        registry.get_plugin_info("missing", "runtime_providers")["available"] is False
    )
    with pytest.raises(ValueError, match="Unknown plugin type"):
        registry.get_plugin_info("missing", "runtime-provider")
    with pytest.raises(KeyError, match="Unknown runtime provider"):
        registry.get_runtime_provider("missing")


def test_runtime_provider_inventory_is_static_and_machine_readable() -> None:
    rows = gather_runtime_provider_inventory_rows(registry=registry_mod.CoreRegistry())

    assert rows[0]["name"] == "hf_transformers"
    assert rows[0]["required_extra"] == "invarlock[hf]"
    assert runtime_provider_inventory_json_items(rows)[0] == {
        "name": "hf_transformers",
        "kind": "runtime_provider",
        "module": "invarlock.runtime_providers.hf_transformers",
        "entry_point": None,
        "origin": "builtin",
        "status": "ready",
        "required_extra": "invarlock[hf]",
        "support_tier": "core_supported",
        "strict_assurance_allowed": True,
        "maintained_catalog": False,
        "deployment_claim": False,
    }


def test_runtime_provider_inventory_reports_missing_hf_extra_without_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = registry_mod.CoreRegistry._check_runtime_dependencies

    def missing_hf(self, deps):  # noqa: ANN001
        if deps == ["torch", "transformers"]:
            return list(deps)
        return original(self, deps)

    monkeypatch.setattr(
        registry_mod.CoreRegistry, "_check_runtime_dependencies", missing_hf
    )
    registry = registry_mod.CoreRegistry()

    rows = gather_runtime_provider_inventory_rows(registry=registry)

    assert rows[0]["status"] == "needs_extra"
    assert rows[0]["enable"] == "pip install 'invarlock[hf]'"
    with pytest.raises(ImportError, match="Needs extra: torch, transformers"):
        registry.get_runtime_provider("hf_transformers")


def test_runtime_provider_loading_requires_separate_exact_abi_and_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Provider:
        name = "test_runtime"
        abi_version = "1"

        def validate_config(self, spec):  # noqa: ANN001
            return None

        def capabilities(self):
            return None

        def identify_artifact(self, spec):  # noqa: ANN001
            return None

        def open(self, spec, context):  # noqa: ANN001
            return None

    module_name = "invarlock_test_runtime_provider"
    Provider.__module__ = module_name
    module = ModuleType(module_name)
    module.INVARLOCK_RUNTIME_PROVIDER_ABI = "1"
    module.Provider = Provider
    monkeypatch.setitem(sys.modules, module_name, module)

    registry = registry_mod.CoreRegistry()
    registry._initialized = True
    registry._runtime_providers["test_runtime"] = registry_mod.PluginInfo(
        name="test_runtime",
        module=module_name,
        class_name="Provider",
        available=True,
        status="Available",
    )

    assert registry.get_runtime_provider("test_runtime").name == "test_runtime"

    module.INVARLOCK_RUNTIME_PROVIDER_ABI = "2"
    with pytest.raises(ImportError, match="ABI mismatch"):
        registry.get_runtime_provider("test_runtime")

    module.INVARLOCK_RUNTIME_PROVIDER_ABI = "1"
    Provider.name = "different"
    with pytest.raises(ImportError, match="identity mismatch"):
        registry.get_runtime_provider("test_runtime")
