from __future__ import annotations

import importlib.util
import sys
from importlib.metadata import EntryPoint, PackageNotFoundError
from types import ModuleType, SimpleNamespace
from typing import cast

import pytest

import invarlock.core.registry as registry_module
from invarlock import __version__ as _VERSION
from invarlock.core.runtime_provider import INVARLOCK_RUNTIME_PROVIDER_ABI


class DistStub:
    def __init__(self, name: str, version: str) -> None:
        self.name = name
        self.version = version


class EntryPointStub:
    def __init__(
        self,
        *,
        name: str,
        value: str,
        dist: DistStub | None = None,
        loader: object | None = None,
    ) -> None:
        self.name = name
        self.value = value
        self.dist = dist
        self._loader = loader

    def load(self) -> object:
        assert self._loader is not None
        return self._loader


def test_entry_point_selection_supports_legacy_mapping_shape() -> None:
    entry = EntryPoint(
        name="vendor_runtime",
        value="vendor.runtime:Provider",
        group="invarlock.runtime_providers",
    )

    assert registry_module._select_entry_points(
        {"invarlock.runtime_providers": [entry]}
    ) == [entry]


def test_shipped_entry_point_requires_exact_distribution_version_and_value() -> None:
    entry = EntryPointStub(
        name="hf_transformers",
        value=("invarlock.runtime_providers.hf_transformers:HFTransformersProvider"),
        dist=DistStub("InvarLock", _VERSION),
    )
    assert registry_module._is_shipped_entry_point(cast(EntryPoint, entry)) is True

    entry.dist = DistStub("invarlock", "wrong-version")
    assert registry_module._is_shipped_entry_point(cast(EntryPoint, entry)) is False
    entry.dist = None
    assert registry_module._is_shipped_entry_point(cast(EntryPoint, entry)) is False


def test_builtin_registration_rejects_duplicate_name() -> None:
    registry = registry_module.CoreRegistry()
    registry._runtime_providers["hf_transformers"] = registry_module.PluginInfo(
        name="hf_transformers",
        module="fixture",
        class_name="Provider",
        required_deps=(),
        available=True,
    )

    with pytest.raises(RuntimeError, match="Duplicate built-in"):
        registry._register_builtins()


def test_missing_dependency_detection_treats_lookup_errors_as_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def find_spec(name: str) -> object | None:
        if name == "broken":
            raise ValueError("invalid module name")
        return None if name == "missing" else object()

    monkeypatch.setattr(importlib.util, "find_spec", find_spec)

    assert registry_module.CoreRegistry._missing_dependencies(
        ("present", "missing", "broken")
    ) == ["missing", "broken"]


@pytest.mark.parametrize("value", [None, "module", ":Provider", "module:"])
def test_entry_point_value_must_be_one_module_and_class_reference(
    value: object,
) -> None:
    entry = SimpleNamespace(value=value)
    with pytest.raises((TypeError, ValueError), match="value must be|string|malformed"):
        registry_module.CoreRegistry._parse_entry_point(entry)  # type: ignore[arg-type]


def test_duplicate_entry_point_accepts_only_exact_shipped_metadata() -> None:
    registry = registry_module.CoreRegistry()
    registry._register_builtins()
    shipped = EntryPointStub(
        name="hf_transformers",
        value=("invarlock.runtime_providers.hf_transformers:HFTransformersProvider"),
        dist=DistStub("invarlock", _VERSION),
    )
    registry._register_entry_point(cast(EntryPoint, shipped))

    spoofed = EntryPointStub(
        name="hf_transformers",
        value="vendor.runtime:Provider",
        dist=DistStub("vendor", "1"),
    )
    with pytest.raises(RuntimeError, match="Duplicate runtime provider"):
        registry._register_entry_point(cast(EntryPoint, spoofed))


def test_entry_point_without_distribution_uses_module_package_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entry = EntryPointStub(
        name="vendor_runtime",
        value="vendor.runtime:Provider",
        dist=None,
    )
    monkeypatch.setattr(
        registry_module,
        "metadata_version",
        lambda _package: (_ for _ in ()).throw(PackageNotFoundError("vendor")),
    )
    registry = registry_module.CoreRegistry()
    registry._register_entry_point(cast(EntryPoint, entry))

    info = registry._runtime_providers["vendor_runtime"]
    assert info.package == "vendor"
    assert info.version is None


def test_entry_point_class_resolution_uses_deferred_loader() -> None:
    class Provider:
        pass

    entry = EntryPointStub(
        name="vendor_runtime",
        value="vendor.runtime:Provider",
        loader=Provider,
    )
    info = registry_module.PluginInfo(
        name="vendor_runtime",
        module="vendor.runtime",
        class_name="Provider",
        required_deps=(),
        available=True,
        entry_point=cast(EntryPoint, entry),
    )

    assert registry_module.CoreRegistry._resolve_provider_class(info) is Provider


def _install_provider_module(
    monkeypatch: pytest.MonkeyPatch,
    *,
    provider_name: str,
    instance_abi: str,
    protocol_complete: bool = True,
) -> registry_module.PluginInfo:
    module_name = f"test_registry_{provider_name}_{instance_abi}"

    class Provider:
        name = provider_name
        abi_version = instance_abi

        if protocol_complete:

            def validate_config(self, _spec: object) -> None:
                return None

            def capabilities(self) -> None:
                return None

            def identify_artifact(self, _spec: object) -> None:
                return None

            def authenticate_artifact(
                self, _spec: object, _artifact_path: object
            ) -> None:
                return None

            def prepare_execution(self, _spec: object, resources: object) -> object:
                return resources

            def open(self, _spec: object, _context: object) -> None:
                return None

    Provider.__module__ = module_name
    module = ModuleType(module_name)
    module.__dict__["INVARLOCK_RUNTIME_PROVIDER_ABI"] = INVARLOCK_RUNTIME_PROVIDER_ABI
    module.__dict__["Provider"] = Provider
    monkeypatch.setitem(sys.modules, module_name, module)
    return registry_module.PluginInfo(
        name=provider_name,
        module=module_name,
        class_name="Provider",
        required_deps=(),
        available=True,
    )


def test_provider_instantiation_rejects_non_protocol_and_instance_abi(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = registry_module.CoreRegistry()
    incomplete = _install_provider_module(
        monkeypatch,
        provider_name="incomplete",
        instance_abi="1",
        protocol_complete=False,
    )
    with pytest.raises(ImportError, match="Expected RuntimeProvider"):
        registry._instantiate(incomplete)

    wrong_abi = _install_provider_module(
        monkeypatch,
        provider_name="wrong_abi",
        instance_abi="9",
    )
    with pytest.raises(ImportError, match="instance ABI"):
        registry._instantiate(wrong_abi)


def test_unavailable_provider_reports_required_dependencies() -> None:
    registry = registry_module.CoreRegistry()
    registry._initialized = True
    registry._runtime_providers["optional"] = registry_module.PluginInfo(
        name="optional",
        module="optional.provider",
        class_name="Provider",
        required_deps=("optional_backend",),
        available=False,
    )

    with pytest.raises(ImportError, match="optional_backend"):
        registry.get_runtime_provider("optional")


def test_global_registry_accessor_returns_stable_instance() -> None:
    assert registry_module.get_registry() is registry_module.get_registry()
