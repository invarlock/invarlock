from __future__ import annotations

import sys
from importlib.metadata import EntryPoint
from pathlib import Path
from types import ModuleType

import pytest

import invarlock.core.registry as registry_mod
from invarlock.core.builtin_plugin_catalog import builtin_plugin_specs
from tests.core._support_registry import DistStub, EntryPointStub, SelectEntryPoints


def test_builtin_runtime_provider_catalog_declares_only_canonical_hf() -> None:
    specs = builtin_plugin_specs("runtime_providers")

    assert [(spec.name, spec.module, spec.class_name) for spec in specs] == [
        (
            "hf_transformers",
            "invarlock.runtime_providers.hf_transformers",
            "HFTransformersProvider",
        ),
    ]
    assert [spec.required_deps for spec in specs] == [()]


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
    monkeypatch.setattr(
        registry_mod,
        "entry_points",
        lambda: SelectEntryPoints(runtime_providers=[]),
    )
    registry = registry_mod.CoreRegistry()

    assert registry.list_runtime_providers() == ["hf_transformers"]
    assert registry.get_plugin_info("hf_transformers", "runtime_providers") == {
        "name": "hf_transformers",
        "module": "invarlock.runtime_providers.hf_transformers",
        "class_name": "HFTransformersProvider",
        "required_deps": (),
        "available": True,
        "package": "invarlock",
        "version": registry_mod.INVARLOCK_VERSION,
        "entry_point": None,
    }
    assert not any(name.startswith("invarlock.runtime_providers") for name in imported)


def test_registry_loads_hf_reference_provider_only_on_request() -> None:
    registry = registry_mod.CoreRegistry()

    provider = registry.get_runtime_provider("hf_transformers")

    assert provider.name == "hf_transformers"
    assert provider.abi_version == "1"


def test_base_install_keeps_hf_import_identity_available_without_backend_packages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_find_spec = registry_mod.importlib.util.find_spec

    def without_execution_backends(name: str, *args, **kwargs):
        if name in {"torch", "transformers"}:
            return None
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(
        registry_mod.importlib.util, "find_spec", without_execution_backends
    )
    registry = registry_mod.CoreRegistry()

    assert (
        registry.get_plugin_info("hf_transformers", "runtime_providers")["available"]
        is True
    )
    assert registry.get_runtime_provider("hf_transformers").name == "hf_transformers"


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


def test_exact_first_party_addins_are_discovered_without_third_party_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    providers = [
        EntryPointStub(
            name="hf_vision_text",
            value="invarlock_addins.multimodal.provider:HFVisionTextProvider",
            dist=DistStub(
                "invarlock-runtime-hf-vision-text", registry_mod.INVARLOCK_VERSION
            ),
        ),
        EntryPointStub(
            name="llama_cpp",
            value="invarlock_addins.gguf.provider:LlamaCppProvider",
            dist=DistStub("invarlock-runtime-gguf", registry_mod.INVARLOCK_VERSION),
        ),
        EntryPointStub(
            name="tensorrt_llm",
            value=("invarlock_addins.tensorrt_llm.provider:TensorRTLLMProvider"),
            dist=DistStub(
                "invarlock-runtime-tensorrt-llm", registry_mod.INVARLOCK_VERSION
            ),
        ),
        EntryPointStub(
            name="unapproved_runtime",
            value="vendor.runtime:Provider",
            dist=DistStub("vendor-runtime", "1.0"),
        ),
    ]
    monkeypatch.setattr(registry_mod, "third_party_plugins_allowed", lambda: False)
    monkeypatch.setattr(
        registry_mod,
        "entry_points",
        lambda: SelectEntryPoints(runtime_providers=providers),
    )

    registry = registry_mod.CoreRegistry()

    assert registry.list_runtime_providers() == [
        "hf_transformers",
        "hf_vision_text",
        "llama_cpp",
        "tensorrt_llm",
    ]
    assert "unapproved_runtime" not in registry.list_runtime_providers()
    for name in ("hf_vision_text", "llama_cpp", "tensorrt_llm"):
        info = registry.get_plugin_info(name, "runtime_providers")
        assert info["name"] == name
        assert info["required_deps"] == ()
        assert info["entry_point"] == name


def test_qualification_ignores_stale_first_party_addin_entry_points(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate_site = tmp_path / "candidate-site"
    stale_site = tmp_path / "stale-site"
    candidate_site.mkdir()
    stale_site.mkdir()

    class LocatedDist(DistStub):
        def __init__(self, location: Path) -> None:
            super().__init__("invarlock-runtime-gguf", registry_mod.INVARLOCK_VERSION)
            self.location = location

        def locate_file(self, path: str) -> Path:
            return self.location / path

    stale = EntryPointStub(
        name="llama_cpp",
        value="invarlock_addins.gguf.provider:LlamaCppProvider",
        dist=LocatedDist(stale_site),
    )
    candidate = EntryPointStub(
        name="llama_cpp",
        value="invarlock_addins.gguf.provider:LlamaCppProvider",
        dist=LocatedDist(candidate_site),
    )
    monkeypatch.setenv("INVARLOCK_QUALIFICATION_CANDIDATE_SITE", str(candidate_site))
    monkeypatch.setattr(registry_mod, "third_party_plugins_allowed", lambda: False)
    monkeypatch.setattr(
        registry_mod,
        "entry_points",
        lambda: SelectEntryPoints(runtime_providers=[stale, candidate]),
    )

    registry = registry_mod.CoreRegistry()

    assert registry.list_runtime_providers() == ["hf_transformers", "llama_cpp"]
    assert registry._runtime_providers["llama_cpp"].entry_point is candidate


@pytest.mark.parametrize(
    ("name", "distribution", "value"),
    [
        (
            "hf_vision_text",
            "spoofed-runtime-hf-vision-text",
            "invarlock_addins.multimodal.provider:HFVisionTextProvider",
        ),
        (
            "hf_vision_text",
            "invarlock-runtime-hf-vision-text",
            "spoofed.runtime:HFVisionTextProvider",
        ),
        (
            "llama_cpp",
            "spoofed-runtime-gguf",
            "invarlock_addins.gguf.provider:LlamaCppProvider",
        ),
        (
            "llama_cpp",
            "invarlock-runtime-gguf",
            "spoofed.runtime:LlamaCppProvider",
        ),
        (
            "tensorrt_llm",
            "spoofed-runtime-tensorrt-llm",
            "invarlock_addins.tensorrt_llm.provider:TensorRTLLMProvider",
        ),
        (
            "tensorrt_llm",
            "invarlock-runtime-tensorrt-llm",
            "spoofed.runtime:TensorRTLLMProvider",
        ),
    ],
)
def test_reserved_first_party_addin_names_reject_distribution_or_value_spoofs(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    distribution: str,
    value: str,
) -> None:
    entry_point = EntryPointStub(
        name=name,
        value=value,
        dist=DistStub(distribution, registry_mod.INVARLOCK_VERSION),
    )
    monkeypatch.setattr(registry_mod, "third_party_plugins_allowed", lambda: False)
    monkeypatch.setattr(
        registry_mod,
        "entry_points",
        lambda: SelectEntryPoints(runtime_providers=[entry_point]),
    )

    with pytest.raises(RuntimeError, match="Invalid first-party runtime add-in"):
        registry_mod.CoreRegistry().list_runtime_providers()


def test_first_party_addin_version_must_match_core(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entry_point = EntryPointStub(
        name="llama_cpp",
        value="invarlock_addins.gguf.provider:LlamaCppProvider",
        dist=DistStub("invarlock-runtime-gguf", "0.0.0"),
    )
    monkeypatch.setattr(registry_mod, "third_party_plugins_allowed", lambda: False)
    monkeypatch.setattr(
        registry_mod,
        "entry_points",
        lambda: SelectEntryPoints(runtime_providers=[entry_point]),
    )

    with pytest.raises(RuntimeError, match="at version"):
        registry_mod.CoreRegistry().list_runtime_providers()


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

        def authenticate_artifact(self, spec, artifact_path):  # noqa: ANN001
            return None

        def prepare_execution(self, spec, resources):  # noqa: ANN001
            return resources

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
        required_deps=(),
        available=True,
    )

    assert registry.get_runtime_provider("test_runtime").name == "test_runtime"

    module.INVARLOCK_RUNTIME_PROVIDER_ABI = "9"
    with pytest.raises(ImportError, match="ABI mismatch"):
        registry.get_runtime_provider("test_runtime")

    module.INVARLOCK_RUNTIME_PROVIDER_ABI = "1"
    Provider.name = "different"
    with pytest.raises(ImportError, match="identity mismatch"):
        registry.get_runtime_provider("test_runtime")
