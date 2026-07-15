import types

import pytest

import invarlock.core.registry as reg
from invarlock.core.builtin_plugin_catalog import builtin_plugin_specs
from tests.core._support_registry import (
    DistStub,
    EntryPointStub,
    MappingEntryPoints,
    SelectEntryPoints,
    install_plain_module,
    install_plugin_module,
)


def test_registry_fails_closed_on_entry_points_error(monkeypatch) -> None:
    monkeypatch.setenv("INVARLOCK_ALLOW_THIRD_PARTY_PLUGINS", "1")
    monkeypatch.setattr(
        reg, "entry_points", lambda: (_ for _ in ()).throw(RuntimeError("boom"))
    )

    with pytest.raises(RuntimeError, match="Plugin discovery failed: boom"):
        reg.CoreRegistry().list_adapters()


def test_check_runtime_dependencies_treats_find_spec_errors_as_missing(
    monkeypatch,
) -> None:
    registry = reg.CoreRegistry()

    def _find_spec(name: str):
        if name == "broken.dep":
            raise RuntimeError("probe failed")
        if name == "present.dep":
            return object()
        return None

    monkeypatch.setattr(reg.importlib.util, "find_spec", _find_spec)

    assert registry._check_runtime_dependencies(
        ["broken.dep", "missing.dep", "present.dep"]
    ) == ["broken.dep", "missing.dep"]


def test_registry_skips_entry_point_lookup_when_third_party_plugins_are_disabled(
    monkeypatch,
) -> None:
    monkeypatch.delenv("INVARLOCK_ALLOW_THIRD_PARTY_PLUGINS", raising=False)
    monkeypatch.setattr(reg, "third_party_plugins_allowed", lambda: False)
    monkeypatch.setattr(
        reg, "entry_points", lambda: (_ for _ in ()).throw(AssertionError("unused"))
    )

    registry = reg.CoreRegistry()

    assert "hf_causal" in registry.list_adapters()
    assert registry.get_plugin_info("hf_causal", "adapters")["status"] == "Built-in"


def test_registry_entry_points_select_and_get_paths(monkeypatch):
    monkeypatch.setenv("INVARLOCK_ALLOW_THIRD_PARTY_PLUGINS", "1")
    # Build stubs that exercise both eps.select(...) and eps.get(...)

    # One entry point that resolves to a valid guard via .load()
    from invarlock.plugins import HelloGuard

    ep_ok = EntryPointStub(
        name="ep_hello_guard",
        value="invarlock.plugins:HelloGuard",
        dist=DistStub("invarlock-plugins", "0.0"),
        loader=HelloGuard,
    )

    # One entry point with a non-importable module to mark available=False
    ep_bad = EntryPointStub(
        name="ep_missing_mod",
        value="totally_missing.module:Thing",
        dist=DistStub("missing", "0.0"),
        loader=None,
    )

    # First, exercise select() code path
    monkeypatch.setattr(
        reg, "entry_points", lambda: SelectEntryPoints(guards=[ep_ok, ep_bad])
    )
    r1 = reg.CoreRegistry()
    names = r1.list_guards()
    assert "ep_hello_guard" in names and "ep_missing_mod" in names

    # Loading via entry_point.load()
    g = r1.get_guard("ep_hello_guard")
    assert isinstance(g, HelloGuard)

    # Unavailable plugin should raise on load
    with pytest.raises(ImportError):
        r1.get_guard("ep_missing_mod")

    # Now, exercise get() mapping code path
    eps = MappingEntryPoints()
    eps["invarlock.guards"] = [ep_ok]
    eps["invarlock.adapters"] = []
    eps["invarlock.edits"] = []
    monkeypatch.setattr(reg, "entry_points", lambda: eps)

    r2 = reg.CoreRegistry()
    assert "ep_hello_guard" in r2.list_guards()


def test_get_plugin_metadata_adds_name_and_type_for_known_plugin() -> None:
    registry = reg.CoreRegistry()

    metadata = registry.get_plugin_metadata("demo_hello_guard", "guards")

    assert metadata["name"] == "demo_hello_guard"
    assert metadata["type"] == "guards"
    assert metadata["available"] is True
    assert metadata["module"] != "unknown"


def test_get_plugin_info_reports_entry_point_group_for_entry_point_plugins(
    monkeypatch,
) -> None:
    monkeypatch.setenv("INVARLOCK_ALLOW_THIRD_PARTY_PLUGINS", "1")

    from invarlock.plugins import HelloGuard

    ep = EntryPointStub(
        name="ep_hello_guard",
        value="invarlock.plugins:HelloGuard",
        dist=DistStub("invarlock-plugins", "0.0"),
        loader=HelloGuard,
    )

    monkeypatch.setattr(reg, "entry_points", lambda: SelectEntryPoints(guards=[ep]))
    registry = reg.CoreRegistry()

    info = registry.get_plugin_info("ep_hello_guard", "guards")

    assert info["entry_point"] == "ep_hello_guard"
    assert info["entry_point_group"] == "invarlock.guards"


def test_registry_rejects_entry_point_name_collisions(monkeypatch) -> None:
    monkeypatch.setenv("INVARLOCK_ALLOW_THIRD_PARTY_PLUGINS", "1")
    adapter_cls, edit_cls, guard_cls = install_plugin_module(
        monkeypatch,
        "invarlock_test_registry_entry_points",
        abi=reg.INVARLOCK_CORE_ABI,
    )

    adapter_ep = EntryPointStub(
        name="hf_causal",
        value="invarlock_test_registry_entry_points:DummyAdapter",
        dist=DistStub("third-party-adapter", "1.2.3"),
        loader=adapter_cls,
    )
    edit_ep = EntryPointStub(
        name="quant_rtn",
        value="invarlock_test_registry_entry_points:DummyEdit",
        dist=DistStub("third-party-edit", "2.3.4"),
        loader=edit_cls,
    )
    guard_ep = EntryPointStub(
        name="hello_guard",
        value="invarlock_test_registry_entry_points:DummyGuard",
        dist=DistStub("third-party-guard", "3.4.5"),
        loader=guard_cls,
    )

    monkeypatch.setattr(
        reg,
        "entry_points",
        lambda: SelectEntryPoints(
            adapters=[adapter_ep],
            edits=[edit_ep],
            guards=[guard_ep],
        ),
    )
    registry = reg.CoreRegistry()

    with pytest.raises(RuntimeError, match="Duplicate adapter plugin name: hf_causal"):
        registry.list_adapters()


def test_registry_ignores_identical_entry_points_from_its_own_distribution(
    monkeypatch,
) -> None:
    monkeypatch.setenv("INVARLOCK_ALLOW_THIRD_PARTY_PLUGINS", "1")
    groups = {
        plugin_type: [
            EntryPointStub(
                name=spec.name,
                value=f"{spec.module}:{spec.class_name}",
                dist=DistStub("InvarLock", reg.INVARLOCK_VERSION),
            )
            for spec in builtin_plugin_specs(plugin_type)
        ]
        for plugin_type in ("adapters", "edits", "guards", "runtime_providers")
    }
    monkeypatch.setattr(reg, "entry_points", lambda: SelectEntryPoints(**groups))

    registry = reg.CoreRegistry()

    assert set(registry.list_adapters()) == {
        spec.name for spec in builtin_plugin_specs("adapters")
    }
    assert set(registry.list_edits()) == {
        spec.name for spec in builtin_plugin_specs("edits")
    }
    assert set(registry.list_guards()) == {
        spec.name for spec in builtin_plugin_specs("guards")
    }
    assert set(registry.list_runtime_providers()) == {
        spec.name for spec in builtin_plugin_specs("runtime_providers")
    }
    assert registry.get_plugin_info("hf_auto", "adapters")["status"] == "Built-in"


@pytest.mark.parametrize(
    ("distribution", "value"),
    [
        ("third-party-adapter", "invarlock.adapters.auto:HF_Auto_Adapter"),
        ("invarlock", "invarlock.adapters.auto:DifferentAdapter"),
    ],
)
def test_registry_rejects_nonidentical_collisions_with_builtin_entry_points(
    monkeypatch, distribution: str, value: str
) -> None:
    monkeypatch.setenv("INVARLOCK_ALLOW_THIRD_PARTY_PLUGINS", "1")
    entry_point = EntryPointStub(
        name="hf_auto",
        value=value,
        dist=DistStub(distribution, "1.0"),
    )
    monkeypatch.setattr(
        reg,
        "entry_points",
        lambda: SelectEntryPoints(adapters=[entry_point]),
    )

    with pytest.raises(RuntimeError, match="Duplicate adapter plugin name: hf_auto"):
        reg.CoreRegistry().list_adapters()


def test_registry_entry_points_with_distinct_names_support_typed_getters(
    monkeypatch,
) -> None:
    monkeypatch.setenv("INVARLOCK_ALLOW_THIRD_PARTY_PLUGINS", "1")
    adapter_cls, edit_cls, guard_cls = install_plugin_module(
        monkeypatch,
        "invarlock_test_registry_distinct_entry_points",
        abi=reg.INVARLOCK_CORE_ABI,
    )

    adapter_ep = EntryPointStub(
        name="custom_adapter",
        value="invarlock_test_registry_distinct_entry_points:DummyAdapter",
        dist=DistStub("third-party-adapter", "1.2.3"),
        loader=adapter_cls,
    )
    edit_ep = EntryPointStub(
        name="custom_edit",
        value="invarlock_test_registry_distinct_entry_points:DummyEdit",
        dist=DistStub("third-party-edit", "2.3.4"),
        loader=edit_cls,
    )
    guard_ep = EntryPointStub(
        name="custom_guard",
        value="invarlock_test_registry_distinct_entry_points:DummyGuard",
        dist=DistStub("third-party-guard", "3.4.5"),
        loader=guard_cls,
    )

    monkeypatch.setattr(
        reg,
        "entry_points",
        lambda: SelectEntryPoints(
            adapters=[adapter_ep],
            edits=[edit_ep],
            guards=[guard_ep],
        ),
    )
    registry = reg.CoreRegistry()

    assert registry.get_plugin_info("custom_adapter", "adapters")["package"] == (
        "third-party-adapter"
    )
    assert registry.get_plugin_info("custom_edit", "edits")["package"] == (
        "third-party-edit"
    )
    assert registry.get_plugin_info("custom_guard", "guards")["package"] == (
        "third-party-guard"
    )

    adapter = registry.get_adapter_typed("custom_adapter")
    edit = registry.get_edit_typed("custom_edit")
    guard = registry.get_guard("custom_guard")
    assert isinstance(adapter, adapter_cls)
    assert isinstance(edit, edit_cls)
    assert isinstance(guard, guard_cls)


def test_registry_optional_plugin_metadata_tracks_missing_dependencies(
    monkeypatch,
) -> None:
    def _fake_missing(self, deps: list[str]) -> list[str]:
        if deps == ["gptqmodel"]:
            return ["gptqmodel"]
        if deps == ["bitsandbytes"]:
            return ["bitsandbytes"]
        if deps == ["torchao"]:
            return ["torchao"]
        if deps == ["hqq"]:
            return ["hqq"]
        if deps == ["optimum.quanto"]:
            return ["optimum.quanto"]
        if deps == ["compressed_tensors"]:
            return ["compressed_tensors"]
        return []

    monkeypatch.setattr(reg.CoreRegistry, "_check_runtime_dependencies", _fake_missing)
    registry = reg.CoreRegistry()

    gptq_info = registry.get_plugin_info("hf_gptq", "adapters")
    awq_info = registry.get_plugin_info("hf_awq", "adapters")
    bnb_info = registry.get_plugin_info("hf_bnb", "adapters")
    torchao_info = registry.get_plugin_info("hf_torchao", "adapters")
    hqq_info = registry.get_plugin_info("hf_hqq", "adapters")
    quanto_info = registry.get_plugin_info("hf_quanto", "adapters")
    ct_info = registry.get_plugin_info("hf_ct", "adapters")

    assert gptq_info["available"] is False
    assert gptq_info["status"] == "Needs extra: gptqmodel"
    assert awq_info["available"] is False
    assert awq_info["status"] == "Needs extra: gptqmodel"
    assert bnb_info["available"] is False
    assert bnb_info["status"] == "Needs extra: bitsandbytes"
    assert torchao_info["available"] is False
    assert torchao_info["status"] == "Needs extra: torchao"
    assert hqq_info["available"] is False
    assert hqq_info["status"] == "Needs extra: hqq"
    assert quanto_info["available"] is False
    assert quanto_info["status"] == "Needs extra: optimum.quanto"
    assert ct_info["available"] is False
    assert ct_info["status"] == "Needs extra: compressed_tensors"


def test_registry_optional_plugin_metadata_tracks_available_dependencies(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        reg.CoreRegistry,
        "_check_runtime_dependencies",
        lambda self, deps: [],
    )

    registry = reg.CoreRegistry()

    gptq_info = registry.get_plugin_info("hf_gptq", "adapters")
    awq_info = registry.get_plugin_info("hf_awq", "adapters")
    bnb_info = registry.get_plugin_info("hf_bnb", "adapters")
    torchao_info = registry.get_plugin_info("hf_torchao", "adapters")
    hqq_info = registry.get_plugin_info("hf_hqq", "adapters")
    quanto_info = registry.get_plugin_info("hf_quanto", "adapters")
    ct_info = registry.get_plugin_info("hf_ct", "adapters")

    assert gptq_info["available"] is True
    assert gptq_info["status"] == "Built-in"
    assert awq_info["available"] is True
    assert awq_info["status"] == "Built-in"
    assert bnb_info["available"] is True
    assert bnb_info["status"] == "Built-in"
    assert torchao_info["available"] is True
    assert torchao_info["status"] == "Built-in"
    assert hqq_info["available"] is True
    assert hqq_info["status"] == "Built-in"
    assert quanto_info["available"] is True
    assert quanto_info["status"] == "Built-in"
    assert ct_info["available"] is True
    assert ct_info["status"] == "Built-in"


def test_registry_additional_paths(monkeypatch):
    r = reg.CoreRegistry()

    # Unknown adapter/edit/guard key errors paths
    with pytest.raises(KeyError):
        r.get_adapter("__nope__")
    with pytest.raises(KeyError):
        r.get_edit("__nope__")
    with pytest.raises(KeyError):
        r.get_guard("__nope__")

    # Unknown plugin type in get_plugin_info
    with pytest.raises(ValueError):
        r.get_plugin_info("hello_guard", "widgets")

    # Guard fallback import path with type-mismatch (not a Guard instance)
    # Use a synthetic module so importlib.import_module can resolve it even though
    # tests/ is not a package.
    module_name = "invarlock_test_registry_type_mismatch"

    class NotGuard:
        pass

    install_plain_module(monkeypatch, module_name, NotGuard=NotGuard)

    r._guards["not_guard"] = reg.PluginInfo(
        name="not_guard",
        module=module_name,
        class_name="NotGuard",
        available=True,
        status="Available",
        package="invarlock",
        version="0",
        entry_point=None,
    )

    with pytest.raises(ImportError):
        r.get_guard("not_guard")

    # Validate configuration success path
    ok, msg = r.validate_configuration("hf_causal", "quant_rtn", ["demo_hello_guard"])
    assert ok and msg.endswith("valid")

    # Validate configuration unavailable paths
    # Temporarily mark certain built-ins as unavailable
    r._adapters["hf_causal"] = reg.PluginInfo(
        name="hf_causal",
        module="invarlock.adapters.hf_causal",
        class_name="HF_Causal_Adapter",
        available=False,
        status="disabled",
    )
    r._edits["quant_rtn"] = reg.PluginInfo(
        name="quant_rtn",
        module="invarlock.edits.quant_rtn",
        class_name="RTNQuantEdit",
        available=False,
        status="disabled",
    )
    r._guards["hello_guard"] = reg.PluginInfo(
        name="hello_guard",
        module="invarlock.plugins",
        class_name="HelloGuard",
        available=False,
        status="disabled",
    )
    ok, msg = r.validate_configuration("hf_causal", "quant_rtn", ["hello_guard"])
    assert not ok
    assert (
        "Adapter unavailable" in msg
        and "Edit unavailable" in msg
        and "Guard unavailable" in msg
    )


def test_registry_typed_wrappers_raise_dependency_error_for_adapter_and_edit_import_failures() -> (
    None
):
    registry = reg.CoreRegistry()
    registry._initialized = True
    registry._adapters["missing_adapter"] = reg.PluginInfo(
        name="missing_adapter",
        module="invarlock_test_registry_missing_adapter_module",
        class_name="MissingAdapter",
        available=True,
        status="Available",
        entry_point=None,
    )
    registry._edits["missing_edit"] = reg.PluginInfo(
        name="missing_edit",
        module="invarlock_test_registry_missing_edit_module",
        class_name="MissingEdit",
        available=True,
        status="Available",
        entry_point=None,
    )

    with pytest.raises(reg.DependencyError) as adapter_exc:
        registry.get_adapter_typed("missing_adapter")
    with pytest.raises(reg.DependencyError) as edit_exc:
        registry.get_edit_typed("missing_edit")

    assert adapter_exc.value.code == "E702"
    assert adapter_exc.value.details == {
        "name": "missing_adapter",
        "kind": "adapter",
        "reason": "ImportError",
    }
    assert edit_exc.value.code == "E702"
    assert edit_exc.value.details == {
        "name": "missing_edit",
        "kind": "edit",
        "reason": "ImportError",
    }


def test_get_registry_returns_global_singleton() -> None:
    assert reg.get_registry() is reg.get_registry()


def test_create_plugin_info_parse_and_metadata_paths(monkeypatch):
    r = reg.CoreRegistry()

    bad_ep = EntryPointStub(name="bad", value="malformed-without-colon")
    with pytest.raises(ValueError, match="malformed entry point value"):
        r._create_plugin_info(bad_ep, "guards")

    # Entry point with dist=None forces package_name from module and metadata_version lookup
    # Simulate PackageNotFoundError from metadata_version
    monkeypatch.setattr(
        reg,
        "metadata_version",
        lambda pkg: (_ for _ in ()).throw(reg.PackageNotFoundError(pkg)),
    )
    ok_ep = EntryPointStub(name="ok", value="invarlock.plugins:HelloGuard", dist=None)
    info_ok = r._create_plugin_info(ok_ep, "guards")
    assert info_ok.available is True
    assert info_ok.status == "Deferred load"
    # Package name inferred from module path → top-level package
    assert info_ok.package == "invarlock"


def test_create_plugin_info_uses_dist_name_when_metadata_name_is_missing() -> None:
    r = reg.CoreRegistry()
    ep = EntryPointStub(
        name="ok",
        value="invarlock.plugins:HelloGuard",
        dist=types.SimpleNamespace(name="fallback-dist", version="1.2.3", metadata={}),
    )

    info = r._create_plugin_info(ep, "guards")

    assert info.package == "fallback-dist"
    assert info.version == "1.2.3"


def test_create_plugin_info_rejects_non_string_values_and_uses_metadata_version(
    monkeypatch,
) -> None:
    r = reg.CoreRegistry()
    bad_ep = EntryPointStub(name="bad", value="ignored")
    bad_ep.value = 123

    with pytest.raises(TypeError, match="entry point value must be a string"):
        r._create_plugin_info(bad_ep, "guards")

    monkeypatch.setattr(reg, "metadata_version", lambda pkg: f"{pkg}-version")
    ep = EntryPointStub(
        name="ok",
        value="thirdparty.guard:Guard",
        dist=types.SimpleNamespace(name=None, version=None, metadata="not-a-dict"),
    )

    info = r._create_plugin_info(ep, "guards")

    assert info.available is True
    assert info.status == "Deferred load"
    assert info.package == "thirdparty"
    assert info.version == "thirdparty-version"


def test_check_runtime_dependencies_find_spec_exception_counts_missing(monkeypatch):
    r = reg.CoreRegistry()

    def _boom(dep: str):
        raise RuntimeError("boom")

    monkeypatch.setattr(reg.importlib.util, "find_spec", _boom)
    assert r._check_runtime_dependencies(["some_dep"]) == ["some_dep"]


def test_registry_get_paths_cover_unavailable_and_type_mismatch_paths(monkeypatch):
    r = reg.CoreRegistry()
    r._initialized = True

    r._adapters["unavailable_adapter"] = reg.PluginInfo(
        name="unavailable_adapter",
        module="invarlock.adapters",
        class_name="HF_Causal_Adapter",
        available=False,
        status="disabled",
        entry_point=None,
    )
    with pytest.raises(ImportError, match="unavailable"):
        r.get_adapter("unavailable_adapter")

    module_name = "invarlock_test_registry_type_mismatch_more"

    class NotAdapter:
        pass

    class NotEdit:
        pass

    install_plain_module(
        monkeypatch,
        module_name,
        NotAdapter=NotAdapter,
        NotEdit=NotEdit,
    )

    r._adapters["bad_adapter"] = reg.PluginInfo(
        name="bad_adapter",
        module=module_name,
        class_name="NotAdapter",
        available=True,
        status="Available",
        entry_point=None,
    )
    with pytest.raises(ImportError):
        r.get_adapter("bad_adapter")

    r._edits["bad_edit"] = reg.PluginInfo(
        name="bad_edit",
        module=module_name,
        class_name="NotEdit",
        available=True,
        status="Available",
        entry_point=None,
    )
    with pytest.raises(ImportError):
        r.get_edit("bad_edit")


def test_registry_unavailable_and_abi_mismatch_paths(monkeypatch):
    bad_adapter_cls, bad_edit_cls, bad_guard_cls = install_plugin_module(
        monkeypatch,
        "invarlock_test_registry_bad_abi",
        abi="9999",
    )
    registry = reg.CoreRegistry()
    registry._initialized = True

    registry._edits["unavailable_edit"] = reg.PluginInfo(
        name="unavailable_edit",
        module="invarlock.edits",
        class_name="NoopEdit",
        available=False,
        status="disabled",
        entry_point=None,
    )
    registry._guards["unavailable_guard"] = reg.PluginInfo(
        name="unavailable_guard",
        module="invarlock.plugins",
        class_name="HelloGuard",
        available=False,
        status="disabled",
        entry_point=None,
    )

    with pytest.raises(ImportError, match="unavailable"):
        registry.get_edit("unavailable_edit")
    with pytest.raises(ImportError, match="unavailable"):
        registry.get_guard("unavailable_guard")

    registry._adapters["bad_abi_adapter"] = reg.PluginInfo(
        name="bad_abi_adapter",
        module="invarlock_test_registry_bad_abi",
        class_name="DummyAdapter",
        available=True,
        status="Available",
        entry_point=EntryPointStub(
            "bad_abi_adapter",
            "invarlock_test_registry_bad_abi:DummyAdapter",
            loader=bad_adapter_cls,
        ),
    )
    registry._edits["bad_abi_edit"] = reg.PluginInfo(
        name="bad_abi_edit",
        module="invarlock_test_registry_bad_abi",
        class_name="DummyEdit",
        available=True,
        status="Available",
        entry_point=EntryPointStub(
            "bad_abi_edit",
            "invarlock_test_registry_bad_abi:DummyEdit",
            loader=bad_edit_cls,
        ),
    )
    registry._guards["bad_abi_guard"] = reg.PluginInfo(
        name="bad_abi_guard",
        module="invarlock_test_registry_bad_abi",
        class_name="DummyGuard",
        available=True,
        status="Available",
        entry_point=EntryPointStub(
            "bad_abi_guard",
            "invarlock_test_registry_bad_abi:DummyGuard",
            loader=bad_guard_cls,
        ),
    )

    with pytest.raises(ImportError, match="ABI mismatch"):
        registry.get_adapter("bad_abi_adapter")
    with pytest.raises(ImportError, match="ABI mismatch"):
        registry.get_edit("bad_abi_edit")
    with pytest.raises(ImportError, match="ABI mismatch"):
        registry.get_guard("bad_abi_guard")


def test_registry_duplicate_builtin_and_coverage_helpers() -> None:
    registry = reg.CoreRegistry()
    registry._adapters["hf_causal"] = reg.PluginInfo(
        name="hf_causal",
        module="invarlock.adapters.hf_causal",
        class_name="HF_Causal_Adapter",
        available=True,
        status="Built-in",
        entry_point=None,
    )

    with pytest.raises(RuntimeError, match="Duplicate built-in plugin registration"):
        registry._register_builtin_plugins()

    assert "quant_rtn" in reg.CoreRegistry().list_edits()


def test_builtin_plugin_catalog_is_table_driven_and_complete() -> None:
    adapter_names = {spec.name for spec in builtin_plugin_specs("adapters")}
    edit_names = {spec.name for spec in builtin_plugin_specs("edits")}
    guard_names = {spec.name for spec in builtin_plugin_specs("guards")}

    assert {"hf_causal", "hf_auto", "hf_multimodal"} <= adapter_names
    assert {"quant_rtn", "noop"} <= edit_names
    assert {"invariants", "spectral", "variance", "rmt"} <= guard_names


def test_registry_plugin_info_and_validation_error_paths(monkeypatch) -> None:
    registry = reg.CoreRegistry()
    registry._initialized = True

    class _NotAdapter:
        pass

    module_name = "invarlock_test_registry_not_adapter"
    install_plain_module(
        monkeypatch,
        module_name,
        INVARLOCK_CORE_ABI=reg.INVARLOCK_CORE_ABI,
        NotAdapter=_NotAdapter,
    )

    registry._adapters["bad_adapter"] = reg.PluginInfo(
        name="bad_adapter",
        module=module_name,
        class_name="NotAdapter",
        available=True,
        status="Available",
        entry_point=None,
    )

    with pytest.raises(ImportError, match="Expected ModelAdapter"):
        registry.get_adapter("bad_adapter")

    assert registry.get_plugin_info("missing_guard", "guards") == {
        "available": False,
        "status": "Not found",
        "module": "unknown",
    }

    with pytest.raises(KeyError, match="Unknown guard plugin 'missing_guard'"):
        registry.get_plugin_metadata("missing_guard", "guards")

    with pytest.raises(reg.PluginError, match="PLUGIN-LOAD-FAILED"):
        registry.get_guard_typed("missing_guard")


def test_registry_validate_configuration_covers_unknown_and_noop_paths() -> None:
    registry = reg.CoreRegistry()

    ok, message = registry.validate_configuration(
        "missing_adapter",
        "missing_edit",
        ["noop", "missing_guard"],
    )

    assert ok is False
    assert "Unknown adapter: missing_adapter" in message
    assert "Unknown edit: missing_edit" in message
    assert "Unknown guard: missing_guard" in message
