import sys
import types
import warnings
from typing import Any

import pytest

import invarlock.core.registry as reg


class _EP:
    """Simple stand-in for importlib.metadata.EntryPoint."""

    def __init__(
        self, name: str, value: str, dist: Any | None = None, loader: Any | None = None
    ):
        self.name = name
        self.value = value
        self.dist = dist
        self._loader = loader

    def load(self):  # pragma: no cover - exercised via get_* calls
        if self._loader is not None:
            return self._loader
        mod, _, attr = self.value.partition(":")
        m = __import__(mod, fromlist=[attr])
        return getattr(m, attr)


class _Dist:
    def __init__(self, name: str, version: str):
        self.name = name
        self.version = version
        self.metadata = {"Name": name}


def _install_plugin_module(
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
    *,
    abi: str,
) -> tuple[type[reg.ModelAdapter], type[reg.ModelEdit], type[reg.Guard]]:
    module = types.ModuleType(module_name)
    module.INVARLOCK_CORE_ABI = abi

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

    DummyAdapter.__module__ = module_name
    DummyEdit.__module__ = module_name
    DummyGuard.__module__ = module_name
    module.DummyAdapter = DummyAdapter
    module.DummyEdit = DummyEdit
    module.DummyGuard = DummyGuard
    monkeypatch.setitem(sys.modules, module_name, module)
    return DummyAdapter, DummyEdit, DummyGuard


def test_registry_fallback_on_entry_points_error(monkeypatch, tmp_path):
    monkeypatch.setenv("INVARLOCK_ALLOW_THIRD_PARTY_PLUGINS", "1")
    # Force entry_points() to error to exercise warning + fallback path
    monkeypatch.setattr(
        reg, "entry_points", lambda: (_ for _ in ()).throw(RuntimeError("boom"))
    )

    captured: list[str] = []
    monkeypatch.setattr(
        warnings, "warn", lambda msg, stacklevel=2: captured.append(str(msg))
    )

    r = reg.CoreRegistry()
    # First call triggers discovery
    adapters = r.list_adapters()
    guards = r.list_guards()
    edits = r.list_edits()

    assert any("Plugin discovery failed" in m for m in captured)
    # Fallback should register a set of built-ins
    assert "hf_causal" in adapters
    assert "hello_guard" in guards
    # Only quant_rtn (and internal noop) remain as core edits
    assert "quant_rtn" in edits

    # Idempotency of lazy init path
    assert set(adapters) == set(r.list_adapters())

    # hello_guard should be loadable without heavy deps
    g = r.get_guard("hello_guard")
    assert g.name == "hello_guard"

    # Unknown plugin info and metadata behavior
    info = r.get_plugin_info("nope", "guards")
    assert info["available"] is False and info["module"] == "unknown"
    assert "Plugin discovery failed" in info["status"]
    with pytest.raises(KeyError):
        r.get_plugin_metadata("nope", "guards")

    ok, msg = r.validate_configuration(
        "nope_adapter", "nope_edit", ["noop", "nope_guard"]
    )
    assert (
        not ok
        and "Plugin discovery failed" in msg
        and "Unknown adapter" in msg
        and "Unknown edit" in msg
        and "Unknown guard" in msg
    )


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
    assert registry.get_plugin_info("hf_causal", "adapters")["status"] == (
        "Available (fallback)"
    )


def test_registry_entry_points_select_and_get_paths(monkeypatch):
    monkeypatch.setenv("INVARLOCK_ALLOW_THIRD_PARTY_PLUGINS", "1")
    # Build stubs that exercise both eps.select(...) and eps.get(...)

    # One entry point that resolves to a valid guard via .load()
    from invarlock.plugins.hello_guard import HelloGuard

    ep_ok = _EP(
        name="ep_hello_guard",
        value="invarlock.plugins.hello_guard:HelloGuard",
        dist=_Dist("invarlock-plugins", "0.0"),
        loader=HelloGuard,
    )

    # One entry point with a non-importable module to mark available=False
    ep_bad = _EP(
        name="ep_missing_mod",
        value="totally_missing.module:Thing",
        dist=_Dist("missing", "0.0"),
        loader=None,
    )

    class _EPContainerSelect:
        def select(self, *, group: str):  # pragma: no cover - covered below
            if group == "invarlock.guards":
                return [ep_ok, ep_bad]
            return []

    class _EPContainerGet(dict):
        pass

    # First, exercise select() code path
    monkeypatch.setattr(reg, "entry_points", lambda: _EPContainerSelect())
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
    eps = _EPContainerGet()
    eps["invarlock.guards"] = [ep_ok]
    eps["invarlock.adapters"] = []
    eps["invarlock.edits"] = []
    monkeypatch.setattr(reg, "entry_points", lambda: eps)

    r2 = reg.CoreRegistry()
    assert "ep_hello_guard" in r2.list_guards()


def test_get_plugin_metadata_adds_name_and_type_for_known_plugin() -> None:
    registry = reg.CoreRegistry()

    metadata = registry.get_plugin_metadata("hello_guard", "guards")

    assert metadata["name"] == "hello_guard"
    assert metadata["type"] == "guards"
    assert metadata["available"] is True
    assert metadata["module"] != "unknown"


def test_get_plugin_info_reports_entry_point_group_for_entry_point_plugins(
    monkeypatch,
) -> None:
    monkeypatch.setenv("INVARLOCK_ALLOW_THIRD_PARTY_PLUGINS", "1")

    from invarlock.plugins.hello_guard import HelloGuard

    ep = _EP(
        name="ep_hello_guard",
        value="invarlock.plugins.hello_guard:HelloGuard",
        dist=_Dist("invarlock-plugins", "0.0"),
        loader=HelloGuard,
    )

    class _EPContainerSelect:
        def select(self, *, group: str):
            if group == "invarlock.guards":
                return [ep]
            return []

    monkeypatch.setattr(reg, "entry_points", lambda: _EPContainerSelect())
    registry = reg.CoreRegistry()

    info = registry.get_plugin_info("ep_hello_guard", "guards")

    assert info["entry_point"] == "ep_hello_guard"
    assert info["entry_point_group"] == "invarlock.guards"


def test_registry_entry_point_collision_and_typed_getters(monkeypatch):
    monkeypatch.setenv("INVARLOCK_ALLOW_THIRD_PARTY_PLUGINS", "1")
    adapter_cls, edit_cls, guard_cls = _install_plugin_module(
        monkeypatch,
        "invarlock_test_registry_entry_points",
        abi=reg.INVARLOCK_CORE_ABI,
    )

    adapter_ep = _EP(
        name="hf_causal",
        value="invarlock_test_registry_entry_points:DummyAdapter",
        dist=_Dist("third-party-adapter", "1.2.3"),
        loader=adapter_cls,
    )
    edit_ep = _EP(
        name="quant_rtn",
        value="invarlock_test_registry_entry_points:DummyEdit",
        dist=_Dist("third-party-edit", "2.3.4"),
        loader=edit_cls,
    )
    guard_ep = _EP(
        name="hello_guard",
        value="invarlock_test_registry_entry_points:DummyGuard",
        dist=_Dist("third-party-guard", "3.4.5"),
        loader=guard_cls,
    )

    class _EPContainerSelect:
        def select(self, *, group: str):
            if group == "invarlock.adapters":
                return [adapter_ep]
            if group == "invarlock.edits":
                return [edit_ep]
            if group == "invarlock.guards":
                return [guard_ep]
            return []

    monkeypatch.setattr(reg, "entry_points", lambda: _EPContainerSelect())
    registry = reg.CoreRegistry()

    assert registry.get_plugin_info("hf_causal", "adapters")["package"] == (
        "third-party-adapter"
    )
    assert registry.get_plugin_info("quant_rtn", "edits")["package"] == (
        "third-party-edit"
    )
    assert registry.get_plugin_info("hello_guard", "guards")["package"] == (
        "third-party-guard"
    )

    adapter = registry.get_adapter_typed("hf_causal")
    edit = registry.get_edit_typed("quant_rtn")
    guard = registry.get_guard("hello_guard")
    assert isinstance(adapter, adapter_cls)
    assert isinstance(edit, edit_cls)
    assert isinstance(guard, guard_cls)


def test_registry_optional_plugin_metadata_tracks_missing_dependencies(
    monkeypatch,
) -> None:
    def _fake_missing(self, deps: list[str]) -> list[str]:
        if deps == ["auto_gptq"]:
            return ["auto_gptq"]
        if deps == ["awq"]:
            return []
        if deps == ["bitsandbytes"]:
            return ["bitsandbytes"]
        return []

    monkeypatch.setattr(reg.CoreRegistry, "_check_runtime_dependencies", _fake_missing)
    registry = reg.CoreRegistry()

    gptq_info = registry.get_plugin_info("hf_gptq", "adapters")
    awq_info = registry.get_plugin_info("hf_awq", "adapters")
    bnb_info = registry.get_plugin_info("hf_bnb", "adapters")

    assert gptq_info["available"] is False
    assert gptq_info["status"] == "Needs extra: auto_gptq"
    assert awq_info["available"] is True
    assert awq_info["status"] == "Available (plugin)"
    assert bnb_info["available"] is False
    assert bnb_info["status"] == "Needs extra: bitsandbytes"


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
    dummy_mod = types.ModuleType(module_name)

    class NotGuard:
        pass

    NotGuard.__module__ = module_name
    dummy_mod.NotGuard = NotGuard
    monkeypatch.setitem(sys.modules, module_name, dummy_mod)

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
    ok, msg = r.validate_configuration("hf_causal", "quant_rtn", ["hello_guard"])
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
        module="invarlock.plugins.hello_guard",
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

    # Entry point with malformed value triggers parse error branch
    bad_ep = _EP(name="bad", value="malformed-without-colon")
    info_bad = r._create_plugin_info(bad_ep, "guards")
    assert info_bad.available is False and "Parse error" in info_bad.status

    # Entry point with dist=None forces package_name from module and metadata_version lookup
    # Simulate PackageNotFoundError from metadata_version
    monkeypatch.setattr(
        reg,
        "metadata_version",
        lambda pkg: (_ for _ in ()).throw(reg.PackageNotFoundError(pkg)),
    )
    ok_ep = _EP(name="ok", value="invarlock.plugins.hello_guard:HelloGuard", dist=None)
    info_ok = r._create_plugin_info(ok_ep, "guards")
    assert info_ok.available is True
    # Package name inferred from module path → top-level package
    assert info_ok.package == "invarlock"


def test_create_plugin_info_uses_dist_name_when_metadata_name_is_missing() -> None:
    r = reg.CoreRegistry()
    ep = _EP(
        name="ok",
        value="invarlock.plugins.hello_guard:HelloGuard",
        dist=types.SimpleNamespace(name="fallback-dist", version="1.2.3", metadata={}),
    )

    info = r._create_plugin_info(ep, "guards")

    assert info.package == "fallback-dist"
    assert info.version == "1.2.3"


def test_check_runtime_dependencies_find_spec_exception_counts_missing(monkeypatch):
    r = reg.CoreRegistry()

    def _boom(dep: str):  # type: ignore[no-untyped-def]
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
    dummy_mod = types.ModuleType(module_name)

    class NotAdapter:
        pass

    class NotEdit:
        pass

    NotAdapter.__module__ = module_name
    NotEdit.__module__ = module_name
    dummy_mod.NotAdapter = NotAdapter
    dummy_mod.NotEdit = NotEdit
    monkeypatch.setitem(sys.modules, module_name, dummy_mod)

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
    bad_adapter_cls, bad_edit_cls, bad_guard_cls = _install_plugin_module(
        monkeypatch,
        "invarlock_test_registry_bad_abi",
        abi="9999",
    )
    registry = reg.CoreRegistry()
    registry._initialized = True

    registry._edits["unavailable_edit"] = reg.PluginInfo(
        name="unavailable_edit",
        module="invarlock.edits.noop",
        class_name="NoopEdit",
        available=False,
        status="disabled",
        entry_point=None,
    )
    registry._guards["unavailable_guard"] = reg.PluginInfo(
        name="unavailable_guard",
        module="invarlock.plugins.hello_guard",
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
        entry_point=_EP(
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
        entry_point=_EP(
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
        entry_point=_EP(
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
