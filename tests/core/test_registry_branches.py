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


def test_registry_fallback_on_entry_points_error(monkeypatch, tmp_path):
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
    assert "hf_gpt2" in adapters
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
    with pytest.raises(KeyError):
        r.get_plugin_metadata("nope", "guards")

    ok, msg = r.validate_configuration(
        "nope_adapter", "nope_edit", ["noop", "nope_guard"]
    )
    assert (
        not ok
        and "Unknown adapter" in msg
        and "Unknown edit" in msg
        and "Unknown guard" in msg
    )


def test_registry_entry_points_select_and_get_paths(monkeypatch):
    # Build stubs that exercise both eps.select(...) and eps.get(...)

    class _Dist:
        def __init__(self, name: str, version: str):
            self.name = name
            self.version = version
            self.metadata = {"Name": name}

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


def test_registry_additional_branches(monkeypatch):
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
    # Inject a dummy plugin that points to a non-Guard class from this test module
    class NotGuard:
        pass

    # Ensure module path for import exists
    module_path = "tests.core.test_registry_branches"
    info = reg.PluginInfo(
        name="not_guard",
        module=module_path,
        class_name="NotGuard",
        available=True,
        status="Available",
        package="invarlock",
        version="0",
        entry_point=None,
    )
    # Bind NotGuard into module globals so importlib can find it
    globals()["NotGuard"] = NotGuard
    r._guards["not_guard"] = info

    with pytest.raises(ImportError):
        r.get_guard("not_guard")

    # Validate configuration success path
    ok, msg = r.validate_configuration("hf_gpt2", "quant_rtn", ["hello_guard"])
    assert ok and msg.endswith("valid")

    # Validate configuration unavailable paths
    # Temporarily mark certain built-ins as unavailable
    r._adapters["hf_gpt2"] = reg.PluginInfo(
        name="hf_gpt2",
        module="invarlock.adapters.hf_gpt2",
        class_name="HF_GPT2_Adapter",
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
    ok, msg = r.validate_configuration("hf_gpt2", "quant_rtn", ["hello_guard"])
    assert not ok
    assert (
        "Adapter unavailable" in msg
        and "Edit unavailable" in msg
        and "Guard unavailable" in msg
    )


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
