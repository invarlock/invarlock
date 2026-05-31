from __future__ import annotations

import io
import json
from types import SimpleNamespace

from rich.console import Console

import invarlock.cli.commands.plugins as plugins_mod
from invarlock.cli.commands.plugins import plugins_command
from invarlock.eval import data as data_mod


class _FakeRegistry:
    def __init__(
        self,
        adapters: dict[str, dict[str, str]] | None = None,
        guards: dict[str, dict[str, str]] | None = None,
        edits: dict[str, dict[str, str]] | None = None,
    ):
        self._adapters = adapters or {}
        self._guards = guards or {}
        self._edits = edits or {}

    def list_adapters(self):
        return list(self._adapters.keys())

    def get_plugin_info(self, name, kind):
        mapping = {
            "adapters": self._adapters,
            "guards": self._guards,
            "edits": self._edits,
        }
        if kind not in mapping or name not in mapping[kind]:
            raise KeyError(f"{kind}:{name}")
        return mapping[kind][name]

    def list_guards(self):
        return list(self._guards.keys())

    def list_edits(self):
        return list(self._edits.keys())


def _patch_registry(monkeypatch, adapters, *, guards=None, edits=None):
    fake = _FakeRegistry(adapters, guards=guards, edits=edits)
    monkeypatch.setattr(plugins_mod, "get_registry", lambda: fake, raising=False)
    monkeypatch.setattr(
        "invarlock.core.registry.get_registry",
        lambda: fake,
        raising=False,
    )
    monkeypatch.setattr(
        "invarlock.core.registry._global_registry",
        fake,
        raising=False,
    )


def _set_console(monkeypatch):
    buf = io.StringIO()
    monkeypatch.setattr(plugins_mod, "console", Console(file=buf), raising=False)
    return buf


class DummyConsole:
    def __init__(self):
        self.lines: list[str] = []

    def print(self, *args, **kwargs):
        text = " ".join(str(arg) for arg in args)
        self.lines.append(text)


def test_plugins_guards_empty_message(monkeypatch):
    _patch_registry(monkeypatch, {})
    buf = _set_console(monkeypatch)
    plugins_command(category="guards")
    assert "No guard plugins" in buf.getvalue()


def test_plugins_guards_verbose_json_and_explain(monkeypatch, capsys):
    guards = {
        "spectral": {
            "module": "invarlock.guards.spectral",
            "entry_point": "guards:spectral",
        },
        "remote_guard": {
            "module": "invarlock.plugins.guard_remote",
            "entry_point": "guards:remote",
        },
    }
    _patch_registry(monkeypatch, {}, guards=guards)
    monkeypatch.setattr(
        "invarlock.cli.commands.plugins._check_plugin_extras",
        lambda name, kind: (
            "⚠️ missing invarlock[guard]" if name == "remote_guard" else ""
        ),
        raising=False,
    )
    buf = _set_console(monkeypatch)

    plugins_command(category="guards", verbose=True)
    assert "Guard Plugins (verbose)" in buf.getvalue()

    plugins_command(category="guards", only="core", json_out=True)
    core_payload = json.loads(capsys.readouterr().out)
    assert {item["name"] for item in core_payload["items"]} == {"spectral"}
    spectral_item = next(
        item for item in core_payload["items"] if item["name"] == "spectral"
    )
    assert spectral_item["support_tier"] == "core_supported"

    plugins_command(category="guards", only="optional", json_out=True)
    optional_payload = json.loads(capsys.readouterr().out)
    assert {item["name"] for item in optional_payload["items"]} == {"remote_guard"}

    dummy_console = DummyConsole()
    monkeypatch.setattr(plugins_mod, "console", dummy_console, raising=False)
    plugins_command(category="guards", explain="remote_guard")
    assert any("Enable" in line for line in dummy_console.lines)


def test_plugins_adapters_explain_variants(monkeypatch):
    """Test explain output for different adapter types."""
    adapters = {
        "hf_auto": {"module": "invarlock.adapters.hf", "entry_point": "auto"},
        "hf_core": {"module": "invarlock.adapters.core", "entry_point": "core"},
        "hf_bnb": {"module": "invarlock.plugins.bitsandbytes", "entry_point": "bnb"},
    }
    _patch_registry(monkeypatch, adapters)

    def fake_extract(name):
        if name == "hf_bnb":
            return SimpleNamespace(library="bitsandbytes", version=None)
        return SimpleNamespace(library="transformers", version="1.0")

    # Patch at the provenance module level so the import inside the function gets it
    monkeypatch.setattr(
        "invarlock.core.backend_inventory.extract_adapter_provenance",
        fake_extract,
        raising=False,
    )
    monkeypatch.setattr(
        plugins_mod,
        "_check_plugin_extras",
        lambda name, kind: "⚠️ missing invarlock[gpu]" if name == "hf_bnb" else "",
        raising=False,
    )
    dummy_console = DummyConsole()
    monkeypatch.setattr(plugins_mod, "console", dummy_console, raising=False)

    plugins_command(category="adapters", explain="hf_auto")
    assert any("hf_auto" in line for line in dummy_console.lines)

    dummy_console.lines.clear()
    plugins_command(category="adapters", explain="hf_core")
    assert any("hf_core" in line for line in dummy_console.lines)

    dummy_console.lines.clear()
    plugins_command(category="adapters", explain="hf_bnb")
    # The explain output shows adapter details; with needs_extra status it may show
    # Enable or Status info depending on the enable field being populated
    assert any("hf_bnb" in line or "Status" in line for line in dummy_console.lines)


def test_plugins_plugins_category_json(monkeypatch, capsys):
    adapters = {
        "hf_core": {"module": "invarlock.adapters.core", "entry_point": "core"},
    }
    guards = {
        "spectral": {"module": "invarlock.guards.spectral", "entry_point": "guard"}
    }
    edits = {"quant_rtn": {"module": "invarlock.edits.quant", "entry_point": "edit"}}
    _patch_registry(monkeypatch, adapters, guards=guards, edits=edits)
    monkeypatch.setattr(
        "invarlock.core.backend_inventory.extract_adapter_provenance",
        lambda name: SimpleNamespace(library="transformers", version="1.0"),
        raising=False,
    )
    plugins_command(category="plugins", json_out=True)
    payload = json.loads(capsys.readouterr().out)
    kinds = {item["kind"] for item in payload["items"]}
    assert kinds == {"adapter", "guard", "edit"}
    quant_item = next(item for item in payload["items"] if item["name"] == "quant_rtn")
    assert quant_item["support_tier"] == "validation_simulation"


def test_plugins_category_none_lists_all(monkeypatch):
    adapters = {"hf_core": {"module": "invarlock.adapters.core", "entry_point": "core"}}
    guards = {
        "spectral": {"module": "invarlock.guards.spectral", "entry_point": "guard"}
    }
    edits = {"quant_rtn": {"module": "invarlock.edits.quant", "entry_point": "edit"}}
    _patch_registry(monkeypatch, adapters, guards=guards, edits=edits)
    monkeypatch.setattr(
        "invarlock.core.backend_inventory.extract_adapter_provenance",
        lambda name: SimpleNamespace(library="transformers", version="1.0"),
        raising=False,
    )
    monkeypatch.setattr(
        plugins_mod,
        "list_providers",
        lambda: ["wikitext2", "synthetic"],
        raising=False,
    )
    monkeypatch.setattr(
        plugins_mod,
        "get_provider",
        lambda name: SimpleNamespace(
            __class__=SimpleNamespace(__module__="invarlock.eval.data")
        ),
        raising=False,
    )
    monkeypatch.setattr(
        data_mod,
        "_PROVIDERS",
        {
            "wikitext2": data_mod.WikiText2Provider,
            "synthetic": data_mod.SyntheticProvider,
        },
        raising=False,
    )
    buf = _set_console(monkeypatch)
    plugins_command(category=None, verbose=True)
    text = buf.getvalue()
    assert "Guard Plugins" in text
    assert "Dataset Providers" in text


def test_plugins_datasets_verbose_table(monkeypatch):
    _patch_registry(monkeypatch, {})
    monkeypatch.setattr(
        plugins_mod, "list_providers", lambda: ["wikitext2", "hf_text"], raising=False
    )
    monkeypatch.setattr(
        plugins_mod,
        "get_provider",
        lambda name: SimpleNamespace(
            __class__=SimpleNamespace(__module__="invarlock.eval.data")
        ),
        raising=False,
    )
    monkeypatch.setattr(
        data_mod,
        "_PROVIDERS",
        {
            "wikitext2": data_mod.WikiText2Provider,
            "hf_text": data_mod.HFTextProvider,
        },
        raising=False,
    )
    buf = _set_console(monkeypatch)
    plugins_command(category="datasets", json_out=False, verbose=True)
    output = buf.getvalue()
    assert "Module" in output


def test_plugins_plugins_category_tables(monkeypatch):
    adapters = {"hf_core": {"module": "invarlock.adapters.core", "entry_point": "core"}}
    guards = {
        "spectral": {"module": "invarlock.guards.spectral", "entry_point": "guard"}
    }
    edits = {"quant_rtn": {"module": "invarlock.edits.quant", "entry_point": "edit"}}
    _patch_registry(monkeypatch, adapters, guards=guards, edits=edits)
    monkeypatch.setattr(
        "invarlock.core.backend_inventory.extract_adapter_provenance",
        lambda name: SimpleNamespace(library="transformers", version="1.0"),
        raising=False,
    )
    buf = io.StringIO()
    monkeypatch.setattr(plugins_mod, "console", Console(file=buf), raising=False)
    plugins_command(category="plugins", json_out=False)
    text = buf.getvalue()
    assert "Adapters — ready" in text
    assert "Guard Plugins" in text
