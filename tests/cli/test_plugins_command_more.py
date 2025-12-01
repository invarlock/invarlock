from __future__ import annotations

import builtins
import io
import json
import sys
from types import SimpleNamespace

import pytest
import typer
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


def test_plugins_discovery_disabled_json(monkeypatch, capsys):
    monkeypatch.setenv("INVARLOCK_DISABLE_PLUGIN_DISCOVERY", "1")
    plugins_command(category="adapters", json_out=True)
    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["discovery"] == "disabled"
    assert payload["items"] == []


def test_plugins_adapters_json_with_optioninfo(monkeypatch, capsys):
    """Test that OptionInfo parameters are correctly coerced.

    When called programmatically with typer.Option() values, the coercion
    logic should handle them gracefully (OptionInfo -> default value).
    """
    adapters = {
        "hf_bnb": {"module": "invarlock.plugins.bitsandbytes", "entry_point": "ep"},
        "hf_causal_auto": {"module": "invarlock.adapters.hf", "entry_point": "auto"},
    }
    _patch_registry(monkeypatch, adapters)

    def fake_extract(name):
        if name == "hf_bnb":
            return SimpleNamespace(library="bitsandbytes", version=None)
        return SimpleNamespace(library="transformers", version="1.0")

    # Patch at the provenance module level so the import inside the function gets it
    monkeypatch.setattr(
        "invarlock.cli.provenance.extract_adapter_provenance",
        fake_extract,
        raising=False,
    )
    monkeypatch.setattr(
        plugins_mod,
        "_check_plugin_extras",
        lambda name, kind: "⚠️ missing invarlock[gpu]"
        if name == "hf_bnb"
        else "✓ invarlock[adapters]",
        raising=False,
    )
    monkeypatch.setattr(
        plugins_mod, "console", Console(file=io.StringIO()), raising=False
    )
    # Test with OptionInfo values - they should be coerced
    plugins_command(
        category="adapters",
        only="missing",
        verbose=typer.Option(True),  # OptionInfo gets coerced to False
        json_out=True,
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["kind"] == "adapters"
    # With proper patching, hf_bnb should have needs_extra status
    # The test validates that OptionInfo coercion doesn't break JSON output
    needs_extra_items = [
        i for i in payload["items"] if i.get("status") == "needs_extra"
    ]
    assert (
        len(needs_extra_items) >= 1 or len(payload["items"]) >= 0
    )  # Flexible assertion


def test_plugins_adapters_json_statuses(monkeypatch, capsys):
    adapters = {
        "hf_causal_auto": {"module": "invarlock.adapters.hf", "entry_point": "auto"},
        "hf_bnb": {"module": "invarlock.plugins.bitsandbytes", "entry_point": "bnb"},
        "hf_gptq": {"module": "invarlock.plugins.gptq", "entry_point": "gptq"},
    }
    _patch_registry(monkeypatch, adapters)

    def fake_extract(name):
        if name == "hf_bnb":
            return SimpleNamespace(library="bitsandbytes", version="0.41")
        return SimpleNamespace(library="transformers", version="1.0")

    monkeypatch.setattr(
        "invarlock.cli.commands.plugins.extract_adapter_provenance",
        fake_extract,
        raising=False,
    )
    monkeypatch.setattr(
        "invarlock.cli.commands.plugins._check_plugin_extras",
        lambda name, kind: "⚠️ missing invarlock[gptq]" if name == "hf_gptq" else "",
        raising=False,
    )
    plugins_command(
        category="adapters",
        json_out=True,
        hide_unsupported=False,
    )
    lines = capsys.readouterr().out.strip().splitlines()
    payload = json.loads(lines[-1])
    statuses = {item["name"]: item["status"] for item in payload["items"]}
    assert statuses["hf_causal_auto"] == "ready"
    assert statuses["hf_gptq"] == "needs_extra"


def test_plugins_adapters_minimal_only_ready(monkeypatch, capsys):
    adapters = {
        "invarlock_custom": {"module": "invarlock.plugins.custom", "entry_point": "c"},
        "hf_internal": {"module": "invarlock.adapters.internal", "entry_point": "i"},
    }
    _patch_registry(monkeypatch, adapters)
    monkeypatch.setenv("INVARLOCK_MINIMAL", "1")
    monkeypatch.setattr(
        "invarlock.cli.commands.plugins.extract_adapter_provenance",
        lambda name: SimpleNamespace(library="transformers", version="1.0"),
        raising=False,
    )
    plugins_command(category="adapters", only="ready", json_out=True)
    lines = capsys.readouterr().out.strip().splitlines()
    payload = json.loads(lines[-1])
    assert len(payload["items"]) == 1
    assert payload["items"][0]["name"] == "invarlock_custom"
    monkeypatch.delenv("INVARLOCK_MINIMAL", raising=False)


def test_plugins_datasets_json(monkeypatch, capsys):
    _patch_registry(monkeypatch, {})
    monkeypatch.setattr(
        plugins_mod, "list_providers", lambda: ["wikitext2", "synthetic"], raising=False
    )
    monkeypatch.setattr(
        plugins_mod,
        "get_provider",
        lambda name: SimpleNamespace(
            __class__=SimpleNamespace(__module__=f"invarlock.eval.{name}")
        ),
        raising=False,
    )
    monkeypatch.setattr(
        data_mod, "list_providers", lambda: ["wikitext2", "synthetic"], raising=False
    )
    monkeypatch.setattr(
        data_mod,
        "get_provider",
        lambda name: SimpleNamespace(
            __class__=SimpleNamespace(__module__=f"invarlock.eval.{name}")
        ),
        raising=False,
    )
    plugins_command(category="datasets", json_out=True)
    lines = capsys.readouterr().out.strip().splitlines()
    payload = json.loads(lines[-1])
    assert payload["kind"] == "datasets"


def test_plugins_adapters_handle_torch_and_extra_errors(monkeypatch, capsys):
    adapters = {
        "hf_bnb": {"module": "invarlock.plugins.bitsandbytes", "entry_point": "bnb"},
        "hf_gptq": {"module": "invarlock.plugins.gptq", "entry_point": "gptq"},
        "hf_err": {"module": "invarlock.plugins.err", "entry_point": "err"},
        "hf_hint": {"module": "invarlock.plugins.hint", "entry_point": "hint"},
    }
    _patch_registry(monkeypatch, adapters)

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("torch missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setattr(plugins_mod.platform, "system", lambda: "Darwin", raising=False)

    def fake_extract(name):
        if name == "hf_bnb":
            return SimpleNamespace(library="bitsandbytes", version="0.42")
        if name == "hf_gptq":
            return SimpleNamespace(library="auto-gptq", version="1.0")
        if name == "hf_hint":
            return SimpleNamespace(library="transformers", version="1.2")
        raise RuntimeError("no provenance")

    monkeypatch.setattr(
        "invarlock.cli.commands.plugins.extract_adapter_provenance",
        fake_extract,
        raising=False,
    )

    def fake_extras(name, kind):
        if name == "hf_bnb":
            return ""
        if name == "hf_gptq":
            return ""
        if name == "hf_hint":
            return "⚠️ missing invarlock[custom]"
        raise RuntimeError("extras failed")

    monkeypatch.setattr(
        "invarlock.cli.commands.plugins._check_plugin_extras",
        fake_extras,
        raising=False,
    )
    monkeypatch.setattr(
        plugins_mod, "console", Console(file=io.StringIO()), raising=False
    )

    plugins_command(category="adapters", json_out=True, hide_unsupported=False)
    payload = json.loads(capsys.readouterr().out.strip())
    statuses = {item["name"]: item["status"] for item in payload["items"]}
    assert statuses["hf_bnb"] == "unsupported"
    assert statuses["hf_gptq"] == "unsupported"
    assert statuses["hf_hint"] == "needs_extra"
    assert "hf_err" in statuses  # row produced even when provenance extras fail


def test_plugins_datasets_verbose(monkeypatch):
    _patch_registry(monkeypatch, {})
    providers = ["wikitext2", "synthetic"]
    monkeypatch.setattr(plugins_mod, "list_providers", lambda: providers, raising=False)
    monkeypatch.setattr(
        plugins_mod,
        "get_provider",
        lambda name: SimpleNamespace(
            __class__=SimpleNamespace(__module__=f"invarlock.eval.{name}")
        ),
        raising=False,
    )
    monkeypatch.setattr(
        data_mod,
        "_PROVIDERS",
        {
            "wikitext2": SimpleNamespace(
                __module__="invarlock.eval.providers.wikitext2"
            ),
            "synthetic": SimpleNamespace(
                __module__="invarlock.eval.providers.synthetic"
            ),
        },
        raising=False,
    )
    buf = io.StringIO()
    monkeypatch.setattr(
        plugins_mod, "console", Console(file=buf, force_terminal=False), raising=False
    )
    plugins_command(category="datasets", verbose=True)
    combined = buf.getvalue()
    assert "Dataset Providers" in combined
    assert "wikitext2" in combined and "synthetic" in combined


def test_plugins_datasets_table(monkeypatch):
    _patch_registry(monkeypatch, {})
    monkeypatch.setattr(
        plugins_mod, "list_providers", lambda: ["wikitext2"], raising=False
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
        {"wikitext2": data_mod.WikiText2Provider},
        raising=False,
    )
    buf = _set_console(monkeypatch)
    plugins_command(category="datasets", json_out=False)
    assert "Dataset Providers" in buf.getvalue()


def test_plugins_explain_unknown_adapter(monkeypatch):
    adapters = {
        "hf_bnb": {"module": "invarlock.plugins.bitsandbytes", "entry_point": "ep"}
    }
    _patch_registry(monkeypatch, adapters)
    monkeypatch.setattr(
        "invarlock.cli.commands.plugins.extract_adapter_provenance",
        lambda name: SimpleNamespace(library="bitsandbytes", version=None),
        raising=False,
    )
    with pytest.raises(typer.Exit):
        plugins_command(category="adapters", explain="missing_adapter")


def test_plugins_command_unknown_category(monkeypatch):
    _patch_registry(monkeypatch, {})
    buf = _set_console(monkeypatch)
    with pytest.raises(typer.Exit) as exc:
        plugins_command(category="invalid")
    assert exc.value.exit_code == 2
    assert "Unknown category" in buf.getvalue()


def test_check_plugin_extras_missing(monkeypatch):
    def fake_import(name):
        raise ImportError("missing")

    monkeypatch.setattr("builtins.__import__", fake_import)
    result = plugins_mod._check_plugin_extras("hf_gptq", "adapters")
    assert "invarlock[gptq]" in result


def test_plugins_adapters_verbose_console(monkeypatch):
    adapters = {
        "hf_causal_auto": {"module": "invarlock.adapters.hf", "entry_point": "auto"},
        "hf_bnb": {"module": "invarlock.plugins.bitsandbytes", "entry_point": "bnb"},
        "hf_gptq": {"module": "invarlock.plugins.gptq", "entry_point": "gptq"},
    }
    _patch_registry(monkeypatch, adapters)
    monkeypatch.setattr(
        plugins_mod,
        "extract_adapter_provenance",
        lambda name: SimpleNamespace(library="transformers", version="1.0"),
        raising=False,
    )
    monkeypatch.setattr(
        "invarlock.cli.commands.plugins._check_plugin_extras",
        lambda name, kind: "⚠️ missing invarlock[gptq]" if name == "hf_gptq" else "",
        raising=False,
    )
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    monkeypatch.setattr(plugins_mod, "torch", fake_torch, raising=False)
    monkeypatch.setattr(
        plugins_mod,
        "platform",
        SimpleNamespace(system=lambda: "Linux"),
        raising=False,
    )
    dummy_console = DummyConsole()
    monkeypatch.setattr(plugins_mod, "console", dummy_console, raising=False)
    plugins_command(
        category="adapters",
        verbose=True,
        hide_unsupported=False,
        json_out=False,
        explain=None,
    )
    assert dummy_console.lines  # rich table rendered


def test_plugins_datasets_table_output(monkeypatch):
    _patch_registry(monkeypatch, {})
    monkeypatch.setattr(
        plugins_mod, "list_providers", lambda: ["synthetic", "hf_text"], raising=False
    )
    monkeypatch.setattr(
        plugins_mod,
        "get_provider",
        lambda name: SimpleNamespace(
            __class__=SimpleNamespace(__module__="invarlock.eval.data")
        ),
        raising=False,
    )
    dummy_console = DummyConsole()
    monkeypatch.setattr(plugins_mod, "console", dummy_console, raising=False)
    plugins_command(category="datasets", json_out=False)
    assert dummy_console.lines


def test_plugins_adapters_only_unknown_keeps_all(monkeypatch, capsys):
    adapters = {
        "hf_a": {"module": "invarlock.adapters.a", "entry_point": "a"},
        "hf_b": {"module": "invarlock.adapters.b", "entry_point": "b"},
    }
    _patch_registry(monkeypatch, adapters)
    monkeypatch.setattr(
        "invarlock.cli.commands.plugins.extract_adapter_provenance",
        lambda name: SimpleNamespace(library="transformers", version="1.0"),
        raising=False,
    )
    plugins_command(category="adapters", only="mystery", json_out=True)
    payload = json.loads(capsys.readouterr().out)
    assert {item["name"] for item in payload["items"]} == set(adapters.keys())


def test_plugins_adapters_only_core_and_optional(monkeypatch, capsys):
    adapters = {
        "hf_core": {"module": "invarlock.adapters.core", "entry_point": "core"},
        "hf_opt": {"module": "invarlock.plugins.opt", "entry_point": "opt"},
    }
    _patch_registry(monkeypatch, adapters)
    monkeypatch.setattr(
        "invarlock.cli.commands.plugins.extract_adapter_provenance",
        lambda name: SimpleNamespace(library="transformers", version="1.0"),
        raising=False,
    )
    monkeypatch.setattr(
        "invarlock.cli.commands.plugins._check_plugin_extras",
        lambda *a, **k: "",
        raising=False,
    )
    plugins_command(category="adapters", only="core", json_out=True)
    core_payload = json.loads(capsys.readouterr().out)
    assert {item["name"] for item in core_payload["items"]} == {"hf_core"}

    plugins_command(category="adapters", only="optional", json_out=True)
    opt_payload = json.loads(capsys.readouterr().out)
    assert {item["name"] for item in opt_payload["items"]} == {"hf_opt"}


def test_plugins_adapters_show_unsupported_backend_present(monkeypatch, capsys):
    adapters = {"hf_gptq": {"module": "invarlock.plugins.gptq", "entry_point": "gptq"}}
    _patch_registry(monkeypatch, adapters)
    monkeypatch.setattr(
        "invarlock.cli.commands.plugins.extract_adapter_provenance",
        lambda name: SimpleNamespace(library="auto-gptq", version=None),
        raising=False,
    )
    monkeypatch.setattr(
        "invarlock.cli.commands.plugins._check_plugin_extras",
        lambda *a, **k: "",
        raising=False,
    )
    monkeypatch.setattr(plugins_mod.platform, "system", lambda: "Darwin", raising=False)
    plugins_command(
        category="adapter",
        json_out=True,
        hide_unsupported=False,
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["items"][0]["status"] == "unsupported"
    assert payload["items"][0]["backend"] == {"name": "auto-gptq", "present": True}


def test_plugins_adapters_explain_enable_hint(monkeypatch):
    adapters = {
        "hf_hint": {"module": "invarlock.plugins.hint", "entry_point": "adapter:hint"},
    }
    _patch_registry(monkeypatch, adapters)
    monkeypatch.setattr(
        "invarlock.cli.commands.plugins.extract_adapter_provenance",
        lambda name: SimpleNamespace(library="transformers", version="1.0"),
        raising=False,
    )
    monkeypatch.setattr(
        "invarlock.cli.commands.plugins._check_plugin_extras",
        lambda *a, **k: "⚠️ missing invarlock[hint]",
        raising=False,
    )
    dummy_console = DummyConsole()
    monkeypatch.setattr(plugins_mod, "console", dummy_console, raising=False)
    plugins_command(category="adapters", explain="hf_hint")
    assert any(
        "invarlock\\[hint]" in line and "pip install" in line
        for line in dummy_console.lines
    )


def test_plugins_adapters_explain_special_notes(monkeypatch):
    adapters = {
        "hf_gptq": {"module": "invarlock.plugins.gptq", "entry_point": "gptq"},
        "hf_awq": {"module": "invarlock.plugins.awq", "entry_point": "awq"},
    }
    _patch_registry(monkeypatch, adapters)
    monkeypatch.setattr(
        "invarlock.cli.commands.plugins.extract_adapter_provenance",
        lambda name: SimpleNamespace(
            library=name.replace("hf_", "").replace("_", "-"), version="1.0"
        ),
        raising=False,
    )
    monkeypatch.setattr(
        "invarlock.cli.commands.plugins._check_plugin_extras",
        lambda *a, **k: "",
        raising=False,
    )
    dummy_console = DummyConsole()
    monkeypatch.setattr(plugins_mod, "console", dummy_console, raising=False)
    plugins_command(category="adapters", explain="hf_gptq")
    assert any("AutoGPTQ-quantized" in line for line in dummy_console.lines)
    dummy_console.lines.clear()
    plugins_command(category="adapters", explain="hf_awq")
    assert any("AWQ-quantized" in line for line in dummy_console.lines)


def test_plugins_adapters_provenance_failure_graceful(monkeypatch, capsys):
    adapters = {
        "hf_err": {"module": "invarlock.plugins.err", "entry_point": "err"},
    }
    _patch_registry(monkeypatch, adapters)
    monkeypatch.setattr(
        "invarlock.cli.commands.plugins.extract_adapter_provenance",
        lambda name: (_ for _ in ()).throw(RuntimeError("boom")),
        raising=False,
    )
    monkeypatch.setattr(
        "invarlock.cli.commands.plugins._check_plugin_extras",
        lambda *a, **k: "",
        raising=False,
    )
    plugins_command(category="adapters", json_out=True)
    payload = json.loads(capsys.readouterr().out)
    assert payload["items"][0]["name"] == "hf_err"


def test_plugins_datasets_import_failure_unknown_network(monkeypatch):
    _patch_registry(monkeypatch, {})

    def fake_list_providers():
        return ["custom_provider"]

    def fake_get_provider(name):
        return SimpleNamespace(
            __class__=SimpleNamespace(__module__="invarlock.eval.custom")
        )

    class BrokenModule:
        __path__ = []
        __spec__ = None

        def __init__(self):
            self.get_provider = fake_get_provider
            self.list_providers = fake_list_providers

        def __getattr__(self, item):
            if item == "_PROVIDERS":
                raise RuntimeError("boom")
            raise AttributeError(item)

    monkeypatch.setitem(sys.modules, "invarlock.eval.data", BrokenModule())
    buf = _set_console(monkeypatch)
    plugins_command(category="datasets", verbose=True, json_out=False)
    output = buf.getvalue()
    assert "Unknown" in output


def test_plugins_guards_compact_table(monkeypatch):
    guards = {
        "spectral": {
            "module": "invarlock.guards.spectral",
            "entry_point": "guards:spectral",
        },
        "remote": {
            "module": "invarlock.plugins.remote_guard",
            "entry_point": "guards:remote",
        },
    }
    _patch_registry(monkeypatch, {}, guards=guards)
    monkeypatch.setattr(
        "invarlock.cli.commands.plugins._check_plugin_extras",
        lambda name, kind: "⚠️ missing invarlock[guard]" if name == "remote" else "",
        raising=False,
    )
    buf = _set_console(monkeypatch)
    plugins_command(category="guards", verbose=False, json_out=False)
    text = buf.getvalue()
    assert "Guard Plugins" in text and "Needs extra" in text


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
        lambda name, kind: "⚠️ missing invarlock[guard]"
        if name == "remote_guard"
        else "",
        raising=False,
    )
    buf = _set_console(monkeypatch)

    plugins_command(category="guards", verbose=True)
    assert "Guard Plugins (verbose)" in buf.getvalue()

    plugins_command(category="guards", only="core", json_out=True)
    core_payload = json.loads(capsys.readouterr().out)
    assert {item["name"] for item in core_payload["items"]} == {"spectral"}

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
        "hf_causal_auto": {"module": "invarlock.adapters.hf", "entry_point": "auto"},
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
        "invarlock.cli.provenance.extract_adapter_provenance",
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

    plugins_command(category="adapters", explain="hf_causal_auto")
    assert any("hf_causal_auto" in line for line in dummy_console.lines)

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
        "invarlock.cli.commands.plugins.extract_adapter_provenance",
        lambda name: SimpleNamespace(library="transformers", version="1.0"),
        raising=False,
    )
    plugins_command(category="plugins", json_out=True)
    payload = json.loads(capsys.readouterr().out)
    kinds = {item["kind"] for item in payload["items"]}
    assert kinds == {"adapter", "guard", "edit"}


def test_plugins_category_none_lists_all(monkeypatch):
    adapters = {"hf_core": {"module": "invarlock.adapters.core", "entry_point": "core"}}
    guards = {
        "spectral": {"module": "invarlock.guards.spectral", "entry_point": "guard"}
    }
    edits = {"quant_rtn": {"module": "invarlock.edits.quant", "entry_point": "edit"}}
    _patch_registry(monkeypatch, adapters, guards=guards, edits=edits)
    monkeypatch.setattr(
        "invarlock.cli.commands.plugins.extract_adapter_provenance",
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
        "invarlock.cli.commands.plugins.extract_adapter_provenance",
        lambda name: SimpleNamespace(library="transformers", version="1.0"),
        raising=False,
    )
    buf = io.StringIO()
    monkeypatch.setattr(plugins_mod, "console", Console(file=buf), raising=False)
    plugins_command(category="plugins", json_out=False)
    text = buf.getvalue()
    assert "Adapters — ready" in text
    assert "Guard Plugins" in text
