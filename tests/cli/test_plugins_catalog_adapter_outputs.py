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
    monkeypatch.setenv("INVARLOCK_ALLOW_THIRD_PARTY_PLUGINS", "0")
    plugins_command(category="adapters", json_out=True)
    payload = json.loads(capsys.readouterr().out.strip())
    assert any(item["name"] == "hf_causal" for item in payload["items"])
    assert payload["contracts"]["model_family_catalog"]["format_version"] == (
        "model-family-catalog-v1"
    )
    assert payload["model_family_catalog"]["format_version"] == (
        "model-family-catalog-v1"
    )


def test_plugins_adapters_json_with_explicit_filters(monkeypatch, capsys):
    adapters = {
        "hf_bnb": {"module": "invarlock.plugins.bitsandbytes", "entry_point": "ep"},
        "hf_auto": {"module": "invarlock.adapters.hf", "entry_point": "auto"},
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
        lambda name, kind: (
            "⚠️ missing invarlock[gpu]" if name == "hf_bnb" else "✓ invarlock[adapters]"
        ),
        raising=False,
    )
    monkeypatch.setattr(
        plugins_mod, "console", Console(file=io.StringIO()), raising=False
    )
    plugins_command(
        category="adapters",
        only="missing",
        verbose=False,
        json_out=True,
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["category"] == "adapters"
    assert "kind" not in payload
    needs_extra_items = [
        i for i in payload["items"] if i.get("status") == "needs_extra"
    ]
    assert [item["name"] for item in needs_extra_items] == ["hf_bnb"]
    assert {item["status"] for item in payload["items"]} == {"needs_extra"}


def test_plugins_adapters_json_statuses(monkeypatch, capsys):
    adapters = {
        "hf_auto": {"module": "invarlock.adapters.hf", "entry_point": "auto"},
        "hf_bnb": {"module": "invarlock.plugins.bitsandbytes", "entry_point": "bnb"},
        "hf_gptq": {"module": "invarlock.plugins.gptq", "entry_point": "gptq"},
    }
    _patch_registry(monkeypatch, adapters)

    def fake_extract(name):
        if name == "hf_bnb":
            return SimpleNamespace(library="bitsandbytes", version="0.41")
        return SimpleNamespace(library="transformers", version="1.0")

    monkeypatch.setattr(
        "invarlock.core.backend_inventory.extract_adapter_provenance",
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
    assert payload["contracts"]["model_family_catalog"]["format_version"] == (
        "model-family-catalog-v1"
    )
    assert payload["model_family_catalog"]["format_version"] == (
        "model-family-catalog-v1"
    )
    statuses = {item["name"]: item["status"] for item in payload["items"]}
    assert statuses["hf_auto"] == "ready"
    assert statuses["hf_gptq"] == "needs_extra"


def test_plugins_adapters_json_bnb_ready_without_cuda_when_runtime_is_available(
    monkeypatch, capsys
):
    adapters = {
        "hf_bnb": {"module": "invarlock.plugins.bitsandbytes", "entry_point": "bnb"},
    }
    _patch_registry(monkeypatch, adapters)
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)),
    )
    monkeypatch.setattr(
        plugins_mod, "bitsandbytes_runtime_available", lambda: True, raising=False
    )
    monkeypatch.setattr(
        "invarlock.core.backend_inventory.extract_adapter_provenance",
        lambda name: SimpleNamespace(library="bitsandbytes", version="0.49.2"),
        raising=False,
    )

    plugins_command(category="adapters", json_out=True, hide_unsupported=False)
    payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    item = next(x for x in payload["items"] if x["name"] == "hf_bnb")
    assert item["status"] == "ready"
    assert item["backend"] == {
        "name": "bitsandbytes",
        "present": True,
        "version": "0.49.2",
    }


def test_plugins_adapters_json_marks_missing_backends_not_present(monkeypatch, capsys):
    adapters = {
        "hf_awq": {"module": "invarlock.plugins.awq", "entry_point": "awq"},
        "hf_gptq": {"module": "invarlock.plugins.gptq", "entry_point": "gptq"},
    }
    _patch_registry(monkeypatch, adapters)
    monkeypatch.setattr(plugins_mod.platform, "system", lambda: "Linux", raising=False)

    def fake_extract(name):
        return SimpleNamespace(library="gptqmodel", version=None)

    monkeypatch.setattr(
        "invarlock.core.backend_inventory.extract_adapter_provenance",
        fake_extract,
        raising=False,
    )

    plugins_command(category="adapters", json_out=True, hide_unsupported=False)
    payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    by_name = {item["name"]: item for item in payload["items"]}
    assert by_name["hf_awq"]["backend"] == {"name": "gptqmodel", "present": False}
    assert by_name["hf_gptq"]["backend"] == {"name": "gptqmodel", "present": False}


def test_plugins_adapters_minimal_only_ready(monkeypatch, capsys):
    adapters = {
        "invarlock_custom": {"module": "invarlock.plugins.custom", "entry_point": "c"},
        "hf_internal": {"module": "invarlock.adapters.internal", "entry_point": "i"},
    }
    _patch_registry(monkeypatch, adapters)
    monkeypatch.setenv("INVARLOCK_MINIMAL", "1")
    monkeypatch.setattr(
        "invarlock.core.backend_inventory.extract_adapter_provenance",
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
    assert payload["category"] == "datasets"
    assert "kind" not in payload


def test_plugins_datasets_json_does_not_instantiate_parameterized_providers(
    monkeypatch, capsys
):
    _patch_registry(monkeypatch, {})

    class _NeedsArgsProvider:
        __module__ = "invarlock.eval.data"

        def __init__(self, dataset_name: str):
            self.dataset_name = dataset_name

    monkeypatch.setattr(
        plugins_mod, "list_providers", lambda: ["hf_seq2seq"], raising=False
    )
    monkeypatch.setattr(
        data_mod, "list_providers", lambda: ["hf_seq2seq"], raising=False
    )
    monkeypatch.setattr(
        data_mod,
        "_PROVIDERS",
        {"hf_seq2seq": _NeedsArgsProvider},
        raising=False,
    )

    plugins_command(category="datasets", json_out=True)
    payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert payload["category"] == "datasets"
    assert "kind" not in payload
    assert payload["items"] == [
        {
            "name": "hf_seq2seq",
            "module": "invarlock.eval.data",
            "status": "available",
        }
    ]


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
            return SimpleNamespace(library="gptqmodel", version="1.0")
        if name == "hf_hint":
            return SimpleNamespace(library="transformers", version="1.2")
        raise RuntimeError("no provenance")

    monkeypatch.setattr(
        "invarlock.core.backend_inventory.extract_adapter_provenance",
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
        plugins_mod, "bitsandbytes_runtime_available", lambda: False, raising=False
    )
    monkeypatch.setattr(
        plugins_mod, "console", Console(file=io.StringIO()), raising=False
    )

    plugins_command(category="adapters", json_out=True, hide_unsupported=False)
    payload = json.loads(capsys.readouterr().out.strip())
    statuses = {item["name"]: item["status"] for item in payload["items"]}
    assert statuses["hf_bnb"] == "unsupported"
    assert statuses["hf_gptq"] == "ready"
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
            "wikitext2": SimpleNamespace(__module__="invarlock.eval.data"),
            "synthetic": SimpleNamespace(__module__="invarlock.eval.data"),
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
        "invarlock.core.backend_inventory.extract_adapter_provenance",
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
    def fake_import(name, *args, **kwargs):
        raise ImportError("missing")

    monkeypatch.setattr("builtins.__import__", fake_import)
    result = plugins_mod._check_plugin_extras("hf_gptq", "adapters")
    assert "invarlock[gptq]" in result
    result = plugins_mod._check_plugin_extras("hf_torchao", "adapters")
    assert "invarlock[torchao]" in result
    result = plugins_mod._check_plugin_extras("hf_hqq", "adapters")
    assert "invarlock[hqq]" in result
    result = plugins_mod._check_plugin_extras("hf_quanto", "adapters")
    assert "invarlock[quanto]" in result
    result = plugins_mod._check_plugin_extras("hf_ct", "adapters")
    assert "invarlock[compressed-tensors]" in result


def test_gptqmodel_plugin_extra_uses_named_runtime_boundary(monkeypatch) -> None:
    from invarlock.cli.commands import plugins_extras

    calls: list[str] = []
    monkeypatch.setattr(
        plugins_extras,
        "require_gptqmodel_runtime",
        lambda: calls.append("runtime"),
    )

    assert plugins_extras._plugin_package_importable("gptqmodel") is True
    assert calls == ["runtime"]


def test_check_plugin_extras_flags_old_multimodal_stack(monkeypatch):
    monkeypatch.setattr(
        plugins_mod,
        "_plugin_package_importable",
        lambda package_name: package_name in {"transformers", "torchvision", "PIL"},
    )
    monkeypatch.setattr(
        plugins_mod,
        "_package_version_at_least",
        lambda package_name, _minimum: package_name != "transformers",
    )

    result = plugins_mod._check_plugin_extras("hf_multimodal", "adapters")

    assert "invarlock[multimodal]" in result


def test_check_plugin_extras_flags_old_core_hf_stack(monkeypatch):
    monkeypatch.setattr(
        plugins_mod,
        "_plugin_package_importable",
        lambda package_name: package_name == "transformers",
    )
    monkeypatch.setattr(
        plugins_mod,
        "_package_version_at_least",
        lambda package_name, _minimum: package_name != "transformers",
    )

    result = plugins_mod._check_plugin_extras("hf_causal", "adapters")

    assert "invarlock[adapters]" in result


def test_plugins_adapters_verbose_console(monkeypatch):
    adapters = {
        "hf_auto": {"module": "invarlock.adapters.hf", "entry_point": "auto"},
        "hf_bnb": {"module": "invarlock.plugins.bitsandbytes", "entry_point": "bnb"},
        "hf_gptq": {"module": "invarlock.plugins.gptq", "entry_point": "gptq"},
    }
    _patch_registry(monkeypatch, adapters)
    monkeypatch.setattr(
        "invarlock.core.backend_inventory.extract_adapter_provenance",
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
        "invarlock.core.backend_inventory.extract_adapter_provenance",
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
        "invarlock.core.backend_inventory.extract_adapter_provenance",
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
        "invarlock.core.backend_inventory.extract_adapter_provenance",
        lambda name: SimpleNamespace(library="gptqmodel", version=None),
        raising=False,
    )
    monkeypatch.setattr(
        "invarlock.cli.commands.plugins._check_plugin_extras",
        lambda *a, **k: "",
        raising=False,
    )
    monkeypatch.setattr(plugins_mod.platform, "system", lambda: "Darwin", raising=False)
    plugins_command(
        category="adapters",
        json_out=True,
        hide_unsupported=False,
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["items"][0]["status"] == "needs_extra"
    assert payload["items"][0]["backend"] == {"name": "gptqmodel", "present": False}


def test_plugins_adapters_explain_enable_hint(monkeypatch):
    adapters = {
        "hf_hint": {"module": "invarlock.plugins.hint", "entry_point": "adapter:hint"},
    }
    _patch_registry(monkeypatch, adapters)
    monkeypatch.setattr(
        "invarlock.core.backend_inventory.extract_adapter_provenance",
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
        "hf_torchao": {
            "module": "invarlock.plugins.torchao",
            "entry_point": "torchao",
        },
        "hf_hqq": {"module": "invarlock.plugins.hqq", "entry_point": "hqq"},
        "hf_quanto": {"module": "invarlock.plugins.quanto", "entry_point": "quanto"},
        "hf_ct": {"module": "invarlock.plugins.ct", "entry_point": "ct"},
    }
    _patch_registry(monkeypatch, adapters)
    monkeypatch.setattr(
        "invarlock.core.backend_inventory.extract_adapter_provenance",
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
    assert any("GPTQModel-compatible" in line for line in dummy_console.lines)
    assert any("Uses GPTQModel" in line for line in dummy_console.lines)
    dummy_console.lines.clear()
    plugins_command(category="adapters", explain="hf_awq")
    assert any("AWQ-quantized" in line for line in dummy_console.lines)
    assert any(
        "Transformers AWQ through GPTQModel" in line for line in dummy_console.lines
    )
    dummy_console.lines.clear()
    plugins_command(category="adapters", explain="hf_torchao")
    assert any("torchao" in line for line in dummy_console.lines)
    assert any("int8 weight-only" in line for line in dummy_console.lines)
    dummy_console.lines.clear()
    plugins_command(category="adapters", explain="hf_hqq")
    assert any("HQQ" in line or "hqq" in line for line in dummy_console.lines)
    assert any("Runtime applies HQQ" in line for line in dummy_console.lines)
    dummy_console.lines.clear()
    plugins_command(category="adapters", explain="hf_quanto")
    assert any("Quanto" in line or "quanto" in line for line in dummy_console.lines)
    assert any("Runtime applies Quanto" in line for line in dummy_console.lines)
    dummy_console.lines.clear()
    plugins_command(category="adapters", explain="hf_ct")
    assert any("compressed-tensors" in line for line in dummy_console.lines)
    assert any("pre-quantized" in line for line in dummy_console.lines)


def test_plugins_adapters_provenance_failure_graceful(monkeypatch, capsys):
    adapters = {
        "hf_err": {"module": "invarlock.plugins.err", "entry_point": "err"},
    }
    _patch_registry(monkeypatch, adapters)
    monkeypatch.setattr(
        "invarlock.core.backend_inventory.extract_adapter_provenance",
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

        @property
        def _PROVIDERS(self):
            raise RuntimeError("boom")

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
    assert "Hints:" not in text
