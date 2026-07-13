from __future__ import annotations

import io
import json
import sys
from types import SimpleNamespace

import pytest
import typer
from rich.console import Console

from invarlock.cli.commands import plugins, plugins_extras
from invarlock.cli.commands import plugins_rendering as rendering


def _console() -> tuple[Console, io.StringIO]:
    stream = io.StringIO()
    return Console(file=stream, width=180), stream


def _adapter_row(**updates):
    row = {
        "name": "adapter",
        "kind": "adapter",
        "module": "pkg.adapter",
        "entry_point": "",
        "support": "core",
        "origin": "core",
        "mode": "adapter",
        "backend": None,
        "backend_version": None,
        "status": "ready",
        "enable": "",
        "support_tier": "core_supported",
        "strict_assurance_allowed": True,
        "deployment_claim": True,
    }
    row.update(updates)
    return row


def test_plugins_runtime_absence_and_command_failure_are_explicit(monkeypatch):
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    assert plugins._detect_current_cuda() is False

    monkeypatch.setattr(
        "invarlock.core.registry.get_registry",
        lambda: (_ for _ in ()).throw(OSError("registry unavailable")),
    )
    console, stream = _console()
    monkeypatch.setattr(plugins, "console", console)
    with pytest.raises(typer.Exit) as exc:
        plugins.plugins_command(category="guards")
    assert exc.value.exit_code == 1
    assert "registry unavailable" in stream.getvalue()


def test_plugin_extra_runtime_success_paths(monkeypatch):
    monkeypatch.setattr(plugins_extras, "bitsandbytes_runtime_available", lambda: True)
    assert plugins_extras._plugin_package_importable("bitsandbytes") is True

    called = []
    monkeypatch.setattr(
        plugins_extras, "require_gptqmodel_runtime", lambda: called.append("gptq")
    )
    assert plugins_extras._plugin_package_importable("gptqmodel") is True
    assert called == ["gptq"]


def test_plugins_json_extra_and_compact_fallback_statuses(monkeypatch, capsys):
    monkeypatch.setattr(rendering, "contract_catalog", lambda: {})
    monkeypatch.setattr(rendering, "load_support_matrix", lambda: {})
    monkeypatch.setattr(rendering, "load_model_family_catalog", lambda: {})
    rendering._emit_plugins_json("guards", [], {"source": "registry"})
    assert json.loads(capsys.readouterr().out)["source"] == "registry"

    console, stream = _console()
    rendering._print_adapters_compact(
        [
            _adapter_row(status="unsupported", support="optional"),
            _adapter_row(name="custom", status="degraded", support="optional"),
        ],
        console=console,
    )
    output = stream.getvalue()
    assert "Unsupported on this platform" in output
    assert "degraded" in output


def test_plugin_explain_and_generic_status_edges():
    console, stream = _console()
    rendering._explain_adapter(
        "adapter",
        rows=[_adapter_row(status="partial")],
        console=console,
    )
    assert "Partial" in stream.getvalue()

    console, stream = _console()
    rendering._print_generic_compact(
        [
            {
                **_adapter_row(status="custom"),
                "mode": "guard",
            }
        ],
        "Guards",
        console=console,
    )
    assert "custom" in stream.getvalue()

    console, stream = _console()
    with pytest.raises(typer.Exit):
        rendering._explain_generic("missing", "guards", rows=[], console=console)
    assert "Unknown guard" in stream.getvalue()

    console, stream = _console()
    rendering._explain_generic(
        "needs",
        "guards",
        rows=[
            {
                **_adapter_row(
                    name="needs", status="needs_extra", enable="invarlock[gpu]"
                ),
                "mode": "guard",
            }
        ],
        console=console,
    )
    assert "Enable" in stream.getvalue()


def test_all_categories_dispatches_nonempty_dataset_provider():
    calls: list[str] = []
    registry = SimpleNamespace(
        list_guards=lambda: [],
        list_edits=lambda: [],
        list_adapters=lambda: [],
    )

    def list_providers() -> list[str]:
        calls.append("list")
        return ["synthetic"]

    def load_provider_registry() -> dict[str, SimpleNamespace]:
        calls.append("registry")
        return {"synthetic": SimpleNamespace(__module__="invarlock.eval.data")}

    rendering.handle_plugins_category(
        category=None,
        registry=registry,
        list_providers_fn=list_providers,
        only=None,
        verbose=False,
        json_out=False,
        explain=None,
        hide_unsupported=True,
        console=_console()[0],
        adapter_rows_loader=lambda _registry: [],
        generic_rows_loader=lambda _registry, _kind: [],
        provider_registry_loader=load_provider_registry,
    )
    assert calls == ["list", "registry"]


def test_plugin_extra_status_handles_unknown_available_and_unversioned_plugins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert plugins_extras.check_plugin_extras("unknown", "adapters") == ""
    assert plugins_extras._plugin_package_importable("json") is True
    monkeypatch.setattr(plugins_extras, "bitsandbytes_runtime_available", lambda: False)
    with pytest.raises(ImportError, match="bitsandbytes not importable"):
        plugins_extras._plugin_package_importable("bitsandbytes")
    monkeypatch.setattr(
        plugins_extras, "_plugin_package_importable", lambda _package: True
    )
    monkeypatch.setattr(
        plugins_extras, "_package_version_at_least", lambda *_args: True
    )
    assert plugins_extras.check_plugin_extras("hf_gptq", "adapters") == (
        "✓ invarlock[gptq]"
    )


def test_plugin_version_check_treats_missing_distribution_as_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        plugins_extras.importlib_metadata,
        "version",
        lambda _package: (_ for _ in ()).throw(
            plugins_extras.importlib_metadata.PackageNotFoundError
        ),
    )
    assert plugins_extras._package_version_at_least("absent", "1.0") is False
    assert plugins_extras._version_key("release-7") == (7, 0, 0)
