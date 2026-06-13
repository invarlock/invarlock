"""
InvarLock CLI Plugins Command
=========================

Handles the 'invarlock advanced plugins' command for listing available plugins.
Supports a minimal view via INVARLOCK_MINIMAL=1 to hide built‑in adapters.
"""

import importlib
import os
import platform
from typing import Any

import typer
from rich.console import Console

from invarlock.cli.commands import plugins_extras as _plugins_extras
from invarlock.core.plugins_inventory import (
    bitsandbytes_runtime_available,
    detect_cuda_available,
    is_minimal_plugins_view,
)

from ..security_helpers import runtime_security_scoped
from .plugins_rendering import (
    gather_adapter_rows as _rendering_gather_adapter_rows,
)
from .plugins_rendering import (
    gather_generic_rows as _rendering_gather_generic_rows,
)
from .plugins_rendering import (
    handle_plugins_category as _rendering_handle_plugins_category,
)

console = Console()
_PLUGIN_REGISTRY_IMPORT_ERRORS = (AttributeError, ImportError, RuntimeError)
_PLUGIN_COMMAND_ERRORS = (
    AttributeError,
    ImportError,
    KeyError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)

# Group: plugins
plugins_app = typer.Typer(
    help="Inspect available adapters, guards, edits, and datasets.",
)
_plugin_package_importable = _plugins_extras._plugin_package_importable
_package_version_at_least = _plugins_extras._package_version_at_least


def _check_plugin_extras(plugin_name: str, plugin_type: str) -> str:
    original_importable = _plugins_extras._plugin_package_importable
    original_version_check = _plugins_extras._package_version_at_least
    try:
        _plugins_extras._plugin_package_importable = _plugin_package_importable
        _plugins_extras._package_version_at_least = _package_version_at_least
        return _plugins_extras.check_plugin_extras(plugin_name, plugin_type)
    finally:
        _plugins_extras._plugin_package_importable = original_importable
        _plugins_extras._package_version_at_least = original_version_check


def _load_provider_registry_map() -> dict[str, Any]:
    try:
        data_mod = importlib.import_module("invarlock.eval.data")
        return getattr(data_mod, "_PROVIDERS", {}) or {}
    except _PLUGIN_REGISTRY_IMPORT_ERRORS:
        return {}


def _is_minimal_plugins_enabled() -> bool:
    return is_minimal_plugins_view(os.environ.get("INVARLOCK_MINIMAL"))


def _detect_current_cuda() -> bool:
    try:
        import torch as _torch

        torch_mod: Any = _torch
        return detect_cuda_available(torch_mod)
    except ImportError:
        return False


def _gather_adapter_rows(registry: Any) -> list[dict[str, Any]]:
    return _rendering_gather_adapter_rows(
        registry=registry,
        minimal=_is_minimal_plugins_enabled(),
        has_cuda=_detect_current_cuda(),
        is_linux=platform.system().lower() == "linux",
        extras_checker=_check_plugin_extras,
        bitsandbytes_runtime_available=bitsandbytes_runtime_available,
    )


def _gather_generic_rows(registry: Any, plugin_type: str) -> list[dict[str, Any]]:
    return _rendering_gather_generic_rows(
        registry,
        plugin_type,
        extras_checker=_check_plugin_extras,
    )


def _handle_plugins_category(
    *,
    category: str | None,
    registry: Any,
    list_providers_fn: Any,
    only: str | None,
    verbose: bool,
    json_out: bool,
    explain: str | None,
    hide_unsupported: bool,
) -> None:
    _rendering_handle_plugins_category(
        category=category,
        registry=registry,
        list_providers_fn=list_providers_fn,
        only=only,
        verbose=verbose,
        json_out=json_out,
        explain=explain,
        hide_unsupported=hide_unsupported,
        console=console,
        adapter_rows_loader=_gather_adapter_rows,
        generic_rows_loader=_gather_generic_rows,
        provider_registry_loader=_load_provider_registry_map,
    )


@runtime_security_scoped
def plugins_command(
    category: str | None = None,
    only: str | None = None,
    verbose: bool = False,
    json_out: bool = False,
    explain: str | None = None,
    hide_unsupported: bool = True,
    allow_third_party_plugins: bool = False,
):
    """
    List available plugins with entry point information.

    Shows plugin names, module paths, and availability status without instantiation.

    Examples:
        invarlock advanced plugins list         # List all plugins
        invarlock advanced plugins guards       # List built-in guard plugins
        invarlock advanced plugins edits        # List built-in edit plugins
        invarlock advanced plugins adapters     # List built-in adapter plugins
        invarlock advanced plugins adapters --allow-third-party-plugins
    """
    try:
        from invarlock.core.registry import get_registry
        from invarlock.eval.data import list_providers

        registry = get_registry()
        _handle_plugins_category(
            category=category,
            registry=registry,
            list_providers_fn=list_providers,
            only=only,
            verbose=verbose,
            json_out=json_out,
            explain=explain,
            hide_unsupported=hide_unsupported,
        )

    except typer.Exit:
        # Propagate intended exit codes from command flow
        raise
    except _PLUGIN_COMMAND_ERRORS as e:
        console.print(f"[red]❌ Plugin listing failed: {e}[/red]")
        raise typer.Exit(1) from e


# Wire subcommands under group
@plugins_app.command("list")
def _plugins_list(
    category: str | None = typer.Argument(
        None, help="Category: adapters|guards|edits|plugins|datasets"
    ),
    json_out: bool = typer.Option(False, "--json", help="Emit JSON output"),
    verbose: bool = typer.Option(False, "--verbose", help="Verbose table output"),
    allow_third_party_plugins: bool = typer.Option(
        False,
        "--allow-third-party-plugins",
        help="Allow third-party plugin discovery for this command.",
    ),
):
    """List installed plugin entry points and adapters for a given category."""
    return plugins_command(
        category,
        verbose=verbose,
        json_out=json_out,
        allow_third_party_plugins=allow_third_party_plugins,
    )


@plugins_app.command("guards")
def _plugins_guards(
    only: str | None = typer.Option(
        None,
        "--only",
        help=(
            "Filter: missing|ready|core|optional|core_supported|demo_only|third_party"
        ),
    ),
    verbose: bool = typer.Option(False, "--verbose", help="Verbose table output"),
    json_out: bool = typer.Option(False, "--json", help="Emit JSON output"),
    allow_third_party_plugins: bool = typer.Option(
        False,
        "--allow-third-party-plugins",
        help="Allow third-party plugin discovery for this command.",
    ),
):
    """List available guard plugins.

    Shows built-in and third-party guards discovered via entry points.
    Use --json for machine-readable output.
    """
    return plugins_command(
        "guards",
        only=only,
        verbose=verbose,
        json_out=json_out,
        allow_third_party_plugins=allow_third_party_plugins,
    )


@plugins_app.command("edits")
def _plugins_edits(
    only: str | None = typer.Option(
        None,
        "--only",
        help=(
            "Filter: missing|ready|core|optional|core_supported|"
            "validation_simulation|internal_baseline_edit|third_party"
        ),
    ),
    verbose: bool = typer.Option(False, "--verbose", help="Verbose table output"),
    json_out: bool = typer.Option(False, "--json", help="Emit JSON output"),
    allow_third_party_plugins: bool = typer.Option(
        False,
        "--allow-third-party-plugins",
        help="Allow third-party plugin discovery for this command.",
    ),
):
    """List available edit plugins.

    Includes built-in edits like quant_rtn and any discovered third-party edits.
    Use --json for machine-readable output.
    """
    return plugins_command(
        "edits",
        only=only,
        verbose=verbose,
        json_out=json_out,
        allow_third_party_plugins=allow_third_party_plugins,
    )


@plugins_app.command("adapters")
def _plugins_adapters(
    only: str | None = typer.Option(
        None,
        "--only",
        help=(
            "Filter: missing|ready|core|optional|core_supported|"
            "optional_backend_loader|third_party"
        ),
    ),
    verbose: bool = typer.Option(False, "--verbose", help="Verbose table output"),
    json_out: bool = typer.Option(False, "--json", help="Emit JSON output"),
    explain: str | None = typer.Option(
        None, "--explain", help="Explain a specific adapter"
    ),
    hide_unsupported: bool = typer.Option(
        True,
        "--hide-unsupported/--show-unsupported",
        help="Hide adapters unsupported on this platform (default: hide)",
    ),
    allow_third_party_plugins: bool = typer.Option(
        False,
        "--allow-third-party-plugins",
        help="Allow third-party plugin discovery for this command.",
    ),
):
    """List available model adapters.

    Supports filtering (--only), verbose view (--verbose), JSON (--json),
    adapter explanation (--explain), and hiding unsupported stacks on this platform.
    """
    return plugins_command(
        "adapters",
        only=only,
        verbose=verbose,
        json_out=json_out,
        explain=explain,
        hide_unsupported=hide_unsupported,
        allow_third_party_plugins=allow_third_party_plugins,
    )


__all__ = [
    "plugins_app",
    "plugins_command",
]
