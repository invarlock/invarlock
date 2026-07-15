"""
InvarLock CLI Plugins Command
=========================

Handles the 'invarlock advanced plugins' command for listing available plugins.
Supports a minimal view via INVARLOCK_MINIMAL=1 to hide built‑in adapters.
"""

import importlib
import importlib.util
import os
import platform
import sys
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
    gather_runtime_provider_rows as _rendering_gather_runtime_provider_rows,
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
    help="Inspect available adapters, guards, edits, runtime providers, and datasets.",
)
_plugin_package_importable = _plugins_extras._plugin_package_importable
_package_version_at_least = _plugins_extras._package_version_at_least
_DEFAULT_PLUGIN_PACKAGE_IMPORTABLE = _plugin_package_importable
_DEFAULT_BITSANDBYTES_RUNTIME_AVAILABLE = bitsandbytes_runtime_available


def _light_import_enabled() -> bool:
    return os.getenv("INVARLOCK_LIGHT_IMPORT", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }


def _package_present_without_import(package_name: str) -> bool:
    try:
        spec = importlib.util.find_spec(package_name)
    except (AttributeError, ImportError, ModuleNotFoundError, ValueError) as error:
        raise ImportError(f"{package_name} not available") from error
    if spec is None:
        raise ImportError(f"{package_name} not available")
    return True


def _bitsandbytes_present_without_import() -> bool:
    try:
        return _package_present_without_import("bitsandbytes")
    except ImportError:
        return False


def _bitsandbytes_inventory_available() -> bool:
    if (
        _light_import_enabled()
        and bitsandbytes_runtime_available is _DEFAULT_BITSANDBYTES_RUNTIME_AVAILABLE
    ):
        return _bitsandbytes_present_without_import()
    return bitsandbytes_runtime_available()


def _check_plugin_extras(plugin_name: str, plugin_type: str) -> str:
    original_importable = _plugins_extras._plugin_package_importable
    original_version_check = _plugins_extras._package_version_at_least
    try:
        _plugins_extras._plugin_package_importable = (
            _package_present_without_import
            if _light_import_enabled()
            and _plugin_package_importable is _DEFAULT_PLUGIN_PACKAGE_IMPORTABLE
            else _plugin_package_importable
        )
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
    # Inventory must not import the runtime it is describing. If a caller has
    # already loaded torch, querying its CUDA state is cheap; otherwise report
    # the conservative metadata-only value and leave runtime probing to doctor.
    torch_mod = sys.modules.get("torch")
    if torch_mod is None:
        return False
    return detect_cuda_available(torch_mod)


def _gather_adapter_rows(registry: Any) -> list[dict[str, Any]]:
    return _rendering_gather_adapter_rows(
        registry=registry,
        minimal=_is_minimal_plugins_enabled(),
        has_cuda=_detect_current_cuda(),
        is_linux=platform.system().lower() == "linux",
        extras_checker=_check_plugin_extras,
        bitsandbytes_runtime_available=_bitsandbytes_inventory_available,
    )


def _gather_generic_rows(registry: Any, plugin_type: str) -> list[dict[str, Any]]:
    return _rendering_gather_generic_rows(
        registry,
        plugin_type,
        extras_checker=_check_plugin_extras,
    )


def _gather_runtime_provider_rows(registry: Any) -> list[dict[str, Any]]:
    return _rendering_gather_runtime_provider_rows(registry)


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
        runtime_provider_rows_loader=_gather_runtime_provider_rows,
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
        None,
        help="Category: adapters|guards|edits|runtime-providers|plugins|datasets",
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


@plugins_app.command("runtime-providers")
def _plugins_runtime_providers(
    only: str | None = typer.Option(
        None,
        "--only",
        help=(
            "Filter: missing|ready|core|optional|core_supported|"
            "first_party_experimental|third_party"
        ),
    ),
    verbose: bool = typer.Option(False, "--verbose", help="Verbose table output"),
    json_out: bool = typer.Option(False, "--json", help="Emit JSON output"),
    explain: str | None = typer.Option(
        None, "--explain", help="Explain a specific runtime provider"
    ),
    allow_third_party_plugins: bool = typer.Option(
        False,
        "--allow-third-party-plugins",
        help="Allow third-party plugin discovery for this command.",
    ),
):
    """List runtime-provider connectors without importing their backends."""
    return plugins_command(
        "runtime-providers",
        only=only,
        verbose=verbose,
        json_out=json_out,
        explain=explain,
        allow_third_party_plugins=allow_third_party_plugins,
    )


__all__ = [
    "plugins_app",
    "plugins_command",
]
