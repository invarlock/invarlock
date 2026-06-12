"""
InvarLock CLI Plugins Command
=========================

Handles the 'invarlock advanced plugins' command for listing available plugins.
Supports a minimal view via INVARLOCK_MINIMAL=1 to hide built‑in adapters.
"""

import importlib
import importlib.metadata as importlib_metadata
import io
import os
import platform
import re
import warnings
from contextlib import redirect_stderr, redirect_stdout
from typing import Any

import typer
from rich.console import Console

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
_EXTRA_CHECK_ERRORS = (AttributeError, ImportError, OSError, RuntimeError, ValueError)

# Group: plugins
plugins_app = typer.Typer(
    help="Inspect available adapters, guards, edits, and datasets.",
)


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


def _check_plugin_extras(plugin_name: str, plugin_type: str) -> str:
    """Check if plugin requires missing optional extras."""
    # Enhanced extras checking without importing heavy modules (avoid noisy warnings)
    # Only include baked-in plugins that are available through entry points
    extras_map = {
        # Edit plugins (baked-in only)
        "quant_rtn": {"packages": [], "extra": ""},
        # Guard plugins (no extra deps typically)
        "invariants": {"packages": [], "extra": ""},
        "spectral": {"packages": [], "extra": ""},
        "variance": {"packages": [], "extra": ""},
        "rmt": {"packages": [], "extra": ""},
        "demo_hello_guard": {"packages": [], "extra": ""},
        # Adapter plugins (baked-in only)
        "hf_causal": {"packages": ["transformers"], "extra": "invarlock[adapters]"},
        "hf_mlm": {"packages": ["transformers"], "extra": "invarlock[adapters]"},
        "hf_seq2seq": {"packages": ["transformers"], "extra": "invarlock[adapters]"},
        "hf_multimodal": {
            "packages": ["transformers", "torchvision", "PIL"],
            "extra": "invarlock[multimodal]",
            "minimum_versions": {
                "transformers": "5.12.0",
                "torchvision": "0.26.0",
            },
        },
        "hf_auto": {"packages": ["transformers"], "extra": "invarlock[adapters]"},
        # Optional adapter plugins
        "hf_gptq": {"packages": ["gptqmodel"], "extra": "invarlock[gptq]"},
        "hf_awq": {"packages": ["gptqmodel"], "extra": "invarlock[awq]"},
        "hf_bnb": {"packages": ["bitsandbytes"], "extra": "invarlock[gpu]"},
        "hf_torchao": {"packages": ["torchao"], "extra": "invarlock[torchao]"},
        "hf_hqq": {"packages": ["hqq"], "extra": "invarlock[hqq]"},
        "hf_quanto": {
            "packages": ["optimum.quanto"],
            "extra": "invarlock[quanto]",
        },
        "hf_ct": {
            "packages": ["compressed_tensors"],
            "extra": "invarlock[compressed-tensors]",
        },
    }

    plugin_info = extras_map.get(plugin_name)
    if not plugin_info or not plugin_info["packages"]:
        return ""  # No extra dependencies needed

    # Check each required package. For most packages we use a light import so
    # tests can monkeypatch __import__. For GPU-only stacks (bitsandbytes), we
    # probe runtime readiness instead of importing.
    missing_packages: list[str] = []
    minimum_versions = plugin_info.get("minimum_versions", {})
    for pkg in plugin_info["packages"]:
        try:
            _plugin_package_importable(str(pkg))
            minimum_version = minimum_versions.get(pkg)
            if minimum_version and not _package_version_at_least(
                pkg, str(minimum_version)
            ):
                raise ImportError(f"{pkg}>={minimum_version} not available")
        except _EXTRA_CHECK_ERRORS:
            missing_packages.append(pkg)

    # Format the result
    if not missing_packages:
        # All dependencies available
        if plugin_info["extra"]:
            return f"✓ {plugin_info['extra']}"
        else:
            return "✓ Available"
    else:
        # Some dependencies missing
        if plugin_info["extra"]:
            return f"⚠️ missing {plugin_info['extra']}"
        else:
            return f"⚠️ missing {', '.join(missing_packages)}"


def _plugin_package_importable(package_name: str) -> bool:
    if package_name == "bitsandbytes":
        if not bitsandbytes_runtime_available():
            raise ImportError("bitsandbytes not importable")
        return True
    if package_name == "gptqmodel":
        from invarlock.plugins import _patch_gptqmodel_transformers_hub_compat

        _patch_gptqmodel_transformers_hub_compat()
    with (
        warnings.catch_warnings(),
        redirect_stdout(io.StringIO()),
        redirect_stderr(io.StringIO()),
    ):
        warnings.simplefilter("ignore")
        __import__(package_name)
    return True


def _package_version_at_least(package_name: str, minimum: str) -> bool:
    try:
        installed = importlib_metadata.version(package_name)
    except importlib_metadata.PackageNotFoundError:
        return False
    return _version_key(installed) >= _version_key(minimum)


def _version_key(value: str) -> tuple[int, int, int]:
    parts = [int(match.group(0)) for match in re.finditer(r"\d+", value)]
    padded = (parts + [0, 0, 0])[:3]
    return (padded[0], padded[1], padded[2])


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
