"""
InvarLock CLI Plugins Command
=========================

Handles the 'invarlock advanced plugins' command for listing available plugins.
Supports a minimal view via INVARLOCK_MINIMAL=1 to hide built‑in adapters.
"""

import json
import importlib
import os
import platform
from typing import Any

import typer
from rich.console import Console
from rich.markup import escape as _escape
from rich.table import Table

from invarlock.public_contracts import (
    adapter_capability,
    contract_catalog,
    load_model_family_catalog,
    load_support_matrix,
)
from invarlock.core.plugins_inventory import (
    adapter_inventory_json_items,
    combined_plugins_json_items,
    dataset_inventory_json_items,
    detect_cuda_available,
    filter_inventory_rows,
    gather_adapter_inventory_rows,
    gather_generic_inventory_rows,
    generic_inventory_json_items,
    is_minimal_plugins_view,
)

from ..backend_runtime import bitsandbytes_runtime_available
from ..constants import PLUGINS_FORMAT_VERSION
from ..security_helpers import runtime_security_scoped

console = Console()

# Group: plugins
plugins_app = typer.Typer(
    help="Inspect available adapters, guards, edits, and datasets.",
)


def _sort_rows(rows):
    """Stable sort by name, kind, then module and entry_point."""
    return sorted(
        rows,
        key=lambda r: (
            str(r.get("name", "")).lower(),
            str(r.get("kind", "")).lower(),
            str(r.get("module", "")).lower(),
            str(r.get("entry_point", "")).lower(),
        ),
    )


def _emit_plugins_json(category: str, rows, extra: dict | None = None) -> None:
    payload = {
        "format_version": PLUGINS_FORMAT_VERSION,
        "category": category,
        "items": _sort_rows(rows),
        "contracts": contract_catalog(),
        "support_matrix": load_support_matrix(),
        "model_family_catalog": load_model_family_catalog(),
    }
    if extra:
        payload.update(extra)
    typer.echo(json.dumps(payload, ensure_ascii=False))


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

        def _is_minimal() -> bool:
            return is_minimal_plugins_view(os.environ.get("INVARLOCK_MINIMAL"))

        def _gather_adapter_rows() -> list[dict]:
            try:
                import torch as _torch

                torch_mod: Any = _torch
                has_cuda = detect_cuda_available(torch_mod)
            except Exception:
                has_cuda = False
            from invarlock.core.adapter_provenance import extract_adapter_provenance

            rows = gather_adapter_inventory_rows(
                registry=registry,
                minimal=_is_minimal(),
                has_cuda=has_cuda,
                is_linux=platform.system().lower() == "linux",
                extras_checker=_check_plugin_extras,
                provenance_extractor=extract_adapter_provenance,
                bitsandbytes_runtime_available=bitsandbytes_runtime_available,
            )
            for row in rows:
                row["capability"] = adapter_capability(str(row.get("name") or ""))
            return rows

        def _filter_only(rows: list[dict]) -> list[dict]:
            return filter_inventory_rows(rows, only)

        def _fmt_backend(backend: str | None, version: str | None) -> tuple[str, str]:
            name = backend or "—"
            if backend and version:
                return backend, f"=={version}"
            return name, "—"

        def _print_adapters_compact(rows: list[dict]) -> None:
            need = sum(1 for r in rows if r["status"] == "needs_extra")
            ready = sum(1 for r in rows if r["status"] == "ready")
            auto = sum(1 for r in rows if r["support"] == "auto")
            unsupported = sum(1 for r in rows if r["status"] == "unsupported")
            title = f"Adapters — ready: {ready} · auto: {auto} · missing-extras: {need} · unsupported: {unsupported}"
            table = Table(title=title)
            table.add_column("Adapter", style="cyan")
            table.add_column("Origin", style="dim")
            table.add_column("Mode", style="dim")
            table.add_column("Backend", style="magenta")
            table.add_column("Version", style="magenta")
            table.add_column("Status / Action", style="green")
            for idx, r in enumerate(rows):
                backend_disp, version_disp = _fmt_backend(
                    r.get("backend"), r.get("backend_version")
                )
                origin_disp = r.get("origin", r.get("support", "")).capitalize()
                mode_disp = (
                    "Auto‑matcher" if r.get("mode") == "auto-matcher" else "Adapter"
                )
                if r["support"] == "auto":
                    status_disp = "Ready"
                elif r["status"] == "ready":
                    status_disp = "Ready"
                elif r["status"] == "needs_extra":
                    status_disp = f"Needs extra: {r['enable'] or ''}".rstrip(": ")
                elif r["status"] == "unsupported":
                    status_disp = "Unsupported on this platform"
                else:
                    status_disp = r["status"]
                next_support = rows[idx + 1]["support"] if idx + 1 < len(rows) else None
                end_section = next_support is not None and next_support != r["support"]
                table.add_row(
                    r["name"],
                    origin_disp,
                    mode_disp,
                    backend_disp,
                    version_disp,
                    _escape(status_disp),
                    end_section=end_section,
                )
            # Hints
            console.print(table)

        def _print_adapters_verbose(rows: list[dict]) -> None:
            table = Table(title="Adapters (verbose)")
            table.add_column("Adapter", style="cyan")
            table.add_column("Origin", style="dim")
            table.add_column("Mode", style="dim")
            table.add_column("Backend", style="magenta")
            table.add_column("Version", style="magenta")
            table.add_column("Status", style="green")
            table.add_column("Module", style="dim")
            table.add_column("Entry Point", style="dim")
            for r in rows:
                backend_disp, version_disp = _fmt_backend(
                    r.get("backend"), r.get("backend_version")
                )
                entry = r["entry_point"] or ""
                entry_disp = (
                    entry
                    if entry
                    else ("(auto matcher)" if r["support"] == "auto" else "")
                )
                table.add_row(
                    r["name"],
                    (r.get("origin") or r.get("support") or "").capitalize(),
                    ("Auto‑matcher" if r.get("mode") == "auto-matcher" else "Adapter"),
                    backend_disp,
                    version_disp,
                    r["status"].replace("needs_extra", "Needs extra").capitalize(),
                    r["module"],
                    entry_disp,
                )
            console.print(table)

        def _print_adapters_json(rows: list[dict]) -> None:
            _emit_plugins_json("adapters", adapter_inventory_json_items(rows))

        def _explain_adapter(name: str) -> None:
            rows = _gather_adapter_rows()
            r = next((x for x in rows if x["name"] == name), None)
            if not r:
                console.print(f"[red]❌ Unknown adapter: {name}[/red]")
                raise typer.Exit(1)
            backend_disp = (
                f"{r['backend']} {r['backend_version']}"
                if r["backend"] and r["backend_version"]
                else (f"{r['backend']} (missing)" if r["backend"] else "-")
            )
            console.print(f"[bold]{r['name']}[/bold]")
            console.print(f"  Support     : {r['support'].capitalize()}")
            console.print(f"  Backend     : {backend_disp}")
            if r["support"] == "auto":
                console.print("  Status      : Ready (auto matcher)")
            elif r["status"] == "ready":
                console.print("  Status      : Ready")
            elif r["status"] == "needs_extra":
                console.print("  Status      : Needs extra")
                if r["enable"]:
                    console.print(f"  Enable      : {_escape(r['enable'])}")
            elif r["status"] == "partial":
                console.print("  Status      : Partial (GPU-only features disabled)")
            if r["name"] == "hf_gptq":
                console.print(
                    "  Matches     : AutoGPTQ-quantized HF repos (from_quantized)"
                )
                console.print(
                    "  Notes       : Linux/CUDA recommended; upstream auto-gptq "
                    "packaging may require a pinned or vendor wheel"
                )
            elif r["name"] == "hf_awq":
                console.print("  Matches     : AWQ-quantized HF repos")
                console.print(
                    "  Notes       : GPU recommended; metadata ingestion on CPU"
                )
            elif r["name"] == "hf_bnb":
                console.print(
                    "  Matches     : Transformers 4/8-bit loading with bitsandbytes"
                )
                console.print(
                    "  Notes       : GPU recommended; falls back to metadata only on CPU"
                )
            else:
                console.print(
                    "  Matches     : Hugging Face Transformers (core adapters)"
                )
            console.print(f"  Module      : {r['module']}")
            entry = r["entry_point"] or ""
            if entry:
                console.print(f"  Entry point : {entry}")

        # Generic (guards/edits) helpers for compact/verbose/json/explain
        def _gather_generic_rows(plugin_type: str) -> list[dict]:
            return gather_generic_inventory_rows(
                registry=registry,
                plugin_type=plugin_type,
                extras_checker=_check_plugin_extras,
            )

        def _print_generic_compact(rows: list[dict], title: str) -> None:
            need = sum(1 for r in rows if r["status"] == "needs_extra")
            ready = sum(1 for r in rows if r["status"] == "ready")
            table = Table(title=f"{title} — ready: {ready} · missing-extras: {need}")
            table.add_column("Name", style="cyan")
            table.add_column("Origin", style="dim")
            table.add_column("Mode", style="dim")
            table.add_column("Backend", style="magenta")
            table.add_column("Version", style="magenta")
            table.add_column("Status / Action", style="green")
            for idx, r in enumerate(rows):
                backend_disp, version_disp = _fmt_backend(
                    r.get("backend"), r.get("backend_version")
                )
                origin_disp = r.get("origin", r.get("support", "")).capitalize()
                mode_disp = "Guard" if r.get("mode") == "guard" else "Edit"
                if r["status"] == "ready":
                    status_disp = "Ready"
                elif r["status"] == "needs_extra":
                    status_disp = f"Needs extra: {r['enable'] or ''}".rstrip(": ")
                else:
                    status_disp = r["status"]
                next_support = rows[idx + 1]["support"] if idx + 1 < len(rows) else None
                end_section = next_support is not None and next_support != r["support"]
                table.add_row(
                    r["name"],
                    origin_disp,
                    mode_disp,
                    backend_disp,
                    version_disp,
                    _escape(status_disp),
                    end_section=end_section,
                )
            console.print(table)

        def _print_generic_verbose(rows: list[dict], title: str) -> None:
            table = Table(title=f"{title} (verbose)")
            table.add_column("Name", style="cyan")
            table.add_column("Origin", style="dim")
            table.add_column("Mode", style="dim")
            table.add_column("Backend", style="magenta")
            table.add_column("Version", style="magenta")
            table.add_column("Status", style="green")
            table.add_column("Module", style="dim")
            table.add_column("Entry Point", style="dim")
            for idx, r in enumerate(rows):
                entry = r["entry_point"] or ""
                next_support = rows[idx + 1]["support"] if idx + 1 < len(rows) else None
                end_section = next_support is not None and next_support != r["support"]
                backend_disp, version_disp = _fmt_backend(
                    r.get("backend"), r.get("backend_version")
                )
                table.add_row(
                    r["name"],
                    (r.get("origin") or r.get("support") or "").capitalize(),
                    ("Guard" if r.get("mode") == "guard" else "Edit"),
                    backend_disp,
                    version_disp,
                    r["status"].replace("needs_extra", "Needs extra").capitalize(),
                    r["module"],
                    entry,
                    end_section=end_section,
                )
            console.print(table)

        def _print_generic_json(rows: list[dict], kind: str) -> None:
            _emit_plugins_json(
                kind,
                generic_inventory_json_items(rows, kind=kind),
            )

        def _render_dataset_table(
            title: str, providers: list[str], *, verbose: bool = False
        ) -> None:
            from invarlock.cli.constants import (
                PROVIDER_KIND as provider_kind,
            )
            from invarlock.cli.constants import (
                PROVIDER_NETWORK as provider_network,
            )
            from invarlock.cli.constants import (
                PROVIDER_PARAMS as provider_params,
            )

            try:
                _data_mod = importlib.import_module("invarlock.eval.data")
                _providers_map = getattr(_data_mod, "_PROVIDERS", {}) or {}
            except Exception:
                _providers_map = {}

            def _net_label(name: str) -> str:
                val = (provider_network.get(name, "") or "").lower()
                if val == "cache":
                    return "Cache/Net"
                if val == "yes":
                    return "Yes"
                if val == "no":
                    return "No"
                return "Unknown"

            rows: list[dict] = []
            for pname in providers:
                rows.append(
                    {
                        "name": pname,
                        "network": _net_label(pname),
                        "kind": provider_kind.get(pname, "-"),
                        "module": getattr(
                            _providers_map.get(pname, None), "__module__", "unknown"
                        ),
                    }
                )
            net_order = {"No": 0, "Cache/Net": 1, "Yes": 2, "Unknown": 3}
            rows.sort(key=lambda r: (net_order.get(r["network"], 9), r["name"]))

            cnt_no = sum(1 for r in rows if r["network"] == "No")
            cnt_cache = sum(1 for r in rows if r["network"] == "Cache/Net")
            cnt_yes = sum(1 for r in rows if r["network"] == "Yes")
            table = Table(
                title=f"{title} — offline: {cnt_no} · cache/net: {cnt_cache} · network: {cnt_yes}"
            )
            table.add_column("Provider", style="cyan")
            table.add_column("Network", style="dim")
            table.add_column("Kind", style="dim")
            table.add_column("Params", style="dim")
            if verbose:
                table.add_column("Module", style="dim")
            table.add_column("Status / Action", style="green")

            for idx, r in enumerate(rows):
                end_section = (
                    idx + 1 < len(rows) and rows[idx + 1]["network"] != r["network"]
                )
                cols = [
                    r["name"],
                    r["network"],
                    r.get("kind", "-"),
                    provider_params.get(r["name"], "-"),
                ]
                if verbose:
                    cols.append(r["module"])
                cols.append("✓ Available")
                table.add_row(*cols, end_section=end_section)

            console.print(table)

        def _explain_generic(name: str, plugin_type: str) -> None:
            rows = _gather_generic_rows(plugin_type)
            r = next((x for x in rows if x["name"] == name), None)
            if not r:
                console.print(f"[red]❌ Unknown {plugin_type[:-1]}: {name}[/red]")
                raise typer.Exit(1)
            console.print(f"[bold]{r['name']}[/bold]")
            console.print(f"  Support     : {r['support'].capitalize()}")
            backend_label = r.get("backend") or "—"
            console.print(f"  Backend     : {backend_label}")
            if r["status"] == "ready":
                console.print("  Status      : Ready")
            elif r["status"] == "needs_extra":
                console.print("  Status      : Needs extra")
                if r["enable"]:
                    console.print(f"  Enable      : {_escape(r['enable'])}")
            console.print(f"  Module      : {r['module']}")
            entry = r["entry_point"] or ""
            if entry:
                console.print(f"  Entry point : {entry}")

        def show_plugins(title: str, plugin_list: list[str], plugin_type: str):
            """Show plugins in a formatted table without instantiation."""
            if not plugin_list and plugin_type != "adapters":
                console.print(f"[yellow]No {title.lower()} plugins found[/yellow]")
                return

            if plugin_type == "adapters":
                if explain:
                    _explain_adapter(explain)
                    return
                rows = _gather_adapter_rows()
                rows = _filter_only(rows)
                if hide_unsupported:
                    rows = [r for r in rows if r.get("status") != "unsupported"]
                if json_out:
                    _print_adapters_json(rows)
                elif verbose:
                    _print_adapters_verbose(rows)
                else:
                    _print_adapters_compact(rows)
                return

            # Guards/Edits: compact/verbose/json/explain
            if plugin_type in {"guards", "edits"}:
                if explain:
                    _explain_generic(explain, plugin_type)
                    return
                rows = _gather_generic_rows(plugin_type)
                rows = _filter_only(rows)
                if json_out:
                    _print_generic_json(rows, plugin_type)
                elif verbose:
                    _print_generic_verbose(rows, title)
                else:
                    _print_generic_compact(rows, title)
                return

            _render_dataset_table(title, plugin_list, verbose=verbose)

        # Show specific category or all
        if category == "guards":
            show_plugins("Guard Plugins", registry.list_guards(), "guards")
        elif category == "edits":
            show_plugins("Edit Plugins", registry.list_edits(), "edits")
        elif category == "adapters":
            # Wrap show to apply hide_unsupported for adapters
            # Inline minimal adapter rendering with filtering/JSON/explain
            rows = _gather_adapter_rows()
            rows = _filter_only(rows)
            if hide_unsupported:
                rows = [r for r in rows if r.get("status") != "unsupported"]
            if json_out:
                _print_adapters_json(rows)
            elif explain:
                _explain_adapter(explain)
            elif verbose:
                _print_adapters_verbose(rows)
            else:
                _print_adapters_compact(rows)
        elif category == "datasets":
            # Show dataset providers
            providers = sorted(list_providers())

            if json_out:
                try:
                    _data_mod = importlib.import_module("invarlock.eval.data")
                    _providers_map = getattr(_data_mod, "_PROVIDERS", {}) or {}
                except Exception:
                    _providers_map = {}
                _emit_plugins_json(
                    "datasets",
                    dataset_inventory_json_items(providers, _providers_map),
                )
            else:
                show_plugins("Dataset Providers", providers, "datasets")
        elif category is None or category in ["list", "all"]:
            # Show all categories
            show_plugins("Guard Plugins", registry.list_guards(), "guards")
            show_plugins("Edit Plugins", registry.list_edits(), "edits")
            show_plugins("Adapter Plugins", registry.list_adapters(), "adapters")

            # Dataset providers
            providers = sorted(list_providers())
            if providers:
                show_plugins("Dataset Providers", providers, "datasets")
        elif category == "plugins":
            if json_out:
                # Combine adapters + guards + edits
                ad_rows = _gather_adapter_rows()
                g_rows = _gather_generic_rows("guards")
                e_rows = _gather_generic_rows("edits")
                _emit_plugins_json(
                    "plugins",
                    combined_plugins_json_items(
                        adapter_rows=ad_rows,
                        guard_rows=g_rows,
                        edit_rows=e_rows,
                    ),
                )
            else:
                # Fallback: print three tables
                show_plugins("Adapter Plugins", registry.list_adapters(), "adapters")
                show_plugins("Guard Plugins", registry.list_guards(), "guards")
                show_plugins("Edit Plugins", registry.list_edits(), "edits")
        else:
            console.print(
                f"[red]❌ Unknown category '{category}'. Valid: guards, edits, adapters, datasets, list, all[/red]"
            )
            raise typer.Exit(2)

    except typer.Exit:
        # Propagate intended exit codes from command flow
        raise
    except Exception as e:
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
        # Adapter plugins (baked-in only)
        "hf_causal": {"packages": ["transformers"], "extra": "invarlock[adapters]"},
        "hf_mlm": {"packages": ["transformers"], "extra": "invarlock[adapters]"},
        "hf_seq2seq": {"packages": ["transformers"], "extra": "invarlock[adapters]"},
        "hf_auto": {"packages": ["transformers"], "extra": "invarlock[adapters]"},
        # Optional adapter plugins
        "hf_gptq": {"packages": ["auto_gptq"], "extra": "invarlock[gptq]"},
        # `autoawq` installs the `awq` import name (not `autoawq`) in practice.
        "hf_awq": {"packages": ["awq"], "extra": "invarlock[awq]"},
        "hf_bnb": {"packages": ["bitsandbytes"], "extra": "invarlock[gpu]"},
    }

    plugin_info = extras_map.get(plugin_name)
    if not plugin_info or not plugin_info["packages"]:
        return ""  # No extra dependencies needed

    # Check each required package. For most packages we use a light import so
    # tests can monkeypatch __import__. For GPU-only stacks (bitsandbytes) and
    # packages with noisy import-time warnings (awq), we probe presence via
    # importlib.util.find_spec instead of importing.
    missing_packages: list[str] = []
    for pkg in plugin_info["packages"]:
        try:
            if pkg == "bitsandbytes":
                if not bitsandbytes_runtime_available():
                    raise ImportError("bitsandbytes not importable")
            elif pkg == "awq":
                import importlib.util as _util

                spec = _util.find_spec(pkg)
                if spec is None:
                    raise ImportError("awq not importable")
            else:
                __import__(pkg)
        except Exception:
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
        None, "--only", help="Filter: missing|ready|core|optional"
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
        None, "--only", help="Filter: missing|ready|core|optional"
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
        None, "--only", help="Filter: missing|ready|core|optional"
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
