"""
InvarLock CLI Plugins Command
=========================

Handles the 'invarlock plugins' command for listing available plugins.
Supports a minimal view via INVARLOCK_MINIMAL=1 to hide built‑in adapters.
"""

import json
import os
import platform
import subprocess
import sys

import typer
from rich.console import Console
from rich.markup import escape as _escape
from rich.table import Table

from ..constants import PLUGINS_FORMAT_VERSION

console = Console()

# Group: plugins
plugins_app = typer.Typer(
    help="Manage optional backends; list adapters/guards/edits.",
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


def _emit_plugins_json(category: str, rows) -> None:
    payload = {
        "format_version": PLUGINS_FORMAT_VERSION,
        # Provide both for back-compat with tests expecting 'kind'
        "category": category,
        "kind": category,
        "items": _sort_rows(rows),
    }
    typer.echo(json.dumps(payload, ensure_ascii=False))


def plugins_command(
    category: str | None = typer.Argument(
        None, help="Plugin category (guards|edits|adapters|datasets)"
    ),
    only: str | None = typer.Option(
        None,
        "--only",
        help="Filter adapters (adapters only): missing|ready|core|optional",
    ),
    verbose: bool = typer.Option(
        False, "--verbose", help="Show detailed columns for adapters"
    ),
    json_out: bool = typer.Option(
        False, "--json", help="Emit JSON instead of a table (adapters only)"
    ),
    explain: str | None = typer.Option(
        None, "--explain", help="Explain a specific adapter (adapters only)"
    ),
    hide_unsupported: bool = typer.Option(
        True,
        "--hide-unsupported/--show-unsupported",
        help="Hide adapters unsupported on this platform (default: hide)",
    ),
):
    """
    List available plugins with entry point information.

    Shows plugin names, module paths, and availability status without instantiation.

    Examples:
        invarlock plugins              # List all plugins
        invarlock plugins guards       # List only guard plugins
        invarlock plugins edits        # List only edit plugins
        invarlock plugins adapters     # List only adapter plugins
        invarlock plugins datasets     # List only dataset providers
    """
    try:
        # Light import guard for docs/tests
        disable_discovery = os.getenv(
            "INVARLOCK_DISABLE_PLUGIN_DISCOVERY", ""
        ).strip().lower() in {
            "1",
            "true",
            "yes",
        }
        if disable_discovery:
            # Validate category even when discovery is disabled so unknown
            # categories still report a proper error code for tests/CLI.
            valid_categories = {
                None,
                "guards",
                "edits",
                "adapters",
                "adapter",
                "datasets",
                "list",
                "all",
                "plugins",
            }
            if category not in valid_categories:
                console.print(
                    f"[red]❌ Unknown category '{category}'. Valid: guards, edits, "
                    "adapters, datasets, list, all[/red]"
                )
                raise typer.Exit(2)

            # Provide a minimal but structurally stable JSON envelope for all
            # categories when discovery is disabled. Tests rely on an empty
            # items array in this mode.
            if json_out:
                import json as _json
                import sys as _sys

                kind = (
                    "adapters"
                    if category in {"adapters", "adapter"}
                    else (category or "list")
                )
                payload = {
                    "format_version": PLUGINS_FORMAT_VERSION,
                    "category": kind,
                    "kind": kind,
                    "items": [],
                    "discovery": "disabled",
                }
                _sys.stdout.write(_json.dumps(payload) + "\n")
                return

            # Fall back to a terse message when discovery is disabled
            console.print(
                "[dim]Plugin discovery disabled (INVARLOCK_DISABLE_PLUGIN_DISCOVERY=1).[/dim]"
            )
            return

        from invarlock.core.registry import get_registry
        from invarlock.eval.data import get_provider, list_providers

        registry = get_registry()

        # Coerce Typer OptionInfo defaults when invoked programmatically
        try:
            from typer.models import OptionInfo as _OptionInfo  # type: ignore
        except Exception:  # pragma: no cover

            class _OptionInfo:  # type: ignore
                pass

        if isinstance(only, _OptionInfo):
            only = None
        if isinstance(verbose, _OptionInfo):
            verbose = False
        if isinstance(json_out, _OptionInfo):
            json_out = False
        if isinstance(explain, _OptionInfo):
            explain = None
        if isinstance(hide_unsupported, _OptionInfo):
            hide_unsupported = True

        def _is_minimal() -> bool:
            val = os.environ.get("INVARLOCK_MINIMAL", "").strip().lower()
            return val not in ("", "0", "false", "no")

        def _gather_adapter_rows() -> list[dict]:
            try:
                import torch  # type: ignore

                has_cuda = bool(
                    getattr(torch, "cuda", None) and torch.cuda.is_available()
                )
            except Exception:
                has_cuda = False

            names = registry.list_adapters()
            is_linux = platform.system().lower() == "linux"
            if _is_minimal():
                names = [
                    n
                    for n in names
                    if str(
                        registry.get_plugin_info(n, "adapters").get("module") or ""
                    ).startswith("invarlock.plugins")
                ]

            rows: list[dict] = []
            for n in names:
                info = registry.get_plugin_info(n, "adapters")
                module = str(info.get("module") or "")
                entry = info.get("entry_point")
                # Classify support level independent of origin
                if module.startswith("invarlock.adapters"):
                    if n in {"hf_causal_auto", "hf_mlm_auto"}:
                        support = "auto"
                    elif n in {"hf_onnx"}:
                        # ONNX relies on optional extras (optimum + onnxruntime)
                        support = "optional"
                    else:
                        support = "core"
                else:
                    support = "optional"

                # Origin reflects where it comes from (builtin vs plugin)
                origin = "core" if module.startswith("invarlock.adapters") else "plugin"
                mode = "auto-matcher" if support == "auto" else "adapter"
                backend_name = ""
                backend_version = None
                present = False
                try:
                    from invarlock.cli.provenance import extract_adapter_provenance

                    prov = extract_adapter_provenance(n)
                    backend_name = prov.library or ""
                    backend_version = prov.version
                    present = backend_version is not None
                except Exception:
                    pass
                status = "ready"
                enable = ""
                if support == "auto":
                    status = "ready"
                elif support == "optional" and not present:
                    status = "needs_extra"
                # Platform gating for Linux-only stacks
                if backend_name in {"auto-gptq", "autoawq"} and not is_linux:
                    status = "unsupported"
                    enable = "Linux-only"
                # Extras completeness for certain adapters (e.g., hf_onnx needs optimum + onnxruntime)
                try:
                    extras_status = _check_plugin_extras(n, "adapters")
                except Exception:
                    extras_status = ""
                if (
                    support == "optional"
                    and extras_status.startswith("⚠️")
                    and "missing" in extras_status
                ):
                    status = "needs_extra"
                    # If we have a normalized extra hint (invarlock[...] ), surface as enable action
                    hint = extras_status.split("missing", 1)[-1].strip()
                    if hint:
                        enable = f"pip install '{hint}'"
                if backend_name == "bitsandbytes" and present and not has_cuda:
                    status = "unsupported"
                    enable = "Requires CUDA"
                extra_hint = {
                    "hf_gptq": "invarlock[gptq]",
                    "hf_awq": "invarlock[awq]",
                    "hf_bnb": "invarlock[gpu]",
                }.get(n)
                if status == "needs_extra" and extra_hint:
                    enable = f"pip install '{extra_hint}'"
                rows.append(
                    {
                        "name": n,
                        "backend": backend_name,
                        "backend_version": backend_version,
                        "support": support,
                        "origin": origin,
                        "mode": mode,
                        "status": status,
                        "enable": enable,
                        "module": module,
                        "entry_point": entry,
                    }
                )
            rows.sort(
                key=lambda r: (
                    {"needs_extra": 0, "partial": 1, "ready": 2}.get(r["status"], 3),
                    {"optional": 0, "core": 1, "auto": 2}.get(r["support"], 3),
                    r["name"],
                )
            )
            return rows

        def _filter_only(rows: list[dict]) -> list[dict]:
            if not only:
                return rows
            m = only.strip().lower()
            if m == "missing":
                return [r for r in rows if r["status"] == "needs_extra"]
            if m == "ready":
                return [r for r in rows if r["status"] == "ready"]
            if m == "core":
                return [r for r in rows if r["support"] == "core"]
            if m == "optional":
                return [r for r in rows if r["support"] == "optional"]
            return rows

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
                    status_disp = "🧩 Auto (selects best hf_* adapter)"
                elif r["status"] == "ready":
                    status_disp = "✅ Ready"
                elif r["status"] == "needs_extra":
                    status_disp = f"⛔ Needs extra → {r['enable'] or ''}".rstrip()
                elif r["status"] == "unsupported":
                    if r["backend"] == "bitsandbytes":
                        status_disp = "🚫 Unsupported (requires CUDA)"
                    else:
                        status_disp = "🚫 Unsupported on this platform"
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
            console.print(
                "[dim]Hints: add --only ready|core|optional|auto|unsupported · use --json for scripting · use adapters (plural)[/dim]"
            )

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
            unified = []
            for r in rows:
                backend_name = r.get("backend") or ""
                backend_ver = r.get("backend_version")
                backend_obj = None
                if backend_name:
                    backend_obj = {"name": backend_name}
                    if backend_ver:
                        backend_obj["version"] = backend_ver
                    else:
                        backend_obj["present"] = True
                unified.append(
                    {
                        "name": r.get("name"),
                        "kind": "adapter",
                        "module": r.get("module"),
                        "entry_point": r.get("entry_point"),
                        "origin": "builtin"
                        if str(r.get("module", "")).startswith("invarlock.")
                        else "third_party",
                        "status": r.get("status"),
                        "backend": backend_obj,
                    }
                )
            _emit_plugins_json("adapters", unified)

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
                console.print("  Status      : 🧩 Ready (matcher)")
            elif r["status"] == "ready":
                console.print("  Status      : ✅ Ready")
            elif r["status"] == "needs_extra":
                console.print("  Status      : ⛔ Needs extra")
                if r["enable"]:
                    console.print(f"  Enable      : {_escape(r['enable'])}")
            elif r["status"] == "partial":
                console.print("  Status      : ⚠️ Partial (GPU-only features disabled)")
            if r["name"] == "hf_gptq":
                console.print(
                    "  Matches     : AutoGPTQ-quantized HF repos (from_quantized)"
                )
                console.print(
                    "  Notes       : GPU recommended; metadata ingestion on CPU"
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
            names = (
                registry.list_guards()
                if plugin_type == "guards"
                else registry.list_edits()
            )
            rows: list[dict] = []
            for n in names:
                info = registry.get_plugin_info(n, plugin_type)
                module = str(info.get("module") or "")
                entry = info.get("entry_point")
                support = (
                    "core"
                    if module.startswith(f"invarlock.{plugin_type}")
                    else "optional"
                )
                origin = "core" if support == "core" else "plugin"
                mode = "guard" if plugin_type == "guards" else "edit"
                extras_status = _check_plugin_extras(n, plugin_type)
                status = "ready"
                enable = ""
                if extras_status.startswith("⚠️") and "missing" in extras_status:
                    status = "needs_extra"
                    hint = extras_status.split("missing", 1)[-1].strip()
                    if hint:
                        enable = f"pip install '{hint}'"
                rows.append(
                    {
                        "name": n,
                        "backend": None,
                        "backend_version": None,
                        "support": support,
                        "origin": origin,
                        "mode": mode,
                        "status": status,
                        "enable": enable,
                        "module": module,
                        "entry_point": entry,
                    }
                )
            # Sort with support first (core → optional), then status, then name
            rows.sort(
                key=lambda r: (
                    {"core": 0, "optional": 1}.get(r["support"], 2),
                    {"needs_extra": 0, "ready": 1}.get(r["status"], 2),
                    r["name"],
                )
            )
            return rows

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
                    status_disp = "✅ Ready"
                elif r["status"] == "needs_extra":
                    status_disp = f"⛔ Needs extra → {r['enable'] or ''}".rstrip()
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
            unified: list[dict] = []
            for r in rows:
                unified.append(
                    {
                        "name": r.get("name"),
                        "kind": "guard" if kind == "guards" else "edit",
                        "module": r.get("module"),
                        "entry_point": r.get("entry_point"),
                        "origin": "builtin"
                        if str(r.get("module", "")).startswith("invarlock.")
                        else "third_party",
                    }
                )
            _emit_plugins_json(kind, unified)

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
                import invarlock.eval.data as _data_mod  # type: ignore

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
                console.print("  Status      : ✅ Ready")
            elif r["status"] == "needs_extra":
                console.print("  Status      : ⛔ Needs extra")
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
        # Accept singular alias
        if category == "adapter":
            category = "adapters"

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
                items = []
                for provider_name in providers:
                    try:
                        provider = get_provider(provider_name)
                        items.append(
                            {
                                "name": provider_name,
                                "module": provider.__class__.__module__,
                                "status": "available",
                            }
                        )
                    except Exception as e:  # pragma: no cover - defensive
                        items.append(
                            {
                                "name": provider_name,
                                "module": "unknown",
                                "status": f"error: {e}",
                            }
                        )
                _emit_plugins_json("datasets", items)
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
                adapters_unified: list[dict] = []
                for r in ad_rows:
                    backend_name = r.get("backend") or ""
                    backend_ver = r.get("backend_version")
                    backend_obj = None
                    if backend_name:
                        backend_obj = {"name": backend_name}
                        if backend_ver:
                            backend_obj["version"] = backend_ver
                        else:
                            backend_obj["present"] = True
                    adapters_unified.append(
                        {
                            "name": r.get("name"),
                            "kind": "adapter",
                            "module": r.get("module"),
                            "entry_point": r.get("entry_point"),
                            "origin": "builtin"
                            if str(r.get("module", "")).startswith("invarlock.")
                            else "third_party",
                            "backend": backend_obj,
                        }
                    )
                g_rows = _gather_generic_rows("guards")
                guards_unified = [
                    {
                        "name": r.get("name"),
                        "kind": "guard",
                        "module": r.get("module"),
                        "entry_point": r.get("entry_point"),
                        "origin": "builtin"
                        if str(r.get("module", "")).startswith("invarlock.")
                        else "third_party",
                    }
                    for r in g_rows
                ]
                e_rows = _gather_generic_rows("edits")
                edits_unified = [
                    {
                        "name": r.get("name"),
                        "kind": "edit",
                        "module": r.get("module"),
                        "entry_point": r.get("entry_point"),
                        "origin": "builtin"
                        if str(r.get("module", "")).startswith("invarlock.")
                        else "third_party",
                    }
                    for r in e_rows
                ]
                _emit_plugins_json(
                    "plugins", adapters_unified + guards_unified + edits_unified
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
        "hf_gpt2": {"packages": ["transformers"], "extra": "invarlock[adapters]"},
        "hf_bert": {"packages": ["transformers"], "extra": "invarlock[adapters]"},
        "hf_llama": {"packages": ["transformers"], "extra": "invarlock[adapters]"},
        "hf_onnx": {"packages": ["optimum", "onnxruntime"], "extra": "invarlock[onnx]"},
        # Optional adapter plugins
        "hf_gptq": {"packages": ["auto_gptq"], "extra": "invarlock[gptq]"},
        "hf_awq": {"packages": ["autoawq"], "extra": "invarlock[awq]"},
        "hf_bnb": {"packages": ["bitsandbytes"], "extra": "invarlock[gpu]"},
    }

    plugin_info = extras_map.get(plugin_name)
    if not plugin_info or not plugin_info["packages"]:
        return ""  # No extra dependencies needed

    # Check each required package. For most packages we use a light import so
    # tests can monkeypatch __import__; for GPU-only stacks like bitsandbytes
    # we only probe presence via importlib.util.find_spec to avoid crashing on
    # CPU-only builds during simple listing.
    missing_packages: list[str] = []
    for pkg in plugin_info["packages"]:
        try:
            if pkg == "bitsandbytes":
                import importlib.util as _util

                spec = _util.find_spec(pkg)
                if spec is None:
                    raise ImportError("bitsandbytes not importable")
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


def list_edits_command():
    """List available edit plugins."""
    plugins_command("edits")


def list_guards_command():
    """List available guard plugins."""
    plugins_command("guards")


def _resolve_uninstall_targets(target: str) -> list[str]:
    """Map a user-provided plugin/extra name to underlying pip packages.

    Supports aliases such as:
    - gptq / hf_gptq / auto-gptq -> ["auto-gptq"]
    - awq / hf_awq / autoawq     -> ["autoawq"]
    - bnb / hf_bnb / gpu         -> ["bitsandbytes"]
    - invarlock[awq] / invarlock[gptq] / invarlock[gpu] -> respective packages
    """
    name = (target or "").strip().lower()
    if name.startswith("invarlock[") and name.endswith("]"):
        name = name[len("invarlock[") : -1]
    # Normalize separators
    name = name.replace("-", "_")
    mapping: dict[str, list[str]] = {
        # GPTQ family
        "gptq": ["auto-gptq"],
        "hf_gptq": ["auto-gptq"],
        "auto_gptq": ["auto-gptq"],
        "auto-gptq": ["auto-gptq"],
        # AWQ family
        "awq": ["autoawq"],
        "hf_awq": ["autoawq"],
        "autoawq": ["autoawq"],
        # bitsandbytes / GPU extra
        "bnb": ["bitsandbytes"],
        "hf_bnb": ["bitsandbytes"],
        "gpu": ["bitsandbytes"],
        "bitsandbytes": ["bitsandbytes"],
        # ONNX/Optimum family
        "onnx": ["onnxruntime"],
        "hf_onnx": ["onnxruntime"],
        "optimum": ["optimum"],
    }
    return mapping.get(name, [])


def _resolve_install_targets(target: str) -> list[str]:
    """Map a user-provided plugin/extra name to `pip install` targets.

    Prefer installing via `invarlock[extra]` to keep pins/markers consistent.

    Supported aliases:
    - gptq / hf_gptq / auto-gptq -> ["invarlock[gptq]"]
    - awq / hf_awq / autoawq     -> ["invarlock[awq]"]
    - bnb / hf_bnb / gpu         -> ["invarlock[gpu]"]
    - adapters / transformers     -> ["invarlock[adapters]"]
    - direct package names pass through: auto-gptq, autoawq, bitsandbytes
    """
    name = (target or "").strip().lower()
    if name.startswith("invarlock[") and name.endswith("]"):
        return [name]
    name = name.replace("-", "_")

    mapping: dict[str, list[str]] = {
        # Extras (preferred)
        "gptq": ["invarlock[gptq]"],
        "hf_gptq": ["invarlock[gptq]"],
        "auto_gptq": ["invarlock[gptq]"],
        "auto-gptq": ["invarlock[gptq]"],
        "awq": ["invarlock[awq]"],
        "hf_awq": ["invarlock[awq]"],
        "autoawq": ["invarlock[awq]"],
        "bnb": ["invarlock[gpu]"],
        "hf_bnb": ["invarlock[gpu]"],
        "gpu": ["invarlock[gpu]"],
        "adapters": ["invarlock[adapters]"],
        "transformers": ["invarlock[adapters]"],
        # ONNX/Optimum
        "onnx": ["invarlock[onnx]"],
        "hf_onnx": ["invarlock[onnx]"],
        "optimum": ["invarlock[onnx]"],
        # Direct packages passthrough
        "bitsandbytes": ["bitsandbytes"],
    }
    return mapping.get(name, [])


def plugins_uninstall_command(
    names: list[str] = typer.Argument(
        ...,
        help="One or more plugin extras or adapter names (e.g., gptq awq gpu hf_bnb)",
    ),
    yes: bool = typer.Option(
        False, "--yes", "-y", help="Proceed without interactive confirmation"
    ),
    dry_run: bool = typer.Option(
        True, "--dry-run", help="Show what would be uninstalled without making changes"
    ),
    apply: bool = typer.Option(False, "--apply", help="Actually uninstall packages"),
):
    """Uninstall optional plugin backends (mirror to extras install).

    Examples:
        invarlock plugins-uninstall gptq
        invarlock plugins-uninstall awq --dry-run
        invarlock plugins-uninstall hf_bnb -y
        invarlock plugins-uninstall 'invarlock[gpu]'
    """
    all_pkgs: list[str] = []
    unknown: list[str] = []
    for name in names:
        pkgs = _resolve_uninstall_targets(name)
        if not pkgs:
            unknown.append(name)
        else:
            for p in pkgs:
                if p not in all_pkgs:
                    all_pkgs.append(p)

    def _print_normalized(action: str, pkgs: list[str], mode: str, result: str) -> None:
        console.print(f"Action: {action}")
        console.print(f"Package: {_escape(' '.join(pkgs))}")
        console.print(f"Mode: {mode}")
        console.print(f"Result: {result}")

    if unknown:
        _print_normalized(
            "uninstall",
            [", ".join(unknown)],
            "dry-run" if (not bool(apply)) or bool(dry_run) else "apply",
            "not-found",
        )
        raise typer.Exit(1)

    force_dry = str(
        os.environ.get("INVARLOCK_PLUGINS_DRY_RUN", "")
    ).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    # Treat non-apply invocations as dry-run to avoid interactive prompts in tests/automation.
    # This also honors explicit --dry-run or INVARLOCK_PLUGINS_DRY_RUN.
    if (not bool(apply)) or bool(dry_run) or force_dry:
        _print_normalized("uninstall", all_pkgs, "dry-run", "ok")
        raise typer.Exit(0)

    if not yes:
        proceed = typer.confirm(
            f"Uninstall the following packages from the current environment? {all_pkgs}"
        )
        if not proceed:
            _print_normalized("uninstall", all_pkgs, "apply", "skipped")
            raise typer.Exit(0)

    try:
        cmd = [sys.executable, "-m", "pip", "uninstall", "-y", *all_pkgs]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            _print_normalized("uninstall", all_pkgs, "apply", "skipped")
            raise typer.Exit(result.returncode)
        _print_normalized("uninstall", all_pkgs, "apply", "ok")
    except Exception as e:
        console.print(f"[red]❌ Uninstall failed: {e}[/red]")
        raise typer.Exit(1) from e


def plugins_install_command(
    names: list[str] = typer.Argument(
        ...,
        help="One or more plugin extras or adapter names (e.g., gptq awq gpu hf_bnb adapters)",
    ),
    upgrade: bool = typer.Option(
        False, "--upgrade", "-U", help="Pass --upgrade to pip"
    ),
    dry_run: bool = typer.Option(
        True, "--dry-run", help="Show what would be installed without making changes"
    ),
    apply: bool = typer.Option(False, "--apply", help="Actually install packages"),
):
    """Install optional plugin extras (mirror to extras install).

    Examples:
        invarlock plugins-install gptq
        invarlock plugins-install awq gpu --upgrade
        invarlock plugins-install hf_bnb --dry-run
        invarlock plugins-install adapters
    """
    all_targets: list[str] = []
    unknown: list[str] = []
    for name in names:
        targets = _resolve_install_targets(name)
        if not targets:
            unknown.append(name)
        else:
            for t in targets:
                if t not in all_targets:
                    all_targets.append(t)

    def _print_normalized(action: str, pkgs: list[str], mode: str, result: str) -> None:
        console.print(f"Action: {action}")
        console.print(f"Package: {_escape(' '.join(pkgs))}")
        console.print(f"Mode: {mode}")
        console.print(f"Result: {result}")

    if unknown:
        _print_normalized(
            "install",
            [", ".join(unknown)],
            "dry-run" if dry_run and not apply else "apply",
            "not-found",
        )
        raise typer.Exit(1)

    force_dry = str(
        os.environ.get("INVARLOCK_PLUGINS_DRY_RUN", "")
    ).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    # Treat any non-apply invocation as a dry-run to avoid accidental changes
    # in CI/docs. This also honors explicit --dry-run or INVARLOCK_PLUGINS_DRY_RUN.
    if (not bool(apply)) or bool(dry_run) or force_dry:
        _print_normalized("install", all_targets, "dry-run", "ok")
        raise typer.Exit(0)

    try:
        cmd = [sys.executable, "-m", "pip", "install", *all_targets]
        if upgrade:
            cmd.insert(4, "--upgrade")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            _print_normalized("install", all_targets, "apply", "skipped")
            raise typer.Exit(result.returncode)
        _print_normalized("install", all_targets, "apply", "ok")
    except Exception as e:
        console.print(f"[red]❌ Install failed: {e}[/red]")
        raise typer.Exit(1) from e


# Wire subcommands under group
@plugins_app.command("list")
def _plugins_list(
    category: str | None = typer.Argument(
        None, help="Category: adapters|guards|edits|plugins|datasets"
    ),
    json_out: bool = typer.Option(False, "--json", help="Emit JSON output"),
    verbose: bool = typer.Option(False, "--verbose", help="Verbose table output"),
):
    """List installed plugin entry points and adapters for a given category."""
    return plugins_command(category, verbose=verbose, json_out=json_out)


@plugins_app.command("guards")
def _plugins_guards(
    only: str | None = typer.Option(
        None, "--only", help="Filter: missing|ready|core|optional"
    ),
    verbose: bool = typer.Option(False, "--verbose", help="Verbose table output"),
    json_out: bool = typer.Option(False, "--json", help="Emit JSON output"),
):
    """List available guard plugins.

    Shows built-in and third-party guards discovered via entry points.
    Use --json for machine-readable output.
    """
    return plugins_command("guards", only=only, verbose=verbose, json_out=json_out)


@plugins_app.command("edits")
def _plugins_edits(
    only: str | None = typer.Option(
        None, "--only", help="Filter: missing|ready|core|optional"
    ),
    verbose: bool = typer.Option(False, "--verbose", help="Verbose table output"),
    json_out: bool = typer.Option(False, "--json", help="Emit JSON output"),
):
    """List available edit plugins.

    Includes built-in edits like quant_rtn and any discovered third-party edits.
    Use --json for machine-readable output.
    """
    return plugins_command("edits", only=only, verbose=verbose, json_out=json_out)


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
    )


# Back-compat singular alias (hidden)
@plugins_app.command(name="adapter", hidden=True, help="Alias for 'adapters' (hidden).")
def _plugins_adapter_alias(
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
):
    return plugins_command(
        "adapters",
        only=only,
        verbose=verbose,
        json_out=json_out,
        explain=explain,
        hide_unsupported=hide_unsupported,
    )


@plugins_app.command("install")
def _plugins_install(
    names: list[str] = typer.Argument(...),
    upgrade: bool = typer.Option(False, "--upgrade", "-U"),
    dry_run: bool = typer.Option(True, "--dry-run"),
    apply: bool = typer.Option(False, "--apply"),
):
    """Install optional plugin backends via pip.

    Examples:
      invarlock plugins install invarlock[gptq]        # Linux + CUDA only
      invarlock plugins install invarlock[awq]         # Linux + CUDA only
      invarlock plugins install invarlock[gpu]         # bitsandbytes (CUDA-only)
      invarlock plugins install invarlock[onnx]        # Optimum + ONNX Runtime

    Use --dry-run (default) to preview the action; pass --apply to execute.
    """
    # Normalize semantics: when --dry-run is provided (default true), ensure apply stays False
    eff_dry_run = bool(dry_run)
    eff_apply = bool(apply) if not eff_dry_run else False
    if eff_dry_run:
        os.environ["INVARLOCK_PLUGINS_DRY_RUN"] = "1"
    try:
        return plugins_install_command(names, upgrade, eff_dry_run, eff_apply)
    finally:
        if os.environ.get("INVARLOCK_PLUGINS_DRY_RUN") == "1":
            os.environ.pop("INVARLOCK_PLUGINS_DRY_RUN", None)


@plugins_app.command("uninstall")
def _plugins_uninstall(
    names: list[str] = typer.Argument(...),
    yes: bool = typer.Option(False, "--yes", "-y"),
    dry_run: bool = typer.Option(True, "--dry-run"),
    apply: bool = typer.Option(False, "--apply"),
):
    """Uninstall optional plugin backends via pip.

    Accepts either extras (invarlock[gptq], invarlock[awq], invarlock[gpu], invarlock[onnx])
    or direct package names (auto-gptq, autoawq, bitsandbytes, onnxruntime).

    Use --dry-run (default) to preview; pass --apply to execute.
    """
    eff_dry_run = bool(dry_run)
    eff_apply = bool(apply) if not eff_dry_run else False
    if eff_dry_run:
        os.environ["INVARLOCK_PLUGINS_DRY_RUN"] = "1"
    try:
        return plugins_uninstall_command(names, yes, eff_dry_run, eff_apply)
    finally:
        if os.environ.get("INVARLOCK_PLUGINS_DRY_RUN") == "1":
            os.environ.pop("INVARLOCK_PLUGINS_DRY_RUN", None)


__all__ = [
    "plugins_app",
    "plugins_command",
    "plugins_install_command",
    "plugins_uninstall_command",
    "list_guards_command",
    "list_edits_command",
]
