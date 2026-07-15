"""
Rendering and category dispatch helpers for the plugins CLI command.
"""

import json
from collections.abc import Callable
from typing import Any, Literal, cast

import typer
from rich.markup import escape as _escape
from rich.table import Table

from invarlock.core.plugins_inventory import (
    PluginCategory,
    adapter_inventory_json_items,
    combined_plugins_json_items,
    dataset_inventory_json_items,
    filter_inventory_rows,
    gather_adapter_inventory_rows,
    gather_generic_inventory_rows,
    generic_inventory_json_items,
)
from invarlock.public_contracts import (
    adapter_capability,
    contract_catalog,
    load_model_family_catalog,
    load_support_matrix,
)

from ..constants import PLUGINS_FORMAT_VERSION

ExtrasChecker = Callable[[str, str], str]
AdapterRowsLoader = Callable[[Any], list[dict[str, Any]]]
GenericPluginCategory = Literal["guards", "edits"]
GenericRowsLoader = Callable[[Any, GenericPluginCategory], list[dict[str, Any]]]
ProviderRegistryLoader = Callable[[], dict[str, Any]]


def _sort_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Stable sort by name, kind, then module and entry_point."""
    return sorted(
        rows,
        key=lambda row: (
            str(row.get("name", "")).lower(),
            str(row.get("kind", "")).lower(),
            str(row.get("module", "")).lower(),
            str(row.get("entry_point", "")).lower(),
        ),
    )


def _emit_plugins_json(
    category: str, rows: list[dict[str, Any]], extra: dict[str, Any] | None = None
) -> None:
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
    typer.echo(json.dumps(payload, ensure_ascii=False, allow_nan=False))


def _all_plugins_json_items(
    *,
    registry: Any,
    providers: list[str],
    only: str | None,
    hide_unsupported: bool,
    adapter_rows_loader: AdapterRowsLoader,
    generic_rows_loader: GenericRowsLoader,
    provider_registry_loader: ProviderRegistryLoader,
) -> list[dict[str, Any]]:
    adapter_rows = _filter_only_rows(adapter_rows_loader(registry), only)
    if hide_unsupported:
        adapter_rows = [
            row for row in adapter_rows if row.get("status") != "unsupported"
        ]
    guard_rows = _filter_only_rows(generic_rows_loader(registry, "guards"), only)
    edit_rows = _filter_only_rows(generic_rows_loader(registry, "edits"), only)
    dataset_rows = [
        {
            **row,
            "kind": "dataset",
            "entry_point": None,
        }
        for row in dataset_inventory_json_items(providers, provider_registry_loader())
    ]
    return [
        *combined_plugins_json_items(
            adapter_rows=adapter_rows,
            guard_rows=guard_rows,
            edit_rows=edit_rows,
        ),
        *dataset_rows,
    ]


def _filter_only_rows(
    rows: list[dict[str, Any]], only: str | None
) -> list[dict[str, Any]]:
    return filter_inventory_rows(rows, only)


def _fmt_backend(backend: str | None, version: str | None) -> tuple[str, str]:
    name = backend or "—"
    if backend and version:
        return backend, f"=={version}"
    return name, "—"


def gather_adapter_rows(
    registry: Any,
    *,
    minimal: bool,
    has_cuda: bool,
    is_linux: bool,
    extras_checker: ExtrasChecker,
    bitsandbytes_runtime_available: Callable[[], bool],
) -> list[dict[str, Any]]:
    from invarlock.core.backend_inventory import extract_adapter_provenance

    rows = gather_adapter_inventory_rows(
        registry=registry,
        minimal=minimal,
        has_cuda=has_cuda,
        is_linux=is_linux,
        extras_checker=extras_checker,
        provenance_extractor=extract_adapter_provenance,
        bitsandbytes_runtime_available=bitsandbytes_runtime_available,
    )
    for row in rows:
        row["capability"] = adapter_capability(str(row.get("name") or ""))
    return rows


def gather_generic_rows(
    registry: Any, plugin_type: GenericPluginCategory, *, extras_checker: ExtrasChecker
) -> list[dict[str, Any]]:
    return gather_generic_inventory_rows(
        registry=registry,
        plugin_type=cast(PluginCategory, plugin_type),
        extras_checker=extras_checker,
    )


def _print_adapters_compact(rows: list[dict[str, Any]], *, console: Any) -> None:
    need = sum(1 for row in rows if row["status"] == "needs_extra")
    ready = sum(1 for row in rows if row["status"] == "ready")
    auto = sum(1 for row in rows if row["support"] == "auto")
    unsupported = sum(1 for row in rows if row["status"] == "unsupported")
    title = (
        "Adapters — ready: "
        f"{ready} · auto: {auto} · missing-extras: {need} · unsupported: {unsupported}"
    )
    table = Table(title=title)
    table.add_column("Adapter", style="cyan")
    table.add_column("Origin", style="dim")
    table.add_column("Mode", style="dim")
    table.add_column("Backend", style="magenta")
    table.add_column("Version", style="magenta")
    table.add_column("Status / Action", style="green")
    for idx, row in enumerate(rows):
        backend_disp, version_disp = _fmt_backend(
            row.get("backend"), row.get("backend_version")
        )
        origin_disp = row.get("origin", row.get("support", "")).capitalize()
        mode_disp = "Auto‑matcher" if row.get("mode") == "auto-matcher" else "Adapter"
        if row["support"] == "auto" or row["status"] == "ready":
            status_disp = "Ready"
        elif row["status"] == "needs_extra":
            status_disp = f"Needs extra: {row['enable'] or ''}".rstrip(": ")
        elif row["status"] == "unsupported":
            status_disp = "Unsupported on this platform"
        else:
            status_disp = row["status"]
        next_support = rows[idx + 1]["support"] if idx + 1 < len(rows) else None
        end_section = next_support is not None and next_support != row["support"]
        table.add_row(
            row["name"],
            origin_disp,
            mode_disp,
            backend_disp,
            version_disp,
            _escape(status_disp),
            end_section=end_section,
        )
    console.print(table)


def _print_adapters_verbose(rows: list[dict[str, Any]], *, console: Any) -> None:
    table = Table(title="Adapters (verbose)")
    table.add_column("Adapter", style="cyan")
    table.add_column("Origin", style="dim")
    table.add_column("Mode", style="dim")
    table.add_column("Backend", style="magenta")
    table.add_column("Version", style="magenta")
    table.add_column("Status", style="green")
    table.add_column("Tier", style="dim")
    table.add_column("Module", style="dim")
    table.add_column("Entry Point", style="dim")
    for row in rows:
        backend_disp, version_disp = _fmt_backend(
            row.get("backend"), row.get("backend_version")
        )
        entry = row["entry_point"] or ""
        entry_disp = (
            entry if entry else ("(auto matcher)" if row["support"] == "auto" else "")
        )
        table.add_row(
            row["name"],
            (row.get("origin") or row.get("support") or "").capitalize(),
            ("Auto‑matcher" if row.get("mode") == "auto-matcher" else "Adapter"),
            backend_disp,
            version_disp,
            row["status"].replace("needs_extra", "Needs extra").capitalize(),
            str(row.get("support_tier") or ""),
            row["module"],
            entry_disp,
        )
    console.print(table)


def _print_adapters_json(rows: list[dict[str, Any]]) -> None:
    _emit_plugins_json("adapters", adapter_inventory_json_items(rows))


def _explain_adapter(name: str, *, rows: list[dict[str, Any]], console: Any) -> None:
    row = next((item for item in rows if item["name"] == name), None)
    if not row:
        console.print(f"[red]❌ Unknown adapter: {name}[/red]")
        raise typer.Exit(1)
    backend_disp = (
        f"{row['backend']} {row['backend_version']}"
        if row["backend"] and row["backend_version"]
        else (f"{row['backend']} (missing)" if row["backend"] else "-")
    )
    console.print(f"[bold]{row['name']}[/bold]")
    console.print(f"  Support     : {row['support'].capitalize()}")
    console.print(f"  Tier        : {row.get('support_tier') or '-'}")
    console.print(
        "  Strict OK   : " + ("yes" if row.get("strict_assurance_allowed") else "no")
    )
    console.print("  Deploys     : " + ("yes" if row.get("deployment_claim") else "no"))
    console.print(f"  Backend     : {backend_disp}")
    if row["support"] == "auto":
        console.print("  Status      : Ready (auto matcher)")
    elif row["status"] == "ready":
        console.print("  Status      : Ready")
    elif row["status"] == "needs_extra":
        console.print("  Status      : Needs extra")
        if row["enable"]:
            console.print(f"  Enable      : {_escape(row['enable'])}")
    elif row["status"] == "partial":
        console.print("  Status      : Partial (GPU-only features disabled)")
    if row["name"] == "hf_gptq":
        console.print("  Matches     : GPTQModel-compatible GPTQ HF repos")
        console.print(
            "  Notes       : Uses GPTQModel; GPU recommended for quantized inference"
        )
    elif row["name"] == "hf_awq":
        console.print("  Matches     : AWQ-quantized HF repos")
        console.print(
            "  Notes       : Uses Transformers AWQ through GPTQModel; GPU recommended"
        )
    elif row["name"] == "hf_bnb":
        console.print("  Matches     : Transformers 4/8-bit loading with bitsandbytes")
        console.print(
            "  Notes       : GPU recommended; falls back to metadata only on CPU"
        )
    elif row["name"] == "hf_torchao":
        console.print("  Matches     : Hugging Face causal LMs quantized with torchao")
        console.print(
            "  Notes       : Runtime applies torchao int8 weight-only quantization"
        )
    elif row["name"] == "hf_hqq":
        console.print("  Matches     : Hugging Face causal LMs quantized with HQQ")
        console.print(
            "  Notes       : Runtime applies HQQ quantization through Transformers"
        )
    elif row["name"] == "hf_quanto":
        console.print("  Matches     : Hugging Face causal LMs quantized with Quanto")
        console.print("  Notes       : Runtime applies Quanto weight-only quantization")
    elif row["name"] == "hf_ct":
        console.print("  Matches     : HF repos using compressed-tensors checkpoints")
        console.print("  Notes       : Loads pre-quantized compressed-tensors subjects")
    else:
        console.print("  Matches     : Hugging Face Transformers (core adapters)")
    console.print(f"  Module      : {row['module']}")
    entry = row["entry_point"] or ""
    if entry:
        console.print(f"  Entry point : {entry}")


def _print_generic_compact(
    rows: list[dict[str, Any]], title: str, *, console: Any
) -> None:
    need = sum(1 for row in rows if row["status"] == "needs_extra")
    ready = sum(1 for row in rows if row["status"] == "ready")
    table = Table(title=f"{title} — ready: {ready} · missing-extras: {need}")
    table.add_column("Name", style="cyan")
    table.add_column("Origin", style="dim")
    table.add_column("Mode", style="dim")
    table.add_column("Backend", style="magenta")
    table.add_column("Version", style="magenta")
    table.add_column("Status / Action", style="green")
    for idx, row in enumerate(rows):
        backend_disp, version_disp = _fmt_backend(
            row.get("backend"), row.get("backend_version")
        )
        origin_disp = row.get("origin", row.get("support", "")).capitalize()
        mode_disp = "Guard" if row.get("mode") == "guard" else "Edit"
        if row["status"] == "ready":
            status_disp = "Ready"
        elif row["status"] == "needs_extra":
            status_disp = f"Needs extra: {row['enable'] or ''}".rstrip(": ")
        else:
            status_disp = row["status"]
        next_support = rows[idx + 1]["support"] if idx + 1 < len(rows) else None
        end_section = next_support is not None and next_support != row["support"]
        table.add_row(
            row["name"],
            origin_disp,
            mode_disp,
            backend_disp,
            version_disp,
            _escape(status_disp),
            end_section=end_section,
        )
    console.print(table)


def _print_generic_verbose(
    rows: list[dict[str, Any]], title: str, *, console: Any
) -> None:
    table = Table(title=f"{title} (verbose)")
    table.add_column("Name", style="cyan")
    table.add_column("Origin", style="dim")
    table.add_column("Mode", style="dim")
    table.add_column("Backend", style="magenta")
    table.add_column("Version", style="magenta")
    table.add_column("Status", style="green")
    table.add_column("Tier", style="dim")
    table.add_column("Module", style="dim")
    table.add_column("Entry Point", style="dim")
    for idx, row in enumerate(rows):
        entry = row["entry_point"] or ""
        next_support = rows[idx + 1]["support"] if idx + 1 < len(rows) else None
        end_section = next_support is not None and next_support != row["support"]
        backend_disp, version_disp = _fmt_backend(
            row.get("backend"), row.get("backend_version")
        )
        table.add_row(
            row["name"],
            (row.get("origin") or row.get("support") or "").capitalize(),
            ("Guard" if row.get("mode") == "guard" else "Edit"),
            backend_disp,
            version_disp,
            row["status"].replace("needs_extra", "Needs extra").capitalize(),
            str(row.get("support_tier") or ""),
            row["module"],
            entry,
            end_section=end_section,
        )
    console.print(table)


def _print_generic_json(
    rows: list[dict[str, Any]], kind: GenericPluginCategory
) -> None:
    _emit_plugins_json(kind, generic_inventory_json_items(rows, kind=kind))


def _render_dataset_table(
    title: str,
    providers: list[str],
    *,
    verbose: bool = False,
    console: Any,
    provider_registry_map: dict[str, Any],
) -> None:
    from invarlock.cli.constants import PROVIDER_KIND as provider_kind
    from invarlock.cli.constants import PROVIDER_NETWORK as provider_network
    from invarlock.cli.constants import PROVIDER_PARAMS as provider_params

    def _net_label(name: str) -> str:
        val = (provider_network.get(name, "") or "").lower()
        if val == "cache":
            return "Cache/Net"
        if val == "yes":
            return "Yes"
        if val == "no":
            return "No"
        return "Unknown"

    rows: list[dict[str, str]] = []
    for provider_name in providers:
        rows.append(
            {
                "name": provider_name,
                "network": _net_label(provider_name),
                "kind": provider_kind.get(provider_name, "-"),
                "module": getattr(
                    provider_registry_map.get(provider_name, None),
                    "__module__",
                    "unknown",
                ),
            }
        )
    net_order = {"No": 0, "Cache/Net": 1, "Yes": 2, "Unknown": 3}
    rows.sort(key=lambda row: (net_order.get(row["network"], 9), row["name"]))

    cnt_no = sum(1 for row in rows if row["network"] == "No")
    cnt_cache = sum(1 for row in rows if row["network"] == "Cache/Net")
    cnt_yes = sum(1 for row in rows if row["network"] == "Yes")
    table = Table(
        title=(
            f"{title} — offline: {cnt_no} · cache/net: {cnt_cache} · network: {cnt_yes}"
        )
    )
    table.add_column("Provider", style="cyan")
    table.add_column("Network", style="dim")
    table.add_column("Kind", style="dim")
    table.add_column("Params", style="dim")
    if verbose:
        table.add_column("Module", style="dim")
    table.add_column("Status / Action", style="green")

    for idx, row in enumerate(rows):
        end_section = idx + 1 < len(rows) and rows[idx + 1]["network"] != row["network"]
        cols = [
            row["name"],
            row["network"],
            row.get("kind", "-"),
            provider_params.get(row["name"], "-"),
        ]
        if verbose:
            cols.append(row["module"])
        cols.append("✓ Available")
        table.add_row(*cols, end_section=end_section)

    console.print(table)


def _explain_generic(
    name: str,
    plugin_type: str,
    *,
    rows: list[dict[str, Any]],
    console: Any,
) -> None:
    row = next((item for item in rows if item["name"] == name), None)
    if not row:
        console.print(f"[red]❌ Unknown {plugin_type[:-1]}: {name}[/red]")
        raise typer.Exit(1)
    console.print(f"[bold]{row['name']}[/bold]")
    console.print(f"  Support     : {row['support'].capitalize()}")
    console.print(f"  Tier        : {row.get('support_tier') or '-'}")
    console.print(
        "  Strict OK   : " + ("yes" if row.get("strict_assurance_allowed") else "no")
    )
    console.print("  Deploys     : " + ("yes" if row.get("deployment_claim") else "no"))
    backend_label = row.get("backend") or "—"
    console.print(f"  Backend     : {backend_label}")
    if row["status"] == "ready":
        console.print("  Status      : Ready")
    elif row["status"] == "needs_extra":
        console.print("  Status      : Needs extra")
        if row["enable"]:
            console.print(f"  Enable      : {_escape(row['enable'])}")
    console.print(f"  Module      : {row['module']}")
    entry = row["entry_point"] or ""
    if entry:
        console.print(f"  Entry point : {entry}")


def _show_plugin_category(
    title: str,
    plugin_list: list[str],
    plugin_type: str,
    *,
    registry: Any,
    only: str | None,
    verbose: bool,
    json_out: bool,
    explain: str | None,
    hide_unsupported: bool,
    console: Any,
    adapter_rows_loader: AdapterRowsLoader,
    generic_rows_loader: GenericRowsLoader,
    provider_registry_loader: ProviderRegistryLoader,
) -> None:
    if not plugin_list and plugin_type != "adapters":
        console.print(f"[yellow]No {title.lower()} plugins found[/yellow]")
        return

    if plugin_type == "adapters":
        all_rows = adapter_rows_loader(registry)
        if explain:
            _explain_adapter(explain, rows=all_rows, console=console)
            return
        rows = _filter_only_rows(all_rows, only)
        if hide_unsupported:
            rows = [row for row in rows if row.get("status") != "unsupported"]
        if json_out:
            _print_adapters_json(rows)
        elif verbose:
            _print_adapters_verbose(rows, console=console)
        else:
            _print_adapters_compact(rows, console=console)
        return

    if plugin_type in {"guards", "edits"}:
        generic_type = cast(GenericPluginCategory, plugin_type)
        rows = generic_rows_loader(registry, generic_type)
        if explain:
            _explain_generic(explain, generic_type, rows=rows, console=console)
            return
        rows = _filter_only_rows(rows, only)
        if json_out:
            _print_generic_json(rows, generic_type)
        elif verbose:
            _print_generic_verbose(rows, title, console=console)
        else:
            _print_generic_compact(rows, title, console=console)
        return

    _render_dataset_table(
        title,
        plugin_list,
        verbose=verbose,
        console=console,
        provider_registry_map=provider_registry_loader(),
    )


def handle_plugins_category(
    *,
    category: str | None,
    registry: Any,
    list_providers_fn: Callable[[], list[str]],
    only: str | None,
    verbose: bool,
    json_out: bool,
    explain: str | None,
    hide_unsupported: bool,
    console: Any,
    adapter_rows_loader: AdapterRowsLoader,
    generic_rows_loader: GenericRowsLoader,
    provider_registry_loader: ProviderRegistryLoader,
) -> None:
    if category == "guards":
        _show_plugin_category(
            "Guard Plugins",
            registry.list_guards(),
            "guards",
            registry=registry,
            only=only,
            verbose=verbose,
            json_out=json_out,
            explain=explain,
            hide_unsupported=hide_unsupported,
            console=console,
            adapter_rows_loader=adapter_rows_loader,
            generic_rows_loader=generic_rows_loader,
            provider_registry_loader=provider_registry_loader,
        )
        return
    if category == "edits":
        _show_plugin_category(
            "Edit Plugins",
            registry.list_edits(),
            "edits",
            registry=registry,
            only=only,
            verbose=verbose,
            json_out=json_out,
            explain=explain,
            hide_unsupported=hide_unsupported,
            console=console,
            adapter_rows_loader=adapter_rows_loader,
            generic_rows_loader=generic_rows_loader,
            provider_registry_loader=provider_registry_loader,
        )
        return
    if category == "adapters":
        _show_plugin_category(
            "Adapter Plugins",
            registry.list_adapters(),
            "adapters",
            registry=registry,
            only=only,
            verbose=verbose,
            json_out=json_out,
            explain=explain,
            hide_unsupported=hide_unsupported,
            console=console,
            adapter_rows_loader=adapter_rows_loader,
            generic_rows_loader=generic_rows_loader,
            provider_registry_loader=provider_registry_loader,
        )
        return
    if category == "datasets":
        providers = sorted(list_providers_fn())
        if json_out:
            _emit_plugins_json(
                "datasets",
                dataset_inventory_json_items(providers, provider_registry_loader()),
            )
        else:
            _show_plugin_category(
                "Dataset Providers",
                providers,
                "datasets",
                registry=registry,
                only=only,
                verbose=verbose,
                json_out=json_out,
                explain=explain,
                hide_unsupported=hide_unsupported,
                console=console,
                adapter_rows_loader=adapter_rows_loader,
                generic_rows_loader=generic_rows_loader,
                provider_registry_loader=provider_registry_loader,
            )
        return
    if category is None or category in ["list", "all"]:
        if json_out:
            _emit_plugins_json(
                "all",
                _all_plugins_json_items(
                    registry=registry,
                    providers=sorted(list_providers_fn()),
                    only=only,
                    hide_unsupported=hide_unsupported,
                    adapter_rows_loader=adapter_rows_loader,
                    generic_rows_loader=generic_rows_loader,
                    provider_registry_loader=provider_registry_loader,
                ),
            )
            return
        _show_plugin_category(
            "Guard Plugins",
            registry.list_guards(),
            "guards",
            registry=registry,
            only=only,
            verbose=verbose,
            json_out=json_out,
            explain=explain,
            hide_unsupported=hide_unsupported,
            console=console,
            adapter_rows_loader=adapter_rows_loader,
            generic_rows_loader=generic_rows_loader,
            provider_registry_loader=provider_registry_loader,
        )
        _show_plugin_category(
            "Edit Plugins",
            registry.list_edits(),
            "edits",
            registry=registry,
            only=only,
            verbose=verbose,
            json_out=json_out,
            explain=explain,
            hide_unsupported=hide_unsupported,
            console=console,
            adapter_rows_loader=adapter_rows_loader,
            generic_rows_loader=generic_rows_loader,
            provider_registry_loader=provider_registry_loader,
        )
        _show_plugin_category(
            "Adapter Plugins",
            registry.list_adapters(),
            "adapters",
            registry=registry,
            only=only,
            verbose=verbose,
            json_out=json_out,
            explain=explain,
            hide_unsupported=hide_unsupported,
            console=console,
            adapter_rows_loader=adapter_rows_loader,
            generic_rows_loader=generic_rows_loader,
            provider_registry_loader=provider_registry_loader,
        )
        providers = sorted(list_providers_fn())
        if providers:
            _show_plugin_category(
                "Dataset Providers",
                providers,
                "datasets",
                registry=registry,
                only=only,
                verbose=verbose,
                json_out=json_out,
                explain=explain,
                hide_unsupported=hide_unsupported,
                console=console,
                adapter_rows_loader=adapter_rows_loader,
                generic_rows_loader=generic_rows_loader,
                provider_registry_loader=provider_registry_loader,
            )
        return
    if category == "plugins":
        if json_out:
            _emit_plugins_json(
                "plugins",
                combined_plugins_json_items(
                    adapter_rows=adapter_rows_loader(registry),
                    guard_rows=generic_rows_loader(registry, "guards"),
                    edit_rows=generic_rows_loader(registry, "edits"),
                ),
            )
        else:
            _show_plugin_category(
                "Adapter Plugins",
                registry.list_adapters(),
                "adapters",
                registry=registry,
                only=only,
                verbose=verbose,
                json_out=json_out,
                explain=explain,
                hide_unsupported=hide_unsupported,
                console=console,
                adapter_rows_loader=adapter_rows_loader,
                generic_rows_loader=generic_rows_loader,
                provider_registry_loader=provider_registry_loader,
            )
            _show_plugin_category(
                "Guard Plugins",
                registry.list_guards(),
                "guards",
                registry=registry,
                only=only,
                verbose=verbose,
                json_out=json_out,
                explain=explain,
                hide_unsupported=hide_unsupported,
                console=console,
                adapter_rows_loader=adapter_rows_loader,
                generic_rows_loader=generic_rows_loader,
                provider_registry_loader=provider_registry_loader,
            )
            _show_plugin_category(
                "Edit Plugins",
                registry.list_edits(),
                "edits",
                registry=registry,
                only=only,
                verbose=verbose,
                json_out=json_out,
                explain=explain,
                hide_unsupported=hide_unsupported,
                console=console,
                adapter_rows_loader=adapter_rows_loader,
                generic_rows_loader=generic_rows_loader,
                provider_registry_loader=provider_registry_loader,
            )
        return
    console.print(
        f"[red]❌ Unknown category '{category}'. "
        "Valid: guards, edits, adapters, datasets, list, all[/red]"
    )
    raise typer.Exit(2)


__all__ = [
    "gather_adapter_rows",
    "gather_generic_rows",
    "handle_plugins_category",
]
