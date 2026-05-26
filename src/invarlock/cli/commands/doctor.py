"""InvarLock doctor CLI command."""

import importlib
import importlib.util
import logging
import os as _os
import platform as _platform
import shutil as _shutil
import sys
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

import invarlock.core.doctor_preflight as _doctor_preflight
from invarlock.core.doctor_findings import (
    DATASET_SPLIT_FALLBACK_WARNING as DATASET_SPLIT_FALLBACK_WARNING,  # re-exported for CLI/tests
)
from invarlock.core.doctor_findings import (
    DoctorAccumulator,
    DoctorFinding,
    build_cross_check_findings,
    build_doctor_result,
    build_split_fallback_findings,
    build_tiny_relax_finding,
    load_explicit_report_input,
)
from invarlock.core.doctor_inventory import (
    build_adapter_inventory_rows,
    build_dataset_inventory_rows,
    build_generic_inventory_rows,
    summarize_inventory_rows,
)
from invarlock.core.doctor_runtime import (
    collect_optional_dependency_facts,
    collect_torch_runtime_facts,
    find_spec_safe,
)
from invarlock.public_contracts import (
    contract_catalog,
    load_adapter_capabilities,
    load_model_family_catalog,
    load_plugin_compatibility,
    load_support_matrix,
)

from .. import output as cli_output
from ..backend_runtime import bitsandbytes_runtime_available
from ..constants import DOCTOR_FORMAT_VERSION
from ..security_helpers import resolve_shell_runtime_security_policy

DETERMINISM_SHARDS_WARNING = _doctor_preflight.DETERMINISM_SHARDS_WARNING
NON_FATAL_EXCEPTIONS = (
    AttributeError,
    TypeError,
    ValueError,
    KeyError,
    RuntimeError,
    OSError,
    ImportError,
    ModuleNotFoundError,
)

console = cli_output.make_console()
LOGGER = logging.getLogger(__name__)


def _doctor_env_flag(name: str, *, environ: dict[str, str] | None = None) -> bool:
    env_map = _os.environ if environ is None else environ
    value = str(env_map.get(name, "")).strip().lower()
    return value in {"1", "true", "yes", "on"}


def _doctor_tiny_relax_enabled(*, environ: dict[str, str] | None = None) -> bool:
    return _doctor_env_flag("INVARLOCK_TINY_RELAX", environ=environ)


def _doctor_third_party_plugins_enabled() -> bool:
    return bool(resolve_shell_runtime_security_policy().allow_third_party_plugins)


def _doctor_output_prefix(severity: str) -> str:
    return {"error": "ERROR:", "warning": "WARNING:"}.get(severity, "NOTE:")


def _doctor_print_message(
    json_out: bool,
    message: str,
    severity: str,
    code: str,
) -> None:
    if json_out:
        return
    typer.echo(f"{_doctor_output_prefix(severity)} {message} [INVARLOCK:{code}]")


def _doctor_add_finding(
    accumulator: DoctorAccumulator,
    json_out: bool,
    code: str,
    severity: str,
    message: str,
    **extra: object,
) -> None:
    accumulator.add(code, severity, message, **extra)
    _doctor_print_message(json_out, message, severity, code)


def _doctor_record_findings(
    accumulator: DoctorAccumulator,
    json_out: bool,
    findings: list[DoctorFinding],
    *,
    mark_error: bool | None = None,
) -> None:
    accumulator.extend(findings, mark_error=mark_error)
    if json_out:
        return
    for finding in findings:
        typer.echo(
            f"{_doctor_output_prefix(finding.severity)} {finding.message} [INVARLOCK:{finding.code}]"
        )


def _doctor_prepare_console(json_out: bool) -> Console:
    global console
    if not json_out:
        return console
    from io import StringIO

    console = Console(file=StringIO())
    return console


def _doctor_load_invarlock_version() -> str:
    try:
        return str(
            getattr(importlib.import_module("invarlock"), "__version__", "unknown")
        )
    except ImportError:
        return "unknown"


def _doctor_print_header(
    json_out: bool, invarlock_version: str, console_obj: Console
) -> None:
    if json_out:
        return
    os_line = f"OS: {_platform.system()} {_platform.release()} ({_platform.machine()})"
    py_line = f"Python: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    console_obj.print("InvarLock Health Check")
    console_obj.print("=" * 50)
    console_obj.print(f"{os_line} · {py_line} · invarlock: {invarlock_version}")


def _doctor_check_core_components(json_out: bool) -> bool:
    try:
        from invarlock.core.registry import get_registry  # noqa: F401
    except ImportError as exc:
        if not json_out:
            cli_output.print_command_event(
                console, "FAIL", f"Core components missing: {exc}"
            )
        return False
    if not json_out:
        cli_output.print_command_event(console, "PASS", "Core components available")
    return True


def _doctor_check_torch_runtime(json_out: bool) -> tuple[bool, bool]:
    try:
        from ..device import get_device_info

        runtime_facts = collect_torch_runtime_facts(
            import_torch_fn=lambda: __import__("torch"),
            get_device_info_fn=get_device_info,
            which_fn=_shutil.which,
        )
    except ImportError:
        if not json_out:
            cli_output.print_command_event(console, "FAIL", "PyTorch not available")
            cli_output.print_command_detail(console, "Install with: pip install torch")
        return False, False

    torch_version = runtime_facts.version
    if not json_out:
        if torch_version:
            cli_output.print_command_event(console, "PASS", f"PyTorch {torch_version}")
        else:
            cli_output.print_command_event(
                console, "WARN", "PyTorch present but version unavailable"
            )

        console.print("\nDevice Information")
    for device_name, info in runtime_facts.device_info.items():
        if device_name == "auto_selected":
            if not json_out:
                console.print(f"  ▶ Auto‑selected device: {info}")
            continue
        if info["available"]:
            if (
                device_name == "cuda"
                and isinstance(info, dict)
                and "device_count" in info
            ):
                if not json_out:
                    console.print(
                        f"  [green]✅ {device_name.upper()}: {info['device_count']} device(s) - {info['device_name']} ({info['memory_total']})[/green]"
                    )
            elif not json_out:
                console.print(f"  [green]✅ {device_name.upper()}: Available[/green]")
        elif not json_out:
            console.print(f"  [dim]❌ {device_name.upper()}: {info['info']}[/dim]")
    if (
        runtime_facts.cuda_toolkit_found is not None
        and runtime_facts.torch_cuda_build is not None
        and runtime_facts.cuda_available is not None
        and not json_out
    ):
        console.print(
            f"  [dim]• CUDA toolkit: {'found' if runtime_facts.cuda_toolkit_found else 'not found'} · "
            f"torch CUDA build: {'yes' if runtime_facts.torch_cuda_build else 'no'} · "
            f"cuda.is_available(): {'true' if runtime_facts.cuda_available else 'false'}[/dim]"
        )
    if runtime_facts.gpu_memory_gb is not None:
        if not json_out:
            console.print(f"\nGPU Memory: {runtime_facts.gpu_memory_gb:.1f} GB total")
        if runtime_facts.gpu_memory_low and not json_out:
            console.print(
                "[yellow]⚠️  Warning: Less than 4GB GPU memory available[/yellow]"
            )
    has_cuda = (
        bool(runtime_facts.cuda_available)
        if runtime_facts.cuda_available is not None
        else False
    )
    return has_cuda, True


def _doctor_load_optional_report_payload(
    path_value: str | None,
) -> dict[str, object] | None:
    if not path_value:
        return None
    try:
        _, payload, _, invalid = load_explicit_report_input(
            path_value,
            label="Report",
            field="report",
        )
    except NON_FATAL_EXCEPTIONS:
        return None
    if invalid:
        return None
    return payload


def _doctor_apply_preflight(
    *,
    config: str | None,
    profile: str | None,
    tier: str | None,
    baseline: str | None,
    json_out: bool,
    accumulator: DoctorAccumulator,
    had_error: bool,
    cfg_metric_kind: str | None,
) -> tuple[bool, str | None]:
    if not config:
        return had_error, cfg_metric_kind
    console.print("\n🧪 Preflight Lints (config)")
    preflight = _doctor_preflight.run_doctor_config_preflight(
        config_path=str(config),
        profile=str(profile) if profile else None,
        tier=str(tier) if tier else None,
        baseline=str(baseline) if baseline else None,
    )
    for line in preflight.lines:
        if not json_out:
            console.print(line)
    _doctor_record_findings(
        accumulator, json_out, list(preflight.findings), mark_error=preflight.had_error
    )
    had_error = had_error or preflight.had_error
    cfg_metric_kind = preflight.metric_kind or cfg_metric_kind
    if preflight.policy_meta is not None:
        global POLICY_META
        POLICY_META = preflight.policy_meta
    return had_error, cfg_metric_kind


def _doctor_apply_baseline_quick_check(
    *,
    config: str | None,
    baseline: str | None,
    baseline_report: str | None,
    json_out: bool,
    accumulator: DoctorAccumulator,
    had_error: bool,
) -> bool:
    if config:
        return had_error
    quick_check_path = baseline or baseline_report
    if not quick_check_path:
        return had_error
    _, baseline_payload, baseline_findings, invalid_baseline = (
        load_explicit_report_input(
            quick_check_path,
            label="Baseline",
            field="baseline" if baseline else "baseline_report",
        )
    )
    split_findings = (
        build_split_fallback_findings(baseline_payload)
        if baseline_payload is not None
        else []
    )
    _doctor_record_findings(
        accumulator,
        json_out,
        list(baseline_findings) + list(split_findings),
        mark_error=invalid_baseline,
    )
    had_error = had_error or invalid_baseline
    if split_findings and not json_out:
        console.print(f"  [yellow]⚠️  {DATASET_SPLIT_FALLBACK_WARNING}[/yellow]")
    return had_error


def _doctor_apply_cross_checks(
    *,
    baseline_report: str | None,
    subject_report: str | None,
    cfg_metric_kind: str | None,
    strict: bool,
    profile: str | None,
    json_out: bool,
    accumulator: DoctorAccumulator,
    had_error: bool,
) -> bool:
    cross_check_findings, cross_check_error = build_cross_check_findings(
        baseline_report,
        subject_report,
        cfg_metric_kind=cfg_metric_kind,
        strict=bool(strict),
        profile=profile,
    )
    _doctor_record_findings(
        accumulator, json_out, cross_check_findings, mark_error=cross_check_error
    )
    return had_error or cross_check_error


def _doctor_apply_tiny_relax(
    *,
    subject_report: str | None,
    baseline_report: str | None,
    json_out: bool,
    accumulator: DoctorAccumulator,
) -> None:
    try:
        tiny_env = _doctor_tiny_relax_enabled()
    except NON_FATAL_EXCEPTIONS:
        tiny_env = False
    tiny_finding = build_tiny_relax_finding(
        subject_report=_doctor_load_optional_report_payload(subject_report),
        baseline_report=_doctor_load_optional_report_payload(baseline_report),
        env_enabled=tiny_env,
    )
    if tiny_finding is not None:
        _doctor_record_findings(accumulator, json_out, [tiny_finding])


def _doctor_format_backend_version(
    backend: str | None, version: str | None
) -> tuple[str, str]:
    return backend or "—", f"=={version}" if backend and version else "—"


def _doctor_inventory_status_action(row: Any) -> str:
    if row.mode == "auto-matcher" or row.status == "ready":
        return "Ready"
    if row.status == "needs_extra":
        if row.required_extra:
            return f"Needs extra: pip install '{row.required_extra}'"
        return "Needs extra"
    if row.detail:
        return row.detail
    return row.status


def _doctor_dataset_network_label(network_mode: str) -> str:
    return {
        "cache": "Cache/Net",
        "yes": "Yes",
        "no": "No",
    }.get(network_mode, "Unknown")


def _doctor_render_registry(
    *,
    json_out: bool,
    has_cuda: bool,
    accumulator: DoctorAccumulator,
) -> bool:
    try:
        from invarlock.core.registry import get_registry

        from .plugins import _check_plugin_extras

        if not json_out:
            console.print("\nPlugin Registry")
        registry = get_registry()
        if not json_out:
            console.print(f"  Adapters: {len(registry.list_adapters())}")
            console.print(f"  Edits: {len(registry.list_edits())}")
            console.print(f"  Guards: {len(registry.list_guards())}")
        if _doctor_third_party_plugins_enabled():
            _doctor_add_finding(
                accumulator,
                json_out,
                "D006",
                "note",
                "Third-party plugin discovery is explicitly enabled by environment; "
                "doctor will include optional third-party adapters in registry checks.",
            )

        try:
            bnb_runtime_ready = bitsandbytes_runtime_available()
        except NON_FATAL_EXCEPTIONS:
            bnb_runtime_ready = False

        adapter_rows = build_adapter_inventory_rows(
            registry,
            has_cuda=bool(has_cuda),
            is_linux=_platform.system().lower() == "linux",
            find_spec_safe=lambda module_name: find_spec_safe(
                module_name, find_spec_fn=importlib.util.find_spec
            ),
            bitsandbytes_runtime_ready=bnb_runtime_ready,
        )
        if adapter_rows:
            adapter_summary = summarize_inventory_rows(adapter_rows)
            total = adapter_summary["total"]
            ready = adapter_summary["ready"]
            need = adapter_summary["needs_extra"]
            unsupported = adapter_summary["unsupported"]
            auto = adapter_summary["auto"]
            rows = [row for row in adapter_rows if row.status != "unsupported"]
            table = Table(
                title=f"Adapters — total: {total} · ready: {ready} · auto: {auto} · missing-extras: {need} · unsupported: {unsupported}"
            )
            table.add_column("Adapter", style="cyan")
            table.add_column("Origin", style="dim")
            table.add_column("Mode", style="dim")
            table.add_column("Backend", style="magenta")
            table.add_column("Version", style="magenta")
            table.add_column("Status / Action", style="green")
            for row in rows:
                backend_disp, ver_disp = _doctor_format_backend_version(
                    row.backend, row.version
                )
                table.add_row(
                    row.name,
                    row.origin.capitalize(),
                    "Auto‑matcher" if row.mode == "auto-matcher" else "Adapter",
                    backend_disp,
                    ver_disp,
                    _doctor_inventory_status_action(row),
                )
            console.print(table)

        for kind, title in (("guards", "Guards"), ("edits", "Edits")):
            grows = build_generic_inventory_rows(
                registry,
                kind=kind,
                check_plugin_extras=_check_plugin_extras,
            )
            if grows:
                summary = summarize_inventory_rows(grows)
                total = summary["total"]
                ready = summary["ready"]
                need = summary["needs_extra"]
                table = Table(
                    title=f"{title} — total: {total} · ready: {ready} · missing-extras: {need}"
                )
                table.add_column("Name", style="cyan")
                table.add_column("Origin", style="dim")
                table.add_column("Mode", style="dim")
                table.add_column("Backend", style="magenta")
                table.add_column("Version", style="magenta")
                table.add_column("Status / Action", style="green")
                for row in grows:
                    backend_disp, ver_disp = _doctor_format_backend_version(
                        row.backend, row.version
                    )
                    table.add_row(
                        row.name,
                        row.origin.capitalize(),
                        ("Guard" if row.mode == "guard" else "Edit"),
                        backend_disp,
                        ver_disp,
                        _doctor_inventory_status_action(row),
                    )
                console.print(table)

        try:
            data_mod = importlib.import_module("invarlock.eval.data")
            list_providers = getattr(data_mod, "list_providers", None)
            if not callable(list_providers):
                raise AttributeError("list_providers unavailable")

            providers = sorted(list_providers())
            if providers:
                dtable = Table(title="Datasets")
                dtable.add_column("Provider", style="cyan")
                dtable.add_column("Network", style="dim")
                dtable.add_column("Status", style="green")
                dtable.add_column("Params", style="dim")
                from invarlock.cli.constants import PROVIDER_NETWORK as provider_network
                from invarlock.cli.constants import PROVIDER_PARAMS as provider_params

                for row in build_dataset_inventory_rows(
                    providers,
                    provider_network=provider_network,
                    provider_params=provider_params,
                ):
                    dtable.add_row(
                        row.provider,
                        _doctor_dataset_network_label(row.network_mode),
                        "✓ Available" if row.available else "Unavailable",
                        row.params,
                    )
                console.print(dtable)
        except NON_FATAL_EXCEPTIONS:
            pass
        return True
    except NON_FATAL_EXCEPTIONS as exc:
        if not json_out:
            console.print(f"  [red]❌ Registry error: {exc}[/red]")
        return False


def doctor_command(
    config: str | None = None,
    profile: str | None = None,
    baseline: str | None = None,
    json_out: bool = False,
    tier: str | None = None,
    baseline_report: str | None = None,
    subject_report: str | None = None,
    strict: bool = False,
):
    """Perform health checks on InvarLock installation."""

    accumulator = DoctorAccumulator()
    console_obj = _doctor_prepare_console(json_out)
    _invarlock_version = _doctor_load_invarlock_version()
    _doctor_print_header(json_out, _invarlock_version, console_obj)
    health_status = True
    had_error = False
    cfg_metric_kind: str | None = None

    if not _doctor_check_core_components(json_out):
        health_status = False
    has_cuda, torch_ok = _doctor_check_torch_runtime(json_out)
    if not torch_ok:
        health_status = False
    if not json_out:
        console_obj.print("\nOptional Dependencies")

    try:
        import torch as _torch

        has_cuda = bool(getattr(_torch, "cuda", None) and _torch.cuda.is_available())
    except NON_FATAL_EXCEPTIONS:
        has_cuda = False

    optional_deps = collect_optional_dependency_facts(
        has_cuda=has_cuda,
        bitsandbytes_runtime_available_fn=bitsandbytes_runtime_available,
        find_spec_fn=importlib.util.find_spec,
    )

    for dep in optional_deps:
        if dep.name == "bitsandbytes":
            runtime_available = bool(dep.runtime_available)
            if runtime_available:
                if not json_out:
                    if has_cuda:
                        console_obj.print(
                            "  [green]✅ bitsandbytes — 8/4-bit loading (GPU)[/green]"
                        )
                    else:
                        console_obj.print(
                            "  [green]✅ bitsandbytes — runtime available on this host[/green]"
                        )
            elif not has_cuda:
                if dep.present:
                    if not json_out:
                        console_obj.print(
                            "  [yellow]⚠️  bitsandbytes — GPU not detected and runtime unavailable on this host[/yellow]"
                        )
                else:
                    if not json_out:
                        console_obj.print(
                            "  [dim]⚠️  bitsandbytes — not installed[/dim]"
                        )
                        console_obj.print(
                            "     → Install: pip install 'invarlock[gpu]'",
                            markup=False,
                        )
            else:
                if not json_out:
                    console_obj.print(
                        "  [yellow]⚠️  bitsandbytes — Present but runtime unavailable on this host[/yellow]"
                    )
                    console_obj.print(
                        "     → Reinstall with: pip install 'invarlock[gpu]' on a compatible host",
                        markup=False,
                    )
            continue

        if not json_out:
            if dep.present:
                console_obj.print(f"  [green]✅ {dep.name} — {dep.description}[/green]")
            else:
                console_obj.print(
                    f"  [yellow]⚠️  {dep.name} — {dep.description}[/yellow]"
                )
                console_obj.print(
                    f"     → Install: pip install 'invarlock[{dep.extra_hint}]'",
                    markup=False,
                )

    had_error, cfg_metric_kind = _doctor_apply_preflight(
        config=config,
        profile=profile,
        tier=tier,
        baseline=baseline,
        json_out=json_out,
        accumulator=accumulator,
        had_error=had_error,
        cfg_metric_kind=cfg_metric_kind,
    )
    had_error = _doctor_apply_baseline_quick_check(
        config=config,
        baseline=baseline,
        baseline_report=baseline_report,
        json_out=json_out,
        accumulator=accumulator,
        had_error=had_error,
    )
    had_error = _doctor_apply_cross_checks(
        baseline_report=baseline_report,
        subject_report=subject_report,
        cfg_metric_kind=cfg_metric_kind,
        strict=bool(strict),
        profile=profile,
        json_out=json_out,
        accumulator=accumulator,
        had_error=had_error,
    )
    _doctor_apply_tiny_relax(
        subject_report=subject_report,
        baseline_report=baseline_report,
        json_out=json_out,
        accumulator=accumulator,
    )
    if not _doctor_render_registry(
        json_out=json_out,
        has_cuda=has_cuda,
        accumulator=accumulator,
    ):
        health_status = False

    # Final status / JSON output
    had_error = had_error or accumulator.had_error
    exit_code = 0 if (health_status and not had_error) else 1
    if json_out:
        import json as _json_out

        result_obj = build_doctor_result(
            format_version=DOCTOR_FORMAT_VERSION,
            findings=accumulator.findings,
            contracts=contract_catalog(),
            support_matrix=load_support_matrix(),
            model_family_catalog=load_model_family_catalog(),
            adapter_capabilities=load_adapter_capabilities(),
            plugin_compatibility=load_plugin_compatibility(),
            policy=POLICY_META
            if "POLICY_META" in globals()
            else {"tier": (tier or "balanced").lower()},
        )
        result_obj["resolution"] = {"exit_code": exit_code}
        typer.echo(_json_out.dumps(result_obj))
        raise typer.Exit(exit_code)
    else:
        console.print("\n" + "=" * 50)
        if exit_code == 0:
            cli_output.print_command_event(
                console, "PASS", "InvarLock installation is healthy (exit code 0)"
            )
        else:
            cli_output.print_command_event(
                console, "FAIL", "InvarLock installation has issues"
            )
            cli_output.print_command_detail(
                console,
                "Run: pip install invarlock[all] to install missing dependencies",
            )
        raise typer.Exit(exit_code)
