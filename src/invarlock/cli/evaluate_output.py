"""Output and render helpers for the evaluate CLI command."""

from __future__ import annotations

import io
import json
import math
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import typer
from rich.console import Console, Group
from rich.panel import Panel
from rich.text import Text

from invarlock import __version__ as INVARLOCK_VERSION
from invarlock.runtime_security import (
    RuntimeManifestExecution,
    resolve_runtime_image,
    resolve_runtime_image_digest,
)


def _render_banner_lines(title: str, context: str) -> list[str]:
    return [
        title,
        context,
        "-" * max(len(title), len(context)),
    ]


def _print_header_banner(
    console: Console, *, version: str, profile: str, tier: str, adapter: str
) -> None:
    title = f"INVARLOCK v{version} · Evaluation Pipeline"
    context = f"Profile: {profile} · Tier: {tier} · Adapter: {adapter}"
    console.print(
        Panel.fit(
            Group(
                Text(title, style="bold"),
                Text(context, style="dim"),
            ),
            border_style="cyan",
            padding=(0, 1),
            title="Evaluate",
        )
    )


def _phase_title(index: int, total: int, title: str) -> str:
    return f"PHASE {index}/{total} · {title}"


def _print_phase_header(console: Console, title: str) -> None:
    console.print(title)
    console.print("-" * max(67, len(title)))


def _format_ratio(value: Any) -> str:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if not math.isfinite(val):
        return "N/A"
    return f"{val:.3f}"


def _evaluation_report_manifest_execution(
    *,
    execution_mode: str,
    allow_network: bool,
    allow_remote_code: bool,
    allow_third_party_plugins: bool,
) -> RuntimeManifestExecution | None:
    normalized_execution_mode = str(execution_mode or "").strip().lower()
    if normalized_execution_mode != "container":
        return None
    return RuntimeManifestExecution(
        execution_mode="container",
        container_execution=True,
        image_ref=resolve_runtime_image(),
        image_digest=resolve_runtime_image_digest(),
        allow_network=allow_network,
        allow_remote_code=allow_remote_code,
        allow_third_party_plugins=allow_third_party_plugins,
    )


def _resolve_verbosity(quiet: bool, verbose: bool, *, console: Console) -> int:
    if quiet and verbose:
        console.print("--quiet and --verbose are mutually exclusive")
        raise typer.Exit(2)
    if quiet:
        return 0
    if verbose:
        return 2
    return 1


@contextmanager
def _override_console(module: Any, new_console: Console) -> Iterator[None]:
    original_console = getattr(module, "console", None)
    module.console = new_console
    try:
        yield
    finally:
        module.console = original_console


@contextmanager
def _suppress_child_output(
    enabled: bool,
    *,
    run_execution_module: Any | None = None,
    report_module: Any | None = None,
    run_module: Any | None = None,
) -> Iterator[io.StringIO | None]:
    if not enabled:
        yield None
        return
    if run_execution_module is None:
        from . import run_execution as run_exec_mod
    else:
        run_exec_mod = run_execution_module
    if report_module is None:
        from .commands import report as report_mod
    else:
        report_mod = report_module
    if run_module is None:
        from .commands import run as run_mod
    else:
        run_mod = run_module

    buffer = io.StringIO()
    quiet_console = Console(file=buffer, force_terminal=False, color_system=None)
    with (
        _override_console(run_mod, quiet_console),
        _override_console(run_exec_mod, quiet_console),
        _override_console(report_mod, quiet_console),
    ):
        yield buffer


def _print_quiet_summary(
    *,
    version: str = INVARLOCK_VERSION,
    console: Console,
    report_out: Path,
    baseline: str,
    subject: str,
    profile: str,
    json_load_fn: Any = json.load,
) -> None:
    report_path = report_out / "evaluation.report.json"
    runtime_manifest_path = report_out / "runtime.manifest.json"
    console.print(f"INVARLOCK v{version} · EVALUATE")
    console.print(f"Baseline: {baseline} -> Subject: {subject} · Profile: {profile}")
    if not report_path.exists():
        console.print(f"Output: {report_out}", soft_wrap=True)
        return
    try:
        with report_path.open("r", encoding="utf-8") as fh:
            evaluation_report = json_load_fn(fh)
    except (json.JSONDecodeError, OSError, TypeError, ValueError):
        console.print(f"Output: {report_path}", soft_wrap=True)
        return
    if not isinstance(evaluation_report, dict):
        console.print(f"Output: {report_path}", soft_wrap=True)
        return
    try:
        from invarlock.reporting.report_summary import (
            compute_console_validation_block as _console_block,
        )

        block = _console_block(evaluation_report)
        rows = block.get("rows", [])
        total = len(rows) if isinstance(rows, list) else 0
        passed = (
            sum(1 for row in rows if row.get("ok")) if isinstance(rows, list) else 0
        )
        status = "PASS" if block.get("overall_pass") else "FAIL"
    except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
        total = 0
        passed = 0
        status = "UNKNOWN"
    pm_ratio = _format_ratio(
        (evaluation_report.get("primary_metric") or {}).get("ratio_vs_baseline")
    )
    gate_summary = f"{passed}/{total} passed" if total else "N/A"
    console.print(f"Status: {status} · Gates: {gate_summary}")
    if pm_ratio != "N/A":
        console.print(f"Primary metric ratio: {pm_ratio}")
    console.print(f"Output: {report_path}", soft_wrap=True)
    if runtime_manifest_path.exists():
        console.print(f"Runtime provenance: {runtime_manifest_path}", soft_wrap=True)
