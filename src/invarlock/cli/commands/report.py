"""
Report operations group
=======================

Provides the `invarlock report` group with:
  - default callback to generate reports from runs
  - subcommands: verify, explain, html, validate
"""

import json
from pathlib import Path

import typer
from rich.console import Console

from invarlock.reporting import certificate as certificate_lib
from invarlock.reporting import report as report_lib

console = Console()


# Group with callback so `invarlock report` still generates reports
report_app = typer.Typer(
    help="Operations on reports and certificates (verify, explain, html, validate).",
    invoke_without_command=True,
)


def _generate_reports(
    *,
    run: str,
    format: str = "json",
    compare: str | None = None,
    baseline: str | None = None,
    output: str | None = None,
) -> None:
    # This callback runs only when invoked without subcommand (default Click behavior)
    try:
        # When invoked programmatically (not via Typer CLI), the default values for
        # parameters defined with `typer.Option(...)` can be instances of
        # `typer.models.OptionInfo`. Coerce them to real Python values to avoid
        # accidentally treating an OptionInfo object as a path.
        try:  # Typer internal type may change between versions
            from typer.models import OptionInfo as _TyperOptionInfo
        except Exception:  # pragma: no cover - defensive fallback
            _TyperOptionInfo = ()  # type: ignore[assignment]

        def _coerce_option(value, fallback=None):
            if isinstance(value, _TyperOptionInfo):
                return getattr(value, "default", fallback)
            return value if value is not None else fallback

        run = _coerce_option(run)
        format = _coerce_option(format, "json")
        compare = _coerce_option(compare)
        baseline = _coerce_option(baseline)
        output = _coerce_option(output)

        # Load primary report
        console.print(f"📊 Loading run report: {run}")
        primary_report = _load_run_report(run)

        # Load comparison report if specified
        compare_report = None
        if compare:
            console.print(f"📊 Loading comparison report: {compare}")
            compare_report = _load_run_report(compare)

        # Load baseline report if specified
        baseline_report = None
        if baseline:
            console.print(f"📊 Loading baseline report: {baseline}")
            baseline_report = _load_run_report(baseline)

        # Determine output directory
        if output is None:
            run_name = Path(run).stem if Path(run).is_file() else Path(run).name
            output_dir = f"reports_{run_name}"
        else:
            output_dir = output

        # Determine formats
        if format == "all":
            formats = ["json", "markdown", "html"]
        else:
            formats = [format]

        # Validate certificate requirements
        if "cert" in formats:
            if baseline_report is None:
                console.print(
                    "[red]❌ Certificate format requires --baseline parameter[/red]"
                )
                console.print(
                    "Use: invarlock report --run <run_dir> --format cert --baseline <baseline_run_dir>"
                )
                raise typer.Exit(1)
            console.print("📜 Generating safety certificate with baseline comparison")

        # Generate reports
        console.print(f"📝 Generating reports in formats: {formats}")
        saved_files = report_lib.save_report(
            primary_report,
            output_dir,
            formats=formats,
            compare=compare_report,
            baseline=baseline_report,
            filename_prefix="evaluation",
        )

        # Show results
        console.print("[green]✅ Reports generated successfully![/green]")
        console.print(f"📁 Output directory: {output_dir}")

        for fmt, file_path in saved_files.items():
            if fmt == "cert":
                console.print(f"  📜 CERTIFICATE (JSON): {file_path}")
            elif fmt == "cert_md":
                console.print(f"  📜 CERTIFICATE (MD): {file_path}")
            else:
                console.print(f"  📄 {fmt.upper()}: {file_path}")

        # Show key metrics (PM-first). Avoid legacy PPL wording.
        console.print("\n📈 Key Metrics:")
        console.print(f"  Model: {primary_report['meta']['model_id']}")
        console.print(f"  Edit: {primary_report['edit']['name']}")
        pm = (primary_report.get("metrics", {}) or {}).get("primary_metric", {})
        if isinstance(pm, dict) and pm:
            kind = str(pm.get("kind") or "primary")
            console.print(f"  Primary Metric: {kind}")
            final = pm.get("final")
            if isinstance(final, int | float):
                console.print(f"  point (final): {final:.3f}")
            dci = pm.get("display_ci")
            if isinstance(dci, tuple | list) and len(dci) == 2:
                try:
                    lo, hi = float(dci[0]), float(dci[1])
                    console.print(f"  CI: {lo:.3f}–{hi:.3f}")
                except Exception:
                    pass
            ratio = pm.get("ratio_vs_baseline")
            if isinstance(ratio, int | float):
                console.print(f"  ratio vs baseline: {ratio:.3f}")

        # Show certificate validation if generated
        if "cert" in formats and baseline_report:
            try:
                certificate = certificate_lib.make_certificate(
                    primary_report, baseline_report
                )
                certificate_lib.validate_certificate(certificate)
                from invarlock.reporting.render import (
                    compute_console_validation_block as _console_block,
                )

                block = _console_block(certificate)
                overall_pass = bool(block.get("overall_pass"))

                console.print("\n📜 Certificate Validation:")
                status_emoji = "✅" if overall_pass else "❌"
                console.print(
                    f"  Overall Status: {status_emoji} {'PASS' if overall_pass else 'FAIL'}"
                )

                rows = block.get("rows", [])
                if isinstance(rows, list) and rows:
                    for row in rows:
                        try:
                            label = row.get("label")
                            status = row.get("status")
                            if label and status:
                                console.print(f"  {label}: {status}")
                        except Exception:
                            continue

                # In CLI report flow, do not hard-exit on validation failure; just display status.
                # CI gating should be handled by dedicated verify commands.

            except Exception as e:
                console.print(
                    f"  [yellow]⚠️  Certificate validation error: {e}[/yellow]"
                )
                # Exit non-zero on certificate generation error
                raise typer.Exit(1) from e

    except Exception as e:
        console.print(f"[red]❌ Report generation failed: {e}[/red]")
        raise typer.Exit(1) from e


@report_app.callback(invoke_without_command=True)
def report_callback(
    ctx: typer.Context,
    run: str | None = typer.Option(
        None, "--run", help="Path to run directory or RunReport JSON"
    ),
    format: str = typer.Option(
        "json", "--format", help="Output format (json|md|html|cert|all)"
    ),
    compare: str | None = typer.Option(
        None, "--compare", help="Path to second run for comparison"
    ),
    baseline: str | None = typer.Option(
        None,
        "--baseline",
        help="Path to baseline run for certificate generation (required for cert format)",
    ),
    output: str | None = typer.Option(None, "--output", "-o", help="Output directory"),
):
    """Generate a report from a run (default callback)."""
    if getattr(ctx, "resilient_parsing", False) or ctx.invoked_subcommand is not None:
        return
    if not run:
        console.print("[red]❌ --run is required when no subcommand is provided[/red]")
        raise typer.Exit(2)
    return _generate_reports(
        run=run, format=format, compare=compare, baseline=baseline, output=output
    )


# Backward-compatible function name expected by tests
def report_command(
    run: str,
    format: str = "json",
    compare: str | None = None,
    baseline: str | None = None,
    output: str | None = None,
):
    return _generate_reports(
        run=run, format=format, compare=compare, baseline=baseline, output=output
    )


def _load_run_report(path: str) -> dict:
    """Load a RunReport from file or directory."""
    path_obj = Path(path)

    if path_obj.is_file():
        with open(path_obj) as f:
            return json.load(f)
    elif path_obj.is_dir():
        # Look for report JSON files
        json_files = list(path_obj.glob("*.json"))
        report_files = [f for f in json_files if "report" in f.name.lower()]

        if not report_files:
            raise FileNotFoundError(f"No report JSON files found in {path}")

        with open(report_files[0]) as f:
            return json.load(f)
    else:
        raise FileNotFoundError(f"Path not found: {path}")


# Subcommands wired from existing modules
@report_app.command(
    name="verify", help="Recompute and verify metrics for a report/cert."
)
def report_verify_command(
    certificates: list[str] = typer.Argument(
        ..., help="One or more certificate JSON files to verify."
    ),
    baseline: str | None = typer.Option(
        None,
        "--baseline",
        help="Optional baseline certificate/report JSON to enforce provider parity.",
    ),
    tolerance: float = typer.Option(
        1e-9, "--tolerance", help="Tolerance for analysis-basis comparisons."
    ),
    profile: str | None = typer.Option(
        "dev",
        "--profile",
        help="Execution profile affecting parity enforcement and exit codes (dev|ci|release).",
    ),
):  # pragma: no cover - thin wrapper around verify_command
    from pathlib import Path as _Path

    from .verify import verify_command as _verify_command

    cert_paths = [_Path(c) for c in certificates]
    baseline_path = _Path(baseline) if isinstance(baseline, str) else None
    return _verify_command(
        certificates=cert_paths,
        baseline=baseline_path,
        tolerance=tolerance,
        profile=profile,
    )


@report_app.command(
    name="explain", help="Explain certificate gates for report vs baseline."
)
def report_explain(
    report: str = typer.Option(..., "--report", help="Path to primary report.json"),
    baseline: str = typer.Option(
        ..., "--baseline", help="Path to baseline report.json"
    ),
):  # pragma: no cover - thin wrapper
    """Explain certificate gates for a report vs baseline."""
    from .explain_gates import explain_gates_command as _explain

    return _explain(report=report, baseline=baseline)


@report_app.command(name="html", help="Render a certificate JSON to HTML.")
def report_html(
    input: str = typer.Option(..., "--input", "-i", help="Path to certificate JSON"),
    output: str = typer.Option(..., "--output", "-o", help="Path to output HTML file"),
    embed_css: bool = typer.Option(
        True, "--embed-css/--no-embed-css", help="Inline a minimal static stylesheet"
    ),
    force: bool = typer.Option(
        False, "--force", help="Overwrite output file if exists"
    ),
):  # pragma: no cover - thin wrapper
    from .export_html import export_html_command as _export

    return _export(input=input, output=output, embed_css=embed_css, force=force)


@report_app.command("validate")
def report_validate(
    report: str = typer.Argument(
        ..., help="Path to certificate JSON to validate against schema v1"
    ),
):
    """Validate a certificate JSON against the current schema (v1)."""
    p = Path(report)
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]❌ Failed to read input JSON: {exc}[/red]")
        raise typer.Exit(1) from exc

    try:
        from invarlock.reporting.certificate import validate_certificate

        ok = validate_certificate(payload)
        if not ok:
            console.print("[red]❌ Certificate schema validation failed[/red]")
            raise typer.Exit(2)
        console.print("✅ Certificate schema is valid")
    except ValueError as exc:
        console.print(f"[red]❌ Certificate validation error: {exc}[/red]")
        raise typer.Exit(2) from exc
    except typer.Exit:
        raise
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]❌ Validation failed: {exc}[/red]")
        raise typer.Exit(1) from exc


__all__ = ["report_app", "report_callback", "report_command"]
