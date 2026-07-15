from __future__ import annotations

from pathlib import Path
from typing import Any, NoReturn

import typer
from rich.console import Console

from invarlock.cli.output import make_command_event_emitter, print_command_detail
from invarlock.core.report_inputs import (
    ReportInputError,
    resolve_report_input_path,
)
from invarlock.evidence_pack_json import StrictJsonError, read_json_object_snapshot

console = Console()
_REPORT_RENDER_ERRORS = (
    AttributeError,
    KeyError,
    OSError,
    RuntimeError,
    TypeError,
)
_JSON_INPUT_ERRORS = (OSError, StrictJsonError)
_EXPORT_OUTPUT_ERRORS = (OSError, UnicodeEncodeError)
_EXPORT_IMPORT_ERRORS = _REPORT_RENDER_ERRORS + (ImportError,)


def _raise_report_input_failure(message: str) -> NoReturn:
    emit = make_command_event_emitter(console)
    emit("FAIL", message)
    raise typer.Exit(2)


def export_report_command(
    *,
    evaluation_report: str,
    format: str,
    output: str | None = None,
    policy_profile: str | None = None,
    report_url: str | None = None,
    evidence_url: str | None = None,
    verify_result: str | None = None,
    force: bool = False,
) -> None:
    """Export an evaluation report for CI and registry handoff surfaces."""
    emit = make_command_event_emitter(console)
    try:
        input_path = resolve_report_input_path(
            evaluation_report,
            expected_kind="evaluation",
        )
        report_bytes, payload = read_json_object_snapshot(
            input_path,
            label="Evaluation report",
        )
        if not isinstance(payload.get("validation"), dict):
            raise ReportInputError(
                "expected_evaluation_payload",
                input_path,
                detail=(
                    "pass the evaluation.report.json artifact emitted by "
                    "invarlock evaluate or invarlock report generate"
                ),
            )
    except (ReportInputError, StrictJsonError) as exc:
        _raise_report_input_failure(str(exc))

    try:
        from invarlock.reporting.oss_exports import (
            build_report_export_context,
            render_report_export,
            serialize_report_export,
        )
    except _EXPORT_IMPORT_ERRORS as exc:
        emit("FAIL", f"Failed to load report exporter: {exc}")
        raise typer.Exit(1) from exc

    verify_payload: dict[str, Any] | None = None
    if verify_result:
        try:
            verify_path = Path(str(verify_result))
            _verify_bytes, raw_verify_payload = read_json_object_snapshot(
                verify_path,
                label="Verify result",
            )
        except _JSON_INPUT_ERRORS as exc:
            emit("FAIL", f"Failed to read verify result: {exc}")
            raise typer.Exit(2) from exc
        verify_payload = raw_verify_payload

    try:
        context = build_report_export_context(
            input_path,
            payload,
            policy_profile=policy_profile,
            report_url=report_url,
            evidence_url=evidence_url,
            verify_result=verify_payload,
            report_bytes=report_bytes,
        )
        rendered = render_report_export(str(format), context, payload)
        text = serialize_report_export(rendered)
    except ValueError as exc:
        emit("FAIL", str(exc))
        raise typer.Exit(2) from exc
    except _REPORT_RENDER_ERRORS as exc:
        emit("FAIL", f"Failed to export report: {exc}")
        raise typer.Exit(1) from exc

    if output is None or str(output).strip() == "-":
        typer.echo(text, nl=False)
        return

    output_path = Path(str(output))
    if output_path.exists() and not force:
        emit("FAIL", "Output file already exists")
        print_command_detail(console, f"Use --force to overwrite: {output_path}")
        raise typer.Exit(1)
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
    except _EXPORT_OUTPUT_ERRORS as exc:
        emit("FAIL", f"Failed to write export file: {exc}")
        raise typer.Exit(1) from exc

    emit("PASS", "Exported evaluation report")
    print_command_detail(console, f"Input: {input_path}")
    print_command_detail(console, f"Output: {output_path}")


def register_report_export_command(report_app: typer.Typer) -> None:
    @report_app.command(
        name="export",
        help=(
            "Export evaluation evidence for MLflow tags, model cards, "
            "or release review."
        ),
    )
    def report_export(
        evaluation_report: str = typer.Option(
            ...,
            "--evaluation-report",
            "-i",
            help=(
                "Path to evaluation report JSON file or directory containing "
                "canonical evaluation.report.json"
            ),
        ),
        format: str = typer.Option(
            "mlflow-tags",
            "--format",
            help="Output format (mlflow-tags|model-card-md|release-review-md)",
        ),
        output: str | None = typer.Option(
            None,
            "--output",
            "-o",
            help="Path to write the export, or '-' / omitted for stdout.",
        ),
        policy_profile: str | None = typer.Option(
            None,
            "--policy-profile",
            help="Policy profile tag to use when the report does not record one.",
        ),
        report_url: str | None = typer.Option(
            None,
            "--report-url",
            help="Public URL to the evaluation report for Markdown exports.",
        ),
        evidence_url: str | None = typer.Option(
            None,
            "--evidence-url",
            help="Public URL to the evidence pack for Markdown exports.",
        ),
        verify_result: str | None = typer.Option(
            None,
            "--verify-result",
            help=(
                "Path to `invarlock verify --json` output. When supplied, "
                "export status and verifier fields come from the verifier result."
            ),
        ),
        force: bool = typer.Option(
            False, "--force", help="Overwrite output file if it exists."
        ),
    ) -> None:
        return export_report_command(
            evaluation_report=evaluation_report,
            format=format,
            output=output,
            policy_profile=policy_profile,
            report_url=report_url,
            evidence_url=evidence_url,
            verify_result=verify_result,
            force=force,
        )


__all__ = ["export_report_command", "register_report_export_command"]
