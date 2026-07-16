"""The InvarLock command line.

The public workflow intentionally has three transactions: evaluate one closed
request, independently verify its evidence, and render that verified evidence.
Provider qualification and repository maintenance are separate tools rather than
alternate user journeys hidden behind this CLI.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import click
import typer
from rich.console import Console
from typer.core import TyperGroup

from invarlock.security import enforce_default_security


class CoreCommandGroup(TyperGroup):
    """Keep the public journey in semantic order."""

    def list_commands(self, ctx: click.Context) -> list[str]:
        del ctx
        return ["evaluate", "verify", "report"]


app = typer.Typer(
    name="invarlock",
    cls=CoreCommandGroup,
    add_completion=False,
    no_args_is_help=True,
    help=(
        "InvarLock authenticates a paired model evaluation, produces one portable "
        "evidence pack, independently verifies it, and renders one human report.\n"
        "\n"
        "  invarlock evaluate request.yaml\n"
        "  invarlock verify evidence/\n"
        "  invarlock report evidence/"
    ),
)
console = Console()


def _emit_version() -> None:
    try:
        resolved = version("invarlock")
    except PackageNotFoundError:
        try:
            from invarlock import __version__ as resolved
        except (ImportError, ModuleNotFoundError):
            resolved = "unknown"
    console.print(f"InvarLock {resolved}")


def _version_callback(value: bool) -> None:
    if value:
        _emit_version()
        raise typer.Exit()


@app.callback()
def _root(
    version_requested: bool = typer.Option(
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Show the installed InvarLock version and exit.",
    ),
) -> None:
    """Apply the fail-closed process defaults shared by all commands."""

    del version_requested
    enforce_default_security()


@app.command(
    name="evaluate",
    help="Execute or import one closed request and atomically publish one evidence pack.",
)
def evaluate(
    request: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help="Closed evaluation request YAML.",
    ),
    signing_key: Path | None = typer.Option(
        None,
        "--signing-key",
        envvar="INVARLOCK_SIGNING_KEY",
        help="Ed25519 evidence-signing key; may also be supplied by INVARLOCK_SIGNING_KEY.",
    ),
    allow_installed_scorers: bool = typer.Option(
        False,
        "--allow-installed-scorers",
        envvar="INVARLOCK_ALLOW_INSTALLED_SCORERS",
        help=(
            "Authorize loading the exact installed scorer extension bound by the "
            "request and policy. Installed scorer code executes in this process."
        ),
    ),
    json_out: bool = typer.Option(
        False,
        "--json",
        help="Emit one machine-readable result object.",
    ),
    runtime_image: str | None = typer.Option(
        None,
        "--runtime-image",
        envvar="INVARLOCK_RUNTIME_IMAGE",
        help="Local OCI image reference; must be digest-bearing or paired with --runtime-image-digest.",
    ),
    runtime_image_digest: str | None = typer.Option(
        None,
        "--runtime-image-digest",
        envvar="INVARLOCK_RUNTIME_IMAGE_DIGEST",
        help="Pinned lowercase OCI sha256 digest for delegated run execution.",
    ),
    baseline_runtime_image: str | None = typer.Option(
        None,
        "--baseline-runtime-image",
        envvar="INVARLOCK_BASELINE_RUNTIME_IMAGE",
        help="Optional digest-pinned baseline image override.",
    ),
    baseline_runtime_image_digest: str | None = typer.Option(
        None,
        "--baseline-runtime-image-digest",
        envvar="INVARLOCK_BASELINE_RUNTIME_IMAGE_DIGEST",
        help="Pinned baseline image digest; defaults to --runtime-image-digest.",
    ),
    subject_runtime_image: str | None = typer.Option(
        None,
        "--subject-runtime-image",
        envvar="INVARLOCK_SUBJECT_RUNTIME_IMAGE",
        help="Optional digest-pinned subject image override.",
    ),
    subject_runtime_image_digest: str | None = typer.Option(
        None,
        "--subject-runtime-image-digest",
        envvar="INVARLOCK_SUBJECT_RUNTIME_IMAGE_DIGEST",
        help="Pinned subject image digest; defaults to --runtime-image-digest.",
    ),
    container_engine: str | None = typer.Option(
        None,
        "--container-engine",
        envvar="INVARLOCK_CONTAINER_ENGINE",
        help="Closed OCI engine selection: docker or podman.",
    ),
    runtime_device: str | None = typer.Option(
        None,
        "--runtime-device",
        envvar="INVARLOCK_RUNTIME_DEVICE",
        help="Default container device: cpu, cuda, or cuda:<index>.",
    ),
    baseline_runtime_device: str | None = typer.Option(
        None,
        "--baseline-runtime-device",
        envvar="INVARLOCK_BASELINE_RUNTIME_DEVICE",
        help="Optional baseline device override.",
    ),
    subject_runtime_device: str | None = typer.Option(
        None,
        "--subject-runtime-device",
        envvar="INVARLOCK_SUBJECT_RUNTIME_DEVICE",
        help="Optional subject device override.",
    ),
    runtime_entrypoint: str | None = typer.Option(
        None,
        "--runtime-entrypoint",
        envvar="INVARLOCK_RUNTIME_ENTRYPOINT",
        help="Worker entrypoint profile: auto, python, or nvidia.",
    ),
    baseline_runtime_entrypoint: str | None = typer.Option(
        None,
        "--baseline-runtime-entrypoint",
        envvar="INVARLOCK_BASELINE_RUNTIME_ENTRYPOINT",
        help="Optional baseline worker entrypoint profile override.",
    ),
    subject_runtime_entrypoint: str | None = typer.Option(
        None,
        "--subject-runtime-entrypoint",
        envvar="INVARLOCK_SUBJECT_RUNTIME_ENTRYPOINT",
        help="Optional subject worker entrypoint profile override.",
    ),
) -> None:
    """Produce the canonical evidence pack described by REQUEST."""

    from invarlock.core.scorer_extension import ScorerExtensionRegistry
    from invarlock.evaluation_oci import (
        OciEvaluationError,
        OciRuntimeExecutor,
        evaluation_request_execution_mode,
        launch_from_environment,
    )
    from invarlock.evaluation_transaction import (
        EvaluationTransactionError,
        evaluate_request_file,
    )

    try:
        runtime_executor = None
        if evaluation_request_execution_mode(request) == "run":
            runtime_executor = OciRuntimeExecutor(
                launch_from_environment(
                    engine=container_engine,
                    image_ref=runtime_image,
                    image_digest=runtime_image_digest,
                    baseline_image_ref=baseline_runtime_image,
                    baseline_image_digest=baseline_runtime_image_digest,
                    subject_image_ref=subject_runtime_image,
                    subject_image_digest=subject_runtime_image_digest,
                    default_device=runtime_device,
                    baseline_device=baseline_runtime_device,
                    subject_device=subject_runtime_device,
                    runtime_entrypoint=runtime_entrypoint,
                    baseline_entrypoint=baseline_runtime_entrypoint,
                    subject_entrypoint=subject_runtime_entrypoint,
                )
            )
        result = evaluate_request_file(
            request,
            signing_key_path=signing_key,
            runtime_executor=runtime_executor,
            scorer_registry=(
                ScorerExtensionRegistry(allow_installed=True)
                if allow_installed_scorers
                else None
            ),
        )
    except (EvaluationTransactionError, OciEvaluationError) as exc:
        failure = (
            exc
            if isinstance(exc, EvaluationTransactionError)
            else EvaluationTransactionError(str(exc))
        )
        if json_out:
            typer.echo(failure.as_json())
        else:
            console.print(f"FAIL {failure}")
        raise typer.Exit(failure.exit_code) from exc
    if json_out:
        typer.echo(result.as_json())
    else:
        console.print("PASS Evidence pack published")
        console.print(str(result.evidence_path))


@app.command(
    name="verify",
    help="Independently verify one evidence pack against caller-supplied trust anchors.",
)
def verify(
    evidence: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        resolve_path=True,
        help="Canonical evidence-pack directory.",
    ),
    policy: Path | None = typer.Option(
        None,
        "--policy",
        envvar="INVARLOCK_POLICY",
        help="Independent policy input; never taken from the submitted pack.",
    ),
    expected_baseline_artifact: str | None = typer.Option(
        None,
        "--expected-baseline-artifact",
        envvar="INVARLOCK_EXPECTED_BASELINE_ARTIFACT",
        help="Independent expected baseline artifact-identity digest.",
    ),
    expected_subject_artifact: str | None = typer.Option(
        None,
        "--expected-subject-artifact",
        envvar="INVARLOCK_EXPECTED_SUBJECT_ARTIFACT",
        help="Independent expected subject artifact-identity digest.",
    ),
    expected_schedule: str | None = typer.Option(
        None,
        "--expected-schedule",
        envvar="INVARLOCK_EXPECTED_SCHEDULE",
        help="Independent expected canonical schedule digest.",
    ),
    expected_baseline_runtime: str | None = typer.Option(
        None,
        "--expected-baseline-runtime",
        envvar="INVARLOCK_EXPECTED_BASELINE_RUNTIME",
        help="Independent expected baseline runtime digest.",
    ),
    expected_subject_runtime: str | None = typer.Option(
        None,
        "--expected-subject-runtime",
        envvar="INVARLOCK_EXPECTED_SUBJECT_RUNTIME",
        help="Independent expected subject runtime digest.",
    ),
    expected_signer: str | None = typer.Option(
        None,
        "--expected-signer",
        envvar="INVARLOCK_EXPECTED_SIGNER",
        help="Independent expected evidence-signing fingerprint.",
    ),
    receipt: Path | None = typer.Option(
        None,
        "--receipt",
        help="Write the signed verification receipt outside the pack.",
    ),
    verifier_signing_key: Path | None = typer.Option(
        None,
        "--verifier-signing-key",
        envvar="INVARLOCK_VERIFIER_SIGNING_KEY",
        help="Independent Ed25519 verifier key used only for the receipt.",
    ),
    verifier_identity: str | None = typer.Option(
        None,
        "--verifier-identity",
        envvar="INVARLOCK_VERIFIER_IDENTITY",
        help="Stable identity asserted by the independent verifier.",
    ),
    allow_installed_scorers: bool = typer.Option(
        False,
        "--allow-installed-scorers",
        envvar="INVARLOCK_ALLOW_INSTALLED_SCORERS",
        help=(
            "Authorize loading the exact installed scorer extension bound by the "
            "evidence, policy, and request. Installed scorer code executes in this "
            "process."
        ),
    ),
    json_out: bool = typer.Option(
        False,
        "--json",
        help="Emit one machine-readable verification result.",
    ),
) -> None:
    """Verify EVIDENCE without trusting its own policy or runtime declarations."""

    from invarlock.core.scorer_extension import ScorerExtensionRegistry
    from invarlock.evidence_verification import (
        EvidenceVerificationError,
        verify_evidence,
    )

    try:
        result = verify_evidence(
            evidence,
            policy_path=policy,
            expected_baseline_artifact=expected_baseline_artifact,
            expected_subject_artifact=expected_subject_artifact,
            expected_schedule=expected_schedule,
            expected_baseline_runtime=expected_baseline_runtime,
            expected_subject_runtime=expected_subject_runtime,
            expected_signer=expected_signer,
            receipt_path=receipt,
            verifier_signing_key_path=verifier_signing_key,
            verifier_identity=verifier_identity,
            scorer_registry=(
                ScorerExtensionRegistry(allow_installed=True)
                if allow_installed_scorers
                else None
            ),
        )
    except EvidenceVerificationError as exc:
        if json_out:
            typer.echo(exc.as_json())
        else:
            console.print(f"FAIL {exc}")
            signed_receipt = exc.payload.get("signed_receipt")
            if isinstance(signed_receipt, str):
                console.print(f"Receipt {signed_receipt}")
        raise typer.Exit(exc.exit_code) from exc
    if json_out:
        typer.echo(result.as_json())
    else:
        console.print("PASS Evidence verified")
        console.print(result.summary)


@app.command(
    name="report",
    help="Render one human-readable report from the canonical report in an evidence pack.",
)
def report(
    evidence: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        resolve_path=True,
        help="Canonical evidence-pack directory.",
    ),
    html: Path | None = typer.Option(
        None,
        "--html",
        help="Write a self-contained HTML report outside the evidence pack.",
    ),
    explain: bool = typer.Option(
        False,
        "--explain",
        help="Include a concise explanation of the decision and evidence bindings.",
    ),
) -> None:
    """Render EVIDENCE without changing any evidence-pack byte."""

    from invarlock.evidence_reporting import EvidenceReportError, render_evidence

    try:
        result = render_evidence(evidence, html_path=html, explain=explain)
    except EvidenceReportError as exc:
        console.print(f"FAIL {exc}")
        raise typer.Exit(exc.exit_code) from exc
    console.print(result.text)
    if result.html_path is not None:
        console.print(f"HTML {result.html_path}")


def main() -> None:
    """Run the installed command-line entry point."""

    app()


if __name__ == "__main__":  # pragma: no cover
    main()


__all__ = ["app", "main"]
