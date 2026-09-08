"""The InvarLock command line.

The controlled workflow evaluates one closed request, independently verifies
its evidence, and renders its recorded result. The pipeline namespace compares
existing evaluator records under its own explicit evidence contract. Provider
qualification and repository maintenance remain separate tools.
"""

from __future__ import annotations

import json
import os
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import click
import typer
from rich.console import Console
from rich.markdown import Markdown
from typer.core import TyperGroup

from invarlock.pipeline.cli import app as pipeline_app
from invarlock.security import enforce_default_security


class CoreCommandGroup(TyperGroup):
    """Keep the public journey in semantic order."""

    def list_commands(self, ctx: click.Context) -> list[str]:
        del ctx
        return ["pipeline", "evaluate", "verify", "report"]


app = typer.Typer(
    name="invarlock",
    cls=CoreCommandGroup,
    add_completion=False,
    no_args_is_help=True,
    help=(
        "Check release policy with your existing evaluation results, or execute/import "
        "a controlled paired evaluation and hand off independently verifiable evidence.\n"
        "\n"
        "  Existing results: invarlock pipeline --help\n"
        "  Controlled evaluation: invarlock evaluate request.yaml\n"
        "  invarlock verify evidence/\n"
        "  invarlock report evidence/"
    ),
)
app.add_typer(
    pipeline_app,
    name="pipeline",
    help="Compare existing evaluation results; no inference required. Separate pipeline evidence contract.",
)
console = Console(markup=False, highlight=False)


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
    ctx: typer.Context,
    request: Path = typer.Argument(
        ...,
        metavar="REQUEST",
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
        rich_help_panel="Signing and execution authorization",
    ),
    allow_installed_scorers: bool = typer.Option(
        False,
        "--allow-installed-scorers",
        envvar="INVARLOCK_ALLOW_INSTALLED_SCORERS",
        help=(
            "Authorize loading the exact installed scorer extension bound by the "
            "request and policy. Installed scorer code executes in this process."
        ),
        rich_help_panel="Signing and execution authorization",
    ),
    json_out: bool = typer.Option(
        False,
        "--json",
        help="Emit one machine-readable result object.",
        rich_help_panel="Output and workflow",
    ),
    preflight: bool = typer.Option(
        False,
        "--preflight",
        help=(
            "Validate request, authenticated inputs, provider resources, local "
            "runtime images, key, and destination without workers or publication."
        ),
        rich_help_panel="Output and workflow",
    ),
    request_root: Path | None = typer.Option(
        None,
        "--request-root",
        hidden=True,
        help="Resolve request-relative inputs against this authenticated directory.",
        rich_help_panel="Output and workflow",
    ),
    runtime_profile: Path | None = typer.Option(
        None,
        "--runtime-profile",
        help="Explicit runtime resource JSON; CLI overrides profile, then environment/defaults. Run mode only.",
        rich_help_panel="Runtime resources (advanced)",
    ),
    runtime_image: str | None = typer.Option(
        None,
        "--runtime-image",
        envvar="INVARLOCK_RUNTIME_IMAGE",
        help="Local OCI image reference; must be digest-bearing or paired with --runtime-image-digest.",
        rich_help_panel="Runtime resources (advanced)",
    ),
    runtime_image_digest: str | None = typer.Option(
        None,
        "--runtime-image-digest",
        envvar="INVARLOCK_RUNTIME_IMAGE_DIGEST",
        help="Pinned lowercase OCI sha256 digest for delegated run execution.",
        rich_help_panel="Runtime resources (advanced)",
    ),
    baseline_runtime_image: str | None = typer.Option(
        None,
        "--baseline-runtime-image",
        envvar="INVARLOCK_BASELINE_RUNTIME_IMAGE",
        help="Optional digest-pinned baseline image override.",
        rich_help_panel="Runtime resources (advanced)",
    ),
    baseline_runtime_image_digest: str | None = typer.Option(
        None,
        "--baseline-runtime-image-digest",
        envvar="INVARLOCK_BASELINE_RUNTIME_IMAGE_DIGEST",
        help="Pinned baseline image digest; defaults to --runtime-image-digest.",
        rich_help_panel="Runtime resources (advanced)",
    ),
    subject_runtime_image: str | None = typer.Option(
        None,
        "--subject-runtime-image",
        envvar="INVARLOCK_SUBJECT_RUNTIME_IMAGE",
        help="Optional digest-pinned subject image override.",
        rich_help_panel="Runtime resources (advanced)",
    ),
    subject_runtime_image_digest: str | None = typer.Option(
        None,
        "--subject-runtime-image-digest",
        envvar="INVARLOCK_SUBJECT_RUNTIME_IMAGE_DIGEST",
        help="Pinned subject image digest; defaults to --runtime-image-digest.",
        rich_help_panel="Runtime resources (advanced)",
    ),
    container_engine: str | None = typer.Option(
        None,
        "--container-engine",
        envvar="INVARLOCK_CONTAINER_ENGINE",
        help="Closed OCI engine selection: docker or podman.",
        rich_help_panel="Runtime resources (advanced)",
    ),
    runtime_device: str | None = typer.Option(
        None,
        "--runtime-device",
        envvar="INVARLOCK_RUNTIME_DEVICE",
        help="Default container device: cpu, cuda, or cuda:<index>.",
        rich_help_panel="Runtime resources (advanced)",
    ),
    baseline_runtime_device: str | None = typer.Option(
        None,
        "--baseline-runtime-device",
        envvar="INVARLOCK_BASELINE_RUNTIME_DEVICE",
        help="Optional baseline device override.",
        rich_help_panel="Runtime resources (advanced)",
    ),
    subject_runtime_device: str | None = typer.Option(
        None,
        "--subject-runtime-device",
        envvar="INVARLOCK_SUBJECT_RUNTIME_DEVICE",
        help="Optional subject device override.",
        rich_help_panel="Runtime resources (advanced)",
    ),
    runtime_entrypoint: str | None = typer.Option(
        None,
        "--runtime-entrypoint",
        envvar="INVARLOCK_RUNTIME_ENTRYPOINT",
        help="Worker entrypoint profile: auto, python, or nvidia.",
        rich_help_panel="Runtime resources (advanced)",
    ),
    baseline_runtime_entrypoint: str | None = typer.Option(
        None,
        "--baseline-runtime-entrypoint",
        envvar="INVARLOCK_BASELINE_RUNTIME_ENTRYPOINT",
        help="Optional baseline worker entrypoint profile override.",
        rich_help_panel="Runtime resources (advanced)",
    ),
    subject_runtime_entrypoint: str | None = typer.Option(
        None,
        "--subject-runtime-entrypoint",
        envvar="INVARLOCK_SUBJECT_RUNTIME_ENTRYPOINT",
        help="Optional subject worker entrypoint profile override.",
        rich_help_panel="Runtime resources (advanced)",
    ),
    runtime_cpus: str | None = typer.Option(
        None,
        "--runtime-cpus",
        envvar="INVARLOCK_RUNTIME_CPUS",
        help="Hard CPU limit applied independently to each OCI worker.",
        rich_help_panel="Runtime resources (advanced)",
    ),
    runtime_memory_mib: str | None = typer.Option(
        None,
        "--runtime-memory-mib",
        envvar="INVARLOCK_RUNTIME_MEMORY_MIB",
        help="Hard memory limit in MiB applied independently to each OCI worker.",
        rich_help_panel="Runtime resources (advanced)",
    ),
    runtime_user: str | None = typer.Option(
        None,
        "--runtime-user",
        envvar="INVARLOCK_RUNTIME_USER",
        help="Non-root numeric UID:GID used by both OCI workers.",
        rich_help_panel="Runtime resources (advanced)",
    ),
) -> None:
    """Produce the canonical evidence pack described by REQUEST."""

    from invarlock.cli.runtime_profile import (
        ResolvedRuntimeProfile,
        RuntimeProfile,
        RuntimeProfileError,
        load_runtime_profile,
        resolve_runtime_profile,
    )
    from invarlock.core.evaluation_request import (
        EvaluationRequestError,
        load_evaluation_request,
    )
    from invarlock.core.registry import CoreRegistry
    from invarlock.core.scorer_extension import ScorerExtensionRegistry
    from invarlock.evaluation_oci import (
        OciEvaluationError,
        OciRuntimeExecutor,
        launch_from_environment,
        preflight_oci_launch,
    )
    from invarlock.evaluation_transaction import (
        EvaluationPreflightError,
        EvaluationPreflightResult,
        EvaluationTransactionError,
        EvaluationTransactionResult,
        evaluate_request_file,
        preflight_evaluation_request,
    )
    from invarlock.evidence_pack_json import StrictJsonError

    profile: RuntimeProfile | None = None
    profile_context: ResolvedRuntimeProfile | None = None
    result: EvaluationPreflightResult | EvaluationTransactionResult
    try:
        scorer_registry = ScorerExtensionRegistry(
            allow_installed=allow_installed_scorers
        )
        registry = CoreRegistry()
        loaded_request = load_evaluation_request(
            request,
            provider_resolver=registry.get_runtime_provider,
            request_root=request_root,
        )
        if runtime_profile is not None:
            if loaded_request.execution.mode != "run":
                raise RuntimeProfileError(
                    "--runtime-profile applies only to run requests; import evidence already binds its runtime"
                )
            profile = load_runtime_profile(runtime_profile)
            explicit = {
                name: value
                for name, value in ctx.params.items()
                if isinstance(value, str)
                and getattr(ctx.get_parameter_source(name), "name", None)
                == "COMMANDLINE"
            }
            profile_context = resolve_runtime_profile(
                profile, explicit=explicit, environment=dict(os.environ)
            )
        runtime_executor = None
        launch = None
        if loaded_request.execution.mode == "run":
            if profile_context is not None:
                launch = launch_from_environment(**profile_context.arguments)
            else:
                launch = launch_from_environment(
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
                    runtime_cpus=runtime_cpus,
                    runtime_memory_mib=runtime_memory_mib,
                    runtime_user=runtime_user,
                )
            runtime_executor = OciRuntimeExecutor(launch)
        if preflight:
            runtime_digests = preflight_oci_launch(launch) if launch else None
            result = preflight_evaluation_request(
                loaded_request,
                signing_key_path=signing_key,
                scorer_registry=scorer_registry,
                runtime_image_digests=runtime_digests,
                resource_resolver=runtime_executor,
                registry=registry,
            )
        else:
            runtime_digests = preflight_oci_launch(launch) if launch else None
            result = evaluate_request_file(
                loaded_request,
                signing_key_path=signing_key,
                runtime_executor=runtime_executor,
                runtime_image_digests=runtime_digests,
                scorer_registry=scorer_registry,
                registry=registry,
            )
    except (
        EvaluationPreflightError,
        EvaluationRequestError,
        EvaluationTransactionError,
        OciEvaluationError,
        StrictJsonError,
    ) as exc:
        failure: EvaluationPreflightError | EvaluationTransactionError
        if preflight:
            failure = (
                exc
                if isinstance(exc, EvaluationPreflightError)
                else EvaluationPreflightError(str(exc))
            )
        else:
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
    elif preflight:
        console.print("Preflight complete")
        if isinstance(result, EvaluationPreflightResult):
            console.print(
                f"Mode: {result.execution_mode}; paired records: {result.record_count}"
            )
            console.print(f"Evidence destination: {result.output}", soft_wrap=True)
            console.print(f"Validated checks: {len(result.checks)}")
        if profile is not None and profile_context is not None and launch is not None:
            console.print(f"Runtime profile: {profile.digest}")
            console.print(f"Container engine: {launch.engine}")
            for side in ("baseline", "subject"):
                resolved_side = getattr(launch, side)
                console.print(
                    f"{side.capitalize()}: {resolved_side.image_ref}; device {resolved_side.device}; entrypoint {resolved_side.entrypoint}"
                )
            limits = launch.worker_limits
            console.print(
                f"Each worker: {limits.cpus} CPUs; {limits.memory_mib} MiB; user {limits.user}"
            )
            for field, source in profile_context.sources.items():
                console.print(f"  {field}: {source}")
        console.print("No execution or publication was performed")
        console.print("Next: run the same evaluate command without --preflight.")
    else:
        assert isinstance(result, EvaluationTransactionResult)
        console.print("Evidence created")
        console.print(
            f"Recorded policy result: {result.policy_verdict or 'unavailable'}"
        )
        console.print("Recipient verification: not performed")
        console.print(f"Evidence: {result.evidence_path}", soft_wrap=True)
        console.print(
            "Next: verify with independently approved trust inputs; use report to inspect the recorded checks."
        )


@app.command(
    name="verify",
    help="Independently verify one evidence pack against caller-supplied trust anchors.",
)
def verify(
    ctx: typer.Context,
    evidence: Path = typer.Argument(
        ...,
        metavar="EVIDENCE",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        resolve_path=True,
        help="Canonical evidence-pack directory.",
    ),
    trust_profile: Path | None = typer.Option(
        None,
        "--trust-profile",
        help=(
            "Closed invarlock/trust-inputs-v1 profile. Explicit trust-anchor "
            "options cannot be mixed with this profile."
        ),
        rich_help_panel="Recipient verification",
    ),
    policy: Path | None = typer.Option(
        None,
        "--policy",
        envvar="INVARLOCK_POLICY",
        help="Independent policy input; never taken from the submitted pack.",
        rich_help_panel="Independent trust anchors",
    ),
    expected_baseline_artifact: str | None = typer.Option(
        None,
        "--expected-baseline-artifact",
        envvar="INVARLOCK_EXPECTED_BASELINE_ARTIFACT",
        help="Independent expected baseline artifact-identity digest.",
        rich_help_panel="Independent trust anchors",
    ),
    expected_subject_artifact: str | None = typer.Option(
        None,
        "--expected-subject-artifact",
        envvar="INVARLOCK_EXPECTED_SUBJECT_ARTIFACT",
        help="Independent expected subject artifact-identity digest.",
        rich_help_panel="Independent trust anchors",
    ),
    expected_schedule: str | None = typer.Option(
        None,
        "--expected-schedule",
        envvar="INVARLOCK_EXPECTED_SCHEDULE",
        help="Independent expected canonical schedule digest.",
        rich_help_panel="Independent trust anchors",
    ),
    expected_baseline_runtime: str | None = typer.Option(
        None,
        "--expected-baseline-runtime",
        envvar="INVARLOCK_EXPECTED_BASELINE_RUNTIME",
        help="Independent expected baseline runtime digest.",
        rich_help_panel="Independent trust anchors",
    ),
    expected_subject_runtime: str | None = typer.Option(
        None,
        "--expected-subject-runtime",
        envvar="INVARLOCK_EXPECTED_SUBJECT_RUNTIME",
        help="Independent expected subject runtime digest.",
        rich_help_panel="Independent trust anchors",
    ),
    expected_signer: str | None = typer.Option(
        None,
        "--expected-signer",
        envvar="INVARLOCK_EXPECTED_SIGNER",
        help="Independent expected evidence-signing fingerprint.",
        rich_help_panel="Independent trust anchors",
    ),
    expected_request_digest: str | None = typer.Option(
        None,
        "--expected-request-digest",
        envvar="INVARLOCK_EXPECTED_REQUEST_DIGEST",
        help=(
            "Independent expected normalized request digest; required when either "
            "side uses llama_cpp."
        ),
        rich_help_panel="Independent trust anchors",
    ),
    receipt: Path | None = typer.Option(
        None,
        "--receipt",
        help="Write the signed verification receipt outside the pack.",
        rich_help_panel="Recipient verification",
    ),
    verifier_signing_key: Path | None = typer.Option(
        None,
        "--verifier-signing-key",
        envvar="INVARLOCK_VERIFIER_SIGNING_KEY",
        help="Independent Ed25519 verifier key used only for the receipt.",
        rich_help_panel="Recipient verification",
    ),
    verifier_identity: str | None = typer.Option(
        None,
        "--verifier-identity",
        envvar="INVARLOCK_VERIFIER_IDENTITY",
        help="Stable identity asserted by the independent verifier.",
        rich_help_panel="Recipient verification",
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
        rich_help_panel="Signing and execution authorization",
    ),
    json_out: bool = typer.Option(
        False,
        "--json",
        help="Emit one machine-readable verification result.",
        rich_help_panel="Output and workflow",
    ),
) -> None:
    """Verify EVIDENCE without trusting its own policy or runtime declarations."""

    from invarlock.core.scorer_extension import ScorerExtensionRegistry
    from invarlock.evidence_verification import (
        EvidenceVerificationError,
        _require_outside_evidence,
        verify_evidence,
    )
    from invarlock.trust_inputs import TrustInputsError, load_trust_inputs

    try:
        trust_profile_digest: str | None = None
        policy_bytes: bytes | None = None
        verifier_signing_key_bytes: bytes | None = None
        if trust_profile is not None:
            explicit_names = (
                "policy",
                "expected_baseline_artifact",
                "expected_subject_artifact",
                "expected_schedule",
                "expected_baseline_runtime",
                "expected_subject_runtime",
                "expected_signer",
                "expected_request_digest",
                "verifier_signing_key",
                "verifier_identity",
                "allow_installed_scorers",
            )
            conflicts = [
                name.replace("_", "-")
                for name in explicit_names
                if getattr(
                    ctx.get_parameter_source(name),
                    "name",
                    None,
                )
                == "COMMANDLINE"
            ]
            if conflicts:
                rendered = ", ".join(f"--{name}" for name in conflicts)
                raise EvidenceVerificationError(
                    f"--trust-profile cannot be mixed with {rendered}"
                )
            _require_outside_evidence(
                evidence,
                trust_profile,
                label="independent trust profile",
            )
            try:
                loaded = load_trust_inputs(trust_profile)
            except TrustInputsError as exc:
                raise EvidenceVerificationError(str(exc)) from exc
            _require_outside_evidence(
                evidence,
                loaded.policy_path,
                label="independent policy",
            )
            _require_outside_evidence(
                evidence,
                loaded.verifier_signing_key_path,
                label="verifier Ed25519 signing key",
            )
            policy = loaded.policy_path
            policy_bytes = loaded.policy_bytes
            expected_baseline_artifact = loaded.expected_artifact_digests["baseline"]
            expected_subject_artifact = loaded.expected_artifact_digests["subject"]
            expected_schedule = loaded.expected_schedule_digest
            expected_baseline_runtime = loaded.expected_runtime_digests["baseline"]
            expected_subject_runtime = loaded.expected_runtime_digests["subject"]
            expected_signer = loaded.expected_signer_fingerprint
            expected_request_digest = loaded.expected_request_digest
            verifier_signing_key = loaded.verifier_signing_key_path
            verifier_signing_key_bytes = loaded.verifier_signing_key_bytes
            verifier_identity = loaded.verifier_identity
            allow_installed_scorers = loaded.allow_installed_scorers
            trust_profile_digest = loaded.profile_digest
        result = verify_evidence(
            evidence,
            policy_path=policy,
            expected_baseline_artifact=expected_baseline_artifact,
            expected_subject_artifact=expected_subject_artifact,
            expected_schedule=expected_schedule,
            expected_baseline_runtime=expected_baseline_runtime,
            expected_subject_runtime=expected_subject_runtime,
            expected_signer=expected_signer,
            expected_request_digest=expected_request_digest,
            receipt_path=receipt,
            verifier_signing_key_path=verifier_signing_key,
            verifier_identity=verifier_identity,
            trust_profile_digest=trust_profile_digest,
            scorer_registry=ScorerExtensionRegistry(
                allow_installed=allow_installed_scorers
            ),
            policy_bytes=policy_bytes,
            verifier_signing_key_bytes=verifier_signing_key_bytes,
        )
    except EvidenceVerificationError as exc:
        if json_out:
            typer.echo(exc.as_json())
        else:
            console.print(f"FAIL {exc}")
            if exc.payload.get("integrity_ok") is True:
                console.print("Evidence integrity: verified")
                verdict = exc.payload.get("policy_verdict")
                if verdict in {"pass", "fail"}:
                    console.print(f"Policy result: {verdict}")
            elif "integrity_ok" in exc.payload:
                console.print(
                    "Evidence integrity: not verified; no acceptance established"
                )
            else:
                console.print(
                    "Verification could not complete; check the required inputs and receipt destination."
                )
            for detail in exc.details:
                console.print(detail)
            signed_receipt = exc.payload.get("signed_receipt")
            if isinstance(signed_receipt, str):
                console.print(
                    f"Receipt {exc.receipt_path or signed_receipt}", soft_wrap=True
                )
        raise typer.Exit(exc.exit_code) from exc
    if json_out:
        typer.echo(result.as_json())
    else:
        console.print("PASS Independent verification complete")
        console.print("Evidence integrity: verified")
        if result.payload.get("policy_verdict") in {"pass", "fail"}:
            console.print(f"Policy result: {result.payload['policy_verdict']}")
        console.print(result.summary, soft_wrap=True)


@app.command(
    name="report",
    help="Summarize an evidence pack in the terminal and optionally write an HTML report.",
)
def report(
    evidence: Path = typer.Argument(
        ...,
        metavar="EVIDENCE",
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
        rich_help_panel="Output and workflow",
    ),
    explain: bool = typer.Option(
        False,
        "--explain",
        help="Include a concise explanation of the decision and evidence bindings.",
        rich_help_panel="Output and workflow",
    ),
    json_out: bool = typer.Option(
        False,
        "--json",
        help="Emit one machine-readable rendering result object.",
        rich_help_panel="Output and workflow",
    ),
) -> None:
    """Render EVIDENCE without changing any evidence-pack byte."""

    from invarlock.evidence_reporting import EvidenceReportError, render_evidence

    try:
        result = render_evidence(evidence, html_path=html, explain=explain)
    except EvidenceReportError as exc:
        if json_out:
            typer.echo(
                json.dumps(
                    {
                        "format_version": "invarlock/evidence-report-v1",
                        "ok": False,
                        "errors": [str(exc)],
                    },
                    allow_nan=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
            )
        else:
            console.print(f"FAIL {exc}")
        raise typer.Exit(exc.exit_code) from exc
    if json_out:
        typer.echo(
            json.dumps(
                {
                    "format_version": "invarlock/evidence-report-v1",
                    "ok": True,
                    "pack_manifest_digest": result.pack_manifest_digest,
                    "html": (
                        str(result.html_path) if result.html_path is not None else None
                    ),
                },
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        )
    else:
        console.print(Markdown(result.text))
        if result.html_path is not None:
            console.print(f"HTML {result.html_path}", soft_wrap=True)


def main() -> None:
    """Run the installed command-line entry point."""

    app()


if __name__ == "__main__":  # pragma: no cover
    main()


__all__ = ["app", "main"]
