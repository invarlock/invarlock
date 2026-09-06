"""The InvarLock command line.

The public workflow intentionally has three transactions: evaluate one closed
request, independently verify its evidence, and render that verified evidence.
Provider qualification and repository maintenance are separate tools rather than
alternate user journeys hidden behind this CLI.
"""

from __future__ import annotations

import json
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
    preflight: bool = typer.Option(
        False,
        "--preflight",
        help=(
            "Validate request, authenticated inputs, provider resources, local "
            "runtime images, key, and destination without workers or publication."
        ),
    ),
    request_root: Path | None = typer.Option(
        None,
        "--request-root",
        hidden=True,
        help="Resolve request-relative inputs against this authenticated directory.",
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
    runtime_cpus: str | None = typer.Option(
        None,
        "--runtime-cpus",
        envvar="INVARLOCK_RUNTIME_CPUS",
        help="Hard CPU limit applied independently to each OCI worker.",
    ),
    runtime_memory_mib: str | None = typer.Option(
        None,
        "--runtime-memory-mib",
        envvar="INVARLOCK_RUNTIME_MEMORY_MIB",
        help="Hard memory limit in MiB applied independently to each OCI worker.",
    ),
    runtime_user: str | None = typer.Option(
        None,
        "--runtime-user",
        envvar="INVARLOCK_RUNTIME_USER",
        help="Non-root numeric UID:GID used by both OCI workers.",
    ),
) -> None:
    """Produce the canonical evidence pack described by REQUEST."""

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
        runtime_executor = None
        launch = None
        if loaded_request.execution.mode == "run":
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
        console.print("PASS Preflight complete")
        console.print("No execution or publication was performed")
    else:
        assert isinstance(result, EvaluationTransactionResult)
        console.print("PASS Evidence pack published")
        console.print(str(result.evidence_path))


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
    expected_request_digest: str | None = typer.Option(
        None,
        "--expected-request-digest",
        envvar="INVARLOCK_EXPECTED_REQUEST_DIGEST",
        help=(
            "Independent expected normalized request digest; required when either "
            "side uses llama_cpp."
        ),
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
    ),
    explain: bool = typer.Option(
        False,
        "--explain",
        help="Include a concise explanation of the decision and evidence bindings.",
    ),
    json_out: bool = typer.Option(
        False,
        "--json",
        help="Emit one machine-readable rendering result object.",
    ),
) -> None:
    """Render EVIDENCE without changing any evidence-pack byte."""

    from invarlock.evidence_reporting import EvidenceReportError, render_evidence

    try:
        result = render_evidence(evidence, html_path=html, explain=explain)
    except EvidenceReportError as exc:
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
        console.print(result.text)
        if result.html_path is not None:
            console.print(f"HTML {result.html_path}")


def main() -> None:
    """Run the installed command-line entry point."""

    app()


if __name__ == "__main__":  # pragma: no cover
    main()


__all__ = ["app", "main"]
