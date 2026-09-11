"""The InvarLock command line."""

from __future__ import annotations

import json
import os
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Literal

import click
import typer
from rich.console import Console
from rich.markdown import Markdown
from typer.core import TyperGroup

from invarlock.evaluation_comparison.capacity import DEFAULT_MAX_BOOTSTRAP_DRAWS
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
        "Evaluate, independently verify, and report on release evidence.\n"
        "\n"
        "  Evaluate: invarlock evaluate request.yaml\n"
        "  invarlock verify evidence/\n"
        "  invarlock report evidence/"
    ),
)
console = Console(markup=False, highlight=False)


def _setup_result(
    action: str | None,
    *,
    details: dict[str, Any] | None = None,
    errors: list[str] | None = None,
) -> dict[str, Any]:
    from jsonschema import Draft202012Validator

    from invarlock.public_contracts import load_evaluation_setup_result_schema

    result = {
        "format_version": "invarlock/evaluation-setup-v1",
        "action": action,
        "ok": not errors,
        "details": details,
        "errors": [error[:1024] for error in (errors or [])[:16]],
    }
    Draft202012Validator(load_evaluation_setup_result_schema()).validate(result)
    return result


def _write_setup_file(path: Path, payload: bytes) -> None:
    from invarlock.captured_contracts import atomic_write

    atomic_write(path, payload)


def _publish_setup_directory(directory: Path, artifacts: dict[str, bytes]) -> None:
    import secrets
    import shutil

    from invarlock.captured_contracts import secure_directory
    from invarlock.filesystem import publish_directory_no_replace

    directory = directory.absolute()
    with secure_directory(directory.parent, create=True) as parent:
        if directory.exists() or directory.is_symlink():
            raise ValueError("setup directory must not already exist")
        name = ".evaluation-setup-" + secrets.token_hex(16)
        os.mkdir(name, mode=0o700, dir_fd=parent)
        staging = directory.parent / name
        try:
            for relative, raw in artifacts.items():
                _write_setup_file(staging / relative, raw)
            publish_directory_no_replace(staging, directory)
        finally:
            try:
                shutil.rmtree(name, dir_fd=parent)
            except FileNotFoundError:
                pass


def _run_setup_action(
    action: str,
    *,
    directory: Path | None,
    example: str,
    case_file: Path | None,
    case_set_output: Path | None,
    json_out: bool,
) -> None:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    from invarlock.captured_contracts import read_file
    from invarlock.evaluation_record_contracts.contracts import (
        MAX_INPUT_BYTES,
        EvaluationRecordsError,
    )
    from invarlock.evaluation_records.cases import canonical_case_set, case_set_digest
    from invarlock.evaluation_records.templates import example_project
    from invarlock.evidence_pack_contract import canonical_json_bytes
    from invarlock.evidence_pack_integrity import public_key_fingerprint
    from invarlock.evidence_pack_json import parse_json_bytes

    details: dict[str, object]
    try:
        if action == "init":
            assert directory is not None
            if directory.exists():
                raise EvaluationRecordsError("init directory must not already exist")
            baseline, subject, policy = example_project(example)
            request = {
                "format_version": "invarlock/evaluation-request-v2",
                "execution": {"mode": "captured"},
                "comparison": {
                    "baseline": {
                        "path": "inputs/baseline.json",
                        "adapter": "invarlock",
                    },
                    "subject": {
                        "path": "inputs/subject.json",
                        "adapter": "invarlock",
                    },
                    "policy": "policy.json",
                },
                "output": {"evidence": "artifacts/evidence"},
            }
            readme = (
                "# InvarLock captured evaluation example\n\n"
                "The JSON inputs are honest synthetic captured results. Replace them "
                "with independently produced records before relying on a decision.\n"
                "Run these commands from this directory:\n\n"
                "invarlock evaluate request.yaml --unsigned --json\n"
                "invarlock report artifacts/evidence --html artifacts/report.html "
                "--markdown artifacts/summary.md --junit artifacts/junit.xml --explain\n"
                "\nUnsigned reports are local, not independent verification or native "
                "acceptance. For a signed handoff use --signing-key, then verify with "
                "a recipient-owned --trust-profile (invarlock/trust-inputs-v2) and "
                "--receipt outside the pack. See docs/user-guide/captured-results.md.\n"
                "Add --fail-on-policy to evaluate for a local gate: 0 pass, 7 adverse "
                "decision, 2 input/work-budget failure. Report only a newly published "
                "pack. Commands default to text; --json emits captured evaluation/"
                "verification/report v2 results. Report output maps are requested_outputs "
                "and written_outputs, with failed_output and errors on write failure.\n"
            )
            _publish_setup_directory(
                directory,
                {
                    "request.yaml": canonical_json_bytes(request),
                    "inputs/baseline.json": canonical_json_bytes(baseline),
                    "inputs/subject.json": canonical_json_bytes(subject),
                    "policy.json": canonical_json_bytes(policy),
                    "README.txt": readme.encode("utf-8"),
                },
            )
            details = {
                "directory": str(directory),
                "request": str(directory / "request.yaml"),
                "example": example,
            }
        elif action == "keygen":
            assert directory is not None
            if directory.exists():
                raise EvaluationRecordsError("keygen directory must not already exist")
            key = Ed25519PrivateKey.generate()
            private = key.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.PKCS8,
                serialization.NoEncryption(),
            )
            public = key.public_key().public_bytes(
                serialization.Encoding.PEM,
                serialization.PublicFormat.SubjectPublicKeyInfo,
            )
            _publish_setup_directory(
                directory, {"private.pem": private, "public.pem": public}
            )
            details = {
                "private_key": str(directory / "private.pem"),
                "public_key": str(directory / "public.pem"),
                "public_key_fingerprint": public_key_fingerprint(key.public_key()),
            }
        else:
            assert case_file is not None
            document = parse_json_bytes(
                read_file(case_file, MAX_INPUT_BYTES), label="case set"
            )
            if not isinstance(document, dict):
                raise EvaluationRecordsError("planned case set must be an object")
            cases = canonical_case_set(document)
            output = None
            if case_set_output is not None:
                _write_setup_file(case_set_output, canonical_json_bytes(cases))
                output = str(case_set_output)
            details = {
                "case_count": len(cases["cases"]),
                "case_set_digest": case_set_digest(cases),
                "output": output,
            }
        setup_result = _setup_result(action, details=details)
    except (
        EvaluationRecordsError,
        OSError,
        TypeError,
        ValueError,
        RuntimeError,
    ) as exc:
        setup_result = _setup_result(action, errors=[str(exc)])
    if json_out:
        typer.echo(canonical_json_bytes(setup_result).decode("utf-8"))
    elif setup_result["ok"]:
        assert setup_result["details"] is not None
        for key, value in setup_result["details"].items():
            console.print(f"{key}: {value}")
    else:
        for error in setup_result["errors"]:
            console.print(f"FAIL {error}")
    if not setup_result["ok"]:
        raise typer.Exit(2)


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


def _finish_policy_gate(verdict: str | None) -> None:
    if verdict == "pass":
        return
    if verdict in {"fail", "regression", "insufficient_evidence"}:
        raise typer.Exit(7)
    typer.echo(
        "Evidence was published, but its policy outcome is unavailable for gating.",
        err=True,
    )
    raise typer.Exit(2)


def _print_captured_metrics(metrics: tuple[dict[str, Any], ...]) -> None:
    for metric in metrics:
        console.print(
            f"{metric['name']} / {metric['slice']}: {metric['decision']}; "
            f"{metric['usable_count']} usable pairs; "
            f"{metric['missing_count']} missing results"
        )
        if metric["reasons"]:
            console.print(f"Recorded reasons: {'; '.join(metric['reasons'])}")


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
    help=(
        "Evaluate one closed request using native execution, authenticated imports, "
        "or captured results.\n\n"
        "Signed handoff: evaluate REQUEST -> verify EVIDENCE -> report EVIDENCE.\n\n"
        "Unsigned local use: evaluate REQUEST --unsigned -> report EVIDENCE."
    ),
)
def evaluate(  # noqa: C901
    ctx: typer.Context,
    request: Path | None = typer.Argument(
        None,
        metavar="REQUEST",
        exists=False,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=False,
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
    fail_on_policy: bool = typer.Option(
        False,
        "--fail-on-policy",
        help="Exit 7 after publication when the recorded policy does not pass.",
        rich_help_panel="Output and workflow",
    ),
    unsigned: bool = typer.Option(
        False,
        "--unsigned",
        help="Explicitly publish captured evaluation as unsigned local evidence.",
        rich_help_panel="Signing and execution authorization",
    ),
    max_bootstrap_draws: int = typer.Option(
        DEFAULT_MAX_BOOTSTRAP_DRAWS,
        "--max-bootstrap-draws",
        min=0,
        help="Caller-owned work allowance for the complete captured comparison.",
        rich_help_panel="Signing and execution authorization",
    ),
    request_root: Path | None = typer.Option(
        None,
        "--request-root",
        hidden=True,
        help="Resolve request-relative inputs against this authenticated directory.",
        rich_help_panel="Output and workflow",
    ),
    baseline_run: Path | None = typer.Option(
        None,
        "--baseline-run",
        help="Captured baseline override, caller-relative and confined to request root.",
        rich_help_panel="Output and workflow",
    ),
    subject_run: Path | None = typer.Option(
        None,
        "--subject-run",
        help="Captured subject override, caller-relative and confined to request root.",
        rich_help_panel="Output and workflow",
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        help="Evidence destination override, caller-relative and confined to request root.",
        rich_help_panel="Output and workflow",
    ),
    init_directory: Path | None = typer.Option(
        None,
        "--init",
        metavar="DIRECTORY",
        help="Create a captured evaluation example without a request.",
        rich_help_panel="Evaluation setup",
    ),
    example: str = typer.Option(
        "classification",
        "--example",
        help="Example for --init: classification, extraction, or judge.",
        rich_help_panel="Evaluation setup",
    ),
    keygen_directory: Path | None = typer.Option(
        None,
        "--keygen",
        metavar="DIRECTORY",
        help="Create generic Ed25519 private/public key material.",
        rich_help_panel="Evaluation setup",
    ),
    freeze_cases: Path | None = typer.Option(
        None,
        "--freeze-cases",
        metavar="FILE",
        help="Validate and optionally canonicalize an evaluation case set.",
        rich_help_panel="Evaluation setup",
    ),
    case_set_output: Path | None = typer.Option(
        None,
        "--case-set-output",
        metavar="FILE",
        help="No-replace output for --freeze-cases.",
        rich_help_panel="Evaluation setup",
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

    setup_values = (init_directory, keygen_directory, freeze_cases)
    selected = [value is not None for value in setup_values]
    action = (
        "init"
        if init_directory is not None
        else "keygen"
        if keygen_directory is not None
        else "freeze_cases"
        if freeze_cases is not None
        else None
    )
    setup_option_names = {
        "baseline_run",
        "subject_run",
        "output",
        "signing_key",
        "allow_installed_scorers",
        "preflight",
        "fail_on_policy",
        "unsigned",
        "max_bootstrap_draws",
        "request_root",
        "runtime_profile",
        "runtime_image",
        "runtime_image_digest",
        "baseline_runtime_image",
        "baseline_runtime_image_digest",
        "subject_runtime_image",
        "subject_runtime_image_digest",
        "container_engine",
        "runtime_device",
        "baseline_runtime_device",
        "subject_runtime_device",
        "runtime_entrypoint",
        "baseline_runtime_entrypoint",
        "subject_runtime_entrypoint",
        "runtime_cpus",
        "runtime_memory_mib",
        "runtime_user",
    }
    explicit_setup_controls = [
        name
        for name in setup_option_names
        if getattr(ctx.get_parameter_source(name), "name", None) == "COMMANDLINE"
    ]
    setup_error: str | None = None
    if sum(selected) > 1:
        setup_error = "setup actions are mutually exclusive"
    elif action is not None:
        if request is not None:
            setup_error = "setup actions do not accept a positional request"
        elif explicit_setup_controls:
            setup_error = (
                "setup actions cannot be combined with evaluation options: "
                + ", ".join(sorted(explicit_setup_controls))
            )
        elif action == "init" and (
            case_set_output is not None
            or freeze_cases is not None
            or keygen_directory is not None
        ):
            setup_error = "--init cannot be combined with other setup actions"
        elif action == "keygen" and (
            getattr(ctx.get_parameter_source("example"), "name", None) == "COMMANDLINE"
            or case_set_output is not None
        ):
            setup_error = "--keygen accepts no setup options"
        elif (
            action == "freeze_cases"
            and getattr(ctx.get_parameter_source("example"), "name", None)
            == "COMMANDLINE"
        ):
            setup_error = "--freeze-cases cannot use --example"
    elif request is None:
        setup_error = "a request or exactly one setup action is required"
    elif (
        case_set_output is not None
        or getattr(ctx.get_parameter_source("example"), "name", None) == "COMMANDLINE"
    ):
        setup_error = "--example and --case-set-output require a setup action"
    if setup_error is not None:
        result = _setup_result(None, errors=[setup_error]) if json_out else None
        if json_out:
            from invarlock.evidence_pack_contract import canonical_json_bytes

            typer.echo(canonical_json_bytes(result).decode("utf-8"))
        else:
            console.print(f"FAIL {setup_error}")
        raise typer.Exit(2)
    if preflight and fail_on_policy:
        raise typer.BadParameter("--fail-on-policy cannot be used with --preflight")
    if action is not None:
        _run_setup_action(
            action,
            directory=init_directory or keygen_directory,
            example=example,
            case_file=freeze_cases,
            case_set_output=case_set_output,
            json_out=json_out,
        )
        return

    from invarlock.captured_evaluation import (
        CapturedEvaluationError,
    )
    from invarlock.cli.runtime_profile import (
        ResolvedRuntimeProfile,
        RuntimeProfile,
        RuntimeProfileError,
        load_runtime_profile,
        resolve_runtime_profile,
    )
    from invarlock.core.evaluation_request import (
        CapturedEvaluationRequest,
        EvaluationRequestError,
        evaluation_request_mode,
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
    evaluation_result: EvaluationPreflightResult | EvaluationTransactionResult
    assert request is not None
    request_path = request
    request_mode: Literal["captured", "runtime", "run", "import"] = "runtime"
    overrides: dict[str, Any] = {
        name: value
        for name, value in (
            ("baseline_run", baseline_run),
            ("subject_run", subject_run),
            ("output", output),
        )
        if value is not None
    }
    try:
        if not request_path.is_file():
            raise EvaluationRequestError(
                f"evaluation request is unavailable: {request_path}"
            )
        # Discriminate the strict request before constructing the runtime registry.
        # The fallback keeps callable-level tests and integrations that replace the
        # loader observable without changing the real strict-loader path.
        loaded_request: object | None = None
        try:
            request_mode = evaluation_request_mode(request_path)
        except EvaluationRequestError:
            loaded_request = load_evaluation_request(
                request_path, request_root=request_root, **overrides
            )
            request_mode = (
                "captured"
                if isinstance(loaded_request, CapturedEvaluationRequest)
                else "runtime"
            )
        if request_mode == "captured":
            if loaded_request is None:
                loaded_request = load_evaluation_request(
                    request_path, request_root=request_root, **overrides
                )
            assert isinstance(loaded_request, CapturedEvaluationRequest)
            runtime_names = setup_option_names - {
                "signing_key",
                "preflight",
                "fail_on_policy",
                "unsigned",
                "max_bootstrap_draws",
                "request_root",
                "baseline_run",
                "subject_run",
                "output",
            }
            if any(
                getattr(ctx.get_parameter_source(name), "name", None) == "COMMANDLINE"
                for name in runtime_names
            ):
                raise CapturedEvaluationError(
                    "runtime options are not valid for captured evaluation"
                )
            effective_signing_key = signing_key
            if (
                unsigned
                and getattr(ctx.get_parameter_source("signing_key"), "name", None)
                != "COMMANDLINE"
            ):
                effective_signing_key = None
            if preflight:
                captured_preflight = preflight_evaluation_request(
                    loaded_request,
                    signing_key_path=effective_signing_key,
                    unsigned=unsigned,
                    max_bootstrap_draws=max_bootstrap_draws,
                )
                if json_out:
                    typer.echo(captured_preflight.as_json())
                else:
                    console.print("Preflight complete")
                    console.print(
                        f"Mode: {captured_preflight.execution_mode}; paired records: "
                        f"{captured_preflight.record_count}"
                    )
                    console.print(
                        "Requested authentication: "
                        f"{captured_preflight.requested_authentication}"
                    )
                    console.print(
                        "No execution, signing, scoring, or publication was performed"
                    )
                return
            captured_result = evaluate_request_file(
                loaded_request,
                signing_key_path=effective_signing_key,
                unsigned=unsigned,
                max_bootstrap_draws=max_bootstrap_draws,
            )
            if json_out:
                typer.echo(captured_result.as_json())
            else:
                console.print("Captured evidence created")
                console.print(
                    f"Recorded policy result: {captured_result.policy_verdict}"
                )
                console.print(
                    "Signing: Signed evidence"
                    if captured_result.authentication == "signed"
                    else "Signing: Unsigned local evidence"
                )
                console.print("Independent verification: not performed")
                _print_captured_metrics(captured_result.metric_summaries)
                console.print(f"Evidence: {captured_result.evidence_path}")
            if fail_on_policy:
                _finish_policy_gate(captured_result.policy_verdict)
            return

        if (
            getattr(ctx.get_parameter_source("max_bootstrap_draws"), "name", None)
            == "COMMANDLINE"
        ):
            raise EvaluationRequestError(
                "--max-bootstrap-draws applies only to captured evaluation"
            )
        if unsigned:
            raise EvaluationRequestError(
                "--unsigned applies only to captured evaluation"
            )
        scorer_registry = ScorerExtensionRegistry(
            allow_installed=allow_installed_scorers
        )
        registry = CoreRegistry()
        loaded_runtime_request = load_evaluation_request(
            request_path,
            provider_resolver=registry.get_runtime_provider,
            request_root=request_root,
            **overrides,
        )
        if isinstance(loaded_runtime_request, CapturedEvaluationRequest):
            raise EvaluationRequestError(
                "captured evaluation requests must use the captured evaluation path"
            )
        if runtime_profile is not None:
            if loaded_runtime_request.execution.mode != "run":
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
        if loaded_runtime_request.execution.mode == "run":
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
            evaluation_result = preflight_evaluation_request(
                loaded_runtime_request,
                signing_key_path=signing_key,
                scorer_registry=scorer_registry,
                runtime_image_digests=runtime_digests,
                resource_resolver=runtime_executor,
                registry=registry,
            )
        else:
            runtime_digests = preflight_oci_launch(launch) if launch else None
            evaluation_result = evaluate_request_file(
                loaded_runtime_request,
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
        CapturedEvaluationError,
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
        if request_mode == "captured":
            failure.captured = True
            if isinstance(failure, EvaluationPreflightError):
                failure.unsigned = unsigned
        if json_out:
            typer.echo(failure.as_json())
        else:
            console.print(f"FAIL {failure}", markup=False)
        raise typer.Exit(failure.exit_code) from exc
    if json_out:
        typer.echo(evaluation_result.as_json())
    elif preflight:
        console.print("Preflight complete")
        assert isinstance(evaluation_result, EvaluationPreflightResult)
        console.print(
            f"Mode: {evaluation_result.execution_mode}; paired records: {evaluation_result.record_count}"
        )
        console.print(
            f"Evidence destination: {evaluation_result.output}", soft_wrap=True
        )
        console.print(f"Validated checks: {len(evaluation_result.checks)}")
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
        assert isinstance(evaluation_result, EvaluationTransactionResult)
        console.print("Evidence created")
        console.print(
            f"Recorded policy result: {evaluation_result.policy_verdict or 'unavailable'}"
        )
        console.print("Recipient verification: not performed")
        console.print(f"Evidence: {evaluation_result.evidence_path}", soft_wrap=True)
        console.print(
            "Next: verify with independently approved trust inputs; use report to inspect the recorded checks."
        )
    if fail_on_policy:
        assert isinstance(evaluation_result, EvaluationTransactionResult)
        _finish_policy_gate(evaluation_result.policy_verdict)


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
        resolve_path=False,
        help="Canonical evidence-pack directory.",
    ),
    trust_profile: Path | None = typer.Option(
        None,
        "--trust-profile",
        help=(
            "Closed native v1 or captured v2 trust profile. Explicit trust-anchor "
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
    expected_baseline_run: str | None = typer.Option(
        None,
        "--expected-baseline-run",
        envvar="INVARLOCK_EXPECTED_BASELINE_RUN",
        help="Independent expected complete captured baseline run digest.",
        rich_help_panel="Independent trust anchors",
    ),
    expected_subject_run: str | None = typer.Option(
        None,
        "--expected-subject-run",
        envvar="INVARLOCK_EXPECTED_SUBJECT_RUN",
        help="Independent expected complete captured subject run digest.",
        rich_help_panel="Independent trust anchors",
    ),
    max_bootstrap_draws: int | None = typer.Option(
        DEFAULT_MAX_BOOTSTRAP_DRAWS,
        "--max-bootstrap-draws",
        help="Recipient-owned captured replay work budget.",
        rich_help_panel="Signing and execution authorization",
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

    from invarlock.captured_reporting import CapturedReportError, is_captured_manifest
    from invarlock.core.scorer_extension import ScorerExtensionRegistry
    from invarlock.evidence_verification import (
        EvidenceVerificationError,
        _require_outside_evidence,
        verify_evidence,
    )
    from invarlock.trust_inputs import (
        CapturedTrustInputs,
        TrustInputsError,
        load_trust_inputs,
    )

    captured = False
    try:
        if not evidence.is_dir() or evidence.is_symlink():
            raise EvidenceVerificationError("evidence must be a real directory")
        try:
            captured = is_captured_manifest(evidence)
        except CapturedReportError as exc:
            raise EvidenceVerificationError(str(exc), exit_code=4) from exc
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
                "expected_baseline_run",
                "expected_subject_run",
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
            if isinstance(loaded, CapturedTrustInputs):
                if not captured:
                    raise EvidenceVerificationError(
                        "captured trust profile requires captured evidence"
                    )
                expected_baseline_run = loaded.expected_run_digests["baseline"]
                expected_subject_run = loaded.expected_run_digests["subject"]
            else:
                if captured:
                    raise EvidenceVerificationError(
                        "native trust profile requires native evidence"
                    )
                expected_baseline_artifact = loaded.expected_artifact_digests[
                    "baseline"
                ]
                expected_subject_artifact = loaded.expected_artifact_digests["subject"]
                expected_schedule = loaded.expected_schedule_digest
                expected_baseline_runtime = loaded.expected_runtime_digests["baseline"]
                expected_subject_runtime = loaded.expected_runtime_digests["subject"]
                allow_installed_scorers = loaded.allow_installed_scorers
            expected_signer = loaded.expected_signer_fingerprint
            expected_request_digest = loaded.expected_request_digest
            verifier_signing_key = loaded.verifier_signing_key_path
            verifier_signing_key_bytes = loaded.verifier_signing_key_bytes
            verifier_identity = loaded.verifier_identity
            trust_profile_digest = loaded.profile_digest
        verification_arguments: dict[str, Any] = {
            "policy_path": policy,
            "expected_signer": expected_signer,
            "expected_request_digest": expected_request_digest,
            "receipt_path": receipt,
            "verifier_signing_key_path": verifier_signing_key,
            "verifier_identity": verifier_identity,
            "trust_profile_digest": trust_profile_digest,
            "policy_bytes": policy_bytes,
            "verifier_signing_key_bytes": verifier_signing_key_bytes,
        }
        irrelevant: tuple[str, ...]
        if captured:
            irrelevant = (
                "expected_baseline_artifact",
                "expected_subject_artifact",
                "expected_schedule",
                "expected_baseline_runtime",
                "expected_subject_runtime",
                "allow_installed_scorers",
            )
            if any(
                getattr(ctx.get_parameter_source(name), "name", None) == "COMMANDLINE"
                for name in irrelevant
            ):
                raise EvidenceVerificationError(
                    "native anchor/scorer flags are not valid for captured verification"
                )
            verification_arguments.update(
                expected_baseline_run=expected_baseline_run,
                expected_subject_run=expected_subject_run,
                max_bootstrap_draws=max_bootstrap_draws,
            )
        else:
            irrelevant = (
                "expected_baseline_run",
                "expected_subject_run",
                "max_bootstrap_draws",
            )
            if any(
                getattr(ctx.get_parameter_source(name), "name", None) == "COMMANDLINE"
                for name in irrelevant
            ):
                raise EvidenceVerificationError(
                    "captured run/work flags are not valid for native verification"
                )
            verification_arguments.update(
                expected_baseline_artifact=expected_baseline_artifact,
                expected_subject_artifact=expected_subject_artifact,
                expected_schedule=expected_schedule,
                expected_baseline_runtime=expected_baseline_runtime,
                expected_subject_runtime=expected_subject_runtime,
                scorer_registry=ScorerExtensionRegistry(
                    allow_installed=allow_installed_scorers
                ),
            )
        result = verify_evidence(evidence, **verification_arguments)
    except EvidenceVerificationError as exc:
        if (
            captured
            and exc.payload.get("format_version")
            == "invarlock/evidence-verification-error-v1"
        ):
            exc = EvidenceVerificationError(
                str(exc), exit_code=exc.exit_code, captured=True
            )
        if json_out:
            typer.echo(exc.as_json())
        else:
            console.print(f"FAIL {exc}")
            if exc.payload.get("integrity_ok") is True:
                console.print("Evidence integrity: verified")
                verdict = exc.payload.get("policy_verdict")
                if verdict in {"pass", "fail"}:
                    console.print(f"Policy result: {verdict}")
                if exc.payload.get("kind") == "captured":
                    console.print("Independent captured replay complete")
                    console.print(f"Recorded decision: {exc.payload['decision']}")
                    _print_captured_metrics(exc.payload["metric_summaries"])
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
        console.print(
            "PASS Independent captured verification complete"
            if result.payload.get("kind") == "captured"
            else "PASS Independent verification complete"
        )
        console.print("Evidence integrity: verified")
        if result.payload.get("policy_verdict") in {"pass", "fail"}:
            console.print(f"Policy result: {result.payload['policy_verdict']}")
        console.print(result.summary, soft_wrap=True)
        if result.payload.get("kind") == "captured":
            console.print(f"Recorded decision: {result.payload['decision']}")
            _print_captured_metrics(result.payload["metric_summaries"])


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
        resolve_path=False,
        help="Canonical evidence-pack directory.",
    ),
    html: Path | None = typer.Option(
        None,
        "--html",
        help="Write a self-contained HTML report outside the evidence pack.",
        rich_help_panel="Output and workflow",
    ),
    markdown: Path | None = typer.Option(
        None, "--markdown", help="Write the report as Markdown outside evidence."
    ),
    junit: Path | None = typer.Option(
        None,
        "--junit",
        help="Write recorded policy checks as JUnit XML outside evidence.",
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

    from invarlock.evidence_reporting import (
        EvidenceReportError,
        EvidenceReportV2,
        render_evidence,
    )

    try:
        destinations = {
            name: value
            for name, value in (("markdown_path", markdown), ("junit_path", junit))
            if value is not None
        }
        result = render_evidence(
            evidence, html_path=html, explain=explain, **destinations
        )
    except EvidenceReportError as exc:
        if json_out:
            typer.echo(
                json.dumps(
                    exc.payload
                    or {
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
            for name, destination in exc.written_outputs.items():
                console.print(f"Written {name}: {destination}", soft_wrap=True)
            if exc.failed_output is not None:
                console.print(f"Failed output: {exc.failed_output}")
        raise typer.Exit(exc.exit_code) from exc
    if isinstance(result, EvidenceReportV2):
        if json_out:
            typer.echo(result.as_json())
        else:
            console.print(Markdown(result.text))
            for name, destination in result.written_outputs.items():
                console.print(f"{name.upper()} {destination}", soft_wrap=True)
        return
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
