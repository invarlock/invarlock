"""Mode selection and execution for the public ``evaluate`` command."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

if TYPE_CHECKING:  # pragma: no cover - imports exist only for static analysis
    from invarlock.captured_evaluation import (
        CapturedEvaluationPreflightResult,
        CapturedEvaluationTransactionResult,
    )
    from invarlock.cli.runtime_profile import ResolvedRuntimeProfile, RuntimeProfile
    from invarlock.evaluation_oci import OciEvaluationLaunch
    from invarlock.evaluation_transaction import (
        EvaluationPreflightResult,
        EvaluationTransactionResult,
    )

type RequestMode = Literal["captured", "runtime", "run", "import"]
type EvaluationResult = (
    CapturedEvaluationPreflightResult
    | CapturedEvaluationTransactionResult
    | EvaluationPreflightResult
    | EvaluationTransactionResult
)


RUNTIME_ONLY_OPTIONS = frozenset(
    {
        "allow_installed_scorers",
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
)
RUN_ONLY_OPTIONS = RUNTIME_ONLY_OPTIONS - {"allow_installed_scorers"}


@dataclass(frozen=True)
class EvaluationOptions:
    """Resolved command values needed after setup-action handling."""

    request: Path
    signing_key: Path | None
    allow_installed_scorers: bool
    preflight: bool
    unsigned: bool
    max_bootstrap_draws: int
    request_root: Path | None
    baseline_run: Path | None
    subject_run: Path | None
    output: Path | None
    runtime_profile: Path | None
    runtime_image: str | None
    runtime_image_digest: str | None
    baseline_runtime_image: str | None
    baseline_runtime_image_digest: str | None
    subject_runtime_image: str | None
    subject_runtime_image_digest: str | None
    container_engine: str | None
    runtime_device: str | None
    baseline_runtime_device: str | None
    subject_runtime_device: str | None
    runtime_entrypoint: str | None
    baseline_runtime_entrypoint: str | None
    subject_runtime_entrypoint: str | None
    runtime_cpus: str | None
    runtime_memory_mib: str | None
    runtime_user: str | None

    def runtime_strings(self) -> dict[str, str]:
        """Return explicit-string candidates using their CLI parameter names."""

        return {
            name: value
            for name in RUNTIME_ONLY_OPTIONS
            if isinstance(value := getattr(self, name), str)
        }


@dataclass(frozen=True)
class EvaluationOutcome:
    """Result plus the runtime context needed by the terminal renderer."""

    result: EvaluationResult
    request_mode: RequestMode
    profile: RuntimeProfile | None = None
    profile_context: ResolvedRuntimeProfile | None = None
    launch: OciEvaluationLaunch | None = None


def execute_evaluation(
    options: EvaluationOptions,
    *,
    command_line: frozenset[str],
    initial_mode: RequestMode | None,
    environment: Mapping[str, str] | None = None,
) -> EvaluationOutcome:
    """Load the request, enforce mode-specific options, and execute or preflight."""

    from invarlock.captured_evaluation import CapturedEvaluationError
    from invarlock.cli.runtime_profile import (
        RuntimeProfileError,
        load_runtime_profile,
        resolve_runtime_profile,
    )
    from invarlock.core.evaluation_request import (
        CapturedEvaluationRequest,
        EvaluationRequest,
        EvaluationRequestError,
        load_evaluation_request,
    )
    from invarlock.core.registry import CoreRegistry
    from invarlock.core.scorer_extension import ScorerExtensionRegistry
    from invarlock.evaluation_oci import (
        OciRuntimeExecutor,
        launch_from_environment,
        preflight_oci_launch,
    )
    from invarlock.evaluation_transaction import (
        evaluate_request_file,
        preflight_evaluation_request,
    )

    request_path = options.request
    if not request_path.is_file():
        raise EvaluationRequestError(
            f"evaluation request is unavailable: {request_path}"
        )
    overrides = {
        name: value
        for name in ("baseline_run", "subject_run", "output")
        if (value := getattr(options, name)) is not None
    }

    loaded_request: object | None = None
    if initial_mode is None:
        loaded_request = load_evaluation_request(
            request_path,
            request_root=options.request_root,
            **overrides,
        )
        request_mode: RequestMode = (
            "captured"
            if isinstance(loaded_request, CapturedEvaluationRequest)
            else "runtime"
        )
    else:
        request_mode = initial_mode

    if request_mode == "captured":
        if loaded_request is None:
            loaded_request = load_evaluation_request(
                request_path,
                request_root=options.request_root,
                **overrides,
            )
        if not isinstance(loaded_request, CapturedEvaluationRequest):
            raise CapturedEvaluationError(
                "captured evaluation request did not load as captured evidence"
            )
        if RUNTIME_ONLY_OPTIONS & command_line:
            raise CapturedEvaluationError(
                "runtime options are not valid for captured evaluation"
            )
        effective_signing_key = options.signing_key
        if options.unsigned and "signing_key" not in command_line:
            effective_signing_key = None
        captured_result: (
            CapturedEvaluationPreflightResult | CapturedEvaluationTransactionResult
        )
        if options.preflight:
            captured_result = preflight_evaluation_request(
                loaded_request,
                signing_key_path=effective_signing_key,
                unsigned=options.unsigned,
                max_bootstrap_draws=options.max_bootstrap_draws,
            )
        else:
            captured_result = evaluate_request_file(
                loaded_request,
                signing_key_path=effective_signing_key,
                unsigned=options.unsigned,
                max_bootstrap_draws=options.max_bootstrap_draws,
            )
        return EvaluationOutcome(result=captured_result, request_mode="captured")

    if "max_bootstrap_draws" in command_line:
        raise EvaluationRequestError(
            "--max-bootstrap-draws applies only to captured evaluation"
        )
    if options.unsigned:
        raise EvaluationRequestError("--unsigned applies only to captured evaluation")

    scorer_registry = ScorerExtensionRegistry(
        allow_installed=options.allow_installed_scorers
    )
    registry = CoreRegistry()
    if loaded_request is None:
        loaded_runtime_request = load_evaluation_request(
            request_path,
            provider_resolver=registry.get_runtime_provider,
            request_root=options.request_root,
            **overrides,
        )
    else:
        loaded_runtime_request = cast(EvaluationRequest, loaded_request)
    if isinstance(loaded_runtime_request, CapturedEvaluationRequest):
        raise EvaluationRequestError(
            "captured evaluation requests must use the captured evaluation path"
        )
    loaded_mode = loaded_runtime_request.execution.mode
    if initial_mode in {"run", "import"} and initial_mode != loaded_mode:
        raise EvaluationRequestError(
            "evaluation request execution mode changed while loading"
        )
    request_mode = loaded_mode
    invalid_run_options = RUN_ONLY_OPTIONS & command_line
    if loaded_mode != "run" and invalid_run_options:
        rendered = ", ".join(
            f"--{name.replace('_', '-')}" for name in sorted(invalid_run_options)
        )
        raise RuntimeProfileError(
            f"{rendered} applies only to run requests; import evidence already binds its runtime"
        )

    profile = None
    profile_context = None
    if options.runtime_profile is not None:
        profile = load_runtime_profile(options.runtime_profile)
        explicit = {
            name: value
            for name, value in options.runtime_strings().items()
            if name in command_line
        }
        profile_context = resolve_runtime_profile(
            profile,
            explicit=explicit,
            environment=dict(os.environ if environment is None else environment),
        )

    launch = None
    runtime_executor = None
    if loaded_mode == "run":
        if profile_context is not None:
            launch = launch_from_environment(**profile_context.arguments)
        else:
            launch = launch_from_environment(
                engine=options.container_engine,
                image_ref=options.runtime_image,
                image_digest=options.runtime_image_digest,
                baseline_image_ref=options.baseline_runtime_image,
                baseline_image_digest=options.baseline_runtime_image_digest,
                subject_image_ref=options.subject_runtime_image,
                subject_image_digest=options.subject_runtime_image_digest,
                default_device=options.runtime_device,
                baseline_device=options.baseline_runtime_device,
                subject_device=options.subject_runtime_device,
                runtime_entrypoint=options.runtime_entrypoint,
                baseline_entrypoint=options.baseline_runtime_entrypoint,
                subject_entrypoint=options.subject_runtime_entrypoint,
                runtime_cpus=options.runtime_cpus,
                runtime_memory_mib=options.runtime_memory_mib,
                runtime_user=options.runtime_user,
            )
        runtime_executor = OciRuntimeExecutor(launch)
    runtime_digests = preflight_oci_launch(launch) if launch else None
    runtime_result: EvaluationPreflightResult | EvaluationTransactionResult
    if options.preflight:
        runtime_result = preflight_evaluation_request(
            loaded_runtime_request,
            signing_key_path=options.signing_key,
            scorer_registry=scorer_registry,
            runtime_image_digests=runtime_digests,
            resource_resolver=runtime_executor,
            registry=registry,
        )
    else:
        runtime_result = evaluate_request_file(
            loaded_runtime_request,
            signing_key_path=options.signing_key,
            runtime_executor=runtime_executor,
            runtime_image_digests=runtime_digests,
            scorer_registry=scorer_registry,
            registry=registry,
        )
    return EvaluationOutcome(
        result=runtime_result,
        request_mode=request_mode,
        profile=profile,
        profile_context=profile_context,
        launch=launch,
    )
