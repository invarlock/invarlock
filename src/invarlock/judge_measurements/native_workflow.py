"""Native answer capture, resumable judge collection and evidence publication."""

from __future__ import annotations

import asyncio
import base64
import fcntl
import hashlib
import os
import stat
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from invarlock.captured_contracts import read_file
from invarlock.core.registry import get_registry
from invarlock.core.runtime_provider import ModelRuntimeSpec
from invarlock.evaluation_runtime import caller_runtime_resources_from_environment
from invarlock.evidence_pack_contract import EvidenceObservation, RuntimeSideEvidence
from invarlock.evidence_pack_json import parse_json_bytes
from invarlock.filesystem.atomic_file import write_file_no_replace
from invarlock.filesystem.paths import entry_identity, pinned_directory
from invarlock.judge_measurements.contracts import canonical_payload
from invarlock.judge_measurements.native_recipe import (
    finalize_native_plan,
    prepare_native_judge,
)
from invarlock.judge_measurements.workflow import (
    JudgeWorkflowError,
    JudgeWorkflowResult,
)
from invarlock.security import network_policy_allows

if TYPE_CHECKING:
    from invarlock.core.evaluation_request import (
        ComparisonSideRequest,
        EvaluationRequest,
    )


def collection_api() -> Any:
    """Load the core collector without importing optional provider SDKs."""
    from invarlock import judge_measurements

    return judge_measurements


def _local_collection(configuration: dict[str, Any]) -> bool:
    return configuration.get("profile") == "runtime-provider-text-frozen-answer-v1"


def _service_collection(configuration: dict[str, Any]) -> bool:
    return configuration.get("profile") == "openai-compatible-text-frozen-answer-v1"


def _judge_network_authorized() -> bool:
    return (
        os.environ.get("INVARLOCK_ALLOW_JUDGE_NETWORK", "").strip().lower()
        in {"1", "true", "yes", "on"}
        or network_policy_allows()
    )


def _require_service_network_authorization(
    configuration: dict[str, Any], environment: dict[str, Any]
) -> None:
    if (
        not _local_collection(configuration)
        and environment.get("network_authorized") is False
    ):
        raise JudgeWorkflowError(
            "service judge collection requires an allowed network policy; set "
            "INVARLOCK_ALLOW_JUDGE_NETWORK=1 only for collection. Preflight and "
            "retained-measurement import remain available offline"
        )


def _collection_context(
    configuration: dict[str, Any], *, model: Any, request_root: Path | None, plan: Any
) -> dict[str, Any]:
    if not _local_collection(configuration) and model is not None:
        raise JudgeWorkflowError(
            "service judge collection cannot use a local artifact model"
        )
    if _local_collection(configuration) or _service_collection(configuration):
        return {"model": model, "request_root": request_root, "plan": plan}
    return {}


def _runtime_provider_inputs(
    *, model: ComparisonSideRequest, request_root: Path
) -> tuple[Any, ModelRuntimeSpec, Any]:
    provider = get_registry().get_runtime_provider(model.runtime.provider)
    spec = ModelRuntimeSpec(
        provider_name=model.runtime.provider,
        model_id=model.artifact.model_id,
        settings=model.runtime.settings,
    )
    resources = caller_runtime_resources_from_environment().resolve(
        request_root=request_root,
        role="judge",
        side=model,
        provider=provider,
    )
    return provider, spec, resources


def collection_preflight(
    configuration: dict[str, Any],
    *,
    integration: str | None = None,
    model: ComparisonSideRequest | None = None,
    request_root: Path | None = None,
    plan: Any | None = None,
) -> dict[str, Any]:
    api = collection_api()
    if integration is not None and integration != (
        "runtime-provider-judge"
        if _local_collection(configuration)
        else "openai-compatible-judge"
        if _service_collection(configuration)
        else "inspect-judge"
    ):
        raise JudgeWorkflowError(
            "judge integration differs from its collection profile"
        )
    if _service_collection(configuration):
        if model is not None or plan is None:
            raise JudgeWorkflowError(
                "endpoint judge preflight requires a plan and no local artifact model"
            )
        result = dict(api.preflight_openai_compatible(configuration, plan))
        result.setdefault("network_authorized", _judge_network_authorized())
        result.setdefault(
            "network_authorization_variable", "INVARLOCK_ALLOW_JUDGE_NETWORK"
        )
        return result
    if not _local_collection(configuration) and model is not None:
        raise JudgeWorkflowError(
            "service judge collection cannot use a local artifact model"
        )
    local = _local_collection(configuration)
    if local:
        if model is None or request_root is None or plan is None:
            raise JudgeWorkflowError(
                "runtime judge preflight requires its model, request root, and plan"
            )
        budgets = api.validate_runtime_provider_collection(configuration, plan)
        provider, spec, resources = _runtime_provider_inputs(
            model=model, request_root=request_root
        )
        result = dict(
            api.preflight_runtime_provider(
                plan, provider=provider, spec=spec, resources=resources
            )
        )
        result["budgets"] = budgets
        return result
    options = api.CollectionOptions(**configuration)
    from invarlock.judge_measurements.runner import (
        _require_qualified_live_provider_model,
    )

    try:
        _require_qualified_live_provider_model(options.grader)
    except ValueError as exc:
        raise JudgeWorkflowError(str(exc)) from exc
    result = dict(api.validate_collection_environment(options))
    result.update(
        network_authorized=_judge_network_authorized(),
        network_authorization_variable="INVARLOCK_ALLOW_JUDGE_NETWORK",
    )
    return result


@contextmanager
def locked_workspace(path: Path) -> Iterator[Callable[[], None]]:
    """Serialize capture and collection under a private, pinned workspace."""
    path = Path(path).absolute()
    with pinned_directory(path, create=True) as directory:
        current = os.fstat(directory)
        if current.st_uid != os.geteuid() or stat.S_IMODE(current.st_mode) & 0o077:
            raise JudgeWorkflowError(
                "judge workspace must be caller-owned and private (0700)"
            )
        bindings = [
            (part, entry_identity(part.stat(follow_symlinks=False)))
            for part in (*reversed(path.parents), path)
        ]

        if bindings[-1][1] != entry_identity(os.fstat(directory)):
            raise JudgeWorkflowError("judge workspace changed while opening")

        def unchanged() -> None:
            if any(
                entry_identity(part.stat(follow_symlinks=False)) != identity
                for part, identity in bindings
            ):
                raise JudgeWorkflowError("judge workspace ancestry changed")
            try:
                named = os.stat(
                    ".evaluation.lock", dir_fd=directory, follow_symlinks=False
                )
            except OSError:
                raise JudgeWorkflowError("judge workspace lock changed") from None
            opened = os.fstat(descriptor)
            if entry_identity(named) != entry_identity(opened) or opened.st_nlink != 1:
                raise JudgeWorkflowError("judge workspace lock changed")

        descriptor = os.open(
            ".evaluation.lock",
            os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o600,
            dir_fd=directory,
        )
        try:
            lock = os.fstat(descriptor)
            if (
                not stat.S_ISREG(lock.st_mode)
                or lock.st_uid != os.geteuid()
                or stat.S_IMODE(lock.st_mode) & 0o077
                or lock.st_nlink != 1
            ):
                raise JudgeWorkflowError("judge workspace lock is unsafe")
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise JudgeWorkflowError(
                    "another evaluation is using this judge workspace"
                ) from None
            unchanged()
            yield unchanged
            unchanged()
        finally:
            os.close(descriptor)


def _retain_identity(path: Path, value: dict[str, Any]) -> None:
    raw = canonical_payload(
        {
            "format": "invarlock/judge-workspace-identity-v1",
            "sha256": hashlib.sha256(canonical_payload(value)).hexdigest(),
        }
    )
    try:
        existing = read_file(path, 1024)
    except FileNotFoundError:
        write_file_no_replace(path, raw)
    else:
        if existing != raw:
            raise JudgeWorkflowError(
                "judge workspace belongs to a different frozen request"
            )


def collect_frozen(
    *,
    plan: Any,
    collection: dict[str, Any],
    runner: dict[str, Any],
    workspace: Path,
    baseline_run: dict[str, Any],
    subject_run: dict[str, Any],
    status: dict[str, str] | None = None,
    integration: str | None = None,
    model: ComparisonSideRequest | None = None,
    request_root: Path | None = None,
) -> Any:
    """Resume only unadmitted calls against the same immutable frozen answers."""
    api = collection_api()
    if integration is not None and integration != (
        "runtime-provider-judge"
        if _local_collection(collection)
        else "openai-compatible-judge"
        if _service_collection(collection)
        else "inspect-judge"
    ):
        raise JudgeWorkflowError(
            "judge integration differs from its collection profile"
        )
    if _service_collection(collection):
        if model is not None or set(runner) != {"scorer_id"}:
            raise JudgeWorkflowError(
                "endpoint judge collection requires only scorer_id and no local artifact model"
            )
        if status is not None:
            status.clear()
        result = api.collect_openai_compatible(
            plan,
            configuration=collection,
            baseline_run=baseline_run,
            subject_run=subject_run,
            options=api.OpenAICompatibleJudgeOptions(
                scorer_id=runner["scorer_id"],
                checkpoint_directory=workspace / "endpoint-collection",
            ),
        )
        if status is not None:
            status["stop_reason"] = "complete"
        return result
    if not _local_collection(collection) and model is not None:
        raise JudgeWorkflowError(
            "service judge collection cannot use a local artifact model"
        )
    local = _local_collection(collection)
    if local:
        if model is None or request_root is None:
            raise JudgeWorkflowError(
                "runtime judge collection requires its model and request root"
            )
        if set(runner) != {"scorer_id"}:
            raise JudgeWorkflowError(
                "runtime judge runner supports only scorer_id; per-record "
                "runtime timeout_seconds is enforced by the provider"
            )
        api.validate_runtime_provider_collection(collection, plan)
        provider, spec, resources = _runtime_provider_inputs(
            model=model, request_root=request_root
        )
        if status is not None:
            status.clear()
        result = api.collect_runtime_provider(
            plan,
            provider=provider,
            spec=spec,
            resources=resources,
            baseline_run=baseline_run,
            subject_run=subject_run,
            options=api.RuntimeProviderJudgeOptions(
                source_id="runtime-provider-judge",
                scorer_id=runner["scorer_id"],
                checkpoint_directory=workspace / "runtime-collection",
            ),
        )
        if status is not None:
            status["stop_reason"] = "complete"
        return result
    options = api.CollectionOptions(**collection)
    run_options = api.RunnerOptions(
        checkpoint_directory=workspace / "collection", **runner
    )
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass
    else:
        raise JudgeWorkflowError(
            "Run synchronous evaluate outside an active event loop"
        )
    if status is not None:
        status.clear()

    def stopped(reason: str) -> None:
        if status is not None:
            status["stop_reason"] = reason

    return asyncio.run(
        api.collect_configured(
            plan, options, run_options, baseline_run, subject_run, on_stop=stopped
        )
    )


def require_completed_collection(
    measurements: Any,
    workspace: Path,
    *,
    stop_reason: str | None = None,
) -> dict[str, Any]:
    """Resume pending work after a deadline or graceful invocation stop."""
    pending = sum(not trial["attempts"] for trial in measurements["trials"])
    if not pending:
        return {"pending_trials": 0, "stop_reason": "complete", "resumable": False}
    if stop_reason == "capacity_exhausted":
        return {
            "pending_trials": pending,
            "stop_reason": "retained_capacity_exhausted",
            "resumable": False,
        }
    resumable = stop_reason in {"deadline", "requested"}
    message = (
        "Judge collection is incomplete; rerun the same request to resume its workspace"
        if resumable
        else "Judge collection stopped without a supported reason for its pending trials"
    )
    raise JudgeWorkflowError(
        message,
        payload={
            "format_version": "invarlock/judge-evaluation-result-v1",
            "kind": "judge",
            "ok": False,
            "resumable": resumable,
            "workspace": str(workspace),
            "pending_trials": pending,
            "stop_reason": stop_reason if resumable else "unknown",
            "evidence": None,
            "errors": [message],
        },
    )


def preflight_native_judge(
    request: EvaluationRequest,
    schedule: dict[str, Any],
    policy_bytes: bytes,
) -> dict[str, Any]:
    prepared = prepare_native_judge(policy_bytes, schedule)
    if request.comparison.judge is None:
        raise JudgeWorkflowError("native judge configuration is required")
    metadata = {
        key: value
        for key, value in prepared.items()
        if key not in {"recipe", "preflight_plan"}
    }
    collection = prepared["recipe"]["collection"]
    metadata["collection"] = collection_preflight(
        collection,
        **_collection_context(
            collection,
            model=getattr(request.comparison.judge, "model", None),
            request_root=getattr(request, "root", None),
            plan=prepared.get("preflight_plan"),
        ),
    )
    metadata.update(workspace=str(request.comparison.judge.workspace), network_calls=0)
    return metadata


def evaluate_native_judge(
    request: EvaluationRequest,
    normalized_request: dict[str, Any],
    schedule: dict[str, Any],
    policy_bytes: bytes,
    signing_key: Ed25519PrivateKey,
    capture: Callable[[], tuple[RuntimeSideEvidence, RuntimeSideEvidence]],
    observations: tuple[EvidenceObservation, ...] = (),
    *,
    expected_artifact_digests: dict[str, str],
    expected_runtime_digests: dict[str, str] | None,
) -> JudgeWorkflowResult:
    """Use the native provider lifecycle once, then retain and judge its outputs."""
    from invarlock.judge_measurements.evidence import publish_judge_evidence
    from invarlock.judge_measurements.native_capture import (
        NATIVE_CAPTURE_MAX_BYTES,
        create_native_capture,
        validate_native_capture,
    )

    prepared = prepare_native_judge(policy_bytes, schedule)
    configuration = request.comparison.judge
    if configuration is None:
        raise JudgeWorkflowError("native judge configuration is required")
    recipe = prepared["recipe"]
    collection_environment = collection_preflight(
        recipe["collection"],
        **_collection_context(
            recipe["collection"],
            model=getattr(configuration, "model", None),
            request_root=getattr(request, "root", None),
            plan=prepared.get("preflight_plan"),
        ),
    )
    _require_service_network_authorization(recipe["collection"], collection_environment)
    workspace = configuration.workspace
    with locked_workspace(workspace) as unchanged:
        _retain_identity(
            workspace / "identity.json",
            {
                "format": "invarlock/native-judge-workspace-v1",
                "request": normalized_request,
                "schedule": schedule,
                "policy_base64": base64.b64encode(policy_bytes).decode("ascii"),
                "artifact_digests": expected_artifact_digests,
                "runtime_digests": expected_runtime_digests,
            },
        )
        frozen_path = workspace / "native_capture.json"
        try:
            raw = read_file(frozen_path, NATIVE_CAPTURE_MAX_BYTES)
        except FileNotFoundError:
            unchanged()
            admission = workspace / "capture-admission.json"
            try:
                write_file_no_replace(
                    admission, canonical_payload({"capture_admitted": True})
                )
            except FileExistsError:
                raise JudgeWorkflowError(
                    "Previous native answer capture did not finish. Inspect the workspace before explicitly starting a new capture in a new workspace."
                ) from None
            baseline, subject = capture()
            unchanged()
            frozen = create_native_capture(
                normalized_request,
                schedule,
                policy_bytes,
                recipe,
                baseline,
                subject,
                observations=observations,
            )
            write_file_no_replace(frozen_path, canonical_payload(frozen))
        else:
            frozen = parse_json_bytes(raw, label="retained native judge capture")
            if not isinstance(frozen, dict):
                raise JudgeWorkflowError(
                    "retained native judge capture must be an object"
                )
            validate_native_capture(frozen)
            if (
                frozen["normalized_request"] != normalized_request
                or frozen["schedule"] != schedule
                or frozen["policy_base64"]
                != base64.b64encode(policy_bytes).decode("ascii")
                or frozen["recipe"] != recipe
            ):
                raise JudgeWorkflowError(
                    "retained native capture differs from the frozen request"
                )
        baseline_run, subject_run = validate_native_capture(frozen)
        for role, run in (("baseline", baseline_run), ("subject", subject_run)):
            if run["artifact_digest"] != expected_artifact_digests[role]:
                raise JudgeWorkflowError(
                    "retained native artifact differs from current preflight"
                )
            if expected_runtime_digests is not None and any(
                row["context"]["runtime_digest"] != expected_runtime_digests[role]
                for row in run["records"]
            ):
                raise JudgeWorkflowError(
                    "retained native runtime differs from current preflight"
                )
        plan, policy = finalize_native_plan(recipe, baseline_run, subject_run)
        unchanged()
        collection_stop: dict[str, str] = {}
        collection_arguments = {
            "plan": plan,
            "collection": recipe["collection"],
            "runner": recipe["runner"],
            "workspace": workspace,
            "baseline_run": baseline_run,
            "subject_run": subject_run,
            "status": collection_stop,
        }
        if _local_collection(recipe["collection"]):
            collection_arguments.update(
                model=configuration.model,
                request_root=request.root,
            )
        measurements = collect_frozen(**collection_arguments)
        unchanged()
        collection_status = require_completed_collection(
            measurements,
            workspace,
            stop_reason=collection_stop.get("stop_reason"),
        )
        publication = publish_judge_evidence(
            request.output.evidence,
            plan=plan,
            measurements=measurements,
            baseline_run=baseline_run,
            subject_run=subject_run,
            analysis_policy=policy,
            signing_key=signing_key,
            signer_identity=configuration.signer_identity,
            native_capture=frozen,
        )
        analysis = publication.analysis_result.to_dict()
        return JudgeWorkflowResult(
            {
                "format_version": "invarlock/judge-evaluation-result-v1",
                "kind": "judge",
                "ok": True,
                "execution_mode": request.execution.mode,
                "evidence": str(publication.path),
                "workspace": str(workspace),
                "authentication": "signed",
                "signer_identity": configuration.signer_identity,
                "independent_verification": "not_performed",
                "decision": analysis["decision"],
                "analysis": analysis,
                "collection": collection_status,
                "errors": [],
            }
        )
