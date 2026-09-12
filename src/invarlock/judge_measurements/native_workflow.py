"""Native answer capture, resumable judge collection and evidence publication."""

from __future__ import annotations

import asyncio
import base64
import fcntl
import hashlib
import importlib
import os
import stat
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from invarlock.captured_contracts import read_file
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

if TYPE_CHECKING:
    from invarlock.core.evaluation_request import EvaluationRequest


def collection_api() -> Any:
    """Load only the installed, optional collector through its fixed entry point."""
    try:
        return importlib.import_module("invarlock_addins.inspect_judge")
    except ImportError:
        raise JudgeWorkflowError(
            "Install invarlock-inspect-judge[inspect] to collect judge measurements"
        ) from None


def collection_preflight(configuration: dict[str, Any]) -> dict[str, Any]:
    api = collection_api()
    options = api.CollectionOptions(**configuration)
    return dict(api.validate_collection_environment(options))


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
) -> Any:
    """Resume only unadmitted calls against the same immutable frozen answers."""
    api = collection_api()
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
    """Resume pending work only after an actual collector deadline."""
    pending = sum(not trial["attempts"] for trial in measurements["trials"])
    if not pending:
        return {"pending_trials": 0, "stop_reason": "complete", "resumable": False}
    if stop_reason == "capacity_exhausted":
        return {
            "pending_trials": pending,
            "stop_reason": "retained_capacity_exhausted",
            "resumable": False,
        }
    resumable = stop_reason == "deadline"
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
            "stop_reason": "deadline" if resumable else "unknown",
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
    metadata = {key: value for key, value in prepared.items() if key != "recipe"}
    metadata["collection"] = collection_preflight(prepared["recipe"]["collection"])
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
    collection_preflight(recipe["collection"])
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
        measurements = collect_frozen(
            plan=plan,
            collection=recipe["collection"],
            runner=recipe["runner"],
            workspace=workspace,
            baseline_run=baseline_run,
            subject_run=subject_run,
            status=collection_stop,
        )
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
