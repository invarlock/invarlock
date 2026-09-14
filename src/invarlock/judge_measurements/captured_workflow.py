"""Judge existing evaluator captures through the same installed transaction."""

from __future__ import annotations

import os
import re
import stat
from pathlib import Path
from typing import Any, cast

import invarlock.judge_measurements.native_workflow as native_workflow
from invarlock.captured_evaluation import _json, _read_input, _run
from invarlock.core.evaluation_request import (
    CapturedEvaluationRequest,
    _reference_parts,
    _resolve_output_reference,
)
from invarlock.evaluation_record_contracts.contracts import MAX_INPUT_BYTES
from invarlock.evaluation_records.io import run_digest
from invarlock.judge_measurement_types import (
    JudgeAnalysisPolicyDocument,
    JudgeMeasurementPlan,
    JudgeMeasurements,
)
from invarlock.judge_measurements.contracts import (
    MEASUREMENTS_MAX_BYTES,
    validate_measurements,
)
from invarlock.judge_measurements.evidence import _private_key, publish_judge_evidence
from invarlock.judge_measurements.native_capture import NATIVE_RUN_SOURCE
from invarlock.judge_measurements.native_recipe import (
    NATIVE_POLICY_MAX_BYTES,
    _recipe,
    finalize_native_plan,
)
from invarlock.judge_measurements.workflow import (
    MAX_WORKFLOW_BYTES,
    JudgeWorkflowError,
    JudgeWorkflowResult,
    _collection_budgets,
)


def prepare_evaluator_judge(
    recipe: dict[str, Any], baseline: dict[str, Any], subject: dict[str, Any]
) -> tuple[JudgeMeasurementPlan, JudgeAnalysisPolicyDocument]:
    """Freeze a declared rubric against captured answers without making calls.

    The returned plan can be used by a caller's retained-measurement collector.
    It binds the complete source records and explicit input projections.
    """
    for run in (baseline, subject):
        run_digest(run)
        if run["source"]["name"] == NATIVE_RUN_SOURCE:
            raise JudgeWorkflowError(
                "Native judge answers require their runtime capture and native request"
            )
    return finalize_native_plan(recipe, baseline, subject)


def _locations(request: CapturedEvaluationRequest, signing_key: Path | None) -> None:
    judge = request.judge
    if request.metric != "judge" or judge is None:
        raise JudgeWorkflowError(
            "Captured judge scoring requires metric: judge and judge configuration"
        )
    if (
        not isinstance(judge.signer_identity, str)
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}", judge.signer_identity)
        is None
    ):
        raise JudgeWorkflowError("Judge signer identity is invalid")
    workspace, evidence = judge.workspace, request.evidence
    for name, path in (("judge workspace", workspace), ("output.evidence", evidence)):
        reference = path.relative_to(request.root).as_posix()
        _reference_parts(reference, label=name)
        # Checks existing ancestry and refuses a symlinked or occupied destination.
        if name == "output.evidence":
            _resolve_output_reference(request.root, reference, label=name)
    if workspace.is_relative_to(evidence) or evidence.is_relative_to(workspace):
        raise JudgeWorkflowError(
            "Judge workspace and evidence destination must be separate"
        )
    paths = [request.baseline.path, request.subject.path, request.policy]
    if judge.measurements is not None:
        paths.append(judge.measurements)
    if signing_key is not None:
        paths.append(Path(signing_key).absolute())
    for path in paths:
        if any(
            path.is_relative_to(target) or target.is_relative_to(path)
            for target in (workspace, evidence)
        ):
            raise JudgeWorkflowError(
                "Judge workspace and evidence must remain separate from input files and signing keys"
            )
    if workspace.exists() and not workspace.is_dir():
        raise JudgeWorkflowError("Judge workspace must be a private directory")
    if workspace.exists():
        metadata = workspace.stat()
        if metadata.st_uid != os.geteuid() or stat.S_IMODE(metadata.st_mode) & 0o077:
            raise JudgeWorkflowError(
                "Judge workspace must be caller-owned and private (0700)"
            )
    # Walk the existing workspace ancestry before even a collection preflight.
    for part in (workspace, *workspace.parents):
        if part.is_symlink():
            raise JudgeWorkflowError(
                "Judge workspace ancestry must not contain symlinks"
            )
        if part.exists() and not part.is_dir():
            raise JudgeWorkflowError(
                "Judge workspace ancestry must contain real directories"
            )


def _prepare(
    request: CapturedEvaluationRequest, signing_key: Path | None, unsigned: bool
) -> tuple[dict[str, Any], dict[str, Any], Any]:
    if type(unsigned) is not bool or unsigned == (signing_key is not None):
        raise JudgeWorkflowError(
            "Choose either a signing key or explicit --unsigned for judge evidence"
        )
    _locations(request, signing_key)
    key = _private_key(signing_key) if signing_key is not None else None
    remaining = MAX_WORKFLOW_BYTES

    def read_input(path: Path, limit: int) -> bytes:
        nonlocal remaining
        if remaining <= 0:
            raise JudgeWorkflowError("Combined judge input byte allowance is exhausted")
        # Charge the immutable bytes that will actually be parsed. Restrict the
        # read itself so a final file cannot allocate past the shared allowance.
        raw = _read_input(request, path, limit=min(limit, remaining))
        remaining -= len(raw)
        return raw

    baseline = _run(
        request, "baseline", raw=read_input(request.baseline.path, MAX_INPUT_BYTES)
    )
    subject = _run(
        request, "subject", raw=read_input(request.subject.path, MAX_INPUT_BYTES)
    )
    for role, run in (("baseline", baseline), ("subject", subject)):
        expected = getattr(request, role).expected_run_digest
        if expected is not None and expected != run_digest(run):
            raise JudgeWorkflowError(f"{role} run digest differs from expected pin")
    recipe = _recipe(
        _json(read_input(request.policy, NATIVE_POLICY_MAX_BYTES), label="judge policy")
    )
    plan, policy = prepare_evaluator_judge(recipe, baseline, subject)
    values: dict[str, Any] = {
        "baseline": baseline,
        "subject": subject,
        "recipe": recipe,
        "plan": plan,
        "policy": policy,
    }
    judge = request.judge
    assert judge is not None
    units = len({item["unit_id"] for item in plan["sampling"]["case_units"]})
    metadata: dict[str, Any] = {
        "format_version": "invarlock/judge-evaluation-preflight-v1",
        "kind": "judge",
        "execution_mode": "captured",
        "ok": True,
        "ready": True,
        "evidence": str(request.evidence),
        "workspace": str(judge.workspace),
        "signer_identity": judge.signer_identity,
        "requested_authentication": "unsigned_local" if unsigned else "signed",
        "cases": len(baseline["records"]),
        "independent_units": units,
        "planned_trials": plan["schedule"]["expected_trials"],
        "judge": plan["judge"],
        "network_calls": 0,
        "baseline_run_digest": run_digest(baseline),
        "subject_run_digest": run_digest(subject),
        "source_assurance": "captured_inputs",
        "errors": [],
    }
    if judge.measurements is not None:
        measurements = _json(
            read_input(judge.measurements, MEASUREMENTS_MAX_BYTES),
            label="judge measurements",
        )
        validate_measurements(
            cast(JudgeMeasurements, measurements),
            plan,
            baseline_run=baseline,
            subject_run=subject,
        )
        values["measurements"] = measurements
        metadata["collection_available"] = False
        metadata["measurements"] = measurements["completeness"]
    else:
        budgets = _collection_budgets(recipe["collection"], plan)
        capacity = min(
            budgets["max_calls"],
            budgets["max_input_tokens"] // budgets["input_tokens_per_call"],
            budgets["max_output_tokens"]
            // plan["judge"]["config"]["max_output_tokens"],
            budgets["max_cost_microusd"] // budgets["cost_microusd_per_call"],
        )
        if capacity < plan["schedule"]["expected_trials"]:
            raise JudgeWorkflowError(
                "Judge collection budgets must reserve every planned call"
            )
        if units < policy["minimum_units"]:
            raise JudgeWorkflowError(
                "Judge cases have fewer independent units than the analysis minimum"
            )
        metadata.update(
            budgets=budgets,
            maximum_admitted_calls=capacity,
            collection_environment=native_workflow.collection_preflight(
                recipe["collection"]
            ),
            collection_available=True,
        )
    return values, metadata, key


def preflight_captured_judge(
    request: CapturedEvaluationRequest, *, signing_key_path: Path | None, unsigned: bool
) -> JudgeWorkflowResult:
    """Validate capture mappings and full collection reserves without writes."""
    try:
        _, result, _ = _prepare(request, signing_key_path, unsigned)
        return JudgeWorkflowResult(result)
    except (OSError, ValueError) as exc:
        if isinstance(exc, JudgeWorkflowError):
            raise
        raise JudgeWorkflowError(str(exc)) from exc


def evaluate_captured_judge(
    request: CapturedEvaluationRequest, *, signing_key_path: Path | None, unsigned: bool
) -> JudgeWorkflowResult:
    """Collect or import judgments on existing answers, then publish once."""
    try:
        values, _, key = _prepare(request, signing_key_path, unsigned)
        judge = request.judge
        assert judge is not None
        status = None
        if "measurements" not in values:
            with native_workflow.locked_workspace(judge.workspace) as unchanged:
                native_workflow._retain_identity(
                    judge.workspace / "identity.json", values
                )
                unchanged()
                stop: dict[str, str] = {}
                values["measurements"] = native_workflow.collect_frozen(
                    plan=values["plan"],
                    collection=values["recipe"]["collection"],
                    runner=values["recipe"]["runner"],
                    workspace=judge.workspace,
                    baseline_run=values["baseline"],
                    subject_run=values["subject"],
                    status=stop,
                )
                unchanged()
                status = native_workflow.require_completed_collection(
                    values["measurements"],
                    judge.workspace,
                    stop_reason=stop.get("stop_reason"),
                )
        publication = publish_judge_evidence(
            request.evidence,
            plan=values["plan"],
            measurements=values["measurements"],
            baseline_run=values["baseline"],
            subject_run=values["subject"],
            analysis_policy=values["policy"],
            signing_key=key,
            signer_identity=judge.signer_identity if key is not None else None,
        )
        analysis = publication.analysis_result.to_dict()
        return JudgeWorkflowResult(
            {
                "format_version": "invarlock/judge-evaluation-result-v1",
                "kind": "judge",
                "ok": True,
                "execution_mode": "captured",
                "evidence": str(publication.path),
                "authentication": "signed" if key is not None else "unsigned_local",
                "signer_identity": judge.signer_identity if key is not None else None,
                "source_assurance": "captured_inputs",
                "independent_verification": "not_performed",
                "decision": analysis["decision"],
                "analysis": analysis,
                **({"collection": status} if status is not None else {}),
                "errors": [],
            }
        )
    except (OSError, ValueError) as exc:
        if isinstance(exc, JudgeWorkflowError):
            raise
        raise JudgeWorkflowError(str(exc)) from exc
