"""Stable embedding facade for the InvarLock evaluation engine.

Embedding applications should import this module rather than package internals.
The surface is intentionally limited to the evaluate/verify/report transactions,
the portable acceptance-attestation transport, their value types, and the
runtime-provider ABI.
"""

from collections.abc import Mapping, Sequence
from typing import Any

from invarlock.acceptance_attestation import (
    ACCEPTANCE_PREDICATE_FORMAT,
    ACCEPTANCE_PREDICATE_TYPE,
    DSSE_PAYLOAD_TYPE,
    IN_TOTO_STATEMENT_TYPE,
    RECIPIENT_POLICY_FORMAT,
    AcceptanceAttestation,
    AcceptanceAttestationError,
    AcceptanceDecision,
    verify_acceptance_attestation,
    write_acceptance_attestation,
)
from invarlock.captured_evaluation import (
    CapturedEvaluationPreflightResult,
    CapturedEvaluationTransactionResult,
)
from invarlock.captured_normalization import (
    captured_request_digest,
    comparison_policy_digest,
    normalize_captured_request,
)
from invarlock.core.checkpoint_identity import checkpoint_tree_sha256
from invarlock.core.evaluation_request import (
    CapturedEvaluationRequest,
    EvaluationRequest,
    EvaluationRequestError,
    ProviderResolver,
    load_evaluation_request,
)
from invarlock.core.runtime_provider import (
    INVARLOCK_RUNTIME_PROVIDER_ABI,
    EvaluationBatch,
    EvaluationInputPart,
    EvaluationRecord,
    GGUFArtifactIdentity,
    HFSnapshotArtifactIdentity,
    ModelArtifactIdentity,
    ModelRuntimeSpec,
    RuntimeArtifactResources,
    RuntimeBackendIdentity,
    RuntimeBehavioralSchedule,
    RuntimeDeviceFacts,
    RuntimeExecutionContext,
    RuntimeExecutionSettings,
    RuntimeProvider,
    RuntimeProviderCapabilities,
    RuntimeProviderPluginIdentity,
    RuntimeProviderReceipt,
    RuntimeScoringRecord,
    RuntimeSession,
    RuntimeTask,
    ScoringObservation,
    TensorRTLLMArtifactIdentity,
    load_runtime_behavioral_schedule,
)
from invarlock.core.schedule_preparation import (
    LocalDatasetRequest,
    prepare_local_evaluation_schedule,
)
from invarlock.core.scorer_extension import (
    SCORER_EXTENSION_ABI_VERSION,
    ScorerExtensionBinding,
    ScorerExtensionDescriptor,
    ScorerExtensionError,
    ScorerExtensionRegistry,
    ScorerExtensionResult,
    ScorerReplayRequest,
    VerifierReplayScorer,
    build_scorer_binding,
    build_scorer_result,
)
from invarlock.evaluation_comparison.capacity import DEFAULT_MAX_BOOTSTRAP_DRAWS
from invarlock.evaluation_comparison.comparison import compare_runs as _compare_runs
from invarlock.evaluation_comparison.comparison import make_run as _make_run
from invarlock.evaluation_oci import (
    OciEvaluationLaunch,
    OciRuntimeExecutor,
    OciSideLaunch,
    OciWorkerLimits,
    launch_from_environment,
)
from invarlock.evaluation_record_contracts.contracts import MAX_RECORDS as _MAX_RECORDS
from invarlock.evaluation_record_contracts.contracts import (
    EvaluationRecordsError,
)
from invarlock.evaluation_records.adapters import load_run
from invarlock.evaluation_records.cases import (
    canonical_case_set as _canonical_case_set,
)
from invarlock.evaluation_records.cases import case_set_digest as _case_set_digest
from invarlock.evaluation_records.cases import (
    validate_run_case_set as _validate_run_case_set,
)
from invarlock.evaluation_records.io import physical_file_digest, run_digest, write_run
from invarlock.evaluation_runtime import RuntimeResourceResolver
from invarlock.evaluation_transaction import (
    EvaluationPreflightError,
    EvaluationPreflightResult,
    EvaluationTransactionError,
    EvaluationTransactionResult,
    evaluate_request_file,
    preflight_evaluation_request,
)
from invarlock.evaluator_qualification import (
    EVALUATOR_EXPORT_FORMAT,
    EVALUATOR_PROFILE_FORMAT,
    EVALUATOR_QUALIFICATION_FORMAT,
    EVALUATOR_SCHEDULE_FORMAT,
    EvaluatorQualificationError,
    EvaluatorQualificationResult,
    qualify_evaluator_export,
)
from invarlock.evidence_pack_contract import EvidenceObservation
from invarlock.evidence_pack_support import EvidencePackResult, EvidencePackStatus
from invarlock.evidence_receipt import (
    EvidenceReceiptError,
    ReceiptVerification,
    verify_signed_verification_receipt,
)
from invarlock.evidence_reporting import (
    EvidenceReport,
    EvidenceReportError,
    EvidenceReportV2,
    render_evidence,
)
from invarlock.evidence_verification import (
    EvidenceVerification,
    EvidenceVerificationError,
    verify_evidence,
)
from invarlock.runtime_import_authoring import (
    RuntimeImportAuthoringError,
    RuntimeImportPairedRecords,
    RuntimeImportSideEvidence,
    build_runtime_import_observation,
    build_runtime_import_receipt,
    load_external_scoring_records_jsonl,
    load_runtime_import_side,
    write_runtime_import_paired_records,
    write_runtime_import_side,
)
from invarlock.runtime_providers.hf_transformers import hf_tokenizer_contract_sha256
from invarlock.trust_inputs import (
    CapturedTrustInputs,
    TrustInputs,
    TrustInputsError,
    load_trust_inputs,
)


def make_run(
    records: Sequence[Mapping[str, Any]],
    *,
    source: Mapping[str, str],
    run_id: str,
    artifact_digest: str,
    source_digest: str | None = None,
    score_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate and detach captured records with caller-supplied source identities."""
    if not 0 < len(records) <= _MAX_RECORDS:
        raise EvaluationRecordsError(f"run must contain 1..{_MAX_RECORDS} records")
    return _make_run(
        [dict(record) for record in records],
        source=dict(source),
        run_id=run_id,
        artifact_digest=artifact_digest,
        source_digest=source_digest,
        score_provenance=dict(score_provenance)
        if score_provenance is not None
        else None,
    )


def freeze_case_set(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Freeze declared cases before inference, sorting only the outer case list."""
    if not 0 < len(cases) <= _MAX_RECORDS:
        raise EvaluationRecordsError(f"case set must contain 1..{_MAX_RECORDS} cases")
    return _canonical_case_set(
        {
            "format": "invarlock/evaluation-case-set-v1",
            "cases": [dict(case) for case in cases],
        }
    )


def case_set_digest(case_set: Mapping[str, Any]) -> str:
    """Hash a validated planned case set, independent of record order."""
    return _case_set_digest(dict(case_set))


def validate_run_case_set(run: Mapping[str, Any], expected_digest: str) -> None:
    """Require exact planned membership without filtering or padding."""
    _validate_run_case_set(dict(run), expected_digest)


def compare_runs(
    baseline: Mapping[str, Any],
    subject: Mapping[str, Any],
    policy: Mapping[str, Any],
    *,
    max_bootstrap_draws: int | None = DEFAULT_MAX_BOOTSTRAP_DRAWS,
) -> dict[str, Any]:
    """Compare complete captured runs under an independent policy and local budget."""
    return _compare_runs(
        dict(baseline),
        dict(subject),
        dict(policy),
        max_bootstrap_draws=max_bootstrap_draws,
    )


__all__ = [
    "CapturedEvaluationPreflightResult",
    "CapturedEvaluationTransactionResult",
    "CapturedTrustInputs",
    "DEFAULT_MAX_BOOTSTRAP_DRAWS",
    "EvaluationRecordsError",
    "EvidenceReportV2",
    "captured_request_digest",
    "comparison_policy_digest",
    "normalize_captured_request",
    "ACCEPTANCE_PREDICATE_FORMAT",
    "ACCEPTANCE_PREDICATE_TYPE",
    "DSSE_PAYLOAD_TYPE",
    "INVARLOCK_RUNTIME_PROVIDER_ABI",
    "IN_TOTO_STATEMENT_TYPE",
    "RECIPIENT_POLICY_FORMAT",
    "AcceptanceAttestation",
    "AcceptanceAttestationError",
    "AcceptanceDecision",
    "CapturedEvaluationRequest",
    "EvaluationBatch",
    "EvaluationInputPart",
    "EvaluationRecord",
    "EvaluationRequest",
    "EvaluationRequestError",
    "EvaluationPreflightError",
    "EvaluationPreflightResult",
    "EvaluationTransactionError",
    "EvaluationTransactionResult",
    "EvidencePackResult",
    "EvidencePackStatus",
    "EvidenceObservation",
    "EvidenceReceiptError",
    "EvidenceReport",
    "EvidenceReportError",
    "EvidenceVerification",
    "EvidenceVerificationError",
    "EVALUATOR_EXPORT_FORMAT",
    "EVALUATOR_PROFILE_FORMAT",
    "EVALUATOR_QUALIFICATION_FORMAT",
    "EVALUATOR_SCHEDULE_FORMAT",
    "EvaluatorQualificationError",
    "EvaluatorQualificationResult",
    "GGUFArtifactIdentity",
    "HFSnapshotArtifactIdentity",
    "LocalDatasetRequest",
    "ReceiptVerification",
    "ModelArtifactIdentity",
    "ModelRuntimeSpec",
    "OciEvaluationLaunch",
    "OciRuntimeExecutor",
    "OciSideLaunch",
    "OciWorkerLimits",
    "ProviderResolver",
    "RuntimeArtifactResources",
    "RuntimeBackendIdentity",
    "RuntimeBehavioralSchedule",
    "RuntimeDeviceFacts",
    "RuntimeExecutionContext",
    "RuntimeExecutionSettings",
    "RuntimeImportAuthoringError",
    "RuntimeImportPairedRecords",
    "RuntimeImportSideEvidence",
    "RuntimeProvider",
    "RuntimeProviderCapabilities",
    "RuntimeProviderPluginIdentity",
    "RuntimeProviderReceipt",
    "RuntimeResourceResolver",
    "RuntimeScoringRecord",
    "RuntimeSession",
    "RuntimeTask",
    "SCORER_EXTENSION_ABI_VERSION",
    "ScorerExtensionBinding",
    "ScorerExtensionDescriptor",
    "ScorerExtensionError",
    "ScorerExtensionRegistry",
    "ScorerExtensionResult",
    "ScorerReplayRequest",
    "VerifierReplayScorer",
    "build_scorer_binding",
    "build_scorer_result",
    "ScoringObservation",
    "TensorRTLLMArtifactIdentity",
    "TrustInputs",
    "TrustInputsError",
    "build_runtime_import_observation",
    "build_runtime_import_receipt",
    "case_set_digest",
    "checkpoint_tree_sha256",
    "compare_runs",
    "evaluate_request_file",
    "freeze_case_set",
    "hf_tokenizer_contract_sha256",
    "load_external_scoring_records_jsonl",
    "load_evaluation_request",
    "load_run",
    "load_runtime_behavioral_schedule",
    "load_runtime_import_side",
    "load_trust_inputs",
    "launch_from_environment",
    "make_run",
    "prepare_local_evaluation_schedule",
    "preflight_evaluation_request",
    "physical_file_digest",
    "qualify_evaluator_export",
    "render_evidence",
    "run_digest",
    "verify_evidence",
    "verify_acceptance_attestation",
    "verify_signed_verification_receipt",
    "validate_run_case_set",
    "write_acceptance_attestation",
    "write_run",
    "write_runtime_import_paired_records",
    "write_runtime_import_side",
]
