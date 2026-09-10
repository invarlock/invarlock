from __future__ import annotations

import invarlock.engine as engine
from invarlock.core.checkpoint_identity import checkpoint_tree_sha256
from invarlock.core.evaluation_request import ProviderResolver
from invarlock.evaluation_oci import (
    OciEvaluationLaunch,
    OciRuntimeExecutor,
    OciSideLaunch,
    OciWorkerLimits,
    launch_from_environment,
)
from invarlock.evaluation_record_contracts.contracts import digest
from invarlock.evaluation_records.templates import example_project
from invarlock.evaluation_runtime import RuntimeResourceResolver
from invarlock.runtime_providers.hf_transformers import hf_tokenizer_contract_sha256


def test_engine_exports_only_supported_transactions_and_provider_contracts() -> None:
    assert set(engine.__all__) == {
        "ACCEPTANCE_PREDICATE_FORMAT",
        "ACCEPTANCE_PREDICATE_TYPE",
        "DEFAULT_MAX_BOOTSTRAP_DRAWS",
        "DSSE_PAYLOAD_TYPE",
        "INVARLOCK_RUNTIME_PROVIDER_ABI",
        "IN_TOTO_STATEMENT_TYPE",
        "RECIPIENT_POLICY_FORMAT",
        "AcceptanceAttestation",
        "AcceptanceAttestationError",
        "AcceptanceDecision",
        "CapturedEvaluationPreflightResult",
        "CapturedEvaluationRequest",
        "CapturedEvaluationTransactionResult",
        "CapturedTrustInputs",
        "EvaluationBatch",
        "EvaluationInputPart",
        "EvaluationRecord",
        "EvaluationRecordsError",
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
        "EvidenceReportV2",
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
        "captured_request_digest",
        "case_set_digest",
        "checkpoint_tree_sha256",
        "compare_runs",
        "comparison_policy_digest",
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
        "normalize_captured_request",
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
    }


def test_engine_exports_resolver_types_used_by_stable_function_signatures() -> None:
    assert engine.ProviderResolver is ProviderResolver
    assert engine.RuntimeResourceResolver is RuntimeResourceResolver


def test_engine_exports_canonical_hf_identity_helpers() -> None:
    assert engine.checkpoint_tree_sha256 is checkpoint_tree_sha256
    assert engine.hf_tokenizer_contract_sha256 is hf_tokenizer_contract_sha256


def test_engine_exports_neutral_captured_helpers(tmp_path) -> None:
    baseline, candidate, policy = example_project("classification")
    assert engine.run_digest(baseline).startswith("sha256:")
    assert engine.case_set_digest(
        {
            "format": "invarlock/evaluation-case-set-v1",
            "cases": [
                {
                    "id": "one",
                    "input": "input",
                    "expected": "expected",
                    "metadata": {},
                }
            ],
        }
    ).startswith("sha256:")
    comparison = engine.compare_runs(baseline, candidate, policy)
    assert comparison["format"] == "invarlock/multi-metric-comparison-v1"
    path = engine.write_run(tmp_path / "run.json", baseline)
    assert engine.load_run(path) == baseline
    assert engine.run_digest(baseline) == digest(baseline)
    assert engine.physical_file_digest(path) == engine.run_digest(baseline)


def test_engine_exports_host_oci_orchestration_without_worker_internals() -> None:
    assert engine.OciEvaluationLaunch is OciEvaluationLaunch
    assert engine.OciSideLaunch is OciSideLaunch
    assert engine.OciRuntimeExecutor is OciRuntimeExecutor
    assert engine.OciWorkerLimits is OciWorkerLimits
    assert engine.launch_from_environment is launch_from_environment
    assert "compose_side_worker_command" not in engine.__all__
