from __future__ import annotations

import invarlock.engine as engine
from invarlock.core.checkpoint_identity import checkpoint_tree_sha256
from invarlock.core.evaluation_request import ProviderResolver
from invarlock.evaluation_oci import (
    OciEvaluationLaunch,
    OciRuntimeExecutor,
    OciSideLaunch,
    launch_from_environment,
)
from invarlock.evaluation_runtime import RuntimeResourceResolver
from invarlock.runtime_providers.hf_transformers import hf_tokenizer_contract_sha256


def test_engine_exports_only_supported_transactions_and_provider_contracts() -> None:
    assert set(engine.__all__) == {
        "INVARLOCK_RUNTIME_PROVIDER_ABI",
        "EvaluationBatch",
        "EvaluationInputPart",
        "EvaluationRecord",
        "EvaluationRequest",
        "EvaluationRequestError",
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
        "GGUFArtifactIdentity",
        "HFSnapshotArtifactIdentity",
        "LocalDatasetRequest",
        "ReceiptVerification",
        "ModelArtifactIdentity",
        "ModelRuntimeSpec",
        "OciEvaluationLaunch",
        "OciRuntimeExecutor",
        "OciSideLaunch",
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
        "build_runtime_import_observation",
        "build_runtime_import_receipt",
        "checkpoint_tree_sha256",
        "evaluate_request_file",
        "hf_tokenizer_contract_sha256",
        "load_external_scoring_records_jsonl",
        "load_evaluation_request",
        "load_runtime_behavioral_schedule",
        "load_runtime_import_side",
        "launch_from_environment",
        "prepare_local_evaluation_schedule",
        "render_evidence",
        "verify_evidence",
        "verify_signed_verification_receipt",
        "write_runtime_import_paired_records",
        "write_runtime_import_side",
    }


def test_engine_exports_resolver_types_used_by_stable_function_signatures() -> None:
    assert engine.ProviderResolver is ProviderResolver
    assert engine.RuntimeResourceResolver is RuntimeResourceResolver


def test_engine_exports_canonical_hf_identity_helpers() -> None:
    assert engine.checkpoint_tree_sha256 is checkpoint_tree_sha256
    assert engine.hf_tokenizer_contract_sha256 is hf_tokenizer_contract_sha256


def test_engine_exports_host_oci_orchestration_without_worker_internals() -> None:
    assert engine.OciEvaluationLaunch is OciEvaluationLaunch
    assert engine.OciSideLaunch is OciSideLaunch
    assert engine.OciRuntimeExecutor is OciRuntimeExecutor
    assert engine.launch_from_environment is launch_from_environment
    assert "compose_side_worker_command" not in engine.__all__
