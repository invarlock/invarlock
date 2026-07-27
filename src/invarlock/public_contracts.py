"""Load the canonical contracts shipped with InvarLock.

Contract readers use the package-owned copies so a working directory or
environment variable cannot substitute a weaker schema during verification.
Repository checks keep these copies byte-identical to the files in ``contracts/``.
"""

from __future__ import annotations

import importlib.resources
import json
from typing import Any

PACKAGE_CONTRACTS_ROOT = importlib.resources.files("invarlock").joinpath(
    "_data", "contracts"
)

EVALUATION_REQUEST_FORMAT_VERSION = "invarlock/evaluation-request-v1"
EVIDENCE_PACK_FORMAT_VERSION = "invarlock/evidence-pack-v1"
EVIDENCE_OBSERVATION_FORMAT_VERSION = "invarlock/evidence-observation-v1"
TRUST_INPUTS_FORMAT_VERSION = "invarlock/trust-inputs-v1"
RUNTIME_MANIFEST_CONTRACT_VERSION = "runtime-manifest-v1"
RUNTIME_PROVIDER_ABI_VERSION = "1"
RUNTIME_PROVIDER_CAPABILITIES_FORMAT_VERSION = "runtime-provider-capabilities-v1"
MODEL_ARTIFACT_IDENTITY_FORMAT_VERSION = "invarlock/model-artifact-identity-v1"
RUNTIME_PROVIDER_RECEIPT_FORMAT_VERSION = "invarlock/runtime-provider-receipt-v1"
RUNTIME_SCORING_OBSERVATION_FORMAT_VERSION = "invarlock/runtime-scoring-observation-v1"
RUNTIME_BEHAVIORAL_SCHEDULE_FORMAT_VERSION = "invarlock/runtime-behavioral-schedule-v1"
SCORER_EXTENSION_DESCRIPTOR_FORMAT_VERSION = "invarlock/scorer-extension-descriptor-v1"
SCORER_EXTENSION_BINDING_FORMAT_VERSION = "invarlock/scorer-extension-binding-v1"
SCORER_EXTENSION_RESULT_FORMAT_VERSION = "invarlock/scorer-extension-result-v1"
ACCEPTANCE_PREDICATE_FORMAT_VERSION = "invarlock/acceptance-predicate-v2"
RECIPIENT_ACCEPTANCE_POLICY_FORMAT_VERSION = "invarlock/recipient-acceptance-policy-v2"
EVALUATOR_QUALIFICATION_PROFILE_FORMAT_VERSION = (
    "invarlock/evaluator-qualification-profile-v1"
)
EVALUATOR_QUALIFICATION_SCHEDULE_FORMAT_VERSION = (
    "invarlock/evaluator-qualification-schedule-v1"
)
EVALUATOR_QUALIFICATION_EXPORT_FORMAT_VERSION = (
    "invarlock/evaluator-qualification-export-v1"
)
EVALUATOR_QUALIFICATION_RESULT_FORMAT_VERSION = (
    "invarlock/evaluator-qualification-result-v1"
)


class ContractLoadError(RuntimeError):
    """Raised when a shipped contract is missing, malformed, or not an object."""

    def __init__(self, filename: str, *, reason: str) -> None:
        super().__init__(f"Failed to load contract '{filename}': {reason}")
        self.filename = filename
        self.reason = reason


def _load_object_contract(filename: str) -> dict[str, Any]:
    try:
        raw = PACKAGE_CONTRACTS_ROOT.joinpath(filename).read_text(encoding="utf-8")
        payload = json.loads(raw)
    except (
        FileNotFoundError,
        ModuleNotFoundError,
        NotADirectoryError,
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
    ) as exc:
        raise ContractLoadError(filename, reason=str(exc)) from exc
    if not isinstance(payload, dict):
        raise ContractLoadError(
            filename,
            reason=f"expected JSON object, got {type(payload).__name__}",
        )
    return payload


def load_evaluation_request_schema() -> dict[str, Any]:
    return _load_object_contract("evaluation_request.schema.json")


def load_evidence_pack_schema() -> dict[str, Any]:
    return _load_object_contract("evidence_pack.schema.json")


def load_evidence_observation_schema() -> dict[str, Any]:
    return _load_object_contract("evidence_observation.schema.json")


def load_trust_inputs_schema() -> dict[str, Any]:
    return _load_object_contract("trust_inputs.schema.json")


def load_runtime_manifest_schema() -> dict[str, Any]:
    return _load_object_contract("runtime_manifest.schema.json")


def load_runtime_provider_capabilities_schema() -> dict[str, Any]:
    return _load_object_contract("runtime_provider_capabilities.json")


def load_model_artifact_identity_schema() -> dict[str, Any]:
    return _load_object_contract("model_artifact_identity.schema.json")


def load_runtime_provider_receipt_schema() -> dict[str, Any]:
    return _load_object_contract("runtime_provider_receipt.schema.json")


def load_runtime_scoring_observation_schema() -> dict[str, Any]:
    return _load_object_contract("runtime_scoring_observation.schema.json")


def load_runtime_behavioral_schedule_schema() -> dict[str, Any]:
    return _load_object_contract("runtime_behavioral_schedule.schema.json")


def load_scorer_extension_descriptor_schema() -> dict[str, Any]:
    return _load_object_contract("scorer_extension_descriptor.schema.json")


def load_scorer_extension_binding_schema() -> dict[str, Any]:
    return _load_object_contract("scorer_extension_binding.schema.json")


def load_scorer_extension_result_schema() -> dict[str, Any]:
    return _load_object_contract("scorer_extension_result.schema.json")


def load_acceptance_predicate_schema() -> dict[str, Any]:
    return _load_object_contract("acceptance_predicate.schema.json")


def load_recipient_acceptance_policy_schema() -> dict[str, Any]:
    return _load_object_contract("recipient_acceptance_policy.schema.json")


def load_evaluator_qualification_profile_schema() -> dict[str, Any]:
    return _load_object_contract("evaluator_qualification_profile.schema.json")


def load_evaluator_qualification_schedule_schema() -> dict[str, Any]:
    return _load_object_contract("evaluator_qualification_schedule.schema.json")


def load_evaluator_qualification_export_schema() -> dict[str, Any]:
    return _load_object_contract("evaluator_qualification_export.schema.json")


def load_evaluator_qualification_result_schema() -> dict[str, Any]:
    return _load_object_contract("evaluator_qualification_result.schema.json")


__all__ = [
    "ContractLoadError",
    "ACCEPTANCE_PREDICATE_FORMAT_VERSION",
    "EVALUATION_REQUEST_FORMAT_VERSION",
    "EVIDENCE_PACK_FORMAT_VERSION",
    "EVIDENCE_OBSERVATION_FORMAT_VERSION",
    "EVALUATOR_QUALIFICATION_EXPORT_FORMAT_VERSION",
    "EVALUATOR_QUALIFICATION_PROFILE_FORMAT_VERSION",
    "EVALUATOR_QUALIFICATION_RESULT_FORMAT_VERSION",
    "EVALUATOR_QUALIFICATION_SCHEDULE_FORMAT_VERSION",
    "MODEL_ARTIFACT_IDENTITY_FORMAT_VERSION",
    "PACKAGE_CONTRACTS_ROOT",
    "RUNTIME_BEHAVIORAL_SCHEDULE_FORMAT_VERSION",
    "RUNTIME_MANIFEST_CONTRACT_VERSION",
    "RUNTIME_PROVIDER_ABI_VERSION",
    "RUNTIME_PROVIDER_CAPABILITIES_FORMAT_VERSION",
    "RUNTIME_PROVIDER_RECEIPT_FORMAT_VERSION",
    "RUNTIME_SCORING_OBSERVATION_FORMAT_VERSION",
    "RECIPIENT_ACCEPTANCE_POLICY_FORMAT_VERSION",
    "SCORER_EXTENSION_BINDING_FORMAT_VERSION",
    "SCORER_EXTENSION_DESCRIPTOR_FORMAT_VERSION",
    "SCORER_EXTENSION_RESULT_FORMAT_VERSION",
    "TRUST_INPUTS_FORMAT_VERSION",
    "load_evaluation_request_schema",
    "load_acceptance_predicate_schema",
    "load_evidence_pack_schema",
    "load_evidence_observation_schema",
    "load_evaluator_qualification_export_schema",
    "load_evaluator_qualification_profile_schema",
    "load_evaluator_qualification_result_schema",
    "load_evaluator_qualification_schedule_schema",
    "load_trust_inputs_schema",
    "load_model_artifact_identity_schema",
    "load_runtime_behavioral_schedule_schema",
    "load_runtime_manifest_schema",
    "load_runtime_provider_capabilities_schema",
    "load_runtime_provider_receipt_schema",
    "load_runtime_scoring_observation_schema",
    "load_recipient_acceptance_policy_schema",
    "load_scorer_extension_binding_schema",
    "load_scorer_extension_descriptor_schema",
    "load_scorer_extension_result_schema",
]
