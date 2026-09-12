"""Additive wire types for narrowly scoped, independently accepted judge evidence."""

from typing import Literal, TypedDict


class JudgeEvidenceBindings(TypedDict):
    baseline_run_sha256: str
    subject_run_sha256: str
    case_set_sha256: str
    plan_sha256: str
    measurements_sha256: str
    analysis_policy_sha256: str
    analysis_result_sha256: str


class JudgeEvidenceSigner(TypedDict):
    identity: str
    public_key_sha256: str
    public_key: str


class JudgeEvidenceEnvelope(TypedDict):
    format: Literal["invarlock/judge-measurement-evidence-v1"]
    decision_scope: Literal["bounded-judge-fixed-benchmark-v1"]
    intended_subject: str
    bindings: JudgeEvidenceBindings
    signature_algorithm: Literal["ed25519"]
    signer: JudgeEvidenceSigner | None
    signature: str | None


class JudgeTrustedSigner(TypedDict):
    identity: str
    public_key_sha256: str


class JudgeRecipientPolicy(TypedDict):
    format: Literal["invarlock/judge-measurement-recipient-policy-v1"]
    decision_scope: Literal["bounded-judge-fixed-benchmark-v1"]
    intended_subject: str
    required_metric_name: str
    trusted_signer: JudgeTrustedSigner
    bindings: JudgeEvidenceBindings
    required_decision: Literal["pass"]
