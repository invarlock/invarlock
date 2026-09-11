"""Trust-input resolution for the public ``verify`` command."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - imports exist only for static analysis
    from invarlock.evidence_verification import EvidenceVerification


PROFILE_CONFLICT_OPTIONS = (
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
CAPTURED_IRRELEVANT_OPTIONS = (
    "expected_baseline_artifact",
    "expected_subject_artifact",
    "expected_schedule",
    "expected_baseline_runtime",
    "expected_subject_runtime",
    "allow_installed_scorers",
)
NATIVE_IRRELEVANT_OPTIONS = (
    "expected_baseline_run",
    "expected_subject_run",
    "max_bootstrap_draws",
)


@dataclass(frozen=True)
class VerificationOptions:
    """All trust inputs accepted by the verification command."""

    evidence: Path
    trust_profile: Path | None
    policy: Path | None
    expected_baseline_artifact: str | None
    expected_subject_artifact: str | None
    expected_schedule: str | None
    expected_baseline_runtime: str | None
    expected_subject_runtime: str | None
    expected_signer: str | None
    expected_baseline_run: str | None
    expected_subject_run: str | None
    max_bootstrap_draws: int | None
    expected_request_digest: str | None
    receipt: Path | None
    verifier_signing_key: Path | None
    verifier_identity: str | None
    allow_installed_scorers: bool


def execute_verification(
    options: VerificationOptions,
    *,
    captured: bool,
    command_line: frozenset[str],
) -> EvidenceVerification:
    """Resolve one closed trust source and execute mode-appropriate verification."""

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

    policy = options.policy
    expected_baseline_artifact = options.expected_baseline_artifact
    expected_subject_artifact = options.expected_subject_artifact
    expected_schedule = options.expected_schedule
    expected_baseline_runtime = options.expected_baseline_runtime
    expected_subject_runtime = options.expected_subject_runtime
    expected_signer = options.expected_signer
    expected_baseline_run = options.expected_baseline_run
    expected_subject_run = options.expected_subject_run
    expected_request_digest = options.expected_request_digest
    verifier_signing_key = options.verifier_signing_key
    verifier_identity = options.verifier_identity
    allow_installed_scorers = options.allow_installed_scorers
    trust_profile_digest: str | None = None
    policy_bytes: bytes | None = None
    verifier_signing_key_bytes: bytes | None = None

    if options.trust_profile is not None:
        conflicts = [
            name.replace("_", "-")
            for name in PROFILE_CONFLICT_OPTIONS
            if name in command_line
        ]
        if conflicts:
            rendered = ", ".join(f"--{name}" for name in conflicts)
            raise EvidenceVerificationError(
                f"--trust-profile cannot be mixed with {rendered}"
            )
        _require_outside_evidence(
            options.evidence,
            options.trust_profile,
            label="independent trust profile",
        )
        try:
            loaded = load_trust_inputs(options.trust_profile)
        except TrustInputsError as exc:
            raise EvidenceVerificationError(str(exc)) from exc
        _require_outside_evidence(
            options.evidence,
            loaded.policy_path,
            label="independent policy",
        )
        _require_outside_evidence(
            options.evidence,
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
            expected_baseline_artifact = loaded.expected_artifact_digests["baseline"]
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

    if captured:
        if command_line.intersection(CAPTURED_IRRELEVANT_OPTIONS):
            raise EvidenceVerificationError(
                "native anchor/scorer flags are not valid for captured verification"
            )
        return verify_evidence(
            options.evidence,
            policy_path=policy,
            expected_baseline_run=expected_baseline_run,
            expected_subject_run=expected_subject_run,
            expected_signer=expected_signer,
            expected_request_digest=expected_request_digest,
            receipt_path=options.receipt,
            verifier_signing_key_path=verifier_signing_key,
            verifier_identity=verifier_identity,
            trust_profile_digest=trust_profile_digest,
            policy_bytes=policy_bytes,
            verifier_signing_key_bytes=verifier_signing_key_bytes,
            max_bootstrap_draws=options.max_bootstrap_draws,
        )
    if command_line.intersection(NATIVE_IRRELEVANT_OPTIONS):
        raise EvidenceVerificationError(
            "captured run/work flags are not valid for native verification"
        )
    return verify_evidence(
        options.evidence,
        policy_path=policy,
        expected_baseline_artifact=expected_baseline_artifact,
        expected_subject_artifact=expected_subject_artifact,
        expected_schedule=expected_schedule,
        expected_baseline_runtime=expected_baseline_runtime,
        expected_subject_runtime=expected_subject_runtime,
        expected_signer=expected_signer,
        expected_request_digest=expected_request_digest,
        receipt_path=options.receipt,
        verifier_signing_key_path=verifier_signing_key,
        verifier_identity=verifier_identity,
        scorer_registry=ScorerExtensionRegistry(
            allow_installed=allow_installed_scorers
        ),
        trust_profile_digest=trust_profile_digest,
        policy_bytes=policy_bytes,
        verifier_signing_key_bytes=verifier_signing_key_bytes,
    )
