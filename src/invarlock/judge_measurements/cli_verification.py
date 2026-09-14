"""CLI adaptation for the independently supplied judge recipient policy."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, NoReturn

from invarlock.evidence_verification import (
    EvidenceVerification,
    EvidenceVerificationError,
    _require_outside_evidence,
)

if TYPE_CHECKING:
    from invarlock.cli.verification_workflow import VerificationOptions


def execute_judge_verification(
    options: VerificationOptions, *, command_line: frozenset[str]
) -> EvidenceVerification:
    from invarlock.cli.verification_workflow import PROFILE_CONFLICT_OPTIONS
    from invarlock.judge_measurements.acceptance import (
        verify_judge_evidence_with_policy,
        write_signed_judge_verification_receipt,
    )

    def fail(message: str, code: int = 2) -> NoReturn:
        raise EvidenceVerificationError(
            message,
            exit_code=code,
            payload={
                "format_version": "invarlock/judge-verification-result-v1",
                "kind": "judge",
                "ok": False,
                "authenticated": False,
                "replayed": False,
                "accepted": False,
                "errors": [message[:1024]],
            },
        )

    if options.trust_profile is None:
        fail(
            "Judge verification requires --trust-profile with an independently maintained judge recipient policy"
        )
    conflicting = [
        name
        for name in PROFILE_CONFLICT_OPTIONS
        if name not in {"verifier_signing_key", "verifier_identity"}
        if name in command_line or getattr(options, name) not in (None, False)
    ]
    if "max_bootstrap_draws" in command_line:
        conflicting.append("max_bootstrap_draws")
    if conflicting:
        fail(
            "Legacy trust options do not apply to judge verification: "
            + ", ".join("--" + name.replace("_", "-") for name in conflicting)
        )
    if options.receipt is None and (
        options.verifier_signing_key is not None
        or options.verifier_identity is not None
    ):
        fail(
            "--verifier-signing-key and --verifier-identity require --receipt for judge verification"
        )
    if options.receipt is not None and (
        options.verifier_signing_key is None or options.verifier_identity is None
    ):
        fail(
            "Judge --receipt requires both --verifier-signing-key and --verifier-identity"
        )
    assert options.trust_profile is not None
    try:
        if options.receipt is not None:
            _require_outside_evidence(
                options.evidence, options.receipt, label="judge receipt"
            )
        result = verify_judge_evidence_with_policy(
            options.evidence, options.trust_profile
        )
        result_value = result.to_dict()
        result_format = result_value.pop("format")
        payload = {
            "format_version": result_format,
            "kind": "judge",
            "ok": result.accepted,
            **result_value,
        }
        if options.receipt is not None:
            assert options.verifier_signing_key is not None
            assert options.verifier_identity is not None
            write_signed_judge_verification_receipt(
                options.evidence,
                result,
                options.receipt,
                recipient_policy_path=options.trust_profile,
                verifier_identity=options.verifier_identity,
                verifier_signing_key_path=options.verifier_signing_key,
            )
            payload["receipt"] = str(options.receipt)
    except (OSError, ValueError) as exc:
        fail(str(exc), 4)
    if not result.accepted:
        exit_code = 4 if not result.verified else 7
        raise EvidenceVerificationError(
            "Judge recipient acceptance was not established",
            exit_code=exit_code,
            payload=payload,
        )
    return EvidenceVerification(Path(options.evidence), payload, options.receipt)
