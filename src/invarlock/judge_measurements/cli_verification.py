"""CLI adaptation for the independently supplied judge recipient policy."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, NoReturn

from invarlock.captured_contracts import atomic_write
from invarlock.evidence_verification import (
    EvidenceVerification,
    EvidenceVerificationError,
    _require_outside_evidence,
)
from invarlock.judge_measurements.contracts import canonical_payload

if TYPE_CHECKING:
    from invarlock.cli.verification_workflow import VerificationOptions


def execute_judge_verification(
    options: VerificationOptions, *, command_line: frozenset[str]
) -> EvidenceVerification:
    from invarlock.cli.verification_workflow import PROFILE_CONFLICT_OPTIONS
    from invarlock.judge_measurements.acceptance import (
        verify_judge_evidence_with_policy,
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
        if name in command_line or getattr(options, name) not in (None, False)
    ]
    if "max_bootstrap_draws" in command_line:
        conflicting.append("max_bootstrap_draws")
    if conflicting:
        fail(
            "Legacy trust or receipt-signing options do not apply to judge verification: "
            + ", ".join("--" + name.replace("_", "-") for name in conflicting)
        )
    assert options.trust_profile is not None
    try:
        if options.receipt is not None:
            _require_outside_evidence(
                options.evidence, options.receipt, label="judge receipt"
            )
        receipt = verify_judge_evidence_with_policy(
            options.evidence, options.trust_profile
        )
        payload = {
            "format_version": "invarlock/judge-verification-result-v1",
            "kind": "judge",
            "ok": receipt.verified,
            **receipt.to_dict(),
        }
        if options.receipt is not None:
            atomic_write(
                Path(options.receipt), canonical_payload(receipt.to_dict()) + b"\n"
            )
            payload["receipt"] = str(options.receipt)
    except (OSError, ValueError) as exc:
        fail(str(exc), 4)
    if not receipt.accepted:
        exit_code = 4 if not receipt.verified else 7
        raise EvidenceVerificationError(
            "Judge recipient acceptance was not established",
            exit_code=exit_code,
            payload=payload,
        )
    return EvidenceVerification(Path(options.evidence), payload, options.receipt)
