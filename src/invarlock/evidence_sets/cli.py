"""Adapt existing verify options without weakening either component's policy."""

from __future__ import annotations

from typing import TYPE_CHECKING, NoReturn

from invarlock.evidence_sets.verification import verify_evidence_set
from invarlock.evidence_verification import (
    EvidenceVerification,
    EvidenceVerificationError,
)

if TYPE_CHECKING:  # pragma: no cover - static imports only
    from invarlock.cli.verification_workflow import VerificationOptions


def execute_evidence_set_verification(
    options: VerificationOptions, *, command_line: frozenset[str]
) -> EvidenceVerification:
    from invarlock.cli.verification_workflow import PROFILE_CONFLICT_OPTIONS

    def fail(message: str, code: int = 2) -> NoReturn:
        raise EvidenceVerificationError(
            message,
            exit_code=code,
            payload={
                "kind": "evidence_set",
                "ok": False,
                "authenticated": False,
                "replayed": False,
                "accepted": False,
                "errors": [message],
            },
        )

    if options.trust_profile is None:
        fail(
            "Evidence set verification requires --trust-profile with an independent composition recipient policy"
        )
    conflicts = [
        name
        for name in PROFILE_CONFLICT_OPTIONS
        if name in command_line or getattr(options, name) not in (None, False)
    ]
    if conflicts:
        fail(
            "Component trust and signing options belong in recipient profiles: "
            + ", ".join("--" + name.replace("_", "-") for name in conflicts)
        )
    assert options.trust_profile is not None
    try:
        result = verify_evidence_set(
            options.evidence,
            recipient_policy=options.trust_profile,
            receipt=options.receipt,
            max_bootstrap_draws=options.max_bootstrap_draws,
        )
    except (OSError, ValueError) as exc:
        fail(str(exc))
    if not result.accepted:
        raise EvidenceVerificationError(
            "Evidence set recipient acceptance was not established",
            exit_code=result.exit_code,
            payload=result.payload,
            receipt_path=options.receipt,
        )
    return EvidenceVerification(options.evidence, result.payload, options.receipt)
