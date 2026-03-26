from __future__ import annotations

from .exceptions import InvarlockError


def enforce_provider_parity(
    subject_digest: dict | None,
    baseline_digest: dict | None,
    *,
    profile: str | None,
    invarlock_error_cls: type[BaseException] = InvarlockError,
) -> None:
    """Enforce tokenizer/masking parity rules for CI and release profiles."""

    prof = (profile or "").strip().lower()
    if prof not in {"ci", "release"}:
        return

    subject = subject_digest or {}
    baseline = baseline_digest or {}
    subj_ids = subject.get("ids_sha256")
    base_ids = baseline.get("ids_sha256")
    subj_tok = subject.get("tokenizer_sha256")
    base_tok = baseline.get("tokenizer_sha256")
    subj_mask = subject.get("masking_sha256")
    base_mask = baseline.get("masking_sha256")

    if not (
        isinstance(subj_ids, str)
        and isinstance(base_ids, str)
        and subj_ids
        and base_ids
        and isinstance(subj_tok, str)
        and isinstance(base_tok, str)
        and subj_tok
        and base_tok
    ):
        raise invarlock_error_cls(
            code="E004",
            message="PROVIDER-DIGEST-MISSING: subject or baseline missing ids/tokenizer digest",
        )

    if subj_ids != base_ids:
        raise invarlock_error_cls(
            code="E006",
            message="IDS-DIGEST-MISMATCH: subject and baseline window IDs differ",
        )

    if subj_tok != base_tok:
        raise invarlock_error_cls(
            code="E002",
            message="TOKENIZER-DIGEST-MISMATCH: subject and baseline tokenizers differ",
        )

    if (
        isinstance(subj_mask, str)
        and isinstance(base_mask, str)
        and subj_mask
        and base_mask
        and subj_mask != base_mask
    ):
        raise invarlock_error_cls(
            code="E003",
            message="MASK-PARITY-MISMATCH: mask positions differ under matched tokenizers",
        )


__all__ = ["enforce_provider_parity"]
