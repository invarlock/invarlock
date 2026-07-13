from __future__ import annotations

import hashlib

from invarlock.core.assurance_guard_validation_variance_provenance import (
    _variance_ab_provenance_errors,
)


def _condition(window_ids: list[str], *, mode: str) -> dict[str, object]:
    return {
        "mode": mode,
        "status": "evaluated",
        "tag": "post_edit",
        "window_ids": window_ids,
        "window_count": len(window_ids),
        "target_fingerprint": "target",
        "pairing_digest": "full-schedule",
        "consumed_pairing_digest": hashlib.blake2s(
            "||".join(window_ids).encode("utf-8"), digest_size=16
        ).hexdigest(),
        "dataset_hash": "dataset",
        "tokenizer_hash": "tokenizer",
        "model_id": "model",
        "run_seed": 7,
    }


def test_variance_provenance_accepts_exact_consumed_schedule_prefix() -> None:
    full_schedule = [f"preview::{index}" for index in range(4)]
    consumed = full_schedule[:2]
    provenance = {
        "condition_a": _condition(consumed, mode="edited_no_ve"),
        "condition_b": _condition(consumed, mode="virtual_ve"),
    }
    metrics = {"ab_provenance": provenance}
    top = {"window_ids": consumed}

    errors = _variance_ab_provenance_errors(
        metrics,
        2,
        source="guards[3]",
        top_provenance=top,
        details_stats=None,
        condition_b_statuses=frozenset({"evaluated"}),
        expected_provenance={
            "window_ids": full_schedule,
            "model_id": "model",
            "run_seed": 7,
            "dataset_hash": "dataset",
            "tokenizer_hash": "tokenizer",
            "pairing_digest": "full-schedule",
        },
    )

    assert errors == []


def test_variance_provenance_rejects_nonprefix_consumed_schedule() -> None:
    full_schedule = [f"preview::{index}" for index in range(4)]
    consumed = [full_schedule[0], full_schedule[2]]
    provenance = {
        "condition_a": _condition(consumed, mode="edited_no_ve"),
        "condition_b": _condition(consumed, mode="virtual_ve"),
    }

    errors = _variance_ab_provenance_errors(
        {"ab_provenance": provenance},
        2,
        source="guards[3]",
        top_provenance={"window_ids": consumed},
        details_stats=None,
        condition_b_statuses=frozenset({"evaluated"}),
        expected_provenance={
            "window_ids": full_schedule,
            "model_id": "model",
            "run_seed": 7,
            "dataset_hash": "dataset",
            "tokenizer_hash": "tokenizer",
            "pairing_digest": "full-schedule",
        },
    )

    assert any("consumed prefix" in error for error in errors)
