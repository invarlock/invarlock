from __future__ import annotations

from types import SimpleNamespace

from invarlock.core.run_provider_dataset_materialization import (
    _calibration_entry,
    _optional_text,
)
from invarlock.core.run_provider_dataset_plan import (
    _materialize_text_provider_dataset_plan,
)
from tests.core._support_run_provider_dataset_plan import _DummyTokenizer


def test_materialization_private_edges_for_blank_tokenizer_and_short_labels() -> None:
    assert _optional_text("   ") is None
    assert _optional_text(123) is None

    entry = _calibration_entry(
        {"input_ids": [1, 2], "attention_mask": [1, 1], "dataset_index": 3},
        arm="final",
        index=1,
        use_mlm=False,
        provider_labels=[[7, 8]],
        tensor_or_list_to_ints_fn=lambda values: list(values),
    )

    assert entry["window_id"] == "final::1"
    assert "labels" not in entry


def test_materialize_text_provider_dataset_plan_handles_provider_attr_failures() -> (
    None
):
    class _BrokenProvider:
        def __getattribute__(self, name: str) -> object:
            if name in {"last_preview_labels", "last_final_labels"}:
                raise TypeError("labels unavailable")
            return super().__getattribute__(name)

    result = _materialize_text_provider_dataset_plan(
        data_provider=_BrokenProvider(),
        resolved_split="validation",
        used_fallback_split=False,
        tokenizer=_DummyTokenizer(),
        tokenizer_hash=None,
        effective_windows={
            "preview_records": [
                {"input_ids": [1, 2], "attention_mask": [1, 1], "dataset_index": 1}
            ],
            "final_records": [
                {"input_ids": [3, 4], "attention_mask": [1, 1], "dataset_index": 2}
            ],
            "actual_preview": 1,
            "actual_final": 1,
            "preview_total_tokens": 2,
            "final_total_tokens": 2,
            "dedupe_adjustments": [],
        },
        requested_preview=1,
        requested_final=1,
        effective_preview=1,
        effective_final=1,
        resolved_loss_type="ppl_causal",
        window_plan=None,
        use_mlm=False,
        mask_prob=0.15,
        mask_seed=43,
        random_token_prob=0.1,
        original_token_prob=0.1,
        tier="balanced",
        profile="dev",
        apply_mlm_masks_fn=lambda *args, **kwargs: (0, []),
        resolve_pm_min_tokens_target_fn=lambda **kwargs: 3,
        hash_sequences_fn=lambda seqs: f"hash-{len(list(seqs))}",
        tokenizer_digest_fn=lambda tokenizer: "digest-fallback",
        safe_int_fn=lambda value, default=0: int(value or default),
        tensor_or_list_to_ints_fn=lambda values: list(values),
        diagnostics=[],
    )

    assert result.dataset_meta["tokenizer_hash"] == "digest-fallback"
    assert result.window_plan is not None
    assert result.window_plan["profile"] == "dev"
    assert result.window_plan["tokens_floor_met"] is True
    assert "labels" not in result.calibration_data[1]


def test_materialize_text_provider_dataset_plan_handles_empty_arms_without_mlm() -> (
    None
):
    result = _materialize_text_provider_dataset_plan(
        data_provider=SimpleNamespace(
            last_preview_labels=None,
            last_final_labels=None,
            stratification_stats=None,
            scorer_profile=None,
        ),
        resolved_split="validation",
        used_fallback_split=False,
        tokenizer=_DummyTokenizer(),
        tokenizer_hash="tokhash",
        effective_windows={
            "preview_records": [],
            "final_records": [],
            "actual_preview": 0,
            "actual_final": 0,
            "preview_total_tokens": 0,
            "final_total_tokens": 0,
            "dedupe_adjustments": [],
        },
        requested_preview=0,
        requested_final=0,
        effective_preview=0,
        effective_final=0,
        resolved_loss_type="ppl_causal",
        window_plan={"profile": "text", "capacity": {}},
        use_mlm=False,
        mask_prob=0.15,
        mask_seed=43,
        random_token_prob=0.1,
        original_token_prob=0.1,
        tier="balanced",
        profile="dev",
        apply_mlm_masks_fn=lambda *args, **kwargs: (0, []),
        resolve_pm_min_tokens_target_fn=lambda **kwargs: 0,
        hash_sequences_fn=lambda seqs: f"hash-{len(list(seqs))}",
        tokenizer_digest_fn=lambda tokenizer: "digest-fallback",
        safe_int_fn=lambda value, default=0: int(value or default),
        tensor_or_list_to_ints_fn=lambda values: list(values),
        diagnostics=[],
    )

    assert result.calibration_data == []
    assert result.preview_mask_counts == []
    assert result.final_mask_counts == []
    assert result.window_plan is not None
    assert result.window_plan["actual_preview"] == 0
    assert "window_capacity" not in result.dataset_meta


def test_materialize_text_provider_dataset_plan_uses_provider_labels_and_metadata() -> (
    None
):
    result = _materialize_text_provider_dataset_plan(
        data_provider=SimpleNamespace(
            last_preview_labels=[[9, 8]],
            last_final_labels=[[7, 6]],
            stratification_stats={"mode": "balanced"},
            scorer_profile={"kind": "seq2seq"},
        ),
        resolved_split="validation",
        used_fallback_split=False,
        tokenizer=_DummyTokenizer(),
        tokenizer_hash="tokhash",
        effective_windows={
            "preview_records": [
                {"input_ids": [1, 2], "attention_mask": [1, 1], "dataset_index": 1}
            ],
            "final_records": [
                {"input_ids": [3, 4], "attention_mask": [1, 1], "dataset_index": 2}
            ],
            "actual_preview": 1,
            "actual_final": 1,
            "preview_total_tokens": 2,
            "final_total_tokens": 2,
            "dedupe_adjustments": [],
        },
        requested_preview=1,
        requested_final=1,
        effective_preview=1,
        effective_final=1,
        resolved_loss_type="ppl_seq2seq",
        window_plan=None,
        use_mlm=False,
        mask_prob=0.15,
        mask_seed=43,
        random_token_prob=0.1,
        original_token_prob=0.1,
        tier="balanced",
        profile="dev",
        apply_mlm_masks_fn=lambda *args, **kwargs: (0, []),
        resolve_pm_min_tokens_target_fn=lambda **kwargs: 4,
        hash_sequences_fn=lambda seqs: f"hash-{len(list(seqs))}",
        tokenizer_digest_fn=lambda tokenizer: "digest-fallback",
        safe_int_fn=lambda value, default=0: int(value or default),
        tensor_or_list_to_ints_fn=lambda values: list(values),
        diagnostics=[],
    )

    assert result.preview_records[0]["labels"] == [9, 8]
    assert result.calibration_data[0]["labels"] == [9, 8]
    assert result.calibration_data[1]["labels"] == [7, 6]
    assert result.dataset_meta["stratification"] == {"mode": "balanced"}
    assert result.dataset_meta["scorer_profile"] == {"kind": "seq2seq"}
    assert result.window_plan is not None
    assert result.window_plan["profile"] == "dev"
