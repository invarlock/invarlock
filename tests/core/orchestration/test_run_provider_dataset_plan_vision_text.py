from __future__ import annotations

from types import SimpleNamespace

import pytest

from invarlock.core.run_provider_dataset_plan import (
    _materialize_text_provider_dataset_plan,
    _split_vision_text_counts,
    _vision_text_dataset_plan,
    build_provider_dataset_plan,
)
from tests.core._support_run_provider_dataset_plan import (
    _cfg,
    _DummyTokenizer,
    _ProviderConfig,
)


def test_build_provider_dataset_plan_supports_vision_text_examples() -> None:
    provider = SimpleNamespace(
        name="vision_text",
        dataset_name="public/vision-test",
        config_name=None,
        revision="a" * 40,
        examples=lambda split="validation": [
            {
                "id": "ex-1",
                "image_path": "/tmp/a.png",
                "prompt": "what is shown?",
                "answer": "cat",
                "answers": ["cat"],
                "image_sha256": "img-a",
                "prompt_sha256": "prompt-a",
                "answer_sha256": "answer-a",
            },
            {
                "id": "ex-2",
                "image_path": "/tmp/b.png",
                "prompt": "what is shown?",
                "answer": "dog",
                "answers": ["dog"],
                "image_sha256": "img-b",
                "prompt_sha256": "prompt-b",
                "answer_sha256": "answer-b",
            },
        ],
        digest=lambda: {"provider": "vision_text", "ids_sha256": "ids"},
    )

    result = build_provider_dataset_plan(
        cfg=_cfg(provider=_ProviderConfig("vision_text", path="demo.jsonl")),
        model_profile=SimpleNamespace(),
        resolved_device="cpu",
        profile="dev",
        profile_normalized="dev",
        requested_preview=1,
        requested_final=1,
        effective_preview=1,
        effective_final=1,
        pairing_schedule_present=False,
        use_mlm=False,
        mask_prob=0.15,
        mask_seed=43,
        random_token_prob=0.1,
        original_token_prob=0.1,
        resolved_loss_type="classification",
        tier="balanced",
        get_provider_fn=object(),
        resolve_provider_and_split_fn=lambda *args, **kwargs: (
            provider,
            "validation",
            False,
        ),
        resolve_tokenizer_fn=lambda profile: (_DummyTokenizer(), "tokhash"),
        maybe_plan_release_windows_fn=lambda **kwargs: {"actual_preview": 1},
        resolve_effective_windows_fn=lambda **kwargs: {},
        apply_mlm_masks_fn=lambda *args, **kwargs: (0, []),
        resolve_pm_min_tokens_target_fn=lambda **kwargs: 4,
        hash_sequences_fn=lambda seqs: f"hash-{len(list(seqs))}",
        tokenizer_digest_fn=lambda tokenizer: "digest",
        safe_int_fn=lambda value, default=0: int(value or default),
        tensor_or_list_to_ints_fn=lambda values: list(values),
    )

    assert result.tokenizer is None
    assert result.calibration_data[0]["example_id"] == "ex-1"
    assert result.preview_records[0]["seq_len"] == 8
    assert result.final_records[0]["example_id"] == "ex-2"
    assert result.dataset_meta["provider_kind"] == "vision_text"
    assert result.dataset_meta["provider_digest"]["provider"] == "vision_text"
    assert result.dataset_meta["dataset_name"] == "public/vision-test"
    assert result.dataset_meta["revision"] == "a" * 40


def test_vision_text_dataset_plan_requires_examples_and_keeps_window_capacity() -> None:
    with pytest.raises(TypeError, match="examples"):
        _vision_text_dataset_plan(
            data_provider=SimpleNamespace(examples=None),
            resolved_split="validation",
            used_fallback_split=False,
            cfg_dataset=SimpleNamespace(seq_len=4),
            requested_preview=1,
            requested_final=1,
            effective_preview=1,
            effective_final=1,
            resolved_loss_type="classification",
            diagnostics=[],
        )

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
        window_plan={"capacity": {"available_examples": 2}},
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

    assert result.dataset_meta["window_plan"]["capacity"] == {"available_examples": 2}
    assert result.dataset_meta["window_capacity"] == {"available_examples": 2}

    result_empty_capacity = _materialize_text_provider_dataset_plan(
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
        window_plan={"capacity": {}, "profile": "text"},
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

    assert result_empty_capacity.dataset_meta["window_plan"]["capacity"] == {}
    assert "window_capacity" not in result_empty_capacity.dataset_meta

    result_without_capacity = _materialize_text_provider_dataset_plan(
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
        window_plan={"profile": "text"},
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

    assert result_without_capacity.dataset_meta["window_plan"]["profile"] == "text"
    assert "window_capacity" not in result_without_capacity.dataset_meta

    class _FalseyWindowPlan(dict[str, object]):
        def __bool__(self) -> bool:
            return False

    result_falsey_window_plan = _materialize_text_provider_dataset_plan(
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
        window_plan=_FalseyWindowPlan(),
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

    assert "window_plan" not in result_falsey_window_plan.dataset_meta
    assert result_falsey_window_plan.window_plan == {
        "actual_preview": 1,
        "actual_final": 1,
        "coverage_ok": True,
        "preview_total_tokens": 2,
        "final_total_tokens": 2,
        "min_tokens_target": 4,
        "tokens_floor_met": True,
    }


def test_split_vision_text_counts_covers_capacity_edges() -> None:
    assert _split_vision_text_counts(
        available=0,
        requested_preview=1,
        requested_final=1,
        effective_preview=1,
        effective_final=1,
    ) == (0, 0)
    assert _split_vision_text_counts(
        available=4,
        requested_preview=1,
        requested_final=1,
        effective_preview=0,
        effective_final=0,
    ) == (0, 0)
    assert _split_vision_text_counts(
        available=2,
        requested_preview=0,
        requested_final=3,
        effective_preview=0,
        effective_final=3,
    ) == (0, 2)
    assert _split_vision_text_counts(
        available=2,
        requested_preview=3,
        requested_final=0,
        effective_preview=3,
        effective_final=0,
    ) == (2, 0)


def test_split_vision_text_counts_distributes_remaining_capacity() -> None:
    assert _split_vision_text_counts(
        available=7,
        requested_preview=5,
        requested_final=5,
        effective_preview=5,
        effective_final=5,
    ) == (4, 3)
    assert _split_vision_text_counts(
        available=3,
        requested_preview=3,
        requested_final=1,
        effective_preview=3,
        effective_final=1,
    ) == (2, 1)


def test_vision_text_dataset_plan_rebalances_when_final_arm_would_be_empty() -> None:
    provider = SimpleNamespace(
        examples=lambda split="validation": [
            {"id": "ex-1", "image_path": "/tmp/a.png", "answers": ["cat"]}
        ],
        digest=lambda: {"provider": "vision_text"},
    )

    result = _vision_text_dataset_plan(
        data_provider=provider,
        resolved_split="validation",
        used_fallback_split=False,
        cfg_dataset=SimpleNamespace(seq_len=4),
        requested_preview=1,
        requested_final=1,
        effective_preview=1,
        effective_final=1,
        resolved_loss_type="classification",
        diagnostics=[],
    )

    assert result.preview_count == 0
    assert result.final_count == 1
    assert result.final_records[0]["example_id"] == "ex-1"
    assert result.window_plan is not None
    assert result.window_plan["coverage_ok"] is True


def test_vision_text_dataset_plan_splits_release_shortage_across_arms() -> None:
    provider = SimpleNamespace(
        examples=lambda split="validation": [
            {"id": f"ex-{index}", "image_path": f"/tmp/{index}.png", "answers": ["cat"]}
            for index in range(64)
        ],
        digest=lambda: {"provider": "vision_text"},
    )

    result = _vision_text_dataset_plan(
        data_provider=provider,
        resolved_split="validation",
        used_fallback_split=False,
        cfg_dataset=SimpleNamespace(seq_len=256),
        requested_preview=400,
        requested_final=400,
        effective_preview=400,
        effective_final=400,
        resolved_loss_type="classification",
        diagnostics=[],
    )

    assert result.preview_count == 32
    assert result.final_count == 32
    assert result.preview_records[0]["example_id"] == "ex-0"
    assert result.final_records[0]["example_id"] == "ex-32"
    assert result.window_plan is not None
    assert result.window_plan["capacity"] == {"available_examples": 64}


def test_vision_text_dataset_plan_normal_path_handles_noncallable_digest() -> None:
    provider = SimpleNamespace(
        examples=lambda split="validation": [
            {"example_id": "ex-1", "prompt": "what", "answer": "cat"},
            {"id": "ex-2", "prompt": "what", "answer": "dog"},
        ],
        digest="not-callable",
    )

    result = _vision_text_dataset_plan(
        data_provider=provider,
        resolved_split="validation",
        used_fallback_split=False,
        cfg_dataset=SimpleNamespace(seq_len=6),
        requested_preview=1,
        requested_final=1,
        effective_preview=1,
        effective_final=1,
        resolved_loss_type="classification",
        diagnostics=[],
    )

    assert result.preview_count == 1
    assert result.final_count == 1
    assert result.preview_records[0]["example_id"] == "ex-1"
    assert result.final_records[0]["example_id"] == "ex-2"
    assert result.calibration_data[0]["window_id"] == "preview::ex-1"
    assert result.calibration_data[1]["window_id"] == "final::ex-2"
    assert result.dataset_meta["provider_digest"] == {}
    assert result.final_records[0]["seq_len"] == 6
