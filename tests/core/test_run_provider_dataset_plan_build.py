from __future__ import annotations

from types import SimpleNamespace

from invarlock.core.run_provider_dataset_plan import (
    ProviderDatasetPlanDiagnostic,
    build_provider_dataset_plan,
)
from invarlock.eval.data_support import DatasetDiagnostic
from tests.core._support_run_provider_dataset_plan import (
    _cfg,
    _DummyTokenizer,
    _ProviderConfig,
    _Seq2SeqProvider,
    _TokenizerWithoutName,
)


def test_build_provider_dataset_plan_collects_diagnostics_and_provider_kwargs() -> None:
    captured: dict[str, object] = {}
    provider = _Seq2SeqProvider()

    def _resolve_provider_and_split(
        cfg: object,
        model_profile: object,
        *,
        get_provider_fn: object,
        provider_kwargs: dict[str, object] | None,
        resolved_device: str | None,
    ) -> tuple[object, str, bool]:
        del cfg, model_profile, get_provider_fn, resolved_device
        captured["provider_kwargs"] = provider_kwargs
        return provider, "validation", False

    def _resolve_effective_windows(**kwargs: object) -> dict[str, object]:
        diagnostic_fn = kwargs["diagnostic_fn"]
        diagnostic_fn(
            DatasetDiagnostic(
                kind="window.dedupe_adjustment",
                severity="warning",
                message="dedupe adjusted",
                metadata={"deficit": 1, "proposed_per_arm": 1},
            )
        )
        return {
            "preview_records": [
                {"input_ids": [1, 2], "attention_mask": [1, 1], "dataset_index": 7}
            ],
            "final_records": [
                {"input_ids": [3, 4], "attention_mask": [1, 1], "dataset_index": 9}
            ],
            "actual_preview": 1,
            "actual_final": 1,
            "preview_total_tokens": 2,
            "final_total_tokens": 2,
            "dedupe_adjustments": [],
        }

    result = build_provider_dataset_plan(
        cfg=_cfg(provider=_ProviderConfig("local_jsonl", path="demo.jsonl")),
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
        resolved_loss_type="ppl_seq2seq",
        tier="balanced",
        get_provider_fn=object(),
        resolve_provider_and_split_fn=_resolve_provider_and_split,
        resolve_tokenizer_fn=lambda profile: (_DummyTokenizer(), "tokhash"),
        maybe_plan_release_windows_fn=lambda **kwargs: {"actual_preview": 1},
        resolve_effective_windows_fn=_resolve_effective_windows,
        apply_mlm_masks_fn=lambda *args, **kwargs: (0, []),
        resolve_pm_min_tokens_target_fn=lambda **kwargs: 4,
        hash_sequences_fn=lambda seqs: f"hash-{len(list(seqs))}",
        tokenizer_digest_fn=lambda tokenizer: "digest",
        safe_int_fn=lambda value, default=0: int(value or default),
        tensor_or_list_to_ints_fn=lambda values: list(values),
    )

    assert captured["provider_kwargs"] == {
        "dataset_name": "demo",
        "path": "demo.jsonl",
    }
    assert result.calibration_data[1]["labels"] == [21, 22]
    assert result.preview_records[0]["labels"] == [11, 12]
    assert result.dataset_meta["stratification"] == {"mode": "balanced"}
    assert result.dataset_meta["scorer_profile"] == {"kind": "seq2seq"}
    assert result.dataset_meta["loss_type"] == "ppl_seq2seq"
    assert result.diagnostics == (
        ProviderDatasetPlanDiagnostic(
            code="provider.resolved",
            summary="provider resolved",
            level="info",
            context={
                "provider": "seq2seq",
                "split": "validation",
                "used_fallback_split": False,
            },
        ),
        ProviderDatasetPlanDiagnostic(
            code="window.dedupe_adjustment",
            summary="dedupe adjusted",
            level="warning",
            context={"deficit": 1, "proposed_per_arm": 1},
        ),
    )


def test_build_provider_dataset_plan_leaves_missing_tokenizer_name_nullable() -> None:
    result = build_provider_dataset_plan(
        cfg=_cfg(provider=_ProviderConfig("local_jsonl", path="demo.jsonl")),
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
        resolved_loss_type="ppl_seq2seq",
        tier="balanced",
        get_provider_fn=object(),
        resolve_provider_and_split_fn=lambda *args, **kwargs: (
            _Seq2SeqProvider(),
            "validation",
            False,
        ),
        resolve_tokenizer_fn=lambda profile: (_TokenizerWithoutName(), None),
        maybe_plan_release_windows_fn=lambda **kwargs: {"actual_preview": 1},
        resolve_effective_windows_fn=lambda **kwargs: {
            "preview_records": [
                {"input_ids": [1, 2], "attention_mask": [1, 1], "dataset_index": 7}
            ],
            "final_records": [
                {"input_ids": [3, 4], "attention_mask": [1, 1], "dataset_index": 9}
            ],
            "actual_preview": 1,
            "actual_final": 1,
            "preview_total_tokens": 2,
            "final_total_tokens": 2,
            "dedupe_adjustments": [],
        },
        apply_mlm_masks_fn=lambda *args, **kwargs: (0, []),
        resolve_pm_min_tokens_target_fn=lambda **kwargs: 4,
        hash_sequences_fn=lambda seqs: f"hash-{len(list(seqs))}",
        tokenizer_digest_fn=lambda tokenizer: "digest",
        safe_int_fn=lambda value, default=0: int(value or default),
        tensor_or_list_to_ints_fn=lambda values: list(values),
    )

    assert result.dataset_meta["tokenizer_name"] is None


def test_build_provider_dataset_plan_release_without_capacity_emits_warning() -> None:
    provider = SimpleNamespace()

    result = build_provider_dataset_plan(
        cfg=_cfg(provider="wikitext2", release=True),
        model_profile=SimpleNamespace(),
        resolved_device="cpu",
        profile="release",
        profile_normalized="release",
        requested_preview=2,
        requested_final=2,
        effective_preview=2,
        effective_final=2,
        pairing_schedule_present=False,
        use_mlm=False,
        mask_prob=0.15,
        mask_seed=43,
        random_token_prob=0.1,
        original_token_prob=0.1,
        resolved_loss_type="ppl_causal",
        tier="balanced",
        get_provider_fn=object(),
        resolve_provider_and_split_fn=lambda *args, **kwargs: (
            provider,
            "validation",
            True,
        ),
        resolve_tokenizer_fn=lambda profile: (_DummyTokenizer(), "tokhash"),
        maybe_plan_release_windows_fn=lambda **kwargs: {"actual_preview": 2},
        resolve_effective_windows_fn=lambda **kwargs: {
            "preview_records": [
                {"input_ids": [1, 2], "attention_mask": [1, 1], "dataset_index": 1},
                {"input_ids": [3, 4], "attention_mask": [1, 1], "dataset_index": 2},
            ],
            "final_records": [
                {"input_ids": [5, 6], "attention_mask": [1, 1], "dataset_index": 3},
                {"input_ids": [7, 8], "attention_mask": [1, 1], "dataset_index": 4},
            ],
            "actual_preview": 2,
            "actual_final": 2,
            "preview_total_tokens": 4,
            "final_total_tokens": 4,
            "dedupe_adjustments": [],
        },
        apply_mlm_masks_fn=lambda *args, **kwargs: (0, []),
        resolve_pm_min_tokens_target_fn=lambda **kwargs: 3,
        hash_sequences_fn=lambda seqs: f"hash-{len(list(seqs))}",
        tokenizer_digest_fn=lambda tokenizer: "digest",
        safe_int_fn=lambda value, default=0: int(value or default),
        tensor_or_list_to_ints_fn=lambda values: list(values),
    )

    assert result.used_fallback_split is True
    assert result.window_plan is not None
    assert result.window_plan["profile"] == "release"
    assert result.window_plan["coverage_ok"] is True
    assert any(
        diagnostic.code == "provider.capacity_missing"
        for diagnostic in result.diagnostics
    )


def test_build_provider_dataset_plan_mlm_populates_mask_counts_and_metadata() -> None:
    class _MlmProvider:
        pass

    def _apply_masks(
        records: list[dict[str, object]],
        *,
        prefix: str,
        **_: object,
    ) -> tuple[int, list[int]]:
        if prefix == "preview":
            records[0]["labels"] = [-100, 99]
            records[0]["mlm_masked"] = 1
            return 1, [1]
        records[0]["labels"] = [77, -100]
        records[0]["mlm_masked"] = 1
        return 1, [1]

    result = build_provider_dataset_plan(
        cfg=_cfg(provider="wikitext2"),
        model_profile=SimpleNamespace(),
        resolved_device="cpu",
        profile="dev",
        profile_normalized="dev",
        requested_preview=1,
        requested_final=1,
        effective_preview=1,
        effective_final=1,
        pairing_schedule_present=False,
        use_mlm=True,
        mask_prob=0.15,
        mask_seed=43,
        random_token_prob=0.1,
        original_token_prob=0.1,
        resolved_loss_type="mlm_ce",
        tier="balanced",
        get_provider_fn=object(),
        resolve_provider_and_split_fn=lambda *args, **kwargs: (
            _MlmProvider(),
            "validation",
            False,
        ),
        resolve_tokenizer_fn=lambda profile: (_DummyTokenizer(), None),
        maybe_plan_release_windows_fn=lambda **kwargs: {"actual_preview": 1},
        resolve_effective_windows_fn=lambda **kwargs: {
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
            "dedupe_adjustments": ["kept"],
        },
        apply_mlm_masks_fn=_apply_masks,
        resolve_pm_min_tokens_target_fn=lambda **kwargs: 5,
        hash_sequences_fn=lambda seqs: f"hash-{len(list(seqs))}",
        tokenizer_digest_fn=lambda tokenizer: "digest-fallback",
        safe_int_fn=lambda value, default=0: int(value or default),
        tensor_or_list_to_ints_fn=lambda values: list(values),
    )

    assert result.preview_mask_counts == [1]
    assert result.final_mask_counts == [1]
    assert result.calibration_data[0]["labels"] == [-100, 99]
    assert result.calibration_data[1]["labels"] == [77, -100]
    assert result.dataset_meta["tokenizer_hash"] == "digest-fallback"
    assert result.dataset_meta["masked_tokens_total"] == 2
    assert result.window_plan is not None
    assert result.window_plan["dedupe_adjustments"] == ["kept"]
