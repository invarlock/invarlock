from __future__ import annotations

from types import SimpleNamespace

from invarlock.core.run_provider_dataset_materialization import _dataset_meta


def test_dataset_meta_attests_hosted_provider_identity() -> None:
    revision = "d" * 40
    provider = SimpleNamespace(
        name="hf_seq2seq",
        dataset_name="google/wmt24pp",
        config_name="de-en",
        revision=revision,
        stratification_stats=None,
        scorer_profile=None,
    )
    tokenizer = SimpleNamespace(
        name_or_path="org/tokenizer",
        vocab_size=100,
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        add_prefix_space=False,
    )

    payload = _dataset_meta(
        data_provider=provider,
        tokenizer=tokenizer,
        tokenizer_hash="tokenizer-digest",
        preview_hash="preview",
        final_hash="final",
        preview_total_tokens=10,
        final_total_tokens=12,
        min_tokens_target=20,
        tokens_floor_met=True,
        resolved_loss_type="seq2seq",
        use_mlm=False,
        preview_mask_total=0,
        final_mask_total=0,
        window_plan={},
        include_window_plan=False,
        tokenizer_digest_fn=lambda _tokenizer: "fallback",
        safe_int_fn=lambda value, _default: int(value),
    )

    assert payload["provider"] == "hf_seq2seq"
    assert payload["dataset_name"] == "google/wmt24pp"
    assert payload["config_name"] == "de-en"
    assert payload["revision"] == revision
