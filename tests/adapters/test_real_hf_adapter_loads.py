"""Offline live-model regression coverage for the built-in HF adapters.

These tests intentionally use locally saved, randomly initialized tiny models.
They verify the adapter's real Hugging Face loading path and a finite forward
pass without downloading a model or turning fixture metadata into evidence.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from invarlock.adapters.auto import HF_Auto_Adapter
from invarlock.adapters.hf_causal import HF_Causal_Adapter
from invarlock.adapters.hf_mlm import HF_MLM_Adapter
from invarlock.adapters.hf_multimodal import HF_Multimodal_Adapter
from invarlock.adapters.hf_seq2seq import HF_Seq2Seq_Adapter

transformers = pytest.importorskip("transformers")


def _save_tiny_causal_model(path: Path) -> None:
    config = transformers.LlamaConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
    )
    transformers.LlamaForCausalLM(config).save_pretrained(
        path,
        safe_serialization=True,
    )


@pytest.mark.parametrize(
    ("adapter", "model_factory", "inputs", "expected_layers"),
    [
        (
            HF_Causal_Adapter,
            _save_tiny_causal_model,
            {"input_ids": torch.tensor([[1, 2, 3, 4]])},
            1,
        ),
        (
            HF_MLM_Adapter,
            lambda path: transformers.BertForMaskedLM(
                transformers.BertConfig(
                    vocab_size=128,
                    hidden_size=32,
                    intermediate_size=64,
                    num_hidden_layers=1,
                    num_attention_heads=4,
                    max_position_embeddings=64,
                )
            ).save_pretrained(path, safe_serialization=True),
            {"input_ids": torch.tensor([[1, 2, 3, 4]])},
            1,
        ),
        (
            HF_Seq2Seq_Adapter,
            lambda path: transformers.T5ForConditionalGeneration(
                transformers.T5Config(
                    vocab_size=128,
                    d_model=32,
                    d_ff=64,
                    num_layers=1,
                    num_decoder_layers=1,
                    num_heads=4,
                    decoder_start_token_id=0,
                    eos_token_id=1,
                    pad_token_id=0,
                )
            ).save_pretrained(path, safe_serialization=True),
            {
                "input_ids": torch.tensor([[1, 2, 3, 4]]),
                "decoder_input_ids": torch.tensor([[0, 1, 2]]),
            },
            2,
        ),
    ],
)
def test_core_hf_adapters_load_local_checkpoints_and_run_finite_inference(
    tmp_path: Path,
    adapter: type[HF_Causal_Adapter | HF_MLM_Adapter | HF_Seq2Seq_Adapter],
    model_factory: object,
    inputs: dict[str, torch.Tensor],
    expected_layers: int,
) -> None:
    checkpoint = tmp_path / "model"
    assert callable(model_factory)
    model_factory(checkpoint)

    loaded_adapter = adapter()
    model = loaded_adapter.load_model(str(checkpoint), device="cpu")
    model.eval()
    with torch.inference_mode():
        output = model(**inputs)

    assert torch.isfinite(output.logits).all()
    assert loaded_adapter.can_handle(model) is True
    assert loaded_adapter.describe(model)["n_layer"] == expected_layers


def test_auto_adapter_loads_a_real_local_causal_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "causal"
    _save_tiny_causal_model(checkpoint)

    adapter = HF_Auto_Adapter()
    model = adapter.load_model(str(checkpoint), device="cpu")
    model.eval()
    with torch.inference_mode():
        output = model(input_ids=torch.tensor([[1, 2, 3, 4]]))

    assert torch.isfinite(output.logits).all()
    assert adapter.describe(model)["n_layer"] == 1


def test_multimodal_adapter_loads_and_uses_a_real_image_conditioned_model(
    tmp_path: Path,
) -> None:
    try:
        llava_config = transformers.LlavaConfig
        llava_model = transformers.LlavaForConditionalGeneration
        vision_config = transformers.CLIPVisionConfig
    except AttributeError as exc:
        pytest.skip(f"current transformers build has no Llava runtime: {exc}")

    checkpoint = tmp_path / "multimodal"
    text_config = transformers.LlamaConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
    )
    image_config = vision_config(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        image_size=32,
        patch_size=16,
        projection_dim=32,
    )
    llava_model(
        llava_config(
            text_config=text_config,
            vision_config=image_config,
            image_token_index=127,
            image_seq_length=4,
        )
    ).save_pretrained(checkpoint, safe_serialization=True)

    adapter = HF_Multimodal_Adapter()
    model = adapter.load_model(str(checkpoint), device="cpu")
    model.eval()
    with torch.inference_mode():
        output = model(
            input_ids=torch.tensor([[127, 127, 127, 127, 2]]),
            pixel_values=torch.rand((1, 3, 32, 32)),
        )

    assert torch.isfinite(output.logits).all()
    assert adapter.describe(model)["n_layer"] == 1
