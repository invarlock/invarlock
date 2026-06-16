from __future__ import annotations

from types import SimpleNamespace

import torch

from invarlock.adapters.hf_seq2seq import HF_Seq2Seq_Adapter


class _BadModelType:
    def __str__(self) -> str:
        raise ValueError("bad model type")


def test_seq2seq_can_handle_falls_back_to_encoder_decoder_heuristic() -> None:
    adapter = HF_Seq2Seq_Adapter()
    model = SimpleNamespace(
        config=SimpleNamespace(
            model_type=_BadModelType(),
            is_encoder_decoder=True,
        ),
        lm_head=object(),
        shared=object(),
    )

    assert adapter.can_handle(model) is True


def test_seq2seq_describe_handles_parameter_probe_failures() -> None:
    adapter = HF_Seq2Seq_Adapter()

    class _BrokenModel:
        def __init__(self) -> None:
            self.config = SimpleNamespace(
                num_layers=2,
                num_heads=4,
                d_model=16,
                d_ff=64,
                vocab_size=128,
            )
            self.lm_head = SimpleNamespace(weight=object())
            self.shared = SimpleNamespace(weight=object())

        def parameters(self):
            raise RuntimeError("parameters unavailable")

    desc = adapter.describe(_BrokenModel())

    assert desc["device"] == "cpu"
    assert desc["total_params"] == 0
    assert desc["n_layer"] == 4


def test_seq2seq_prepare_generation_inputs_preserves_decoder_labels() -> None:
    adapter = HF_Seq2Seq_Adapter()

    prepared = adapter.prepare_generation_inputs(
        {
            "input_ids": [1, 2, 3],
            "attention_mask": [1, 1, 1],
            "labels": [5, 6, -100],
        },
        torch.device("cpu"),
    )

    assert tuple(prepared["input_ids"].shape) == (1, 3)
    assert tuple(prepared["attention_mask"].shape) == (1, 3)
    assert prepared["labels"].tolist() == [[5, 6, -100]]
