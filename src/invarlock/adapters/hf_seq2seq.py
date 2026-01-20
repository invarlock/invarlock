"""
HuggingFace encoder-decoder adapter.
===================================

Role-based adapter for HuggingFace encoder-decoder (seq2seq) models.
"""

from __future__ import annotations

from .hf_t5 import HF_T5_Adapter


class HF_Seq2Seq_Adapter(HF_T5_Adapter):
    name = "hf_seq2seq"


__all__ = ["HF_Seq2Seq_Adapter"]

