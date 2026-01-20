"""
HuggingFace masked LM adapter.
==============================

Role-based adapter for HuggingFace masked language models (e.g., BERT-family).
"""

from __future__ import annotations

from .hf_bert import HF_BERT_Adapter


class HF_MLM_Adapter(HF_BERT_Adapter):
    name = "hf_mlm"


__all__ = ["HF_MLM_Adapter"]

