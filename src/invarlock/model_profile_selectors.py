from __future__ import annotations


def bert_selectors() -> dict[str, list[str]]:
    return {
        "attention": [
            "attention.self.query",
            "attention.self.key",
            "attention.self.value",
            "attention.output.dense",
        ],
        "ffn": [
            "intermediate.dense",
            "output.dense",
        ],
    }


def gpt2_selectors() -> dict[str, list[str]]:
    return {
        "attention": [
            "attn.c_attn",
            "attn.c_proj",
        ],
        "ffn": [
            "mlp.c_fc",
            "mlp.c_proj",
        ],
    }


def falcon_selectors() -> dict[str, list[str]]:
    """Selectors for Transformers' Falcon decoder blocks.

    Falcon fuses Q/K/V into ``query_key_value`` and uses two dense FFN
    projections.  Those names are not compatible with either GPT-2's Conv1D
    layout or the split projections used by the RoPE decoder families.
    """

    return {
        "attention": [
            "self_attention.query_key_value",
            "self_attention.dense",
        ],
        "ffn": [
            "mlp.dense_h_to_4h",
            "mlp.dense_4h_to_h",
        ],
    }


def dense_rope_decoder_selectors() -> dict[str, list[str]]:
    return {
        "attention": [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
        ],
        "ffn": [
            "mlp.up_proj",
            "mlp.down_proj",
            "mlp.gate_proj",
        ],
    }


def rope_decoder_selectors() -> dict[str, list[str]]:
    selectors = dense_rope_decoder_selectors()
    selectors["attention"].extend(
        [
            "linear_attn.in_proj_qkv",
            "linear_attn.out_proj",
        ]
    )
    return selectors


def gpt_oss_selectors() -> dict[str, list[str]]:
    return {
        "attention": [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
        ],
        "ffn": [
            "mlp.router",
            "mlp.experts",
        ],
    }


def seq2seq_selectors() -> dict[str, list[str]]:
    return {
        "attention": [
            "SelfAttention.q",
            "SelfAttention.k",
            "SelfAttention.v",
            "SelfAttention.o",
            "EncDecAttention.q",
            "EncDecAttention.k",
            "EncDecAttention.v",
            "EncDecAttention.o",
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.out_proj",
            "encoder_attn.q_proj",
            "encoder_attn.k_proj",
            "encoder_attn.v_proj",
            "encoder_attn.out_proj",
        ],
        "ffn": [
            "DenseReluDense.wi",
            "DenseReluDense.wi_0",
            "DenseReluDense.wi_1",
            "DenseReluDense.wo",
            "fc1",
            "fc2",
        ],
    }


def phi_selectors() -> dict[str, list[str]]:
    return {
        "attention": [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.dense",
            "self_attn.o_proj",
            "self_attn.qkv_proj",
        ],
        "ffn": [
            "mlp.fc1",
            "mlp.fc2",
            "mlp.gate_up_proj",
            "mlp.down_proj",
        ],
    }


def unknown_selectors() -> dict[str, list[str]]:
    return {
        "attention": ["attention"],
        "ffn": [],
    }
