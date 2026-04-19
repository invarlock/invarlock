from __future__ import annotations

_SCOPE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "all": (
        "linear",
        "dense",
        "proj",
        "fc",
        "mlp",
        "attn",
        "wqkv",
        "query_key_value",
    ),
    "ffn": (
        "mlp",
        "fc",
        "dense",
        "gate",
        "up_proj",
        "down_proj",
        "dense_h_to_4h",
        "dense_4h_to_h",
    ),
    "attn": (
        "attn",
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "wqkv",
        "out_proj",
        "query_key_value",
    ),
}


def matches_edit_scope(name: str, scope: str) -> bool:
    name_lower = name.lower()
    keywords = _SCOPE_KEYWORDS.get(scope)
    if not keywords:
        return False
    return any(keyword in name_lower for keyword in keywords)
