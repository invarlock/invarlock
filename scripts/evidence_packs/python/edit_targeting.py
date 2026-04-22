from __future__ import annotations

import re

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

_EXCLUDED_PATH_SEGMENTS = frozenset(
    {
        "connector",
        "mm_projector",
        "multi_modal_projector",
        "vision_encoder",
        "vision_model",
        "vision_resampler",
        "vision_tower",
    }
)


def _path_segments(name: str) -> tuple[str, ...]:
    return tuple(
        segment
        for segment in re.split(r"[^a-z0-9_]+", name.lower())
        if segment
    )


def _is_excluded_multimodal_path(name: str) -> bool:
    segments = _path_segments(name)
    return any(segment in _EXCLUDED_PATH_SEGMENTS for segment in segments)


def matches_edit_scope(name: str, scope: str) -> bool:
    if _is_excluded_multimodal_path(name):
        return False
    name_lower = name.lower()
    keywords = _SCOPE_KEYWORDS.get(scope)
    if not keywords:
        return False
    return any(keyword in name_lower for keyword in keywords)
