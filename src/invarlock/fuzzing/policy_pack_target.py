from __future__ import annotations

import json

import yaml

from invarlock.policy_pack import _load_structured_text, verify_policy_pack

_STRUCTURED_SUFFIXES = (".json", ".yaml", ".yml")


def _choose_suffix(data: bytes) -> str:
    if not data:
        return ".json"
    return _STRUCTURED_SUFFIXES[data[0] % len(_STRUCTURED_SUFFIXES)]


def exercise_policy_pack_bytes(data: bytes) -> None:
    text = data.decode("utf-8", errors="ignore")
    try:
        payload = _load_structured_text(text, suffix=_choose_suffix(data))
    except (
        json.JSONDecodeError,
        RecursionError,
        TypeError,
        ValueError,
        yaml.YAMLError,
    ):
        return

    verify_policy_pack(payload)
