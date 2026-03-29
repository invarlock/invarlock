from __future__ import annotations

from copy import deepcopy
from typing import Any

DEFAULT_EVALUATE_GUARDS_ORDER = [
    "invariants",
    "spectral",
    "rmt",
    "variance",
    "invariants",
]


def normalize_model_id(model_id: str, adapter_name: str) -> str:
    """Normalize model identifiers for adapters."""

    mid = str(model_id or "").strip()
    try:
        if str(adapter_name).startswith("hf_") and mid.startswith("hf:"):
            return mid.split(":", 1)[1]
    except Exception:
        pass
    return mid


def default_preset_data_for_adapter(adapter_name: str) -> dict[str, Any]:
    seq_len = 128 if adapter_name == "hf_mlm" else 512
    return {
        "dataset": {
            "provider": "wikitext2",
            "split": "validation",
            "seq_len": seq_len,
            "stride": seq_len,
            "preview_n": 64,
            "final_n": 64,
            "seed": 43,
        }
    }


def sanitize_preset_data_for_evaluate(preset_data: dict[str, Any]) -> dict[str, Any]:
    """Remove evaluate-local overrides that should be chosen by the runtime."""

    sanitized = deepcopy(preset_data)
    model_block = sanitized.get("model")
    if isinstance(model_block, dict) and "device" in model_block:
        model_block = dict(model_block)
        model_block.pop("device", None)
        sanitized["model"] = model_block
    return sanitized


def resolve_guards_order(
    preset_data: dict[str, Any],
    *,
    default_order: list[str] | None = None,
) -> list[str]:
    guards_block = preset_data.get("guards")
    preset_order = guards_block.get("order") if isinstance(guards_block, dict) else None
    if (
        isinstance(preset_order, list)
        and preset_order
        and all(isinstance(item, str) for item in preset_order)
    ):
        return list(preset_order)
    return list(default_order or DEFAULT_EVALUATE_GUARDS_ORDER)


def determine_subject_label(
    *,
    edit_label: str | None,
    edit_config: str | None,
    source_model_id: str,
    subject_model_id: str,
) -> str | None:
    if edit_label:
        return edit_label
    if not edit_config:
        return "custom" if source_model_id != subject_model_id else "noop"
    return None


def deep_merge_dicts(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge nested dictionaries without mutating inputs."""

    merged = deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge_dicts(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def build_baseline_run_config(
    preset_data: dict[str, Any],
    *,
    model_id: str,
    adapter_name: str,
    output_dir: str,
    profile: str,
    tier: str,
    guards_order: list[str],
) -> dict[str, Any]:
    return deep_merge_dicts(
        preset_data,
        {
            "model": {
                "id": model_id,
                "adapter": adapter_name,
            },
            "edit": {"name": "noop", "plan": {}},
            "eval": {},
            "guards": {"order": guards_order},
            "output": {"dir": output_dir},
            "context": {"profile": profile, "tier": tier},
        },
    )


def build_subject_noop_run_config(
    preset_data: dict[str, Any],
    *,
    model_id: str,
    adapter_name: str,
    output_dir: str,
    profile: str,
    tier: str,
    guards_order: list[str],
) -> dict[str, Any]:
    return build_baseline_run_config(
        preset_data,
        model_id=model_id,
        adapter_name=adapter_name,
        output_dir=output_dir,
        profile=profile,
        tier=tier,
        guards_order=guards_order,
    )


def build_subject_edit_run_config(
    preset_data: dict[str, Any],
    loaded_edit_config: dict[str, Any],
    *,
    subject_model_id: str,
    adapter_name: str,
    output_dir: str,
    profile: str,
    tier: str,
    guards_order: list[str],
) -> dict[str, Any]:
    cfg_loaded = deepcopy(loaded_edit_config)
    model_block = dict(cfg_loaded.get("model") or {})
    raw_model_id = model_block.get("id")
    if not isinstance(raw_model_id, str) or raw_model_id.startswith("<"):
        model_block["id"] = subject_model_id
    else:
        model_block["id"] = normalize_model_id(str(raw_model_id), adapter_name)
    if not isinstance(model_block.get("adapter"), str) or not model_block.get(
        "adapter"
    ):
        model_block["adapter"] = adapter_name
    cfg_loaded["model"] = model_block

    merged = deep_merge_dicts(
        deep_merge_dicts(preset_data, cfg_loaded),
        {
            "output": {"dir": output_dir},
            "context": {"profile": profile, "tier": tier},
        },
    )
    guards_block = merged.get("guards")
    guards_order_cfg = (
        guards_block.get("order") if isinstance(guards_block, dict) else None
    )
    if not (
        isinstance(guards_order_cfg, list)
        and guards_order_cfg
        and all(isinstance(item, str) for item in guards_order_cfg)
    ):
        merged = deep_merge_dicts(merged, {"guards": {"order": guards_order}})
    return merged


__all__ = [
    "DEFAULT_EVALUATE_GUARDS_ORDER",
    "build_baseline_run_config",
    "build_subject_edit_run_config",
    "build_subject_noop_run_config",
    "deep_merge_dicts",
    "default_preset_data_for_adapter",
    "determine_subject_label",
    "normalize_model_id",
    "resolve_guards_order",
    "sanitize_preset_data_for_evaluate",
]
