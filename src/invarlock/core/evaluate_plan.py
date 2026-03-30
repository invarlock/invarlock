from __future__ import annotations

import tempfile
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_EVALUATE_GUARDS_ORDER = [
    "invariants",
    "spectral",
    "rmt",
    "variance",
    "invariants",
]


@dataclass(frozen=True)
class EvaluateExecutionPolicy:
    mode: str
    allow_host_execution: bool
    prefer_local_files_only: bool


@dataclass(frozen=True)
class EvaluateCommandPlan:
    profile_name: str
    tier_name: str
    adapter_name: str
    adapter_auto: bool
    preset_path: Path
    preset_data: dict[str, Any]
    guards_order: list[str]
    source_model_id: str
    subject_model_id: str
    baseline_config: dict[str, Any]
    baseline_label: str
    subject_label: str | None
    tmp_dir: Path


def normalize_model_id(model_id: str, adapter_name: str) -> str:
    """Normalize model identifiers for adapters."""

    mid = str(model_id or "").strip()
    try:
        if str(adapter_name).startswith("hf_") and mid.startswith("hf:"):
            return mid.split(":", 1)[1]
    except Exception:
        pass
    return mid


def stable_text(value: object, fallback: str = "") -> str:
    if isinstance(value, str):
        return value
    try:
        return str(value)
    except Exception:
        return fallback


def resolve_evaluate_execution_policy(
    *,
    mode: str,
    allow_host_execution: bool,
) -> EvaluateExecutionPolicy:
    normalized_mode = stable_text(mode, "attested").strip().lower()
    if normalized_mode not in {"attested", "local"}:
        raise ValueError("Execution mode must be one of: attested, local.")
    return EvaluateExecutionPolicy(
        mode=normalized_mode,
        allow_host_execution=allow_host_execution or normalized_mode == "local",
        prefer_local_files_only=normalized_mode == "local",
    )


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


def default_evaluate_preset_path(adapter_name: str) -> Path:
    return (
        Path("configs/presets/masked_lm/wikitext2_128.yaml")
        if adapter_name == "hf_mlm"
        else Path("configs/presets/causal_lm/wikitext2_512.yaml")
    )


def load_evaluate_preset_data(
    *,
    adapter_name: str,
    preset: str | None,
    load_yaml_fn: Any,
) -> tuple[Path, dict[str, Any]]:
    preset_path = (
        Path(preset)
        if preset is not None
        else default_evaluate_preset_path(adapter_name)
    )
    if preset is None and not preset_path.exists():
        return preset_path, default_preset_data_for_adapter(str(adapter_name))
    if not preset_path.exists():
        raise FileNotFoundError(str(preset_path))
    return preset_path, sanitize_preset_data_for_evaluate(load_yaml_fn(preset_path))


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


def resolve_evaluate_tmp_dir(
    candidate: str | None,
    *,
    scratch_root: Path | None = None,
) -> Path:
    if candidate:
        tmp_dir = Path(candidate).expanduser()
    else:
        root = scratch_root or (Path("tmp") / ".evaluate")
        root.mkdir(parents=True, exist_ok=True)
        tmp_dir = Path(tempfile.mkdtemp(prefix="run-", dir=str(root))).resolve()
    tmp_dir.mkdir(parents=True, exist_ok=True)
    return tmp_dir


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


def build_evaluate_command_plan(
    *,
    baseline_model_id: str,
    subject_model_id: str,
    adapter: str,
    profile: object,
    tier: object,
    preset: str | None,
    out: str,
    edit_config: str | None,
    edit_label: str | None,
    resolve_auto_adapter_fn: Any,
    load_yaml_fn: Any,
    tmp_dir_candidate: str | None = None,
) -> EvaluateCommandPlan:
    profile_name = stable_text(profile, "dev")
    tier_name = stable_text(tier, "balanced")
    adapter_name = adapter
    adapter_auto = str(adapter).strip().lower() in {"auto", "auto_hf"}
    if adapter_auto:
        adapter_name = resolve_auto_adapter_fn(baseline_model_id)

    preset_path, preset_data = load_evaluate_preset_data(
        adapter_name=str(adapter_name),
        preset=preset,
        load_yaml_fn=load_yaml_fn,
    )
    guards_order = resolve_guards_order(preset_data)
    normalized_source_model_id = normalize_model_id(baseline_model_id, adapter_name)
    normalized_subject_model_id = normalize_model_id(subject_model_id, adapter_name)
    baseline_config = build_baseline_run_config(
        preset_data,
        model_id=normalized_source_model_id,
        adapter_name=str(adapter_name),
        output_dir=str(Path(out) / "source"),
        profile=profile_name,
        tier=tier_name,
        guards_order=guards_order,
    )
    return EvaluateCommandPlan(
        profile_name=profile_name,
        tier_name=tier_name,
        adapter_name=str(adapter_name),
        adapter_auto=adapter_auto,
        preset_path=preset_path,
        preset_data=preset_data,
        guards_order=guards_order,
        source_model_id=normalized_source_model_id,
        subject_model_id=normalized_subject_model_id,
        baseline_config=baseline_config,
        baseline_label="noop",
        subject_label=determine_subject_label(
            edit_label=edit_label,
            edit_config=edit_config,
            source_model_id=normalized_source_model_id,
            subject_model_id=normalized_subject_model_id,
        ),
        tmp_dir=resolve_evaluate_tmp_dir(tmp_dir_candidate),
    )


__all__ = [
    "DEFAULT_EVALUATE_GUARDS_ORDER",
    "EvaluateCommandPlan",
    "EvaluateExecutionPolicy",
    "build_baseline_run_config",
    "build_evaluate_command_plan",
    "build_subject_edit_run_config",
    "build_subject_noop_run_config",
    "default_evaluate_preset_path",
    "deep_merge_dicts",
    "default_preset_data_for_adapter",
    "determine_subject_label",
    "load_evaluate_preset_data",
    "normalize_model_id",
    "resolve_evaluate_execution_policy",
    "resolve_evaluate_tmp_dir",
    "resolve_guards_order",
    "sanitize_preset_data_for_evaluate",
    "stable_text",
]
