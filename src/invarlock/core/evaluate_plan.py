from __future__ import annotations

import re
import tempfile
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .assurance_contract import (
    CANONICAL_GUARD_CHAIN,
    normalize_assurance_mode,
    strict_evaluate_policy_errors,
)
from .checkpoint_identity import LEGACY_MODEL_IDENTITY_FIELDS, resolve_model_identity

_TEXT_NORMALIZATION_ERRORS = (RuntimeError, TypeError, ValueError)
_RUNTIME_PROVIDER_NAME = re.compile(r"^[a-z][a-z0-9_]{0,63}$")

DEFAULT_EVALUATE_GUARDS_ORDER = list(CANONICAL_GUARD_CHAIN)


@dataclass(frozen=True)
class EvaluateExecutionPolicy:
    execution_mode: str
    allow_host_execution: bool
    prefer_local_files_only: bool
    allow_unverified_provenance: bool


@dataclass(frozen=True)
class EvaluateCommandPlan:
    profile_name: str
    tier_name: str
    baseline_adapter_name: str
    subject_adapter_name: str
    baseline_runtime_provider_name: str
    subject_runtime_provider_name: str
    adapter_auto: bool
    baseline_adapter_auto: bool
    subject_adapter_auto: bool
    preset_path: Path
    preset_data: dict[str, Any]
    guards_order: list[str]
    source_model_id: str
    subject_model_id: str
    baseline_config: dict[str, Any]
    baseline_identity: dict[str, str] | None
    subject_identity: dict[str, str] | None
    baseline_label: str
    subject_label: str | None
    tmp_dir: Path
    assurance_mode: str


def normalize_model_id(model_id: str, adapter_name: str) -> str:
    """Normalize model identifiers for adapters."""

    mid = str(model_id or "").strip()
    try:
        if str(adapter_name).startswith("hf_") and mid.startswith("hf:"):
            return mid.split(":", 1)[1]
    except _TEXT_NORMALIZATION_ERRORS:
        pass
    return mid


def stable_text(value: object, fallback: str = "") -> str:
    if isinstance(value, str):
        return value
    try:
        return str(value)
    except _TEXT_NORMALIZATION_ERRORS:
        return fallback


def normalize_runtime_provider_name(value: object) -> str:
    """Return a canonical provider name for evaluate planning."""

    normalized = stable_text(value, "hf_transformers").strip()
    if _RUNTIME_PROVIDER_NAME.fullmatch(normalized) is None:
        raise ValueError(
            "Runtime provider names must be lowercase plugin names containing only "
            "letters, digits, and underscores."
        )
    return normalized


def _runtime_provider_block(
    provider_name: str,
    *,
    model_block: dict[str, Any] | None = None,
) -> dict[str, Any]:
    settings: dict[str, Any] = {}
    existing = model_block.get("runtime_provider") if model_block else None
    if (
        isinstance(existing, dict)
        and existing.get("name", provider_name) == provider_name
    ):
        existing_settings = existing.get("settings")
        if isinstance(existing_settings, dict):
            settings = deepcopy(existing_settings)
    return {"name": provider_name, "settings": settings}


def resolve_evaluate_execution_policy(
    *,
    execution_mode: str,
    allow_host_execution: bool,
) -> EvaluateExecutionPolicy:
    normalized_execution_mode = stable_text(execution_mode, "container").strip().lower()
    host_mode = normalized_execution_mode == "host"
    if normalized_execution_mode not in {"container", "host"}:
        raise ValueError("Execution mode must be one of: container, host.")
    return EvaluateExecutionPolicy(
        execution_mode=normalized_execution_mode,
        allow_host_execution=allow_host_execution or host_mode,
        prefer_local_files_only=host_mode,
        allow_unverified_provenance=host_mode,
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
    return preset_path, sanitize_preset_data_for_evaluate(
        load_yaml_fn(preset_path),
        adapter_name=str(adapter_name),
    )


def sanitize_preset_data_for_evaluate(
    preset_data: dict[str, Any],
    *,
    adapter_name: str | None = None,
) -> dict[str, Any]:
    """Remove evaluate-local overrides that should be chosen by the runtime."""

    sanitized = deepcopy(preset_data)
    model_block = sanitized.get("model")
    if isinstance(model_block, dict) and "device" in model_block:
        model_block = dict(model_block)
        model_block.pop("device", None)
        sanitized["model"] = model_block
    if adapter_name:
        default_dataset = default_preset_data_for_adapter(adapter_name)["dataset"]
        dataset_block = sanitized.get("dataset")
        if dataset_block is None:
            sanitized["dataset"] = deepcopy(default_dataset)
        elif isinstance(dataset_block, dict):
            sanitized["dataset"] = deep_merge_dicts(default_dataset, dataset_block)
    return sanitized


def resolve_guards_order(
    preset_data: dict[str, Any],
    *,
    default_order: list[str] | None = None,
    require_canonical: bool = False,
) -> list[str]:
    guards_block = preset_data.get("guards")
    preset_order = guards_block.get("order") if isinstance(guards_block, dict) else None
    if (
        isinstance(preset_order, list)
        and preset_order
        and all(isinstance(item, str) for item in preset_order)
    ):
        if require_canonical and list(preset_order) != DEFAULT_EVALUATE_GUARDS_ORDER:
            raise ValueError("Strict assurance requires the canonical guard chain.")
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


def _reject_legacy_model_identity(config: dict[str, Any], *, label: str) -> None:
    model = config.get("model")
    if not isinstance(model, dict):
        return
    legacy_fields = sorted(
        field for field in LEGACY_MODEL_IDENTITY_FIELDS if field in model
    )
    if legacy_fields:
        raise ValueError(
            f"{label} uses legacy model identity field(s): " + ", ".join(legacy_fields)
        )


def build_baseline_run_config(
    preset_data: dict[str, Any],
    *,
    model_id: str,
    adapter_name: str,
    runtime_provider_name: str = "hf_transformers",
    model_identity: dict[str, str] | None = None,
    output_dir: str,
    profile: str,
    tier: str,
    guards_order: list[str],
    assurance_mode: str = "off",
    execution_mode: str = "unknown",
) -> dict[str, Any]:
    _reject_legacy_model_identity(preset_data, label="Preset config")
    normalized_runtime_provider = normalize_runtime_provider_name(runtime_provider_name)
    preset_model = preset_data.get("model")
    preset_model_block = preset_model if isinstance(preset_model, dict) else None
    model_config: dict[str, Any] = {
        "id": model_id,
        "adapter": adapter_name,
        "runtime_provider": _runtime_provider_block(
            normalized_runtime_provider,
            model_block=preset_model_block,
        ),
    }
    if model_identity is not None:
        model_config["model_identity"] = deepcopy(model_identity)
    return deep_merge_dicts(
        preset_data,
        {
            "model": model_config,
            "edit": {"name": "noop", "plan": {}},
            "eval": {},
            "guards": {"order": guards_order},
            "output": {"dir": output_dir},
            "context": {
                "profile": profile,
                "tier": tier,
                "assurance": {"mode": assurance_mode},
                "runtime": {
                    "execution_mode": stable_text(execution_mode, "unknown")
                    .strip()
                    .lower()
                },
            },
            "assurance": {"mode": assurance_mode},
        },
    )


def build_subject_noop_run_config(
    preset_data: dict[str, Any],
    *,
    model_id: str,
    adapter_name: str,
    runtime_provider_name: str = "hf_transformers",
    model_identity: dict[str, str] | None = None,
    output_dir: str,
    profile: str,
    tier: str,
    guards_order: list[str],
    assurance_mode: str = "off",
    execution_mode: str = "unknown",
) -> dict[str, Any]:
    return build_baseline_run_config(
        preset_data,
        model_id=model_id,
        adapter_name=adapter_name,
        runtime_provider_name=runtime_provider_name,
        model_identity=model_identity,
        output_dir=output_dir,
        profile=profile,
        tier=tier,
        guards_order=guards_order,
        assurance_mode=assurance_mode,
        execution_mode=execution_mode,
    )


def build_subject_edit_run_config(
    preset_data: dict[str, Any],
    loaded_edit_config: dict[str, Any],
    *,
    subject_model_id: str,
    adapter_name: str,
    runtime_provider_name: str = "hf_transformers",
    model_identity: dict[str, str] | None = None,
    output_dir: str,
    profile: str,
    tier: str,
    guards_order: list[str],
    assurance_mode: str = "off",
    execution_mode: str = "unknown",
) -> dict[str, Any]:
    _reject_legacy_model_identity(preset_data, label="Preset config")
    cfg_loaded = deepcopy(loaded_edit_config)
    _reject_legacy_model_identity(cfg_loaded, label="Edit config")
    model_block = dict(cfg_loaded.get("model") or {})
    normalized_runtime_provider = normalize_runtime_provider_name(runtime_provider_name)
    raw_model_id = model_block.get("id")
    if not isinstance(raw_model_id, str) or raw_model_id.startswith("<"):
        model_block["id"] = subject_model_id
    else:
        model_block["id"] = normalize_model_id(str(raw_model_id), adapter_name)
    if not isinstance(model_block.get("adapter"), str) or not model_block.get(
        "adapter"
    ):
        model_block["adapter"] = adapter_name
    model_block["runtime_provider"] = _runtime_provider_block(
        normalized_runtime_provider,
        model_block=model_block,
    )
    model_block.pop("model_identity", None)
    if model_identity is not None:
        model_block["model_identity"] = deepcopy(model_identity)
    cfg_loaded["model"] = model_block

    merged = deep_merge_dicts(
        deep_merge_dicts(preset_data, cfg_loaded),
        {
            "output": {"dir": output_dir},
            "context": {
                "profile": profile,
                "tier": tier,
                "assurance": {"mode": assurance_mode},
                "runtime": {
                    "execution_mode": stable_text(execution_mode, "unknown")
                    .strip()
                    .lower()
                },
            },
            "assurance": {"mode": assurance_mode},
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
    elif (
        assurance_mode == "strict" and guards_order_cfg != DEFAULT_EVALUATE_GUARDS_ORDER
    ):
        raise ValueError("Strict assurance requires the canonical guard chain.")
    return merged


def build_evaluate_command_plan(
    *,
    baseline_model_id: str,
    subject_model_id: str,
    profile: object,
    tier: object,
    preset: str | None,
    out: str,
    edit_config: str | None,
    edit_label: str | None,
    resolve_auto_adapter_fn: Any,
    load_yaml_fn: Any,
    baseline_revision: str | None = None,
    subject_revision: str | None = None,
    baseline_adapter: str = "auto",
    subject_adapter: str = "auto",
    baseline_runtime_provider: str = "hf_transformers",
    subject_runtime_provider: str = "hf_transformers",
    tmp_dir_candidate: str | None = None,
    assurance_mode: str = "strict",
    execution_mode: str = "container",
    allow_unverified_provenance: bool = False,
    evaluation_input_binding: dict[str, object] | None = None,
) -> EvaluateCommandPlan:
    profile_name = stable_text(profile, "dev")
    tier_name = stable_text(tier, "balanced")
    normalized_assurance_mode = normalize_assurance_mode(assurance_mode)
    baseline_runtime_provider_name = normalize_runtime_provider_name(
        baseline_runtime_provider
    )
    subject_runtime_provider_name = normalize_runtime_provider_name(
        subject_runtime_provider
    )

    def _resolve_side_adapter(
        raw_adapter: str,
        model_id: str,
        runtime_provider_name: str,
    ) -> tuple[str, bool]:
        raw_adapter_name = stable_text(raw_adapter, "auto")
        is_auto = raw_adapter_name.strip().lower() in {"auto", "auto_hf"}
        if is_auto:
            if runtime_provider_name == "hf_transformers":
                return str(resolve_auto_adapter_fn(model_id)), True
            return "auto", True
        return raw_adapter_name, False

    baseline_adapter_name, baseline_adapter_auto = _resolve_side_adapter(
        baseline_adapter,
        baseline_model_id,
        baseline_runtime_provider_name,
    )
    subject_adapter_name, subject_adapter_auto = _resolve_side_adapter(
        subject_adapter,
        subject_model_id,
        subject_runtime_provider_name,
    )
    adapter_auto = baseline_adapter_auto or subject_adapter_auto

    preset_path, preset_data = load_evaluate_preset_data(
        adapter_name=str(subject_adapter_name),
        preset=preset,
        load_yaml_fn=load_yaml_fn,
    )
    if evaluation_input_binding is not None:
        preset_data = deepcopy(preset_data)
        context = preset_data.setdefault("context", {})
        if not isinstance(context, dict):
            raise ValueError("Preset context must be an object")
        context["evaluation_inputs"] = deepcopy(evaluation_input_binding)
    guards_order = resolve_guards_order(
        preset_data,
        require_canonical=normalized_assurance_mode == "strict",
    )
    assurance_errors = strict_evaluate_policy_errors(
        assurance_mode=normalized_assurance_mode,
        profile=profile_name,
        tier=tier_name,
        guards_order=guards_order,
        execution_mode=execution_mode,
        allow_unverified_provenance=allow_unverified_provenance,
    )
    if assurance_errors:
        raise ValueError("; ".join(assurance_errors))
    normalized_source_model_id = normalize_model_id(
        baseline_model_id, baseline_adapter_name
    )
    normalized_subject_model_id = normalize_model_id(
        subject_model_id, subject_adapter_name
    )
    baseline_identity = resolve_model_identity(
        normalized_source_model_id,
        revision=baseline_revision,
        strict=normalized_assurance_mode == "strict",
        side="baseline",
    )
    subject_identity = resolve_model_identity(
        normalized_subject_model_id,
        revision=subject_revision,
        strict=normalized_assurance_mode == "strict",
        side="subject",
    )
    baseline_config = build_baseline_run_config(
        preset_data,
        model_id=normalized_source_model_id,
        adapter_name=str(baseline_adapter_name),
        runtime_provider_name=baseline_runtime_provider_name,
        model_identity=baseline_identity,
        output_dir=str(Path(out) / "source"),
        profile=profile_name,
        tier=tier_name,
        guards_order=guards_order,
        assurance_mode=normalized_assurance_mode,
        execution_mode=execution_mode,
    )
    return EvaluateCommandPlan(
        profile_name=profile_name,
        tier_name=tier_name,
        baseline_adapter_name=str(baseline_adapter_name),
        subject_adapter_name=str(subject_adapter_name),
        baseline_runtime_provider_name=baseline_runtime_provider_name,
        subject_runtime_provider_name=subject_runtime_provider_name,
        adapter_auto=adapter_auto,
        baseline_adapter_auto=baseline_adapter_auto,
        subject_adapter_auto=subject_adapter_auto,
        preset_path=preset_path,
        preset_data=preset_data,
        guards_order=guards_order,
        source_model_id=normalized_source_model_id,
        subject_model_id=normalized_subject_model_id,
        baseline_config=baseline_config,
        baseline_identity=baseline_identity,
        subject_identity=subject_identity,
        baseline_label="noop",
        subject_label=determine_subject_label(
            edit_label=edit_label,
            edit_config=edit_config,
            source_model_id=normalized_source_model_id,
            subject_model_id=normalized_subject_model_id,
        ),
        tmp_dir=resolve_evaluate_tmp_dir(tmp_dir_candidate),
        assurance_mode=normalized_assurance_mode,
    )


__all__ = [
    "DEFAULT_EVALUATE_GUARDS_ORDER",
    "EvaluateExecutionPolicy",
    "EvaluateCommandPlan",
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
    "normalize_runtime_provider_name",
    "resolve_evaluate_execution_policy",
    "resolve_evaluate_tmp_dir",
    "resolve_guards_order",
    "sanitize_preset_data_for_evaluate",
    "stable_text",
]
