"""Run-command config and provider resolution helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import typer
import yaml
from rich.console import Console

from invarlock.core.exceptions import ConfigError, ValidationError
from invarlock.core.metric_provider_resolution import resolve_provider_kind_and_kwargs
from invarlock.core.run_policy import coerce_mapping as _coerce_mapping_impl
from invarlock.runtime_security import remote_code_allowed

from .security_helpers import resolve_shell_runtime_security_policy

SPLIT_ALIASES: tuple[str, ...] = ("validation", "val", "dev", "eval", "test")

__all__ = [
    "SPLIT_ALIASES",
    "prepare_config_for_run",
    "resolve_device_and_output",
    "resolve_provider_and_split",
    "remote_code_allowed",
    "_coerce_mapping",
    "_prune_none_values",
    "_to_serialisable_dict",
]


def _resolve_requested_edit_name(kind: str) -> str:
    normalized = kind.lower().strip()
    try:
        from invarlock.edits import get_registry

        registry = get_registry()
        if registry.get_plugin(normalized) is not None:
            return normalized
    except ImportError:
        pass
    known_edits = {"quant_rtn", "noop", "orchestrator"}
    if normalized in known_edits:
        return normalized
    raise ValueError(f"Unknown edit kind: {kind}")


def _coerce_mapping(obj: object) -> dict[str, Any]:
    """Best-effort conversion of config-like objects to plain dicts."""
    return _coerce_mapping_impl(obj)


def _prune_none_values(value: Any) -> Any:
    """Recursively drop keys/items whose value is None."""

    if isinstance(value, dict):
        return {
            key: _prune_none_values(val)
            for key, val in value.items()
            if val is not None
        }
    if isinstance(value, list):
        return [_prune_none_values(item) for item in value if item is not None]
    if isinstance(value, tuple):
        return tuple(_prune_none_values(item) for item in value if item is not None)
    return value


def _to_serialisable_dict(section: object) -> dict[str, Any]:
    """Coerce config fragments to plain dicts."""

    model_dump = getattr(section, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        if isinstance(dumped, dict):
            return _coerce_mapping(dumped)
    as_dict = getattr(section, "dict", None)
    if callable(as_dict):
        try:
            dumped = as_dict()
            if isinstance(dumped, dict):
                return _coerce_mapping(dumped)
        except (TypeError, ValueError):
            pass
    raw = getattr(section, "_data", None)
    if isinstance(raw, dict):
        return raw
    if isinstance(section, dict):
        return section
    try:
        data = vars(section)
        if isinstance(data, dict) and isinstance(data.get("_data"), dict):
            return data["_data"]
        if isinstance(data, dict):
            return _coerce_mapping(_prune_none_values(data))
        return {}
    except TypeError:
        return {}


def _apply_requested_edit_override(
    cfg: Any, edit_name: str, *, config_cls: type
) -> Any:
    cfg_dict = cfg.model_dump()
    edit_section = cfg_dict.setdefault("edit", {})
    if not isinstance(edit_section, dict):
        edit_section = {}
        cfg_dict["edit"] = edit_section
    edit_section["name"] = edit_name
    return config_cls(cfg_dict)


def prepare_config_for_run(
    *,
    config_path: str,
    profile: str | None,
    edit: str | None,
    tier: str | None,
    probes: int | None,
    console: Console | None = None,
    event_fn: Any | None = None,
    invarlock_config_cls: type | None = None,
    load_config_fn: Any | None = None,
    apply_profile_fn: Any | None = None,
    apply_auto_adapter_fn: Any | None = None,
) -> Any:
    """Load config and apply profile/CLI overrides deterministically."""
    shell_mode = console is not None
    if event_fn is None and shell_mode:
        from invarlock.cli.run_shell_output import _event as event_fn
    if invarlock_config_cls is None:
        from invarlock.core.config_runtime import InvarLockConfig

        invarlock_config_cls = InvarLockConfig
    if load_config_fn is None:
        from invarlock.core.config_loader import load_config as load_config_fn
    if apply_profile_fn is None:
        from invarlock.core.config_loader import apply_profile as apply_profile_fn
    if apply_auto_adapter_fn is None:
        try:
            import importlib

            adapter_auto_mod = importlib.import_module("invarlock.adapters.auto")
            apply_auto_adapter_fn = adapter_auto_mod.apply_auto_adapter_if_needed
        except ImportError:  # pragma: no cover - optional adapter path
            apply_auto_adapter_fn = None

    if shell_mode and callable(event_fn):
        event_fn(
            console,
            "INIT",
            f"Loading configuration: {config_path}",
            emoji="📋",
            profile=profile,
        )
    try:
        cfg = load_config_fn(config_path)
    except (ValueError, yaml.YAMLError) as exc:
        if shell_mode and callable(event_fn):
            event_fn(console, "FAIL", str(exc), emoji="❌", profile=profile)
            raise typer.Exit(2) from exc
        raise ConfigError(code="E002", message=str(exc)) from exc

    if profile and str(profile).lower() not in {"dev"}:
        if shell_mode and callable(event_fn):
            event_fn(
                console,
                "INIT",
                f"Applying profile: {profile}",
                emoji="🎯",
                profile=profile,
            )
        try:
            cfg = apply_profile_fn(cfg, profile)
        except (ConfigError, TypeError, ValueError, ValidationError) as exc:
            if shell_mode and callable(event_fn):
                event_fn(console, "FAIL", str(exc), emoji="❌", profile=profile)
                raise typer.Exit(1) from exc
            raise ValidationError(code="E003", message=str(exc)) from exc

    if edit:
        try:
            edit_name = _resolve_requested_edit_name(edit)
            if shell_mode and callable(event_fn):
                event_fn(
                    console,
                    "EXEC",
                    f"Edit override: {edit_name}",
                    emoji="✂️",
                    profile=profile,
                )
            cfg = _apply_requested_edit_override(
                cfg,
                edit_name,
                config_cls=invarlock_config_cls,
            )
        except ValueError as exc:
            if shell_mode and callable(event_fn):
                event_fn(console, "FAIL", str(exc), emoji="❌", profile=profile)
                raise typer.Exit(2) from exc
            raise ConfigError(code="E002", message=str(exc)) from exc

    if tier or probes is not None:
        if tier and tier not in ["conservative", "balanced", "aggressive", "none"]:
            message = f"Invalid tier '{tier}'. Valid options: conservative, balanced, aggressive, none"
            if shell_mode and callable(event_fn):
                event_fn(
                    console,
                    "FAIL",
                    message,
                    emoji="❌",
                    profile=profile,
                )
                raise typer.Exit(1)
            raise ValidationError(code="E003", message=message)
        if probes is not None and (probes < 0 or probes > 10):
            message = f"Invalid probes '{probes}'. Must be between 0 and 10"
            if shell_mode and callable(event_fn):
                event_fn(
                    console,
                    "FAIL",
                    message,
                    emoji="❌",
                    profile=profile,
                )
                raise typer.Exit(1)
            raise ValidationError(code="E003", message=message)

        try:
            cfg_dict = cfg.model_dump()
        except (AttributeError, TypeError, ValueError):
            cfg_dict = {}
        auto_section = (
            cfg_dict.get("auto") if isinstance(cfg_dict.get("auto"), dict) else {}
        )
        cfg_dict["auto"] = auto_section
        if tier:
            auto_section["tier"] = tier
            if shell_mode and callable(event_fn):
                event_fn(
                    console,
                    "INIT",
                    f"Auto tier override: {tier}",
                    emoji="🎛️",
                    profile=profile,
                )
        if probes is not None:
            auto_section["probes"] = probes
            if shell_mode and callable(event_fn):
                event_fn(
                    console,
                    "INIT",
                    f"Auto probes override: {probes}",
                    emoji="🔬",
                    profile=profile,
                )
        cfg = invarlock_config_cls(cfg_dict)

    if apply_auto_adapter_fn is not None:
        cfg = apply_auto_adapter_fn(cfg)

    return cfg


def resolve_device_and_output(
    cfg: Any,
    *,
    device: str | None,
    out: str | None,
    console: Console | None = None,
    event_fn: Any | None = None,
    format_kv_line_fn: Any | None = None,
    device_resolution_note_fn: Any | None = None,
    resolve_device_fn: Any | None = None,
    validate_device_fn: Any | None = None,
) -> tuple[str, Path]:
    """Resolve device and output directory with validation and logging."""
    shell_mode = console is not None
    if event_fn is None and shell_mode:
        from invarlock.cli.run_shell_output import _event as event_fn
    if format_kv_line_fn is None and shell_mode:
        from invarlock.cli.run_shell_output import _format_kv_line as format_kv_line_fn
    if device_resolution_note_fn is None:
        from invarlock.cli.run_shell_output import (
            _device_resolution_note as device_resolution_note_fn,
        )
    if resolve_device_fn is None:
        from invarlock.cli.device import resolve_device as resolve_device_fn
    if validate_device_fn is None:
        from invarlock.cli.device import (
            validate_device_for_config as validate_device_fn,
        )

    try:
        cfg_device = getattr(cfg.model, "device", None)
    except AttributeError:
        cfg_device = None
    target_device = device or cfg_device or "auto"
    resolved_device = resolve_device_fn(target_device)
    resolution_note = device_resolution_note_fn(target_device, resolved_device)
    if shell_mode and format_kv_line_fn is not None:
        console.print(
            format_kv_line_fn("Device", f"{resolved_device} ({resolution_note})")
        )
    is_valid, error_msg = validate_device_fn(resolved_device)
    if not is_valid:
        message = f"Device validation failed: {error_msg}"
        if shell_mode and callable(event_fn):
            event_fn(console, "FAIL", message, emoji="❌")
            raise typer.Exit(1)
        raise ValidationError(code="E003", message=message)

    if out:
        output_dir = Path(out)
    else:
        try:
            output_dir = Path(cfg.output.dir)
        except AttributeError:
            output_dir = Path("runs")
    output_dir.mkdir(parents=True, exist_ok=True)
    return str(resolved_device), output_dir


def resolve_provider_and_split(
    cfg: Any,
    model_profile: Any,
    *,
    get_provider_fn: Any | None = None,
    choose_dataset_split_fn: Any | None = None,
    provider_kwargs: dict[str, Any] | None = None,
    resolved_device: str | None = None,
    emit: Any = None,
) -> tuple[Any, str, bool]:
    """Resolve dataset provider/split and return provider, split, fallback flag."""
    if get_provider_fn is None:
        from invarlock.eval.data import get_provider as get_provider_fn
    if choose_dataset_split_fn is None:
        from invarlock.core.run_policy import (
            choose_dataset_split as choose_dataset_split_fn,
        )

    provider_kwargs = dict(provider_kwargs or {})
    try:
        provider_val = cfg.dataset.provider
    except AttributeError:
        provider_val = None
    provider_name, explicit_provider_kwargs = resolve_provider_kind_and_kwargs(
        provider_val
    )
    provider_kwargs.update(explicit_provider_kwargs)
    if not provider_name:
        provider_name = getattr(model_profile, "default_provider", None) or "wikitext2"

    if resolved_device and provider_name == "wikitext2":
        provider_kwargs.setdefault("device_hint", resolved_device)
    _ = emit
    data_provider = get_provider_fn(provider_name, **provider_kwargs)

    requested_split = None
    try:
        requested_split = getattr(cfg.dataset, "split", None)
    except AttributeError:
        requested_split = None
    available_splits = None
    available_splits_fn = getattr(data_provider, "available_splits", None)
    if callable(available_splits_fn):
        try:
            available_splits = list(available_splits_fn())
        except (AttributeError, TypeError, ValueError):
            available_splits = None
    resolved_split, used_fallback_split = choose_dataset_split_fn(
        requested=requested_split,
        available=available_splits,
        split_aliases=SPLIT_ALIASES,
    )
    return data_provider, resolved_split, used_fallback_split


def extract_model_load_kwargs(
    cfg: Any,
    *,
    invarlock_error_cls: type[BaseException] | None = None,
) -> dict[str, Any]:
    """Return adapter.load_model kwargs from config excluding core fields."""
    if invarlock_error_cls is None:
        from invarlock.core.exceptions import InvarlockError

        invarlock_error_cls = InvarlockError
    try:
        data = cfg.model_dump()
    except (AttributeError, TypeError, ValueError):
        data = {}
    model = data.get("model") if isinstance(data, dict) else None
    if not isinstance(model, dict):
        return {}

    extra = {
        key: value
        for key, value in model.items()
        if key not in {"id", "adapter", "device", "baseline_id", "subject_id"}
        and value is not None
    }

    trust_remote_code = extra.get("trust_remote_code")
    if (
        isinstance(trust_remote_code, bool)
        and trust_remote_code
        and not resolve_shell_runtime_security_policy().allow_remote_code
    ):
        raise invarlock_error_cls(
            code="E008",
            message=(
                "REMOTE-CODE-DISABLED: model.trust_remote_code requires "
                "--allow-remote-code or INVARLOCK_ALLOW_REMOTE_CODE=1."
            ),
            details={"key": "model.trust_remote_code"},
        )

    removed_keys: list[str] = []
    for key in ("torch_dtype", "load_in_8bit", "load_in_4bit"):
        if key in extra:
            removed_keys.append(key)
    if removed_keys:
        raise invarlock_error_cls(
            code="E007",
            message=(
                "CONFIG-KEY-REMOVED: "
                + ", ".join(removed_keys)
                + ". Use model.dtype and/or model.quantization_config."
            ),
            details={"removed_keys": removed_keys},
        )

    if "dtype" in extra and isinstance(extra.get("dtype"), str):
        dtype_str = str(extra.get("dtype") or "").strip().lower()
        removed_dtype_aliases = {
            "fp16": "float16",
            "half": "float16",
            "bf16": "bfloat16",
            "fp32": "float32",
        }
        if dtype_str in removed_dtype_aliases:
            canonical = removed_dtype_aliases[dtype_str]
            raise invarlock_error_cls(
                code="E007",
                message=(
                    "CONFIG-VALUE-REMOVED: "
                    f"model.dtype={dtype_str}. Use model.dtype={canonical}."
                ),
                details={
                    "removed_values": [f"model.dtype={dtype_str}"],
                    "replacement": f"model.dtype={canonical}",
                },
            )
        if dtype_str:
            extra["dtype"] = dtype_str

    return extra
