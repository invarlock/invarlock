"""Run-command config and provider resolution helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import typer
from rich.console import Console

from invarlock.runtime_security import remote_code_allowed


def prepare_config_for_run(
    *,
    config_path: str,
    profile: str | None,
    edit: str | None,
    tier: str | None,
    probes: int | None,
    console: Console,
    event_fn: Any,
    invarlock_config_cls: type,
    load_config_fn: Any,
    apply_profile_fn: Any,
    resolve_edit_kind_fn: Any,
    apply_edit_override_fn: Any,
    apply_auto_adapter_fn: Any | None = None,
) -> Any:
    """Load config and apply profile/CLI overrides deterministically."""
    event_fn(
        console,
        "INIT",
        f"Loading configuration: {config_path}",
        emoji="📋",
        profile=profile,
    )
    try:
        cfg = load_config_fn(config_path)
    except ValueError as exc:
        event_fn(console, "FAIL", str(exc), emoji="❌", profile=profile)
        raise typer.Exit(2) from exc

    if profile and str(profile).lower() not in {"dev"}:
        event_fn(
            console,
            "INIT",
            f"Applying profile: {profile}",
            emoji="🎯",
            profile=profile,
        )
        try:
            cfg = apply_profile_fn(cfg, profile)
        except Exception as exc:
            event_fn(console, "FAIL", str(exc), emoji="❌", profile=profile)
            raise typer.Exit(1) from exc

    if edit:
        try:
            edit_name = resolve_edit_kind_fn(edit)
            event_fn(
                console,
                "EXEC",
                f"Edit override: {edit_name}",
                emoji="✂️",
                profile=profile,
            )
            cfg = apply_edit_override_fn(cfg, edit)
        except ValueError as exc:
            event_fn(console, "FAIL", str(exc), emoji="❌", profile=profile)
            raise typer.Exit(2) from exc

    if tier or probes is not None:
        if tier and tier not in ["conservative", "balanced", "aggressive", "none"]:
            event_fn(
                console,
                "FAIL",
                f"Invalid tier '{tier}'. Valid options: conservative, balanced, aggressive, none",
                emoji="❌",
                profile=profile,
            )
            raise typer.Exit(1)
        if probes is not None and (probes < 0 or probes > 10):
            event_fn(
                console,
                "FAIL",
                f"Invalid probes '{probes}'. Must be between 0 and 10",
                emoji="❌",
                profile=profile,
            )
            raise typer.Exit(1)

        try:
            cfg_dict = cfg.model_dump()
        except Exception:
            cfg_dict = {}
        auto_section = (
            cfg_dict.get("auto") if isinstance(cfg_dict.get("auto"), dict) else {}
        )
        cfg_dict["auto"] = auto_section
        if tier:
            auto_section["tier"] = tier
            event_fn(
                console,
                "INIT",
                f"Auto tier override: {tier}",
                emoji="🎛️",
                profile=profile,
            )
        if probes is not None:
            auto_section["probes"] = probes
            event_fn(
                console,
                "INIT",
                f"Auto probes override: {probes}",
                emoji="🔬",
                profile=profile,
            )
        cfg = invarlock_config_cls(cfg_dict)

    if apply_auto_adapter_fn is not None:
        try:
            cfg = apply_auto_adapter_fn(cfg)
        except Exception:
            pass

    return cfg


def resolve_device_and_output(
    cfg: Any,
    *,
    device: str | None,
    out: str | None,
    console: Console,
    event_fn: Any,
    format_kv_line_fn: Any,
    device_resolution_note_fn: Any,
    resolve_device_fn: Any,
    validate_device_fn: Any,
) -> tuple[str, Path]:
    """Resolve device and output directory with validation and logging."""
    try:
        cfg_device = getattr(cfg.model, "device", None)
    except Exception:
        cfg_device = None
    target_device = device or cfg_device or "auto"
    resolved_device = resolve_device_fn(target_device)
    resolution_note = device_resolution_note_fn(target_device, resolved_device)
    console.print(format_kv_line_fn("Device", f"{resolved_device} ({resolution_note})"))
    is_valid, error_msg = validate_device_fn(resolved_device)
    if not is_valid:
        event_fn(console, "FAIL", f"Device validation failed: {error_msg}", emoji="❌")
        raise typer.Exit(1)

    if out:
        output_dir = Path(out)
    else:
        try:
            output_dir = Path(cfg.output.dir)
        except Exception:
            output_dir = Path("runs")
    output_dir.mkdir(parents=True, exist_ok=True)
    return str(resolved_device), output_dir


def resolve_provider_and_split(
    cfg: Any,
    model_profile: Any,
    *,
    get_provider_fn: Any,
    choose_dataset_split_fn: Any,
    provider_kwargs: dict[str, Any] | None = None,
    resolved_device: str | None = None,
    emit: Any = None,
) -> tuple[Any, str, bool]:
    """Resolve dataset provider/split and return provider, split, fallback flag."""
    provider_name = None
    provider_kwargs = dict(provider_kwargs or {})
    try:
        provider_val = cfg.dataset.provider
    except Exception:
        provider_val = None
    if isinstance(provider_val, str) and provider_val:
        provider_name = provider_val
    else:
        try:
            provider_name = provider_val.kind  # type: ignore[attr-defined]
            try:
                for key, value in provider_val.items():  # type: ignore[attr-defined]
                    if key != "kind" and value is not None and value != "":
                        provider_kwargs[key] = value
            except Exception:
                pass
        except Exception:
            provider_name = None
    if not provider_name:
        provider_name = getattr(model_profile, "default_provider", None) or "wikitext2"

    if resolved_device and provider_name == "wikitext2":
        provider_kwargs.setdefault("device_hint", resolved_device)
    if emit is not None and provider_name == "wikitext2":
        data_provider = get_provider_fn(provider_name, emit=emit, **provider_kwargs)
    else:
        data_provider = get_provider_fn(provider_name, **provider_kwargs)

    requested_split = None
    try:
        requested_split = getattr(cfg.dataset, "split", None)
    except Exception:
        requested_split = None
    available_splits = None
    if hasattr(data_provider, "available_splits"):
        try:
            available_splits = list(data_provider.available_splits())  # type: ignore[attr-defined]
        except Exception:
            available_splits = None
    resolved_split, used_fallback_split = choose_dataset_split_fn(
        requested=requested_split,
        available=available_splits,
    )
    return data_provider, resolved_split, used_fallback_split


def extract_model_load_kwargs(
    cfg: Any,
    *,
    invarlock_error_cls: type[BaseException],
) -> dict[str, Any]:
    """Return adapter.load_model kwargs from config excluding core fields."""
    try:
        data = cfg.model_dump()
    except Exception:
        data = {}
    model = data.get("model") if isinstance(data, dict) else None
    if not isinstance(model, dict):
        return {}

    extra = {
        key: value
        for key, value in model.items()
        if key not in {"id", "adapter", "device"} and value is not None
    }

    trust_remote_code = extra.get("trust_remote_code")
    if (
        isinstance(trust_remote_code, bool)
        and trust_remote_code
        and not remote_code_allowed()
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
        aliases = {
            "fp16": "float16",
            "half": "float16",
            "bf16": "bfloat16",
            "fp32": "float32",
        }
        if dtype_str in aliases:
            extra["dtype"] = aliases[dtype_str]
        elif dtype_str:
            extra["dtype"] = dtype_str

    return extra
