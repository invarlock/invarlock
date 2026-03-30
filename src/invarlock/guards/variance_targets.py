from __future__ import annotations

import fnmatch
import hashlib
import re
from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn

from .variance_scaling import iter_transformer_layers, unwrap_model


def normalize_module_name(name: str) -> str:
    """Normalize module names to transformer.h.<idx>.<branch>.c_proj form."""
    if not isinstance(name, str):
        return ""

    normalized = name.strip()
    if not normalized:
        return normalized

    if normalized.startswith("block"):
        parts = normalized.split(".")
        if len(parts) >= 2 and parts[0].startswith("block"):
            layer_idx = parts[0][5:]
            branch = parts[1]
            branch = "attn" if branch.startswith("attn") else "mlp"
            return f"transformer.h.{layer_idx}.{branch}.c_proj"

    if normalized.startswith("transformer.h."):
        if normalized.endswith(".c_proj"):
            return normalized
        if ".mlp" in normalized and ".c_proj" not in normalized:
            return f"{normalized}.c_proj"
        if ".attn" in normalized and ".c_proj" not in normalized:
            return f"{normalized}.c_proj"

    return normalized


def matches_tap(guard: Any, name: str) -> bool:
    """Return True if a module name matches configured tap patterns."""
    normalized = normalize_module_name(name)
    candidates = {normalized, name}
    match = re.match(r"^transformer\.h\.(\d+)\.(attn|mlp)\.c_proj$", normalized)
    if match:
        layer_idx = match.group(1)
        branch = match.group(2)
        prefixes = (
            f"transformer.h.{layer_idx}",
            f"model.layers.{layer_idx}",
            f"model.model.layers.{layer_idx}",
            f"decoder.layers.{layer_idx}",
            f"layers.{layer_idx}",
        )
        suffixes = (
            ("mlp.c_proj", "mlp.down_proj", "mlp.fc2")
            if branch == "mlp"
            else ("attn.c_proj", "attn.out_proj", "attn.o_proj")
        )
        for prefix in prefixes:
            for suffix in suffixes:
                candidates.add(f"{prefix}.{suffix}")

    for pattern in guard._tap_patterns:
        for candidate in candidates:
            if fnmatch.fnmatch(candidate, pattern):
                return True
    return False


def normalize_pairing_ids(prefix: str, window_ids: Sequence[Any]) -> list[str]:
    normalized: list[str] = []
    for idx in window_ids:
        token = str(idx)
        if "::" in token:
            normalized.append(token)
        else:
            normalized.append(f"{prefix}::{token}")
    return normalized


def expected_window_ids(guard: Any) -> list[str]:
    return list(guard._pairing_reference)


def normalize_scale_name(name: str) -> str:
    return normalize_module_name(name)


def scale_matches_target(scale_name: str, target_name: str) -> bool:
    """Check if a scale name from equalise_residual_variance matches a target."""
    normalized_scale = normalize_scale_name(scale_name)
    if normalized_scale == target_name:
        return True
    if scale_name.startswith("block") and ("attn" in scale_name or "mlp" in scale_name):
        parts = scale_name.split(".")
        if len(parts) == 2:
            layer_part = parts[0]
            component = parts[1]
            if layer_part.startswith("block"):
                try:
                    layer_num = layer_part[5:]
                    if f"h.{layer_num}.{component}" in target_name:
                        return True
                except (ValueError, IndexError):
                    pass
    return False


def is_focus_match(guard: Any, name: str) -> bool:
    """Check whether a module name matches the configured focus list."""
    if not guard._focus_modules:
        return True
    return normalize_module_name(name) in guard._focus_modules


def fingerprint_targets(guard: Any) -> str | None:
    """Compute a lightweight fingerprint of targeted module weights."""
    if not guard._target_modules:
        return None

    hasher = hashlib.sha256()
    try:
        for name in sorted(guard._target_modules.keys()):
            module = guard._target_modules[name]
            state = getattr(module, "state_dict", None)
            if not callable(state):
                continue
            module_state = state()
            for key in sorted(module_state.keys()):
                tensor = module_state[key]
                if hasattr(tensor, "detach"):
                    data = tensor.detach().cpu().numpy().tobytes()
                else:
                    data = bytes(str(tensor), "utf-8")
                hasher.update(name.encode("utf-8"))
                hasher.update(key.encode("utf-8"))
                hasher.update(data)
        return hasher.hexdigest()[:16]
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return None


def record_ab_provenance(
    guard: Any,
    condition: str,
    *,
    tag: str,
    window_ids: Sequence[str],
    fingerprint: str | None,
    mode: str,
    status: str,
) -> None:
    """Record provenance metadata for A/B evaluation conditions."""
    provenance = guard._stats.setdefault("ab_provenance", {})
    window_list = list(window_ids)
    provenance[condition] = {
        "tag": tag,
        "mode": mode,
        "window_ids": window_list,
        "window_count": len(window_list),
        "target_fingerprint": fingerprint,
        "status": status,
        "pairing_digest": guard._pairing_digest,
        "dataset_hash": (guard._dataset_meta or {}).get("dataset_hash"),
        "tokenizer_hash": (guard._dataset_meta or {}).get("tokenizer_hash"),
        "model_id": (guard._report_meta or {}).get("model_id"),
        "run_seed": (guard._report_meta or {}).get("seed"),
    }


def resolve_target_modules(
    guard: Any, model: nn.Module, adapter: Any | None = None
) -> dict[str, nn.Module]:
    """Resolve target modules based on scope policy."""
    targets: dict[str, nn.Module] = {}
    scope = guard._policy["scope"]
    audit_candidates: list[dict[str, Any]] = []
    audit_rejections: list[dict[str, Any]] = []

    def record_match(name: str, module: nn.Module) -> None:
        audit_candidates.append(
            {"name": name, "class": module.__class__.__name__, "source": "direct"}
        )

    def record_rejection(name: str, reason: str, module: Any | None) -> None:
        audit_rejections.append(
            {
                "name": name,
                "reason": reason,
                "class": getattr(module, "__class__", type(None)).__name__
                if module is not None
                else None,
            }
        )

    try:
        from transformers.pytorch_utils import Conv1D

        module_types = (nn.Linear, nn.Conv1d, Conv1D)
    except ImportError:
        module_types = (nn.Linear, nn.Conv1d)

    def is_supported_module(module: Any) -> bool:
        experts = getattr(module, "experts", None)
        if experts is not None:
            for attr in ("w2", "down_proj", "c_proj", "fc2"):
                proj = getattr(experts, attr, None)
                if proj is None:
                    continue
                weight = getattr(proj, "weight", None)
                candidate = weight if isinstance(weight, torch.Tensor) else None
                if candidate is None and isinstance(proj, torch.Tensor):
                    candidate = proj
                if candidate is None:
                    continue
                try:
                    dim = candidate.dim()
                except (AttributeError, RuntimeError, TypeError, ValueError):
                    dim = getattr(candidate, "ndim", None)
                if dim in (2, 3):
                    return True

            if isinstance(experts, nn.Module) and hasattr(experts, "_modules"):
                iterable = experts._modules.values()  # type: ignore[attr-defined]
            else:
                try:
                    iterable = list(experts)
                except TypeError:
                    iterable = []
            for expert in iterable:
                for attr in ("w2", "down_proj", "c_proj", "fc2"):
                    proj = getattr(expert, attr, None)
                    weight = getattr(proj, "weight", None) if proj is not None else None
                    if weight is None:
                        continue
                    try:
                        dim = weight.dim()
                    except (AttributeError, RuntimeError, TypeError, ValueError):
                        dim = getattr(weight, "ndim", None)
                    if dim in (2, 3):
                        return True

        if isinstance(module, module_types):
            return True
        class_name = module.__class__.__name__ if module is not None else ""
        if class_name in {"Conv1D", "Linear"}:
            return True
        weight = getattr(module, "weight", None)
        if weight is None:
            return False
        try:
            dim = weight.dim()
        except (AttributeError, RuntimeError, TypeError, ValueError):
            dim = getattr(weight, "ndim", None)
        return dim == 2

    for index, block in enumerate(iter_transformer_layers(model)):
        if scope in ["attn", "both"] and hasattr(block, "attn"):
            attn_proj = getattr(block.attn, "c_proj", None) or getattr(
                block.attn, "out_proj", None
            )
            name = f"transformer.h.{index}.attn.c_proj"
            if attn_proj is None:
                record_rejection(name, "missing_module", None)
            elif not matches_tap(guard, name):
                record_rejection(name, "tap_mismatch", attn_proj)
            elif not is_supported_module(attn_proj):
                record_rejection(name, "unsupported_type", attn_proj)
            else:
                targets[name] = attn_proj
                record_match(name, attn_proj)

        if scope in ["ffn", "both"]:
            mlp_container = None
            if hasattr(block, "mlp"):
                mlp_container = block.mlp  # type: ignore[attr-defined]
            elif hasattr(block, "block_sparse_moe"):
                mlp_container = block.block_sparse_moe  # type: ignore[attr-defined]
            if mlp_container is None:
                continue

            mlp_proj = (
                getattr(mlp_container, "c_proj", None)
                or getattr(mlp_container, "down_proj", None)
                or getattr(mlp_container, "fc2", None)
            )
            name = f"transformer.h.{index}.mlp.c_proj"
            if mlp_proj is None:
                if is_supported_module(mlp_container):
                    if not matches_tap(guard, name):
                        record_rejection(name, "tap_mismatch", mlp_container)
                    else:
                        targets[name] = mlp_container
                        record_match(name, mlp_container)
                else:
                    record_rejection(name, "missing_module", None)
            elif not matches_tap(guard, name):
                record_rejection(name, "tap_mismatch", mlp_proj)
            elif not is_supported_module(mlp_proj):
                record_rejection(name, "unsupported_type", mlp_proj)
            else:
                targets[name] = mlp_proj
                record_match(name, mlp_proj)

    fallback_used = False
    if not targets and adapter is not None and hasattr(adapter, "get_layer_modules"):
        try:
            n_layers = 0
            if hasattr(adapter, "describe"):
                try:
                    desc = adapter.describe(model)
                    if isinstance(desc, dict):
                        n_layers = int(desc.get("n_layer", 0) or 0)
                except (AttributeError, RuntimeError, TypeError, ValueError) as desc_exc:
                    guard._log_event(
                        "adapter_describe_error",
                        level="DEBUG",
                        message=f"adapter.describe() failed: {desc_exc}",
                    )
            if n_layers == 0:
                try:
                    n_layers = sum(1 for _ in iter_transformer_layers(model))
                except (AttributeError, RuntimeError, TypeError, ValueError):
                    pass
            if n_layers == 0:
                config = getattr(unwrap_model(model), "config", None)
                if config is not None:
                    n_layers = (
                        getattr(config, "n_layer", 0)
                        or getattr(config, "num_hidden_layers", 0)
                        or getattr(config, "num_layers", 0)
                        or 0
                    )
            if n_layers == 0:
                guard._log_event(
                    "adapter_fallback_no_layers",
                    level="WARN",
                    message="Adapter fallback: could not determine layer count",
                )

            for index in range(n_layers):
                try:
                    modules = adapter.get_layer_modules(model, index) or {}
                except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
                    record_rejection(
                        f"transformer.h.{index}", f"adapter_error:{exc}", None
                    )
                    continue

                for key, module in modules.items():
                    if not isinstance(key, str) or not key.endswith("c_proj"):
                        continue
                    branch = "attn" if "attn" in key else "mlp"
                    name = f"transformer.h.{index}.{branch}.c_proj"
                    if not matches_tap(guard, name):
                        record_rejection(name, "tap_mismatch", module)
                        continue
                    if not is_supported_module(module):
                        record_rejection(name, "unsupported_type", module)
                        continue
                    targets[name] = module
                    audit_candidates.append(
                        {
                            "name": name,
                            "class": module.__class__.__name__,
                            "source": "adapter_fallback",
                        }
                    )
            if targets:
                fallback_used = True
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            guard._log_event(
                "target_resolution_fallback_error",
                level="WARN",
                message="Adapter fallback failed during VE target resolution",
                error=str(exc),
            )

    if guard._focus_modules:
        focused: dict[str, nn.Module] = {}
        for name, module in targets.items():
            if normalize_module_name(name) in guard._focus_modules:
                focused[name] = module
        if not focused:
            guard._log_event(
                "focus_miss",
                level="WARN",
                message="No target modules matched focus list",
                focus_modules=sorted(guard._focus_modules),
                available=list(targets.keys()),
            )
        else:
            targets = focused

    rejected_summary: dict[str, Any] = {}
    for item in audit_rejections:
        reason = item["reason"]
        bucket = rejected_summary.setdefault(reason, {"count": 0, "examples": []})
        bucket["count"] += 1
        if len(bucket["examples"]) < 5:
            bucket["examples"].append({"name": item["name"], "class": item["class"]})

    guard._stats["target_resolution"] = {
        "scope": scope,
        "tap": list(guard._tap_patterns),
        "total_matched": len(targets),
        "matched": sorted(targets.keys()),
        "fallback_used": fallback_used,
        "candidates_recorded": len(audit_candidates),
        "rejected": rejected_summary,
    }
    guard._log_event(
        "target_resolution",
        message="Resolved variance guard targets",
        scope=scope,
        tap=list(guard._tap_patterns),
        matched=len(targets),
        rejected=sum(item["count"] for item in rejected_summary.values())
        if rejected_summary
        else 0,
        fallback_used=fallback_used,
    )
    return targets


__all__ = [
    "expected_window_ids",
    "fingerprint_targets",
    "is_focus_match",
    "matches_tap",
    "normalize_module_name",
    "normalize_pairing_ids",
    "normalize_scale_name",
    "record_ab_provenance",
    "resolve_target_modules",
    "scale_matches_target",
]
