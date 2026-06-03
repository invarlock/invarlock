from __future__ import annotations

import os
import random
import re
from pathlib import Path

import torch

try:
    from .basic_injections import (
        _break_weight_tying,
        _inject_extreme_quant,
        _inject_inf_injection,
        _inject_missing_tensors,
        _inject_nan_injection,
        _inject_norm_collapse,
        _inject_rank_collapse,
        _inject_scale_explosion,
    )
    from .common import (
        _default_last_layers,
        _get_norm_weight,
        _is_norm_module,
        _parse_layer_indices,
        _select_row_indices,
    )
    from .probe_injections_spectral_variance import (
        _inject_spectral_moderate_scale,
        _inject_ve_mlp_scale_skew,
    )
except ImportError:  # pragma: no cover - direct script-path loading in tests
    from error_model.basic_injections import (
        _break_weight_tying,
        _inject_extreme_quant,
        _inject_inf_injection,
        _inject_missing_tensors,
        _inject_nan_injection,
        _inject_norm_collapse,
        _inject_rank_collapse,
        _inject_scale_explosion,
    )
    from error_model.common import (
        _default_last_layers,
        _get_norm_weight,
        _is_norm_module,
        _parse_layer_indices,
        _select_row_indices,
    )
    from error_model.probe_injections_spectral_variance import (
        _inject_spectral_moderate_scale,
        _inject_ve_mlp_scale_skew,
    )


def _inject_rmt_norm_noise(
    *, model: torch.nn.Module, error_info: dict[str, object]
) -> None:
    noise_scale = float(os.environ.get("INVARLOCK_RMT_NORM_NOISE_SCALE", "0.05"))
    target_layers_spec = os.environ.get("INVARLOCK_RMT_NORM_TARGET_LAYERS", "")
    max_modules = int(os.environ.get("INVARLOCK_RMT_NORM_MAX_MODULES", "32"))
    seed = int(os.environ.get("INVARLOCK_RMT_NORM_SEED", "42"))
    include_global = os.environ.get("INVARLOCK_RMT_NORM_INCLUDE_GLOBAL", "0") == "1"
    mult_clamp = float(os.environ.get("INVARLOCK_RMT_NORM_MULT_CLAMP", "0.5"))

    random.seed(seed)
    torch.manual_seed(seed)

    layer_pattern = re.compile(r"(?:layers|blocks|h)\.(\d+)\.")
    norm_modules_by_layer: dict[int, list[tuple[str, torch.nn.Module]]] = {}
    global_norm_modules: list[tuple[str, torch.nn.Module]] = []

    for name, module in model.named_modules():
        if not _is_norm_module(module):
            continue
        weight = _get_norm_weight(module)
        if weight is None:
            continue
        match = layer_pattern.search(name)
        if match:
            layer_idx = int(match.group(1))
            norm_modules_by_layer.setdefault(layer_idx, []).append((name, module))
        else:
            global_norm_modules.append((name, module))

    all_layer_indices = sorted(norm_modules_by_layer.keys())
    max_layer = max(all_layer_indices) + 1 if all_layer_indices else 0
    if target_layers_spec:
        target_layers = _parse_layer_indices(target_layers_spec, max_layer)
    else:
        target_layers = _default_last_layers(all_layer_indices, 4)

    target_modules: list[tuple[str, torch.nn.Module]] = []
    for layer_idx in target_layers:
        if layer_idx in norm_modules_by_layer:
            target_modules.extend(norm_modules_by_layer[layer_idx])
    if include_global:
        target_modules.extend(global_norm_modules)

    random.shuffle(target_modules)
    if len(target_modules) > max_modules:
        target_modules = target_modules[:max_modules]

    modified_count = 0
    modified_names: list[str] = []
    for name, module in target_modules:
        weight = _get_norm_weight(module)
        if weight is None or not weight.is_floating_point():
            continue
        with torch.no_grad():
            base = weight.detach().clone()
            noise = torch.randn(base.shape, device=base.device, dtype=torch.float32)
            noise *= float(noise_scale)
            multiplier = (1.0 + noise).clamp(1.0 - mult_clamp, 1.0 + mult_clamp)
            weight.mul_(multiplier.to(dtype=base.dtype))
            if not torch.isfinite(weight).all():
                weight.copy_(base)
                continue
        modified_count += 1
        modified_names.append(name)
        if modified_count <= 5:
            print(f"Perturbed norm: {name} (scale={noise_scale})")

    if modified_count > 5:
        print(f"  ... and {modified_count - 5} more norm modules")

    if modified_count > 0:
        error_info.update(
            {
                "injected": True,
                "mode": "norm_noise",
                "noise_scale": noise_scale,
                "seed": seed,
                "target_layers": target_layers,
                "max_modules": max_modules,
                "include_global": include_global,
                "mult_clamp": mult_clamp,
                "modified_count": modified_count,
                "modified_modules": modified_names[:20],
                "total_layers": max_layer,
            }
        )
        print(
            f"Applied RMT norm noise to {modified_count} modules "
            f"(scale={noise_scale}, seed={seed})"
        )
    else:
        print("WARNING: rmt_norm_noise not injected (no norm modules found)")


def _inject_rmt_row_scale(
    *, model: torch.nn.Module, error_info: dict[str, object]
) -> None:
    scale_factor = float(os.environ.get("INVARLOCK_RMT_ROW_SCALE_FACTOR", "1.6"))
    scale_factor = min(max(scale_factor, 0.1), 10.0)
    row_frac = float(os.environ.get("INVARLOCK_RMT_ROW_SCALE_ROW_FRAC", "0.10"))
    row_frac = min(max(row_frac, 0.01), 0.95)
    max_rows = int(os.environ.get("INVARLOCK_RMT_ROW_SCALE_MAX_ROWS", "2048"))
    max_params = int(os.environ.get("INVARLOCK_RMT_ROW_SCALE_MAX_PARAMS", "8"))
    target_family = (
        os.environ.get("INVARLOCK_RMT_ROW_SCALE_TARGET_FAMILY", "ffn").strip().lower()
    )
    target_param_substrings_spec = os.environ.get(
        "INVARLOCK_RMT_ROW_SCALE_TARGET_PARAMS", ""
    ).strip()
    target_param_substrings = [
        part.strip().lower()
        for part in target_param_substrings_spec.split(",")
        if part.strip()
    ]
    target_layers_spec = os.environ.get("INVARLOCK_RMT_ROW_SCALE_TARGET_LAYERS", "")
    include_qkv = os.environ.get("INVARLOCK_RMT_ROW_SCALE_INCLUDE_QKV", "0") == "1"
    row_selection = os.environ.get("INVARLOCK_RMT_ROW_SCALE_ROW_SELECTION", "first")
    seed = int(os.environ.get("INVARLOCK_RMT_ROW_SCALE_SEED", "42"))

    random.seed(seed)
    torch.manual_seed(seed)

    layer_pattern = re.compile(r"(?:layers|blocks|h)\.(\d+)\.")
    attn_patterns = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "c_attn",
        "c_proj",
        "out_proj",
        "query_key_value",
        "self_attn",
        "attention",
    )
    ffn_patterns = (
        "mlp",
        "ffn",
        "feed_forward",
        "gate_proj",
        "up_proj",
        "down_proj",
        "fc1",
        "fc2",
        "c_fc",
        "experts",
    )
    embed_patterns = ("embed", "wte", "wpe")

    def is_target_family(name: str) -> bool:
        lname = name.lower()
        if target_family == "attn":
            return any(p in lname for p in attn_patterns)
        if target_family == "ffn":
            return any(p in lname for p in ffn_patterns)
        if target_family == "embed":
            return any(p in lname for p in embed_patterns)
        if target_family == "all":
            return any(
                p in lname for p in attn_patterns + ffn_patterns + embed_patterns
            )
        return any(p in lname for p in attn_patterns + ffn_patterns)

    def is_qkv_param(name: str) -> bool:
        lname = name.lower()
        return any(
            p in lname
            for p in ("q_proj", "k_proj", "v_proj", "c_attn", "query_key_value")
        )

    def matches_param_filter(name: str) -> bool:
        if not target_param_substrings:
            return True
        lname = name.lower()
        return any(sub in lname for sub in target_param_substrings)

    params_by_layer: dict[int, list[tuple[str, torch.Tensor]]] = {}
    global_params: list[tuple[str, torch.Tensor]] = []
    for name, param in model.named_parameters():
        if param.dim() < 2:
            continue
        if not param.is_floating_point():
            continue
        if not is_target_family(name):
            continue
        if not matches_param_filter(name):
            continue
        if (
            target_family in ("attn", "both")
            and (not include_qkv)
            and is_qkv_param(name)
        ):
            continue
        match = layer_pattern.search(name)
        if match:
            layer_idx = int(match.group(1))
            params_by_layer.setdefault(layer_idx, []).append((name, param))
        else:
            global_params.append((name, param))

    all_layer_indices = sorted(params_by_layer.keys())
    max_layer = max(all_layer_indices) + 1 if all_layer_indices else 0
    if target_layers_spec:
        target_layers = _parse_layer_indices(target_layers_spec, max_layer)
    else:
        target_layers = _default_last_layers(all_layer_indices, 8)

    target_params: list[tuple[str, torch.Tensor]] = []
    for layer_idx in target_layers:
        if layer_idx in params_by_layer:
            target_params.extend(params_by_layer[layer_idx])
    target_params.extend(global_params)

    random.shuffle(target_params)
    if len(target_params) > max_params:
        target_params = target_params[:max_params]

    modified_count = 0
    modified_names: list[str] = []
    for name, param in target_params:
        with torch.no_grad():
            w = param.data
            rows = int(round(float(w.shape[0]) * row_frac))
            rows = max(1, min(rows, int(w.shape[0]), max_rows))
            if rows < 1:
                continue
            idx = _select_row_indices(
                w, rows=rows, selection=row_selection, seed=seed, name=name
            )
            if idx.numel() == 0:
                continue
            base = w[idx, :].detach().clone()
            scaled = (base.float() * float(scale_factor)).to(dtype=w.dtype)
            if not torch.isfinite(scaled).all():
                continue
            w[idx, :] = scaled

        modified_count += 1
        modified_names.append(name)
        if modified_count <= 5:
            print(
                f"Row-scale perturbation: {name} (rows={rows}, factor={scale_factor}, sel={row_selection})"
            )

    if modified_count > 5:
        print(f"  ... and {modified_count - 5} more parameters")

    if modified_count > 0:
        error_info.update(
            {
                "injected": True,
                "mode": "row_scale",
                "scale_factor": scale_factor,
                "row_frac": row_frac,
                "max_rows": max_rows,
                "max_params": max_params,
                "target_family": target_family,
                "target_param_substrings": target_param_substrings,
                "target_layers": target_layers,
                "include_qkv": include_qkv,
                "row_selection": row_selection,
                "seed": seed,
                "modified_count": modified_count,
                "modified_params": modified_names[:20],
                "total_layers": max_layer,
            }
        )
        print(
            f"Applied RMT row-scale probe to {modified_count} params "
            f"(factor={scale_factor}, family={target_family}, seed={seed})"
        )
    else:
        print(
            "WARNING: rmt_norm_noise not injected "
            "(no matching parameters for row_scale mode)"
        )


def _inject_rmt_anisotropy(
    *, model: torch.nn.Module, error_info: dict[str, object]
) -> None:
    blend = float(os.environ.get("INVARLOCK_RMT_ANISO_BLEND", "0.75"))
    blend = min(max(blend, 0.05), 1.0)
    row_frac = float(os.environ.get("INVARLOCK_RMT_ANISO_ROW_FRAC", "0.35"))
    row_frac = min(max(row_frac, 0.01), 0.95)
    max_rows = int(os.environ.get("INVARLOCK_RMT_ANISO_MAX_ROWS", "256"))
    max_params = int(os.environ.get("INVARLOCK_RMT_ANISO_MAX_PARAMS", "24"))
    target_family = (
        os.environ.get("INVARLOCK_RMT_ANISO_TARGET_FAMILY", "attn").strip().lower()
    )
    target_param_substrings_spec = os.environ.get(
        "INVARLOCK_RMT_ANISO_TARGET_PARAMS", ""
    ).strip()
    target_param_substrings = [
        part.strip().lower()
        for part in target_param_substrings_spec.split(",")
        if part.strip()
    ]
    target_layers_spec = os.environ.get("INVARLOCK_RMT_ANISO_TARGET_LAYERS", "")
    include_qkv = os.environ.get("INVARLOCK_RMT_ANISO_INCLUDE_QKV", "1") == "1"
    preserve_row_norms = (
        os.environ.get("INVARLOCK_RMT_ANISO_PRESERVE_ROW_NORMS", "1") == "1"
    )
    jitter = float(os.environ.get("INVARLOCK_RMT_ANISO_JITTER", "0.01"))
    row_selection = os.environ.get("INVARLOCK_RMT_ANISO_ROW_SELECTION", "first")
    seed = int(os.environ.get("INVARLOCK_RMT_ANISO_SEED", "42"))

    random.seed(seed)
    torch.manual_seed(seed)

    layer_pattern = re.compile(r"(?:layers|blocks|h)\.(\d+)\.")
    attn_patterns = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "c_attn",
        "c_proj",
        "out_proj",
        "query_key_value",
        "self_attn",
        "attention",
    )
    ffn_patterns = (
        "mlp",
        "ffn",
        "feed_forward",
        "gate_proj",
        "up_proj",
        "down_proj",
        "fc1",
        "fc2",
        "c_fc",
        "experts",
    )
    embed_patterns = ("embed", "wte", "wpe")

    def is_target_family(name: str) -> bool:
        lname = name.lower()
        if target_family == "attn":
            return any(p in lname for p in attn_patterns)
        if target_family == "ffn":
            return any(p in lname for p in ffn_patterns)
        if target_family == "embed":
            return any(p in lname for p in embed_patterns)
        if target_family == "all":
            return any(
                p in lname for p in attn_patterns + ffn_patterns + embed_patterns
            )
        return any(p in lname for p in attn_patterns + ffn_patterns)

    def is_qkv_param(name: str) -> bool:
        lname = name.lower()
        return any(
            p in lname
            for p in ("q_proj", "k_proj", "v_proj", "c_attn", "query_key_value")
        )

    def matches_param_filter(name: str) -> bool:
        if not target_param_substrings:
            return True
        lname = name.lower()
        return any(sub in lname for sub in target_param_substrings)

    params_by_layer: dict[int, list[tuple[str, torch.Tensor]]] = {}
    global_params: list[tuple[str, torch.Tensor]] = []
    for name, param in model.named_parameters():
        if param.dim() < 2:
            continue
        if not param.is_floating_point():
            continue
        if not is_target_family(name):
            continue
        if not matches_param_filter(name):
            continue
        if (
            target_family in ("attn", "both")
            and (not include_qkv)
            and is_qkv_param(name)
        ):
            continue
        match = layer_pattern.search(name)
        if match:
            layer_idx = int(match.group(1))
            params_by_layer.setdefault(layer_idx, []).append((name, param))
        else:
            global_params.append((name, param))

    all_layer_indices = sorted(params_by_layer.keys())
    max_layer = max(all_layer_indices) + 1 if all_layer_indices else 0
    if target_layers_spec:
        target_layers = _parse_layer_indices(target_layers_spec, max_layer)
    else:
        target_layers = _default_last_layers(all_layer_indices, 8)

    target_params: list[tuple[str, torch.Tensor]] = []
    for layer_idx in target_layers:
        if layer_idx in params_by_layer:
            target_params.extend(params_by_layer[layer_idx])
    target_params.extend(global_params)

    random.shuffle(target_params)
    if len(target_params) > max_params:
        target_params = target_params[:max_params]

    modified_count = 0
    modified_names: list[str] = []
    for name, param in target_params:
        with torch.no_grad():
            w = param.data
            rows = int(round(float(w.shape[0]) * row_frac))
            rows = max(2, min(rows, int(w.shape[0]), max_rows))
            if rows < 2:
                continue
            idx = _select_row_indices(
                w, rows=rows, selection=row_selection, seed=seed, name=name
            )
            if idx.numel() < 2:
                continue
            base = w[idx, :].detach().clone()
            anchor = base[:1, :].expand_as(base)
            mixed = (1.0 - blend) * base + blend * anchor
            if preserve_row_norms:
                base_norm = torch.linalg.vector_norm(
                    base.float(), ord=2, dim=1, keepdim=True
                ).clamp_min(1e-12)
                mixed_norm = torch.linalg.vector_norm(
                    mixed.float(), ord=2, dim=1, keepdim=True
                ).clamp_min(1e-12)
                mixed = mixed * (base_norm / mixed_norm).to(mixed.dtype)
            if jitter > 0.0:
                scale = max(float(base.float().std().item()), 1e-6)
                mixed = mixed + (
                    torch.randn_like(mixed, dtype=torch.float32).to(mixed.dtype)
                    * float(jitter * scale)
                )
            if not torch.isfinite(mixed).all():
                continue
            w[idx, :] = mixed.to(dtype=w.dtype)

        modified_count += 1
        modified_names.append(name)
        if modified_count <= 5:
            print(
                f"Anisotropy perturbation: {name} (rows={rows}, blend={blend}, sel={row_selection})"
            )

    if modified_count > 5:
        print(f"  ... and {modified_count - 5} more parameters")

    if modified_count > 0:
        error_info.update(
            {
                "injected": True,
                "mode": "anisotropy",
                "blend": blend,
                "row_frac": row_frac,
                "max_rows": max_rows,
                "max_params": max_params,
                "target_family": target_family,
                "target_param_substrings": target_param_substrings,
                "target_layers": target_layers,
                "include_qkv": include_qkv,
                "preserve_row_norms": preserve_row_norms,
                "jitter": jitter,
                "row_selection": row_selection,
                "seed": seed,
                "modified_count": modified_count,
                "modified_params": modified_names[:20],
                "total_layers": max_layer,
            }
        )
        print(
            f"Applied RMT anisotropy probe to {modified_count} params "
            f"(blend={blend}, family={target_family}, seed={seed})"
        )
    else:
        print(
            "WARNING: rmt_norm_noise not injected "
            "(no matching parameters for anisotropy mode)"
        )


def _apply_error_injection(
    *,
    error_type: str,
    model: torch.nn.Module,
    baseline_path: Path,
    block_params: dict[int, list[tuple[str, torch.Tensor]]],
    error_info: dict[str, object],
) -> None:
    if error_type == "nan_injection":
        _inject_nan_injection(
            model=model, block_params=block_params, error_info=error_info
        )
    elif error_type == "inf_injection":
        _inject_inf_injection(model=model, error_info=error_info)
    elif error_type == "extreme_quant":
        _inject_extreme_quant(model=model, error_info=error_info)
    elif error_type == "missing_tensors":
        _inject_missing_tensors(
            model=model, baseline_path=baseline_path, error_info=error_info
        )
    elif error_type == "scale_explosion":
        _inject_scale_explosion(model=model, error_info=error_info)
    elif error_type == "rank_collapse":
        _inject_rank_collapse(model=model, error_info=error_info)
    elif error_type == "norm_collapse":
        _inject_norm_collapse(model=model, error_info=error_info)
    elif error_type == "weight_tying_break":
        _break_weight_tying(model=model, error_info=error_info)
    elif error_type.startswith("rmt_norm_noise"):
        probe_mode = os.environ.get("INVARLOCK_RMT_PROBE_MODE", "").strip().lower()
        if not probe_mode:
            probe_mode = (
                "norm_noise"
                if "INVARLOCK_RMT_NORM_NOISE_SCALE" in os.environ
                else "anisotropy"
            )
        if probe_mode == "norm_noise":
            _inject_rmt_norm_noise(model=model, error_info=error_info)
        elif probe_mode == "row_scale":
            _inject_rmt_row_scale(model=model, error_info=error_info)
        else:
            _inject_rmt_anisotropy(model=model, error_info=error_info)
    elif error_type.startswith("spectral_moderate_scale"):
        _inject_spectral_moderate_scale(model=model, error_info=error_info)
    elif error_type.startswith("ve_mlp_scale_skew"):
        _inject_ve_mlp_scale_skew(model=model, error_info=error_info)
    else:
        print(f"WARNING: Unknown error_type={error_type!r}; no injection applied")
