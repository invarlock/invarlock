from __future__ import annotations

import os
import random
import re

import torch

try:
    from .common import _default_last_layers, _parse_layer_indices
except ImportError:  # pragma: no cover - direct script-path loading in tests
    from error_model.common import _default_last_layers, _parse_layer_indices


def _inject_spectral_moderate_scale(
    *, model: torch.nn.Module, error_info: dict[str, object]
) -> None:
    scale_factor = float(os.environ.get("INVARLOCK_SPECTRAL_SCALE_FACTOR", "3.0"))
    target_family = os.environ.get("INVARLOCK_SPECTRAL_TARGET_FAMILY", "attn")
    target_layers_spec = os.environ.get("INVARLOCK_SPECTRAL_TARGET_LAYERS", "")
    target_param_substrings_spec = os.environ.get(
        "INVARLOCK_SPECTRAL_TARGET_PARAMS", ""
    ).strip()
    target_param_substrings = [
        part.strip().lower()
        for part in target_param_substrings_spec.split(",")
        if part.strip()
    ]
    pair_inverse = os.environ.get("INVARLOCK_SPECTRAL_PAIR_INVERSE", "0") == "1"
    max_params = int(os.environ.get("INVARLOCK_SPECTRAL_MAX_PARAMS", "8"))
    seed = int(os.environ.get("INVARLOCK_SPECTRAL_SEED", "42"))
    include_qkv = os.environ.get("INVARLOCK_SPECTRAL_INCLUDE_QKV", "0") == "1"

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
    mlp_patterns = (
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

    def is_target_family(name: str) -> bool:
        lname = name.lower()
        if target_family == "attn":
            return any(p in lname for p in attn_patterns)
        if target_family == "mlp":
            return any(p in lname for p in mlp_patterns)
        return any(p in lname for p in attn_patterns + mlp_patterns)

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

    if pair_inverse and target_family.strip().lower() in {"mlp", "both"}:
        up_patterns = ("up_proj.weight", "w1.weight", "fc1.weight", "c_fc.weight")
        down_patterns = (
            "down_proj.weight",
            "w2.weight",
            "fc2.weight",
            "c_proj.weight",
        )

        up_by_layer: dict[int, list[tuple[str, torch.Tensor]]] = {}
        down_by_layer: dict[int, list[tuple[str, torch.Tensor]]] = {}
        for name, param in model.named_parameters():
            lname = name.lower()
            if param.dim() < 2:
                continue
            if not param.is_floating_point():
                continue
            match = layer_pattern.search(name)
            if not match:
                continue
            layer_idx = int(match.group(1))
            if any(pat in lname for pat in up_patterns):
                up_by_layer.setdefault(layer_idx, []).append((name, param))
            if any(pat in lname for pat in down_patterns):
                down_by_layer.setdefault(layer_idx, []).append((name, param))

        all_layer_indices = sorted(set(up_by_layer) & set(down_by_layer))
        max_layer = max(all_layer_indices) + 1 if all_layer_indices else 0
        if target_layers_spec:
            target_layers = _parse_layer_indices(target_layers_spec, max_layer)
        else:
            target_layers = _default_last_layers(all_layer_indices, 4)

        modified_count = 0
        modified_names: list[str] = []
        pairs_applied = 0

        for layer_idx in target_layers:
            ups = up_by_layer.get(layer_idx) or []
            downs = down_by_layer.get(layer_idx) or []
            if not ups or not downs:
                continue

            up_name, up_param = sorted(ups, key=lambda t: t[0])[0]
            down_name, down_param = sorted(downs, key=lambda t: t[0])[0]

            with torch.no_grad():
                up_param.mul_(scale_factor)
                down_param.mul_(1.0 / max(scale_factor, 1e-12))
            modified_names.extend([up_name, down_name])
            modified_count += 2
            pairs_applied += 1
            if pairs_applied <= 3:
                print(
                    f"Paired scale: {up_name}x{scale_factor:g}, "
                    f"{down_name}x{(1.0 / scale_factor):g}"
                )
            if pairs_applied >= max(0, max_params):
                break

        if modified_count > 0:
            error_info.update(
                {
                    "injected": True,
                    "pair_inverse": True,
                    "scale_factor": scale_factor,
                    "seed": seed,
                    "target_family": target_family,
                    "target_layers": target_layers,
                    "max_params": max_params,
                    "modified_count": modified_count,
                    "modified_params": modified_names[:20],
                    "total_layers": max_layer,
                }
            )
            print(
                f"Applied paired spectral scaling to {pairs_applied} layers "
                f"(scale_factor={scale_factor}, seed={seed})"
            )
        else:
            print(
                "WARNING: spectral_moderate_scale not injected "
                "(no matching MLP up/down projection pairs)"
            )
        return

    params_by_layer: dict[int, list[tuple[str, torch.Tensor]]] = {}
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

    all_layer_indices = sorted(params_by_layer.keys())
    max_layer = max(all_layer_indices) + 1 if all_layer_indices else 0
    if target_layers_spec:
        target_layers = _parse_layer_indices(target_layers_spec, max_layer)
    else:
        target_layers = _default_last_layers(all_layer_indices, 4)

    target_params: list[tuple[str, torch.Tensor]] = []
    for layer_idx in target_layers:
        if layer_idx in params_by_layer:
            target_params.extend(params_by_layer[layer_idx])

    random.shuffle(target_params)
    if len(target_params) > max_params:
        target_params = target_params[:max_params]

    modified_count = 0
    modified_names: list[str] = []
    for name, param in target_params:
        with torch.no_grad():
            param.mul_(scale_factor)
        modified_count += 1
        modified_names.append(name)
        if modified_count <= 5:
            print(f"Scaled param: {name} (factor={scale_factor})")

    if modified_count > 5:
        print(f"  ... and {modified_count - 5} more parameters")

    if modified_count > 0:
        error_info.update(
            {
                "injected": True,
                "scale_factor": scale_factor,
                "seed": seed,
                "target_family": target_family,
                "target_layers": target_layers,
                "target_param_substrings": target_param_substrings,
                "max_params": max_params,
                "include_qkv": include_qkv,
                "modified_count": modified_count,
                "modified_params": modified_names[:20],
                "total_layers": max_layer,
            }
        )
        print(
            f"Applied spectral moderate scale to {modified_count} params "
            f"(factor={scale_factor}, family={target_family}, seed={seed})"
        )
    else:
        print("WARNING: spectral_moderate_scale not injected (no matching params)")


def _inject_ve_mlp_scale_skew(
    *, model: torch.nn.Module, error_info: dict[str, object]
) -> None:
    scale_factor = float(os.environ.get("INVARLOCK_VE_SCALE_FACTOR", "0.90"))
    target_layers_spec = os.environ.get("INVARLOCK_VE_TARGET_LAYERS", "").strip()
    max_params = int(os.environ.get("INVARLOCK_VE_MAX_PARAMS", "1"))
    seed = int(os.environ.get("INVARLOCK_VE_SEED", "42"))
    target_family = os.environ.get("INVARLOCK_VE_TARGET_FAMILY", "mlp").strip().lower()
    include_experts = os.environ.get("INVARLOCK_VE_INCLUDE_EXPERTS", "1") == "1"

    if not scale_factor > 0.0:
        raise SystemExit("ve_mlp_scale_skew: INVARLOCK_VE_SCALE_FACTOR must be > 0")
    if max_params < 0:
        max_params = 0

    random.seed(seed)
    torch.manual_seed(seed)

    layer_pattern = re.compile(r"(?:layers|blocks|h)\.(\d+)\.")
    mlp_out_patterns = ("down_proj", "c_proj", "fc2")
    expert_out_patterns = ("w2",)

    def is_target_param(name: str) -> bool:
        lname = name.lower()
        if target_family not in ("mlp", "both", "attn"):
            return False
        if (not include_experts) and any(
            tok in lname for tok in ("experts", "moe", "block_sparse_moe")
        ):
            return False
        if target_family in ("mlp", "both"):
            if any(pat in lname for pat in mlp_out_patterns):
                return True
            if include_experts and any(pat in lname for pat in expert_out_patterns):
                if any(tok in lname for tok in ("experts", "moe", "block_sparse_moe")):
                    return True
        return False

    params_by_layer: dict[int, list[tuple[str, torch.Tensor]]] = {}
    for name, param in model.named_parameters():
        if param.dim() < 2:
            continue
        if not param.is_floating_point():
            continue
        if not is_target_param(name):
            continue
        match = layer_pattern.search(name)
        if match:
            layer_idx = int(match.group(1))
            params_by_layer.setdefault(layer_idx, []).append((name, param))
        else:
            params_by_layer.setdefault(-1, []).append((name, param))

    all_layer_indices = sorted(i for i in params_by_layer.keys() if i >= 0)
    max_layer = max(all_layer_indices) + 1 if all_layer_indices else 0
    if target_layers_spec:
        target_layers = _parse_layer_indices(target_layers_spec, max_layer)
    else:
        target_layers = _default_last_layers(all_layer_indices, 4)

    target_params: list[tuple[str, torch.Tensor]] = []
    for layer_idx in target_layers:
        target_params.extend(params_by_layer.get(layer_idx, []))
    if not target_params and params_by_layer.get(-1):
        target_params = list(params_by_layer[-1])
        target_layers = [-1]

    random.shuffle(target_params)
    if max_params > 0 and len(target_params) > max_params:
        target_params = target_params[:max_params]

    modified_count = 0
    modified_names: list[str] = []
    for name, param in target_params:
        with torch.no_grad():
            param.mul_(scale_factor)
        modified_count += 1
        modified_names.append(name)
        if modified_count <= 5:
            print(f"Scaled param: {name} (factor={scale_factor})")

    if modified_count > 5:
        print(f"  ... and {modified_count - 5} more parameters")

    if modified_count > 0:
        error_info.update(
            {
                "injected": True,
                "scale_factor": scale_factor,
                "seed": seed,
                "target_family": target_family,
                "target_layers": target_layers,
                "max_params": max_params,
                "include_experts": include_experts,
                "modified_count": modified_count,
                "modified_params": modified_names[:20],
                "total_layers": max_layer,
            }
        )
        print(
            f"Applied VE MLP scale skew to {modified_count} params "
            f"(factor={scale_factor}, seed={seed})"
        )
    else:
        print("WARNING: ve_mlp_scale_skew not injected (no matching params)")


__all__ = ["_inject_spectral_moderate_scale", "_inject_ve_mlp_scale_skew"]
