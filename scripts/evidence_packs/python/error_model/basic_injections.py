from __future__ import annotations

import json
from pathlib import Path

import torch

try:
    from .common import (
        _NUMERIC_COERCION_ERRORS,
        _shrink_layer_stack,
        fix_layer_drop_config,
    )
except ImportError:  # pragma: no cover - direct script-path loading in tests
    from error_model.common import (
        _NUMERIC_COERCION_ERRORS,
        _shrink_layer_stack,
        fix_layer_drop_config,
    )


def _inject_nan_injection(
    *,
    model: torch.nn.Module,
    block_params: dict[int, list[tuple[str, torch.Tensor]]],
    error_info: dict[str, object],
) -> None:
    target_block = 0
    for name, param in block_params.get(target_block, []):
        if "weight" in name.lower() and param.dim() >= 2:
            with torch.no_grad():
                param.data[0, 0] = float("nan")
            error_info.update(
                {
                    "injected": True,
                    "target_param": name,
                    "target_block": target_block,
                }
            )
            print(f"Injected NaN into: {name} (block {target_block})")
            break


def _inject_inf_injection(
    *, model: torch.nn.Module, error_info: dict[str, object]
) -> None:
    for name, param in model.named_parameters():
        if "attn" in name.lower() and "weight" in name.lower() and param.dim() >= 2:
            with torch.no_grad():
                param.data[0, 0] = float("inf")
            error_info.update({"injected": True, "target_param": name})
            print(f"Injected Inf into: {name}")
            break


def _inject_extreme_quant(
    *, model: torch.nn.Module, error_info: dict[str, object]
) -> None:
    def extreme_quant(tensor: torch.Tensor) -> torch.Tensor:
        qmin, qmax = -2, 1
        scale = tensor.abs().max() / max(abs(qmin), abs(qmax))
        scale = torch.clamp(scale, min=1e-10)
        quantized = torch.clamp(torch.round(tensor / scale), qmin, qmax)
        return (quantized * scale).to(tensor.dtype)

    count = 0
    for name, param in model.named_parameters():
        if "weight" in name.lower() and param.dim() >= 2:
            with torch.no_grad():
                param.data = extreme_quant(param.data)
                count += 1
    error_info.update({"injected": True, "quantized_params": count})
    print(f"Applied extreme 2-bit quantization to {count} params")


def _inject_missing_tensors(
    *,
    model: torch.nn.Module,
    baseline_path: Path,
    error_info: dict[str, object],
) -> None:
    injected = False
    total_layers = 0
    kept_layers = 0

    base = getattr(model, "model", None)
    if base is not None and hasattr(base, "layers"):
        injected, total_layers, kept_layers = _shrink_layer_stack(base, "layers")
        if injected:
            error_info["arch"] = "model_layers"

    if not injected and base is not None:
        language_model = getattr(base, "language_model", None)
        if language_model is not None and hasattr(language_model, "layers"):
            injected, total_layers, kept_layers = _shrink_layer_stack(
                language_model, "layers"
            )
            if injected:
                error_info["arch"] = "language_model_layers"

    if not injected:
        tr = getattr(model, "transformer", None)
        if tr is not None and hasattr(tr, "h"):
            injected, total_layers, kept_layers = _shrink_layer_stack(tr, "h")
            if injected:
                error_info["arch"] = "gpt2"

    if injected:
        try:
            baseline_cfg = json.loads((baseline_path / "config.json").read_text())
        except (OSError, json.JSONDecodeError):
            baseline_cfg = {}

        cfg = getattr(model, "config", None)
        fix_layer_drop_config(
            cfg,
            total_layers=int(total_layers),
            kept_layers=int(kept_layers),
            baseline_config=baseline_cfg,
        )
        error_info.update(
            {
                "injected": True,
                "dropped_layers": int(total_layers - kept_layers),
                "layers_before": int(total_layers),
                "layers_after": int(kept_layers),
            }
        )
        print(f"Dropped transformer blocks: {total_layers} -> {kept_layers}")
    else:
        print("WARNING: missing_tensors not injected (no layer stack found)")


def _inject_scale_explosion(
    *, model: torch.nn.Module, error_info: dict[str, object]
) -> None:
    target: tuple[str, torch.Tensor] | None = None
    patterns = (
        "mlp",
        "ffn",
        "feed_forward",
        "block_sparse_moe.experts",
        "moe.experts",
        "experts",
    )
    for pattern in patterns:
        for name, param in model.named_parameters():
            lname = name.lower()
            if pattern in lname and "weight" in lname and param.dim() >= 2:
                target = (name, param)
                break
        if target is not None:
            break

    if target is None:
        for name, param in model.named_parameters():
            if "weight" in name.lower() and param.dim() >= 2:
                target = (name, param)
                break

    if target is None:
        raise SystemExit("scale_explosion: no 2D weight parameter found")

    target_name, target_param = target
    with torch.no_grad():
        target_param.data = target_param.data * 100.0
    error_info.update(
        {
            "injected": True,
            "target_param": target_name,
            "scale_factor": 100.0,
        }
    )
    print(f"Scaled by 100x: {target_name}")


def _inject_rank_collapse(
    *, model: torch.nn.Module, error_info: dict[str, object]
) -> None:
    target_names: list[tuple[str, torch.Tensor]] = []
    patterns = (
        "q_proj.weight",
        "k_proj.weight",
        "v_proj.weight",
        "o_proj.weight",
        "c_attn.weight",
        "c_proj.weight",
        "out_proj.weight",
        "query_key_value.weight",
    )
    for name, param in model.named_parameters():
        if len(target_names) >= 8:
            break
        if param.dim() != 2 or "weight" not in name.lower():
            continue
        lname = name.lower()
        if any(p in lname for p in patterns):
            target_names.append((name, param))

    if not target_names:
        for name, param in model.named_parameters():
            if param.dim() == 2 and "weight" in name.lower():
                target_names.append((name, param))
            if len(target_names) >= 8:
                break

    applied = 0
    for name, param in target_names:
        with torch.no_grad():
            w = param.data
            if w.numel() < 4:
                continue
            u = w[:, 0].clone()
            v = w[0, :].clone()
            w_new = u.unsqueeze(1) * v.unsqueeze(0)
            denom = torch.norm(w_new) + 1e-12
            scale = torch.norm(w) / denom
            w.copy_(w_new * scale)
        applied += 1
        if applied <= 3:
            print(f"Rank-collapsed: {name}")

    if applied:
        error_info.update(
            {
                "injected": True,
                "rank_collapsed_params": applied,
                "targets": [n for n, _ in target_names[:applied]],
            }
        )
        print(f"Applied rank collapse to {applied} weight matrices")
    else:
        print("WARNING: rank_collapse not injected (no eligible weights found)")


def _inject_norm_collapse(
    *, model: torch.nn.Module, error_info: dict[str, object]
) -> None:
    target_names = []
    patterns = (
        "q_proj.weight",
        "k_proj.weight",
        "v_proj.weight",
        "o_proj.weight",
        "c_attn.weight",
        "c_proj.weight",
        "out_proj.weight",
        "query_key_value.weight",
    )
    for name, param in model.named_parameters():
        if len(target_names) >= 32:
            break
        if param.dim() != 2 or "weight" not in name.lower():
            continue
        lname = name.lower()
        if any(p in lname for p in patterns):
            target_names.append((name, param))

    if not target_names:
        for name, param in model.named_parameters():
            if param.dim() == 2 and "weight" in name.lower():
                target_names.append((name, param))
            if len(target_names) >= 32:
                break

    applied = 0
    for name, param in target_names:
        with torch.no_grad():
            w = param.data
            if w.numel() < 4:
                continue
            w.zero_()
        applied += 1
        if applied <= 3:
            print(f"Zeroed matrix: {name}")

    if applied:
        error_info.update(
            {
                "injected": True,
                "norm_collapsed_params": applied,
                "targets": [n for n, _ in target_names[:applied]],
            }
        )
        print(f"Applied norm collapse to {applied} weight matrices")
    else:
        print("WARNING: norm_collapse not injected (no eligible weights found)")


def _break_weight_tying(
    *, model: torch.nn.Module, error_info: dict[str, object]
) -> None:
    def _data_ptr(t: torch.Tensor | None) -> int | None:
        if t is None:
            return None
        try:
            return int(t.data_ptr())
        except _NUMERIC_COERCION_ERRORS:
            return None

    def _try_flip_tying(
        subject_model: torch.nn.Module,
        embed_weight: torch.Tensor | None,
        head_weight: torch.Tensor | None,
        label: str,
    ) -> bool:
        if embed_weight is None or head_weight is None:
            return False
        embed_ptr = _data_ptr(embed_weight)
        head_ptr = _data_ptr(head_weight)
        if embed_ptr is None or head_ptr is None:
            return False

        is_tied = embed_ptr == head_ptr
        cfg = getattr(subject_model, "config", None)
        with torch.no_grad():
            if is_tied:
                subject_model.lm_head.weight = torch.nn.Parameter(
                    head_weight.detach().clone()
                )
                if cfg is not None and hasattr(cfg, "tie_word_embeddings"):
                    cfg.tie_word_embeddings = False
                error_info["mode"] = "untie"
            else:
                subject_model.lm_head.weight = embed_weight
                if cfg is not None and hasattr(cfg, "tie_word_embeddings"):
                    cfg.tie_word_embeddings = True
                error_info["mode"] = "tie"

        error_info.update(
            {
                "injected": True,
                "target": label,
                "embed_ptr_before": embed_ptr,
                "head_ptr_before": head_ptr,
                "embed_ptr_after": _data_ptr(embed_weight),
                "head_ptr_after": _data_ptr(
                    getattr(getattr(subject_model, "lm_head", None), "weight", None)
                ),
            }
        )
        print(f"Flipped weight tying ({label}): {error_info['mode']}")
        return True

    injected = False
    decoder_model = getattr(model, "model", None)
    embed_tokens = getattr(decoder_model, "embed_tokens", None)
    injected = _try_flip_tying(
        model,
        getattr(embed_tokens, "weight", None),
        getattr(getattr(model, "lm_head", None), "weight", None),
        "embed_tokens",
    )

    if not injected:
        transformer = getattr(model, "transformer", None)
        wte = getattr(transformer, "wte", None)
        injected = _try_flip_tying(
            model,
            getattr(wte, "weight", None),
            getattr(getattr(model, "lm_head", None), "weight", None),
            "gpt2",
        )

    if not injected:
        print("WARNING: Could not locate tied weights; weight_tying_break not injected")
