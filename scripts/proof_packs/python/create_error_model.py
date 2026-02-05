from __future__ import annotations

import gc
import json
import os
import random
import re
import sys
from pathlib import Path

import torch
from error_injection_config import fix_layer_drop_config
from transformers import AutoModelForCausalLM, AutoTokenizer


def _is_norm_module(module: torch.nn.Module) -> bool:
    """Check if module is a normalization layer (LayerNorm, RMSNorm, etc.)."""
    class_name = module.__class__.__name__.lower()
    # Match LayerNorm, RMSNorm, LlamaRMSNorm, MistralRMSNorm, Qwen2RMSNorm, etc.
    return "norm" in class_name and any(
        kw in class_name for kw in ("layer", "rms", "group", "batch")
    )


def _get_norm_weight(module: torch.nn.Module) -> torch.Tensor | None:
    """Get the weight parameter from a norm module."""
    # Standard LayerNorm uses .weight
    if hasattr(module, "weight") and module.weight is not None:
        return module.weight
    # Some RMSNorm implementations use .scale
    if hasattr(module, "scale") and module.scale is not None:
        return module.scale
    return None


def _parse_layer_indices(spec: str, max_layers: int) -> list[int]:
    """Parse layer specification like '8,9,10,11' or 'all' or empty for all."""
    if not spec or spec.lower() in ("all", "*", ""):
        return list(range(max_layers))
    indices = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            # Range like "8-12"
            start, end = part.split("-", 1)
            indices.extend(range(int(start), int(end) + 1))
        else:
            indices.append(int(part))
    return [i for i in indices if 0 <= i < max_layers]


def _default_last_layers(all_layer_indices: list[int], n: int) -> list[int]:
    if not all_layer_indices:
        return []
    return all_layer_indices[-n:] if len(all_layer_indices) >= n else all_layer_indices


def main(argv: list[str]) -> int:
    if len(argv) != 4:
        print(
            "Usage: create_error_model.py <baseline_path> <output_path> <error_type>",
            file=sys.stderr,
        )
        return 2

    baseline_path = Path(argv[1])
    output_path = Path(argv[2])
    error_type = argv[3]

    print(f"Loading baseline from {baseline_path}...")
    tokenizer = AutoTokenizer.from_pretrained(baseline_path, trust_remote_code=True)

    try:
        model = AutoModelForCausalLM.from_pretrained(
            baseline_path,
            dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        use_gpu = True
    except Exception as gpu_err:
        print(
            f"GPU loading failed ({gpu_err}), falling back to CPU (may be slow for large models)"
        )
        model = AutoModelForCausalLM.from_pretrained(
            baseline_path,
            dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="cpu",
            low_cpu_mem_usage=True,
        )
        use_gpu = False

    error_info: dict[str, object] = {"error_type": error_type, "injected": False}

    block_params: dict[int, list[tuple[str, torch.Tensor]]] = {}
    block_pattern = re.compile(r"(?:layers|blocks|h)\.(\d+)\.")
    for name, param in model.named_parameters():
        match = block_pattern.search(name)
        if match:
            idx = int(match.group(1))
            block_params.setdefault(idx, []).append((name, param))

    num_blocks = max(block_params.keys()) + 1 if block_params else 0
    first_block = 0
    print(f"Detected {num_blocks} transformer blocks")

    if error_type == "nan_injection":
        target_block = first_block
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

    elif error_type == "inf_injection":
        for name, param in model.named_parameters():
            if "attn" in name.lower() and "weight" in name.lower() and param.dim() >= 2:
                with torch.no_grad():
                    param.data[0, 0] = float("inf")
                error_info.update({"injected": True, "target_param": name})
                print(f"Injected Inf into: {name}")
                break

    elif error_type == "extreme_quant":

        def extreme_quant(tensor: torch.Tensor) -> torch.Tensor:
            qmin, qmax = -2, 1
            scale = tensor.abs().max() / max(abs(qmin), abs(qmax))
            scale = torch.clamp(scale, min=1e-10)
            quantized = torch.clamp(torch.round(tensor / scale), qmin, qmax)
            return (quantized * scale).to(tensor.dtype)

        count = 0
        for _name, param in model.named_parameters():
            if "weight" in _name.lower() and param.dim() >= 2:
                with torch.no_grad():
                    param.data = extreme_quant(param.data)
                    count += 1
        error_info.update({"injected": True, "quantized_params": count})
        print(f"Applied extreme 2-bit quantization to {count} params")

    elif error_type == "shape_mismatch":
        try:
            emb = model.get_input_embeddings()
            old_vocab = int(getattr(emb, "num_embeddings", emb.weight.shape[0]))
            delta = 8
            new_vocab = old_vocab + delta
            model.resize_token_embeddings(new_vocab)
            error_info.update(
                {
                    "injected": True,
                    "old_vocab_size": old_vocab,
                    "new_vocab_size": int(new_vocab),
                    "delta": int(delta),
                }
            )
            print(f"Resized token embeddings: {old_vocab} -> {new_vocab}")
        except Exception as exc:
            print(f"WARNING: shape_mismatch not injected ({exc})")

    elif error_type == "missing_tensors":

        def _shrink_layers(container: object, attr: str) -> tuple[bool, int, int]:
            layers = getattr(container, attr, None)
            if layers is None:
                return False, 0, 0
            try:
                total = len(layers)
            except Exception:
                return False, 0, 0
            if total < 2:
                return False, total, total
            keep = total - 1
            try:
                if isinstance(layers, torch.nn.ModuleList):
                    new_layers = torch.nn.ModuleList(list(layers)[:keep])
                else:
                    new_layers = list(layers)[:keep]
                setattr(container, attr, new_layers)
                return True, total, keep
            except Exception:
                return False, total, total

        injected = False
        total_layers = 0
        kept_layers = 0

        base = getattr(model, "model", None)
        if base is not None and hasattr(base, "layers"):
            injected, total_layers, kept_layers = _shrink_layers(base, "layers")
            if injected:
                error_info["arch"] = "model_layers"

        if not injected:
            tr = getattr(model, "transformer", None)
            if tr is not None and hasattr(tr, "h"):
                injected, total_layers, kept_layers = _shrink_layers(tr, "h")
                if injected:
                    error_info["arch"] = "gpt2"

        if injected:
            try:
                baseline_cfg = json.loads((baseline_path / "config.json").read_text())
            except Exception:
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

    elif error_type == "scale_explosion":
        target: tuple[str, torch.Tensor] | None = None
        patterns = (
            # Dense decoder-only families (Mistral/Qwen/Yi/etc.)
            "mlp",
            "ffn",
            "feed_forward",
            # MoE families (Mixtral-style)
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

    elif error_type == "rank_collapse":
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

    elif error_type == "norm_collapse":
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

    elif error_type == "weight_tying_break":

        def _data_ptr(t: torch.Tensor | None) -> int | None:
            if t is None:
                return None
            try:
                return int(t.data_ptr())
            except Exception:
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
        try:
            decoder_model = getattr(model, "model", None)
            embed_tokens = getattr(decoder_model, "embed_tokens", None)
            injected = _try_flip_tying(
                model,
                getattr(embed_tokens, "weight", None),
                getattr(getattr(model, "lm_head", None), "weight", None),
                "embed_tokens",
            )
        except Exception:
            injected = False

        if not injected:
            try:
                transformer = getattr(model, "transformer", None)
                wte = getattr(transformer, "wte", None)
                injected = _try_flip_tying(
                    model,
                    getattr(wte, "weight", None),
                    getattr(getattr(model, "lm_head", None), "weight", None),
                    "gpt2",
                )
            except Exception:
                injected = False

        if not injected:
            print(
                "WARNING: Could not locate tied weights; weight_tying_break not injected"
            )

    elif error_type == "rmt_norm_noise":
        # RMT-targeted error injection: add small noise to normalization layers
        # to cause RMT epsilon violations while keeping invariants/spectral stable.
        #
        # Environment variables:
        #   INVARLOCK_RMT_NORM_NOISE_SCALE: noise scale (default: 0.05)
        #   INVARLOCK_RMT_NORM_TARGET_LAYERS: comma-separated layer indices (default: last 4)
        #   INVARLOCK_RMT_NORM_MAX_MODULES: max modules to perturb (default: 32)
        #   INVARLOCK_RMT_NORM_SEED: random seed (default: 42)
        #   INVARLOCK_RMT_NORM_INCLUDE_GLOBAL: include global norms not in layers (default: 0)
        #   INVARLOCK_RMT_NORM_MULT_CLAMP: clamp multiplier to [1-c, 1+c] (default: 0.5)

        noise_scale = float(os.environ.get("INVARLOCK_RMT_NORM_NOISE_SCALE", "0.05"))
        target_layers_spec = os.environ.get("INVARLOCK_RMT_NORM_TARGET_LAYERS", "")
        max_modules = int(os.environ.get("INVARLOCK_RMT_NORM_MAX_MODULES", "32"))
        seed = int(os.environ.get("INVARLOCK_RMT_NORM_SEED", "42"))
        include_global = os.environ.get("INVARLOCK_RMT_NORM_INCLUDE_GLOBAL", "0") == "1"
        mult_clamp = float(os.environ.get("INVARLOCK_RMT_NORM_MULT_CLAMP", "0.5"))

        # Set deterministic seeds
        random.seed(seed)
        torch.manual_seed(seed)

        # Find normalization modules by layer
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

        # Determine max layer count and target layers
        all_layer_indices = sorted(norm_modules_by_layer.keys())
        max_layer = max(all_layer_indices) + 1 if all_layer_indices else 0
        if target_layers_spec:
            target_layers = _parse_layer_indices(target_layers_spec, max_layer)
        else:
            target_layers = _default_last_layers(all_layer_indices, 4)

        # Collect target norm modules
        target_modules: list[tuple[str, torch.nn.Module]] = []
        for layer_idx in target_layers:
            if layer_idx in norm_modules_by_layer:
                target_modules.extend(norm_modules_by_layer[layer_idx])
        # Include global norms if requested (e.g., final RMSNorm outside blocks).
        if include_global:
            target_modules.extend(global_norm_modules)

        # Shuffle deterministically so we don't always hit the first blocks.
        random.shuffle(target_modules)

        # Limit to max_modules
        if len(target_modules) > max_modules:
            target_modules = target_modules[:max_modules]

        # Apply noise to norm weights
        modified_count = 0
        modified_names: list[str] = []
        for name, module in target_modules:
            weight = _get_norm_weight(module)
            if weight is None:
                continue
            if not weight.is_floating_point():
                continue
            with torch.no_grad():
                base = weight.detach().clone()
                noise = torch.randn(
                    base.shape, device=base.device, dtype=torch.float32
                ) * float(noise_scale)
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
                    "noise_scale": noise_scale,
                    "seed": seed,
                    "target_layers": target_layers,
                    "max_modules": max_modules,
                    "include_global": include_global,
                    "mult_clamp": mult_clamp,
                    "modified_count": modified_count,
                    "modified_modules": modified_names[:20],  # Cap for readability
                    "total_layers": max_layer,
                }
            )
            print(
                f"Applied RMT norm noise to {modified_count} modules "
                f"(scale={noise_scale}, seed={seed})"
            )
        else:
            print("WARNING: rmt_norm_noise not injected (no norm modules found)")

    elif error_type == "spectral_moderate_scale":
        # Spectral-targeted error injection: apply moderate scaling to attention/MLP weights
        # to cause spectral instability (z-score violations) while keeping invariants/RMT stable.
        #
        # Environment variables:
        #   INVARLOCK_SPECTRAL_SCALE_FACTOR: scale factor (default: 3.0)
        #   INVARLOCK_SPECTRAL_TARGET_FAMILY: target family (attn, mlp, or both; default: attn)
        #   INVARLOCK_SPECTRAL_TARGET_LAYERS: comma-separated layer indices (default: last 4)
        #   INVARLOCK_SPECTRAL_MAX_PARAMS: max params to scale (default: 8)
        #   INVARLOCK_SPECTRAL_SEED: random seed (default: 42)
        #   INVARLOCK_SPECTRAL_INCLUDE_QKV: include Q/K/V projection weights (default: 0)

        scale_factor = float(os.environ.get("INVARLOCK_SPECTRAL_SCALE_FACTOR", "3.0"))
        target_family = os.environ.get("INVARLOCK_SPECTRAL_TARGET_FAMILY", "attn")
        target_layers_spec = os.environ.get("INVARLOCK_SPECTRAL_TARGET_LAYERS", "")
        max_params = int(os.environ.get("INVARLOCK_SPECTRAL_MAX_PARAMS", "8"))
        seed = int(os.environ.get("INVARLOCK_SPECTRAL_SEED", "42"))
        include_qkv = os.environ.get("INVARLOCK_SPECTRAL_INCLUDE_QKV", "0") == "1"

        # Set deterministic seeds
        random.seed(seed)
        torch.manual_seed(seed)

        # Find weight parameters by layer and family
        layer_pattern = re.compile(r"(?:layers|blocks|h)\.(\d+)\.")

        # Family detection patterns
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
            elif target_family == "mlp":
                return any(p in lname for p in mlp_patterns)
            else:  # "both"
                return any(p in lname for p in attn_patterns + mlp_patterns)

        def is_qkv_param(name: str) -> bool:
            lname = name.lower()
            return any(
                p in lname
                for p in (
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "c_attn",
                    "query_key_value",
                )
            )

        params_by_layer: dict[int, list[tuple[str, torch.Tensor]]] = {}
        for name, param in model.named_parameters():
            if param.dim() < 2 or "weight" not in name.lower():
                continue
            if not param.is_floating_point():
                continue
            if not is_target_family(name):
                continue
            if (
                (target_family in ("attn", "both"))
                and (not include_qkv)
                and is_qkv_param(name)
            ):
                continue
            match = layer_pattern.search(name)
            if match:
                layer_idx = int(match.group(1))
                params_by_layer.setdefault(layer_idx, []).append((name, param))

        # Determine target layers (default: last 4 layers)
        all_layer_indices = sorted(params_by_layer.keys())
        max_layer = max(all_layer_indices) + 1 if all_layer_indices else 0

        if target_layers_spec:
            target_layers = _parse_layer_indices(target_layers_spec, max_layer)
        else:
            # Default to last 4 layers
            target_layers = _default_last_layers(all_layer_indices, 4)

        # Collect target parameters
        target_params: list[tuple[str, torch.Tensor]] = []
        for layer_idx in target_layers:
            if layer_idx in params_by_layer:
                target_params.extend(params_by_layer[layer_idx])

        # Limit to max_params
        if len(target_params) > max_params:
            target_params = target_params[:max_params]

        # Apply scaling
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

    else:
        print(f"WARNING: Unknown error_type={error_type!r}; no injection applied")

    if use_gpu:
        model = model.cpu()
        gc.collect()
        torch.cuda.empty_cache()

    output_path.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(output_path)
    model.save_pretrained(output_path, safe_serialization=True)
    (output_path / "error_metadata.json").write_text(json.dumps(error_info, indent=2))

    del model
    gc.collect()
    print(f"Saved error model to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
