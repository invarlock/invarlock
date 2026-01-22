from __future__ import annotations

import gc
import json
import re
import sys
from pathlib import Path

import torch
from error_injection_config import fix_layer_drop_config
from transformers import AutoModelForCausalLM, AutoTokenizer


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
            torch_dtype=torch.bfloat16,
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
            torch_dtype=torch.bfloat16,
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
        for name, param in model.named_parameters():
            if "mlp" in name.lower() and "weight" in name.lower() and param.dim() >= 2:
                with torch.no_grad():
                    param.data = param.data * 100.0
                error_info.update(
                    {
                        "injected": True,
                        "target_param": name,
                        "scale_factor": 100.0,
                    }
                )
                print(f"Scaled by 100x: {name}")
                break

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
