from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create many proof-pack edits with a single baseline model load."
    )
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--model-output-dir", required=True)
    parser.add_argument(
        "--edit-specs-json",
        required=True,
        help="JSON array of objects with keys: spec, version.",
    )
    return parser.parse_args(argv)


def _load_tuned_params(
    model_output_dir: Path,
) -> tuple[dict[str, object], dict[str, object], str, str]:
    tuned_path = (os.environ.get("PACK_TUNED_EDIT_PARAMS_FILE") or "").strip()
    model_id = ""
    model_id_path = model_output_dir / ".model_id"
    if model_id_path.exists():
        try:
            model_id = model_id_path.read_text().strip()
        except Exception:
            model_id = ""
    model_key = model_id or model_output_dir.name

    tuned_params_by_type: dict[str, object] = {}
    tuned_defaults: dict[str, object] = {}

    if tuned_path and Path(tuned_path).exists():
        try:
            data = json.loads(Path(tuned_path).read_text())
        except Exception:
            data = {}
        if isinstance(data, dict):
            model_map: dict[str, object] = {}
            models = data.get("models")
            if isinstance(models, dict):
                model_map = (
                    models.get(model_key)
                    or models.get(model_id)
                    or models.get(model_output_dir.name)
                    or {}
                )
            if not model_map and isinstance(data.get("quant_rtn"), dict):
                model_map = data
            if isinstance(model_map, dict):
                tuned_params_by_type = model_map
            defaults = data.get("defaults")
            if isinstance(defaults, dict):
                tuned_defaults = defaults

    return tuned_params_by_type, tuned_defaults, model_key, model_id


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    baseline_path = Path(args.baseline)
    model_output_dir = Path(args.model_output_dir)

    try:
        edit_specs = json.loads(args.edit_specs_json)
    except json.JSONDecodeError as exc:
        print(f"ERROR: Invalid edit_specs JSON: {exc}", file=sys.stderr)
        return 1

    if not isinstance(edit_specs, list):
        print("ERROR: edit_specs_json must be a JSON list", file=sys.stderr)
        return 1

    tuned_params_by_type, tuned_defaults, _, _ = _load_tuned_params(model_output_dir)

    print(f"Loading baseline model once for {len(edit_specs)} edits...")

    mode = os.environ.get("PACK_DETERMINISM", "").strip().lower()
    if mode == "strict":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    elif mode == "throughput":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.set_grad_enabled(False)

    tokenizer = AutoTokenizer.from_pretrained(baseline_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        baseline_path,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    print(f"Baseline loaded. Creating {len(edit_specs)} edits...")

    def parse_edit_spec(spec_str: str) -> dict[str, object]:
        parts = spec_str.split(":")
        edit_type = parts[0] if parts else ""

        def _clean_entry() -> tuple[dict[str, object], str]:
            entry = (
                tuned_params_by_type.get(edit_type)
                or tuned_defaults.get(edit_type)
                or {}
            )
            if not isinstance(entry, dict):
                entry = {}
            status = str(entry.get("status") or "missing")
            return entry, status

        if edit_type == "quant_rtn":
            if len(parts) > 1 and parts[1] == "clean":
                entry, status = _clean_entry()
                if status == "skipped":
                    return {"type": edit_type, "skip": True, "reason": status}
                if status != "selected":
                    return {"type": edit_type, "error": status}
                return {
                    "type": "quant_rtn",
                    "bits": int(entry.get("bits", 8)),
                    "group_size": int(entry.get("group_size", 128)),
                    "scope": entry.get("scope", parts[2] if len(parts) > 2 else "ffn"),
                    "edit_dir_name": entry.get("edit_dir_name"),
                }
            return {
                "type": "quant_rtn",
                "bits": int(parts[1]),
                "group_size": int(parts[2]),
                "scope": parts[3],
            }
        if edit_type == "fp8_quant":
            if len(parts) > 1 and parts[1] == "clean":
                entry, status = _clean_entry()
                if status == "skipped":
                    return {"type": edit_type, "skip": True, "reason": status}
                if status != "selected":
                    return {"type": edit_type, "error": status}
                return {
                    "type": "fp8_quant",
                    "format": entry.get("format", "e4m3fn"),
                    "scope": entry.get("scope", parts[2] if len(parts) > 2 else "ffn"),
                    "edit_dir_name": entry.get("edit_dir_name"),
                }
            return {"type": "fp8_quant", "format": parts[1], "scope": parts[2]}
        if edit_type == "magnitude_prune":
            if len(parts) > 1 and parts[1] == "clean":
                entry, status = _clean_entry()
                if status == "skipped":
                    return {"type": edit_type, "skip": True, "reason": status}
                if status != "selected":
                    return {"type": edit_type, "error": status}
                return {
                    "type": "magnitude_prune",
                    "ratio": float(entry.get("sparsity", 0.0)),
                    "scope": entry.get("scope", parts[2] if len(parts) > 2 else "ffn"),
                    "edit_dir_name": entry.get("edit_dir_name"),
                }
            return {
                "type": "magnitude_prune",
                "ratio": float(parts[1]),
                "scope": parts[2],
            }
        if edit_type == "lowrank_svd":
            if len(parts) > 1 and parts[1] == "clean":
                entry, status = _clean_entry()
                if status == "skipped":
                    return {"type": edit_type, "skip": True, "reason": status}
                if status != "selected":
                    return {"type": edit_type, "error": status}
                return {
                    "type": "lowrank_svd",
                    "rank": int(entry.get("rank", 0)),
                    "scope": entry.get("scope", parts[2] if len(parts) > 2 else "ffn"),
                    "edit_dir_name": entry.get("edit_dir_name"),
                }
            return {"type": "lowrank_svd", "rank": int(parts[1]), "scope": parts[2]}

        return {"type": edit_type, "params": parts[1:]}

    def get_edit_dir_name(parsed_spec: dict[str, object], version: str) -> str:
        if parsed_spec.get("edit_dir_name"):
            return str(parsed_spec["edit_dir_name"])
        t = str(parsed_spec["type"])
        if t == "quant_rtn":
            return f"quant_{parsed_spec['bits']}bit_{version}"
        if t == "fp8_quant":
            return f"fp8_{parsed_spec['format']}_{version}"
        if t == "magnitude_prune":
            pct = int(float(parsed_spec["ratio"]) * 100)
            return f"prune_{pct}pct_{version}"
        if t == "lowrank_svd":
            return f"svd_rank{parsed_spec['rank']}_{version}"
        return f"{t}_{version}"

    def _target_modules(scope: str) -> list[str]:
        if scope == "ffn":
            return ["mlp", "feed_forward", "ffn"]
        if scope == "all":
            return ["q_proj", "k_proj", "v_proj", "o_proj", "mlp", "gate", "up", "down"]
        return []

    def apply_quantization(model, bits: int, group_size: int, scope: str):
        import copy

        edited = copy.deepcopy(model)
        target_modules = _target_modules(scope)

        qmin = -(2 ** (bits - 1))
        qmax = max((2 ** (bits - 1)) - 1, 1)
        for name, param in edited.named_parameters():
            if not any(t in name.lower() for t in target_modules):
                continue
            if param.dim() < 2:
                continue
            orig_shape = param.shape
            flat = param.reshape(orig_shape[0], -1)
            in_features = flat.shape[1]
            eff_group_size = group_size if group_size > 0 else in_features
            if eff_group_size >= in_features:
                eff_group_size = in_features
            num_groups = (in_features + eff_group_size - 1) // eff_group_size
            pad = (num_groups * eff_group_size) - in_features
            if pad > 0:
                flat = torch.nn.functional.pad(flat, (0, pad))
            grouped = flat.reshape(orig_shape[0], num_groups, eff_group_size)
            max_abs = grouped.abs().amax(dim=-1, keepdim=True)
            scale = torch.clamp(max_abs / qmax, min=1e-10)
            quantized = torch.round(grouped / scale).clamp(qmin, qmax) * scale
            quantized = quantized.reshape(orig_shape[0], num_groups * eff_group_size)
            if pad > 0:
                quantized = quantized[:, :in_features]
            param.data = quantized.reshape(orig_shape).to(param.dtype)
        return edited

    def apply_pruning(model, ratio: float, scope: str):
        import copy

        edited = copy.deepcopy(model)
        target_modules = _target_modules(scope)

        for name, param in edited.named_parameters():
            if not any(t in name.lower() for t in target_modules):
                continue
            if param.dim() < 2:
                continue
            param_abs = param.detach().float().abs()
            flat = param_abs.view(-1)
            if flat.numel() > 10_000_000:
                sample_size = min(1_000_000, flat.numel())
                idx = torch.randint(0, flat.numel(), (sample_size,), device=flat.device)
                flat_for_quantile = flat[idx]
            else:
                flat_for_quantile = flat
            threshold = torch.quantile(flat_for_quantile, ratio)
            mask = param_abs > threshold
            param.data = (param * mask).to(param.dtype)
        return edited

    def apply_lowrank(model, rank: int, scope: str):
        import copy

        edited = copy.deepcopy(model)
        target_modules = _target_modules(scope)

        for name, param in edited.named_parameters():
            if not any(t in name.lower() for t in target_modules):
                continue
            if param.dim() != 2:
                continue
            if min(param.shape) <= rank:
                continue
            W = param.data.float()
            k = min(rank, min(W.shape))
            U, S, V = torch.svd_lowrank(W, q=k, niter=2)
            param.data = ((U * S) @ V.T).to(param.dtype)
        return edited

    def apply_fp8(model, format_type: str, scope: str):
        import copy

        edited = copy.deepcopy(model)
        target_modules = _target_modules(scope)

        dtype = None
        if format_type in {"e4m3", "e4m3fn", "e4m3fnuz"}:
            dtype = getattr(torch, "float8_e4m3fn", None)
        elif format_type in {"e5m2", "e5m2fn", "e5m2fnuz"}:
            dtype = getattr(torch, "float8_e5m2", None)

        def _quantize(tensor):
            if dtype is None:
                return tensor.to(torch.float16).to(tensor.dtype)
            return tensor.to(dtype).to(tensor.dtype)

        for name, param in edited.named_parameters():
            if not any(t in name.lower() for t in target_modules):
                continue
            if param.dim() < 2:
                continue
            param.data = _quantize(param.data)
        return edited

    created_count = 0
    failed_count = 0

    for spec_entry in edit_specs:
        if not isinstance(spec_entry, dict):
            continue
        spec_str = str(spec_entry.get("spec", ""))
        version = str(spec_entry.get("version", "clean"))

        parsed = parse_edit_spec(spec_str)
        if parsed.get("skip"):
            print(f"  Skip (tuned edit preset skipped): {spec_str}")
            continue
        if parsed.get("error"):
            raise ValueError(
                f"Tuned edit preset missing for {spec_str}: {parsed['error']}"
            )
        edit_dir_name = get_edit_dir_name(parsed, version)
        edit_path = model_output_dir / "models" / edit_dir_name

        if (edit_path / "config.json").exists():
            print(f"  Skip (exists): {edit_dir_name}")
            created_count += 1
            continue

        print(f"  Creating: {edit_dir_name}...")

        try:
            edit_path.mkdir(parents=True, exist_ok=True)

            t = str(parsed["type"])
            if t == "quant_rtn":
                edited_model = apply_quantization(
                    model,
                    int(parsed["bits"]),
                    int(parsed["group_size"]),
                    str(parsed["scope"]),
                )
            elif t == "magnitude_prune":
                edited_model = apply_pruning(
                    model, float(parsed["ratio"]), str(parsed["scope"])
                )
            elif t == "lowrank_svd":
                edited_model = apply_lowrank(
                    model, int(parsed["rank"]), str(parsed["scope"])
                )
            elif t == "fp8_quant":
                edited_model = apply_fp8(
                    model, str(parsed["format"]), str(parsed["scope"])
                )
            else:
                raise ValueError(f"Unknown edit type: {t}")

            edited_model.save_pretrained(edit_path, safe_serialization=True)
            tokenizer.save_pretrained(edit_path)

            del edited_model
            gc.collect()
            torch.cuda.empty_cache()

            print(f"    Saved: {edit_path}")
            created_count += 1

        except Exception as exc:
            print(f"    ERROR: {exc}", file=sys.stderr)
            failed_count += 1

    del model
    gc.collect()
    torch.cuda.empty_cache()

    print(f"Batch complete: {created_count} created, {failed_count} failed")

    if failed_count > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
