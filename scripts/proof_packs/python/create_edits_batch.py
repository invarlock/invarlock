from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch

try:
    from runtime_tools import require_remote_code_opt_in
except ImportError:  # pragma: no cover - direct module load under pytest
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from runtime_tools import require_remote_code_opt_in
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
        except OSError:
            model_id = ""
    model_key = model_id or model_output_dir.name

    tuned_params_by_type: dict[str, object] = {}
    tuned_defaults: dict[str, object] = {}

    if tuned_path and Path(tuned_path).exists():
        try:
            data = json.loads(Path(tuned_path).read_text())
        except (OSError, json.JSONDecodeError):
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


def _parse_edit_specs_json(raw_payload: str) -> list[object]:
    try:
        edit_specs = json.loads(raw_payload)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid edit_specs JSON: {exc}") from exc

    if not isinstance(edit_specs, list):
        raise ValueError("edit_specs_json must be a JSON list")
    return edit_specs


def _configure_determinism() -> None:
    mode = os.environ.get("PACK_DETERMINISM", "").strip().lower()
    if mode == "strict":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    elif mode == "throughput":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.set_grad_enabled(False)


def _load_baseline_artifacts(baseline_path: Path) -> tuple[Any, Any]:
    trust_remote_code = require_remote_code_opt_in("create_edits_batch.py")
    tokenizer = AutoTokenizer.from_pretrained(
        baseline_path,
        trust_remote_code=trust_remote_code,
    )
    model = AutoModelForCausalLM.from_pretrained(
        baseline_path,
        dtype=torch.bfloat16,
        trust_remote_code=trust_remote_code,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    return tokenizer, model


def _clean_tuned_entry(
    edit_type: str,
    tuned_params_by_type: dict[str, object],
    tuned_defaults: dict[str, object],
) -> tuple[dict[str, object], str]:
    entry = tuned_params_by_type.get(edit_type) or tuned_defaults.get(edit_type) or {}
    if not isinstance(entry, dict):
        entry = {}
    status = str(entry.get("status") or "missing")
    return entry, status


def _parse_clean_edit_spec(
    edit_type: str,
    parts: list[str],
    tuned_params_by_type: dict[str, object],
    tuned_defaults: dict[str, object],
) -> dict[str, object]:
    entry, status = _clean_tuned_entry(edit_type, tuned_params_by_type, tuned_defaults)
    if status == "skipped":
        return {"type": edit_type, "skip": True, "reason": status}
    if status != "selected":
        return {"type": edit_type, "error": status}

    scope = entry.get("scope", parts[2] if len(parts) > 2 else "ffn")
    if edit_type == "quant_rtn":
        return {
            "type": "quant_rtn",
            "bits": int(entry.get("bits", 8)),
            "group_size": int(entry.get("group_size", 128)),
            "scope": scope,
            "edit_dir_name": entry.get("edit_dir_name"),
        }
    if edit_type == "fp8_quant":
        return {
            "type": "fp8_quant",
            "format": entry.get("format", "e4m3fn"),
            "scope": scope,
            "edit_dir_name": entry.get("edit_dir_name"),
        }
    if edit_type == "magnitude_prune":
        return {
            "type": "magnitude_prune",
            "ratio": float(entry.get("sparsity", 0.0)),
            "scope": scope,
            "edit_dir_name": entry.get("edit_dir_name"),
        }
    if edit_type == "lowrank_svd":
        return {
            "type": "lowrank_svd",
            "rank": int(entry.get("rank", 0)),
            "scope": scope,
            "edit_dir_name": entry.get("edit_dir_name"),
        }
    return {"type": edit_type, "params": parts[1:]}


def _parse_edit_spec(
    spec_str: str,
    tuned_params_by_type: dict[str, object],
    tuned_defaults: dict[str, object],
) -> dict[str, object]:
    parts = spec_str.split(":")
    edit_type = parts[0] if parts else ""

    if len(parts) > 1 and parts[1] == "clean":
        return _parse_clean_edit_spec(
            edit_type,
            parts,
            tuned_params_by_type,
            tuned_defaults,
        )

    if edit_type == "quant_rtn":
        return {
            "type": "quant_rtn",
            "bits": int(parts[1]),
            "group_size": int(parts[2]),
            "scope": parts[3],
        }
    if edit_type == "fp8_quant":
        return {"type": "fp8_quant", "format": parts[1], "scope": parts[2]}
    if edit_type == "magnitude_prune":
        return {
            "type": "magnitude_prune",
            "ratio": float(parts[1]),
            "scope": parts[2],
        }
    if edit_type == "lowrank_svd":
        return {"type": "lowrank_svd", "rank": int(parts[1]), "scope": parts[2]}
    return {"type": edit_type, "params": parts[1:]}


def _get_edit_dir_name(parsed_spec: dict[str, object], version: str) -> str:
    if parsed_spec.get("edit_dir_name"):
        return str(parsed_spec["edit_dir_name"])

    edit_type = str(parsed_spec["type"])
    if edit_type == "quant_rtn":
        return f"quant_{parsed_spec['bits']}bit_{version}"
    if edit_type == "fp8_quant":
        return f"fp8_{parsed_spec['format']}_{version}"
    if edit_type == "magnitude_prune":
        pct = int(float(parsed_spec["ratio"]) * 100)
        return f"prune_{pct}pct_{version}"
    if edit_type == "lowrank_svd":
        return f"svd_rank{parsed_spec['rank']}_{version}"
    return f"{edit_type}_{version}"


def _target_modules(scope: str) -> list[str]:
    if scope == "ffn":
        return ["mlp", "feed_forward", "ffn"]
    if scope == "all":
        return ["q_proj", "k_proj", "v_proj", "o_proj", "mlp", "gate", "up", "down"]
    return []


def _apply_quantization(model: Any, bits: int, group_size: int, scope: str) -> Any:
    edited = copy.deepcopy(model)
    target_modules = _target_modules(scope)

    qmin = -(2 ** (bits - 1))
    qmax = max((2 ** (bits - 1)) - 1, 1)
    for name, param in edited.named_parameters():
        if not any(target in name.lower() for target in target_modules):
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


def _apply_pruning(model: Any, ratio: float, scope: str) -> Any:
    edited = copy.deepcopy(model)
    target_modules = _target_modules(scope)

    for name, param in edited.named_parameters():
        if not any(target in name.lower() for target in target_modules):
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


def _apply_lowrank(model: Any, rank: int, scope: str) -> Any:
    edited = copy.deepcopy(model)
    target_modules = _target_modules(scope)

    for name, param in edited.named_parameters():
        if not any(target in name.lower() for target in target_modules):
            continue
        if param.dim() != 2:
            continue
        if min(param.shape) <= rank:
            continue
        weights = param.data.float()
        k = min(rank, min(weights.shape))
        left, singular, right = torch.svd_lowrank(weights, q=k, niter=2)
        param.data = ((left * singular) @ right.T).to(param.dtype)
    return edited


def _fp8_dtype(format_type: str) -> torch.dtype | None:
    if format_type in {"e4m3", "e4m3fn", "e4m3fnuz"}:
        return getattr(torch, "float8_e4m3fn", None)
    if format_type in {"e5m2", "e5m2fn", "e5m2fnuz"}:
        return getattr(torch, "float8_e5m2", None)
    return None


def _apply_fp8(model: Any, format_type: str, scope: str) -> Any:
    edited = copy.deepcopy(model)
    target_modules = _target_modules(scope)
    dtype = _fp8_dtype(format_type)

    for name, param in edited.named_parameters():
        if not any(target in name.lower() for target in target_modules):
            continue
        if param.dim() < 2:
            continue
        if dtype is None:
            param.data = param.data.to(torch.float16).to(param.dtype)
        else:
            param.data = param.data.to(dtype).to(param.dtype)
    return edited


def _build_edited_model(model: Any, parsed_spec: dict[str, object]) -> Any:
    edit_type = str(parsed_spec["type"])
    if edit_type == "quant_rtn":
        return _apply_quantization(
            model,
            int(parsed_spec["bits"]),
            int(parsed_spec["group_size"]),
            str(parsed_spec["scope"]),
        )
    if edit_type == "magnitude_prune":
        return _apply_pruning(
            model,
            float(parsed_spec["ratio"]),
            str(parsed_spec["scope"]),
        )
    if edit_type == "lowrank_svd":
        return _apply_lowrank(
            model,
            int(parsed_spec["rank"]),
            str(parsed_spec["scope"]),
        )
    if edit_type == "fp8_quant":
        return _apply_fp8(
            model,
            str(parsed_spec["format"]),
            str(parsed_spec["scope"]),
        )
    raise ValueError(f"Unknown edit type: {edit_type}")


def _clear_memory() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def _create_edit_artifact(
    *,
    model: Any,
    tokenizer: Any,
    parsed_spec: dict[str, object],
    edit_path: Path,
) -> None:
    edit_path.mkdir(parents=True, exist_ok=True)
    edited_model = _build_edited_model(model, parsed_spec)
    edited_model.save_pretrained(edit_path, safe_serialization=True)
    tokenizer.save_pretrained(edit_path)
    del edited_model
    _clear_memory()


def _process_spec_entry(
    *,
    spec_entry: object,
    model_output_dir: Path,
    model: Any,
    tokenizer: Any,
    tuned_params_by_type: dict[str, object],
    tuned_defaults: dict[str, object],
) -> tuple[int, int]:
    if not isinstance(spec_entry, dict):
        return 0, 0

    spec_str = str(spec_entry.get("spec", ""))
    version = str(spec_entry.get("version", "clean"))
    parsed = _parse_edit_spec(spec_str, tuned_params_by_type, tuned_defaults)

    if parsed.get("skip"):
        print(f"  Skip (tuned edit preset skipped): {spec_str}")
        return 0, 0
    if parsed.get("error"):
        raise ValueError(f"Tuned edit preset missing for {spec_str}: {parsed['error']}")

    edit_dir_name = _get_edit_dir_name(parsed, version)
    edit_path = model_output_dir / "models" / edit_dir_name
    if (edit_path / "config.json").exists():
        print(f"  Skip (exists): {edit_dir_name}")
        return 1, 0

    print(f"  Creating: {edit_dir_name}...")
    try:
        _create_edit_artifact(
            model=model,
            tokenizer=tokenizer,
            parsed_spec=parsed,
            edit_path=edit_path,
        )
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"    ERROR: {exc}", file=sys.stderr)
        return 0, 1

    print(f"    Saved: {edit_path}")
    return 1, 0


def _process_edit_specs(
    *,
    edit_specs: list[object],
    model_output_dir: Path,
    model: Any,
    tokenizer: Any,
    tuned_params_by_type: dict[str, object],
    tuned_defaults: dict[str, object],
) -> tuple[int, int]:
    created_count = 0
    failed_count = 0
    for spec_entry in edit_specs:
        created, failed = _process_spec_entry(
            spec_entry=spec_entry,
            model_output_dir=model_output_dir,
            model=model,
            tokenizer=tokenizer,
            tuned_params_by_type=tuned_params_by_type,
            tuned_defaults=tuned_defaults,
        )
        created_count += created
        failed_count += failed
    return created_count, failed_count


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    baseline_path = Path(args.baseline)
    model_output_dir = Path(args.model_output_dir)

    try:
        edit_specs = _parse_edit_specs_json(args.edit_specs_json)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    tuned_params_by_type, tuned_defaults, _, _ = _load_tuned_params(model_output_dir)
    print(f"Loading baseline model once for {len(edit_specs)} edits...")

    _configure_determinism()

    model: Any | None = None
    try:
        tokenizer, model = _load_baseline_artifacts(baseline_path)
        print(f"Baseline loaded. Creating {len(edit_specs)} edits...")
        created_count, failed_count = _process_edit_specs(
            edit_specs=edit_specs,
            model_output_dir=model_output_dir,
            model=model,
            tokenizer=tokenizer,
            tuned_params_by_type=tuned_params_by_type,
            tuned_defaults=tuned_defaults,
        )
    finally:
        if model is not None:
            del model
        _clear_memory()

    print(f"Batch complete: {created_count} created, {failed_count} failed")
    return 1 if failed_count > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
