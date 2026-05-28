from __future__ import annotations

import argparse
import copy
import gc
import os
import sys
from pathlib import Path
from typing import Any

import torch

try:
    from edit_implementations import (
        apply_dense_lowrank_approximation,
        apply_dense_magnitude_prune,
        apply_fp8_dequantized_simulation,
        apply_rtn_dequantized_simulation,
    )
    from edit_metadata import build_validation_edit_metadata
    from edit_specs import (
        parse_edit_specs_json,
        resolve_batch_entry,
    )
    from runtime_tools import require_remote_code_opt_in
    from save_subject_artifact import save_edited_subject_artifact
    from validate_edit_artifact import validate_edit_artifact
except ImportError:  # pragma: no cover - direct module load under pytest
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from edit_implementations import (
        apply_dense_lowrank_approximation,
        apply_dense_magnitude_prune,
        apply_fp8_dequantized_simulation,
        apply_rtn_dequantized_simulation,
    )
    from edit_metadata import build_validation_edit_metadata
    from edit_specs import (
        parse_edit_specs_json,
        resolve_batch_entry,
    )
    from runtime_tools import require_remote_code_opt_in
    from save_subject_artifact import save_edited_subject_artifact
    from validate_edit_artifact import validate_edit_artifact
from transformers import AutoModelForCausalLM, AutoTokenizer

_BATCH_EDIT_STRATEGIES = {"reload", "deepcopy"}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create many evidence-pack edits from one baseline checkpoint."
    )
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--model-output-dir", required=True)
    parser.add_argument(
        "--edit-specs-json",
        required=True,
        help="JSON array of objects with keys: spec, version.",
    )
    return parser.parse_args(argv)


def _parse_edit_specs_json(raw_payload: str) -> list[object]:
    return parse_edit_specs_json(raw_payload)


def _configure_determinism() -> None:
    mode = os.environ.get("PACK_DETERMINISM", "").strip().lower()
    if mode == "strict":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    elif mode == "throughput":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.set_grad_enabled(False)


def _load_tokenizer(baseline_path: Path) -> Any:
    trust_remote_code = require_remote_code_opt_in("create_edits_batch.py")
    return AutoTokenizer.from_pretrained(
        baseline_path,
        trust_remote_code=trust_remote_code,
    )


def _load_baseline_model(baseline_path: Path) -> Any:
    trust_remote_code = require_remote_code_opt_in("create_edits_batch.py")
    return AutoModelForCausalLM.from_pretrained(
        baseline_path,
        dtype=torch.bfloat16,
        trust_remote_code=trust_remote_code,
        device_map="auto",
        low_cpu_mem_usage=True,
    )


def _batch_edit_strategy() -> str:
    raw = os.environ.get("PACK_BATCH_EDIT_STRATEGY", "reload").strip().lower()
    if raw not in _BATCH_EDIT_STRATEGIES:
        raise ValueError(
            "PACK_BATCH_EDIT_STRATEGY must be one of: "
            + ", ".join(sorted(_BATCH_EDIT_STRATEGIES))
        )
    return raw


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


def _build_edited_model_and_metadata(
    model: Any,
    parsed_spec: dict[str, object],
    *,
    clone_model: bool = True,
) -> tuple[Any, dict[str, object]]:
    edited = copy.deepcopy(model) if clone_model else model
    edit_type = str(parsed_spec["type"])
    if edit_type == "quant_rtn":
        bits = int(parsed_spec["bits"])
        group_size = int(parsed_spec["group_size"])
        scope = str(parsed_spec["scope"])
        stats = apply_rtn_dequantized_simulation(
            edited,
            bits=bits,
            group_size=group_size,
            scope=scope,
        )
        metadata = build_validation_edit_metadata(
            edit_type="quant_rtn",
            scope=scope,
            parameters={"bits": bits, "group_size": group_size},
            coverage=stats.coverage_payload(),
            extra={
                "quantization_mode": "rtn_dequantized_external_subject_simulation",
                "quantized_params": stats.edited_tensors,
            },
        )
        return edited, metadata
    if edit_type == "magnitude_prune":
        ratio = float(parsed_spec["ratio"])
        scope = str(parsed_spec["scope"])
        stats = apply_dense_magnitude_prune(edited, sparsity=ratio, scope=scope)
        metadata = build_validation_edit_metadata(
            edit_type="magnitude_prune",
            scope=scope,
            parameters={"target_sparsity": ratio},
            coverage=stats.coverage_payload(),
            extra={
                "target_sparsity": ratio,
                "actual_sparsity": stats.details.get("actual_sparsity"),
                "pruned_params": stats.edited_tensors,
            },
        )
        return edited, metadata
    if edit_type == "lowrank_svd":
        rank = int(parsed_spec["rank"])
        scope = str(parsed_spec["scope"])
        stats = apply_dense_lowrank_approximation(edited, rank=rank, scope=scope)
        metadata = build_validation_edit_metadata(
            edit_type="lowrank_svd",
            scope=scope,
            parameters={"rank": rank},
            coverage=stats.coverage_payload(),
            extra={
                "rank": rank,
                "modified_matrices": stats.edited_tensors,
                "avg_energy_retained": stats.details.get("avg_energy_retained"),
                "base_scope": stats.details.get("base_scope"),
                "layer_limit": stats.details.get("layer_limit"),
                "layer": stats.details.get("layer"),
            },
        )
        return edited, metadata
    if edit_type == "fp8_quant":
        format_type = str(parsed_spec["format"])
        scope = str(parsed_spec["scope"])
        stats = apply_fp8_dequantized_simulation(
            edited,
            format_type=format_type,
            scope=scope,
        )
        metadata = build_validation_edit_metadata(
            edit_type="fp8_quant",
            scope=scope,
            parameters={"format": format_type},
            coverage=stats.coverage_payload(),
            extra={
                "quantization_mode": "fp8_dequantized_external_subject_simulation",
                "format": format_type,
                "quantized_tensors": stats.edited_tensors,
                "avg_relative_error": stats.details.get("avg_relative_error"),
                "torch_fp8_dtype_available": stats.details.get(
                    "torch_fp8_dtype_available"
                ),
            },
        )
        return edited, metadata
    raise ValueError(f"Unknown edit type: {edit_type}")


def _clear_memory() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def _edit_artifact_complete(edit_path: Path) -> bool:
    return bool(validate_edit_artifact(edit_path, require_metadata=True))


def _create_edit_artifact(
    *,
    model: Any,
    tokenizer: Any,
    parsed_spec: dict[str, object],
    edit_path: Path,
    clone_model: bool = True,
) -> None:
    edited_model, metadata = _build_edited_model_and_metadata(
        model,
        parsed_spec,
        clone_model=clone_model,
    )
    save_edited_subject_artifact(
        model=edited_model,
        tokenizer=tokenizer,
        output_path=edit_path,
        metadata=metadata,
    )
    del edited_model
    _clear_memory()


def _process_spec_entry(
    *,
    spec_entry: object,
    model_output_dir: Path,
    model: Any,
    tokenizer: Any,
    clone_model: bool = True,
) -> tuple[int, int]:
    pending, created, failed = _resolve_pending_spec_entry(
        spec_entry=spec_entry,
        model_output_dir=model_output_dir,
    )
    if pending is None:
        return created, failed

    parsed, edit_path = pending
    try:
        _create_edit_artifact(
            model=model,
            tokenizer=tokenizer,
            parsed_spec=parsed,
            edit_path=edit_path,
            clone_model=clone_model,
        )
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"    ERROR: {exc}", file=sys.stderr)
        return 0, 1

    print(f"    Saved: {edit_path}")
    return 1, 0


def _resolve_pending_spec_entry(
    *,
    spec_entry: object,
    model_output_dir: Path,
) -> tuple[tuple[dict[str, object], Path] | None, int, int]:
    if not isinstance(spec_entry, dict):
        return None, 0, 0

    spec_str = str(spec_entry.get("spec", ""))
    version = str(spec_entry.get("version", "clean"))
    parsed_resolved = resolve_batch_entry(
        spec_entry=spec_entry,
        model_output_dir=model_output_dir,
    )
    if parsed_resolved is None:
        return None, 0, 0
    parsed = parsed_resolved.to_batch_payload()

    if parsed_resolved.skip:
        print(f"  Skip (tuned edit preset skipped): {spec_str}")
        return None, 0, 0
    if not parsed_resolved.selected:
        raise ValueError(
            f"Tuned edit preset missing for {spec_str}: {parsed_resolved.status}"
        )

    edit_dir_name = _get_edit_dir_name(parsed, version)
    edit_path = model_output_dir / "models" / edit_dir_name
    if _edit_artifact_complete(edit_path):
        print(f"  Skip (exists): {edit_dir_name}")
        return None, 1, 0

    print(f"  Creating: {edit_dir_name}...")
    return (parsed, edit_path), 0, 0


def _process_edit_specs(
    *,
    edit_specs: list[object],
    model_output_dir: Path,
    model: Any,
    tokenizer: Any,
    clone_model: bool = True,
) -> tuple[int, int]:
    created_count = 0
    failed_count = 0
    for spec_entry in edit_specs:
        created, failed = _process_spec_entry(
            spec_entry=spec_entry,
            model_output_dir=model_output_dir,
            model=model,
            tokenizer=tokenizer,
            clone_model=clone_model,
        )
        created_count += created
        failed_count += failed
    return created_count, failed_count


def _process_edit_specs_reloading_model(
    *,
    edit_specs: list[object],
    baseline_path: Path,
    model_output_dir: Path,
    tokenizer: Any,
) -> tuple[int, int]:
    created_count = 0
    failed_count = 0
    for spec_entry in edit_specs:
        model: Any | None = None
        try:
            pending, created, failed = _resolve_pending_spec_entry(
                spec_entry=spec_entry,
                model_output_dir=model_output_dir,
            )
            if pending is None:
                created_count += created
                failed_count += failed
                continue
            parsed, edit_path = pending
            model = _load_baseline_model(baseline_path)
            _create_edit_artifact(
                model=model,
                tokenizer=tokenizer,
                parsed_spec=parsed,
                edit_path=edit_path,
                clone_model=False,
            )
            print(f"    Saved: {edit_path}")
            created_count += 1
        except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
            print(f"    ERROR: {exc}", file=sys.stderr)
            failed_count += 1
        finally:
            if model is not None:
                del model
            _clear_memory()
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

    _configure_determinism()
    try:
        strategy = _batch_edit_strategy()
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Creating {len(edit_specs)} edits with strategy={strategy}...")

    model: Any | None = None
    try:
        tokenizer = _load_tokenizer(baseline_path)
        if strategy == "deepcopy":
            model = _load_baseline_model(baseline_path)
            print("Baseline loaded once. Creating edits via model deepcopy...")
            created_count, failed_count = _process_edit_specs(
                edit_specs=edit_specs,
                model_output_dir=model_output_dir,
                model=model,
                tokenizer=tokenizer,
                clone_model=True,
            )
        else:
            print(
                "Reloading baseline per edit to avoid model deepcopy memory spikes..."
            )
            created_count, failed_count = _process_edit_specs_reloading_model(
                edit_specs=edit_specs,
                baseline_path=baseline_path,
                model_output_dir=model_output_dir,
                tokenizer=tokenizer,
            )
    finally:
        if model is not None:
            del model
        _clear_memory()

    print(f"Batch complete: {created_count} created, {failed_count} failed")
    return 1 if failed_count > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
