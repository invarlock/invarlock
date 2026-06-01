#!/usr/bin/env python3
"""Materialize a tiny PEFT LoRA-merged subject checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a deterministic tiny LoRA adapter, merge it into the base "
            "checkpoint, and write a HF-loadable subject directory."
        )
    )
    parser.add_argument(
        "--baseline",
        default="sshleifer/tiny-gpt2",
        help="Baseline model ID or local path. Default: sshleifer/tiny-gpt2.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where the merged subject checkpoint will be written.",
    )
    parser.add_argument(
        "--target-module",
        action="append",
        default=None,
        help=(
            "Module name for PEFT LoRA injection. Can be repeated. "
            "Default: c_attn for GPT-2 style attention projections."
        ),
    )
    parser.add_argument("--rank", type=int, default=2, help="LoRA rank.")
    parser.add_argument("--alpha", type=int, default=4, help="LoRA alpha.")
    parser.add_argument(
        "--seed",
        type=int,
        default=43,
        help="Seed for deterministic adapter materialization.",
    )
    parser.add_argument(
        "--lora-init-scale",
        type=float,
        default=1.0,
        help="Standard deviation for deterministic normal LoRA initialization.",
    )
    parser.add_argument(
        "--allow-network",
        action="store_true",
        help="Allow Hugging Face downloads instead of using local cache only.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace the output directory if it already exists.",
    )
    return parser.parse_args()


def _require_dependencies() -> tuple[Any, Any, Any, Any, Any]:
    try:
        import torch
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except (ImportError, ModuleNotFoundError) as exc:
        raise SystemExit(
            "Missing example dependency. Install PEFT in your example "
            "environment, for example: python -m pip install peft"
        ) from exc
    return (
        torch,
        AutoModelForCausalLM,
        AutoTokenizer,
        LoraConfig,
        TaskType,
        get_peft_model,
    )


def _prepare_output_dir(output_dir: Path, *, force: bool) -> None:
    if output_dir.exists():
        if not force:
            raise SystemExit(
                f"Output directory already exists: {output_dir}. "
                "Pass --force to replace it."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def _version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def _checkpoint_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file())


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _selected_weights(model: Any, target_modules: list[str]) -> dict[str, Any]:
    selected: dict[str, Any] = {}
    for name, parameter in model.named_parameters():
        if any(f".{target}." in f".{name}." for target in target_modules):
            selected[name] = parameter.detach().cpu().clone()
    return selected


def _initialize_lora_weights(
    peft_model: Any,
    *,
    torch: Any,
    lora_init_scale: float,
    seed: int,
) -> int:
    initialized = 0
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    with torch.no_grad():
        for module in peft_model.modules():
            lora_a = getattr(module, "lora_A", None)
            lora_b = getattr(module, "lora_B", None)
            if lora_a is None or lora_b is None:
                continue
            for layer in lora_a.values():
                layer.weight.copy_(
                    torch.randn(
                        layer.weight.shape,
                        dtype=layer.weight.dtype,
                        device=layer.weight.device,
                        generator=generator,
                    )
                    * float(lora_init_scale)
                )
                initialized += 1
            for layer in lora_b.values():
                layer.weight.copy_(
                    torch.randn(
                        layer.weight.shape,
                        dtype=layer.weight.dtype,
                        device=layer.weight.device,
                        generator=generator,
                    )
                    * float(lora_init_scale)
                )
                initialized += 1
    return initialized


def _delta_summary(before: dict[str, Any], merged_model: Any) -> dict[str, Any]:
    merged_state = dict(merged_model.named_parameters())
    max_abs_delta = 0.0
    changed_parameters = 0
    checked_parameters = 0
    by_parameter: dict[str, float] = {}
    for name, before_value in before.items():
        after_value = merged_state.get(name)
        if after_value is None:
            continue
        checked_parameters += 1
        delta = (after_value.detach().cpu().to(before_value.dtype) - before_value).abs()
        parameter_delta = float(delta.max().item())
        by_parameter[name] = parameter_delta
        max_abs_delta = max(max_abs_delta, parameter_delta)
        if parameter_delta > 0.0:
            changed_parameters += 1
    return {
        "checked_parameters": checked_parameters,
        "changed_parameters": changed_parameters,
        "max_abs_delta": max_abs_delta,
        "by_parameter": by_parameter,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    target_modules = args.target_module or ["c_attn"]
    local_files_only = not bool(args.allow_network)

    torch, auto_model, auto_tokenizer, lora_config, task_type, get_peft_model = (
        _require_dependencies()
    )

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    _prepare_output_dir(output_dir, force=bool(args.force))

    tokenizer = auto_tokenizer.from_pretrained(
        args.baseline,
        local_files_only=local_files_only,
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = auto_model.from_pretrained(
        args.baseline,
        local_files_only=local_files_only,
    )
    before = _selected_weights(model, target_modules)
    if not before:
        raise SystemExit(
            "No baseline parameters matched the target module list: "
            + ", ".join(target_modules)
        )

    config = lora_config(
        task_type=task_type.CAUSAL_LM,
        r=int(args.rank),
        lora_alpha=int(args.alpha),
        lora_dropout=0.0,
        target_modules=target_modules,
        fan_in_fan_out=True,
        bias="none",
    )
    peft_model = get_peft_model(model, config)
    initialized_layers = _initialize_lora_weights(
        peft_model,
        torch=torch,
        lora_init_scale=float(args.lora_init_scale),
        seed=int(args.seed),
    )
    if initialized_layers == 0:
        raise SystemExit("PEFT model did not expose LoRA layers to initialize.")

    merged_model = peft_model.merge_and_unload()
    delta_summary = _delta_summary(before, merged_model)
    if float(delta_summary["max_abs_delta"]) <= 0.0:
        raise SystemExit("Merged subject checkpoint did not change the target weights.")

    merged_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    files = {
        str(path.relative_to(output_dir)): _sha256(path)
        for path in _checkpoint_files(output_dir)
    }
    timestamp = datetime.now(UTC).replace(microsecond=0).isoformat()
    summary = {
        "format_version": "integration-example-edit-summary-v1",
        "created_at": timestamp,
        "baseline": str(args.baseline),
        "subject_checkpoint_path": str(output_dir),
        "external_edit_type": "lora_merge",
        "toolchain": "peft",
        "toolchain_versions": {
            "peft": _version("peft"),
            "torch": _version("torch"),
            "transformers": _version("transformers"),
        },
        "lora": {
            "rank": int(args.rank),
            "alpha": int(args.alpha),
            "dropout": 0.0,
            "target_modules": target_modules,
            "fan_in_fan_out": True,
            "initialized_layers": initialized_layers,
            "init_distribution": "normal",
            "init_scale": float(args.lora_init_scale),
        },
        "delta_summary": delta_summary,
        "files": files,
    }
    checkpoint_refs = {
        "format_version": "checkpoint-refs-v1",
        "lane_id": "tiny-gpt2-peft-lora-merge",
        "created_at": timestamp,
        "baseline": {
            "kind": "hf_model_ref",
            "model_id": str(args.baseline),
            "purpose": "Baseline causal LM reference for the PEFT LoRA example.",
        },
        "subject": {
            "kind": "byoe_checkpoint_ref",
            "path": str(output_dir),
            "edit_workflow": "External PEFT LoRA merge",
            "external_edit_type": "lora_merge",
            "purpose": (
                "HF-loadable checkpoint produced by merging a deterministic "
                "PEFT LoRA adapter into the baseline model."
            ),
        },
        "artifacts": {
            "external_edit_summary": "external_edit_summary.json",
            "files": files,
        },
    }
    _write_json(output_dir / "external_edit_summary.json", summary)
    _write_json(output_dir / "checkpoint_refs.json", checkpoint_refs)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
