#!/usr/bin/env python3
"""Convert one pinned Qwen3 checkpoint into a TensorRT-LLM engine."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections.abc import Iterator
from pathlib import Path

import modelopt.torch.quantization as mtq
import torch
from modelopt.torch.export import export_tensorrt_llm_checkpoint
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from examples.integrations.bounded_command import run_bounded_command
except ModuleNotFoundError as exc:
    if exc.name != "examples":
        raise
    from bounded_command import run_bounded_command  # type: ignore[no-redef]


def _canonical_tokenizer_contract(model: Path) -> bytes:
    tokenizer = AutoTokenizer.from_pretrained(
        model,
        local_files_only=True,
        trust_remote_code=False,
        use_fast=True,
    )
    if not tokenizer.is_fast or tokenizer.eos_token_id is None:
        raise RuntimeError("the pinned checkpoint lacks a usable fast tokenizer")
    backend = json.loads(tokenizer.backend_tokenizer.to_str())
    payload = {
        "add_special_tokens": False,
        "clean_up_tokenization_spaces": False,
        "eos_token_id": tokenizer.eos_token_id,
        "format_version": "invarlock/tensorrt-llm-tokenizer-contract-v1",
        "pad_token_id": (
            tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else tokenizer.eos_token_id
        ),
        "skip_special_tokens": True,
        "tokenizer_json": backend,
    }
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _calibration_batches(
    tokenizer: object,
    records_path: Path,
    *,
    batch_size: int = 8,
) -> Iterator[dict[str, torch.Tensor]]:
    records = json.loads(records_path.read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        raise RuntimeError("FP8 calibration records must be a non-empty JSON array")
    prompts: list[str] = []
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("prompt"), str):
            raise RuntimeError("each FP8 calibration record must contain a prompt")
        prompts.append(record["prompt"])
    for offset in range(0, len(prompts), batch_size):
        encoded = tokenizer(
            prompts[offset : offset + batch_size],
            add_special_tokens=False,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        yield {name: value.to("cuda") for name, value in encoded.items()}


def _convert(
    model: Path,
    checkpoint: Path,
    *,
    quantization: str,
    calibration_records: Path | None,
) -> None:
    loaded = AutoModelForCausalLM.from_pretrained(
        model,
        dtype=torch.bfloat16,
        local_files_only=True,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )
    if quantization == "fp8":
        if calibration_records is None:
            raise RuntimeError("FP8 conversion requires calibration records")
        tokenizer = AutoTokenizer.from_pretrained(
            model,
            local_files_only=True,
            trust_remote_code=False,
            use_fast=True,
        )
        loaded.to("cuda")

        def calibrate() -> None:
            with torch.inference_mode():
                for batch in _calibration_batches(tokenizer, calibration_records):
                    loaded(**batch)

        mtq.quantize(loaded, mtq.FP8_DEFAULT_CFG, forward_loop=calibrate)
    elif quantization != "none":
        raise RuntimeError(f"unsupported TensorRT quantization: {quantization}")
    export_tensorrt_llm_checkpoint(
        loaded,
        decoder_type="qwen",
        dtype=torch.bfloat16,
        export_dir=checkpoint,
        inference_tensor_parallel=1,
        inference_pipeline_parallel=1,
    )
    config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    observed = config.get("quantization", {}).get("quant_algo")
    expected = "FP8" if quantization == "fp8" else None
    if observed != expected:
        raise RuntimeError(
            "TensorRT checkpoint quantization metadata does not match the "
            f"requested {quantization} variant"
        )


def _build(checkpoint: Path, engine: Path, *, quantization: str) -> None:
    executable = shutil.which("trtllm-build")
    if executable is None:
        raise RuntimeError("trtllm-build is unavailable in the runtime image")
    run_bounded_command(
        [
            executable,
            "--checkpoint_dir",
            str(checkpoint),
            "--output_dir",
            str(engine),
            "--output_timing_cache",
            str(checkpoint.parent / "model.cache"),
            "--gemm_plugin",
            "disable" if quantization == "fp8" else "bfloat16",
            "--max_batch_size",
            "1",
            "--max_input_len",
            "1024",
            "--max_seq_len",
            "1026",
            "--max_num_tokens",
            "1026",
            "--opt_num_tokens",
            "1024",
        ],
        check=True,
        label="TensorRT-LLM engine build",
    )
    expected = {"config.json", "rank0.engine"}
    observed = {path.name for path in engine.iterdir() if path.is_file()}
    if observed != expected:
        raise RuntimeError("TensorRT-LLM produced an unexpected engine layout")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--tokenizer-contract", type=Path, required=True)
    parser.add_argument("--quantization", choices=("none", "fp8"), required=True)
    parser.add_argument("--calibration-records", type=Path)
    arguments = parser.parse_args(argv)
    for output in (
        arguments.checkpoint,
        arguments.engine,
        arguments.tokenizer_contract,
    ):
        if os.path.lexists(output):
            raise RuntimeError(f"output already exists: {output}")
    arguments.tokenizer_contract.write_bytes(
        _canonical_tokenizer_contract(arguments.model)
    )
    _convert(
        arguments.model,
        arguments.checkpoint,
        quantization=arguments.quantization,
        calibration_records=arguments.calibration_records,
    )
    _build(
        arguments.checkpoint,
        arguments.engine,
        quantization=arguments.quantization,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
