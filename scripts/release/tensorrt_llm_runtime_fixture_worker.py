#!/usr/bin/env python3
"""Container-side builder and probe for the TensorRT-LLM release fixture."""

from __future__ import annotations

import argparse
import json
import os
import stat
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final

MODEL_REPOSITORY: Final = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MODEL_REVISION: Final = "fe8a4ea1ffedaf415f4da2f062534de366a451e6"
TOKENIZER_FORMAT: Final = "invarlock/tensorrt-llm-tokenizer-contract-v1"
REQUEST_FORMAT: Final = "invarlock/tensorrt-llm-runner-request-v1"
RESPONSE_FORMAT: Final = "invarlock/tensorrt-llm-runner-response-v1"
RUNNER_PROTOCOL: Final = "invarlock/tensorrt-llm-runner-v1"
BUILD_RESULT_FORMAT: Final = "invarlock/tensorrt-llm-fixture-build-result-v1"
PROBE_RESULT_FORMAT: Final = "invarlock/tensorrt-llm-fixture-probe-result-v1"
CONVERTER: Final = Path(
    "/app/tensorrt_llm/examples/models/core/llama/convert_checkpoint.py"
)
RUNNER: Final = Path("/opt/invarlock/bin/tensorrt-llm-runner")
PROMPT: Final = "InvarLock"
_MAX_OUTPUT = 2 * 1024 * 1024


class FixtureWorkerError(RuntimeError):
    """Raised when the closed fixture worker contract cannot be satisfied."""


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _require_new_directory(path: Path) -> None:
    try:
        path.mkdir(mode=0o700, parents=False, exist_ok=False)
    except OSError as exc:
        raise FixtureWorkerError(
            "the output directory cannot be created safely"
        ) from exc


def _write_new(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except OSError as exc:
        raise FixtureWorkerError("a fixture output cannot be written safely") from exc


def _bounded_run(command: Sequence[str], *, timeout: int) -> bytes:
    try:
        completed = subprocess.run(  # noqa: S603 - fixed executable/argument contract
            list(command),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            check=False,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise FixtureWorkerError("the fixture subprocess failed") from exc
    if len(completed.stdout) > _MAX_OUTPUT or len(completed.stderr) > _MAX_OUTPUT:
        raise FixtureWorkerError("the fixture subprocess exceeded its output limit")
    if completed.returncode != 0:
        raise FixtureWorkerError(
            f"the fixture subprocess exited with status {completed.returncode}"
        )
    return completed.stdout


def _validate_local_model(path: Path) -> None:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise FixtureWorkerError("the local model snapshot is unavailable") from exc
    if not stat.S_ISDIR(metadata.st_mode) or path.is_symlink():
        raise FixtureWorkerError("the local model snapshot must be a directory")


def _tokenizer_contract(model: Path) -> bytes:
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            str(model),
            local_files_only=True,
            trust_remote_code=False,
            use_fast=True,
        )
        backend = tokenizer.backend_tokenizer
        tokenizer_json = json.loads(backend.to_str())
        eos_token_id = tokenizer.eos_token_id
        pad_token_id = tokenizer.pad_token_id
    except (AttributeError, OSError, TypeError, ValueError) as exc:
        raise FixtureWorkerError(
            "the pinned tokenizer cannot be loaded locally"
        ) from exc
    if not isinstance(tokenizer_json, Mapping) or not tokenizer_json:
        raise FixtureWorkerError("the pinned tokenizer JSON is invalid")
    if not isinstance(eos_token_id, int) or eos_token_id < 0:
        raise FixtureWorkerError("the pinned tokenizer has no valid EOS token")
    if pad_token_id is None:
        pad_token_id = eos_token_id
    if not isinstance(pad_token_id, int) or pad_token_id < 0:
        raise FixtureWorkerError("the pinned tokenizer has no valid pad token")
    return _canonical_json(
        {
            "add_special_tokens": False,
            "clean_up_tokenization_spaces": False,
            "eos_token_id": eos_token_id,
            "format_version": TOKENIZER_FORMAT,
            "pad_token_id": pad_token_id,
            "skip_special_tokens": True,
            "tokenizer_json": tokenizer_json,
        }
    )


def build_fixture(
    *, model: Path, output: Path, repository: str, revision: str
) -> dict[str, object]:
    """Build one TP=1 FP16 engine from an already local pinned snapshot."""

    if repository != MODEL_REPOSITORY or revision != MODEL_REVISION:
        raise FixtureWorkerError("the model source does not match the pinned fixture")
    _validate_local_model(model)
    _require_new_directory(output)
    checkpoint = output / "checkpoint"
    engine = output / "engine"
    tokenizer_path = output / "tokenizer.json"
    _write_new(tokenizer_path, _tokenizer_contract(model))
    _bounded_run(
        (
            sys.executable,
            str(CONVERTER),
            "--model_dir",
            str(model),
            "--output_dir",
            str(checkpoint),
            "--dtype",
            "float16",
            "--tp_size",
            "1",
        ),
        timeout=3600,
    )
    _bounded_run(
        (
            "trtllm-build",
            "--checkpoint_dir",
            str(checkpoint),
            "--output_dir",
            str(engine),
            "--gemm_plugin",
            "auto",
            "--max_batch_size",
            "1",
            "--max_input_len",
            "8",
            "--max_seq_len",
            "9",
            "--max_num_tokens",
            "9",
            "--opt_num_tokens",
            "8",
            "--output_timing_cache",
            "/tmp/invarlock-tensorrt-llm-model.cache",
        ),
        timeout=3600,
    )
    expected = {"config.json", "rank0.engine"}
    try:
        actual = {item.name for item in engine.iterdir()}
    except OSError as exc:
        raise FixtureWorkerError("the engine output cannot be inspected") from exc
    if actual != expected or not all((engine / name).is_file() for name in expected):
        raise FixtureWorkerError(
            "the engine builder did not produce the closed TP=1 layout"
        )
    return {
        "backend_version": "1.2.1",
        "format_version": BUILD_RESULT_FORMAT,
        "ok": True,
    }


def probe_fixture(*, engine: Path, tokenizer: Path) -> dict[str, object]:
    """Execute the fixed prompt through the installed closed runner protocol."""

    request = {
        "engine_bundle": str(engine),
        "format_version": REQUEST_FORMAT,
        "input_text": PROMPT,
        "protocol_version": RUNNER_PROTOCOL,
        "settings": {
            "allow_network": False,
            "batch_size": 1,
            "context_length": 8,
            "max_output_tokens": 1,
            "seed": 0,
            "timeout_seconds": 300,
        },
        "tokenizer_contract": str(tokenizer),
    }
    try:
        completed = subprocess.run(  # noqa: S603 - fixed installed runner path
            (str(RUNNER), "--invarlock-score-v1"),
            input=_canonical_json(request),
            capture_output=True,
            check=False,
            timeout=360,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise FixtureWorkerError("the fixture probe failed") from exc
    if (
        completed.returncode != 0
        or completed.stderr
        or not completed.stdout
        or len(completed.stdout) > _MAX_OUTPUT
    ):
        raise FixtureWorkerError("the fixture probe did not complete cleanly")
    try:
        response = json.loads(completed.stdout)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FixtureWorkerError("the fixture probe response is invalid") from exc
    if (
        not isinstance(response, dict)
        or set(response) != {"format_version", "output_text"}
        or response.get("format_version") != RESPONSE_FORMAT
        or not isinstance(response.get("output_text"), str)
    ):
        raise FixtureWorkerError("the fixture probe response has an unexpected schema")
    return {
        "format_version": PROBE_RESULT_FORMAT,
        "ok": True,
        "output_text": response["output_text"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--model", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--repository", required=True)
    build.add_argument("--revision", required=True)
    probe = subparsers.add_parser("probe")
    probe.add_argument("--engine", type=Path, required=True)
    probe.add_argument("--tokenizer", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "build":
            result = build_fixture(
                model=args.model,
                output=args.output,
                repository=args.repository,
                revision=args.revision,
            )
        else:
            result = probe_fixture(engine=args.engine, tokenizer=args.tokenizer)
    except (FixtureWorkerError, OSError, TypeError, ValueError) as exc:
        print(f"TensorRT-LLM fixture worker failed: {exc}", file=sys.stderr)
        return 2
    sys.stdout.buffer.write(_canonical_json(result) + b"\n")
    return 0


if __name__ == "__main__":  # pragma: no cover - container entrypoint
    raise SystemExit(main())
