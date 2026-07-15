"""Static contract authentication for TensorRT-LLM runtime inspection."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path

from invarlock.runtime_providers._tensorrt_llm_execution import (
    TensorRTLLMExecutionError,
    _PinnedFile,
)

_MAX_ENGINE_CONFIG_BYTES = 16 * 1024 * 1024
_MAX_TOKENIZER_CONTRACT_BYTES = 128 * 1024 * 1024
_MAX_JSON_DEPTH = 64
_MAX_JSON_ITEMS = 1_000_000
_IO_CHUNK_BYTES = 64 * 1024
_TOKENIZER_CONTRACT_FORMAT = "invarlock/tensorrt-llm-tokenizer-contract-v1"


def _strict_json_object(payload: bytes, *, label: str) -> dict[str, object]:
    try:
        text = payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise TensorRTLLMExecutionError(f"{label} is not UTF-8") from exc

    def reject_duplicates(items: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in items:
            if key in result:
                raise TensorRTLLMExecutionError(f"{label} contains a duplicate key")
            result[key] = value
        return result

    def reject_constant(value: str) -> object:
        raise TensorRTLLMExecutionError(
            f"{label} contains non-finite JSON number {value!r}"
        )

    def finite_float(value: str) -> float:
        parsed = float(value)
        if not math.isfinite(parsed):
            raise TensorRTLLMExecutionError(
                f"{label} contains a non-finite JSON number"
            )
        return parsed

    def validate_budget(value: object, *, depth: int = 0) -> int:
        if depth > _MAX_JSON_DEPTH:
            raise TensorRTLLMExecutionError(
                f"{label} exceeds the maximum nesting depth"
            )
        if isinstance(value, dict):
            count = len(value)
            for key, child in value.items():
                if not isinstance(key, str):
                    raise TensorRTLLMExecutionError(
                        f"{label} contains a non-text object key"
                    )
                count += validate_budget(child, depth=depth + 1)
        elif isinstance(value, list):
            count = len(value)
            for child in value:
                count += validate_budget(child, depth=depth + 1)
        else:
            count = 1
        if count > _MAX_JSON_ITEMS:
            raise TensorRTLLMExecutionError(f"{label} exceeds the maximum item count")
        return count

    try:
        value = json.loads(
            text,
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
            parse_float=finite_float,
        )
    except TensorRTLLMExecutionError:
        raise
    except (json.JSONDecodeError, RecursionError) as exc:
        raise TensorRTLLMExecutionError(f"{label} is not strict JSON") from exc
    if not isinstance(value, dict):
        raise TensorRTLLMExecutionError(f"{label} must be a JSON object")
    validate_budget(value)
    return value


def _read_pinned_bytes(pinned: _PinnedFile, *, label: str) -> bytes:
    expected_size = pinned.initial_stat.st_size
    os.lseek(pinned.descriptor, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    remaining = expected_size
    while remaining:
        chunk = os.read(pinned.descriptor, min(remaining, _IO_CHUNK_BYTES))
        if not chunk:
            raise TensorRTLLMExecutionError(f"{label} changed while being read")
        chunks.append(chunk)
        remaining -= len(chunk)
    if os.read(pinned.descriptor, 1):
        raise TensorRTLLMExecutionError(f"{label} grew while being read")
    pinned.recheck()
    return b"".join(chunks)


def _nonnegative_integer(value: object, *, label: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value <= (2**31) - 1
    ):
        raise TensorRTLLMExecutionError(f"{label} is outside the supported bound")
    return value


def _positive_integer(value: object, *, label: str, maximum: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 1 <= value <= maximum
    ):
        raise TensorRTLLMExecutionError(f"{label} is outside the supported bound")
    return value


def _validate_tokenizer_contract(payload: dict[str, object]) -> None:
    expected_fields = {
        "add_special_tokens",
        "clean_up_tokenization_spaces",
        "eos_token_id",
        "format_version",
        "pad_token_id",
        "skip_special_tokens",
        "tokenizer_json",
    }
    if set(payload) != expected_fields:
        raise TensorRTLLMExecutionError("tokenizer contract fields are not closed")
    if payload["format_version"] != _TOKENIZER_CONTRACT_FORMAT:
        raise TensorRTLLMExecutionError("tokenizer contract version is unsupported")
    if payload["add_special_tokens"] is not False:
        raise TensorRTLLMExecutionError(
            "tokenizer contract requires add_special_tokens=false"
        )
    if payload["skip_special_tokens"] is not True:
        raise TensorRTLLMExecutionError(
            "tokenizer contract requires skip_special_tokens=true"
        )
    if payload["clean_up_tokenization_spaces"] is not False:
        raise TensorRTLLMExecutionError(
            "tokenizer contract requires clean_up_tokenization_spaces=false"
        )
    _nonnegative_integer(payload["eos_token_id"], label="eos_token_id")
    _nonnegative_integer(payload["pad_token_id"], label="pad_token_id")
    tokenizer_json = payload["tokenizer_json"]
    if not isinstance(tokenizer_json, dict) or not tokenizer_json:
        raise TensorRTLLMExecutionError("tokenizer_json must be a non-empty object")
    version = tokenizer_json.get("version")
    model = tokenizer_json.get("model")
    if not isinstance(version, str) or not version or version != version.strip():
        raise TensorRTLLMExecutionError(
            "tokenizer_json must declare a non-empty version"
        )
    if not isinstance(model, dict) or not model:
        raise TensorRTLLMExecutionError("tokenizer_json must declare a non-empty model")
    model_type = model.get("type")
    if (
        not isinstance(model_type, str)
        or not model_type
        or model_type != model_type.strip()
    ):
        raise TensorRTLLMExecutionError(
            "tokenizer_json model must declare a non-empty type"
        )


def _validate_engine_contract(config: dict[str, object]) -> tuple[int, int, int]:
    if set(config) != {"build_config", "pretrained_config", "version"}:
        raise TensorRTLLMExecutionError("engine config fields are not closed")
    pretrained = config["pretrained_config"]
    build = config["build_config"]
    if not isinstance(pretrained, dict) or not isinstance(build, dict):
        raise TensorRTLLMExecutionError("engine config sections must be objects")
    mapping = pretrained.get("mapping")
    if not isinstance(mapping, dict):
        raise TensorRTLLMExecutionError("engine mapping must be an object")
    for name in ("world_size", "tp_size", "pp_size"):
        if mapping.get(name) != 1 or isinstance(mapping.get(name), bool):
            raise TensorRTLLMExecutionError(
                "the runner supports only single-rank engines"
            )
    if mapping.get("cp_size", 1) != 1 or isinstance(mapping.get("cp_size", 1), bool):
        raise TensorRTLLMExecutionError("the runner supports only single-rank engines")
    return (
        _positive_integer(
            build.get("max_batch_size"),
            label="engine max_batch_size",
            maximum=1024,
        ),
        _positive_integer(
            build.get("max_input_len"),
            label="engine max_input_len",
            maximum=1024 * 1024,
        ),
        _positive_integer(
            build.get("max_seq_len"),
            label="engine max_seq_len",
            maximum=2 * 1024 * 1024,
        ),
    )


@dataclass
class _ValidatedTensorRTLLMStaticInputs:
    tokenizer: _PinnedFile = field(repr=False)
    engine_config: _PinnedFile = field(repr=False)
    engine_max_batch_size: int
    engine_max_input_len: int
    engine_max_seq_len: int

    @property
    def tokenizer_sha256(self) -> str:
        return self.tokenizer.sha256

    def recheck(self) -> None:
        self.tokenizer.recheck()
        self.engine_config.recheck()

    def close(self) -> None:
        errors: list[Exception] = []
        for resource in (self.engine_config, self.tokenizer):
            try:
                resource.close()
            except Exception as exc:
                errors.append(exc)
        if errors:
            raise TensorRTLLMExecutionError(
                "TensorRT-LLM static inspection cleanup did not complete"
            ) from errors[0]


def _open_validated_tensorrt_llm_static_inputs(
    *,
    engine_bundle_path: Path,
    tokenizer_contract_path: Path,
) -> _ValidatedTensorRTLLMStaticInputs:
    tokenizer: _PinnedFile | None = None
    engine_config: _PinnedFile | None = None
    try:
        tokenizer = _PinnedFile.open(
            tokenizer_contract_path,
            expected_sha256=None,
            require_executable=False,
            max_bytes=_MAX_TOKENIZER_CONTRACT_BYTES,
        )
        engine_config = _PinnedFile.open(
            engine_bundle_path / "config.json",
            expected_sha256=None,
            require_executable=False,
            max_bytes=_MAX_ENGINE_CONFIG_BYTES,
        )
        _validate_tokenizer_contract(
            _strict_json_object(
                _read_pinned_bytes(tokenizer, label="tokenizer contract"),
                label="tokenizer contract",
            )
        )
        engine_limits = _validate_engine_contract(
            _strict_json_object(
                _read_pinned_bytes(engine_config, label="engine config"),
                label="engine config",
            )
        )
        return _ValidatedTensorRTLLMStaticInputs(
            tokenizer=tokenizer,
            engine_config=engine_config,
            engine_max_batch_size=engine_limits[0],
            engine_max_input_len=engine_limits[1],
            engine_max_seq_len=engine_limits[2],
        )
    except BaseException:
        cleanup_errors: list[Exception] = []
        for resource in (engine_config, tokenizer):
            if resource is None:
                continue
            try:
                resource.close()
            except Exception as exc:
                cleanup_errors.append(exc)
        if cleanup_errors:
            raise TensorRTLLMExecutionError(
                "TensorRT-LLM static inspection cleanup did not complete"
            ) from cleanup_errors[0]
        raise
