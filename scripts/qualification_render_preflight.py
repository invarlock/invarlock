#!/usr/bin/env python3
"""Authenticate provider-specific rendering before expensive qualification runs.

This maintainer tool intentionally sits outside the runtime-provider ABI.  It
loads a canonical schedule, authenticates the exact tokenizer or processor
contract used by a candidate runtime, and rejects inputs that would truncate or
could not reproduce their byte-exact expected output.  GGUF runtimes expose no
separate tokenizer executable, so their supported path is a small signed live
prefix statement produced through the pinned runtime.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib
import io
import json
import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

from invarlock.core.runtime_provider import EvaluationRecord, RuntimeBehavioralSchedule
from invarlock.core.runtime_provider.behavioral_schedule import (
    MAX_RUNTIME_BEHAVIORAL_SCHEDULE_BYTES,
    build_runtime_behavioral_schedule,
)
from invarlock.evidence_pack_integrity import public_key_fingerprint
from invarlock.evidence_pack_json import parse_json_bytes, read_regular_file_bytes
from invarlock.output_text_contract import exact_match_output_text

FORMAT_VERSION = "invarlock/qualification-render-preflight-v1"
GGUF_PREFIX_FORMAT = "invarlock/qualification-gguf-prefix-v1"
_SHA256 = re.compile(r"^[a-f0-9]{64}$")
_PREFIXED_SHA256 = re.compile(r"^sha256:[a-f0-9]{64}$")
_MAX_CONTRACT_BYTES = 64 * 1024 * 1024
_MAX_IMAGE_BYTES = 64 * 1024 * 1024


class QualificationRenderPreflightError(ValueError):
    """Raised when exact rendering is not qualified for execution."""


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _digest(value: str, *, label: str, prefixed: bool = False) -> str:
    pattern = _PREFIXED_SHA256 if prefixed else _SHA256
    if pattern.fullmatch(value) is None:
        qualifier = "sha256:" if prefixed else ""
        raise QualificationRenderPreflightError(
            f"{label} must be a lowercase {qualifier}SHA-256 digest"
        )
    return value


def _positive(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise QualificationRenderPreflightError(f"{label} must be a positive integer")
    return value


def load_bound_schedule(
    path: Path,
    *,
    expected_file_sha256: str,
    expected_schedule_sha256: str,
) -> tuple[RuntimeBehavioralSchedule, str]:
    """Load one immutable schedule snapshot and bind both file and semantic hashes."""

    expected_file = _digest(
        expected_file_sha256, label="schedule file digest", prefixed=True
    )
    expected_schedule = _digest(expected_schedule_sha256, label="schedule digest")
    payload = read_regular_file_bytes(
        path,
        label="qualification schedule",
        max_bytes=MAX_RUNTIME_BEHAVIORAL_SCHEDULE_BYTES,
    )
    observed_file = "sha256:" + _sha256(payload)
    if observed_file != expected_file:
        raise QualificationRenderPreflightError(
            "qualification schedule file digest does not match"
        )
    decoded = parse_json_bytes(payload, label="qualification schedule")
    if not isinstance(decoded, Mapping):
        raise QualificationRenderPreflightError(
            "qualification schedule must be a JSON object"
        )
    schedule = build_runtime_behavioral_schedule(decoded)
    if schedule.schedule_sha256 != expected_schedule:
        raise QualificationRenderPreflightError(
            "qualification schedule semantic digest does not match"
        )
    return schedule, observed_file


def _single_token_ids(value: object, *, label: str) -> list[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes, bytearray))
        and len(value) == 1
        and isinstance(value[0], Sequence)
        and not isinstance(value[0], (str, bytes, bytearray))
    ):
        value = value[0]
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise QualificationRenderPreflightError(f"{label} returned invalid token IDs")
    result = list(value)
    if any(isinstance(item, bool) or not isinstance(item, int) for item in result):
        raise QualificationRenderPreflightError(f"{label} returned invalid token IDs")
    return result


def _tokenize(
    tokenizer: object,
    text: str,
    *,
    add_special_tokens: bool,
    label: str,
) -> list[int]:
    if not callable(tokenizer):
        raise QualificationRenderPreflightError("tokenizer is not callable")
    try:
        encoded = tokenizer(
            text,
            add_special_tokens=add_special_tokens,
            truncation=False,
            return_tensors=None,
        )
    except Exception as exc:
        raise QualificationRenderPreflightError(f"{label} tokenization failed") from exc
    if not isinstance(encoded, Mapping) or "input_ids" not in encoded:
        raise QualificationRenderPreflightError(f"{label} returned no input_ids")
    return _single_token_ids(encoded["input_ids"], label=label)


def _decode(
    tokenizer: object,
    token_ids: Sequence[int],
    *,
    skip_special_tokens: bool,
    label: str,
) -> str:
    decode = getattr(tokenizer, "decode", None)
    if not callable(decode):
        raise QualificationRenderPreflightError("tokenizer has no decode API")
    try:
        value = decode(
            list(token_ids),
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=False,
        )
    except Exception as exc:
        raise QualificationRenderPreflightError(f"{label} decoding failed") from exc
    if not isinstance(value, str):
        raise QualificationRenderPreflightError(f"{label} decoding returned non-text")
    return value


def _text_record(record: EvaluationRecord, *, profile: str) -> tuple[str, str, str]:
    parts = record.input_parts
    if len(parts) != 1 or parts[0].kind != "text" or parts[0].role != "prompt":
        raise QualificationRenderPreflightError(
            f"{profile} requires one prompt text part"
        )
    expected = record.expected_output
    if not isinstance(expected, str) or not expected.strip():
        raise QualificationRenderPreflightError(
            f"{profile} requires non-empty expected output"
        )
    return record.record_id, record.input_text, expected


def _result_integer(
    item: Mapping[str, object], field: str, *, default: int | None = None
) -> int:
    value = item.get(field, default)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise QualificationRenderPreflightError(
            f"internal preflight result {field} is invalid"
        )
    return value


def _result(
    *,
    profile: str,
    schedule: RuntimeBehavioralSchedule,
    schedule_file_sha256: str,
    bindings: Mapping[str, object],
    record_results: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    schedule_file_sha256 = _digest(
        schedule_file_sha256, label="schedule file digest", prefixed=True
    )
    body: dict[str, object] = {
        "format_version": FORMAT_VERSION,
        "ok": True,
        "profile": profile,
        "schedule_sha256": schedule.schedule_sha256,
        "schedule_file_sha256": schedule_file_sha256,
        "record_count": len(schedule.records),
        "bindings": dict(bindings),
        "record_results_sha256": _sha256(_canonical_json(list(record_results))),
        "maximum_prompt_tokens": max(
            _result_integer(item, "prompt_tokens") for item in record_results
        ),
        "maximum_expected_tokens": max(
            _result_integer(item, "expected_tokens", default=0)
            for item in record_results
        ),
    }
    return {**body, "result_sha256": _sha256(_canonical_json(body))}


def preflight_hf_text(
    schedule: RuntimeBehavioralSchedule,
    *,
    schedule_file_sha256: str,
    tokenizer: object,
    expected_tokenizer_sha256: str,
    context_length: int,
    max_output_tokens: int,
    metric: str,
    tokenizer_digest: Callable[[object], str],
) -> dict[str, object]:
    """Replay the exact built-in HF prompt and expected-continuation contract."""

    if schedule.task != "text_causal":
        raise QualificationRenderPreflightError(
            "HF text preflight requires text_causal"
        )
    if metric not in {"exact_match", "normalized_nll_per_utf8_byte"}:
        raise QualificationRenderPreflightError("HF text metric is unsupported")
    context_length = _positive(context_length, label="context_length")
    max_output_tokens = _positive(max_output_tokens, label="max_output_tokens")
    expected_digest = _digest(
        expected_tokenizer_sha256, label="tokenizer contract digest"
    )
    try:
        observed_digest = tokenizer_digest(tokenizer)
    except Exception as exc:
        raise QualificationRenderPreflightError(
            "HF tokenizer contract could not be authenticated"
        ) from exc
    if observed_digest != expected_digest:
        raise QualificationRenderPreflightError("HF tokenizer contract digest mismatch")

    results: list[dict[str, object]] = []
    for record in schedule.records:
        record_id, prompt, expected = _text_record(record, profile="HF text")
        prompt_ids = _tokenize(
            tokenizer,
            prompt,
            add_special_tokens=True,
            label=f"record {record_id!r} prompt",
        )
        if not prompt_ids:
            raise QualificationRenderPreflightError(
                f"record {record_id!r} prompt tokenization is empty"
            )
        if len(prompt_ids) > context_length:
            raise QualificationRenderPreflightError(
                f"record {record_id!r} prompt would be truncated"
            )
        expected_ids = _tokenize(
            tokenizer,
            expected,
            add_special_tokens=False,
            label=f"record {record_id!r} expected output",
        )
        if not expected_ids or len(expected_ids) > max_output_tokens:
            raise QualificationRenderPreflightError(
                f"record {record_id!r} expected output exceeds the token bound"
            )
        if metric == "normalized_nll_per_utf8_byte":
            decoded_prompt = _decode(
                tokenizer,
                prompt_ids,
                skip_special_tokens=False,
                label=f"record {record_id!r} prompt",
            )
            decoded_combined = _decode(
                tokenizer,
                [*prompt_ids, *expected_ids],
                skip_special_tokens=False,
                label=f"record {record_id!r} continuation",
            )
            if decoded_combined != decoded_prompt + expected:
                raise QualificationRenderPreflightError(
                    f"record {record_id!r} expected output is not an exact tokenizer "
                    "continuation"
                )
        else:
            decoded_expected = _decode(
                tokenizer,
                expected_ids,
                skip_special_tokens=True,
                label=f"record {record_id!r} expected output",
            )
            if decoded_expected != expected:
                raise QualificationRenderPreflightError(
                    f"record {record_id!r} exact output does not round-trip"
                )
        results.append(
            {
                "record_id": record_id,
                "prompt_tokens": len(prompt_ids),
                "expected_tokens": len(expected_ids),
            }
        )
    return _result(
        profile="hf_text",
        schedule=schedule,
        schedule_file_sha256=schedule_file_sha256,
        bindings={
            "metric": metric,
            "tokenizer_contract_sha256": observed_digest,
            "context_length": context_length,
            "max_output_tokens": max_output_tokens,
        },
        record_results=results,
    )


def _engine_limits(config: Mapping[str, object]) -> tuple[int, int]:
    if set(config) != {"build_config", "pretrained_config", "version"}:
        raise QualificationRenderPreflightError("TensorRT engine config is not closed")
    build = config.get("build_config")
    pretrained = config.get("pretrained_config")
    if not isinstance(build, Mapping) or not isinstance(pretrained, Mapping):
        raise QualificationRenderPreflightError("TensorRT engine config is invalid")
    mapping = pretrained.get("mapping")
    if not isinstance(mapping, Mapping) or any(
        mapping.get(name, 1) != 1
        for name in ("world_size", "tp_size", "pp_size", "cp_size")
    ):
        raise QualificationRenderPreflightError(
            "TensorRT preflight requires a single-rank engine"
        )
    maximum_input = _positive(
        build.get("max_input_len"),
        label="engine max_input_len",
    )
    maximum_sequence = _positive(
        build.get("max_seq_len"),
        label="engine max_seq_len",
    )
    return maximum_input, maximum_sequence


def preflight_tensorrt(
    schedule: RuntimeBehavioralSchedule,
    *,
    schedule_file_sha256: str,
    tokenizer: object,
    tokenizer_contract_sha256: str,
    engine_config: Mapping[str, object],
    engine_config_sha256: str,
    context_length: int,
    max_output_tokens: int,
) -> dict[str, object]:
    """Replay TensorRT-LLM's no-special-token prompt contract and engine limits."""

    if schedule.task != "text_causal":
        raise QualificationRenderPreflightError(
            "TensorRT preflight requires text_causal"
        )
    tokenizer_digest = _digest(
        tokenizer_contract_sha256, label="TensorRT tokenizer contract digest"
    )
    config_digest = _digest(engine_config_sha256, label="TensorRT engine config digest")
    context_length = _positive(context_length, label="context_length")
    max_output_tokens = _positive(max_output_tokens, label="max_output_tokens")
    maximum_input, maximum_sequence = _engine_limits(engine_config)
    if context_length > maximum_input:
        raise QualificationRenderPreflightError(
            "context_length exceeds the TensorRT engine input limit"
        )
    if context_length + max_output_tokens > maximum_sequence:
        raise QualificationRenderPreflightError(
            "context and output bounds exceed the TensorRT engine sequence limit"
        )

    encode = getattr(tokenizer, "encode", None)
    if not callable(encode):
        raise QualificationRenderPreflightError("TensorRT tokenizer has no encode API")
    results: list[dict[str, object]] = []
    for record in schedule.records:
        record_id, prompt, expected = _text_record(record, profile="TensorRT")
        try:
            prompt_ids = _single_token_ids(
                encode(prompt, add_special_tokens=False),
                label=f"record {record_id!r} prompt",
            )
            expected_ids = _single_token_ids(
                encode(expected, add_special_tokens=False),
                label=f"record {record_id!r} expected output",
            )
        except QualificationRenderPreflightError:
            raise
        except Exception as exc:
            raise QualificationRenderPreflightError(
                f"record {record_id!r} TensorRT tokenization failed"
            ) from exc
        if not prompt_ids or len(prompt_ids) > context_length:
            raise QualificationRenderPreflightError(
                f"record {record_id!r} prompt exceeds the TensorRT context"
            )
        if not expected_ids or len(expected_ids) > max_output_tokens:
            raise QualificationRenderPreflightError(
                f"record {record_id!r} expected output exceeds the token bound"
            )
        if (
            _decode(
                tokenizer,
                expected_ids,
                skip_special_tokens=True,
                label=f"record {record_id!r} expected output",
            )
            != expected
        ):
            raise QualificationRenderPreflightError(
                f"record {record_id!r} exact output does not round-trip"
            )
        results.append(
            {
                "record_id": record_id,
                "prompt_tokens": len(prompt_ids),
                "expected_tokens": len(expected_ids),
            }
        )
    return _result(
        profile="tensorrt_llm",
        schedule=schedule,
        schedule_file_sha256=schedule_file_sha256,
        bindings={
            "tokenizer_contract_sha256": tokenizer_digest,
            "engine_config_sha256": config_digest,
            "context_length": context_length,
            "max_output_tokens": max_output_tokens,
            "engine_max_input_len": maximum_input,
            "engine_max_seq_len": maximum_sequence,
        },
        record_results=results,
    )


def preflight_multimodal(
    schedule: RuntimeBehavioralSchedule,
    *,
    schedule_file_sha256: str,
    processor: object,
    expected_processor_sha256: str,
    context_length: int,
    max_output_tokens: int,
    processor_digest: Callable[[object], str],
    image_resolver: Callable[[EvaluationRecord], object],
) -> dict[str, object]:
    """Replay the exact multimodal chat rendering without loading model weights."""

    if schedule.task != "vision_text_generation":
        raise QualificationRenderPreflightError(
            "multimodal preflight requires vision_text_generation"
        )
    expected_digest = _digest(
        expected_processor_sha256, label="processor contract digest"
    )
    try:
        observed_digest = processor_digest(processor)
    except Exception as exc:
        raise QualificationRenderPreflightError(
            "multimodal processor contract could not be authenticated"
        ) from exc
    if observed_digest != expected_digest:
        raise QualificationRenderPreflightError(
            "multimodal processor contract digest mismatch"
        )
    context_length = _positive(context_length, label="context_length")
    max_output_tokens = _positive(max_output_tokens, label="max_output_tokens")
    apply_template = getattr(processor, "apply_chat_template", None)
    if not callable(apply_template) or not callable(processor):
        raise QualificationRenderPreflightError(
            "multimodal processor APIs are unavailable"
        )
    tokenizer = getattr(processor, "tokenizer", None)
    results: list[dict[str, object]] = []
    for record in schedule.records:
        parts = record.input_parts
        prompts = [
            part for part in parts if part.kind == "text" and part.role == "prompt"
        ]
        images = [
            part for part in parts if part.kind == "content" and part.role == "image"
        ]
        if len(parts) != 2 or len(prompts) != 1 or len(images) != 1:
            raise QualificationRenderPreflightError(
                "multimodal record requires exactly one prompt and one image"
            )
        prompt = prompts[0].text
        expected = record.expected_output
        if not isinstance(prompt, str) or not isinstance(expected, str) or not expected:
            raise QualificationRenderPreflightError(
                "multimodal record requires prompt text and expected output"
            )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        try:
            rendered = apply_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except Exception as exc:
            raise QualificationRenderPreflightError(
                f"record {record.record_id!r} chat rendering failed"
            ) from exc
        if not isinstance(rendered, str) or not rendered:
            raise QualificationRenderPreflightError(
                f"record {record.record_id!r} chat rendering returned no text"
            )
        image = image_resolver(record)
        try:
            encoded = processor(
                text=rendered,
                images=image,
                return_tensors="pt",
                truncation=False,
            )
        except Exception as exc:
            raise QualificationRenderPreflightError(
                f"record {record.record_id!r} processor encoding failed"
            ) from exc
        finally:
            close = getattr(image, "close", None)
            if callable(close):
                close()
        if not isinstance(encoded, Mapping) or "input_ids" not in encoded:
            raise QualificationRenderPreflightError(
                f"record {record.record_id!r} processor returned no input_ids"
            )
        prompt_ids = _single_token_ids(
            encoded["input_ids"], label=f"record {record.record_id!r} multimodal input"
        )
        if not prompt_ids or len(prompt_ids) > context_length:
            raise QualificationRenderPreflightError(
                f"record {record.record_id!r} multimodal input would be truncated"
            )
        expected_ids = _tokenize(
            tokenizer,
            expected,
            add_special_tokens=False,
            label=f"record {record.record_id!r} expected output",
        )
        if not expected_ids or len(expected_ids) > max_output_tokens:
            raise QualificationRenderPreflightError(
                f"record {record.record_id!r} expected output exceeds the token bound"
            )
        if (
            _decode(
                tokenizer,
                expected_ids,
                skip_special_tokens=True,
                label=f"record {record.record_id!r} expected output",
            )
            != expected
        ):
            raise QualificationRenderPreflightError(
                f"record {record.record_id!r} exact output does not round-trip"
            )
        results.append(
            {
                "record_id": record.record_id,
                "prompt_tokens": len(prompt_ids),
                "expected_tokens": len(expected_ids),
                "rendered_sha256": _sha256(rendered.encode("utf-8")),
            }
        )
    return _result(
        profile="multimodal",
        schedule=schedule,
        schedule_file_sha256=schedule_file_sha256,
        bindings={
            "processor_contract_sha256": observed_digest,
            "context_length": context_length,
            "max_output_tokens": max_output_tokens,
        },
        record_results=results,
    )


def verify_gguf_prefix(
    schedule: RuntimeBehavioralSchedule,
    *,
    schedule_file_sha256: str,
    statement: Mapping[str, object],
    signature: bytes,
    public_key: ed25519.Ed25519PublicKey,
    expected_signer_fingerprint: str,
    minimum_exact_matches: int,
) -> dict[str, object]:
    """Verify a signed live GGUF prefix against the first schedule records."""

    expected_fields = {
        "format_version",
        "schedule_sha256",
        "artifact_sha256",
        "backend_binary_sha256",
        "runtime_image_digest",
        "prefix_record_count",
        "records",
    }
    if (
        set(statement) != expected_fields
        or statement.get("format_version") != GGUF_PREFIX_FORMAT
    ):
        raise QualificationRenderPreflightError("GGUF prefix statement is invalid")
    fingerprint = public_key_fingerprint(public_key)
    if fingerprint != _digest(
        expected_signer_fingerprint,
        label="GGUF prefix signer fingerprint",
        prefixed=True,
    ):
        raise QualificationRenderPreflightError("GGUF prefix signer is not pinned")
    try:
        public_key.verify(signature, _canonical_json(statement))
    except InvalidSignature as exc:
        raise QualificationRenderPreflightError(
            "GGUF prefix signature is invalid"
        ) from exc
    if statement.get("schedule_sha256") != schedule.schedule_sha256:
        raise QualificationRenderPreflightError(
            "GGUF prefix does not bind the qualification schedule"
        )
    _digest(str(statement.get("artifact_sha256")), label="GGUF artifact digest")
    _digest(str(statement.get("backend_binary_sha256")), label="GGUF backend digest")
    _digest(
        str(statement.get("runtime_image_digest")),
        label="GGUF runtime image digest",
        prefixed=True,
    )
    count = statement.get("prefix_record_count")
    records = statement.get("records")
    if (
        isinstance(count, bool)
        or not isinstance(count, int)
        or count <= 0
        or count > len(schedule.records)
        or not isinstance(records, list)
        or len(records) != count
    ):
        raise QualificationRenderPreflightError("GGUF prefix record count is invalid")
    minimum_exact_matches = _positive(
        minimum_exact_matches, label="minimum_exact_matches"
    )
    if minimum_exact_matches > count:
        raise QualificationRenderPreflightError(
            "minimum_exact_matches exceeds the GGUF prefix"
        )
    exact_matches = 0
    results: list[dict[str, object]] = []
    for index, (observed, expected) in enumerate(
        zip(records, schedule.records[:count], strict=True)
    ):
        if not isinstance(observed, Mapping) or set(observed) != {
            "record_id",
            "input_sha256",
            "output_text",
            "output_sha256",
        }:
            raise QualificationRenderPreflightError(
                f"GGUF prefix record {index} is invalid"
            )
        if (
            observed.get("record_id") != expected.record_id
            or observed.get("input_sha256") != expected.input_sha256
        ):
            raise QualificationRenderPreflightError(
                f"GGUF prefix record {index} pairing does not match"
            )
        try:
            output = exact_match_output_text(observed.get("output_text"))
        except ValueError as exc:
            raise QualificationRenderPreflightError(
                f"GGUF prefix record {index} output is invalid"
            ) from exc
        if observed.get("output_sha256") != _sha256(output.encode("utf-8")):
            raise QualificationRenderPreflightError(
                f"GGUF prefix record {index} output digest does not match"
            )
        matched = output == expected.expected_output
        exact_matches += int(matched)
        results.append(
            {
                "record_id": expected.record_id,
                "prompt_tokens": 0,
                "expected_tokens": 0,
                "exact_match": matched,
            }
        )
    if exact_matches < minimum_exact_matches:
        raise QualificationRenderPreflightError(
            "GGUF live prefix did not meet the exact-match minimum"
        )
    return _result(
        profile="gguf_live_prefix",
        schedule=schedule,
        schedule_file_sha256=schedule_file_sha256,
        bindings={
            "artifact_sha256": statement["artifact_sha256"],
            "backend_binary_sha256": statement["backend_binary_sha256"],
            "runtime_image_digest": statement["runtime_image_digest"],
            "signer_fingerprint": fingerprint,
            "prefix_record_count": count,
            "exact_matches": exact_matches,
            "minimum_exact_matches": minimum_exact_matches,
            "probe_statement_sha256": _sha256(_canonical_json(statement)),
        },
        record_results=results,
    )


def _load_hf_tokenizer(checkpoint: Path) -> object:
    transformers = importlib.import_module("transformers")
    return transformers.AutoTokenizer.from_pretrained(
        str(checkpoint), local_files_only=True, trust_remote_code=False
    )


def _load_processor(checkpoint: Path) -> object:
    transformers = importlib.import_module("transformers")
    return transformers.AutoProcessor.from_pretrained(
        str(checkpoint), local_files_only=True, trust_remote_code=False
    )


def _load_tensorrt_contract(path: Path) -> tuple[object, str]:
    payload = read_regular_file_bytes(
        path, label="TensorRT tokenizer contract", max_bytes=_MAX_CONTRACT_BYTES
    )
    decoded = parse_json_bytes(payload, label="TensorRT tokenizer contract")
    expected = {
        "add_special_tokens",
        "clean_up_tokenization_spaces",
        "eos_token_id",
        "format_version",
        "pad_token_id",
        "skip_special_tokens",
        "tokenizer_json",
    }
    if not isinstance(decoded, Mapping) or set(decoded) != expected:
        raise QualificationRenderPreflightError(
            "TensorRT tokenizer contract is not closed"
        )
    if (
        decoded.get("add_special_tokens") is not False
        or decoded.get("skip_special_tokens") is not True
        or decoded.get("clean_up_tokenization_spaces") is not False
    ):
        raise QualificationRenderPreflightError(
            "TensorRT tokenizer contract flags are unsupported"
        )
    tokenizers = importlib.import_module("tokenizers")
    transformers = importlib.import_module("transformers")
    raw = tokenizers.Tokenizer.from_str(
        _canonical_json(decoded["tokenizer_json"]).decode()
    )
    eos_id = decoded.get("eos_token_id")
    pad_id = decoded.get("pad_token_id")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in (eos_id, pad_id)
    ):
        raise QualificationRenderPreflightError(
            "TensorRT tokenizer special-token IDs are invalid"
        )
    eos = raw.id_to_token(eos_id)
    pad = raw.id_to_token(pad_id)
    if not isinstance(eos, str) or not isinstance(pad, str):
        raise QualificationRenderPreflightError(
            "TensorRT tokenizer special tokens are unavailable"
        )
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=raw,
        eos_token=eos,
        pad_token=pad,
        clean_up_tokenization_spaces=False,
    )
    if tokenizer.eos_token_id != eos_id or tokenizer.pad_token_id != pad_id:
        raise QualificationRenderPreflightError(
            "TensorRT tokenizer special-token IDs do not round-trip"
        )
    return tokenizer, _sha256(payload)


def _image_resolver(content_store: Path) -> Callable[[EvaluationRecord], object]:
    def resolve(record: EvaluationRecord) -> object:
        content = next(part for part in record.input_parts if part.kind == "content")
        assert isinstance(content.content_id, str)
        payload = read_regular_file_bytes(
            content_store / content.content_id,
            label=f"multimodal content {content.content_id!r}",
            max_bytes=_MAX_IMAGE_BYTES,
        )
        if len(payload) != content.byte_length or _sha256(payload) != content.sha256:
            raise QualificationRenderPreflightError(
                f"multimodal content {content.content_id!r} identity does not match"
            )
        image_module = importlib.import_module("PIL.Image")
        image = image_module.open(io.BytesIO(payload))
        try:
            image.load()
            converted = image.convert("RGB")
        finally:
            image.close()
        return converted

    return resolve


def _read_json_object(path: Path, *, label: str) -> tuple[bytes, Mapping[str, object]]:
    payload = read_regular_file_bytes(path, label=label, max_bytes=_MAX_CONTRACT_BYTES)
    value = parse_json_bytes(payload, label=label)
    if not isinstance(value, Mapping):
        raise QualificationRenderPreflightError(f"{label} must be a JSON object")
    return payload, value


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--schedule-file-sha256", required=True)
    parser.add_argument("--schedule-sha256", required=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="profile", required=True)
    hf = subparsers.add_parser("hf-text")
    _common(hf)
    hf.add_argument("--checkpoint", type=Path, required=True)
    hf.add_argument("--tokenizer-sha256", required=True)
    hf.add_argument("--context-length", type=int, required=True)
    hf.add_argument("--max-output-tokens", type=int, required=True)
    hf.add_argument(
        "--metric",
        choices=("exact_match", "normalized_nll_per_utf8_byte"),
        required=True,
    )
    trt = subparsers.add_parser("tensorrt-llm")
    _common(trt)
    trt.add_argument("--tokenizer-contract", type=Path, required=True)
    trt.add_argument("--tokenizer-sha256", required=True)
    trt.add_argument("--engine-config", type=Path, required=True)
    trt.add_argument("--engine-config-sha256", required=True)
    trt.add_argument("--context-length", type=int, required=True)
    trt.add_argument("--max-output-tokens", type=int, required=True)
    mm = subparsers.add_parser("multimodal")
    _common(mm)
    mm.add_argument("--checkpoint", type=Path, required=True)
    mm.add_argument("--processor-sha256", required=True)
    mm.add_argument("--content-store", type=Path, required=True)
    mm.add_argument("--context-length", type=int, required=True)
    mm.add_argument("--max-output-tokens", type=int, required=True)
    gguf = subparsers.add_parser("gguf-prefix")
    _common(gguf)
    gguf.add_argument("--statement", type=Path, required=True)
    gguf.add_argument("--statement-sha256", required=True)
    gguf.add_argument("--signature", type=Path, required=True)
    gguf.add_argument("--public-key", type=Path, required=True)
    gguf.add_argument("--signer-fingerprint", required=True)
    gguf.add_argument("--minimum-exact-matches", type=int, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    schedule, schedule_file_sha256 = load_bound_schedule(
        arguments.schedule,
        expected_file_sha256=arguments.schedule_file_sha256,
        expected_schedule_sha256=arguments.schedule_sha256,
    )
    if arguments.profile == "hf-text":
        identity = importlib.import_module(
            "invarlock.runtime_providers.hf_transformers"
        ).hf_tokenizer_contract_sha256
        result = preflight_hf_text(
            schedule,
            schedule_file_sha256=schedule_file_sha256,
            tokenizer=_load_hf_tokenizer(arguments.checkpoint),
            expected_tokenizer_sha256=arguments.tokenizer_sha256,
            context_length=arguments.context_length,
            max_output_tokens=arguments.max_output_tokens,
            metric=arguments.metric,
            tokenizer_digest=identity,
        )
    elif arguments.profile == "tensorrt-llm":
        tokenizer, observed_tokenizer_sha256 = _load_tensorrt_contract(
            arguments.tokenizer_contract
        )
        if observed_tokenizer_sha256 != arguments.tokenizer_sha256:
            raise QualificationRenderPreflightError(
                "TensorRT tokenizer contract digest mismatch"
            )
        config_bytes, config = _read_json_object(
            arguments.engine_config, label="TensorRT engine config"
        )
        if _sha256(config_bytes) != arguments.engine_config_sha256:
            raise QualificationRenderPreflightError(
                "TensorRT engine config digest mismatch"
            )
        result = preflight_tensorrt(
            schedule,
            schedule_file_sha256=schedule_file_sha256,
            tokenizer=tokenizer,
            tokenizer_contract_sha256=observed_tokenizer_sha256,
            engine_config=config,
            engine_config_sha256=_sha256(config_bytes),
            context_length=arguments.context_length,
            max_output_tokens=arguments.max_output_tokens,
        )
    elif arguments.profile == "multimodal":
        provider = importlib.import_module("invarlock_addins.multimodal.provider")
        result = preflight_multimodal(
            schedule,
            schedule_file_sha256=schedule_file_sha256,
            processor=_load_processor(arguments.checkpoint),
            expected_processor_sha256=arguments.processor_sha256,
            context_length=arguments.context_length,
            max_output_tokens=arguments.max_output_tokens,
            processor_digest=provider.processor_contract_sha256,
            image_resolver=_image_resolver(arguments.content_store),
        )
    else:
        statement_bytes, statement = _read_json_object(
            arguments.statement, label="GGUF prefix statement"
        )
        if statement_bytes != _canonical_json(statement) + b"\n":
            raise QualificationRenderPreflightError(
                "GGUF prefix statement must be canonical JSON with one final newline"
            )
        if "sha256:" + _sha256(statement_bytes) != arguments.statement_sha256:
            raise QualificationRenderPreflightError(
                "GGUF prefix statement file digest mismatch"
            )
        signature = base64.b64decode(
            read_regular_file_bytes(
                arguments.signature, label="GGUF prefix signature", max_bytes=4096
            ).strip(),
            validate=True,
        )
        loaded_key = serialization.load_pem_public_key(
            read_regular_file_bytes(
                arguments.public_key,
                label="GGUF prefix public key",
                max_bytes=16 * 1024,
            )
        )
        if not isinstance(loaded_key, ed25519.Ed25519PublicKey):
            raise QualificationRenderPreflightError(
                "GGUF prefix public key must be Ed25519"
            )
        result = verify_gguf_prefix(
            schedule,
            schedule_file_sha256=schedule_file_sha256,
            statement=statement,
            signature=signature,
            public_key=loaded_key,
            expected_signer_fingerprint=arguments.signer_fingerprint,
            minimum_exact_matches=arguments.minimum_exact_matches,
        )
    print(_canonical_json(result).decode("utf-8"))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
