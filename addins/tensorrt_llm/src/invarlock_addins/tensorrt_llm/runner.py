"""Closed TensorRT-LLM 1.2.1 add-in connector used by the runtime image.

Importing this module is deliberately torch-, transformers-, CUDA-, and
TensorRT-free. Optional backend imports occur only after a strict score request
has authenticated its local engine/tokenizer inputs and execution boundary.
"""

from __future__ import annotations

import ctypes
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import re
import stat
import sys
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Protocol

from invarlock.output_text_contract import exact_match_output_text

_PROTOCOL_VERSION = "invarlock/tensorrt-llm-runner-v1"
_INFO_FORMAT_VERSION = "invarlock/tensorrt-llm-runner-info-v1"
_REQUEST_FORMAT_VERSION = "invarlock/tensorrt-llm-runner-request-v1"
_RESPONSE_FORMAT_VERSION = "invarlock/tensorrt-llm-runner-response-v1"
_BATCH_REQUEST_FORMAT_VERSION = "invarlock/tensorrt-llm-runner-batch-request-v1"
_BATCH_RESPONSE_FORMAT_VERSION = "invarlock/tensorrt-llm-runner-batch-response-v1"
_TOKENIZER_FORMAT_VERSION = "invarlock/tensorrt-llm-tokenizer-contract-v1"
_BACKEND_VERSION = "1.2.1"
_MAX_REQUEST_BYTES = 1024 * 1024
_MAX_CONFIG_BYTES = 1024 * 1024
_MAX_TOKENIZER_BYTES = 128 * 1024 * 1024
_MAX_JSON_DEPTH = 64
_MAX_JSON_ITEMS = 1_250_000
_MAX_TEXT_BYTES = 1024 * 1024
_MAX_RECORD_ID_BYTES = 4096
_MAX_BATCH_RECORDS = 1024
_MAX_OUTPUT_BYTES = 2 * 1024 * 1024
_MAX_BATCH_RESPONSE_BYTES = 2 * 1024 * 1024
_MAX_SETTING_VALUE = 1024 * 1024
_MAX_TIMEOUT_SECONDS = 24 * 60 * 60
_IPV4_ROUTE_PATH = Path("/proc/net/route")
_IPV6_ROUTE_PATH = Path("/proc/net/ipv6_route")
_DRIVER_VERSION = re.compile(
    r"^NVRM version: NVIDIA UNIX "
    r"(?:(?:x86_64|aarch64) Kernel Module|"
    r"Open Kernel Module for (?:x86_64|aarch64))"
    r"[ \t]+([0-9]+(?:\.[0-9]+)+)[ \t]+"
    r"(?:Release Build[ \t]+\([A-Za-z0-9_.@/-]+\)[ \t]+)?"
    r"(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)[ \t]+"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[ \t]+"
    r"(?:[1-9]|[12][0-9]|3[01])[ \t]+"
    r"(?:[01][0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9][ \t]+UTC[ \t]+"
    r"(?:19|20)[0-9]{2}$",
    re.MULTILINE,
)
_CRITICAL_BACKEND_FILES = (
    "tensorrt_llm/__init__.py",
    "tensorrt_llm/version.py",
    "tensorrt_llm/llmapi/llm.py",
    "tensorrt_llm/llmapi/llm_args.py",
    "tensorrt_llm/_tensorrt_engine/__init__.py",
    "tensorrt_llm/sampling_params.py",
)


class TensorRTLLMRunnerError(RuntimeError):
    """Raised when the connector cannot satisfy its closed contract."""


class _Constructor(Protocol):
    def __call__(self, *args: object, **kwargs: object) -> object: ...


@dataclass(frozen=True)
class _Backend:
    llm: _Constructor
    sampling_params: _Constructor
    fast_tokenizer: _Constructor
    raw_tokenizer_from_str: Callable[[str], object]


@dataclass(frozen=True)
class _TokenizerContract:
    tokenizer_json: Mapping[str, object]
    eos_token_id: int
    pad_token_id: int
    add_special_tokens: bool
    skip_special_tokens: bool
    clean_up_tokenization_spaces: bool


@dataclass(frozen=True)
class _ExecutionSettings:
    allow_network: bool
    batch_size: int
    context_length: int
    max_output_tokens: int
    seed: int
    timeout_seconds: int


@dataclass(frozen=True)
class _ObservedDevice:
    device_name: str
    compute_capability: str
    driver_version: str
    cuda_runtime_version: str


@dataclass(frozen=True)
class _ScoreRequest:
    engine_bundle: Path
    tokenizer_contract_path: Path
    tokenizer_contract: _TokenizerContract
    engine_config: Mapping[str, object]
    input_text: str
    settings: _ExecutionSettings


@dataclass(frozen=True)
class _BatchScoreRecord:
    record_id: str
    input_text: str


@dataclass(frozen=True)
class _BatchScoreRequest:
    engine_bundle: Path
    tokenizer_contract_path: Path
    tokenizer_contract: _TokenizerContract
    engine_config: Mapping[str, object]
    records: tuple[_BatchScoreRecord, ...]
    settings: _ExecutionSettings


def _reject_duplicate_pairs(items: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in items:
        if key in result:
            raise TensorRTLLMRunnerError("JSON contains a duplicate object key")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> object:
    raise TensorRTLLMRunnerError(f"JSON contains non-finite number {value!r}")


def _finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise TensorRTLLMRunnerError("JSON contains a non-finite number")
    return parsed


def _validate_json_budget(value: object, *, depth: int = 0) -> int:
    if depth > _MAX_JSON_DEPTH:
        raise TensorRTLLMRunnerError("JSON exceeds the maximum nesting depth")
    if isinstance(value, dict):
        count = len(value)
        for key, child in value.items():
            if not isinstance(key, str):
                raise TensorRTLLMRunnerError("JSON object key is not text")
            count += _validate_json_budget(child, depth=depth + 1)
    elif isinstance(value, list):
        count = len(value)
        for child in value:
            count += _validate_json_budget(child, depth=depth + 1)
    elif isinstance(value, float) and not math.isfinite(value):
        raise TensorRTLLMRunnerError("JSON contains a non-finite number")
    else:
        count = 1
    if count > _MAX_JSON_ITEMS:
        raise TensorRTLLMRunnerError("JSON exceeds the maximum item count")
    return count


def _strict_json_object(payload: bytes, *, label: str) -> dict[str, object]:
    try:
        text = payload.decode("utf-8", errors="strict")
        decoded = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_nonfinite,
            parse_float=_finite_float,
        )
    except TensorRTLLMRunnerError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise TensorRTLLMRunnerError(f"{label} is not strict JSON") from exc
    if not isinstance(decoded, dict):
        raise TensorRTLLMRunnerError(f"{label} must be a JSON object")
    _validate_json_budget(decoded)
    return decoded


def _read_bounded(stream: BinaryIO, limit: int, *, label: str) -> bytes:
    payload = stream.read(limit + 1)
    if len(payload) > limit:
        raise TensorRTLLMRunnerError(f"{label} exceeds the byte limit")
    return payload


def _read_regular_file(path: Path, limit: int, *, label: str) -> bytes:
    try:
        opened = path.lstat()
    except OSError as exc:
        raise TensorRTLLMRunnerError(f"{label} is unavailable") from exc
    if not stat.S_ISREG(opened.st_mode) or opened.st_size > limit:
        raise TensorRTLLMRunnerError(f"{label} is not a bounded regular file")
    try:
        with path.open("rb") as stream:
            payload = _read_bounded(stream, limit, label=label)
            final = os.fstat(stream.fileno())
    except OSError as exc:
        raise TensorRTLLMRunnerError(f"{label} cannot be read") from exc
    if (
        final.st_dev != opened.st_dev
        or final.st_ino != opened.st_ino
        or final.st_size != opened.st_size
        or final.st_mtime_ns != opened.st_mtime_ns
        or final.st_ctime_ns != opened.st_ctime_ns
    ):
        raise TensorRTLLMRunnerError(f"{label} changed while being read")
    return payload


def _canonical_path(value: object, *, label: str) -> Path:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise TensorRTLLMRunnerError(f"{label} path is invalid")
    supplied = Path(value)
    if not supplied.is_absolute() or ".." in supplied.parts:
        raise TensorRTLLMRunnerError(f"{label} path must be canonical and absolute")
    try:
        resolved = supplied.resolve(strict=True)
    except OSError as exc:
        raise TensorRTLLMRunnerError(f"{label} path is unavailable") from exc
    if resolved != supplied:
        raise TensorRTLLMRunnerError(f"{label} path must not contain symlinks")
    return resolved


def _positive_integer(value: object, *, label: str, maximum: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 1 <= value <= maximum
    ):
        raise TensorRTLLMRunnerError(f"{label} is outside the supported bound")
    return value


def _nonnegative_integer(value: object, *, label: str, maximum: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value <= maximum
    ):
        raise TensorRTLLMRunnerError(f"{label} is outside the supported bound")
    return value


def _parse_settings(value: object) -> _ExecutionSettings:
    if not isinstance(value, dict) or set(value) != {
        "allow_network",
        "batch_size",
        "context_length",
        "max_output_tokens",
        "seed",
        "timeout_seconds",
    }:
        raise TensorRTLLMRunnerError("request settings are not closed")
    if value["allow_network"] is not False:
        raise TensorRTLLMRunnerError("TensorRT-LLM network access must be disabled")
    return _ExecutionSettings(
        allow_network=False,
        batch_size=_positive_integer(
            value["batch_size"], label="batch_size", maximum=1024
        ),
        context_length=_positive_integer(
            value["context_length"],
            label="context_length",
            maximum=_MAX_SETTING_VALUE,
        ),
        max_output_tokens=_positive_integer(
            value["max_output_tokens"],
            label="max_output_tokens",
            maximum=_MAX_SETTING_VALUE,
        ),
        seed=_nonnegative_integer(value["seed"], label="seed", maximum=(2**63) - 1),
        timeout_seconds=_positive_integer(
            value["timeout_seconds"],
            label="timeout_seconds",
            maximum=_MAX_TIMEOUT_SECONDS,
        ),
    )


def _parse_tokenizer_contract(path: Path) -> _TokenizerContract:
    payload = _strict_json_object(
        _read_regular_file(path, _MAX_TOKENIZER_BYTES, label="tokenizer contract"),
        label="tokenizer contract",
    )
    if set(payload) != {
        "add_special_tokens",
        "clean_up_tokenization_spaces",
        "eos_token_id",
        "format_version",
        "pad_token_id",
        "skip_special_tokens",
        "tokenizer_json",
    }:
        raise TensorRTLLMRunnerError("tokenizer contract fields are not closed")
    if payload["format_version"] != _TOKENIZER_FORMAT_VERSION:
        raise TensorRTLLMRunnerError("tokenizer contract version is unsupported")
    tokenizer_json = payload["tokenizer_json"]
    if not isinstance(tokenizer_json, dict) or not tokenizer_json:
        raise TensorRTLLMRunnerError("tokenizer_json must be a non-empty object")
    add_special_tokens = payload["add_special_tokens"]
    skip_special_tokens = payload["skip_special_tokens"]
    clean_up = payload["clean_up_tokenization_spaces"]
    if add_special_tokens is not False:
        raise TensorRTLLMRunnerError(
            "the v1 tokenizer contract requires add_special_tokens=false"
        )
    if skip_special_tokens is not True:
        raise TensorRTLLMRunnerError(
            "the v1 tokenizer contract requires skip_special_tokens=true"
        )
    if clean_up is not False:
        raise TensorRTLLMRunnerError(
            "the v1 tokenizer contract requires clean_up_tokenization_spaces=false"
        )
    return _TokenizerContract(
        tokenizer_json=tokenizer_json,
        eos_token_id=_nonnegative_integer(
            payload["eos_token_id"], label="eos_token_id", maximum=(2**31) - 1
        ),
        pad_token_id=_nonnegative_integer(
            payload["pad_token_id"], label="pad_token_id", maximum=(2**31) - 1
        ),
        add_special_tokens=False,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )


def _parse_engine_config(engine_bundle: Path) -> dict[str, object]:
    try:
        opened = engine_bundle.lstat()
        entries = sorted(engine_bundle.iterdir(), key=lambda item: item.name)
    except OSError as exc:
        raise TensorRTLLMRunnerError("engine bundle cannot be inspected") from exc
    if not stat.S_ISDIR(opened.st_mode) or {entry.name for entry in entries} != {
        "config.json",
        "rank0.engine",
    }:
        raise TensorRTLLMRunnerError(
            "the runner requires a closed single-rank engine bundle"
        )
    for entry in entries:
        try:
            metadata = entry.lstat()
        except OSError as exc:
            raise TensorRTLLMRunnerError("engine bundle entry is unavailable") from exc
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size <= 0:
            raise TensorRTLLMRunnerError(
                "engine bundle contains a non-regular or empty entry"
            )
    config = _strict_json_object(
        _read_regular_file(
            engine_bundle / "config.json",
            _MAX_CONFIG_BYTES,
            label="engine config",
        ),
        label="engine config",
    )
    if set(config) != {"build_config", "pretrained_config", "version"}:
        raise TensorRTLLMRunnerError("engine config fields are not closed")
    pretrained = config["pretrained_config"]
    build = config["build_config"]
    if not isinstance(pretrained, dict) or not isinstance(build, dict):
        raise TensorRTLLMRunnerError("engine config sections must be objects")
    mapping = pretrained.get("mapping")
    if not isinstance(mapping, dict):
        raise TensorRTLLMRunnerError("engine mapping must be an object")
    for name in ("world_size", "tp_size", "pp_size"):
        if mapping.get(name) != 1:
            raise TensorRTLLMRunnerError("the runner supports only single-rank engines")
    if mapping.get("cp_size", 1) != 1:
        raise TensorRTLLMRunnerError("the runner supports only single-rank engines")
    return config


def _engine_limits(config: Mapping[str, object]) -> tuple[int, int, int]:
    build = config["build_config"]
    assert isinstance(build, dict)
    return (
        _positive_integer(
            build.get("max_batch_size"),
            label="engine max_batch_size",
            maximum=1024,
        ),
        _positive_integer(
            build.get("max_input_len"),
            label="engine max_input_len",
            maximum=_MAX_SETTING_VALUE,
        ),
        _positive_integer(
            build.get("max_seq_len"),
            label="engine max_seq_len",
            maximum=_MAX_SETTING_VALUE * 2,
        ),
    )


def _canonical_request(payload: bytes) -> dict[str, object]:
    request = _strict_json_object(payload, label="runner request")
    try:
        canonical = json.dumps(
            request,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise TensorRTLLMRunnerError("runner request is not canonical JSON") from exc
    if payload != canonical:
        raise TensorRTLLMRunnerError("runner request is not canonical JSON")
    return request


def _validated_runtime_inputs(
    request: Mapping[str, object],
) -> tuple[
    Path,
    Path,
    _TokenizerContract,
    Mapping[str, object],
    _ExecutionSettings,
]:
    engine_bundle = _canonical_path(request["engine_bundle"], label="engine bundle")
    tokenizer_path = _canonical_path(
        request["tokenizer_contract"], label="tokenizer contract"
    )
    if (
        engine_bundle.name != "engine"
        or tokenizer_path.name != "tokenizer.json"
        or engine_bundle.parent != tokenizer_path.parent
    ):
        raise TensorRTLLMRunnerError("runtime inputs do not use the closed layout")
    engine_config = _parse_engine_config(engine_bundle)
    tokenizer_contract = _parse_tokenizer_contract(tokenizer_path)
    settings = _parse_settings(request["settings"])
    max_batch_size, max_input_len, max_seq_len = _engine_limits(engine_config)
    if settings.batch_size > max_batch_size:
        raise TensorRTLLMRunnerError("batch_size exceeds the engine build limit")
    if settings.context_length > max_input_len:
        raise TensorRTLLMRunnerError("context_length exceeds the engine build limit")
    if settings.context_length + settings.max_output_tokens > max_seq_len:
        raise TensorRTLLMRunnerError(
            "context and output lengths exceed the engine sequence limit"
        )
    return (
        engine_bundle,
        tokenizer_path,
        tokenizer_contract,
        engine_config,
        settings,
    )


def _input_text(value: object) -> str:
    if not isinstance(value, str):
        raise TensorRTLLMRunnerError("input_text must be text")
    try:
        input_bytes = value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise TensorRTLLMRunnerError("input_text is not valid UTF-8") from exc
    if len(input_bytes) > _MAX_TEXT_BYTES:
        raise TensorRTLLMRunnerError("input_text exceeds the byte limit")
    return value


def _record_id(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise TensorRTLLMRunnerError("record_id must be non-empty text")
    try:
        encoded = value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise TensorRTLLMRunnerError("record_id is not valid UTF-8") from exc
    if len(encoded) > _MAX_RECORD_ID_BYTES:
        raise TensorRTLLMRunnerError("record_id exceeds the byte limit")
    return value


def _parse_request(payload: bytes) -> _ScoreRequest:
    request = _canonical_request(payload)
    if set(request) != {
        "engine_bundle",
        "format_version",
        "input_text",
        "protocol_version",
        "settings",
        "tokenizer_contract",
    }:
        raise TensorRTLLMRunnerError("runner request fields are not closed")
    if request["format_version"] != _REQUEST_FORMAT_VERSION:
        raise TensorRTLLMRunnerError("runner request format is unsupported")
    if request["protocol_version"] != _PROTOCOL_VERSION:
        raise TensorRTLLMRunnerError("runner protocol version is unsupported")
    input_text = _input_text(request["input_text"])
    (
        engine_bundle,
        tokenizer_path,
        tokenizer_contract,
        engine_config,
        settings,
    ) = _validated_runtime_inputs(request)
    return _ScoreRequest(
        engine_bundle=engine_bundle,
        tokenizer_contract_path=tokenizer_path,
        tokenizer_contract=tokenizer_contract,
        engine_config=engine_config,
        input_text=input_text,
        settings=settings,
    )


def _parse_batch_request(payload: bytes) -> _BatchScoreRequest:
    request = _canonical_request(payload)
    if set(request) != {
        "engine_bundle",
        "format_version",
        "protocol_version",
        "records",
        "settings",
        "tokenizer_contract",
    }:
        raise TensorRTLLMRunnerError("batch runner request fields are not closed")
    if request["format_version"] != _BATCH_REQUEST_FORMAT_VERSION:
        raise TensorRTLLMRunnerError("batch runner request format is unsupported")
    if request["protocol_version"] != _PROTOCOL_VERSION:
        raise TensorRTLLMRunnerError("runner protocol version is unsupported")
    raw_records = request["records"]
    if (
        not isinstance(raw_records, list)
        or not raw_records
        or len(raw_records) > _MAX_BATCH_RECORDS
    ):
        raise TensorRTLLMRunnerError(
            "batch records count is outside the supported bound"
        )
    records: list[_BatchScoreRecord] = []
    observed_ids: set[str] = set()
    for raw_record in raw_records:
        if not isinstance(raw_record, dict) or set(raw_record) != {
            "input_text",
            "record_id",
        }:
            raise TensorRTLLMRunnerError("batch record fields are not closed")
        record_id = _record_id(raw_record["record_id"])
        if record_id in observed_ids:
            raise TensorRTLLMRunnerError("batch record IDs must be unique")
        observed_ids.add(record_id)
        records.append(
            _BatchScoreRecord(
                record_id=record_id,
                input_text=_input_text(raw_record["input_text"]),
            )
        )
    (
        engine_bundle,
        tokenizer_path,
        tokenizer_contract,
        engine_config,
        settings,
    ) = _validated_runtime_inputs(request)
    return _BatchScoreRequest(
        engine_bundle=engine_bundle,
        tokenizer_contract_path=tokenizer_path,
        tokenizer_contract=tokenizer_contract,
        engine_config=engine_config,
        records=tuple(records),
        settings=settings,
    )


def _require_isolated_network_namespace(
    *,
    ipv4_route_path: Path = _IPV4_ROUTE_PATH,
    ipv6_route_path: Path = _IPV6_ROUTE_PATH,
) -> None:
    try:
        ipv4_lines = ipv4_route_path.read_text(
            encoding="ascii", errors="strict"
        ).splitlines()
        ipv6_lines = ipv6_route_path.read_text(
            encoding="ascii", errors="strict"
        ).splitlines()
    except (OSError, UnicodeError) as exc:
        raise TensorRTLLMRunnerError(
            "runner cannot verify the network namespace"
        ) from exc
    if (
        not ipv4_lines
        or not ipv4_lines[0].split()
        or ipv4_lines[0].split()[0] != "Iface"
    ):
        raise TensorRTLLMRunnerError("runner cannot verify the IPv4 route table")
    ipv4_interfaces: set[str] = set()
    for line in ipv4_lines[1:]:
        fields = line.split()
        if not fields:
            continue
        if len(fields) < 11:
            raise TensorRTLLMRunnerError("runner cannot verify the IPv4 route table")
        ipv4_interfaces.add(fields[0])
    ipv6_interfaces: set[str] = set()
    for line in ipv6_lines:
        fields = line.split()
        if not fields:
            continue
        if len(fields) != 10:
            raise TensorRTLLMRunnerError("runner cannot verify the IPv6 route table")
        ipv6_interfaces.add(fields[-1])
    if (ipv4_interfaces | ipv6_interfaces) - {"lo"}:
        raise TensorRTLLMRunnerError("runner requires a network-disabled container")


def _require_runtime_boundary() -> None:
    from invarlock.runtime_security_helpers import strict_container_boundary_present

    if not strict_container_boundary_present():
        raise TensorRTLLMRunnerError(
            "runner requires the authenticated container boundary"
        )
    _require_isolated_network_namespace()


def _require_backend_version() -> None:
    try:
        version = importlib.metadata.version("tensorrt_llm")
    except importlib.metadata.PackageNotFoundError as exc:
        raise TensorRTLLMRunnerError("TensorRT-LLM is not installed") from exc
    if version != _BACKEND_VERSION:
        raise TensorRTLLMRunnerError("TensorRT-LLM version is not pinned")


def _hash_regular_backend_file(path: Path, *, logical_name: str) -> tuple[int, str]:
    try:
        opened = path.lstat()
    except OSError as exc:
        raise TensorRTLLMRunnerError(
            "TensorRT-LLM backend file is unavailable"
        ) from exc
    if not stat.S_ISREG(opened.st_mode) or opened.st_size <= 0:
        raise TensorRTLLMRunnerError(
            "TensorRT-LLM backend file is not a non-empty regular file"
        )
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
            final = os.fstat(stream.fileno())
    except OSError as exc:
        raise TensorRTLLMRunnerError(
            "TensorRT-LLM backend file cannot be hashed"
        ) from exc
    if (
        final.st_dev != opened.st_dev
        or final.st_ino != opened.st_ino
        or final.st_size != opened.st_size
        or final.st_mtime_ns != opened.st_mtime_ns
        or final.st_ctime_ns != opened.st_ctime_ns
    ):
        raise TensorRTLLMRunnerError(
            "TensorRT-LLM backend file changed while being hashed"
        )
    if (
        not logical_name
        or logical_name.startswith("/")
        or ".." in Path(logical_name).parts
    ):
        raise TensorRTLLMRunnerError("TensorRT-LLM backend inventory is invalid")
    return opened.st_size, digest.hexdigest()


def _observed_backend_build_sha256() -> str:
    """Hash the live HLAPI sources and native extension bytes used by the runner."""

    _require_backend_version()
    try:
        distribution = importlib.metadata.distribution("tensorrt_llm")
        files = distribution.files
    except importlib.metadata.PackageNotFoundError as exc:
        raise TensorRTLLMRunnerError("TensorRT-LLM is not installed") from exc
    if files is None:
        raise TensorRTLLMRunnerError("TensorRT-LLM file inventory is unavailable")
    available = {str(path).replace("\\", "/"): path for path in files}
    native_names = sorted(
        name
        for name in available
        if name.startswith("tensorrt_llm/") and name.endswith(".so")
    )
    selected_names = [*_CRITICAL_BACKEND_FILES, *native_names]
    if not native_names or any(
        name not in available for name in _CRITICAL_BACKEND_FILES
    ):
        raise TensorRTLLMRunnerError(
            "TensorRT-LLM critical backend inventory is incomplete"
        )
    if len(selected_names) > 256:
        raise TensorRTLLMRunnerError(
            "TensorRT-LLM critical backend inventory exceeds the file bound"
        )
    inventory: list[dict[str, object]] = []
    for name in selected_names:
        path = Path(str(distribution.locate_file(available[name])))
        size, sha256 = _hash_regular_backend_file(path, logical_name=name)
        inventory.append({"byte_length": size, "name": name, "sha256": sha256})
    encoded = json.dumps(
        inventory,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(
        b"invarlock/tensorrt-llm-backend-build-v1\0" + encoded
    ).hexdigest()


def _read_driver_version(
    *, version_path: Path = Path("/proc/driver/nvidia/version")
) -> str:
    try:
        payload = version_path.read_text(encoding="ascii", errors="strict")
    except (OSError, UnicodeError) as exc:
        raise TensorRTLLMRunnerError("NVIDIA driver version is unavailable") from exc
    match = _DRIVER_VERSION.search(payload)
    if match is None:
        raise TensorRTLLMRunnerError("NVIDIA driver version is not canonical")
    return match.group(1)


def _read_cuda_runtime_version(
    *, library_loader: Callable[[str], object] = ctypes.CDLL
) -> str:
    """Query the live CUDA Runtime instance through ``cudaRuntimeGetVersion``."""

    try:
        library = library_loader("libcudart.so")
        get_version = library.cudaRuntimeGetVersion  # type: ignore[attr-defined]
        get_version.argtypes = [ctypes.POINTER(ctypes.c_int)]
        get_version.restype = ctypes.c_int
        encoded = ctypes.c_int()
        status = get_version(ctypes.byref(encoded))
    except (AttributeError, OSError, TypeError) as exc:
        raise TensorRTLLMRunnerError("CUDA runtime version cannot be observed") from exc
    if isinstance(status, bool) or not isinstance(status, int) or status != 0:
        raise TensorRTLLMRunnerError("CUDA runtime version probe failed")
    if encoded.value < 1000:
        raise TensorRTLLMRunnerError("CUDA runtime version is invalid")
    major = encoded.value // 1000
    minor = (encoded.value % 1000) // 10
    patch = encoded.value % 10
    version = f"{major}.{minor}" if patch == 0 else f"{major}.{minor}.{patch}"
    return version


def _observe_cuda_device() -> _ObservedDevice:
    try:
        torch = importlib.import_module("torch")
        cuda = torch.cuda
        if cuda.is_available() is not True or cuda.device_count() < 1:
            raise TensorRTLLMRunnerError("CUDA device is unavailable")
        index = cuda.current_device()
        name = cuda.get_device_name(index)
        capability = cuda.get_device_capability(index)
    except TensorRTLLMRunnerError:
        raise
    except (AttributeError, ImportError, RuntimeError) as exc:
        raise TensorRTLLMRunnerError("CUDA device facts cannot be observed") from exc
    if (
        isinstance(index, bool)
        or not isinstance(index, int)
        or index < 0
        or not isinstance(name, str)
        or not name
        or name != name.strip()
        or any(ord(character) < 32 for character in name)
    ):
        raise TensorRTLLMRunnerError("CUDA device identity is invalid")
    if (
        not isinstance(capability, tuple)
        or len(capability) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in capability
        )
        or not all(0 <= value <= 99 for value in capability)
    ):
        raise TensorRTLLMRunnerError("CUDA compute capability is invalid")
    return _ObservedDevice(
        device_name=name,
        compute_capability=f"{capability[0]}.{capability[1]}",
        driver_version=_read_driver_version(),
        cuda_runtime_version=_read_cuda_runtime_version(),
    )


def _load_backend() -> _Backend:
    _require_backend_version()
    try:
        tensorrt_llm = importlib.import_module("tensorrt_llm")
        tensorrt_engine = importlib.import_module("tensorrt_llm._tensorrt_engine")
        transformers = importlib.import_module("transformers")
        tokenizers = importlib.import_module("tokenizers")
        llm = tensorrt_engine.LLM
        sampling_params = tensorrt_llm.SamplingParams
        fast_tokenizer = transformers.PreTrainedTokenizerFast
        raw_tokenizer = tokenizers.Tokenizer
        from_str = raw_tokenizer.from_str
    except (AttributeError, ImportError) as exc:
        raise TensorRTLLMRunnerError(
            "TensorRT-LLM 1.2.1 connector API is unavailable"
        ) from exc
    for value in (llm, sampling_params, fast_tokenizer, from_str):
        if not callable(value):
            raise TensorRTLLMRunnerError(
                "TensorRT-LLM 1.2.1 connector API is unavailable"
            )
    return _Backend(
        llm=llm,
        sampling_params=sampling_params,
        fast_tokenizer=fast_tokenizer,
        raw_tokenizer_from_str=from_str,
    )


@contextmanager
def _silence_backend_output() -> Iterator[None]:
    """Keep vendor Python/C++ logs outside the strict JSON response channel."""

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except (AttributeError, OSError):
            pass
    saved_stdout = os.dup(1)
    saved_stderr = os.dup(2)
    sink = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(sink, 1)
        os.dup2(sink, 2)
        yield
    finally:
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.flush()
            except (AttributeError, OSError):
                pass
        os.dup2(saved_stdout, 1)
        os.dup2(saved_stderr, 2)
        os.close(saved_stdout)
        os.close(saved_stderr)
        os.close(sink)


def _tokenizer_from_contract(contract: _TokenizerContract, backend: _Backend) -> object:
    tokenizer_payload = json.dumps(
        contract.tokenizer_json,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    try:
        raw_tokenizer = backend.raw_tokenizer_from_str(tokenizer_payload)
        id_to_token = getattr(raw_tokenizer, "id_to_token")  # noqa: B009
        eos_token = id_to_token(contract.eos_token_id)
        pad_token = id_to_token(contract.pad_token_id)
        if not isinstance(eos_token, str) or not isinstance(pad_token, str):
            raise TensorRTLLMRunnerError(
                "tokenizer contract special token IDs are unavailable"
            )
        tokenizer = backend.fast_tokenizer(
            tokenizer_object=raw_tokenizer,
            eos_token=eos_token,
            pad_token=pad_token,
            clean_up_tokenization_spaces=contract.clean_up_tokenization_spaces,
        )
        if (
            getattr(tokenizer, "eos_token_id", None) != contract.eos_token_id
            or getattr(tokenizer, "pad_token_id", None) != contract.pad_token_id
        ):
            raise TensorRTLLMRunnerError(
                "tokenizer contract special token IDs do not round-trip"
            )
    except TensorRTLLMRunnerError:
        raise
    except Exception as exc:
        raise TensorRTLLMRunnerError(
            "tokenizer contract cannot initialize the pinned tokenizer API"
        ) from exc
    return tokenizer


def _prompt_token_ids(
    tokenizer: object,
    input_text: str,
    *,
    add_special_tokens: bool,
) -> tuple[int, ...]:
    try:
        encode = getattr(tokenizer, "encode")  # noqa: B009
        token_ids = encode(
            input_text,
            add_special_tokens=add_special_tokens,
        )
    except Exception as exc:
        raise TensorRTLLMRunnerError("prompt tokenization failed") from exc
    if not isinstance(token_ids, list) or any(
        isinstance(token_id, bool) or not isinstance(token_id, int)
        for token_id in token_ids
    ):
        raise TensorRTLLMRunnerError("prompt tokenization returned invalid IDs")
    return tuple(token_ids)


def _sampling_parameters(
    backend: _Backend,
    tokenizer_contract: _TokenizerContract,
    settings: _ExecutionSettings,
) -> object:
    return backend.sampling_params(
        add_special_tokens=tokenizer_contract.add_special_tokens,
        best_of=1,
        detokenize=True,
        end_id=tokenizer_contract.eos_token_id,
        exclude_input_from_output=True,
        max_tokens=settings.max_output_tokens,
        n=1,
        pad_id=tokenizer_contract.pad_token_id,
        seed=settings.seed,
        skip_special_tokens=tokenizer_contract.skip_special_tokens,
        temperature=0.0,
        top_k=1,
        use_beam_search=False,
    )


def _validated_generation_output(
    output: object,
    *,
    expected_prompt: str | None = None,
    expected_prompt_token_ids: tuple[int, ...] | None = None,
) -> str:
    if getattr(output, "finished", None) is not True:
        raise TensorRTLLMRunnerError("generation did not finish")
    if expected_prompt is not None:
        if getattr(output, "prompt", None) != expected_prompt:
            raise TensorRTLLMRunnerError(
                "batched generation output prompt order does not match the request"
            )
        observed_prompt_token_ids = getattr(output, "prompt_token_ids", None)
        if (
            not isinstance(observed_prompt_token_ids, list)
            or any(
                isinstance(token_id, bool) or not isinstance(token_id, int)
                for token_id in observed_prompt_token_ids
            )
            or tuple(observed_prompt_token_ids) != expected_prompt_token_ids
        ):
            raise TensorRTLLMRunnerError(
                "batched generation output prompt tokens do not match the request"
            )
    completions = getattr(output, "outputs", None)
    if not isinstance(completions, list) or len(completions) != 1:
        raise TensorRTLLMRunnerError(
            "generation returned an unsupported completion count"
        )
    text = getattr(completions[0], "text", None)
    try:
        text = exact_match_output_text(text)
    except ValueError as exc:
        raise TensorRTLLMRunnerError(
            "generation output is not valid user-visible text"
        ) from exc
    if len(text.encode("utf-8")) > _MAX_OUTPUT_BYTES:
        raise TensorRTLLMRunnerError("generation output exceeds the byte limit")
    return text


def _execute_prompts(
    *,
    engine_bundle: Path,
    tokenizer_contract: _TokenizerContract,
    engine_config: Mapping[str, object],
    settings: _ExecutionSettings,
    prompts: tuple[str, ...],
    batched: bool,
    backend: _Backend | None,
) -> tuple[str, ...]:
    _require_runtime_boundary()
    if os.environ.get("FORCE_DETERMINISTIC") != "1":
        raise TensorRTLLMRunnerError(
            "TensorRT-LLM scoring requires FORCE_DETERMINISTIC=1"
        )
    os.environ["TLLM_LOG_LEVEL"] = "error"
    llm: object | None = None
    try:
        with _silence_backend_output():
            selected_backend = backend if backend is not None else _load_backend()
            tokenizer = _tokenizer_from_contract(tokenizer_contract, selected_backend)
            _max_batch, _max_input, max_seq_len = _engine_limits(engine_config)
            prompt_token_ids: list[tuple[int, ...]] = []
            for prompt in prompts:
                token_ids = _prompt_token_ids(
                    tokenizer,
                    prompt,
                    add_special_tokens=tokenizer_contract.add_special_tokens,
                )
                prompt_token_ids.append(token_ids)
                if len(token_ids) > settings.context_length:
                    raise TensorRTLLMRunnerError(
                        "prompt exceeds the authenticated context length"
                    )
                if len(token_ids) + settings.max_output_tokens > max_seq_len:
                    raise TensorRTLLMRunnerError(
                        "prompt and output exceed the engine sequence limit"
                    )
            llm = selected_backend.llm(
                model=engine_bundle,
                tokenizer=tokenizer,
                tokenizer_mode="auto",
                skip_tokenizer_init=False,
                trust_remote_code=False,
                tensor_parallel_size=1,
            )
            sampling = _sampling_parameters(
                selected_backend, tokenizer_contract, settings
            )
            generate = getattr(llm, "generate")  # noqa: B009
            output = generate(
                list(prompts) if batched else prompts[0],
                sampling_params=sampling,
                use_tqdm=False,
            )
            if not batched and isinstance(output, list):
                raise TensorRTLLMRunnerError(
                    "single-record generation returned a batched response"
                )
            if batched:
                if not isinstance(output, list) or len(output) != len(prompts):
                    raise TensorRTLLMRunnerError(
                        "batched generation output count does not match the request"
                    )
                return tuple(
                    _validated_generation_output(
                        item,
                        expected_prompt=prompt,
                        expected_prompt_token_ids=token_ids,
                    )
                    for item, prompt, token_ids in zip(
                        output, prompts, prompt_token_ids, strict=True
                    )
                )
            return (_validated_generation_output(output),)
    except TensorRTLLMRunnerError:
        raise
    except Exception as exc:
        raise TensorRTLLMRunnerError("TensorRT-LLM execution failed") from exc
    finally:
        if llm is not None:
            try:
                shutdown = getattr(llm, "shutdown")  # noqa: B009
                with _silence_backend_output():
                    shutdown()
            except Exception:
                pass


def _execute_request(request: _ScoreRequest, *, backend: _Backend | None = None) -> str:
    return _execute_prompts(
        engine_bundle=request.engine_bundle,
        tokenizer_contract=request.tokenizer_contract,
        engine_config=request.engine_config,
        settings=request.settings,
        prompts=(request.input_text,),
        batched=False,
        backend=backend,
    )[0]


def _execute_batch_request(
    request: _BatchScoreRequest,
    *,
    backend: _Backend | None = None,
) -> tuple[tuple[str, str], ...]:
    outputs = _execute_prompts(
        engine_bundle=request.engine_bundle,
        tokenizer_contract=request.tokenizer_contract,
        engine_config=request.engine_config,
        settings=request.settings,
        prompts=tuple(record.input_text for record in request.records),
        batched=True,
        backend=backend,
    )
    if len(outputs) != len(request.records):
        raise TensorRTLLMRunnerError(
            "batched generation output count does not match the request"
        )
    return tuple(
        (record.record_id, output)
        for record, output in zip(request.records, outputs, strict=True)
    )


def _info_payload() -> dict[str, str]:
    _require_runtime_boundary()
    with _silence_backend_output():
        backend_build_sha256 = _observed_backend_build_sha256()
        device = _observe_cuda_device()
    return {
        "backend_build_sha256": backend_build_sha256,
        "backend_name": "TensorRT-LLM",
        "backend_version": _BACKEND_VERSION,
        "cuda_compute_capability": device.compute_capability,
        "cuda_device_name": device.device_name,
        "cuda_driver_version": device.driver_version,
        "cuda_runtime_version": device.cuda_runtime_version,
        "device_kind": "cuda",
        "format_version": _INFO_FORMAT_VERSION,
        "protocol_version": _PROTOCOL_VERSION,
    }


def _write_json(
    value: Mapping[str, object],
    *,
    max_bytes: int | None = None,
) -> None:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    if max_bytes is not None and len(encoded.encode("utf-8")) + 1 > max_bytes:
        raise TensorRTLLMRunnerError("batch runner response exceeds the byte limit")
    sys.stdout.write(encoded + "\n")
    sys.stdout.flush()


def main(argv: Sequence[str] | None = None) -> int:
    arguments = tuple(sys.argv[1:] if argv is None else argv)
    try:
        if arguments == ("--invarlock-runtime-info-v1",):
            _write_json(_info_payload())
            return 0
        if arguments not in {
            ("--invarlock-score-v1",),
            ("--invarlock-score-batch-v1",),
        }:
            return 64
        payload = _read_bounded(
            sys.stdin.buffer, _MAX_REQUEST_BYTES, label="runner request"
        )
        if arguments == ("--invarlock-score-batch-v1",):
            batch_request = _parse_batch_request(payload)
            outputs = _execute_batch_request(batch_request)
            _write_json(
                {
                    "format_version": _BATCH_RESPONSE_FORMAT_VERSION,
                    "outputs": [
                        {"output_text": output_text, "record_id": record_id}
                        for record_id, output_text in outputs
                    ],
                },
                max_bytes=_MAX_BATCH_RESPONSE_BYTES,
            )
            return 0
        request = _parse_request(payload)
        output_text = _execute_request(request)
        _write_json(
            {
                "format_version": _RESPONSE_FORMAT_VERSION,
                "output_text": output_text,
            }
        )
        return 0
    except TensorRTLLMRunnerError as exc:
        sys.stderr.write(f"TensorRT-LLM runner failed closed: {exc}\n")
        sys.stderr.flush()
        return 70
    except Exception:
        sys.stderr.write("TensorRT-LLM runner failed closed\n")
        sys.stderr.flush()
        return 70


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["TensorRTLLMRunnerError", "main"]
