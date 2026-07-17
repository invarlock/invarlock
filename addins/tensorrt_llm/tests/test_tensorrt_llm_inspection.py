from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from _support import (
    _BACKEND_BUILD_SHA256,
    _BACKEND_VERSION,
    _IMAGE_DIGEST,
    _REQUIRES_POSIX_PINNING,
    _runtime_inputs,
    _write_fake_vendor_python,
)
from invarlock_addins.tensorrt_llm import execution as tensorrt_llm_execution
from invarlock_addins.tensorrt_llm import inspection as tensorrt_llm_inspection
from invarlock_addins.tensorrt_llm import provider as tensorrt_llm_provider
from invarlock_addins.tensorrt_llm import session as tensorrt_llm_session
from invarlock_addins.tensorrt_llm.provider import TensorRTLLMProvider
from invarlock_addins.tensorrt_llm.session import (
    TensorRTLLMExecutionError,
)


@pytest.fixture(autouse=True)
def _authenticated_container_boundary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("INVARLOCK_RUNTIME_IMAGE_DIGEST", _IMAGE_DIGEST)
    monkeypatch.setenv("INVARLOCK_RUNTIME_IMAGE", f"invarlock-runtime@{_IMAGE_DIGEST}")
    monkeypatch.setattr(
        tensorrt_llm_provider,
        "strict_container_boundary_present",
        lambda: True,
    )
    vendor_python = tmp_path / "private-vendor-python"
    _write_fake_vendor_python(vendor_python)
    monkeypatch.setattr(tensorrt_llm_execution, "_VENDOR_PYTHON", vendor_python)
    monkeypatch.setattr(
        tensorrt_llm_execution,
        "_REQUIRED_EXECUTABLE_OWNER",
        (os.getuid(), os.getgid()),
    )
    monkeypatch.setattr(
        tensorrt_llm_execution,
        "_OFFICIAL_RUNNER_PATH",
        tmp_path / "private-tensorrt-runner",
        raising=False,
    )
    monkeypatch.setattr(
        tensorrt_llm_execution,
        "_require_readonly_descriptor",
        lambda *_args, **_kwargs: None,
        raising=False,
    )
    monkeypatch.setattr(
        tensorrt_llm_execution,
        "_descriptor_mount_id",
        lambda _descriptor: 533,
        raising=False,
    )
    monkeypatch.setattr(
        tensorrt_llm_execution,
        "_require_restricted_process_status",
        lambda: None,
        raising=False,
    )
    monkeypatch.setattr(
        tensorrt_llm_session,
        "_require_isolated_network_namespace",
        lambda: None,
    )


@_REQUIRES_POSIX_PINNING
def test_tensorrt_llm_inspection_derives_complete_path_free_spec(
    tmp_path: Path,
) -> None:
    expected, bindings, _context = _runtime_inputs(tmp_path)

    observed = TensorRTLLMProvider().inspect_runtime_spec(
        bindings,
        seed=7,
        context_length=256,
        batch_size=4,
        max_output_tokens=16,
        timeout_seconds=1,
    )

    assert observed == expected
    assert all(str(tmp_path) not in str(value) for value in observed.settings.values())


def test_tensorrt_llm_inspection_rejects_unexpected_runner_info(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _expected, bindings, _context = _runtime_inputs(tmp_path)
    monkeypatch.setattr(
        tensorrt_llm_session,
        "_authenticated_official_runner_info",
        lambda _path: ({"unexpected": "value"}, "a" * 64),
    )

    with pytest.raises(TensorRTLLMExecutionError, match="unexpected fields"):
        TensorRTLLMProvider().inspect_runtime_spec(
            bindings,
            seed=7,
            context_length=256,
            batch_size=1,
            max_output_tokens=16,
            timeout_seconds=1,
        )


def test_tensorrt_llm_static_inputs_remain_pinned_across_runner_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _expected, bindings, _context = _runtime_inputs(tmp_path)
    real_probe = tensorrt_llm_session._authenticated_official_runner_info  # noqa: SLF001

    def mutate_tokenizer_after_probe(path: Path) -> tuple[dict[str, object], str]:
        observed = real_probe(path)
        bindings.tokenizer_contract_path.write_text("{}", encoding="utf-8")
        return observed

    monkeypatch.setattr(
        tensorrt_llm_session,
        "_authenticated_official_runner_info",
        mutate_tokenizer_after_probe,
    )

    with pytest.raises(TensorRTLLMExecutionError, match="pinned file identity changed"):
        TensorRTLLMProvider().inspect_runtime_spec(
            bindings,
            seed=7,
            context_length=256,
            batch_size=1,
            max_output_tokens=16,
            timeout_seconds=1,
        )


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        ({"backend_name": "other"}, "pinned contract"),
        ({"cuda_device_name": ""}, "not canonical"),
        ({"backend_build_sha256": "not-a-digest"}, "build identity"),
        ({"cuda_compute_capability": "future"}, "compute capability"),
        ({"cuda_runtime_version": "unknown"}, "CUDA runtime version"),
    ],
)
def test_tensorrt_llm_inspection_rejects_noncanonical_runner_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    replacement: dict[str, str],
    message: str,
) -> None:
    _expected, bindings, _context = _runtime_inputs(tmp_path)
    info = {
        "backend_build_sha256": _BACKEND_BUILD_SHA256,
        "backend_name": "TensorRT-LLM",
        "backend_version": _BACKEND_VERSION,
        "cuda_compute_capability": "9.0",
        "cuda_device_name": "Observed NVIDIA H200",
        "cuda_driver_version": "570.00",
        "cuda_runtime_version": "12.8",
        "device_kind": "cuda",
        "format_version": "invarlock/tensorrt-llm-runner-info-v1",
        "protocol_version": "invarlock/tensorrt-llm-runner-v1",
    }
    info.update(replacement)
    monkeypatch.setattr(
        tensorrt_llm_session,
        "_authenticated_official_runner_info",
        lambda _path: (info, "a" * 64),
    )

    with pytest.raises(TensorRTLLMExecutionError, match=message):
        TensorRTLLMProvider().inspect_runtime_spec(
            bindings,
            seed=7,
            context_length=256,
            batch_size=1,
            max_output_tokens=16,
            timeout_seconds=1,
        )


@pytest.mark.parametrize(
    ("tokenizer_contract", "message"),
    [
        ({}, "fields are not closed"),
        (
            {
                "add_special_tokens": False,
                "clean_up_tokenization_spaces": False,
                "eos_token_id": 1,
                "format_version": "invarlock/tensorrt-llm-tokenizer-contract-v1",
                "pad_token_id": 0,
                "skip_special_tokens": True,
                "tokenizer_json": {
                    "replace_with_unexpected_tokenizer_json": True,
                },
            },
            "declare a non-empty version",
        ),
    ],
)
def test_tensorrt_llm_inspection_rejects_malformed_or_placeholder_tokenizer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tokenizer_contract: dict[str, object],
    message: str,
) -> None:
    _expected, bindings, _context = _runtime_inputs(tmp_path)
    bindings.tokenizer_contract_path.write_text(
        json.dumps(tokenizer_contract), encoding="utf-8"
    )
    monkeypatch.setattr(
        tensorrt_llm_session,
        "_authenticated_official_runner_info",
        lambda _path: pytest.fail("runner probe executed before static validation"),
    )

    with pytest.raises(TensorRTLLMExecutionError, match=message):
        TensorRTLLMProvider().inspect_runtime_spec(
            bindings,
            seed=7,
            context_length=256,
            batch_size=1,
            max_output_tokens=16,
            timeout_seconds=1,
        )


def test_tensorrt_llm_inspection_json_rejects_float_overflow() -> None:
    with pytest.raises(TensorRTLLMExecutionError, match="non-finite"):
        tensorrt_llm_session._strict_json_object(  # noqa: SLF001
            b'{"value":1e100000}', label="tokenizer contract"
        )


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (b"\xff", "not UTF-8"),
        (b'{"value":1,"value":2}', "duplicate key"),
        (b'{"value":NaN}', "non-finite JSON number"),
        (b"not-json", "not strict JSON"),
        (b"[]", "must be a JSON object"),
    ],
)
def test_static_inspection_json_rejects_ambiguous_inputs(
    payload: bytes, message: str
) -> None:
    with pytest.raises(TensorRTLLMExecutionError, match=message):
        tensorrt_llm_inspection._strict_json_object(  # noqa: SLF001
            payload, label="static input"
        )


def test_static_inspection_json_enforces_depth_and_item_budgets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    nested: object = {"leaf": True}
    for _ in range(66):
        nested = {"child": nested}
    with pytest.raises(TensorRTLLMExecutionError, match="nesting depth"):
        tensorrt_llm_inspection._strict_json_object(  # noqa: SLF001
            json.dumps(nested).encode(), label="static input"
        )

    monkeypatch.setattr(tensorrt_llm_inspection, "_MAX_JSON_ITEMS", 2)
    with pytest.raises(TensorRTLLMExecutionError, match="item count"):
        tensorrt_llm_inspection._strict_json_object(  # noqa: SLF001
            b'{"values":[1,2]}', label="static input"
        )


@pytest.mark.parametrize("value", [True, -1, 2**31, "1"])
def test_static_inspection_integer_bounds_are_closed(value: object) -> None:
    with pytest.raises(TensorRTLLMExecutionError, match="supported bound"):
        tensorrt_llm_inspection._nonnegative_integer(  # noqa: SLF001
            value, label="value"
        )
    with pytest.raises(TensorRTLLMExecutionError, match="supported bound"):
        tensorrt_llm_inspection._positive_integer(  # noqa: SLF001
            value, label="value", maximum=16
        )


def _valid_tokenizer_contract() -> dict[str, object]:
    return {
        "add_special_tokens": False,
        "clean_up_tokenization_spaces": False,
        "eos_token_id": 1,
        "format_version": "invarlock/tensorrt-llm-tokenizer-contract-v1",
        "pad_token_id": 0,
        "skip_special_tokens": True,
        "tokenizer_json": {"model": {"type": "BPE"}, "version": "1.0"},
    }


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("format_version",), "unknown", "version is unsupported"),
        (("add_special_tokens",), True, "add_special_tokens=false"),
        (("skip_special_tokens",), False, "skip_special_tokens=true"),
        (
            ("clean_up_tokenization_spaces",),
            True,
            "clean_up_tokenization_spaces=false",
        ),
        (("eos_token_id",), True, "supported bound"),
        (("tokenizer_json",), {}, "non-empty object"),
        (("tokenizer_json", "version"), " ", "non-empty version"),
        (("tokenizer_json", "model"), {}, "non-empty model"),
        (("tokenizer_json", "model", "type"), " ", "non-empty type"),
    ],
)
def test_tokenizer_contract_rejects_incompatible_semantics(
    path: tuple[str, ...], value: object, message: str
) -> None:
    payload = _valid_tokenizer_contract()
    target: dict[str, object] = payload
    for component in path[:-1]:
        child = target[component]
        assert isinstance(child, dict)
        target = child
    target[path[-1]] = value

    with pytest.raises(TensorRTLLMExecutionError, match=message):
        tensorrt_llm_inspection._validate_tokenizer_contract(payload)  # noqa: SLF001


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda config: config.update(extra=True), "fields are not closed"),
        (
            lambda config: config.update(pretrained_config=[]),
            "sections must be objects",
        ),
        (
            lambda config: config["pretrained_config"].update(mapping=[]),
            "mapping must be an object",
        ),
        (
            lambda config: config["pretrained_config"]["mapping"].update(
                world_size=True
            ),
            "single-rank engines",
        ),
    ],
)
def test_engine_contract_rejects_incompatible_structure(
    mutation: Callable[[dict[str, Any]], None], message: str
) -> None:
    config: dict[str, Any] = {
        "build_config": {
            "max_batch_size": 1,
            "max_input_len": 8,
            "max_seq_len": 16,
        },
        "pretrained_config": {
            "mapping": {"cp_size": 1, "pp_size": 1, "tp_size": 1, "world_size": 1}
        },
        "version": "1.0.0",
    }
    mutation(config)

    with pytest.raises(TensorRTLLMExecutionError, match=message):
        tensorrt_llm_inspection._validate_engine_contract(config)  # noqa: SLF001


def _remove_tp_size(config: dict[str, Any]) -> None:
    config["pretrained_config"]["mapping"].pop("tp_size")


def _remove_max_input_len(config: dict[str, Any]) -> None:
    config["build_config"].pop("max_input_len")


def _set_incompatible_cp_size(config: dict[str, Any]) -> None:
    config["pretrained_config"]["mapping"]["cp_size"] = 2


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (_remove_tp_size, "single-rank"),
        (_remove_max_input_len, "max_input_len"),
        (_set_incompatible_cp_size, "single-rank"),
    ],
)
def test_tensorrt_llm_inspection_rejects_runner_incompatible_engine_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: Callable[[dict[str, Any]], None],
    message: str,
) -> None:
    _expected, bindings, _context = _runtime_inputs(tmp_path)
    config_path = bindings.engine_bundle_path / "config.json"
    config: dict[str, Any] = json.loads(config_path.read_text(encoding="utf-8"))
    mutation(config)
    config_path.write_text(
        json.dumps(config, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        tensorrt_llm_session,
        "_authenticated_official_runner_info",
        lambda _path: pytest.fail("runner probe executed before static validation"),
    )

    with pytest.raises(TensorRTLLMExecutionError, match=message):
        TensorRTLLMProvider().inspect_runtime_spec(
            bindings,
            seed=7,
            context_length=128,
            batch_size=1,
            max_output_tokens=16,
            timeout_seconds=1,
        )


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        ({"batch_size": 9}, "batch_size exceeds"),
        ({"context_length": 257}, "context_length exceeds"),
        (
            {"context_length": 256, "max_output_tokens": 257},
            "sequence limit",
        ),
    ],
)
def test_tensorrt_llm_inspection_rejects_settings_incompatible_with_engine(
    tmp_path: Path,
    settings: dict[str, int],
    message: str,
) -> None:
    _expected, bindings, _context = _runtime_inputs(tmp_path)
    arguments = {
        "seed": 7,
        "context_length": 128,
        "batch_size": 1,
        "max_output_tokens": 16,
        "timeout_seconds": 1,
        **settings,
    }

    with pytest.raises(ValueError, match=message):
        TensorRTLLMProvider().inspect_runtime_spec(bindings, **arguments)
