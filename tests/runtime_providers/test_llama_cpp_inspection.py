from __future__ import annotations

import hashlib
import struct
from pathlib import Path

import pytest

from invarlock.core.runtime_provider import ModelRuntimeSpec
from invarlock.runtime_providers import llama_cpp, llama_cpp_session
from invarlock.runtime_providers.gguf_identity import read_gguf_artifact_identity
from invarlock.runtime_providers.llama_cpp import (
    LlamaCppProvider,
    LlamaCppRuntimeBindings,
)
from invarlock.runtime_providers.llama_cpp_session import LlamaCppExecutionError

_BACKEND_VERSION = "version: 4242 (test) built with TestCompiler for TestOS"
_IMAGE_DIGEST = "sha256:" + "a" * 64


@pytest.fixture(autouse=True)
def _authenticated_inspection_boundary(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INVARLOCK_RUNTIME_IMAGE_DIGEST", _IMAGE_DIGEST)
    monkeypatch.setenv("INVARLOCK_RUNTIME_IMAGE", "invarlock-runtime@" + _IMAGE_DIGEST)
    monkeypatch.setattr(llama_cpp, "strict_container_boundary_present", lambda: True)
    monkeypatch.setattr(llama_cpp, "_require_isolated_network_namespace", lambda: None)


def _string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def _metadata(key: str, value_type: int, value: bytes) -> bytes:
    return _string(key) + struct.pack("<I", value_type) + value


def _gguf_fixture() -> bytes:
    metadata = [
        _metadata("general.architecture", 8, _string("llama")),
        _metadata("general.alignment", 4, struct.pack("<I", 32)),
        _metadata("tokenizer.ggml.model", 8, _string("llama")),
        _metadata(
            "tokenizer.ggml.tokens",
            9,
            struct.pack("<IQ", 8, 2) + _string("a") + _string("b"),
        ),
    ]
    tensor = _string("token_embd.weight") + struct.pack("<IQQIQ", 2, 2, 2, 0, 0)
    header = (
        b"GGUF" + struct.pack("<IQQ", 3, 1, len(metadata)) + b"".join(metadata) + tensor
    )
    padding = b"\x00" * ((32 - len(header) % 32) % 32)
    return header + padding + b"\x00" * 16


def _runtime_inputs(tmp_path: Path) -> tuple[ModelRuntimeSpec, LlamaCppRuntimeBindings]:
    model_path = tmp_path / "private-model-name.gguf"
    model_path.write_bytes(_gguf_fixture())
    executable_path = tmp_path / "private-llama-completion"
    executable_path.write_bytes(b"authenticated llama.cpp executable")
    executable_path.chmod(0o700)
    source_archive_path = tmp_path / "private-llama-cpp-source.tar"
    source_archive_path.write_bytes(b"llama.cpp-source-commit-4242")
    identity = read_gguf_artifact_identity(model_path)
    expected = ModelRuntimeSpec(
        provider_name="llama_cpp",
        model_id=identity.artifact_name,
        settings={
            "artifact_sha256": identity.sha256,
            "artifact_byte_length": identity.byte_length,
            "gguf_metadata_sha256": identity.gguf_metadata_sha256,
            "tensor_inventory_sha256": identity.tensor_inventory_sha256,
            "tokenizer_metadata_sha256": identity.tokenizer_metadata_sha256,
            "backend_binary_sha256": hashlib.sha256(
                executable_path.read_bytes()
            ).hexdigest(),
            "backend_source_sha256": hashlib.sha256(
                source_archive_path.read_bytes()
            ).hexdigest(),
            "backend_version": _BACKEND_VERSION,
            "seed": 7,
            "context_length": 256,
            "batch_size": 32,
            "max_output_tokens": 16,
            "timeout_seconds": 1,
        },
    )
    return expected, LlamaCppRuntimeBindings(
        gguf_path=model_path,
        executable_path=executable_path,
        source_archive_path=source_archive_path,
    )


def test_inspection_derives_complete_path_free_spec(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected, bindings = _runtime_inputs(tmp_path)
    monkeypatch.setattr(
        llama_cpp_session,
        "probe_llama_cpp_version",
        lambda _executable, _run_directory: _BACKEND_VERSION,
    )

    observed = LlamaCppProvider().inspect_runtime_spec(
        bindings,
        seed=7,
        context_length=256,
        batch_size=32,
        max_output_tokens=16,
        timeout_seconds=1,
    )

    assert observed == expected
    assert all(str(tmp_path) not in str(value) for value in observed.settings.values())


def test_inspection_rejects_artifact_changed_during_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _expected, bindings = _runtime_inputs(tmp_path)
    monkeypatch.setattr(
        llama_cpp_session,
        "probe_llama_cpp_version",
        lambda _executable, _run_directory: _BACKEND_VERSION,
    )
    inspect_backend = llama_cpp.inspect_llama_cpp_backend

    def mutate_after_backend_probe(
        runtime_bindings: LlamaCppRuntimeBindings,
    ) -> llama_cpp_session.LlamaCppBackendInspection:
        inspection = inspect_backend(runtime_bindings)
        payload = bytearray(runtime_bindings.gguf_path.read_bytes())
        payload[-1] ^= 1
        runtime_bindings.gguf_path.write_bytes(payload)
        return inspection

    monkeypatch.setattr(
        llama_cpp,
        "inspect_llama_cpp_backend",
        mutate_after_backend_probe,
    )

    with pytest.raises(ValueError, match="changed during runtime inspection"):
        LlamaCppProvider().inspect_runtime_spec(
            bindings,
            seed=7,
            context_length=256,
            batch_size=1,
            max_output_tokens=16,
            timeout_seconds=1,
        )


def test_inspection_rejects_host_execution_before_version_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _expected, bindings = _runtime_inputs(tmp_path)
    probes: list[object] = []
    monkeypatch.setattr(llama_cpp, "strict_container_boundary_present", lambda: False)
    monkeypatch.setattr(
        llama_cpp_session,
        "probe_llama_cpp_version",
        lambda *_args: probes.append(object()) or _BACKEND_VERSION,
    )

    with pytest.raises(ValueError, match="authenticated container boundary"):
        LlamaCppProvider().inspect_runtime_spec(
            bindings,
            seed=7,
            context_length=256,
            batch_size=1,
            max_output_tokens=16,
            timeout_seconds=1,
        )

    assert probes == []


def test_backend_version_accepts_pinned_b10015_shape() -> None:
    value = (
        "version: 10015 (12127defda4f41b7679cb2477a4b0d65ee6a0c8f) "
        "built with GNU 12.2.0 for Linux x86_64"
    )

    assert llama_cpp_session.validate_llama_cpp_backend_version(value) == value


@pytest.mark.parametrize(
    "value",
    [
        "version: 10015 (abcdef0) built with /private/build/cc for Linux x86_64",
        r"version: 10015 (abcdef0) built with C:\\private\\cc for Linux x86_64",
        "version: 10015 (abcdef0) built with https://example.test/cc for Linux",
        "version: 10015 (abcdef0) built with cc\nsecret=exposed for Linux",
        "version: 10015 (abcdef0) built with api_key=exposed for Linux",
        "version: 10015 (abcdef0) built with cc for Linux \u202e",
    ],
)
def test_backend_version_rejects_private_or_noncanonical_text(value: str) -> None:
    with pytest.raises(ValueError, match="backend version"):
        llama_cpp_session.validate_llama_cpp_backend_version(value)


def test_version_probe_rejects_extra_diagnostic_lines() -> None:
    with pytest.raises(LlamaCppExecutionError, match="exact version/build lines"):
        llama_cpp_session._normalize_version_output(  # noqa: SLF001
            b"version: 10015 (abcdef0)\nbuilt with cc for Linux x86_64\nwarning",
            b"",
        )
