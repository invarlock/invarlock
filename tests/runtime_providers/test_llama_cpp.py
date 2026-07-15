from __future__ import annotations

import hashlib
import json
import struct
import sys
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from invarlock.core.runtime_provider import (
    EvaluationBatch,
    EvaluationRecord,
    ModelRuntimeSpec,
    RuntimeExecutionContext,
    RuntimeProvider,
    RuntimeSession,
    artifact_identity_sha256,
)
from invarlock.reporting.validation.runtime_behavioral_observation import (
    runtime_scoring_records_sha256,
)
from invarlock.runtime_providers import llama_cpp, llama_cpp_session
from invarlock.runtime_providers.gguf_identity import read_gguf_artifact_identity
from invarlock.runtime_providers.llama_cpp import (
    LlamaCppProvider,
    LlamaCppRuntimeBindings,
)
from invarlock.runtime_providers.llama_cpp_session import LlamaCppExecutionError

_IMAGE_DIGEST = "sha256:" + "a" * 64
_BACKEND_VERSION = "version: 4242 (test) built with TestCompiler for TestOS"
_REQUIRES_FD_EXECUTION = pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="descriptor-backed executable launch requires Linux /proc/self/fd",
)
_REQUIRE_ISOLATED_NETWORK_NAMESPACE = llama_cpp._require_isolated_network_namespace
_OBSERVE_LINUX_CPU = llama_cpp._observe_linux_cpu


@pytest.fixture(autouse=True)
def _authenticated_test_container(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INVARLOCK_CONTAINER_EXECUTION", "1")
    monkeypatch.setenv("INVARLOCK_RUNTIME_IMAGE_DIGEST", _IMAGE_DIGEST)
    monkeypatch.setenv("INVARLOCK_RUNTIME_IMAGE", "invarlock-runtime@" + _IMAGE_DIGEST)
    monkeypatch.setattr(llama_cpp, "strict_container_boundary_present", lambda: True)
    monkeypatch.setattr(llama_cpp, "_require_isolated_network_namespace", lambda: None)
    monkeypatch.setattr(
        llama_cpp,
        "_observe_linux_cpu",
        lambda: llama_cpp.RuntimeDeviceFacts(
            device_kind="cpu",
            device_name="observed test CPU",
        ),
    )


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


def _write_fake_llama_completion(path: Path) -> None:
    path.write_text(
        f"""#!{sys.executable}
import json
import os
import sys
import time

if "--version" in sys.argv:
    print("version: 4242 (test)")
    print("built with TestCompiler for TestOS")
    raise SystemExit(0)

prompt = sys.stdin.read()

def write_output(payload, *, framed=True):
    os.write(1, payload)
    if framed:
        os.write(1, b"\\n\\n")

if prompt == "__sleep__":
    time.sleep(30)
elif prompt == "__flood__":
    os.write(1, b"x" * (2 * 1024 * 1024))
elif prompt == "__stderr__":
    os.write(2, b"unexpected diagnostic")
elif prompt == "__fail__":
    raise SystemExit(7)
elif prompt == "__invalid_utf8__":
    write_output(b"\\xff")
elif prompt == "__bad_framing__":
    write_output(b"bad", framed=False)
elif prompt == "__argv__":
    write_output(json.dumps(sys.argv[1:], separators=(",", ":")).encode("utf-8"))
elif prompt == "__env__":
    write_output(json.dumps(dict(os.environ), sort_keys=True, separators=(",", ":")).encode("utf-8"))
else:
    write_output(("OUT:" + prompt).encode("utf-8"))
""",
        encoding="utf-8",
    )
    path.chmod(0o700)


def _runtime_inputs(
    tmp_path: Path,
) -> tuple[
    ModelRuntimeSpec,
    LlamaCppRuntimeBindings,
    RuntimeExecutionContext,
]:
    model_path = tmp_path / "private-model-name.gguf"
    model_path.write_bytes(_gguf_fixture())
    executable_path = tmp_path / "private-llama-completion"
    _write_fake_llama_completion(executable_path)
    source_archive_path = tmp_path / "private-llama-cpp-source.tar"
    source_archive_path.write_bytes(b"llama.cpp-source-commit-4242")
    identity = read_gguf_artifact_identity(model_path)
    executable_sha256 = hashlib.sha256(executable_path.read_bytes()).hexdigest()
    source_sha256 = hashlib.sha256(source_archive_path.read_bytes()).hexdigest()
    spec = ModelRuntimeSpec(
        provider_name="llama_cpp",
        model_id=identity.artifact_name,
        settings={
            "artifact_sha256": identity.sha256,
            "artifact_byte_length": identity.byte_length,
            "gguf_metadata_sha256": identity.gguf_metadata_sha256,
            "tensor_inventory_sha256": identity.tensor_inventory_sha256,
            "tokenizer_metadata_sha256": identity.tokenizer_metadata_sha256,
            "backend_binary_sha256": executable_sha256,
            "backend_source_sha256": source_sha256,
            "backend_version": _BACKEND_VERSION,
            "seed": 7,
            "context_length": 256,
            "batch_size": 32,
            "max_output_tokens": 16,
            "timeout_seconds": 1,
        },
    )
    bindings = LlamaCppRuntimeBindings(
        gguf_path=model_path,
        executable_path=executable_path,
        source_archive_path=source_archive_path,
    )
    context = RuntimeExecutionContext(
        strict=True,
        allow_network=False,
        container_image_digest=_IMAGE_DIGEST,
        device_kind="cpu",
        artifact_identity_sha256=artifact_identity_sha256(identity),
        native_model=bindings,
    )
    return spec, bindings, context


def _record(record_id: str, text: str) -> EvaluationRecord:
    return EvaluationRecord(
        record_id=record_id,
        input_text=text,
        input_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        expected_output=f"OUT:{text}",
    )


def _batch(*records: EvaluationRecord) -> EvaluationBatch:
    return EvaluationBatch(schedule_sha256="b" * 64, records=tuple(records))


def test_llama_cpp_observes_linux_cpu_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cpuinfo = tmp_path / "cpuinfo"
    payload = b"processor: 0\nmodel name: Test CPU 9000\nfeatures: test\n"
    cpuinfo.write_bytes(payload)
    monkeypatch.setattr(llama_cpp.platform, "system", lambda: "Linux")
    monkeypatch.setattr(llama_cpp, "_CPU_INFO_PATH", cpuinfo)
    monkeypatch.setattr(
        llama_cpp.os, "uname", lambda: type("U", (), {"machine": "x86_64"})()
    )

    facts = _OBSERVE_LINUX_CPU()

    assert facts.device_kind == "cpu"
    canonical_identity = b'{"fields":{"features":["test"],"model name":["Test CPU 9000"]},"machine":"x86_64"}'
    assert facts.device_name == (
        "Test CPU 9000 [x86_64; cpu_identity_sha256="
        + hashlib.sha256(canonical_identity).hexdigest()
        + "]"
    )


def test_llama_cpp_cpu_identity_ignores_transient_frequency(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cpuinfo = tmp_path / "cpuinfo"
    monkeypatch.setattr(llama_cpp.platform, "system", lambda: "Linux")
    monkeypatch.setattr(llama_cpp, "_CPU_INFO_PATH", cpuinfo)
    monkeypatch.setattr(
        llama_cpp.os, "uname", lambda: type("U", (), {"machine": "x86_64"})()
    )

    cpuinfo.write_text(
        "processor: 0\nmodel name: Stable CPU\ncpu MHz: 800.125\nflags: a b\n",
        encoding="utf-8",
    )
    first = _OBSERVE_LINUX_CPU()
    cpuinfo.write_text(
        "processor: 0\nmodel name: Stable CPU\ncpu MHz: 4200.875\nflags: a b\n",
        encoding="utf-8",
    )
    second = _OBSERVE_LINUX_CPU()

    assert first == second


def test_llama_cpp_rejects_unobservable_cpu_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "cpuinfo-target"
    target.write_text("model name: Test CPU\n", encoding="utf-8")
    link = tmp_path / "cpuinfo"
    link.symlink_to(target)
    monkeypatch.setattr(llama_cpp.platform, "system", lambda: "Linux")
    monkeypatch.setattr(llama_cpp, "_CPU_INFO_PATH", link)

    with pytest.raises(ValueError, match="cannot observe"):
        _OBSERVE_LINUX_CPU()


def test_llama_cpp_config_identity_capabilities_and_private_bindings(
    tmp_path: Path,
) -> None:
    spec, bindings, _context = _runtime_inputs(tmp_path)
    provider = LlamaCppProvider()

    provider.validate_config(spec)
    identity = provider.identify_artifact(spec)

    assert isinstance(provider, RuntimeProvider)
    assert identity.artifact_name == spec.model_id
    assert identity.sha256 == spec.settings["artifact_sha256"]
    assert provider.capabilities().artifact_formats == ("gguf",)
    assert provider.capabilities().metrics == ("exact_match",)
    assert provider.capabilities().execution_modes == ("container", "local_process")
    assert provider.capabilities().platform_constraints == (
        "linux",
        "cpu",
        "llama_cpp_b10015",
    )
    assert "unpinned_outer_runtime" in provider.capabilities().degraded_modes
    assert provider.capabilities().unavailable_modes == (
        "gpu_execution",
        "networked_execution",
        "non_linux_execution",
    )
    assert str(bindings.gguf_path) not in repr(bindings)
    assert str(bindings.executable_path) not in repr(bindings)
    assert str(bindings.source_archive_path) not in repr(bindings)
    assert all("path" not in name for name in spec.settings)


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        ({"strict": True, "allow_network": True}, "disable network"),
        ({"strict": True, "container_image_digest": None}, "container image"),
        ({"strict": True, "artifact_identity_sha256": None}, "artifact identity"),
        ({"strict": True, "artifact_identity_sha256": "0" * 64}, "does not match"),
        ({"strict": True, "device_kind": "cuda"}, "CPU device"),
        ({"strict": True, "native_model": None}, "runtime bindings"),
    ],
)
def test_llama_cpp_strict_open_rejects_missing_security_bindings(
    tmp_path: Path, replacement: dict[str, object], message: str
) -> None:
    spec, _bindings, context = _runtime_inputs(tmp_path)
    context = replace(context, **replacement)

    with pytest.raises((ValueError, LlamaCppExecutionError), match=message):
        LlamaCppProvider().open(spec, context)


def test_llama_cpp_strict_open_rejects_host_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec, _bindings, context = _runtime_inputs(tmp_path)
    monkeypatch.delenv("INVARLOCK_CONTAINER_EXECUTION", raising=False)
    monkeypatch.setattr(llama_cpp, "strict_container_boundary_present", lambda: False)

    with pytest.raises(ValueError, match="container boundary"):
        LlamaCppProvider().open(spec, context)


@pytest.mark.parametrize(
    ("runtime_image_digest", "message"),
    [
        (None, "canonical INVARLOCK_RUNTIME_IMAGE_DIGEST"),
        ("SHA256:" + "a" * 64, "canonical INVARLOCK_RUNTIME_IMAGE_DIGEST"),
        ("sha256:" + "b" * 64, "does not match"),
    ],
)
def test_llama_cpp_strict_open_authenticates_runtime_image_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    runtime_image_digest: str | None,
    message: str,
) -> None:
    spec, _bindings, context = _runtime_inputs(tmp_path)
    if runtime_image_digest is None:
        monkeypatch.delenv("INVARLOCK_RUNTIME_IMAGE_DIGEST", raising=False)
    else:
        monkeypatch.setenv("INVARLOCK_RUNTIME_IMAGE_DIGEST", runtime_image_digest)

    with pytest.raises(ValueError, match=message):
        LlamaCppProvider().open(spec, context)


@pytest.mark.parametrize(
    "runtime_image",
    [
        "",
        "invarlock-runtime:mutable",
        "invarlock-runtime@sha256:" + "b" * 64,
    ],
)
def test_llama_cpp_strict_open_authenticates_runtime_image_reference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    runtime_image: str,
) -> None:
    spec, _bindings, context = _runtime_inputs(tmp_path)
    monkeypatch.setenv("INVARLOCK_RUNTIME_IMAGE", runtime_image)

    with pytest.raises(ValueError, match="INVARLOCK_RUNTIME_IMAGE"):
        LlamaCppProvider().open(spec, context)


def _write_route_tables(
    root: Path, *, ipv4_interface: str | None, ipv6_interface: str | None
) -> tuple[Path, Path]:
    ipv4 = root / "route"
    ipv6 = root / "ipv6_route"
    header = "Iface Destination Gateway Flags RefCnt Use Metric Mask MTU Window IRTT\n"
    row = ""
    if ipv4_interface is not None:
        row = f"{ipv4_interface} 00000000 00000000 0001 0 0 0 00000000 0 0 0\n"
    ipv4.write_text(header + row, encoding="ascii")
    ipv6_row = ""
    if ipv6_interface is not None:
        ipv6_row = (
            "00000000000000000000000000000001 80 "
            "00000000000000000000000000000000 00 "
            "00000000000000000000000000000000 "
            f"00000000 00000000 00000000 00000001 {ipv6_interface}\n"
        )
    ipv6.write_text(ipv6_row, encoding="ascii")
    return ipv4, ipv6


@pytest.mark.parametrize(
    ("ipv4_interface", "ipv6_interface"),
    [("eth0", None), (None, "eth0")],
)
def test_llama_cpp_strict_open_rejects_routable_network_namespace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ipv4_interface: str | None,
    ipv6_interface: str | None,
) -> None:
    spec, _bindings, context = _runtime_inputs(tmp_path)
    ipv4, ipv6 = _write_route_tables(
        tmp_path, ipv4_interface=ipv4_interface, ipv6_interface=ipv6_interface
    )
    monkeypatch.setattr(
        llama_cpp,
        "_require_isolated_network_namespace",
        lambda: _REQUIRE_ISOLATED_NETWORK_NAMESPACE(
            ipv4_route_path=ipv4, ipv6_route_path=ipv6
        ),
    )

    with pytest.raises(ValueError, match="network-disabled container"):
        LlamaCppProvider().open(spec, context)


def test_llama_cpp_strict_open_accepts_loopback_only_network_namespace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec, _bindings, context = _runtime_inputs(tmp_path)
    ipv4, ipv6 = _write_route_tables(tmp_path, ipv4_interface="lo", ipv6_interface="lo")
    monkeypatch.setattr(
        llama_cpp,
        "_require_isolated_network_namespace",
        lambda: _REQUIRE_ISOLATED_NETWORK_NAMESPACE(
            ipv4_route_path=ipv4, ipv6_route_path=ipv6
        ),
    )
    sentinel = object()
    monkeypatch.setattr(llama_cpp, "LlamaCppSession", lambda _config: sentinel)

    assert LlamaCppProvider().open(spec, context) is sentinel


@_REQUIRES_FD_EXECUTION
def test_llama_cpp_strict_open_authenticates_executable_and_version(
    tmp_path: Path,
) -> None:
    spec, bindings, context = _runtime_inputs(tmp_path)

    bad_digest = ModelRuntimeSpec(
        provider_name=spec.provider_name,
        model_id=spec.model_id,
        settings={**spec.settings, "backend_binary_sha256": "0" * 64},
    )
    with pytest.raises(LlamaCppExecutionError, match="digest"):
        LlamaCppProvider().open(bad_digest, context)

    bad_version = ModelRuntimeSpec(
        provider_name=spec.provider_name,
        model_id=spec.model_id,
        settings={
            **spec.settings,
            "backend_version": (
                "version: 4243 (test) built with TestCompiler for TestOS"
            ),
        },
    )
    with pytest.raises(LlamaCppExecutionError, match="version"):
        LlamaCppProvider().open(bad_version, context)

    bindings.gguf_path.write_bytes(_gguf_fixture() + b"changed")
    with pytest.raises((ValueError, LlamaCppExecutionError), match="identity|digest"):
        LlamaCppProvider().open(spec, context)


def test_llama_cpp_rejects_wrong_source_archive_digest(tmp_path: Path) -> None:
    spec, _bindings, context = _runtime_inputs(tmp_path)
    bad_source = ModelRuntimeSpec(
        provider_name=spec.provider_name,
        model_id=spec.model_id,
        settings={**spec.settings, "backend_source_sha256": "0" * 64},
    )

    with pytest.raises(LlamaCppExecutionError, match="digest"):
        LlamaCppProvider().open(bad_source, context)


@_REQUIRES_FD_EXECUTION
def test_llama_cpp_scores_in_order_and_emits_bound_receipt(tmp_path: Path) -> None:
    spec, _bindings, context = _runtime_inputs(tmp_path)
    provider = LlamaCppProvider()
    session = provider.open(spec, context)

    with pytest.raises(RuntimeError, match="unavailable before scoring"):
        session.runtime_receipt()
    observation = session.score(_batch(_record("a", "alpha"), _record("b", "beta")))
    receipt = session.runtime_receipt()

    assert isinstance(session, RuntimeSession)
    assert tuple(record.record_id for record in observation.records) == ("a", "b")
    assert tuple(record.output_text for record in observation.records) == (
        "OUT:alpha",
        "OUT:beta",
    )
    for record in observation.records:
        assert (
            record.output_sha256
            == hashlib.sha256(record.output_text.encode("utf-8")).hexdigest()
        )
    assert observation.aggregate_source_sha256 == runtime_scoring_records_sha256(
        [asdict(record) for record in observation.records]
    )
    assert receipt.backend.name == "llama.cpp"
    assert receipt.backend.version == _BACKEND_VERSION
    assert receipt.backend.binary_sha256 == spec.settings["backend_binary_sha256"]
    assert receipt.backend.source_sha256 == spec.settings["backend_source_sha256"]
    assert receipt.artifact_identity == provider.identify_artifact(spec)
    assert receipt.outer_image_digest == _IMAGE_DIGEST
    assert receipt.device.device_kind == "cpu"
    assert "cpu_identity_sha256=" in receipt.device.device_name
    assert receipt.device.compute_capability is None
    assert receipt.device.driver_version is None
    assert (
        receipt.scoring_observation_sha256
        == hashlib.sha256(
            json.dumps(
                asdict(observation),
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
    )
    session.close()
    session.close()
    with pytest.raises(RuntimeError, match="closed"):
        session.score(_batch(_record("c", "gamma")))


@_REQUIRES_FD_EXECUTION
def test_llama_cpp_uses_fixed_argv_and_sanitized_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec, _bindings, context = _runtime_inputs(tmp_path)
    popen_calls: list[dict[str, object]] = []
    real_popen = llama_cpp_session.subprocess.Popen

    def recording_popen(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        popen_calls.append(dict(kwargs))
        return real_popen(*args, **kwargs)

    monkeypatch.setattr(llama_cpp_session.subprocess, "Popen", recording_popen)
    monkeypatch.setenv("LLAMA_ARG_MODEL", "untrusted-model")
    monkeypatch.setenv("HF_TOKEN", "private-token")
    monkeypatch.setenv("INVARLOCK_TEST_SECRET", "private-value")
    session = LlamaCppProvider().open(spec, context)

    argv_output = session.score(_batch(_record("argv", "__argv__"))).records[0]
    argv = json.loads(argv_output.output_text)
    assert argv == [
        "--model",
        argv[1],
        "--file",
        "/dev/stdin",
        "--seed",
        "7",
        "--ctx-size",
        "256",
        "--batch-size",
        "32",
        "--ubatch-size",
        "32",
        "--n-predict",
        "16",
        "--threads",
        "1",
        "--threads-batch",
        "1",
        "--temp",
        "0",
        "--device",
        "none",
        "--fit",
        "off",
        "--no-conversation",
        "--no-display-prompt",
        "--ignore-eos",
        "--no-warmup",
        "--no-context-shift",
        "--no-perf",
        "--no-escape",
        "--verbosity",
        "0",
        "--offline",
    ]
    assert argv[1].startswith(("/dev/fd/", "/proc/self/fd/"))

    environment_output = session.score(_batch(_record("env", "__env__"))).records[0]
    environment = json.loads(environment_output.output_text)
    assert environment["LANG"] == "C"
    assert environment["LC_ALL"] == "C"
    assert environment["NO_COLOR"] == "1"
    assert environment["HOME"] == environment["TMPDIR"]
    assert environment["HOME"] == environment["XDG_CACHE_HOME"]
    assert "LLAMA_ARG_MODEL" not in environment
    assert "HF_TOKEN" not in environment
    assert "INVARLOCK_TEST_SECRET" not in environment
    assert popen_calls
    assert all(call["shell"] is False for call in popen_calls)
    assert all(call["start_new_session"] is True for call in popen_calls)
    assert all(
        set(call["env"])
        == {
            "HOME",
            "LANG",
            "LC_ALL",
            "NO_COLOR",
            "TMPDIR",
            "XDG_CACHE_HOME",
        }
        for call in popen_calls
    )
    session.close()


@_REQUIRES_FD_EXECUTION
@pytest.mark.parametrize(
    ("prompt", "message"),
    [
        ("__sleep__", "timed out"),
        ("__flood__", "stdout limit"),
        ("__stderr__", "stderr"),
        ("__fail__", "status 7"),
        ("__invalid_utf8__", "UTF-8"),
        ("__bad_framing__", "final framing"),
    ],
)
def test_llama_cpp_fails_closed_on_subprocess_errors(
    tmp_path: Path, prompt: str, message: str
) -> None:
    spec, _bindings, context = _runtime_inputs(tmp_path)
    session = LlamaCppProvider().open(spec, context)

    with pytest.raises(LlamaCppExecutionError, match=message):
        session.score(_batch(_record("record", prompt)))

    with pytest.raises(RuntimeError, match="unavailable before scoring"):
        session.runtime_receipt()
    session.close()


@_REQUIRES_FD_EXECUTION
def test_llama_cpp_timeout_kills_the_child_process_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec, _bindings, context = _runtime_inputs(tmp_path)
    killed_pids: list[int] = []
    real_kill = llama_cpp_session._kill_process_group

    def recording_kill(process):  # noqa: ANN001, ANN202
        killed_pids.append(process.pid)
        return real_kill(process)

    monkeypatch.setattr(llama_cpp_session, "_kill_process_group", recording_kill)
    session = LlamaCppProvider().open(spec, context)

    with pytest.raises(LlamaCppExecutionError, match="timed out"):
        session.score(_batch(_record("record", "__sleep__")))

    assert killed_pids
    session.close()


@_REQUIRES_FD_EXECUTION
def test_llama_cpp_rechecks_input_and_artifact_before_and_after_score(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec, bindings, context = _runtime_inputs(tmp_path)
    session = LlamaCppProvider().open(spec, context)
    invalid_record = EvaluationRecord(
        record_id="bad",
        input_text="actual",
        input_sha256=hashlib.sha256(b"different").hexdigest(),
    )
    with pytest.raises(ValueError, match="input_sha256"):
        session.score(_batch(invalid_record))

    original_execute = session._execute_record

    def mutate_after_execution(record: EvaluationRecord) -> bytes:
        output = original_execute(record)
        bindings.gguf_path.write_bytes(_gguf_fixture() + b"mutation")
        return output

    monkeypatch.setattr(session, "_execute_record", mutate_after_execution)
    with pytest.raises((ValueError, LlamaCppExecutionError), match="identity|digest"):
        session.score(_batch(_record("record", "value")))
    with pytest.raises(RuntimeError, match="unavailable before scoring"):
        session.runtime_receipt()
    session.close()


@_REQUIRES_FD_EXECUTION
def test_llama_cpp_rechecks_source_archive_before_score(tmp_path: Path) -> None:
    spec, bindings, context = _runtime_inputs(tmp_path)
    session = LlamaCppProvider().open(spec, context)
    bindings.source_archive_path.write_bytes(b"changed-source-archive")

    with pytest.raises(LlamaCppExecutionError, match="identity|digest"):
        session.score(_batch(_record("record", "value")))

    with pytest.raises(RuntimeError, match="unavailable before scoring"):
        session.runtime_receipt()
    session.close()


@_REQUIRES_FD_EXECUTION
def test_llama_cpp_non_strict_unpinned_execution_is_explicitly_degraded(
    tmp_path: Path,
) -> None:
    spec, _bindings, context = _runtime_inputs(tmp_path)
    context = replace(
        context,
        strict=False,
        container_image_digest=None,
        artifact_identity_sha256=None,
    )
    session = LlamaCppProvider().open(spec, context)

    session.score(_batch(_record("record", "value")))
    receipt = session.runtime_receipt()

    assert receipt.outer_image_digest is None
    assert "unpinned_outer_runtime" in receipt.capabilities.degraded_modes
    session.close()


@_REQUIRES_FD_EXECUTION
def test_llama_cpp_executes_pinned_descriptor_and_fails_closed_on_path_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec, bindings, context = _runtime_inputs(tmp_path)
    session = LlamaCppProvider().open(spec, context)
    replacement_marker = tmp_path / "replacement-ran"
    replacement_path = tmp_path / "replacement-cli"
    replacement_path.write_text(
        "#!/usr/bin/env python3\n"
        "from pathlib import Path\n"
        f"Path({str(replacement_marker)!r}).write_text('ran', encoding='utf-8')\n"
        "print('REPLACEMENT')\n",
        encoding="utf-8",
    )
    replacement_path.chmod(0o700)
    real_popen = llama_cpp_session.subprocess.Popen

    def replacing_popen(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        monkeypatch.setattr(llama_cpp_session.subprocess, "Popen", real_popen)
        replacement_path.replace(bindings.executable_path)
        return real_popen(*args, **kwargs)

    monkeypatch.setattr(llama_cpp_session.subprocess, "Popen", replacing_popen)

    with pytest.raises(LlamaCppExecutionError, match="identity changed"):
        session.score(_batch(_record("record", "value")))

    assert not replacement_marker.exists()
    with pytest.raises(RuntimeError, match="unavailable before scoring"):
        session.runtime_receipt()
    session.close()


@pytest.mark.skipif(
    sys.platform.startswith("linux"),
    reason="Linux provides descriptor-backed executable launch",
)
def test_llama_cpp_fails_closed_without_descriptor_execution_support(
    tmp_path: Path,
) -> None:
    spec, _bindings, context = _runtime_inputs(tmp_path)

    with pytest.raises(LlamaCppExecutionError, match="descriptor-backed"):
        LlamaCppProvider().open(spec, context)
