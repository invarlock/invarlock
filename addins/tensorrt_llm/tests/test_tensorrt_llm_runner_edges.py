from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from invarlock_addins.tensorrt_llm import runner


def test_canonical_request_and_text_identifiers_reject_unsafe_unicode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(runner.TensorRTLLMRunnerError, match="canonical JSON"):
        runner._canonical_request(b'{"value":"\\ud800"}')  # noqa: SLF001

    with pytest.raises(runner.TensorRTLLMRunnerError, match="input_text must be text"):
        runner._input_text(1)  # noqa: SLF001
    with pytest.raises(runner.TensorRTLLMRunnerError, match="valid UTF-8"):
        runner._input_text("\ud800")  # noqa: SLF001
    monkeypatch.setattr(runner, "_MAX_TEXT_BYTES", 1)
    with pytest.raises(runner.TensorRTLLMRunnerError, match="byte limit"):
        runner._input_text("ab")  # noqa: SLF001

    with pytest.raises(runner.TensorRTLLMRunnerError, match="non-empty text"):
        runner._record_id("")  # noqa: SLF001
    with pytest.raises(runner.TensorRTLLMRunnerError, match="valid UTF-8"):
        runner._record_id("\ud800")  # noqa: SLF001
    monkeypatch.setattr(runner, "_MAX_RECORD_ID_BYTES", 1)
    with pytest.raises(runner.TensorRTLLMRunnerError, match="byte limit"):
        runner._record_id("ab")  # noqa: SLF001


@pytest.mark.parametrize(
    ("payload_object", "message"),
    [
        ({}, "fields are not closed"),
        (
            {
                "engine_bundle": "/engine",
                "format_version": "wrong",
                "input_text": "prompt",
                "protocol_version": "invarlock/tensorrt-llm-runner-protocol-v1",
                "settings": {},
                "tokenizer_contract": "/tokenizer.json",
            },
            "format is unsupported",
        ),
        (
            {
                "engine_bundle": "/engine",
                "format_version": "invarlock/tensorrt-llm-runner-request-v1",
                "input_text": "prompt",
                "protocol_version": "wrong",
                "settings": {},
                "tokenizer_contract": "/tokenizer.json",
            },
            "protocol version is unsupported",
        ),
    ],
)
def test_single_request_rejects_protocol_drift(
    payload_object: dict[str, object], message: str
) -> None:
    payload = json.dumps(payload_object, sort_keys=True, separators=(",", ":")).encode()
    with pytest.raises(runner.TensorRTLLMRunnerError, match=message):
        runner._parse_request(payload)  # noqa: SLF001


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("format_version", "wrong", "format is unsupported"),
        ("protocol_version", "wrong", "protocol version is unsupported"),
    ],
)
def test_batch_request_rejects_protocol_drift(
    field: str,
    value: str,
    message: str,
) -> None:
    request = {
        "engine_bundle": "/engine",
        "format_version": "invarlock/tensorrt-llm-runner-batch-request-v1",
        "protocol_version": "invarlock/tensorrt-llm-runner-protocol-v1",
        "records": [{"input_text": "prompt", "record_id": "one"}],
        "settings": {},
        "tokenizer_contract": "/tokenizer.json",
    }
    request[field] = value
    with pytest.raises(runner.TensorRTLLMRunnerError, match=message):
        runner._parse_batch_request(  # noqa: SLF001
            json.dumps(request, sort_keys=True, separators=(",", ":")).encode()
        )


@pytest.mark.parametrize(
    ("ipv4", "ipv6", "message"),
    [
        ("", "", "IPv4 route table"),
        ("Iface Destination\neth0 short\n", "", "IPv4 route table"),
        (
            "Iface Destination Gateway Flags RefCnt Use Metric Mask MTU Window IRTT\n",
            "short ipv6\n",
            "IPv6 route table",
        ),
        (
            "Iface Destination Gateway Flags RefCnt Use Metric Mask MTU Window IRTT\n"
            "eth0 0 0 0 0 0 0 0 0 0 0\n",
            "",
            "network-disabled",
        ),
    ],
)
def test_runner_rejects_unverifiable_or_routable_network_tables(
    tmp_path: Path,
    ipv4: str,
    ipv6: str,
    message: str,
) -> None:
    ipv4_path = tmp_path / "route"
    ipv6_path = tmp_path / "ipv6_route"
    ipv4_path.write_text(ipv4, encoding="ascii")
    ipv6_path.write_text(ipv6, encoding="ascii")
    with pytest.raises(runner.TensorRTLLMRunnerError, match=message):
        runner._require_isolated_network_namespace(  # noqa: SLF001
            ipv4_route_path=ipv4_path,
            ipv6_route_path=ipv6_path,
        )


def test_runtime_boundary_and_backend_version_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import invarlock.runtime_security_helpers as security

    monkeypatch.setattr(security, "strict_container_boundary_present", lambda: False)
    with pytest.raises(runner.TensorRTLLMRunnerError, match="container boundary"):
        runner._require_runtime_boundary()  # noqa: SLF001

    calls: list[bool] = []
    monkeypatch.setattr(security, "strict_container_boundary_present", lambda: True)
    monkeypatch.setattr(
        runner,
        "_require_isolated_network_namespace",
        lambda: calls.append(True),
    )
    runner._require_runtime_boundary()  # noqa: SLF001
    assert calls == [True]

    monkeypatch.setattr(
        runner.importlib.metadata,
        "version",
        lambda _name: (_ for _ in ()).throw(
            importlib.metadata.PackageNotFoundError("missing")
        ),
    )
    with pytest.raises(runner.TensorRTLLMRunnerError, match="not installed"):
        runner._require_backend_version()  # noqa: SLF001

    monkeypatch.setattr(runner.importlib.metadata, "version", lambda _name: "wrong")
    with pytest.raises(runner.TensorRTLLMRunnerError, match="not pinned"):
        runner._require_backend_version()  # noqa: SLF001


def test_backend_file_identity_rejects_missing_empty_and_unsafe_inventory(
    tmp_path: Path,
) -> None:
    with pytest.raises(runner.TensorRTLLMRunnerError, match="unavailable"):
        runner._hash_regular_backend_file(  # noqa: SLF001
            tmp_path / "missing",
            logical_name="backend.py",
        )
    directory = tmp_path / "directory"
    directory.mkdir()
    with pytest.raises(runner.TensorRTLLMRunnerError, match="non-empty regular"):
        runner._hash_regular_backend_file(directory, logical_name="backend.py")  # noqa: SLF001
    candidate = tmp_path / "backend.py"
    candidate.write_text("code", encoding="utf-8")
    with pytest.raises(runner.TensorRTLLMRunnerError, match="inventory is invalid"):
        runner._hash_regular_backend_file(candidate, logical_name="../backend.py")  # noqa: SLF001


def test_backend_inventory_rejects_missing_unavailable_and_oversized_sets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner, "_require_backend_version", lambda: None)
    monkeypatch.setattr(
        runner.importlib.metadata,
        "distribution",
        lambda _name: (_ for _ in ()).throw(
            importlib.metadata.PackageNotFoundError("missing")
        ),
    )
    with pytest.raises(runner.TensorRTLLMRunnerError, match="not installed"):
        runner._observed_backend_build_sha256()  # noqa: SLF001

    monkeypatch.setattr(
        runner.importlib.metadata,
        "distribution",
        lambda _name: SimpleNamespace(files=None),
    )
    with pytest.raises(runner.TensorRTLLMRunnerError, match="inventory is unavailable"):
        runner._observed_backend_build_sha256()  # noqa: SLF001

    monkeypatch.setattr(
        runner.importlib.metadata,
        "distribution",
        lambda _name: SimpleNamespace(files=[]),
    )
    with pytest.raises(runner.TensorRTLLMRunnerError, match="inventory is incomplete"):
        runner._observed_backend_build_sha256()  # noqa: SLF001

    files = [*runner._CRITICAL_BACKEND_FILES]  # noqa: SLF001
    files.extend(f"tensorrt_llm/native_{index}.so" for index in range(257))
    monkeypatch.setattr(
        runner.importlib.metadata,
        "distribution",
        lambda _name: SimpleNamespace(files=files),
    )
    with pytest.raises(runner.TensorRTLLMRunnerError, match="exceeds the file bound"):
        runner._observed_backend_build_sha256()  # noqa: SLF001


def test_driver_and_cuda_runtime_version_failures_are_actionable(
    tmp_path: Path,
) -> None:
    with pytest.raises(runner.TensorRTLLMRunnerError, match="unavailable"):
        runner._read_driver_version(version_path=tmp_path / "missing")  # noqa: SLF001
    version = tmp_path / "version"
    version.write_text("not a driver version", encoding="ascii")
    with pytest.raises(runner.TensorRTLLMRunnerError, match="not canonical"):
        runner._read_driver_version(version_path=version)  # noqa: SLF001

    with pytest.raises(runner.TensorRTLLMRunnerError, match="cannot be observed"):
        runner._read_cuda_runtime_version(  # noqa: SLF001
            library_loader=lambda _name: (_ for _ in ()).throw(OSError("missing"))
        )


def test_cuda_device_observation_rejects_absent_and_invalid_devices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unavailable = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False, device_count=lambda: 0)
    )
    monkeypatch.setattr(runner.importlib, "import_module", lambda _name: unavailable)
    with pytest.raises(runner.TensorRTLLMRunnerError, match="device is unavailable"):
        runner._observe_cuda_device()  # noqa: SLF001

    invalid = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 1,
            current_device=lambda: -1,
            get_device_name=lambda _index: "GPU",
            get_device_capability=lambda _index: (9, 0),
        )
    )
    monkeypatch.setattr(runner.importlib, "import_module", lambda _name: invalid)
    with pytest.raises(runner.TensorRTLLMRunnerError, match="identity is invalid"):
        runner._observe_cuda_device()  # noqa: SLF001

    invalid.cuda.current_device = lambda: 0
    invalid.cuda.get_device_capability = lambda _index: (True, 0)
    with pytest.raises(runner.TensorRTLLMRunnerError, match="capability is invalid"):
        runner._observe_cuda_device()  # noqa: SLF001
