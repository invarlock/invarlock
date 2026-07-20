from __future__ import annotations

import hashlib
import io
import json
import os
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest
from invarlock_addins.tensorrt_llm import execution, inspection, runner, session


def test_runtime_resource_budgets_match_static_inspection_for_qwen3_contract() -> None:
    assert runner._MAX_CONFIG_BYTES == inspection._MAX_ENGINE_CONFIG_BYTES  # noqa: SLF001
    assert (  # noqa: SLF001
        runner._MAX_TOKENIZER_BYTES == inspection._MAX_TOKENIZER_CONTRACT_BYTES
    )
    assert runner._MAX_JSON_DEPTH == inspection._MAX_JSON_DEPTH  # noqa: SLF001
    assert runner._MAX_JSON_ITEMS == inspection._MAX_JSON_ITEMS  # noqa: SLF001
    assert runner._MAX_JSON_ITEMS >= 1_060_694  # noqa: SLF001
    assert runner._MAX_JSON_ITEMS <= 1_250_000  # noqa: SLF001


def test_strict_json_rejects_ambiguous_or_unbounded_payloads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(runner.TensorRTLLMRunnerError, match="duplicate object key"):
        runner._strict_json_object(b'{"value":1,"value":2}', label="request")  # noqa: SLF001

    with pytest.raises(runner.TensorRTLLMRunnerError, match="non-finite number"):
        runner._strict_json_object(b'{"value":NaN}', label="request")  # noqa: SLF001

    with pytest.raises(runner.TensorRTLLMRunnerError, match="not strict JSON"):
        runner._strict_json_object(b"\xff", label="request")  # noqa: SLF001

    with pytest.raises(runner.TensorRTLLMRunnerError, match="must be a JSON object"):
        runner._strict_json_object(b"[]", label="request")  # noqa: SLF001

    monkeypatch.setattr(runner, "_MAX_JSON_DEPTH", 0)
    with pytest.raises(runner.TensorRTLLMRunnerError, match="maximum nesting depth"):
        runner._validate_json_budget({"nested": {}})  # noqa: SLF001

    monkeypatch.setattr(runner, "_MAX_JSON_DEPTH", 64)
    monkeypatch.setattr(runner, "_MAX_JSON_ITEMS", 1)
    with pytest.raises(runner.TensorRTLLMRunnerError, match="maximum item count"):
        runner._validate_json_budget({"value": 1})  # noqa: SLF001

    monkeypatch.setattr(runner, "_MAX_JSON_ITEMS", 100)
    with pytest.raises(runner.TensorRTLLMRunnerError, match="not text"):
        runner._validate_json_budget({1: "value"})  # type: ignore[dict-item]  # noqa: SLF001

    assert runner._validate_json_budget([1]) == 2  # noqa: SLF001
    with pytest.raises(runner.TensorRTLLMRunnerError, match="non-finite number"):
        runner._validate_json_budget(float("nan"))  # noqa: SLF001


def test_runner_file_and_path_helpers_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(runner.TensorRTLLMRunnerError, match="exceeds the byte limit"):
        runner._read_bounded(  # noqa: SLF001
            stream=io.BytesIO(b"too-long"),
            limit=3,
            label="request",
        )

    missing = tmp_path / "missing.json"
    with pytest.raises(runner.TensorRTLLMRunnerError, match="is unavailable"):
        runner._read_regular_file(missing, 32, label="request")  # noqa: SLF001

    directory = tmp_path / "directory"
    directory.mkdir()
    with pytest.raises(runner.TensorRTLLMRunnerError, match="bounded regular file"):
        runner._read_regular_file(directory, 32, label="request")  # noqa: SLF001

    with pytest.raises(runner.TensorRTLLMRunnerError, match="path is invalid"):
        runner._canonical_path("", label="request")  # noqa: SLF001
    with pytest.raises(runner.TensorRTLLMRunnerError, match="canonical and absolute"):
        runner._canonical_path("relative.json", label="request")  # noqa: SLF001
    with pytest.raises(runner.TensorRTLLMRunnerError, match="path is unavailable"):
        runner._canonical_path(str(missing), label="request")  # noqa: SLF001

    target = tmp_path / "target.json"
    target.write_text("{}", encoding="utf-8")
    link = tmp_path / "link.json"
    link.symlink_to(target)
    with pytest.raises(
        runner.TensorRTLLMRunnerError, match="must not contain symlinks"
    ):
        runner._canonical_path(str(link), label="request")  # noqa: SLF001

    with pytest.raises(runner.TensorRTLLMRunnerError, match="supported bound"):
        runner._positive_integer(True, label="value", maximum=8)  # noqa: SLF001
    with pytest.raises(runner.TensorRTLLMRunnerError, match="supported bound"):
        runner._nonnegative_integer(-1, label="value", maximum=8)  # noqa: SLF001


@pytest.mark.skipif(
    os.name != "posix" or not hasattr(os, "O_NOFOLLOW"),
    reason="secure file pinning requires POSIX nofollow support",
)
def test_pinned_file_rejects_size_mode_and_digest_drift(tmp_path: Path) -> None:
    candidate = tmp_path / "runner"
    candidate.write_bytes(b"authenticated-runner")

    with pytest.raises(execution.TensorRTLLMExecutionError, match="size bound"):
        execution._PinnedFile.open(  # noqa: SLF001
            candidate,
            expected_sha256=None,
            require_executable=False,
            max_bytes=4,
        )

    with pytest.raises(execution.TensorRTLLMExecutionError, match="not executable"):
        execution._PinnedFile.open(  # noqa: SLF001
            candidate,
            expected_sha256=None,
            require_executable=True,
        )

    candidate.chmod(0o700)
    with pytest.raises(
        execution.TensorRTLLMExecutionError, match="digest does not match"
    ):
        execution._PinnedFile.open(  # noqa: SLF001
            candidate,
            expected_sha256=hashlib.sha256(b"different").hexdigest(),
            require_executable=True,
        )

    pinned = execution._PinnedFile.open(  # noqa: SLF001
        candidate,
        expected_sha256=hashlib.sha256(b"authenticated-runner").hexdigest(),
        require_executable=True,
    )
    pinned.close()
    pinned.close()
    with pytest.raises(
        execution.TensorRTLLMExecutionError, match="pinned file is closed"
    ):
        pinned.recheck()


def test_proc_fact_reader_rejects_missing_oversized_and_non_ascii_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(execution.TensorRTLLMExecutionError, match="is unavailable"):
        execution._read_bounded_proc_text(  # noqa: SLF001
            tmp_path / "missing",
            label="process fact",
        )

    oversized = tmp_path / "oversized"
    oversized.write_bytes(b"abc")
    monkeypatch.setattr(execution, "_MAX_PROC_FACT_BYTES", 2)
    with pytest.raises(execution.TensorRTLLMExecutionError, match="size limit"):
        execution._read_bounded_proc_text(oversized, label="process fact")  # noqa: SLF001

    non_ascii = tmp_path / "non-ascii"
    non_ascii.write_bytes(b"\xff")
    monkeypatch.setattr(execution, "_MAX_PROC_FACT_BYTES", 128)
    with pytest.raises(execution.TensorRTLLMExecutionError, match="canonical ASCII"):
        execution._read_bounded_proc_text(non_ascii, label="process fact")  # noqa: SLF001


def test_process_status_and_vendor_interpreter_contracts_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(execution.TensorRTLLMExecutionError, match="incomplete"):
        execution._parse_restricted_process_status("ignored line\n")  # noqa: SLF001

    duplicated = (
        "NoNewPrivs:\t1\n"
        "NoNewPrivs:\t1\n"
        "CapInh:\t0000000000000000\n"
        "CapPrm:\t0000000000000000\n"
        "CapEff:\t0000000000000000\n"
        "CapBnd:\t0000000000000000\n"
        "CapAmb:\t0000000000000000\n"
    )
    with pytest.raises(execution.TensorRTLLMExecutionError, match="not canonical"):
        execution._parse_restricted_process_status(duplicated)  # noqa: SLF001

    monkeypatch.setattr(execution, "_VENDOR_PYTHON", tmp_path / "missing-python")
    with pytest.raises(execution.TensorRTLLMExecutionError, match="cannot be resolved"):
        execution._resolve_vendor_python()  # noqa: SLF001


def test_run_directory_rejects_use_after_cleanup() -> None:
    run_directory = execution._RunDirectory.create()  # noqa: SLF001
    assert run_directory.environment()["HOME"] == str(run_directory.path)
    run_directory.close()
    run_directory.close()
    with pytest.raises(
        execution.TensorRTLLMExecutionError, match="directory is closed"
    ):
        run_directory.recheck()


def test_finite_float_parser_rejects_nonfinite_values() -> None:
    assert runner._finite_float("1.25") == 1.25  # noqa: SLF001
    with pytest.raises(runner.TensorRTLLMRunnerError, match="non-finite number"):
        runner._finite_float("nan")  # noqa: SLF001


def test_strict_json_accepts_canonical_finite_object() -> None:
    payload = json.dumps(
        {"nested": [1, 2.5]},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    assert runner._strict_json_object(payload, label="request") == {  # noqa: SLF001
        "nested": [1, 2.5]
    }


def test_execution_identity_helpers_cover_protected_and_rejected_entries() -> None:
    protected = SimpleNamespace(st_mode=stat.S_IFDIR | 0o755, st_uid=os.geteuid())
    entry = SimpleNamespace(st_uid=os.geteuid())
    assert execution._parent_entry_is_protected(protected, entry) is True  # noqa: SLF001

    writable = SimpleNamespace(st_mode=stat.S_IFDIR | 0o777, st_uid=99999)
    assert execution._parent_entry_is_protected(writable, entry) is False  # noqa: SLF001

    assert execution._parse_mount_id("mnt_id:\t42\n") == 42  # noqa: SLF001
    for payload in ("", "mnt_id: 0\n", "mnt_id: 1\nmnt_id: 2\n"):
        with pytest.raises(
            execution.TensorRTLLMExecutionError,
            match="mount identity is not canonical",
        ):
            execution._parse_mount_id(payload)  # noqa: SLF001


def test_readonly_and_process_status_checks_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        execution.os,
        "fstatvfs",
        lambda _descriptor: SimpleNamespace(f_flag=0),
    )
    with pytest.raises(execution.TensorRTLLMExecutionError, match="read-only"):
        execution._require_readonly_descriptor(1, label="runtime")  # noqa: SLF001

    restricted = (
        "NoNewPrivs:\t0\n"
        "CapInh:\t0000000000000000\n"
        "CapPrm:\t0000000000000000\n"
        "CapEff:\t0000000000000000\n"
        "CapBnd:\t0000000000000000\n"
        "CapAmb:\t0000000000000000\n"
    )
    with pytest.raises(
        execution.TensorRTLLMExecutionError,
        match="permits privilege acquisition",
    ):
        execution._parse_restricted_process_status(restricted)  # noqa: SLF001


def test_official_runner_and_trusted_executable_checks_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(execution.TensorRTLLMExecutionError, match="official installed"):
        execution._pin_official_runner(  # noqa: SLF001
            tmp_path / "other",
            expected_sha256=None,
        )

    executable = tmp_path / "runner"
    executable.write_bytes(b"runner")
    executable.chmod(0o755)
    monkeypatch.setattr(execution, "_REQUIRED_EXECUTABLE_OWNER", (99999, 99999))
    with pytest.raises(
        execution.TensorRTLLMExecutionError, match="ownership is invalid"
    ):
        execution._pin_trusted_executable(  # noqa: SLF001
            executable,
            expected_sha256=None,
            label="runner",
        )


def test_pinned_file_recheck_detects_unavailability_and_digest_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = tmp_path / "runner"
    candidate.write_bytes(b"authenticated-runner")
    pinned = execution._PinnedFile.open(  # noqa: SLF001
        candidate,
        expected_sha256=None,
        require_executable=False,
    )
    try:
        real_stat = execution.os.stat

        def unavailable(*args: object, **kwargs: object) -> os.stat_result:
            if kwargs.get("dir_fd") == pinned.parent_descriptor:
                raise FileNotFoundError("gone")
            return real_stat(*args, **kwargs)

        monkeypatch.setattr(execution.os, "stat", unavailable)
        with pytest.raises(execution.TensorRTLLMExecutionError, match="unavailable"):
            pinned.recheck()
    finally:
        pinned.close()


def test_immutable_boundary_and_run_directory_reject_changed_state(
    tmp_path: Path,
) -> None:
    boundary = execution._ImmutableExecutionBoundary(  # noqa: SLF001
        root_descriptor=-1,
        root_initial_stat=tmp_path.stat(),
        mount_id=1,
    )
    boundary._closed = True  # noqa: SLF001
    with pytest.raises(execution.TensorRTLLMExecutionError, match="boundary is closed"):
        boundary.recheck(SimpleNamespace(), SimpleNamespace())  # type: ignore[arg-type]

    run_directory = execution._RunDirectory.create()  # noqa: SLF001
    original = run_directory.path
    moved = original.with_name(original.name + "-moved")
    original.rename(moved)
    original.mkdir()
    try:
        with pytest.raises(
            execution.TensorRTLLMExecutionError,
            match="runtime directory changed",
        ):
            run_directory.recheck()
    finally:
        original.rmdir()
        moved.rename(original)
        run_directory.close()


def test_hash_descriptor_rejects_truncation_and_growth(tmp_path: Path) -> None:
    candidate = tmp_path / "payload"
    candidate.write_bytes(b"abc")
    descriptor = os.open(candidate, os.O_RDONLY)
    try:
        with pytest.raises(execution.TensorRTLLMExecutionError, match="changed"):
            execution._hash_descriptor(descriptor, 4)  # noqa: SLF001
        with pytest.raises(execution.TensorRTLLMExecutionError, match="grew"):
            execution._hash_descriptor(descriptor, 2)  # noqa: SLF001
    finally:
        os.close(descriptor)


def test_pinned_file_rejects_unsupported_or_invalid_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(execution.os, "name", "nt")
    with pytest.raises(execution.TensorRTLLMExecutionError, match="POSIX"):
        execution._PinnedFile.open(  # noqa: SLF001
            "candidate",
            expected_sha256=None,
            require_executable=False,
        )

    monkeypatch.setattr(execution.os, "name", "posix")

    class InvalidPath:
        def __fspath__(self) -> str:
            raise TypeError("invalid")

    with pytest.raises(execution.TensorRTLLMExecutionError, match="path is invalid"):
        execution._PinnedFile.open(  # type: ignore[arg-type]  # noqa: SLF001
            InvalidPath(),
            expected_sha256=None,
            require_executable=False,
        )


def test_pinned_file_recheck_detects_content_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate = tmp_path / "payload"
    candidate.write_bytes(b"one")
    pinned = execution._PinnedFile.open(  # noqa: SLF001
        candidate,
        expected_sha256=None,
        require_executable=False,
    )
    try:
        candidate.write_bytes(b"two")
        monkeypatch.setattr(
            execution,
            "_stat_identity",
            lambda value: (value.st_dev, value.st_ino, value.st_size),
        )
        with pytest.raises(execution.TensorRTLLMExecutionError, match="digest changed"):
            pinned.recheck()
    finally:
        pinned.close()


def test_proc_and_descriptor_fact_helpers_cover_success_and_io_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        execution,
        "_read_bounded_proc_text",
        lambda *_args, **_kwargs: "mnt_id:\t533\n",
    )
    assert execution._descriptor_mount_id(3) == 533  # noqa: SLF001

    monkeypatch.setattr(
        execution.os,
        "fstatvfs",
        lambda _descriptor: (_ for _ in ()).throw(OSError("unavailable")),
    )
    with pytest.raises(
        execution.TensorRTLLMExecutionError, match="facts are unavailable"
    ):
        execution._require_readonly_descriptor(3, label="runtime")  # noqa: SLF001


def test_restricted_process_status_reader_accepts_closed_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    restricted = (
        "NoNewPrivs:\t1\n"
        "CapInh:\t0000000000000000\n"
        "CapPrm:\t0000000000000000\n"
        "CapEff:\t0000000000000000\n"
        "CapBnd:\t0000000000000000\n"
        "CapAmb:\t0000000000000000\n"
    )
    monkeypatch.setattr(
        execution,
        "_read_bounded_proc_text",
        lambda *_args, **_kwargs: restricted,
    )
    execution._require_restricted_process_status()  # noqa: SLF001


def test_execution_boundary_create_and_recheck_report_filesystem_failures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    real_open = execution.os.open

    def denied(path: object, *args: object, **kwargs: object) -> int:
        if path == "/":
            raise PermissionError("denied")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(execution.os, "open", denied)
    with pytest.raises(execution.TensorRTLLMExecutionError, match="cannot be opened"):
        execution._ImmutableExecutionBoundary.create(  # noqa: SLF001
            SimpleNamespace(),  # type: ignore[arg-type]
            SimpleNamespace(),  # type: ignore[arg-type]
        )

    monkeypatch.setattr(execution.os, "open", real_open)
    descriptor = os.open(tmp_path, os.O_RDONLY)
    boundary = execution._ImmutableExecutionBoundary(  # noqa: SLF001
        root_descriptor=descriptor,
        root_initial_stat=os.fstat(descriptor),
        mount_id=1,
    )
    monkeypatch.setattr(
        execution.os,
        "fstat",
        lambda _descriptor: (_ for _ in ()).throw(OSError("gone")),
    )
    try:
        with pytest.raises(execution.TensorRTLLMExecutionError, match="unavailable"):
            boundary.recheck(  # type: ignore[arg-type]
                SimpleNamespace(),
                SimpleNamespace(),
            )
    finally:
        monkeypatch.undo()
        boundary.close()
        boundary.close()


def test_selector_stream_close_tolerates_unregistered_stream() -> None:
    stream = io.BytesIO()

    class Selector:
        def unregister(self, _stream: object) -> None:
            raise KeyError("not registered")

    execution._close_selector_stream(Selector(), stream)  # type: ignore[arg-type]  # noqa: SLF001
    assert stream.closed is True


def test_runner_info_probe_rejects_exit_and_stderr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resources = (SimpleNamespace(),) * 4
    monkeypatch.setattr(
        session,
        "_run_bounded_process",
        lambda **_kwargs: (7, b"{}", b""),
    )
    with pytest.raises(execution.TensorRTLLMExecutionError, match="status 7"):
        session._probe_runner_info_object(*resources)  # type: ignore[arg-type]  # noqa: SLF001

    monkeypatch.setattr(
        session,
        "_run_bounded_process",
        lambda **_kwargs: (0, b"{}", b"warning"),
    )
    with pytest.raises(execution.TensorRTLLMExecutionError, match="emitted stderr"):
        session._probe_runner_info_object(*resources)  # type: ignore[arg-type]  # noqa: SLF001


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (
            {"format_version": "wrong", "outputs": []},
            "format is unsupported",
        ),
        (
            {
                "format_version": "invarlock/tensorrt-llm-runner-batch-response-v1",
                "outputs": "wrong",
            },
            "outputs must be a list",
        ),
        (
            {
                "format_version": "invarlock/tensorrt-llm-runner-batch-response-v1",
                "outputs": [{"record_id": 1, "output_text": "answer"}],
            },
            "record_id must be text",
        ),
    ],
)
def test_batch_response_rejects_invalid_protocol_fields(
    payload: dict[str, object], message: str
) -> None:
    with pytest.raises(execution.TensorRTLLMExecutionError, match=message):
        session._validated_batch_response(  # noqa: SLF001
            json.dumps(payload).encode(),
            expected_record_ids=("record",) if payload.get("outputs") else (),
        )


def test_tensorrt_inspection_rejects_non_bindings() -> None:
    with pytest.raises(ValueError, match="native runtime bindings"):
        session.inspect_tensorrt_llm_inputs(object())  # type: ignore[arg-type]


def test_snapshot_copy_rejects_truncated_and_growing_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(session, "_fcntl", None)
    source = tmp_path / "source"
    source.write_bytes(b"abc")
    descriptor = os.open(source, os.O_RDONLY)
    try:
        with pytest.raises(execution.TensorRTLLMExecutionError, match="changed"):
            session._copy_from_descriptor(  # noqa: SLF001
                descriptor,
                tmp_path / "truncated",
                4,
            )
        with pytest.raises(execution.TensorRTLLMExecutionError, match="changed"):
            session._copy_from_descriptor(  # noqa: SLF001
                descriptor,
                tmp_path / "growing",
                2,
            )
    finally:
        os.close(descriptor)


@pytest.mark.parametrize("entries", [(), ("unexpected",)])
def test_snapshot_bundle_rejects_invalid_layouts(
    tmp_path: Path, entries: tuple[str, ...]
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    for name in entries:
        source.joinpath(name).write_bytes(b"value")
    with pytest.raises(execution.TensorRTLLMExecutionError, match="count|single-rank"):
        session._snapshot_bundle(source, tmp_path / "snapshot")  # noqa: SLF001


def test_tensorrt_network_namespace_requires_loopback_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def route_text(path: Path, **_kwargs: object) -> str:
        if path.name == "route":
            return "Iface Destination\nlo 00000000\n"
        return "prefix values lo\n"

    monkeypatch.setattr(Path, "read_text", route_text)
    session._require_isolated_network_namespace()  # noqa: SLF001

    def routable(path: Path, **_kwargs: object) -> str:
        if path.name == "route":
            return "Iface Destination\neth0 00000000\n"
        return ""

    monkeypatch.setattr(Path, "read_text", routable)
    with pytest.raises(execution.TensorRTLLMExecutionError, match="network-disabled"):
        session._require_isolated_network_namespace()  # noqa: SLF001
