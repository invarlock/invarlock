from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path

import pytest
from invarlock_addins.tensorrt_llm import execution, runner


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
