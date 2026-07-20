from __future__ import annotations

import hashlib
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from invarlock_addins.gguf import session

from invarlock.core.runtime_provider import (
    EvaluationBatch,
    EvaluationInputPart,
    EvaluationRecord,
    RuntimeExecutionSettings,
    evaluation_input_parts_sha256,
)


def test_hash_descriptor_rejects_truncation_and_growth(tmp_path: Path) -> None:
    candidate = tmp_path / "payload"
    candidate.write_bytes(b"abc")
    descriptor = os.open(candidate, os.O_RDONLY)
    try:
        with pytest.raises(session.LlamaCppExecutionError, match="changed"):
            session._hash_descriptor(descriptor, 4)  # noqa: SLF001
        with pytest.raises(session.LlamaCppExecutionError, match="grew"):
            session._hash_descriptor(descriptor, 2)  # noqa: SLF001
    finally:
        os.close(descriptor)


def test_pinned_file_rejects_unsupported_invalid_and_closed_use(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(session.os, "name", "nt")
    with pytest.raises(session.LlamaCppExecutionError, match="POSIX"):
        session._PinnedFile.open(  # noqa: SLF001
            "candidate",
            expected_sha256=None,
            require_executable=False,
        )
    monkeypatch.setattr(session.os, "name", "posix")

    class InvalidPath:
        def __fspath__(self) -> str:
            raise TypeError("invalid")

    with pytest.raises(session.LlamaCppExecutionError, match="path is invalid"):
        session._PinnedFile.open(  # type: ignore[arg-type]  # noqa: SLF001
            InvalidPath(),
            expected_sha256=None,
            require_executable=False,
        )

    candidate = tmp_path / "payload"
    candidate.write_bytes(b"value")
    pinned = session._PinnedFile.open(  # noqa: SLF001
        candidate,
        expected_sha256=hashlib.sha256(b"value").hexdigest(),
        require_executable=False,
    )
    pinned.close()
    pinned.close()
    with pytest.raises(session.LlamaCppExecutionError, match="pinned file is closed"):
        pinned.recheck()


def test_pinned_file_fd_path_and_digest_drift_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    candidate = tmp_path / "payload"
    candidate.write_bytes(b"one")
    pinned = session._PinnedFile.open(  # noqa: SLF001
        candidate,
        expected_sha256=None,
        require_executable=False,
    )
    try:
        monkeypatch.setattr(session.os.path, "isdir", lambda _path: False)
        with pytest.raises(session.LlamaCppExecutionError, match="path is unavailable"):
            _ = pinned.fd_path

        candidate.write_bytes(b"two")
        monkeypatch.setattr(
            session,
            "_stat_identity",
            lambda value: (value.st_dev, value.st_ino, value.st_size),
        )
        with pytest.raises(session.LlamaCppExecutionError, match="digest changed"):
            pinned.recheck()
    finally:
        pinned.close()


def test_run_directory_rejects_closed_and_changed_state() -> None:
    run_directory = session._RunDirectory.create()  # noqa: SLF001
    original = run_directory.path
    moved = original.with_name(original.name + "-moved")
    original.rename(moved)
    original.mkdir()
    try:
        with pytest.raises(session.LlamaCppExecutionError, match="directory changed"):
            run_directory.recheck()
    finally:
        original.rmdir()
        moved.rename(original)
        run_directory.close()
        run_directory.close()
    with pytest.raises(session.LlamaCppExecutionError, match="directory is closed"):
        run_directory.recheck()


def test_process_cleanup_still_terminates_descendants_after_leader_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    signals: list[tuple[int, int]] = []
    monkeypatch.setattr(
        session.os,
        "killpg",
        lambda process_group, sent: signals.append((process_group, sent)),
    )
    process = SimpleNamespace(pid=123, wait=lambda timeout: 0)

    session._kill_process_group(process)  # type: ignore[arg-type]  # noqa: SLF001

    assert signals == [(123, signal.SIGKILL)]


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX process groups")
def test_bounded_cleanup_kills_pipe_holding_descendant(tmp_path: Path) -> None:
    marker = tmp_path / "descendant-survived"
    code = (
        "import os,time,pathlib; child=os.fork(); "
        f"(time.sleep(1.5), pathlib.Path({str(marker)!r}).write_text('alive'), "
        "time.sleep(5)) if child == 0 else os._exit(0)"
    )
    process = subprocess.Popen(
        [sys.executable, "-c", code],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )

    with pytest.raises(session.LlamaCppExecutionError, match="timed out"):
        session.communicate_bounded(
            process,
            input_bytes=b"",
            timeout_seconds=1,
            stdout_limit=1024,
            stderr_limit=1024,
            error_type=session.LlamaCppExecutionError,
            timeout_label="llama.cpp record",
            output_label="llama.cpp",
            pipes_message="llama.cpp pipes could not be established",
            terminate=session._kill_process_group,  # noqa: SLF001
        )

    time.sleep(0.75)
    assert not marker.exists()


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (None, "non-empty text"),
        ("version: 1 built with Café for Linux", "canonical ASCII"),
        ("not a version", "closed grammar"),
        (
            "version: 4242 (secret) built with GCC for Linux",
            "sensitive-looking",
        ),
    ],
)
def test_backend_version_rejects_noncanonical_values(
    value: object, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        session.validate_llama_cpp_backend_version(value)


@pytest.mark.parametrize(
    ("stdout", "stderr", "message"),
    [
        (b"\xff", b"", "canonical ASCII"),
        (b"one line", b"", "exact version/build lines"),
        (b"version: wrong\nbuilt with wrong", b"", "closed grammar"),
    ],
)
def test_version_output_rejects_invalid_streams(
    stdout: bytes, stderr: bytes, message: str
) -> None:
    with pytest.raises(session.LlamaCppExecutionError, match=message):
        session._normalize_version_output(stdout, stderr)  # noqa: SLF001


def test_version_probe_rejects_failed_process(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        session,
        "_run_bounded_process",
        lambda **_kwargs: (9, b"", b"error"),
    )
    with pytest.raises(session.LlamaCppExecutionError, match="status 9"):
        session.probe_llama_cpp_version(
            SimpleNamespace(descriptor=3),  # type: ignore[arg-type]
            SimpleNamespace(),  # type: ignore[arg-type]
        )


def test_version_probe_accepts_a_successful_canonical_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        session,
        "_run_bounded_process",
        lambda **_kwargs: (
            0,
            b"version: 4242 (test)\nbuilt with GCC for Linux\n",
            b"",
        ),
    )
    assert (
        session.probe_llama_cpp_version(  # type: ignore[arg-type]
            SimpleNamespace(descriptor=3),
            SimpleNamespace(),
        )
        == "version: 4242 (test) built with GCC for Linux"
    )


def test_backend_resource_cleanup_continues_and_reports_failure() -> None:
    calls: list[str] = []

    class Resource:
        def __init__(self, name: str, *, fail: bool) -> None:
            self.name = name
            self.fail = fail

        def close(self) -> None:
            calls.append(self.name)
            if self.fail:
                raise OSError("cleanup")

    resources = [Resource("first", fail=False), Resource("second", fail=True)]
    with pytest.raises(
        session.LlamaCppExecutionError, match="cleanup did not complete"
    ):
        session._close_backend_resources(  # type: ignore[arg-type]  # noqa: SLF001
            resources,
            operation="test",
        )
    assert calls == ["second", "first"]


def test_backend_operations_require_native_bindings() -> None:
    with pytest.raises(ValueError, match="native runtime bindings"):
        session.authenticate_llama_cpp_backend_files(  # type: ignore[arg-type]
            object(),
            expected_binary_sha256="a" * 64,
            expected_source_sha256="b" * 64,
        )
    with pytest.raises(ValueError, match="native runtime bindings"):
        session.inspect_llama_cpp_backend(object())  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (b"answer", "lacks the pinned final framing"),
        (b"\xff\n\n", "not UTF-8"),
        (b"answer [end of text]\n\n\n", "ambiguous backend EOG marker"),
    ],
)
def test_generated_output_rejects_unframed_or_ambiguous_bytes(
    payload: bytes, message: str
) -> None:
    with pytest.raises(session.LlamaCppExecutionError, match=message):
        session._extract_generated_output(payload)  # noqa: SLF001


def _record(*, role: str = "prompt", digest: str | None = None) -> EvaluationRecord:
    text = "prompt"
    part = EvaluationInputPart(
        kind="text",
        role=role,
        text=text,
        sha256=hashlib.sha256(text.encode()).hexdigest(),
    )
    return EvaluationRecord(
        record_id="record",
        input_text=text,
        input_sha256=digest or evaluation_input_parts_sha256((part,)),
        input_parts=(part,),
    )


def _bare_session() -> session.LlamaCppSession:
    candidate = object.__new__(session.LlamaCppSession)
    candidate._closed = False  # noqa: SLF001
    candidate._executable = SimpleNamespace(  # noqa: SLF001
        descriptor=10,
        fd_path="/dev/fd/10",
    )
    candidate._model = SimpleNamespace(descriptor=11, fd_path="/dev/fd/11")  # noqa: SLF001
    candidate._source_archive = SimpleNamespace()  # noqa: SLF001
    candidate._run_directory = SimpleNamespace()  # noqa: SLF001
    candidate._score_lock = threading.Lock()  # noqa: SLF001
    candidate._config = SimpleNamespace(  # noqa: SLF001
        execution_settings=RuntimeExecutionSettings(
            seed=1,
            context_length=32,
            batch_size=1,
            max_output_tokens=8,
            timeout_seconds=2,
        )
    )
    return candidate


def test_record_execution_reports_input_process_and_stderr_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = _bare_session()
    record = _record()

    monkeypatch.setattr(session, "_MAX_INPUT_BYTES", 0)
    with pytest.raises(ValueError, match="input exceeds"):
        candidate._execute_record(record)  # noqa: SLF001

    monkeypatch.setattr(session, "_MAX_INPUT_BYTES", 1024)
    monkeypatch.setattr(
        session,
        "_run_bounded_process",
        lambda **_kwargs: (7, b"", b"private detail"),
    )
    with pytest.raises(session.LlamaCppExecutionError, match="status 7"):
        candidate._execute_record(record)  # noqa: SLF001

    monkeypatch.setattr(
        session,
        "_run_bounded_process",
        lambda **_kwargs: (0, b"answer\n\n", b"private detail"),
    )
    with pytest.raises(session.LlamaCppExecutionError, match="unexpected stderr"):
        candidate._execute_record(record)  # noqa: SLF001


def test_session_score_rejects_wrong_batch_shape_and_pairing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = _bare_session()
    candidate._latest_observation_sha256 = None  # noqa: SLF001

    with pytest.raises(TypeError, match="EvaluationBatch"):
        candidate.score(object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="text_causal"):
        candidate.score(
            EvaluationBatch(
                schedule_sha256="a" * 64,
                records=(_record(),),
                task="vision_text_generation",
            )
        )

    monkeypatch.setattr(session, "_MAX_BATCH_RECORDS", 0)
    with pytest.raises(ValueError, match="record limit"):
        candidate.score(EvaluationBatch("a" * 64, (_record(),)))
    monkeypatch.setattr(session, "_MAX_BATCH_RECORDS", 1024)

    with pytest.raises(ValueError, match="one prompt text input part"):
        candidate.score(EvaluationBatch("a" * 64, (_record(role="context"),)))
    with pytest.raises(ValueError, match="input_sha256 does not match"):
        candidate.score(EvaluationBatch("a" * 64, (_record(digest="0" * 64),)))


def test_session_open_state_receipt_and_optional_cleanup_contracts() -> None:
    candidate = _bare_session()
    candidate._latest_observation_sha256 = None  # noqa: SLF001
    with pytest.raises(RuntimeError, match="unavailable before scoring"):
        candidate.runtime_receipt()

    for missing in ("_executable", "_model", "_source_archive"):
        probe = _bare_session()
        setattr(probe, missing, None)
        with pytest.raises(RuntimeError, match="session is closed"):
            probe._require_open()  # noqa: SLF001

    calls: list[str] = []
    candidate._model = SimpleNamespace(close=lambda: calls.append("model"))  # noqa: SLF001
    candidate._source_archive = None  # noqa: SLF001
    candidate._executable = SimpleNamespace(  # noqa: SLF001
        close=lambda: calls.append("executable")
    )
    candidate._run_directory = SimpleNamespace(  # noqa: SLF001
        close=lambda: calls.append("directory")
    )
    candidate.close()
    candidate.close()
    assert calls == ["model", "executable", "directory"]


def test_process_group_cleanup_handles_disappeared_and_stubborn_children(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    class Process:
        pid = 123

        def poll(self) -> None:
            return None

        def wait(self, timeout: int) -> None:
            calls.append(f"wait:{timeout}")
            if len(calls) == 2:
                raise subprocess.TimeoutExpired("runner", timeout)

        def kill(self) -> None:
            calls.append("kill")

    monkeypatch.setattr(
        session.os,
        "killpg",
        lambda *_args: (_ for _ in ()).throw(ProcessLookupError()),
    )
    process = Process()
    session._kill_process_group(process)  # type: ignore[arg-type]  # noqa: SLF001
    session._kill_process_group(process)  # type: ignore[arg-type]  # noqa: SLF001
    assert calls == ["wait:2", "wait:2", "kill", "wait:2"]


@pytest.mark.parametrize("observed_version", ["expected version", "different version"])
def test_session_construction_authenticates_the_observed_backend_version(
    monkeypatch: pytest.MonkeyPatch,
    observed_version: str,
) -> None:
    closed: list[str] = []

    class Resource:
        def __init__(self, name: str) -> None:
            self.name = name

        def close(self) -> None:
            closed.append(self.name)

    resources = [Resource("executable"), Resource("model"), Resource("source")]
    run_directory = Resource("directory")
    monkeypatch.setattr(session, "artifact_identity_sha256", lambda _identity: "a" * 64)
    monkeypatch.setattr(
        session._RunDirectory,  # noqa: SLF001
        "create",
        classmethod(lambda _cls: run_directory),
    )
    monkeypatch.setattr(
        session._PinnedFile,  # noqa: SLF001
        "open",
        classmethod(lambda _cls, *_args, **_kwargs: resources.pop(0)),
    )
    monkeypatch.setattr(
        session,
        "probe_llama_cpp_version",
        lambda *_args: observed_version,
    )
    monkeypatch.setattr(session.LlamaCppSession, "_recheck_runtime", lambda _self: None)
    config = SimpleNamespace(
        artifact_identity=SimpleNamespace(sha256="d" * 64),
        backend_version="expected version",
        backend_binary_sha256="b" * 64,
        backend_source_sha256="c" * 64,
        bindings=SimpleNamespace(
            executable_path=Path("/runtime/llama"),
            gguf_path=Path("/runtime/model.gguf"),
            source_archive_path=Path("/runtime/source.tar"),
        ),
    )

    if observed_version != config.backend_version:
        with pytest.raises(session.LlamaCppExecutionError, match="observed version"):
            session.LlamaCppSession(config)  # type: ignore[arg-type]
        assert closed == ["model", "source", "executable", "directory"]
    else:
        candidate = session.LlamaCppSession(config)  # type: ignore[arg-type]
        candidate.close()
        assert closed == ["model", "source", "executable", "directory"]


def test_session_runtime_recheck_authenticates_identity_and_all_pins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = _bare_session()
    authenticated_identity = object()
    candidate._config.artifact_identity = authenticated_identity  # noqa: SLF001
    candidate._config.bindings = SimpleNamespace(gguf_path=Path("/model.gguf"))  # noqa: SLF001
    calls: list[str] = []
    candidate._model = SimpleNamespace(recheck=lambda: calls.append("model"))  # noqa: SLF001
    candidate._executable = SimpleNamespace(  # noqa: SLF001
        recheck=lambda: calls.append("executable")
    )
    candidate._source_archive = SimpleNamespace(  # noqa: SLF001
        recheck=lambda: calls.append("source")
    )
    candidate._run_directory = SimpleNamespace(  # noqa: SLF001
        recheck=lambda: calls.append("directory")
    )

    monkeypatch.setattr(session, "read_gguf_artifact_identity", lambda _path: object())
    with pytest.raises(session.LlamaCppExecutionError, match="authenticated identity"):
        candidate._recheck_runtime()  # noqa: SLF001
    assert calls == []

    monkeypatch.setattr(
        session,
        "read_gguf_artifact_identity",
        lambda _path: authenticated_identity,
    )
    candidate._recheck_runtime()  # noqa: SLF001
    assert calls == ["model", "executable", "source", "directory"]


def test_session_score_supports_structured_and_legacy_authenticated_inputs() -> None:
    candidate = _bare_session()
    candidate._latest_observation_sha256 = None  # noqa: SLF001
    candidate._artifact_identity_sha256 = "d" * 64  # noqa: SLF001
    candidate._config.capabilities = SimpleNamespace(provider_name="llama_cpp")  # noqa: SLF001
    rechecks: list[bool] = []
    candidate._recheck_runtime = lambda: rechecks.append(True)  # type: ignore[method-assign]  # noqa: SLF001
    candidate._execute_record = (  # type: ignore[method-assign]  # noqa: SLF001
        lambda record: f"answer:{record.record_id}"
    )
    legacy_text = "legacy prompt"
    legacy = EvaluationRecord(
        record_id="legacy",
        input_text=legacy_text,
        input_sha256=hashlib.sha256(legacy_text.encode()).hexdigest(),
    )

    observation = candidate.score(
        EvaluationBatch("e" * 64, (_record(), legacy), task="text_causal")
    )

    assert [record.record_id for record in observation.records] == ["record", "legacy"]
    assert [record.output_text for record in observation.records] == [
        "answer:record",
        "answer:legacy",
    ]
    assert len(rechecks) == 2
    assert candidate._latest_observation_sha256 is not None  # noqa: SLF001


def test_session_cleanup_handles_a_failure_before_any_pin_is_opened() -> None:
    candidate = _bare_session()
    candidate._model = None  # noqa: SLF001
    candidate._source_archive = None  # noqa: SLF001
    candidate._executable = None  # noqa: SLF001
    calls: list[str] = []
    candidate._run_directory = SimpleNamespace(close=lambda: calls.append("directory"))  # noqa: SLF001

    candidate.close()

    assert calls == ["directory"]
