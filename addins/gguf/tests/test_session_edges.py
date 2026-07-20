from __future__ import annotations

import hashlib
import io
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from invarlock_addins.gguf import session


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
    with pytest.raises(session.LlamaCppExecutionError, match="directory is closed"):
        run_directory.recheck()


def test_process_and_selector_cleanup_tolerate_absent_resources() -> None:
    process = SimpleNamespace(poll=lambda: 0)
    session._kill_process_group(process)  # type: ignore[arg-type]  # noqa: SLF001

    stream = io.BytesIO()

    class Selector:
        def unregister(self, _stream: object) -> None:
            raise KeyError("absent")

    session._close_selector_stream(Selector(), stream)  # type: ignore[arg-type]  # noqa: SLF001
    assert stream.closed is True


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
