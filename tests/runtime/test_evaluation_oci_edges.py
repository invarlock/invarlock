from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from invarlock import evaluation_oci as oci


def _error(message: str):
    return pytest.raises(oci.OciEvaluationError, match=message)


def test_execution_discriminator_and_worker_limit_edge_types(tmp_path: Path) -> None:
    request = tmp_path / "request.yaml"
    request.write_text(
        "format_version: invarlock/evaluation-request-v1\n"
        "comparison: {}\nexecution: []\noutput: {}\n",
        encoding="utf-8",
    )
    assert oci.evaluation_request_execution_mode(request) is None

    with _error("memory limit must be an integer"):
        oci._memory_limit_mib(True)
    with _error("memory limit must be an integer"):
        oci._memory_limit_mib(1.5)
    with _error("portable 32-bit"):
        oci._runtime_user("4294967295:1")


def test_execution_discriminator_rejects_non_mapping_execution(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    request = tmp_path / "request.yaml"
    request.write_text("placeholder", encoding="utf-8")
    monkeypatch.setattr(oci, "_load_yaml", lambda _payload: {"execution": []})
    monkeypatch.setattr(oci, "_validate_schema", lambda value: value)

    assert oci.evaluation_request_execution_mode(request) is None


def test_bounded_engine_command_streams_output_and_cleans_failed_destination(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    destination = tmp_path / "inspect.json"
    completed = oci._run_bounded_command(  # noqa: SLF001
        [sys.executable, "-c", "print('ok')"],
        timeout_seconds=5,
        stdout_limit=1024,
        stdout_path=destination,
    )
    assert completed.returncode == 0
    assert completed.stdout == b""
    assert destination.read_bytes() == b"ok\n"

    def fail_to_start(*_args: object, **_kwargs: object) -> subprocess.Popen[bytes]:
        raise OSError("engine unavailable")

    monkeypatch.setattr(oci.subprocess, "Popen", fail_to_start)
    with _error("could not complete"):
        oci._run_bounded_command(  # noqa: SLF001
            ["docker", "version"],
            timeout_seconds=5,
            stdout_limit=1024,
            stdout_path=tmp_path / "failed.json",
        )
    assert not (tmp_path / "failed.json").exists()


def test_bounded_engine_termination_escalates_after_stop_timeout() -> None:
    class HungProcess:
        def __init__(self) -> None:
            self.terminated = False
            self.killed = False
            self.waits = 0

        def poll(self) -> None:
            return None

        def terminate(self) -> None:
            self.terminated = True

        def kill(self) -> None:
            self.killed = True

        def wait(self, *, timeout: int) -> int:
            self.waits += 1
            if self.waits == 1:
                raise subprocess.TimeoutExpired("docker", timeout)
            return 0

    process = HungProcess()
    oci._terminate_bounded_process(process)  # type: ignore[arg-type]  # noqa: SLF001
    assert process.terminated
    assert process.killed
    assert process.waits == 2


def test_completed_engine_process_needs_no_termination() -> None:
    class CompletedProcess:
        def poll(self) -> int:
            return 0

        def terminate(self) -> None:
            pytest.fail("completed process must not be terminated")

    oci._terminate_bounded_process(CompletedProcess())  # type: ignore[arg-type]


def test_bounded_engine_command_can_return_captured_stdout() -> None:
    completed = oci._run_bounded_command(  # noqa: SLF001
        [sys.executable, "-c", "print('captured')"],
        timeout_seconds=5,
        stdout_limit=1024,
    )

    assert completed.returncode == 0
    assert completed.stdout == b"captured\n"


def test_provider_bindings_require_complete_optional_resources(
    tmp_path: Path,
) -> None:
    with _error("RESOURCE_ROOT"):
        oci._provider_bindings(  # noqa: SLF001
            {"INVARLOCK_HF_VISION_TEXT_CONTENT_STORE": str(tmp_path)}
        )
    with _error("CONTENT_STORE"):
        oci._provider_bindings(  # noqa: SLF001
            {"INVARLOCK_HF_VISION_TEXT_RESOURCE_ROOT": str(tmp_path)}
        )


def test_image_inspection_scalar_normalization_and_repository_validation() -> None:
    assert oci._inspect_output_bytes("text", label="stdout") == b"text"
    with _error("stdout is invalid"):
        oci._inspect_output_bytes(1, label="stdout")
    assert oci._normalized_config_id("a" * 64) == "sha256:" + "a" * 64
    with _error("config ID is invalid"):
        oci._normalized_config_id(1)
    with _error("config ID is invalid"):
        oci._normalized_config_id("bad")

    assert oci._repository_digest("registry.example/model@sha256:" + "a" * 64)[0] == (
        "registry.example/model"
    )
    for value in (1, " spaced @sha256:" + "a" * 64, "/root@sha256:" + "a" * 64):
        with _error("repository digest is invalid"):
            oci._repository_digest(value)
    with _error("repository digest is invalid"):
        oci._repository_digest("registry.example/model@sha256:bad")
    with _error("repository name"):
        oci._tag_repository(":latest")


def test_local_image_inspection_reports_process_and_payload_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        oci,
        "_run_bounded_command",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            oci.OciEvaluationError("timed out")
        ),
    )
    with _error("could not be inspected locally"):
        oci._inspect_local_image("docker", "image")

    def completed(stdout: object, stderr: object = b"", returncode: int = 0):
        monkeypatch.setattr(
            oci,
            "_run_bounded_command",
            lambda *_args, **_kwargs: subprocess.CompletedProcess(
                [], returncode, stdout, stderr
            ),
        )

    completed(b"", b"missing", 1)
    with _error("missing"):
        oci._inspect_local_image("docker", "image")
    completed(b"x" * (oci._MAX_IMAGE_INSPECT_BYTES + 1))
    with _error("bounded size"):
        oci._inspect_local_image("docker", "image")
    completed(b"{")
    with _error("valid JSON"):
        oci._inspect_local_image("docker", "image")
    completed(b"[]")
    with _error("exactly one image object"):
        oci._inspect_local_image("docker", "image")
    completed(b"1")
    with _error("exactly one image object"):
        oci._inspect_local_image("docker", "image")
    completed(json.dumps({"Id": "a" * 64, "RepoDigests": {}}).encode())
    with _error("RepoDigests are invalid"):
        oci._inspect_local_image("docker", "image")
    completed(json.dumps({"Id": "a" * 64}).encode())
    inspection = oci._inspect_local_image("docker", "image")
    assert inspection.repo_digests == ()


def test_inspected_image_resolution_rejects_unbound_local_id_and_tag() -> None:
    digest = "sha256:" + "a" * 64
    mismatched = oci._LocalImageInspection(  # noqa: SLF001
        config_id="sha256:" + "b" * 64,
        repo_digests=(),
    )

    with _error("config ID and supplied digest do not agree"):
        oci._resolve_inspected_image(digest, digest, mismatched, allow_tag=False)
    with _error("must be an immutable"):
        oci._resolve_inspected_image(
            "registry/model:latest", digest, mismatched, allow_tag=False
        )


def test_mount_sources_reject_missing_and_unrepresentable_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with _error("is unavailable"):
        oci._directory_mount_source(tmp_path / "missing", label="resources")
    with _error("is unavailable"):
        oci._artifact_mount_source(tmp_path / "missing", label="artifact")

    directory = tmp_path / "directory"
    directory.mkdir()
    file_path = tmp_path / "file"
    file_path.write_bytes(b"value")
    directory_link = tmp_path / "directory-link"
    directory_link.symlink_to(directory, target_is_directory=True)
    with _error("must be a directory"):
        oci._directory_mount_source(directory_link, label="resources")
    with _error("regular file or directory"):
        oci._artifact_mount_source(directory_link, label="artifact")
    monkeypatch.setattr(Path, "resolve", lambda self, **_kwargs: Path("/bad,path"))
    with _error("cannot be represented"):
        oci._directory_mount_source(directory, label="resources")
    with _error("cannot be represented"):
        oci._artifact_mount_source(file_path, label="artifact")


def test_worker_permission_resolution_covers_owner_group_and_other() -> None:
    owner = SimpleNamespace(st_uid=1, st_gid=2, st_mode=stat.S_IRUSR | stat.S_IXUSR)
    group = SimpleNamespace(st_uid=9, st_gid=2, st_mode=stat.S_IRGRP | stat.S_IXGRP)
    other = SimpleNamespace(st_uid=9, st_gid=8, st_mode=stat.S_IROTH | stat.S_IXOTH)
    unreadable = SimpleNamespace(st_uid=9, st_gid=8, st_mode=0)
    assert oci._worker_grants_read(owner, uid=1, gid=2, directory=True)
    assert oci._worker_grants_read(group, uid=1, gid=2, directory=True)
    assert oci._worker_grants_read(other, uid=1, gid=2, directory=True)
    assert not oci._worker_grants_read(unreadable, uid=1, gid=2, directory=False)
    no_search = SimpleNamespace(st_uid=1, st_gid=2, st_mode=stat.S_IRUSR)
    assert not oci._worker_grants_read(no_search, uid=1, gid=2, directory=True)


def test_worker_readability_lists_multiple_unreadable_entries(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir(mode=0o700)
    for index in range(4):
        (root / f"file-{index}").write_bytes(b"x")
        os.chmod(root / f"file-{index}", 0o600)
    with _error("and 2 more"):
        oci._assert_worker_readable(root, user="65532:65532", label="artifact")


def test_worker_readability_rejects_nested_symlinks(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir(mode=0o755)
    target = tmp_path / "target"
    target.write_bytes(b"secret")
    (root / "link").symlink_to(target)

    with _error("not readable"):
        oci._assert_worker_readable(root, user="65532:65532", label="artifact")


def test_worker_readability_checks_nested_directories(tmp_path: Path) -> None:
    root = tmp_path / "root"
    nested = root / "private"
    nested.mkdir(parents=True)
    root.chmod(0o755)
    nested.chmod(0o700)

    with _error("private"):
        oci._assert_worker_readable(root, user="65532:65532", label="artifact")


def test_tensorrt_scratch_rejects_special_bundle_entries(tmp_path: Path) -> None:
    engine = tmp_path / "engine"
    engine.mkdir()
    os.mkfifo(engine / "control.pipe")

    with _error("unsupported entry"):
        oci._worker_tmpfs_size_gib(
            provider_name="tensorrt_llm",
            artifact_source=engine,
        )


@pytest.mark.parametrize(
    "timeout", [True, 0, oci._MAX_OUTER_WORKER_TIMEOUT_SECONDS + 1]
)
def test_side_worker_rejects_invalid_outer_timeout_before_launch(
    timeout: object,
) -> None:
    with _error("outer timeout is invalid"):
        oci.run_side_worker([], timeout_seconds=timeout)  # type: ignore[arg-type]


def test_gpu_arguments_and_worker_stream_fail_closed_edges() -> None:
    assert oci._gpu_arguments("docker", "cpu") == []
    assert oci._gpu_arguments("docker", "cuda:2") == ["--gpus", "device=2"]
    assert oci._gpu_arguments("podman", "cuda") == [
        "--device",
        "nvidia.com/gpu=all",
    ]

    destination = bytearray()
    oci._read_bounded_stream(object(), destination, threading.Event())
    assert destination == b""

    with _error("at least one record"):
        oci._worker_outer_timeout_seconds({"timeout_seconds": 1}, record_count=0)
