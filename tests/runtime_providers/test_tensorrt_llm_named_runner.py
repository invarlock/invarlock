from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from invarlock.runtime_providers import (
    _tensorrt_llm_execution as tensorrt_llm_execution,
)
from invarlock.runtime_providers import tensorrt_llm as tensorrt_llm_provider
from invarlock.runtime_providers import tensorrt_llm_session
from invarlock.runtime_providers.tensorrt_llm import TensorRTLLMProvider
from invarlock.runtime_providers.tensorrt_llm_session import (
    TensorRTLLMExecutionError,
)
from tests.runtime_providers._tensorrt_llm_support import (
    _IMAGE_DIGEST,
    _batch,
    _record,
    _runtime_inputs,
    _write_fake_vendor_python,
)

_REQUIRE_READONLY_DESCRIPTOR = tensorrt_llm_execution._require_readonly_descriptor  # noqa: SLF001

pytestmark = pytest.mark.skipif(
    os.name != "posix",
    reason="the pinned TensorRT-LLM runtime requires POSIX nofollow support",
)


@pytest.fixture(autouse=True)
def _closed_runtime_boundary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INVARLOCK_RUNTIME_IMAGE_DIGEST", _IMAGE_DIGEST)
    monkeypatch.setenv("INVARLOCK_RUNTIME_IMAGE", f"invarlock-runtime@{_IMAGE_DIGEST}")
    monkeypatch.setattr(
        tensorrt_llm_provider,
        "strict_container_boundary_present",
        lambda: True,
    )
    monkeypatch.setattr(
        tensorrt_llm_session,
        "_require_isolated_network_namespace",
        lambda: None,
    )
    monkeypatch.setattr(
        tensorrt_llm_execution,
        "_REQUIRED_EXECUTABLE_OWNER",
        (os.getuid(), os.getgid()),
    )
    monkeypatch.setattr(
        tensorrt_llm_execution,
        "_OFFICIAL_RUNNER_PATH",
        tmp_path / "official-tensorrt-llm-runner",
        raising=False,
    )
    vendor_python = tmp_path / "private-vendor-python"
    _write_fake_vendor_python(vendor_python)
    monkeypatch.setattr(tensorrt_llm_execution, "_VENDOR_PYTHON", vendor_python)
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


def test_runner_binding_rejects_nonofficial_and_symlink_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec, bindings, context = _runtime_inputs(tmp_path)
    nonofficial = tmp_path / "other-runner"
    nonofficial.write_bytes(bindings.runner_executable_path.read_bytes())
    nonofficial.chmod(0o700)
    nonofficial_bindings = replace(bindings, runner_executable_path=nonofficial)
    with pytest.raises(TensorRTLLMExecutionError, match="official installed path"):
        TensorRTLLMProvider().open(
            spec,
            replace(context, native_model=nonofficial_bindings),
        )

    official = bindings.runner_executable_path
    target = tmp_path / "runner-target"
    official.replace(target)
    official.symlink_to(target)
    with pytest.raises(TensorRTLLMExecutionError, match="without following symlinks"):
        TensorRTLLMProvider().open(spec, context)


def test_runner_and_interpreter_reject_untrusted_ownership_or_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec, bindings, context = _runtime_inputs(tmp_path)
    bindings.runner_executable_path.chmod(0o722)
    with pytest.raises(TensorRTLLMExecutionError, match="group- or other-writable"):
        TensorRTLLMProvider().open(spec, context)

    bindings.runner_executable_path.chmod(0o700)
    monkeypatch.setattr(
        tensorrt_llm_execution,
        "_REQUIRED_EXECUTABLE_OWNER",
        (os.getuid() + 1, os.getgid()),
    )
    with pytest.raises(TensorRTLLMExecutionError, match="ownership"):
        TensorRTLLMProvider().open(spec, context)

    vendor_python = tmp_path / "vendor-python"
    _write_fake_vendor_python(vendor_python)
    vendor_python.chmod(0o722)
    monkeypatch.setattr(
        tensorrt_llm_execution,
        "_REQUIRED_EXECUTABLE_OWNER",
        (os.getuid(), os.getgid()),
    )
    monkeypatch.setattr(tensorrt_llm_execution, "_VENDOR_PYTHON", vendor_python)
    with pytest.raises(TensorRTLLMExecutionError, match="group- or other-writable"):
        tensorrt_llm_session._resolve_vendor_python()  # noqa: SLF001


def test_pinned_file_allows_only_owned_entries_under_trusted_sticky_parent(
    tmp_path: Path,
) -> None:
    sticky_parent = tmp_path / "sticky"
    sticky_parent.mkdir()
    sticky_parent.chmod(0o1777)
    executable = sticky_parent / "runner"
    executable.write_text("runner", encoding="utf-8")
    executable.chmod(0o700)

    pinned = tensorrt_llm_execution._PinnedFile.open(  # noqa: SLF001
        executable,
        expected_sha256=None,
        require_executable=True,
        require_secure_parents=True,
    )
    pinned.close()

    sticky_parent.chmod(0o0777)
    with pytest.raises(
        TensorRTLLMExecutionError, match="group- or other-writable parent"
    ):
        tensorrt_llm_execution._PinnedFile.open(  # noqa: SLF001
            executable,
            expected_sha256=None,
            require_executable=True,
            require_secure_parents=True,
        )


def test_sticky_parent_requires_trusted_directory_and_entry_owners() -> None:
    current_uid = os.geteuid()
    untrusted_uid = max(current_uid, 0) + 1000

    def facts(*, mode: int, uid: int) -> os.stat_result:
        return os.stat_result((mode, 1, 1, 1, uid, 1, 0, 0, 0, 0))

    trusted_parent = facts(mode=0o41777, uid=0)
    trusted_entry = facts(mode=0o40700, uid=current_uid)
    untrusted_parent = facts(mode=0o41777, uid=untrusted_uid)
    untrusted_entry = facts(mode=0o40700, uid=untrusted_uid)

    assert (
        tensorrt_llm_execution._parent_entry_is_protected(  # noqa: SLF001
            trusted_parent, trusted_entry
        )
        is True
    )
    assert (
        tensorrt_llm_execution._parent_entry_is_protected(  # noqa: SLF001
            untrusted_parent, trusted_entry
        )
        is False
    )
    assert (
        tensorrt_llm_execution._parent_entry_is_protected(  # noqa: SLF001
            trusted_parent, untrusted_entry
        )
        is False
    )


def test_provider_rejects_writable_root_mount(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec, _bindings, context = _runtime_inputs(tmp_path)

    def require_readonly(_descriptor: int, *, label: str) -> None:
        if label == "root":
            raise TensorRTLLMExecutionError("root requires a read-only filesystem")

    monkeypatch.setattr(
        tensorrt_llm_execution,
        "_require_readonly_descriptor",
        require_readonly,
    )
    with pytest.raises(TensorRTLLMExecutionError, match="root requires a read-only"):
        TensorRTLLMProvider().open(spec, context)


@pytest.mark.parametrize(
    "payload",
    [
        "NoNewPrivs:\t0\n"
        "CapInh:\t0000000000000000\n"
        "CapPrm:\t0000000000000000\n"
        "CapEff:\t0000000000000000\n"
        "CapBnd:\t0000000000000000\n"
        "CapAmb:\t0000000000000000\n",
        "NoNewPrivs:\t1\n"
        "CapInh:\t0000000000000001\n"
        "CapPrm:\t0000000000000000\n"
        "CapEff:\t0000000000000000\n"
        "CapBnd:\t0000000000000000\n"
        "CapAmb:\t0000000000000000\n",
    ],
)
def test_process_status_parser_rejects_privilege_drift(payload: str) -> None:
    with pytest.raises(TensorRTLLMExecutionError, match="process security status"):
        tensorrt_llm_execution._parse_restricted_process_status(payload)  # noqa: SLF001


def test_process_status_parser_accepts_restricted_thread() -> None:
    assert (
        tensorrt_llm_execution._parse_restricted_process_status(  # noqa: SLF001
            "NoNewPrivs:\t1\n"
            "CapInh:\t0000000000000000\n"
            "CapPrm:\t0000000000000000\n"
            "CapEff:\t0000000000000000\n"
            "CapBnd:\t0000000000000000\n"
            "CapAmb:\t0000000000000000\n"
        )
        is None
    )


@pytest.mark.parametrize(
    "payload",
    [
        "pos:\t0\nflags:\t02100000\n",
        "mnt_id:\t533\nmnt_id:\t534\n",
        "mnt_id:\t0\n",
        "mnt_id:\tnot-a-number\n",
    ],
)
def test_mount_id_parser_rejects_noncanonical_values(payload: str) -> None:
    with pytest.raises(TensorRTLLMExecutionError, match="mount identity"):
        tensorrt_llm_execution._parse_mount_id(payload)  # noqa: SLF001


def test_mount_fact_parsers_reject_writable_and_mismatched_mounts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert (
        tensorrt_llm_execution._parse_mount_id(  # noqa: SLF001
            "pos:\t0\nflags:\t02100000\nmnt_id:\t533\n"
        )
        == 533
    )
    monkeypatch.setattr(
        tensorrt_llm_execution.os,
        "fstatvfs",
        lambda _descriptor: SimpleNamespace(f_flag=0),
    )
    with pytest.raises(TensorRTLLMExecutionError, match="read-only filesystem"):
        _REQUIRE_READONLY_DESCRIPTOR(1, label="runner")
    monkeypatch.setattr(
        tensorrt_llm_execution.os,
        "fstatvfs",
        lambda _descriptor: SimpleNamespace(f_flag=os.ST_RDONLY),
    )
    _REQUIRE_READONLY_DESCRIPTOR(1, label="runner")

    runner_path = tmp_path / "runner"
    vendor_path = tmp_path / "vendor-python"
    runner_path.write_text("runner", encoding="utf-8")
    runner_path.chmod(0o700)
    _write_fake_vendor_python(vendor_path)
    runner = tensorrt_llm_execution._PinnedFile.open(  # noqa: SLF001
        runner_path,
        expected_sha256=None,
        require_executable=True,
    )
    vendor = tensorrt_llm_execution._PinnedFile.open(  # noqa: SLF001
        vendor_path,
        expected_sha256=None,
        require_executable=True,
    )
    monkeypatch.setattr(
        tensorrt_llm_execution,
        "_require_readonly_descriptor",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        tensorrt_llm_execution,
        "_descriptor_mount_id",
        lambda descriptor: 534 if descriptor == runner.descriptor else 533,
    )
    try:
        with pytest.raises(TensorRTLLMExecutionError, match="same root mount"):
            tensorrt_llm_execution._ImmutableExecutionBoundary.create(  # noqa: SLF001
                runner, vendor
            )
    finally:
        runner.close()
        vendor.close()


def test_launch_precheck_blocks_new_writable_boundary_before_spawn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec, _bindings, context = _runtime_inputs(tmp_path)
    session = TensorRTLLMProvider().open(spec, context)
    boundary = session._execution_boundary  # noqa: SLF001
    real_recheck = boundary.recheck
    recheck_calls = 0
    spawn_calls = 0

    def recheck(runner, vendor_python):  # noqa: ANN001, ANN202
        nonlocal recheck_calls
        recheck_calls += 1
        if recheck_calls == 2:
            raise TensorRTLLMExecutionError("runner requires a read-only filesystem")
        real_recheck(runner, vendor_python)

    def unexpected_spawn(*_args, **_kwargs):  # noqa: ANN002, ANN003, ANN202
        nonlocal spawn_calls
        spawn_calls += 1
        raise AssertionError("writable execution boundary reached process spawn")

    monkeypatch.setattr(boundary, "recheck", recheck)
    monkeypatch.setattr(tensorrt_llm_execution.subprocess, "Popen", unexpected_spawn)
    with pytest.raises(TensorRTLLMExecutionError, match="read-only filesystem"):
        session.score(_batch(_record("a", "alpha")))
    assert recheck_calls == 2
    assert spawn_calls == 0
    session.close()


def test_launch_precheck_blocks_privilege_drift_before_spawn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec, _bindings, context = _runtime_inputs(tmp_path)
    session = TensorRTLLMProvider().open(spec, context)
    spawn_calls = 0

    def reject_privileges() -> None:
        raise TensorRTLLMExecutionError(
            "process security status permits privilege acquisition"
        )

    def unexpected_spawn(*_args, **_kwargs):  # noqa: ANN002, ANN003, ANN202
        nonlocal spawn_calls
        spawn_calls += 1
        raise AssertionError("privilege drift reached process spawn")

    monkeypatch.setattr(
        tensorrt_llm_execution,
        "_require_restricted_process_status",
        reject_privileges,
    )
    monkeypatch.setattr(tensorrt_llm_execution.subprocess, "Popen", unexpected_spawn)
    with pytest.raises(TensorRTLLMExecutionError, match="privilege acquisition"):
        session.score(_batch(_record("a", "alpha")))
    assert spawn_calls == 0
    session.close()


def test_post_run_mount_drift_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec, _bindings, context = _runtime_inputs(tmp_path)
    session = TensorRTLLMProvider().open(spec, context)
    real_popen = tensorrt_llm_execution.subprocess.Popen
    drifted = False

    def mount_id(descriptor: int) -> int:
        if drifted and descriptor == session._runner.descriptor:  # noqa: SLF001
            return 534
        return 533

    def drift_after_spawn(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        nonlocal drifted
        process = real_popen(*args, **kwargs)
        drifted = True
        return process

    monkeypatch.setattr(tensorrt_llm_execution, "_descriptor_mount_id", mount_id)
    monkeypatch.setattr(tensorrt_llm_execution.subprocess, "Popen", drift_after_spawn)
    with pytest.raises(TensorRTLLMExecutionError, match="same root mount"):
        session.score(_batch(_record("a", "alpha")))
    session.close()


@pytest.mark.parametrize("replacement_mode", [0o755, 0o500])
def test_run_directory_mode_mutation_fails_before_spawning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    replacement_mode: int,
) -> None:
    spec, _bindings, context = _runtime_inputs(tmp_path)
    session = TensorRTLLMProvider().open(spec, context)
    spawn_calls = 0

    def unexpected_spawn(*_args, **_kwargs):  # noqa: ANN002, ANN003, ANN202
        nonlocal spawn_calls
        spawn_calls += 1
        raise AssertionError("tampered runtime directory reached process spawn")

    monkeypatch.setattr(tensorrt_llm_execution.subprocess, "Popen", unexpected_spawn)
    session._run_directory.path.chmod(replacement_mode)  # noqa: SLF001
    with pytest.raises(TensorRTLLMExecutionError, match="runtime directory changed"):
        session.score(_batch(_record("a", "alpha")))
    assert spawn_calls == 0
    session._run_directory.path.chmod(0o700)  # noqa: SLF001
    session.close()


def test_successful_close_and_constructor_rollback_remove_private_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec, _bindings, context = _runtime_inputs(tmp_path)
    session = TensorRTLLMProvider().open(spec, context)
    successful_root = session._run_directory.path  # noqa: SLF001
    session.close()
    assert not successful_root.exists()

    run_roots: list[Path] = []
    real_create = tensorrt_llm_execution._RunDirectory.create  # noqa: SLF001

    def recording_create():  # noqa: ANN202
        run_directory = real_create()
        run_roots.append(run_directory.path)
        return run_directory

    monkeypatch.setattr(
        tensorrt_llm_execution._RunDirectory,  # noqa: SLF001
        "create",
        staticmethod(recording_create),
    )
    monkeypatch.setattr(
        tensorrt_llm_session,
        "_probe_runner",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            TensorRTLLMExecutionError("probe failed")
        ),
    )
    with pytest.raises(TensorRTLLMExecutionError, match="probe failed"):
        TensorRTLLMProvider().open(spec, context)
    assert len(run_roots) == 1
    assert not run_roots[0].exists()


def test_close_continues_cleanup_after_resource_close_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec, _bindings, context = _runtime_inputs(tmp_path)
    session = TensorRTLLMProvider().open(spec, context)
    run_root = session._run_directory.path  # noqa: SLF001
    runner = session._runner  # noqa: SLF001
    real_runner_close = runner.close

    def fail_runner_close() -> None:
        raise OSError("injected runner close failure")

    monkeypatch.setattr(runner, "close", fail_runner_close)
    with pytest.raises(TensorRTLLMExecutionError, match="cleanup did not complete"):
        session.close()

    assert not run_root.exists()
    assert session._execution_boundary._closed is True  # noqa: SLF001
    assert session._vendor_python._closed is True  # noqa: SLF001
    assert session._tokenizer_source._closed is True  # noqa: SLF001
    real_runner_close()


def test_run_directory_removal_is_attempted_after_preparation_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_directory = tensorrt_llm_execution._RunDirectory.create()  # noqa: SLF001
    run_root = run_directory.path
    run_root.joinpath("payload").write_text("x", encoding="utf-8")

    def fail_walk(*_args, **_kwargs):  # noqa: ANN002, ANN003, ANN202
        raise OSError("injected walk failure")

    monkeypatch.setattr(tensorrt_llm_execution.os, "walk", fail_walk)
    with pytest.raises(TensorRTLLMExecutionError, match="cleanup failed"):
        run_directory.close()
    assert not run_root.exists()


def test_run_directory_create_closes_descriptor_when_identity_read_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_root = tmp_path / "run"
    target_descriptor: int | None = None
    closed_descriptors: list[int] = []
    real_fstat = tensorrt_llm_execution.os.fstat
    real_close = tensorrt_llm_execution.os.close

    def make_run_directory(**_kwargs: object) -> str:
        run_root.mkdir()
        return str(run_root)

    def fail_first_fstat(descriptor: int):
        nonlocal target_descriptor
        if target_descriptor is None:
            target_descriptor = descriptor
            raise OSError("injected identity failure")
        return real_fstat(descriptor)

    def record_close(descriptor: int) -> None:
        closed_descriptors.append(descriptor)
        real_close(descriptor)

    monkeypatch.setattr(tensorrt_llm_execution.tempfile, "mkdtemp", make_run_directory)
    monkeypatch.setattr(tensorrt_llm_execution.os, "fstat", fail_first_fstat)
    monkeypatch.setattr(tensorrt_llm_execution.os, "close", record_close)

    with pytest.raises(OSError, match="injected identity failure"):
        tensorrt_llm_execution._RunDirectory.create()  # noqa: SLF001

    assert target_descriptor in closed_descriptors
    assert not run_root.exists()
