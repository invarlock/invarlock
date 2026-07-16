from __future__ import annotations

import hashlib
import json
import os
import signal
import threading
import time
from dataclasses import asdict, replace
from pathlib import Path

import pytest
from _support import (
    _BACKEND_BUILD_SHA256,
    _BACKEND_VERSION,
    _IMAGE_DIGEST,
    _REQUIRES_POSIX_PINNING,
    _batch,
    _bundle,
    _linux_process_state_is_running,
    _parse_linux_process_stat,
    _process_diagnostic,
    _process_is_running,
    _record,
    _runtime_inputs,
    _write_fake_runner,
    _write_fake_vendor_python,
)
from invarlock_addins.tensorrt_llm import execution as tensorrt_llm_execution
from invarlock_addins.tensorrt_llm import provider as tensorrt_llm_provider
from invarlock_addins.tensorrt_llm import session as tensorrt_llm_session
from invarlock_addins.tensorrt_llm.provider import TensorRTLLMProvider
from invarlock_addins.tensorrt_llm.session import (
    TensorRTLLMExecutionError,
)

from invarlock.core.runtime_provider import (
    EvaluationRecord,
    ModelRuntimeSpec,
    RuntimeArtifactResources,
    RuntimeProvider,
    RuntimeSession,
)
from invarlock.core.runtime_provider.behavioral_observation import (
    runtime_scoring_records_sha256,
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


def test_tensorrt_llm_closed_environment_pins_vendor_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LD_LIBRARY_PATH", "/tmp/ambient-libraries")
    monkeypatch.setenv("OPAL_PREFIX", "/tmp/ambient-opal")
    monkeypatch.setenv("PATH", "/tmp/ambient-bin")
    monkeypatch.setenv("INVARLOCK_TEST_SECRET", "must-not-cross-boundary")
    run_directory = tensorrt_llm_execution._RunDirectory(  # noqa: SLF001
        path=tmp_path,
        descriptor=-1,
        initial_stat=tmp_path.stat(),
    )
    monkeypatch.setattr(run_directory, "recheck", lambda: None)

    environment = run_directory.environment()

    rendered = str(tmp_path)
    assert environment == {
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "DO_NOT_TRACK": "1",
        "FORCE_DETERMINISTIC": "1",
        "HF_DATASETS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "HF_HUB_OFFLINE": "1",
        "HOME": rendered,
        "INVARLOCK_CONTAINER_EXECUTION": "1",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "LD_LIBRARY_PATH": "/usr/local/tensorrt/lib",
        "NO_COLOR": "1",
        "NO_PROXY": "*",
        "OPAL_PREFIX": "/opt/hpcx/ompi",
        "PATH": "/opt/hpcx/ompi/bin:/usr/bin:/bin",
        "TELEMETRY_DISABLED": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "TRANSFORMERS_OFFLINE": "1",
        "TRTLLM_NO_USAGE_STATS": "1",
        "TMPDIR": rendered,
        "XDG_CACHE_HOME": rendered,
    }
    fixed_paths = [environment["OPAL_PREFIX"]]
    fixed_paths.extend(
        entry
        for variable in ("LD_LIBRARY_PATH", "PATH")
        for entry in environment[variable].split(":")
    )
    assert all(
        Path(entry).is_absolute() and entry not in {rendered, "/tmp", "/var/tmp"}
        for entry in fixed_paths
    )


def test_tensorrt_llm_config_identity_capabilities_and_private_bindings(
    tmp_path: Path,
) -> None:
    spec, bindings, _context = _runtime_inputs(tmp_path)
    provider = TensorRTLLMProvider()

    provider.validate_config(spec)
    identity = provider.identify_artifact(spec)

    assert isinstance(provider, RuntimeProvider)
    assert identity.bundle_name == spec.model_id
    assert (
        identity.engine_bundle_tree_sha256
        == (spec.settings["engine_bundle_tree_sha256"])
    )
    assert provider.capabilities().artifact_formats == ("tensorrt_llm_engine",)
    assert provider.capabilities().metrics == ("exact_match",)
    assert provider.capabilities().execution_modes == ("container",)
    assert provider.capabilities().tasks == ("text_causal",)
    assert str(bindings.engine_bundle_path) not in repr(bindings)
    assert str(bindings.tokenizer_contract_path) not in repr(bindings)
    assert str(bindings.runner_executable_path) not in repr(bindings)
    assert all("path" not in name for name in spec.settings)


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        ({"strict": False}, "strict mode"),
        ({"allow_network": True}, "disable network"),
        ({"container_image_digest": None}, "container image"),
        ({"artifact_identity_sha256": None}, "artifact identity"),
        ({"artifact_identity_sha256": "0" * 64}, "does not match"),
        ({"device_kind": "cpu"}, "CUDA device"),
        ({"provider_state": None}, "runtime bindings"),
    ],
)
def test_tensorrt_llm_open_rejects_missing_security_bindings(
    tmp_path: Path, replacement: dict[str, object], message: str
) -> None:
    spec, _bindings, context = _runtime_inputs(tmp_path)
    with pytest.raises((ValueError, TensorRTLLMExecutionError), match=message):
        TensorRTLLMProvider().open(spec, replace(context, **replacement))


def test_tensorrt_llm_rejects_host_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec, _bindings, context = _runtime_inputs(tmp_path)
    monkeypatch.delenv("INVARLOCK_CONTAINER_EXECUTION", raising=False)
    monkeypatch.setattr(
        tensorrt_llm_provider,
        "strict_container_boundary_present",
        lambda: False,
    )
    with pytest.raises(ValueError, match="container boundary"):
        TensorRTLLMProvider().open(spec, context)


def test_tensorrt_llm_rejects_marker_only_container_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec, _bindings, context = _runtime_inputs(tmp_path)
    monkeypatch.setenv("INVARLOCK_CONTAINER_EXECUTION", "1")
    monkeypatch.setattr(
        tensorrt_llm_provider,
        "strict_container_boundary_present",
        lambda: False,
    )
    with pytest.raises(ValueError, match="container boundary"):
        TensorRTLLMProvider().open(spec, context)


def test_tensorrt_llm_rejects_unobserved_runtime_image_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec, _bindings, context = _runtime_inputs(tmp_path)
    monkeypatch.setenv("INVARLOCK_RUNTIME_IMAGE_DIGEST", "sha256:" + "f" * 64)

    with pytest.raises(ValueError, match="does not match the container context"):
        TensorRTLLMProvider().open(spec, context)

    monkeypatch.setenv("INVARLOCK_RUNTIME_IMAGE_DIGEST", _IMAGE_DIGEST)
    monkeypatch.setenv("INVARLOCK_RUNTIME_IMAGE", "mutable:tag")
    with pytest.raises(ValueError, match="embed the exact runtime image digest"):
        TensorRTLLMProvider().open(spec, context)


def test_tensorrt_llm_rejects_tokenizer_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec, bindings, context = _runtime_inputs(tmp_path)
    monkeypatch.setenv("INVARLOCK_CONTAINER_EXECUTION", "1")
    bindings.tokenizer_contract_path.write_bytes(b"changed")
    with pytest.raises(TensorRTLLMExecutionError, match="digest"):
        TensorRTLLMProvider().open(spec, context)


def test_tensorrt_llm_snapshot_rejects_multi_rank_engine(tmp_path: Path) -> None:
    source = _bundle(tmp_path / "source")
    config = json.loads(source.joinpath("config.json").read_text(encoding="utf-8"))
    config["pretrained_config"]["mapping"] = {
        "pp_size": 1,
        "tp_size": 2,
        "world_size": 2,
    }
    source.joinpath("config.json").write_text(
        json.dumps(config, sort_keys=True, separators=(",", ":")), encoding="utf-8"
    )
    source.joinpath("rank1.engine").write_bytes(b"second rank")

    with pytest.raises(TensorRTLLMExecutionError, match="single-rank"):
        tensorrt_llm_session._snapshot_bundle(  # noqa: SLF001
            source, tmp_path / "destination"
        )


def test_tensorrt_llm_prepare_execution_binds_root_confined_resources(
    tmp_path: Path,
) -> None:
    spec, bindings, _context = _runtime_inputs(tmp_path)
    root = Path("/")

    context = TensorRTLLMProvider().prepare_execution(
        spec,
        RuntimeArtifactResources(
            root=root,
            primary_artifact=str(bindings.engine_bundle_path).removeprefix("/"),
            support_resources={
                "runner_executable": str(bindings.runner_executable_path).removeprefix(
                    "/"
                ),
                "tokenizer_contract": str(
                    bindings.tokenizer_contract_path
                ).removeprefix("/"),
            },
            device_kind="cuda",
            container_image_digest=_IMAGE_DIGEST,
        ),
    )

    assert context.provider_state == bindings
    assert context.allow_network is False
    assert str(tmp_path) not in repr(context)


def test_tensorrt_llm_prepare_execution_rejects_missing_tokenizer(
    tmp_path: Path,
) -> None:
    spec, bindings, _context = _runtime_inputs(tmp_path)
    root = Path("/")
    resources = RuntimeArtifactResources(
        root=root,
        primary_artifact=str(bindings.engine_bundle_path).removeprefix("/"),
        support_resources={
            "runner_executable": str(bindings.runner_executable_path).removeprefix("/")
        },
        device_kind="cuda",
        container_image_digest=_IMAGE_DIGEST,
    )

    with pytest.raises(ValueError, match="tokenizer_contract"):
        TensorRTLLMProvider().prepare_execution(spec, resources)


@_REQUIRES_POSIX_PINNING
def test_tensorrt_llm_scores_in_order_and_emits_bound_receipt(
    tmp_path: Path,
) -> None:
    spec, bindings, _context = _runtime_inputs(tmp_path)
    provider = TensorRTLLMProvider()
    root = Path("/")
    context = provider.prepare_execution(
        spec,
        RuntimeArtifactResources(
            root=root,
            primary_artifact=str(bindings.engine_bundle_path).removeprefix("/"),
            support_resources={
                "runner_executable": str(bindings.runner_executable_path).removeprefix(
                    "/"
                ),
                "tokenizer_contract": str(
                    bindings.tokenizer_contract_path
                ).removeprefix("/"),
            },
            device_kind="cuda",
            container_image_digest=_IMAGE_DIGEST,
        ),
    )
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
    assert observation.aggregate_source_sha256 == runtime_scoring_records_sha256(
        [asdict(record) for record in observation.records]
    )
    assert receipt.backend.name == "TensorRT-LLM"
    assert receipt.backend.version == _BACKEND_VERSION
    assert receipt.backend.binary_sha256 == spec.settings["runner_binary_sha256"]
    assert receipt.backend.build_sha256 == _BACKEND_BUILD_SHA256
    assert receipt.backend.source_sha256 is None
    assert receipt.artifact_identity == provider.identify_artifact(spec)
    assert receipt.outer_image_digest == _IMAGE_DIGEST
    assert receipt.device.device_name == "Observed NVIDIA H200"
    assert receipt.device.compute_capability == "9.0"
    assert receipt.device.driver_version == "570.00"
    assert receipt.device.cuda_runtime_version == "12.8"
    assert (
        receipt.scoring_observation_sha256
        == hashlib.sha256(
            json.dumps(
                asdict(observation),
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
    )
    session.close()
    session.close()
    with pytest.raises(RuntimeError, match="closed"):
        session.score(_batch(_record("c", "gamma")))


@_REQUIRES_POSIX_PINNING
def test_tensorrt_llm_rejects_runner_identity_mismatch(tmp_path: Path) -> None:
    spec, _bindings, context = _runtime_inputs(tmp_path)
    wrong_version = ModelRuntimeSpec(
        provider_name=spec.provider_name,
        model_id=spec.model_id,
        settings={**spec.settings, "backend_version": "1.2.2"},
    )
    with pytest.raises(TensorRTLLMExecutionError, match="identity"):
        TensorRTLLMProvider().open(wrong_version, context)


@_REQUIRES_POSIX_PINNING
def test_tensorrt_llm_rejects_observed_compute_capability_mismatch(
    tmp_path: Path,
) -> None:
    spec, bindings, context = _runtime_inputs(tmp_path)
    _write_fake_runner(bindings.runner_executable_path, compute_capability="8.9")
    runner_sha256 = hashlib.sha256(
        bindings.runner_executable_path.read_bytes()
    ).hexdigest()
    mismatched_runner = ModelRuntimeSpec(
        provider_name=spec.provider_name,
        model_id=spec.model_id,
        settings={**spec.settings, "runner_binary_sha256": runner_sha256},
    )

    with pytest.raises(TensorRTLLMExecutionError, match="observed CUDA"):
        TensorRTLLMProvider().open(mismatched_runner, context)


@_REQUIRES_POSIX_PINNING
def test_tensorrt_llm_rejects_input_digest_and_runner_path_swap(
    tmp_path: Path,
) -> None:
    spec, bindings, context = _runtime_inputs(tmp_path)
    session = TensorRTLLMProvider().open(spec, context)
    invalid = EvaluationRecord(
        record_id="bad-digest",
        input_text="alpha",
        input_sha256="0" * 64,
    )
    with pytest.raises(ValueError, match="does not match authenticated input material"):
        session.score(_batch(invalid))
    with pytest.raises(ValueError, match="supports only text_causal"):
        session.score(
            replace(
                _batch(_record("wrong-task", "alpha")),
                task="vision_text_generation",
            )
        )

    bindings.runner_executable_path.unlink()
    _write_fake_runner(bindings.runner_executable_path)
    with pytest.raises(TensorRTLLMExecutionError, match="identity changed"):
        session.score(_batch(_record("a", "alpha")))


@_REQUIRES_POSIX_PINNING
def test_tensorrt_llm_uses_closed_request_and_sanitized_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec, _bindings, context = _runtime_inputs(tmp_path)
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    real_popen = tensorrt_llm_execution.subprocess.Popen

    def recording_popen(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        calls.append((args, dict(kwargs)))
        return real_popen(*args, **kwargs)

    monkeypatch.setattr(tensorrt_llm_execution.subprocess, "Popen", recording_popen)
    monkeypatch.setenv("HF_TOKEN", "private-token")
    monkeypatch.setenv("HTTP_PROXY", "http://private-proxy")
    monkeypatch.setenv("INVARLOCK_TEST_SECRET", "private-value")
    session = TensorRTLLMProvider().open(spec, context)

    request_record = session.score(_batch(_record("request", "__request__"))).records[0]
    request = json.loads(request_record.output_text)
    assert request["format_version"] == "invarlock/tensorrt-llm-runner-request-v1"
    assert request["protocol_version"] == "invarlock/tensorrt-llm-runner-v1"
    assert request["settings"] == {
        "allow_network": False,
        "batch_size": 4,
        "context_length": 256,
        "max_output_tokens": 16,
        "seed": 7,
        "timeout_seconds": 1,
    }
    assert request["engine_bundle"].endswith("/engine")
    assert request["tokenizer_contract"].endswith("/tokenizer.json")

    env_record = session.score(_batch(_record("env", "__env__"))).records[0]
    environment = json.loads(env_record.output_text)
    assert environment["DO_NOT_TRACK"] == "1"
    assert environment["FORCE_DETERMINISTIC"] == "1"
    assert environment["HF_HUB_OFFLINE"] == "1"
    assert environment["TRANSFORMERS_OFFLINE"] == "1"
    assert environment["NO_PROXY"] == "*"
    assert environment["INVARLOCK_CONTAINER_EXECUTION"] == "1"
    assert environment["TELEMETRY_DISABLED"] == "1"
    assert environment["TRTLLM_NO_USAGE_STATS"] == "1"
    assert environment["HOME"] == environment["TMPDIR"]
    assert environment["HOME"] == environment["XDG_CACHE_HOME"]
    assert "HF_TOKEN" not in environment
    assert "HTTP_PROXY" not in environment
    assert "INVARLOCK_TEST_SECRET" not in environment
    assert calls
    vendor_python_path = str(session._vendor_python.path)  # noqa: SLF001
    runner_path = str(session._runner.path)  # noqa: SLF001
    run_path = session._run_directory.path  # noqa: SLF001
    assert [tuple(positional[0][2:]) for positional, _keywords in calls] == [
        ("--invarlock-runtime-info-v1",),
        ("--invarlock-score-v1",),
        ("--invarlock-score-v1",),
    ]
    for positional, keywords in calls:
        argv = positional[0]
        assert keywords["executable"] == argv[0] == vendor_python_path
        assert argv[1] == runner_path
        assert "/proc/self/fd/" not in " ".join(argv)
        assert keywords["cwd"] == run_path
        process_environment = keywords["env"]
        assert process_environment["HOME"] == str(run_path)
        assert process_environment["TMPDIR"] == str(run_path)
        assert process_environment["XDG_CACHE_HOME"] == str(run_path)
        assert keywords["shell"] is False
        assert keywords["close_fds"] is True
        assert keywords["pass_fds"] == ()
        assert keywords["start_new_session"] is True


@pytest.mark.parametrize(
    ("prompt", "message"),
    [
        ("__sleep__", "timed out"),
        ("__flood__", "limit exceeded"),
        ("__stderr__", "emitted stderr"),
        ("__fail__", "status 7"),
        ("__bad_json__", "strict JSON"),
        ("__duplicate__", "duplicate key"),
        ("__extra__", "unexpected fields"),
    ],
)
@_REQUIRES_POSIX_PINNING
def test_tensorrt_llm_fails_closed_on_runner_errors(
    tmp_path: Path, prompt: str, message: str
) -> None:
    spec, _bindings, context = _runtime_inputs(tmp_path)
    session = TensorRTLLMProvider().open(spec, context)
    with pytest.raises(TensorRTLLMExecutionError, match=message):
        session.score(_batch(_record("bad", prompt)))


@_REQUIRES_POSIX_PINNING
def test_tensorrt_llm_timeout_kills_the_child_process_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec, _bindings, context = _runtime_inputs(tmp_path)
    killed: list[int] = []
    real_kill = tensorrt_llm_execution._kill_process_group

    def recording_kill(process):  # noqa: ANN001, ANN202
        killed.append(process.pid)
        real_kill(process)

    monkeypatch.setattr(tensorrt_llm_execution, "_kill_process_group", recording_kill)
    session = TensorRTLLMProvider().open(spec, context)
    with pytest.raises(TensorRTLLMExecutionError, match="timed out"):
        session.score(_batch(_record("sleep", "__sleep__")))
    assert killed


def test_tensorrt_llm_kills_process_group_after_leader_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[int, signal.Signals]] = []

    class ExitedLeader:
        pid = 24680

        @staticmethod
        def wait(*, timeout: float) -> int:
            assert timeout == 2
            return 0

        @staticmethod
        def kill() -> None:
            raise AssertionError(
                "the already-exited leader must not need a direct kill"
            )

    monkeypatch.setattr(
        tensorrt_llm_execution.os,
        "killpg",
        lambda process_group, sent_signal: calls.append((process_group, sent_signal)),
    )

    tensorrt_llm_execution._kill_process_group(ExitedLeader())  # type: ignore[arg-type]

    assert calls == [(ExitedLeader.pid, signal.SIGKILL)]


@pytest.mark.parametrize(
    ("state", "expected_running"),
    [("R", True), ("S", True), ("D", True), ("Z", False), ("X", False), ("x", False)],
)
def test_linux_process_state_classification(state: str, expected_running: bool) -> None:
    assert _linux_process_state_is_running(state) is expected_running


def test_linux_process_stat_parser_handles_complex_command_names() -> None:
    fields = ["S", "1", "77", *(["0"] * 16), "999"]
    process_stat = f"123 (worker ) with spaces) {' '.join(fields)}"

    assert _parse_linux_process_stat(process_stat) == ("S", 77, 999)


@pytest.mark.parametrize(
    "process_stat",
    [
        "123 worker S 1 2",
        "123 (worker) S 1 2",
        "123 (worker) S 1 invalid " + "0 " * 17,
    ],
)
def test_linux_process_stat_parser_rejects_malformed_records(
    process_stat: str,
) -> None:
    with pytest.raises(ValueError, match="Linux process stat"):
        _parse_linux_process_stat(process_stat)


@_REQUIRES_POSIX_PINNING
def test_tensorrt_llm_timeout_kills_descendant_after_leader_exits(
    tmp_path: Path,
) -> None:
    spec, _bindings, context = _runtime_inputs(tmp_path)
    session = TensorRTLLMProvider().open(spec, context)
    pid_path = session._run_directory.path / "grandchild.pid"  # noqa: SLF001

    with pytest.raises(TensorRTLLMExecutionError, match="timed out"):
        session.score(_batch(_record("orphan", "__orphan_pipe__")))

    grandchild_pid = int(pid_path.read_text(encoding="ascii"))
    try:
        deadline = time.monotonic() + 10
        while _process_is_running(grandchild_pid) and time.monotonic() < deadline:
            time.sleep(0.01)
        assert not _process_is_running(grandchild_pid), _process_diagnostic(
            grandchild_pid
        )
    finally:
        if _process_is_running(grandchild_pid):
            os.kill(grandchild_pid, signal.SIGKILL)
        session.close()


@_REQUIRES_POSIX_PINNING
@pytest.mark.parametrize("_attempt", range(5))
def test_tensorrt_llm_success_kills_descendant_that_closed_inherited_fds(
    tmp_path: Path,
    _attempt: int,
) -> None:
    spec, _bindings, context = _runtime_inputs(tmp_path)
    session = TensorRTLLMProvider().open(spec, context)
    pid_path = session._run_directory.path / "detached.pid"  # noqa: SLF001

    observation = session.score(_batch(_record("detached", "__detached_success__")))

    assert observation.records[0].output_text == "OUT:__detached_success__"
    grandchild_pid = int(pid_path.read_text(encoding="ascii"))
    try:
        deadline = time.monotonic() + 10
        while _process_is_running(grandchild_pid) and time.monotonic() < deadline:
            time.sleep(0.01)
        assert not _process_is_running(grandchild_pid), _process_diagnostic(
            grandchild_pid
        )
    finally:
        if _process_is_running(grandchild_pid):
            os.kill(grandchild_pid, signal.SIGKILL)
        session.close()


@_REQUIRES_POSIX_PINNING
def test_tensorrt_llm_close_waits_for_active_score_and_then_removes_tree(
    tmp_path: Path,
) -> None:
    spec, _bindings, context = _runtime_inputs(tmp_path)
    session = TensorRTLLMProvider().open(spec, context)
    run_root = session._run_directory.path  # noqa: SLF001
    started_path = run_root / "score.started"
    release_path = run_root / "score.release"
    score_results: list[str] = []
    thread_errors: list[BaseException] = []
    close_started = threading.Event()
    close_lock_attempted = threading.Event()
    close_finished = threading.Event()

    class ObservedLock:
        def __init__(self) -> None:
            self._lock = threading.Lock()

        def __enter__(self) -> ObservedLock:
            if threading.current_thread().name == "tensorrt-close":
                close_lock_attempted.set()
            self._lock.acquire()
            return self

        def __exit__(
            self,
            _exc_type: object,
            _exc: object,
            _traceback: object,
        ) -> None:
            self._lock.release()

    session._score_lock = ObservedLock()  # type: ignore[assignment]  # noqa: SLF001

    def score() -> None:
        try:
            observation = session.score(_batch(_record("wait", "__wait_for_release__")))
            score_results.append(observation.records[0].output_text or "")
        except BaseException as exc:
            thread_errors.append(exc)

    def close() -> None:
        close_started.set()
        try:
            session.close()
        except BaseException as exc:
            thread_errors.append(exc)
        finally:
            close_finished.set()

    score_thread = threading.Thread(target=score, name="tensorrt-score")
    score_thread.start()
    close_thread = threading.Thread(target=close, name="tensorrt-close")
    close_thread_started = False
    try:
        deadline = time.monotonic() + 1
        while not started_path.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert started_path.exists()

        close_thread.start()
        close_thread_started = True
        assert close_started.wait(timeout=1)
        assert close_lock_attempted.wait(timeout=1)
        assert not close_finished.wait(timeout=0.1)
        assert run_root.exists()
    finally:
        if run_root.exists():
            release_path.write_text("release", encoding="ascii")
        score_thread.join(timeout=2)
        if close_thread_started:
            close_thread.join(timeout=2)
        elif run_root.exists():
            session.close()

    assert not score_thread.is_alive()
    assert not close_thread.is_alive()
    assert thread_errors == []
    assert score_results == ["released"]
    assert not run_root.exists()


@_REQUIRES_POSIX_PINNING
def test_tensorrt_llm_uses_private_engine_and_tokenizer_snapshots(
    tmp_path: Path,
) -> None:
    spec, bindings, context = _runtime_inputs(tmp_path)
    session = TensorRTLLMProvider().open(spec, context)

    # Engine execution uses the authenticated private bundle snapshot.
    bindings.engine_bundle_path.joinpath("rank0.engine").write_bytes(b"source changed")
    observation = session.score(_batch(_record("a", "alpha")))
    assert observation.records[0].output_text == "OUT:alpha"

    session._tokenizer_snapshot.chmod(0o600)  # noqa: SLF001
    session._tokenizer_snapshot.write_bytes(b"snapshot changed")  # noqa: SLF001
    with pytest.raises(TensorRTLLMExecutionError, match="tokenizer contract changed"):
        session.score(_batch(_record("b", "beta")))


def test_tensorrt_llm_modules_remain_torch_free() -> None:
    root = (
        Path.cwd()
        / "addins"
        / "tensorrt_llm"
        / "src"
        / "invarlock_addins"
        / "tensorrt_llm"
    )
    source = "\n".join(
        root.joinpath(name).read_text(encoding="utf-8")
        for name in (
            "provider.py",
            "execution.py",
            "inspection.py",
            "runner.py",
            "session.py",
        )
    )
    assert "import torch" not in source
    assert "import tensorrt" not in source
    assert "import transformers" not in source
