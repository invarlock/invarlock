"""Opt-in adversarial checks through the real OCI composer and worker lifecycle.

The probe replaces only the model payload. Security flags, mount projection,
bounded diagnostics, deadlines and cleanup all come from production code.
"""

from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import threading
import time
import uuid
from pathlib import Path

import pytest

import invarlock.evaluation_oci as oci

pytestmark = pytest.mark.skipif(
    os.environ.get("INVARLOCK_RUN_CONTAINER_SMOKE") != "1",
    reason="set INVARLOCK_RUN_CONTAINER_SMOKE=1 for actual OCI isolation checks",
)

NETWORK_PROBE = r"""
import ctypes, errno, json, socket, struct, subprocess, sys
from pathlib import Path
config = json.loads(Path('/invarlock/job/probe.json').read_text())
host, port = config['host'], config['port']
results = {}
try:
    with socket.create_connection((host, port), timeout=1):
        results['python'] = True
except OSError:
    results['python'] = False
libc = ctypes.CDLL(None, use_errno=True)
descriptor = libc.socket(2, 1 | 2048, 0)
assert descriptor >= 0
address = struct.pack('H', 2) + struct.pack('!H', port) + socket.inet_aton(host) + bytes(8)
buffer = ctypes.create_string_buffer(address)
status = libc.connect(descriptor, buffer, len(address))
if status == 0:
    results['native'] = True
elif ctypes.get_errno() == errno.EINPROGRESS:
    import select
    _, ready, _ = select.select([], [descriptor], [], 1)
    error = ctypes.c_int(-1)
    length = ctypes.c_uint(ctypes.sizeof(error))
    status = libc.getsockopt(descriptor, 1, 4, ctypes.byref(error), ctypes.byref(length))
    results['native'] = bool(ready) and status == 0 and error.value == 0
else:
    results['native'] = False
libc.close(descriptor)
child = subprocess.run([sys.executable, '-c',
    'import socket,sys; socket.create_connection((sys.argv[1],int(sys.argv[2])),timeout=1).close()',
    host, str(port)], capture_output=True, timeout=5)
results['subprocess'] = child.returncode == 0
print(json.dumps(results), flush=True)
"""

FILESYSTEM_PROBE = r"""
import json, os
from pathlib import Path
config = json.loads(Path('/invarlock/job/probe.json').read_text())
results = {'uid': os.getuid(), 'key_visible': Path(config['key']).exists(),
           'signing_environment': os.environ.get('INVARLOCK_SIGNING_KEY')}
for label, path in [('artifact', '/invarlock-resources/artifact/input.txt'),
                    ('job', '/invarlock/job/mutable.txt')]:
    try:
        Path(path).write_text('changed')
        results[label + '_writable'] = True
    except OSError as error:
        results[label + '_writable'] = False
        results[label + '_errno'] = error.errno
Path('/invarlock/output-root/probe-output.json').write_text(json.dumps(results))
print(json.dumps(results), flush=True)
"""


def _engine(engine: str, *arguments: str, check: bool = True):
    return subprocess.run(
        [engine, *arguments], capture_output=True, text=True, timeout=30, check=check
    )


@pytest.fixture(scope="module")
def runtime():
    engine = os.environ.get("INVARLOCK_CONTAINER_ENGINE", "docker")
    image = os.environ.get("INVARLOCK_RUNTIME_IMAGE", "invarlock-runtime:local")
    metadata = json.loads(_engine(engine, "image", "inspect", image).stdout)[0]
    image_id = metadata["Id"]
    for variable, label in (
        ("RUNTIME_SOURCE_COMMIT", "org.opencontainers.image.revision"),
        ("RUNTIME_SOURCE_BUNDLE_SHA256", "dev.invarlock.source-bundle-sha256"),
    ):
        if expected := os.environ.get(variable):
            assert metadata["Config"]["Labels"][label] == expected
    side = oci.OciSideLaunch(image_id, image_id, "cpu", "python")
    launch = oci.OciEvaluationLaunch(
        engine=engine,
        baseline=side,
        subject=side,
        worker_limits=oci.OciWorkerLimits(cpus="1", memory_mib=256),
    )
    assert oci.preflight_oci_launch(launch) == {
        "baseline": image_id,
        "subject": image_id,
    }
    yield launch


@pytest.fixture(scope="module")
def reachable_service(runtime):
    """An owned internal network supplies a reachable, internet-free control."""
    engine = runtime.engine
    name = "invarlock-isolation-" + uuid.uuid4().hex[:12]
    network = _engine(engine, "network", "create", "--internal", name).stdout.strip()
    container = None
    try:
        container = _engine(
            engine,
            "run",
            "--detach",
            "--rm",
            "--pull=never",
            "--network",
            name,
            "--read-only",
            "--cap-drop=ALL",
            "--security-opt",
            "no-new-privileges",
            "--cpus",
            "0.25",
            "--memory",
            "128m",
            "--pids-limit",
            "32",
            "--user",
            "65532:65532",
            "--entrypoint",
            "python",
            runtime.baseline.image_ref,
            "-m",
            "http.server",
            "9000",
            "--bind",
            "0.0.0.0",
        ).stdout.strip()
        metadata = json.loads(
            _engine(engine, "container", "inspect", container).stdout
        )[0]
        address = metadata["NetworkSettings"]["Networks"][name]["IPAddress"]
        ready = _engine(
            engine,
            "exec",
            container,
            "python",
            "-c",
            "import socket,time; "
            "exec('for attempt in range(50):\\n"
            ' try:\\n  socket.create_connection(("127.0.0.1",9000),timeout=1).close(); break\\n'
            " except OSError:\\n  time.sleep(0.1)\\n"
            "else:\\n raise SystemExit(1)')",
            check=False,
        )
        assert ready.returncode == 0, (
            ready.stderr + _engine(engine, "logs", container).stdout
        )
        yield {"name": name, "address": address, "container": container}
    finally:
        if container:
            _engine(engine, "stop", "--time", "1", container, check=False)
        _engine(engine, "network", "rm", network, check=False)


def _command(runtime, root: Path, payload: str, config=None):
    root.mkdir()
    job, artifact, output = (root / name for name in ("job", "artifact", "output"))
    for path in (job, artifact, output):
        path.mkdir(mode=0o777)
        path.chmod(0o777)
    for path in (job / "mutable.txt", artifact / "input.txt"):
        path.write_text("original")
        path.chmod(0o666)
    (job / "probe.json").write_text(json.dumps(config or {}))
    command = oci.compose_side_worker_command(
        launch=runtime,
        side_launch=runtime.baseline,
        provider_name="hf_transformers",
        artifact_source=artifact,
        support_sources={},
        job_root=job,
        output_root=output,
    )
    assert command[-3:] == [
        "-m",
        "invarlock.evaluation_side_worker",
        "/invarlock/job/job.json",
    ]
    command[-3:] = ["-c", payload]
    return command


def _record(name, runtime, command, result, **observations):
    destination = os.environ.get("INVARLOCK_OCI_ISOLATION_RESULTS")
    if not destination:
        return
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    payload = {
        "case": name,
        "image_id": runtime.baseline.image_digest,
        "source_commit": os.environ.get("RUNTIME_SOURCE_COMMIT"),
        "source_bundle_sha256": os.environ.get("RUNTIME_SOURCE_BUNDLE_SHA256"),
        "engine": _engine(runtime.engine, "version", "--format", "{{json .}}").stdout,
        "command": command,
        "returncode": result.returncode if result is not None else None,
        "stdout": result.stdout if result is not None else None,
        "stderr": result.stderr if result is not None else None,
        "test_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "host_oci_source_sha256": hashlib.sha256(
            Path(oci.__file__).read_bytes()
        ).hexdigest(),
        "observations": observations,
    }
    (root / f"{name}.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )


def test_python_native_and_subprocess_network_isolation_with_positive_control(
    runtime, reachable_service, tmp_path
):
    config = {"host": reachable_service["address"], "port": 9000}
    for label, network, expected in (
        ("network-control", reachable_service["name"], True),
        ("network-isolated", "none", False),
    ):
        command = _command(runtime, tmp_path / label, NETWORK_PROBE, config)
        command[command.index("--network") + 1] = network
        result = oci.run_side_worker(command, timeout_seconds=20)
        assert result.returncode == 0, result.stderr
        observed = json.loads(result.stdout)
        _record(label, runtime, command, result, network_attempts=observed)
        assert observed == dict.fromkeys(("python", "native", "subprocess"), expected)


def test_readonly_inputs_and_signing_key_absence_with_positive_controls(
    runtime, tmp_path, monkeypatch
):
    key = tmp_path / "caller-signing-key-canary.pem"
    key.write_text("test-only-key-canary")
    key.chmod(0o644)
    monkeypatch.setenv("INVARLOCK_SIGNING_KEY", "test-only-signing-environment")
    for label, writable in (("mount-control", True), ("mount-isolated", False)):
        command = _command(
            runtime, tmp_path / label, FILESYSTEM_PROBE, {"key": str(key)}
        )
        if writable:
            command = [argument.removesuffix(",readonly") for argument in command]
            index = command.index("--entrypoint")
            command[index:index] = [
                "--mount",
                f"type=bind,source={key},target={key},readonly",
                "-e",
                "INVARLOCK_SIGNING_KEY=test-only-signing-environment",
            ]
        result = oci.run_side_worker(command, timeout_seconds=20)
        assert result.returncode == 0, result.stderr
        observed = json.loads(result.stdout)
        assert observed["uid"] == 65532
        assert observed["artifact_writable"] is writable
        assert observed["job_writable"] is writable
        assert observed["key_visible"] is writable
        assert observed["signing_environment"] == (
            "test-only-signing-environment" if writable else None
        )
        assert (tmp_path / label / "output/probe-output.json").is_file()
        if not writable:
            assert (tmp_path / label / "artifact/input.txt").read_text() == "original"
            assert (tmp_path / label / "job/mutable.txt").read_text() == "original"
        _record(label, runtime, command, result, mounts=observed)


def test_cgroup_resource_limits_and_memory_enforcement(runtime, tmp_path):
    payload = """
import json
from pathlib import Path
root = Path('/sys/fs/cgroup')
print(json.dumps({name: (root / name).read_text().strip()
    for name in ('memory.max', 'cpu.max', 'pids.max')}), flush=True)
"""
    command = _command(runtime, tmp_path / "resources", payload)
    result = oci.run_side_worker(command, timeout_seconds=20)
    assert result.returncode == 0, result.stderr
    limits = json.loads(result.stdout)
    assert int(limits["memory.max"]) == 256 * 1024 * 1024
    quota, period = map(int, limits["cpu.max"].split())
    assert quota / period == 1
    assert int(limits["pids.max"]) == 1024
    _record("resource-limits", runtime, command, result, cgroups=limits)
    for label, size, expected in (
        ("memory-control", 8, 0),
        ("memory-exhausted", 512, 137),
    ):
        command = _command(
            runtime,
            tmp_path / label,
            f"data=bytearray({size}*1024*1024); "
            "data[::4096]=b'x'*len(data[::4096]); print(len(data), flush=True)",
        )
        result = oci.run_side_worker(command, timeout_seconds=20)
        assert result.returncode == expected, result.stderr
        _record(label, runtime, command, result, allocated_mib=size)


def test_output_flood_is_drained_with_bounded_diagnostics(runtime, tmp_path):
    payload = "import os; block=b'x'*8192; [(os.write(1,block),os.write(2,block)) for _ in range(256)]"
    command = _command(runtime, tmp_path / "output-flood", payload)
    result = oci.run_side_worker(command, timeout_seconds=20)
    assert result.returncode == 0
    assert len(result.stdout) == len(result.stderr) == oci._MAX_WORKER_DIAGNOSTIC_BYTES
    _record(
        "output-flood",
        runtime,
        command,
        result,
        emitted_bytes_per_stream=2 * 1024 * 1024,
    )


@pytest.mark.parametrize("mode", ["deadline", "cancel", "interrupt"])
def test_cleanup_uses_exact_container_id_and_preserves_other_containers(
    runtime, reachable_service, tmp_path, monkeypatch, mode
):
    command = _command(
        runtime,
        tmp_path / mode,
        "import signal,time; from pathlib import Path; "
        "signal.signal(signal.SIGTERM,signal.SIG_IGN); "
        "Path('/invarlock/output-root/ready').touch(); time.sleep(120)",
    )
    cidfile = Path(command[command.index("--cidfile") + 1])
    identifiers = []
    controls = []
    errors = []
    cancellation = threading.Event()
    original = oci._container_control

    def control(engine, action, identifier):
        controls.append((action, identifier))
        return original(engine, action, identifier)

    monkeypatch.setattr(oci, "_container_control", control)

    def trigger():
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            identifier = oci._read_worker_container_id(cidfile)
            if (
                identifier is not None
                and len(identifier) == 64
                and (tmp_path / mode / "output/ready").exists()
            ):
                identifiers.append(identifier)
                if mode == "cancel":
                    cancellation.set()
                elif mode == "interrupt":
                    os.kill(os.getpid(), signal.SIGINT)
                return
            time.sleep(0.02)
        errors.append("engine never issued a container ID")

    watcher = threading.Thread(target=trigger, daemon=True)
    watcher.start()
    started = time.monotonic()
    result = None
    try:
        if mode == "interrupt":
            with pytest.raises(KeyboardInterrupt):
                oci.run_side_worker(command, timeout_seconds=30)
        else:
            result = oci.run_side_worker(
                command,
                timeout_seconds=2 if mode == "deadline" else 30,
                cancellation_event=cancellation if mode == "cancel" else None,
            )
            assert result.returncode == (124 if mode == "deadline" else 125)
    finally:
        watcher.join(timeout=16)
    elapsed = time.monotonic() - started
    assert not errors
    assert len(identifiers) == 1
    assert controls and {identifier for _, identifier in controls} == set(identifiers)
    assert not cidfile.exists()
    assert (
        _engine(
            runtime.engine, "container", "inspect", identifiers[0], check=False
        ).returncode
        != 0
    )
    service = json.loads(
        _engine(
            runtime.engine, "container", "inspect", reachable_service["container"]
        ).stdout
    )[0]
    assert service["State"]["Running"] is True
    assert elapsed < 25
    _record(
        mode,
        runtime,
        command,
        result,
        elapsed_seconds=elapsed,
        container_ids=identifiers,
        cleanup_controls=controls,
        unrelated_container_running=True,
    )


@pytest.mark.parametrize("mode", ["nonzero", "malformed"])
def test_failed_real_worker_cannot_publish_completed_transaction(
    runtime, tmp_path, monkeypatch, mode
):
    from invarlock.evaluation_transaction import (
        EvaluationTransactionError,
        evaluate_request_file,
    )
    from tests.integration.test_container_front_door_journey import (
        _private_key,
        _request,
        _tiny_checkpoint,
    )

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _checkpoint, checkpoint_digest, tokenizer_digest = _tiny_checkpoint(workspace)
    key, _fingerprint = _private_key(tmp_path / "private.pem")
    request = _request(
        workspace,
        checkpoint_digest=checkpoint_digest,
        tokenizer_digest=tokenizer_digest,
        output="rejected-evidence",
    )
    original = oci.compose_side_worker_command
    observed = []
    payload = (
        "raise SystemExit(7)"
        if mode == "nonzero"
        else "from pathlib import Path; p=Path('/invarlock/output-root/side'); p.mkdir(); (p/'report.json').write_text('{}')"
    )

    def compose(**kwargs):
        command = original(**kwargs)
        command[-3:] = ["-c", payload]
        observed.append(command)
        return command

    monkeypatch.setattr(oci, "compose_side_worker_command", compose)
    executor = oci.OciRuntimeExecutor(runtime)
    with pytest.raises(
        EvaluationTransactionError, match="worker exited|six-file bundle"
    ) as failure:
        evaluate_request_file(
            request,
            signing_key_path=key,
            runtime_executor=executor,
            runtime_image_digests=oci.preflight_oci_launch(runtime),
        )
    assert len(observed) == 2
    assert not (workspace / "rejected-evidence").exists()
    assert not list(workspace.rglob("pack.manifest.json"))
    _record(
        "transaction-" + mode,
        runtime,
        observed,
        None,
        rejected=str(failure.value),
        evidence_published=False,
    )
