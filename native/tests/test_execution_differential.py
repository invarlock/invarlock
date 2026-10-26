"""Exercise real pipe boundaries against the retained Python transport."""

from __future__ import annotations

import subprocess
import sys
import threading
import time

import pytest

from invarlock._bounded_subprocess import communicate_bounded

native = pytest.importorskip("invarlock_native")


def _python(script: str, payload: bytes, stdout_limit: int, stderr_limit: int):
    process = subprocess.Popen(
        [sys.executable, "-c", script],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    def terminate(child):
        child.kill()
        child.wait(timeout=5)

    return communicate_bounded(
        process,
        input_bytes=payload,
        timeout_seconds=5,
        stdout_limit=stdout_limit,
        stderr_limit=stderr_limit,
        error_type=ValueError,
        timeout_label="process",
        output_label="process",
        pipes_message="process pipes missing",
        terminate=terminate,
    )


def _native(script: str, payload: bytes, stdout_limit: int, stderr_limit: int):
    return native.run_process(
        sys.executable, ["-c", script], payload, 5.0, stdout_limit, stderr_limit
    )


@pytest.mark.parametrize(
    ("script", "payload"),
    [
        (
            "import sys; sys.stdout.buffer.write(sys.stdin.buffer.read())",
            b"x" * 300_000,
        ),
        (
            "import os,sys,threading; "
            "t=threading.Thread(target=lambda: os.write(2,b'e'*300000)); t.start(); "
            "sys.stdout.buffer.write(sys.stdin.buffer.read()); t.join()",
            b"x" * 500_000,
        ),
        (
            "import time,sys; time.sleep(.05); "
            "sys.stdout.buffer.write(sys.stdin.buffer.read())",
            b"x" * 500_000,
        ),
        ("import sys; sys.exit(7)", b"x" * 300_000),
        ("import os; os.write(1,b'abc'); os.write(2,b'def')", b""),
    ],
)
def test_real_pipe_output_matches_python(script, payload):
    assert _native(script, payload, 1_000_000, 1_000_000) == _python(
        script, payload, 1_000_000, 1_000_000
    )


@pytest.mark.parametrize(("stream", "fd"), [("stdout", 1), ("stderr", 2)])
def test_overflow_classification_matches_python(stream, fd):
    script = f"import os; os.write({fd},b'x'*100000)"
    for runner in (_python, _native):
        with pytest.raises(ValueError, match=f"process {stream} limit exceeded"):
            runner(script, b"", 4096, 4096)


def test_native_cancel_releases_gil_and_stops_process():
    token = native.CancellationToken()
    timer = threading.Timer(0.05, token.cancel)
    timer.start()
    started = time.monotonic()
    try:
        with pytest.raises(ValueError, match="process cancelled"):
            native.run_process(
                sys.executable,
                ["-c", "import time; time.sleep(20)"],
                b"",
                5.0,
                4096,
                4096,
                cancellation=token,
            )
    finally:
        timer.join()
    assert token.cancelled
    assert time.monotonic() - started < 2


def test_native_deadline_and_exact_output_boundary():
    with pytest.raises(ValueError, match="process timed out"):
        native.run_process(
            sys.executable,
            ["-c", "import time; time.sleep(20)"],
            b"",
            0.05,
            4096,
            4096,
        )
    assert _native("import os; os.write(1,b'abc')", b"", 3, 0) == (0, b"abc", b"")


@pytest.fixture
def inspect_checkpoint_inputs(tmp_path):
    import json
    from pathlib import Path

    from invarlock.judge_measurements import runner
    from invarlock.judge_measurements.collector import CollectionOptions

    fixture_root = (
        Path(__file__).resolve().parents[2] / "tests/judge_measurements/fixtures"
    )
    plan = json.loads((fixture_root / "plan.json").read_bytes())
    exported = json.loads((fixture_root / "export.json").read_bytes())
    options = CollectionOptions.from_mapping(exported["collection"])
    directory = tmp_path.resolve() / "checkpoint"
    runner_options = runner.RunnerOptions(directory, "correctness", 30)
    return runner, plan, options, runner_options, exported


def _checkpoint_templates(inputs):
    from invarlock.judge_measurements.contracts import canonical_payload

    runner, plan, options, runner_options, _ = inputs
    return (
        canonical_payload(runner._header(plan, options, runner_options)),
        canonical_payload(runner._empty_export(plan, options, runner_options)),
    )


def _source_checkpoint_export(inputs):
    from invarlock.filesystem.paths import pinned_directory
    from invarlock.judge_measurements.contracts import canonical_payload

    runner, plan, options, runner_options, _ = inputs
    with pinned_directory(runner_options.checkpoint_directory) as descriptor:
        return canonical_payload(
            runner._checkpoint_export(plan, options, runner_options, descriptor)
        )


def test_inspect_checkpoint_admit_complete_resume_matches_source(
    inspect_checkpoint_inputs,
):
    import json

    from invarlock.judge_measurements.contracts import canonical_payload

    inputs = inspect_checkpoint_inputs
    runner, _, options, settings, fixture_export = inputs
    header, empty = _checkpoint_templates(inputs)
    checkpoint = native.InspectCheckpoint(
        str(settings.checkpoint_directory), header, empty
    )
    assert checkpoint.export_json() == _source_checkpoint_export(inputs)
    slot = json.loads(empty)["samples"][0]
    event = next(
        sample for sample in fixture_export["samples"] if sample["id"] == slot["id"]
    )["events"][0]
    admission = runner._admission_event(
        trial_id=slot["id"],
        attempt=1,
        request=event["call"]["request"],
        options=options,
    )
    checkpoint.admit(slot["id"], 1, canonical_payload(admission))
    assert checkpoint.pending_admissions() == [(slot["id"], 1)]
    assert checkpoint.export_json() == _source_checkpoint_export(inputs)
    checkpoint.close()
    checkpoint = native.InspectCheckpoint(
        str(settings.checkpoint_directory), header, empty
    )
    assert checkpoint.pending_admissions() == [(slot["id"], 1)]
    with pytest.raises(ValueError, match="prior admission remains ambiguous"):
        checkpoint.admit(slot["id"], 2, canonical_payload(admission))
    checkpoint.complete(slot["id"], 1, canonical_payload(event))
    assert checkpoint.pending_admissions() == []
    assert checkpoint.export_json() == _source_checkpoint_export(inputs)
    checkpoint.close()
    assert (settings.checkpoint_directory / "collection.json").read_bytes() == header


@pytest.mark.parametrize(
    "defect", ["orphan_result", "missing_attempt", "duplicate_admission"]
)
def test_inspect_checkpoint_replay_rejects_same_invalid_sequences(
    inspect_checkpoint_inputs, defect
):
    import json

    from invarlock.judge_measurements.contracts import canonical_payload

    inputs = inspect_checkpoint_inputs
    _, _, _, settings, _ = inputs
    header, empty = _checkpoint_templates(inputs)
    checkpoint = native.InspectCheckpoint(
        str(settings.checkpoint_directory), header, empty
    )
    checkpoint.close()
    trial = json.loads(empty)["samples"][0]["id"]
    record = {
        "trial_id": trial,
        "attempt": 2 if defect == "missing_attempt" else 1,
        "event": {},
    }
    prefix = "result-" if defect == "orphan_result" else "admission-"
    (settings.checkpoint_directory / f"{prefix}{trial}-01.json").write_bytes(
        canonical_payload(record)
    )
    if defect == "duplicate_admission":
        (settings.checkpoint_directory / "admission-copy.json").write_bytes(
            canonical_payload(record)
        )
    with pytest.raises(ValueError):
        native.InspectCheckpoint(str(settings.checkpoint_directory), header, empty)
    with pytest.raises(ValueError):
        _source_checkpoint_export(inputs)


def test_inspect_checkpoint_rejects_lock_contention_and_source_replacement(
    inspect_checkpoint_inputs,
):
    import json

    _, _, _, settings, _ = inspect_checkpoint_inputs
    header, empty = _checkpoint_templates(inspect_checkpoint_inputs)
    checkpoint = native.InspectCheckpoint(
        str(settings.checkpoint_directory), header, empty
    )
    with pytest.raises(ValueError, match="another collector"):
        native.InspectCheckpoint(str(settings.checkpoint_directory), header, empty)
    trial = json.loads(empty)["samples"][0]["id"]
    checkpoint.admit(trial, 1, b"{}")
    path = settings.checkpoint_directory / f"admission-{trial}-01.json"
    raw = path.read_bytes()
    path.unlink()
    path.write_bytes(raw)
    with pytest.raises(ValueError, match="identity changed"):
        checkpoint.export_json()
    checkpoint.close()


@pytest.fixture
def native_command_path():
    import os

    path = os.environ.get("INVARLOCK_NATIVE_COMMAND")
    if not path:
        pytest.skip("set INVARLOCK_NATIVE_COMMAND to exercise the built control CLI")
    return path


def _control_request(binary, path, request):
    import json

    path.write_text(json.dumps(request))
    return subprocess.run(
        [binary, "request", str(path)], capture_output=True, timeout=10
    )


@pytest.mark.parametrize("spent,retained", [(0, 0), (101, 100), (0, 384 * 1024 * 1024)])
def test_collection_capacity_matches_source_integer_bounds(
    native_command_path, tmp_path, spent, retained
):
    import json
    from types import SimpleNamespace

    from invarlock.judge_measurements.collector import _LiveCheckpoint

    reservation = {
        "max_calls": 100,
        "max_input_tokens": 1000,
        "max_output_tokens": 500,
        "max_cost_microusd": 10000,
        "input_tokens_per_call": 10,
        "output_tokens_per_call": 5,
        "cost_microusd_per_call": 100,
    }
    sizes = {
        "header_bytes": 500,
        "samples": 100,
        "record_bytes": 100,
        "retained_bytes": retained,
        "spent_calls": spent,
    }
    source = object.__new__(_LiveCheckpoint)
    source.options = SimpleNamespace(**reservation)
    source.plan = {"judge": {"config": {"max_output_tokens": 5}}}
    source.samples = [None] * sizes["samples"]
    for key, value in sizes.items():
        if key != "samples":
            setattr(source, key, value)
    completed = _control_request(
        native_command_path,
        tmp_path.resolve() / "capacity.json",
        {
            "command": "collection-capacity",
            "reservations": reservation,
            "retained": sizes,
        },
    )
    assert completed.returncode == 0, completed.stderr
    observed = json.loads(completed.stdout)
    expected = source.capacity_details()
    assert {key: observed[key] for key in expected} == expected
    retained_bound, source_bound = source._storage_bounds()
    assert (observed["retained_bound"], observed["source_bound"]) == (
        retained_bound,
        source_bound,
    )


def test_control_transaction_persists_once_and_preserves_unknown_usage(
    native_command_path, tmp_path
):
    import json

    root = tmp_path.resolve()
    request = {
        "command": "process-transaction",
        "directory": str(root / "transaction"),
        "identity": {"plan": "fixture-v1"},
        "id": "one",
        "maximum": {
            "calls": 2,
            "input_tokens": 100,
            "output_tokens": 100,
            "cost_microusd": 100,
            "retained_bytes": 4096,
        },
        "reservation": {
            "calls": 1,
            "input_tokens": 10,
            "output_tokens": 10,
            "cost_microusd": 10,
            "retained_bytes": 2048,
        },
        "process": {
            "program": sys.executable,
            "args": ["-c", "print('retained')"],
            "input_hex": "",
            "timeout_ms": 1000,
            "stdout_limit": 128,
            "stderr_limit": 128,
        },
    }
    result = _control_request(native_command_path, root / "request.json", request)
    assert result.returncode == 0, result.stderr
    value = json.loads(result.stdout)
    assert value["result"]["process"]["stdout_hex"] == b"retained\n".hex()
    assert value["checkpoint"]["transactions"][0]["observed"] is None
    retry = _control_request(native_command_path, root / "request.json", request)
    assert retry.returncode == 2
    assert b"already admitted" in retry.stderr
    assert len(list((root / "transaction").glob("*.event"))) == 2


@pytest.mark.parametrize(
    "kind", ["config", "tag", "repository", "tag_config", "wrong_digest"]
)
def test_oci_manifest_resolution_matches_source(kind):
    import json

    from invarlock import evaluation_oci as source

    config = "sha256:" + "a" * 64
    manifest = "sha256:" + "b" * 64
    repository = "registry:5000/org/model"
    inspection = source._LocalImageInspection(config, (f"{repository}@{manifest}",))
    image, digest = {
        "config": (config, config),
        "tag": (f"{repository}:latest", manifest),
        "repository": (f"{repository}@{manifest}", manifest),
        "tag_config": (f"{repository}:latest", config),
        "wrong_digest": (f"{repository}@{manifest}", config),
    }[kind]
    raw = json.dumps(
        [{"Id": config, "RepoDigests": list(inspection.repo_digests)}]
    ).encode()
    if kind in {"tag_config", "wrong_digest"}:
        with pytest.raises(ValueError) as expected:
            source._resolve_inspected_image(image, digest, inspection, allow_tag=True)
        with pytest.raises(ValueError) as observed:
            native.resolve_oci_inspection(image, digest, raw)
        assert str(observed.value) == str(expected.value)
    else:
        assert native.resolve_oci_inspection(
            image, digest, raw
        ) == source._resolve_inspected_image(image, digest, inspection, allow_tag=True)


@pytest.fixture
def coordinator_inputs():
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[2] / "tests/judge_measurements/fixtures"
    return {
        name: json.loads((root / f"{name}.json").read_bytes())
        for name in ("plan", "export", "baseline_run", "subject_run")
    }


def test_native_coordinator_owns_durable_admission_and_retained_validation(
    tmp_path, coordinator_inputs
):
    import json

    from invarlock.judge_measurements.collector import CollectionOptions, import_export
    from invarlock.judge_measurements.contracts import canonical_payload

    data = coordinator_inputs
    options = data["export"]["collection"]
    root = tmp_path.resolve() / "coordinator"
    coordinator = native.CollectionCoordinator(
        str(root),
        canonical_payload(data["plan"]),
        canonical_payload(options),
        canonical_payload(data["baseline_run"]),
        canonical_payload(data["subject_run"]),
        "correctness",
        30,
    )
    samples = {sample["id"]: sample for sample in data["export"]["samples"]}
    calls = json.loads(coordinator.begin_batch_json())
    assert len(calls) == 2
    for call in calls:
        assert (root / f"admission-{call['trial_id']}-01.json").is_file()
        with pytest.raises(ValueError, match="started call"):
            coordinator.complete(
                call["trial_id"],
                canonical_payload(samples[call["trial_id"]]["events"][0]),
            )
        coordinator.wait_start(call["trial_id"])
        event = canonical_payload(samples[call["trial_id"]]["events"][0])
        coordinator.complete(call["trial_id"], event)
        coordinator.complete(call["trial_id"], event)
    assert json.loads(coordinator.begin_batch_json()) == []
    assert json.loads(coordinator.status_json())["stop_reason"] == "complete"
    actual = json.loads(coordinator.measurements_json())
    expected = import_export(
        canonical_payload(data["export"]),
        plan=data["plan"],
        options=CollectionOptions.from_mapping(options),
        baseline_run=data["baseline_run"],
        subject_run=data["subject_run"],
    )
    assert actual == expected
    coordinator.close()
    assert len(list(root.glob("admission-*.json"))) == 2
    assert len(list(root.glob("result-*.json"))) == 2


@pytest.mark.parametrize("configured_route", [False, True])
@pytest.mark.parametrize("interrupt", [False, True])
def test_explicit_rust_adapter_routes_callbacks_without_python_coordination(
    tmp_path, monkeypatch, coordinator_inputs, configured_route, interrupt
):
    import asyncio
    import copy
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from invarlock.judge_measurements import configured, rust_runner
    from invarlock.judge_measurements import runner as source
    from invarlock.judge_measurements.collector import CollectionOptions
    from invarlock.judge_measurements.contracts import canonical_payload

    data = coordinator_inputs
    options = CollectionOptions.from_mapping(data["export"]["collection"])
    directory = tmp_path.resolve() / "adapter"
    settings = source.RunnerOptions(directory, "correctness", 1 if interrupt else 30)
    source_import_module = rust_runner.importlib.import_module
    source_version = rust_runner.importlib.metadata.version
    monkeypatch.setattr(
        source, "_require_qualified_live_provider_model", lambda _: None
    )
    monkeypatch.setattr(source, "_require_provider_retries_disabled", lambda _: None)
    monkeypatch.setattr(source, "_require_clean_model_configuration", lambda _: None)
    monkeypatch.setattr(
        rust_runner.importlib.metadata,
        "version",
        lambda name: "0.3.263" if name == "inspect-ai" else source_version(name),
    )
    monkeypatch.setattr(
        rust_runner.importlib,
        "import_module",
        lambda name: (
            SimpleNamespace(
                GenerateConfig=lambda **kw: SimpleNamespace(**kw),
                get_model=lambda *a, **kw: model,
            )
            if name == "inspect_ai.model"
            else source_import_module(name)
        ),
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("Python coordinator path was used")

    monkeypatch.setattr(source, "_LiveCheckpoint", forbidden)
    monkeypatch.setattr(source, "_checkpoint_export", forbidden)
    monkeypatch.setattr(source, "write_file_no_replace", forbidden)
    by_id = {sample["id"]: sample for sample in data["export"]["samples"]}
    invoked = []

    async def provider_callback(
        model, *, request, config, options, pacer, check_directory
    ):
        await pacer.wait()
        check_directory()
        sample = by_id[pacer.trial_id]
        assert canonical_payload(request) == canonical_payload(
            sample["events"][0]["call"]["request"]
        )
        assert (directory / f"admission-{sample['id']}-01.json").exists()
        invoked.append(sample["id"])
        if interrupt:
            await asyncio.sleep(30)
        return copy.deepcopy(sample["events"][0])

    monkeypatch.setattr(source, "_call_one", provider_callback)

    class Model:
        def __str__(self):
            return options.grader

    model = Model()
    client = SimpleNamespace(base_url="https://api.openai.com/v1", close=AsyncMock())
    model.api = SimpleNamespace(
        client=client, service_tier="default", aclose=client.close
    )
    monkeypatch.setattr(configured, "network_policy_allows", lambda: True)
    monkeypatch.setattr(configured, "validate_collection_environment", lambda *a: None)
    monkeypatch.setattr(
        configured, "_require_qualified_live_provider_model", lambda *a: None
    )
    monkeypatch.setattr(
        configured,
        "_provider_config",
        lambda *a: ("openai", configured._PROVIDERS["openai"]),
    )
    monkeypatch.setattr(configured, "collect", forbidden)

    async def run(**kwargs):
        if configured_route:
            kwargs.pop("model")
            return await configured.collect_configured(
                **kwargs,
                execution_backend="rust",
                environment={"OPENAI_API_KEY": "offline-fixture"},
            )
        return await rust_runner.collect(**kwargs)

    status = {}
    result = asyncio.run(
        run(
            plan=data["plan"],
            options=options,
            runner=settings,
            model=model,
            baseline_run=data["baseline_run"],
            subject_run=data["subject_run"],
            status=status,
        )
    )
    assert 1 <= len(invoked) <= 2
    assert len(list(directory.glob("admission-*.json"))) == 2
    assert result["completeness"]["completed_trials"] == (0 if interrupt else 2)
    assert status == {"stop_reason": "deadline" if interrupt else "complete"}
    if configured_route:
        client.close.assert_awaited_once()
    invoked.clear()
    resumed = asyncio.run(
        run(
            plan=data["plan"],
            options=options,
            runner=settings,
            model=model,
            baseline_run=data["baseline_run"],
            subject_run=data["subject_run"],
            status=status,
        )
    )
    assert invoked == []
    assert resumed == result


def test_native_coordinator_rejects_invalid_callback_without_retaining_result(
    tmp_path, coordinator_inputs
):
    import copy
    import json

    from invarlock.judge_measurements.contracts import canonical_payload

    data = coordinator_inputs
    root = tmp_path.resolve() / "invalid-callback"
    coordinator = native.CollectionCoordinator(
        str(root),
        canonical_payload(data["plan"]),
        canonical_payload(data["export"]["collection"]),
        canonical_payload(data["baseline_run"]),
        canonical_payload(data["subject_run"]),
        "correctness",
        30,
    )
    calls = json.loads(coordinator.begin_batch_json())
    call = calls[0]
    coordinator.wait_start(call["trial_id"])
    sample = next(s for s in data["export"]["samples"] if s["id"] == call["trial_id"])
    event = copy.deepcopy(sample["events"][0])
    event["call"]["request"] = {"altered": True}
    with pytest.raises(ValueError):
        coordinator.complete(call["trial_id"], canonical_payload(event))
    coordinator.close()
    assert len(list(root.glob("admission-*.json"))) == 2
    assert list(root.glob("result-*.json")) == []


@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize("failure", [False, True])
def test_native_oci_pair_owns_launch_cancellation_and_resource_checks(
    tmp_path, monkeypatch, parallel, failure
):
    import json
    from pathlib import Path

    from invarlock import evaluation_oci

    root = tmp_path.resolve()
    engine = root / "engine"
    log = root / "calls.jsonl"
    engine.write_text(
        "#!"
        + sys.executable
        + "\n"
        + f"""
import sys,json,pathlib,time,os
args=sys.argv[1:]
log=pathlib.Path({str(log)!r})
if args[:2]==['image','inspect']:
 print(json.dumps([{{'Id':args[2],'RepoDigests':[]}}]));sys.exit(0)
with log.open('a') as f:f.write(json.dumps(args)+'\\n')
if args[0] in ['stop','kill']:sys.exit(0)
cid=pathlib.Path(args[args.index('--cidfile')+1]);cid.write_text('a'*64)
role=cid.parent.name
if {failure!r}:
 if role=='baseline':sys.exit(7)
 time.sleep(30)
os.write(1,b'o'*200000);os.write(2,b'e'*200000)
"""
    )
    engine.chmod(0o755)
    artifact = root / "artifact"
    artifact.write_bytes(b"model")
    artifact.chmod(0o444)
    workers = {}
    for role in ("baseline", "subject"):
        job = root / role
        job.mkdir()
        (job / "job.json").write_text("{}")
        (job / "schedule.json").write_text("{}")
        output = root / (role + "-output")
        output.mkdir()
        workers[role] = {
            "engine": str(engine),
            "engine_kind": "docker",
            "image": "sha256:" + "a" * 64,
            "device": "cpu",
            "cpus": "1",
            "memory_mib": 512,
            "user": "65532:65532",
            "artifact": str(artifact),
            "support": {},
            "job": str(job),
            "output": str(output),
            "cidfile": str(job / "container.cid"),
            "tmpfs_gib": 4,
            "entrypoint": "python",
            "arguments": ["-m", "invarlock.evaluation_side_worker"],
            "timeout_seconds": 5,
        }
    monkeypatch.setattr(
        evaluation_oci,
        "run_side_worker",
        lambda *a, **kw: (_ for _ in ()).throw(
            AssertionError("Python launch path used")
        ),
    )
    started = time.monotonic()
    result = evaluation_oci._run_native_workers(workers, parallel=parallel)
    assert time.monotonic() - started < 5
    if failure:
        assert result["baseline"].returncode == 7
        assert result["subject"].returncode == 125
        commands = [json.loads(line) for line in log.read_text().splitlines()]
        if not parallel:
            assert len([cmd for cmd in commands if cmd[0] == "run"]) == 1
    else:
        for outcome in result.values():
            assert outcome.returncode == 0
            assert len(outcome.stdout) == 65536
            assert len(outcome.stderr) == 65536
    # Pins are checked before dispatch; a substituted artifact never launches.
    artifact.unlink()
    artifact.symlink_to(Path(sys.executable))
    for role in workers:
        (root / role / "container.cid").unlink(missing_ok=True)
    with pytest.raises(evaluation_oci.OciEvaluationError):
        evaluation_oci._run_native_workers(workers, parallel=parallel)


def test_native_provider_transport_keeps_owned_file_after_caller_fd_reuse(tmp_path):
    import os

    root = tmp_path.resolve()
    original = root / "original"
    original.write_bytes(b"owned input")
    replacement = root / "replacement"
    replacement.write_bytes(b"substituted input")
    started = root / "started"
    fd = os.open(original, os.O_RDONLY)
    outcome = []

    def invoke():
        outcome.append(
            native.run_process(
                sys.executable,
                [
                    "-c",
                    "import sys,time,pathlib;pathlib.Path(sys.argv[2]).touch();time.sleep(.3);print(open(sys.argv[1]).read(),end='')",
                    f"/dev/fd/{fd}",
                    str(started),
                ],
                b"",
                5,
                100,
                100,
                pass_fds=[fd],
                cwd=str(root),
                environment={"PATH": "/usr/bin:/bin"},
            )
        )

    task = threading.Thread(target=invoke)
    task.start()
    deadline = time.monotonic() + 3
    while not started.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert started.exists()
    os.close(fd)
    new_fd = os.open(replacement, os.O_RDONLY)
    try:
        task.join(5)
        assert not task.is_alive()
        assert outcome == [(0, b"owned input", b"")]
    finally:
        os.close(new_fd)


@pytest.mark.parametrize("provider", ["llama", "tensorrt"])
def test_provider_process_routes_native_and_preserves_public_errors(
    tmp_path, monkeypatch, provider
):
    from types import SimpleNamespace

    from invarlock.runtime_providers import _tensorrt_llm_execution as trt
    from invarlock.runtime_providers import llama_cpp_session as llama

    root = tmp_path.resolve()
    monkeypatch.setenv("INVARLOCK_EXECUTION_BACKEND", "rust")

    def forbidden(*a, **kw):
        raise AssertionError("Python Popen transport used")

    monkeypatch.setattr(subprocess, "Popen", forbidden)
    checks = []
    directory = SimpleNamespace(
        path=root,
        recheck=lambda: checks.append("directory"),
        environment=lambda: {"PATH": "/usr/bin:/bin"},
    )
    if provider == "llama":
        executable = SimpleNamespace(
            fd_path=sys.executable,
            descriptor=-1,
            recheck=lambda: checks.append("executable"),
        )
        # The process entrypoint uses a normal executable in this host fixture;
        # descriptor transfer and descriptor-path rewriting are covered above.
        import os

        fd = os.open(sys.executable, os.O_RDONLY)
        executable.descriptor = fd

        def call(args, limit=100):
            return llama._run_bounded_process(
                executable=executable,
                arguments=args,
                input_bytes=b"",
                pass_fds=(),
                run_directory=directory,
                timeout_seconds=1,
                stdout_limit=limit,
                stderr_limit=100,
            )

        error = llama.LlamaCppExecutionError
    else:
        fd = None
        runner = root / "runner.py"
        runner.write_text(
            "import sys,time\nif sys.argv[1]=='slow':time.sleep(10)\nelse: print('x'*200,end='')\n"
        )

        def call(args, limit=100):
            return trt._run_bounded_process(
                runner=SimpleNamespace(path=runner),
                vendor_python=SimpleNamespace(path=sys.executable),
                execution_boundary=SimpleNamespace(
                    recheck=lambda *a: checks.append("boundary")
                ),
                arguments=args,
                input_bytes=b"",
                run_directory=directory,
                timeout_seconds=1,
                stdout_limit=limit,
                stderr_limit=100,
            )

        error = trt.TensorRTLLMExecutionError
    try:
        args = ["-c", "print('x'*200,end='')"] if provider == "llama" else ["burst"]
        with pytest.raises(error, match="stdout limit exceeded"):
            call(args)
        args = ["-c", "import time;time.sleep(10)"] if provider == "llama" else ["slow"]
        with pytest.raises(error, match="timed out"):
            call(args)
        assert len(checks) >= 8
    finally:
        if fd is not None:
            os.close(fd)


def test_oci_executor_explicit_backend_uses_native_pair_actual_path(
    tmp_path, monkeypatch
):
    import importlib.util
    import json
    from dataclasses import replace
    from pathlib import Path
    from types import SimpleNamespace

    from invarlock import evaluation_oci
    from invarlock.evidence_pack_contract import (
        schedule_bytes as canonical_schedule_bytes,
    )

    fixture_path = (
        Path(__file__).resolve().parents[2] / "tests/runtime/test_evaluation_oci.py"
    )
    spec = importlib.util.spec_from_file_location(
        "oci_native_route_fixtures", fixture_path
    )
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    root = tmp_path.resolve()
    request = fixture._request(root)
    (root / "models/subject/tokenizer.json").write_text("{}")
    engine = root / "engine"
    log = root / "launches.jsonl"
    engine.write_text(
        "#!"
        + sys.executable
        + "\n"
        + f"""
import sys,json,pathlib
args=sys.argv[1:]
if args[:2]==['image','inspect']:
 print(json.dumps([{{'Id':'sha256:'+'c'*64,'RepoDigests':[args[2]]}}]));sys.exit(0)
with open({str(log)!r},'a') as f:f.write(json.dumps(args)+'\\n')
pathlib.Path(args[args.index('--cidfile')+1]).write_text('a'*64)
"""
    )
    engine.chmod(0o755)

    def forbidden(*a, **kw):
        raise AssertionError("Python OCI process or command composition used")

    monkeypatch.setattr(evaluation_oci, "run_side_worker", forbidden)
    monkeypatch.setattr(evaluation_oci, "compose_side_worker_command", forbidden)
    monkeypatch.setattr(
        evaluation_oci,
        "load_runtime_side_evidence",
        lambda path: fixture._evidence(path.parent.name.removesuffix("-output")),
    )
    monkeypatch.setattr(
        evaluation_oci,
        "decode_runtime_provider_receipt",
        lambda raw: SimpleNamespace(
            outer_image_digest=fixture._BASELINE_DIGEST
            if raw == b"baseline"
            else fixture._SUBJECT_DIGEST
        ),
    )
    launch = replace(fixture._launch(), engine_path=str(engine))
    result = evaluation_oci.OciRuntimeExecutor(
        launch,
        environment={
            "INVARLOCK_TENSORRT_LLM_RESOURCE_ROOT": str(root),
            "INVARLOCK_TENSORRT_LLM_TOKENIZER_CONTRACT": "models/subject/tokenizer.json",
        },
        execution_backend="rust",
    ).execute(
        request,
        registry=fixture._Registry(),
        schedule_bytes=canonical_schedule_bytes(fixture._schedule()),
        policy_digest="sha256:" + "c" * 64,
    )
    assert result.baseline.run_report == b"baseline-report"
    assert result.subject.run_report == b"subject-report"
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(calls) == 2
    assert all("INVARLOCK_EXECUTION_BACKEND=rust" in args for args in calls)
    assert all("--pull=never" in args and "--read-only" in args for args in calls)


def test_compatible_collector_native_owns_checkpoint_resume_and_source_validation(
    tmp_path, monkeypatch
):
    from invarlock.judge_measurements import openai_compatible as source
    from tests.judge_measurements import test_openai_compatible_collector as fixture

    root = tmp_path.resolve()
    plan, collection, baseline, subject = fixture._inputs(root)
    calls = fixture._install_transport(monkeypatch)
    arguments = {
        "configuration": collection,
        "baseline_run": baseline,
        "subject_run": subject,
    }
    expected = source.collect_openai_compatible(
        plan,
        **arguments,
        options=source.OpenAICompatibleJudgeOptions(
            checkpoint_directory=root / "python"
        ),
    )
    calls.clear()

    def forbidden(*a, **kw):
        raise AssertionError("Python checkpoint/control path used")

    for name in (
        "_checkpoint_sources",
        "_admit",
        "_retain",
        "_validate_cached",
        "_assemble",
    ):
        monkeypatch.setattr(source, name, forbidden)
    options = source.OpenAICompatibleJudgeOptions(checkpoint_directory=root / "rust")
    actual = source.collect_openai_compatible(
        plan, **arguments, options=options, execution_backend="rust"
    )
    assert actual == expected
    assert len(calls) == plan["schedule"]["expected_trials"]
    calls.clear()
    assert (
        source.collect_openai_compatible(
            plan, **arguments, options=options, execution_backend="rust"
        )
        == expected
    )
    assert calls == []
    assert {p.name: p.read_bytes() for p in (root / "python").iterdir()} == {
        p.name: p.read_bytes() for p in (root / "rust").iterdir()
    }
    # An admitted but interrupted call must not become a fresh HTTP retry.
    (root / "rust" / "result-000002.json").unlink()
    with pytest.raises(source.OpenAICompatibleJudgeError, match="cannot be retried"):
        source.collect_openai_compatible(
            plan, **arguments, options=options, execution_backend="rust"
        )
    assert calls == []


def test_runtime_collector_native_owns_checkpoint_and_provider_callbacks(
    tmp_path, monkeypatch
):
    from invarlock.judge_measurements import runtime_provider as source
    from tests.judge_measurements import test_runtime_provider_judge as fixture

    root = tmp_path.resolve()
    for name, value in (
        ("strict_container_boundary_present", True),
        ("network_allowed", False),
        ("remote_code_allowed", False),
        ("third_party_plugins_allowed", False),
    ):
        monkeypatch.setattr(source, name, lambda value=value: value)
    args = {
        "spec": fixture._spec(),
        "resources": fixture._resources(root),
        "baseline_run": fixture._load("baseline_run.json"),
        "subject_run": fixture._load("subject_run.json"),
    }
    plan = fixture._plan()
    expected = source.collect_runtime_provider(
        plan,
        provider=fixture._Provider(),
        **args,
        options=source.RuntimeProviderJudgeOptions(
            checkpoint_directory=root / "python"
        ),
    )

    def forbidden(*a, **kw):
        raise AssertionError("Python checkpoint/control path used")

    for name in (
        "_prepare_checkpoint",
        "_admit_checkpoint_batch",
        "_retain_checkpoint_batch",
        "_validate_cached_sources",
        "_retained_measurement_size",
    ):
        monkeypatch.setattr(source, name, forbidden)
    options = source.RuntimeProviderJudgeOptions(checkpoint_directory=root / "rust")
    provider = fixture._Provider()
    actual = source.collect_runtime_provider(
        plan, provider=provider, **args, options=options, execution_backend="rust"
    )
    assert actual == expected
    assert provider.open_calls == 1
    resumed = fixture._Provider()
    assert (
        source.collect_runtime_provider(
            plan, provider=resumed, **args, options=options, execution_backend="rust"
        )
        == expected
    )
    assert resumed.open_calls == 0
    assert {p.name: p.read_bytes() for p in (root / "python").iterdir()} == {
        p.name: p.read_bytes() for p in (root / "rust").iterdir()
    }
    (root / "rust" / "result-000002.json").unlink()
    with pytest.raises(source.RuntimeProviderJudgeError, match="cannot be retried"):
        source.collect_runtime_provider(
            plan, provider=resumed, **args, options=options, execution_backend="rust"
        )
    assert resumed.open_calls == 0


def test_native_judge_workflow_publication_uses_owned_key_transaction(
    tmp_path, monkeypatch
):
    import json

    from invarlock.judge_measurements import evidence, native_publication
    from invarlock.judge_measurements.acceptance import verify_judge_evidence
    from tests.judge_measurements.test_evidence_acceptance import KEY, _publish

    root = tmp_path.resolve()
    expected, policy = _publish(root)
    values = {
        name: json.loads((expected.path / filename).read_bytes())
        for name, filename in {
            "plan": "plan.json",
            "measurements": "measurements.json",
            "baseline_run": "baseline_run.json",
            "subject_run": "subject_run.json",
            "analysis_policy": "analysis_policy.json",
        }.items()
    }
    monkeypatch.setattr(
        evidence,
        "publish_judge_evidence",
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("Python publisher used")),
    )
    actual = native_publication.publish(
        root / "native", **values, signing_key=KEY, signer_identity="example-signer"
    )
    assert actual.analysis_result.to_dict() == expected.analysis_result.to_dict()
    assert {p.name: p.read_bytes() for p in expected.path.iterdir()} == {
        p.name: p.read_bytes() for p in actual.path.iterdir()
    }
    assert verify_judge_evidence(actual.path, recipient_policy_path=policy).accepted


def test_public_native_judge_evaluation_routes_prepare_capture_and_publication(
    tmp_path, monkeypatch
):
    import json

    from invarlock.evaluation_transaction import evaluate_request_file
    from invarlock.judge_measurements import evidence, native_capture, native_workflow
    from tests.core.test_native_judge_transaction import (
        _completed_measurements,
        _import,
    )

    root = tmp_path.resolve()
    request, key, _ = _import(root)
    monkeypatch.setattr(
        native_workflow,
        "collection_preflight",
        lambda *a, **kw: {"credential_available": True, "network_calls": 0},
    )
    seen = []

    def collect(**kwargs):
        assert kwargs.pop("execution_backend") == "rust"
        seen.append(True)
        return _completed_measurements(**kwargs)

    monkeypatch.setattr(native_workflow, "collect_frozen", collect)

    def forbidden(*a, **kw):
        raise AssertionError("Python judge prepare/replay/publication path used")

    monkeypatch.setattr(native_workflow, "pinned_directory", forbidden)
    monkeypatch.setattr(native_workflow, "write_file_no_replace", forbidden)
    monkeypatch.setattr(native_workflow, "prepare_native_judge", forbidden)
    monkeypatch.setattr(native_workflow, "finalize_native_plan", forbidden)
    monkeypatch.setattr(native_capture, "create_native_capture", forbidden)
    monkeypatch.setattr(native_capture, "validate_native_capture", forbidden)
    monkeypatch.setattr(evidence, "publish_judge_evidence", forbidden)
    result = evaluate_request_file(request, signing_key_path=key, backend="rust")
    assert result.payload["ok"] is True
    assert seen == [True]
    capture = json.loads((root / "judge-workspace/native_capture.json").read_bytes())
    assert capture["format"] == "invarlock/native-judge-capture-v1"
    assert (root / "artifacts/evidence/envelope.json").is_file()


def test_native_workspace_owns_lock_identity_and_interrupted_capture(tmp_path):
    import hashlib
    import os

    from invarlock.judge_measurements.contracts import canonical_payload

    root = tmp_path.resolve() / "workspace"
    workspace = native.Workspace(str(root))
    value = {"request": "same-frozen-inputs"}
    workspace.retain_identity("identity.json", canonical_payload(value))
    assert (root / "identity.json").read_bytes() == canonical_payload(
        {
            "format": "invarlock/judge-workspace-identity-v1",
            "sha256": hashlib.sha256(canonical_payload(value)).hexdigest(),
        }
    )
    with pytest.raises(ValueError, match="another evaluation"):
        native.Workspace(str(root))
    workspace.write("capture-admission.json", b'{"capture_admitted":true}')
    workspace.close()
    resumed = native.Workspace(str(root))
    with pytest.raises(FileNotFoundError):
        resumed.read("native_capture.json", 1024)
    with pytest.raises(FileExistsError):
        resumed.write("capture-admission.json", b'{"capture_admitted":true}')
    resumed.retain_identity("identity.json", canonical_payload(value))
    (root / "identity.json").rename(root / "previous.json")
    (root / "identity.json").write_bytes(b"substitution")
    with pytest.raises(ValueError):
        resumed.verify()
    resumed.close()
    os.link(root / ".evaluation.lock", root / "hardlinked-lock")
    with pytest.raises(ValueError, match="lock is unsafe"):
        native.Workspace(str(root))


def test_public_captured_judge_cli_uses_native_preparation_and_publication(
    tmp_path, monkeypatch
):
    import json

    from typer.testing import CliRunner

    from invarlock.cli.app import app
    from invarlock.judge_measurements import (
        captured_workflow,
        evidence,
        native_workflow,
    )
    from tests.core.test_native_judge_transaction import _completed_measurements
    from tests.judge_measurements.test_captured_scorer_workflow import _request

    root = tmp_path.resolve()
    request, *_ = _request(root)
    monkeypatch.setattr(
        native_workflow,
        "collection_preflight",
        lambda *a, **kw: {"credential_available": True, "network_calls": 0},
    )

    def collect(**kwargs):
        assert kwargs.pop("execution_backend") == "rust"
        return _completed_measurements(**kwargs)

    monkeypatch.setattr(native_workflow, "collect_frozen", collect)

    def forbidden(*a, **kw):
        raise AssertionError("Python captured prepare/publication/workspace path used")

    monkeypatch.setattr(captured_workflow, "prepare_evaluator_judge", forbidden)
    monkeypatch.setattr(captured_workflow, "publish_judge_evidence", forbidden)
    monkeypatch.setattr(evidence, "publish_judge_evidence", forbidden)
    monkeypatch.setattr(native_workflow, "pinned_directory", forbidden)
    monkeypatch.setattr(native_workflow, "write_file_no_replace", forbidden)
    result = CliRunner().invoke(
        app, ["evaluate", str(request), "--backend", "rust", "--unsigned", "--json"]
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["ok"] is True
    assert (root / "evidence/envelope.json").is_file()


def test_native_oci_output_cleanup_never_uses_python_process_transport(
    tmp_path, monkeypatch
):
    from dataclasses import replace

    from invarlock import evaluation_oci
    from tests.runtime.test_evaluation_oci import _launch

    root = tmp_path.resolve() / "worker"
    root.mkdir()
    (root / "baseline-output").mkdir()
    engine = tmp_path.resolve() / "cleanup-engine"
    engine.write_text(
        "#!"
        + sys.executable
        + "\nimport pathlib,sys\nargs=sys.argv[1:]\nif '--cidfile' in args:pathlib.Path(args[args.index('--cidfile')+1]).write_text('a'*64)\n"
    )
    engine.chmod(0o755)
    observed = []

    def remove(path):
        observed.append(path)
        if len(observed) == 1:
            raise OSError("simulate worker-owned output")

    monkeypatch.setattr(evaluation_oci, "_remove_collectible_output", remove)
    monkeypatch.setattr(
        evaluation_oci,
        "run_side_worker",
        lambda *a, **kw: (_ for _ in ()).throw(
            AssertionError("Python cleanup process used")
        ),
    )
    evaluation_oci._cleanup_worker_directory(
        root,
        replace(_launch(), engine_path=str(engine)),
        {"baseline": "hf_transformers", "subject": "tensorrt_llm"},
        execution_backend="rust",
    )
    assert len(observed) == 2
    assert not root.exists()


@pytest.mark.parametrize("backend", ["python", "rust"])
@pytest.mark.parametrize("fail", [False, True])
def test_public_preflight_scopes_host_provider_backend(
    tmp_path, monkeypatch, backend, fail
):
    from invarlock.evaluation_transaction import (
        EvaluationPreflightError,
        preflight_evaluation_request,
    )
    from invarlock.runtime_providers.hf_transformers import HFTransformersProvider
    from invarlock.runtime_providers.native_execution import (
        execution_backend,
        using_execution_backend,
    )
    from tests.cli.test_evaluation_preflight import _materialize_run_request

    root = tmp_path.resolve()
    request, key = _materialize_run_request(root)
    authenticate = HFTransformersProvider.authenticate_artifact
    observed = []

    def hook(provider, spec, path):
        observed.append(execution_backend())
        if fail:
            raise ValueError("controlled host authentication interruption")
        return authenticate(provider, spec, path)

    monkeypatch.setattr(HFTransformersProvider, "authenticate_artifact", hook)
    runtime = "sha256:" + "a" * 64
    kwargs = {"backend": "rust"} if backend == "rust" else {}
    previous = execution_backend()
    enclosing = "python" if backend == "rust" else "rust"
    with using_execution_backend(enclosing):
        if fail:
            with pytest.raises(
                EvaluationPreflightError,
                match="controlled host authentication interruption",
            ):
                preflight_evaluation_request(
                    request,
                    signing_key_path=key,
                    runtime_image_digests={"baseline": runtime, "subject": runtime},
                    **kwargs,
                )
        else:
            result = preflight_evaluation_request(
                request,
                signing_key_path=key,
                runtime_image_digests={"baseline": runtime, "subject": runtime},
                **kwargs,
            )
            assert result.execution_mode == "run"
        assert execution_backend() == enclosing
    assert execution_backend() == previous
    assert observed == [backend] * (1 if fail else 2)
    assert not (root / "artifacts").exists()
