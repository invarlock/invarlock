"""Campaign execution enforces resource isolation and preserves interruption state."""

import contextlib
import copy
import hashlib
import json
import signal
from pathlib import Path
from types import SimpleNamespace

import pytest

from examples.qualification import campaign_execution as execution
from examples.qualification import campaign_scheduling as scheduling


def test_container_command_limits_real_devices_cpu_memory_and_network(tmp_path):
    job = {
        "id": "sentinel",
        "timeout_seconds": 60,
        "container": {
            "image": "sha256:" + "a" * 64,
            "command": ["python", "-c", "print(1)"],
            "environment": {},
            "mounts": [],
        },
    }
    allocation = {
        "gpu_ids": ["GPU-test"],
        "cpu_ids": [2, 4],
        "memory_mib": 4096,
        "io_slots": 0,
        "exclusive": False,
    }
    command = execution.container_command(
        job, allocation, tmp_path, "campaign-test", 999, 987
    )
    assert "--gpus=all" not in command
    assert command[command.index("--gpus") + 1] == '"device=GPU-test"'
    assert "--cpuset-cpus=2,4" in command
    assert "--memory=4096m" in command and "--memory-swap=4096m" in command
    assert "--network=none" in command and "--read-only" in command
    assert "--user=999:987" in command
    allocation["gpu_ids"] = []
    command = execution.container_command(
        job, allocation, tmp_path, "campaign-test", 999, 987
    )
    assert "--gpus" not in command and "--runtime=runc" in command
    assert "NVIDIA_VISIBLE_DEVICES=void" in command


GPUS = [
    "GPU-00000000-0000-0000-0000-000000000001",
    "GPU-00000000-0000-0000-0000-000000000002",
]


def _job(name, dependencies=(), *, gpus=1, seconds=1, timeout=5):
    return {
        "id": name,
        "depends_on": list(dependencies),
        "resources": {
            "gpus": gpus,
            "cpus": 2,
            "memory_mib": 400,
            "io_slots": 1,
            "exclusive": False,
        },
        "timeout_seconds": timeout,
        "estimate_seconds": seconds,
        "workload": {"key": "fixed-model-profile", "units": 10, "fixed_seconds": 0.1},
        "container": {
            "image": "sha256:" + "a" * 64,
            "command": ["python", "/check.py"],
            "mounts": [],
            "environment": {},
        },
        "outputs": ["result.json"],
    }


def _manifest(jobs=None):
    return {
        "format": scheduling.FORMAT,
        "id": "test-run",
        "host": {
            "gpu_ids": GPUS.copy(),
            "cpu_ids": [0, 1, 2, 3],
            "memory_mib": 1000,
            "io_slots": 2,
            "deadline_epoch": 1100,
            "reserve_seconds": 10,
            "hourly_cost_usd": 3.6,
        },
        "jobs": jobs
        if jobs is not None
        else [_job("a"), _job("b"), _job("c", ["a", "b"])],
    }


class Clock:
    def __init__(self):
        self.now = 0.0
        self.offset = 1000.0

    def wall(self):
        return self.offset + self.now

    def mono(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class FakeEngine:
    def __init__(self, clock, *, durations=None, codes=None, payloads=None):
        self.clock = clock
        self.durations = durations or {}
        self.codes = codes or {}
        self.payloads = payloads or {}
        self.active = {}
        self.starts = []
        self.removals = []
        self.max_active = 0
        self.remove_error = False
        self.start_error = False
        self.on_poll = None
        self.probes = 0

    def probe(self, host):
        self.probes += 1
        return {"gpu_ids": host["gpu_ids"]}

    def start(self, job, allocation, output, name):
        intent = execution.read_json(output.parents[2] / "state.json")["jobs"][
            job["id"]
        ]
        assert intent["container_name"] == name
        assert intent["allocation"] == allocation
        self.active[name] = (job, output, self.clock.now)
        self.starts.append((job["id"], self.clock.now, copy.deepcopy(allocation)))
        self.max_active = max(self.max_active, len(self.active))
        if self.start_error:
            raise RuntimeError("simulated interruption after launch")
        return {"Id": name, "fake_engine": True}

    def poll(self, name):
        job, output, started = self.active[name]
        if self.on_poll:
            self.on_poll(job, output)
        if self.clock.now - started < self.durations.get(job["id"], 0.5):
            return None
        code = self.codes.get(job["id"], 0)
        if code == 0:
            for path in job["outputs"]:
                target = output / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(
                    json.dumps(self.payloads.get(job["id"], {"result": job["id"]}))
                )
        return code

    def logs(self, name, output):
        (output / "container.log").write_text("retained log\n")

    def remove(self, name):
        self.removals.append(name)
        if self.remove_error:
            raise ValueError("cleanup uncertain")
        self.active.pop(name, None)


def _run(monkeypatch, manifest, output, *, fake=None, clock=None, **kwargs):
    monkeypatch.setattr(execution.os, "getuid", lambda: 999)
    monkeypatch.setattr(execution.os, "getgid", lambda: 987)
    clock = clock or Clock()
    fake = fake or FakeEngine(clock)
    state = execution.run(
        manifest,
        output,
        engine=fake,
        clock=clock.wall,
        monotonic=clock.mono,
        sleep=clock.sleep,
        **kwargs,
    )
    return state, fake, clock


def test_fake_engine_dispatches_dependencies_concurrently_and_resumes(
    monkeypatch, tmp_path
):
    manifest = _manifest()
    state, engine, clock = _run(monkeypatch, manifest, tmp_path / "run")
    assert engine.max_active == 2
    assert engine.starts[0][1] == engine.starts[1][1] == 0
    assert engine.starts[2][1] == 0.5
    assert all(r["status"] == "succeeded" for r in state["jobs"].values())
    assert not engine.active
    assert state["observations"] == []
    remaining = execution.read_json(tmp_path / "run" / "remaining.json")
    assert remaining["remaining_seconds"] == 0
    assert remaining["complete_campaign_possible"] is True
    resumed, second, _ = _run(monkeypatch, manifest, tmp_path / "run", clock=clock)
    assert not second.starts
    assert resumed["jobs"] == state["jobs"]


def test_resource_contention_serializes_and_dependency_failures_block(
    monkeypatch, tmp_path
):
    manifest = _manifest()
    manifest["jobs"][0]["resources"]["gpus"] = 2
    clock = Clock()
    engine = FakeEngine(clock, codes={"a": 9})
    state, _, _ = _run(
        monkeypatch, manifest, tmp_path / "run", clock=clock, fake=engine
    )
    assert state["jobs"]["a"]["status"] == "failed"
    assert state["jobs"]["c"]["status"] == "blocked_dependency"
    assert [key for key, _, _ in engine.starts] == ["a", "b"]
    assert engine.starts[1][1] >= 0.5
    assert not execution.read_json(tmp_path / "run" / "remaining.json")[
        "complete_campaign_possible"
    ]


def test_deadline_reserves_entire_downstream_timeout_chain(monkeypatch, tmp_path):
    m = _manifest([_job("a", timeout=2), _job("b", ["a"], timeout=2)])
    m["host"]["deadline_epoch"] = 1013  # Only 3 seconds before the reserve.
    state, engine, _ = _run(monkeypatch, m, tmp_path / "run")
    assert not engine.starts
    assert state["jobs"]["a"]["status"] == "deferred_budget"
    assert state["jobs"]["b"]["status"] == "blocked_dependency"


def test_timeout_and_cancellation_clean_all_owned_containers(monkeypatch, tmp_path):
    m = _manifest([_job("a", timeout=1), _job("b", timeout=1)])
    clock = Clock()
    engine = FakeEngine(clock, durations={"a": 10, "b": 10})
    state, _, _ = _run(monkeypatch, m, tmp_path / "timeout", clock=clock, fake=engine)
    assert all(r["status"] == "cancelled" for r in state["jobs"].values())
    assert len(engine.removals) == 2 and not engine.active
    clock = Clock()
    engine = FakeEngine(clock, durations={"a": 10, "b": 10})
    state, _, _ = _run(
        monkeypatch,
        _manifest(),
        tmp_path / "cancel",
        clock=clock,
        fake=engine,
        cancelled=lambda: clock.now >= 0.25,
    )
    assert all(state["jobs"][key]["status"] == "cancelled" for key in ("a", "b"))
    assert state["jobs"]["c"]["status"] == "deferred_budget"
    assert not engine.active


def test_hard_deadline_uses_monotonic_time_even_with_frozen_wall(monkeypatch, tmp_path):
    m = _manifest([_job("a", timeout=2)])
    m["host"]["deadline_epoch"] = 1012
    clock = Clock()
    clock.wall = lambda: 1000
    clock.sleep = lambda seconds: setattr(clock, "now", clock.now + 3)
    engine = FakeEngine(clock, durations={"a": 10})
    state, _, _ = _run(monkeypatch, m, tmp_path / "run", clock=clock, fake=engine)
    assert state["jobs"]["a"]["status"] == "cancelled"
    assert not engine.active


def test_uncertain_cleanup_prevents_more_admission_and_retains_recovery_target(
    monkeypatch, tmp_path
):
    m = _manifest([_job("a", gpus=2), _job("b", gpus=2)])
    clock = Clock()
    engine = FakeEngine(clock)
    engine.remove_error = True
    with pytest.raises(ValueError, match="cleanup uncertain"):
        _run(monkeypatch, m, tmp_path / "run", fake=engine, clock=clock)
    assert len(engine.starts) == 1
    state = execution.read_json(tmp_path / "run" / "state.json")
    assert state["jobs"]["a"]["status"] == "cleanup_uncertain"
    target = state["jobs"]["a"]["container_name"]
    assert target in engine.removals
    recovery = FakeEngine(clock)
    state, _, _ = _run(monkeypatch, m, tmp_path / "run", fake=recovery, clock=clock)
    assert recovery.removals[0] == target
    assert state["jobs"]["a"]["status"] == "cancelled"
    assert all(key != "a" for key, _, _ in recovery.starts)


def test_exception_after_launch_cleans_and_records_interruption(monkeypatch, tmp_path):
    clock = Clock()
    engine = FakeEngine(clock)
    engine.start_error = True
    with pytest.raises(RuntimeError, match="interruption"):
        _run(monkeypatch, _manifest(), tmp_path / "run", clock=clock, fake=engine)
    state = execution.read_json(tmp_path / "run" / "state.json")
    assert state["jobs"]["a"]["status"] == "cancelled"
    assert not engine.active
    assert (
        execution.read_json(tmp_path / "run" / "remaining.json")["active_job_ids"] == []
    )


@pytest.mark.parametrize(
    "tamper", ["output", "mounted_file", "manifest", "backward_clock"]
)
def test_completed_resume_rejects_changed_identities_before_any_launch(
    monkeypatch, tmp_path, tamper
):
    m = _manifest([_job("a")])
    source = tmp_path / "source.json"
    source.write_text("original")
    m["jobs"][0]["container"]["mounts"] = [
        {"source": str(source), "target": "/source.json"}
    ]
    _, _, clock = _run(monkeypatch, m, tmp_path / "run")
    if tamper == "output":
        (tmp_path / "run/jobs/a/output/result.json").write_text("changed")
    elif tamper == "mounted_file":
        source.write_text("changed")
    elif tamper == "manifest":
        m["jobs"][0]["estimate_seconds"] += 1
    else:
        clock.offset -= 10
    engine = FakeEngine(clock)
    with pytest.raises(ValueError, match="changed|clock"):
        _run(monkeypatch, m, tmp_path / "run", fake=engine, clock=clock)
    assert not engine.starts


def test_root_operator_rejected_without_output_side_effect(monkeypatch, tmp_path):
    monkeypatch.setattr(execution.os, "getuid", lambda: 0)
    with pytest.raises(ValueError, match="non-root"):
        execution.run(_manifest(), tmp_path / "absent")
    assert not (tmp_path / "absent").exists()


def test_mounted_input_changed_during_execution_cannot_succeed(monkeypatch, tmp_path):
    source = tmp_path / "input"
    source.write_text("original")
    m = _manifest([_job("a")])
    m["jobs"][0]["container"]["mounts"] = [{"source": str(source), "target": "/input"}]
    clock = Clock()
    engine = FakeEngine(clock)
    engine.on_poll = lambda job, output: source.write_text("changed")
    with pytest.raises(ValueError, match="source file changed"):
        _run(monkeypatch, m, tmp_path / "run", clock=clock, fake=engine)
    assert (
        execution.read_json(tmp_path / "run/state.json")["jobs"]["a"]["status"]
        != "succeeded"
    )
    assert not engine.active


def _sentinel(job, **extra):
    return {
        "format": "invarlock/campaign-sentinel-v1",
        "workload_key": job["workload"]["key"],
        "units": job["workload"]["units"],
        "fixed_seconds": 0.1,
        "complete": True,
        "semantic_ready": True,
        **extra,
    }


def test_successful_sentinel_calibrates_only_after_validated_outputs(
    monkeypatch, tmp_path
):
    a = _job("a")
    a["outputs"] = ["sentinel.json"]
    clock = Clock()
    engine = FakeEngine(clock, payloads={"a": _sentinel(a)})
    state, _, _ = _run(
        monkeypatch, _manifest([a]), tmp_path / "run", fake=engine, clock=clock
    )
    obs = state["observations"][0]
    assert obs["job_id"] == "a" and obs["elapsed_seconds"] == 0.5
    assert obs["manifest_digest"] == scheduling.manifest_digest(_manifest([a]))
    assert (
        execution.read_json(tmp_path / "run/forecast.json")["jobs"][0]["source"]
        == "same_workload_sentinels"
    )


def test_postvalidator_attributes_timing_and_resources_to_direct_completed_dependency(
    monkeypatch, tmp_path
):
    target = _job("work", gpus=2)
    validator = _job("validate", ["work"], gpus=0)
    validator["outputs"] = ["sentinel.json"]
    validator["workload"]["key"] = "validator-only"
    clock = Clock()
    engine = FakeEngine(
        clock,
        durations={"work": 1, "validate": 0.25},
        payloads={"validate": _sentinel(target, observed_job_id="work")},
    )
    state, _, _ = _run(
        monkeypatch,
        _manifest([target, validator]),
        tmp_path / "run",
        clock=clock,
        fake=engine,
    )
    obs = state["observations"][0]
    assert obs["job_id"] == "work" and obs["elapsed_seconds"] == 1
    assert obs["allocation"]["gpu_ids"] == GPUS
    assert state["jobs"]["validate"]["allocation"]["gpu_ids"] == []


@pytest.mark.parametrize(
    "change",
    [
        {"units": True},
        {"complete": 1},
        {"semantic_ready": 1},
        {"fixed_seconds": True},
        {"fixed_seconds": 1},
        {"format": "bad"},
        {"extra": 1},
        {"observed_job_id": "other"},
    ],
)
def test_malformed_sentinels_are_not_evidence(tmp_path, change):
    job = _job("a")
    record = {
        "status": "succeeded",
        "elapsed_seconds": 0.5,
        "output_hashes": {"sentinel.json": "sha256:" + "0" * 64},
        "allocation": scheduling.allocate(job, _manifest()["host"], []),
    }
    execution.write_json(tmp_path / "sentinel.json", _sentinel(job, **change))
    with pytest.raises(ValueError):
        execution.observation(job, record, tmp_path)


def test_missing_unhashed_or_incomplete_sentinel_never_silently_means_ready(tmp_path):
    job = _job("a")
    record = {
        "status": "succeeded",
        "elapsed_seconds": 0.5,
        "output_hashes": {},
        "allocation": scheduling.allocate(job, _manifest()["host"], []),
    }
    assert execution.observation(job, record, tmp_path) is None
    execution.write_json(
        tmp_path / "sentinel.json", _sentinel(job, semantic_ready=False)
    )
    assert execution.observation(job, record, tmp_path) is None
    record["output_hashes"] = execution.output_hashes(tmp_path, ["sentinel.json"])
    assert execution.observation(job, record, tmp_path)["semantic_ready"] is False


def test_remaining_forecast_preserves_active_leases_and_excludes_completed_work():
    m = _manifest(
        [_job("a", seconds=10), _job("b", seconds=10), _job("c", ["a", "b"], seconds=5)]
    )
    a = scheduling.allocate(m["jobs"][0], m["host"], [])
    state = {
        "observations": [],
        "jobs": {"a": {"status": "running"}, "b": {"status": "succeeded"}},
    }
    result = execution.remaining_forecast(
        m, state, {"a": {"started": 0, "allocation": a}}, 4
    )
    assert result["remaining_seconds"] == 11
    assert result["active_job_ids"] == ["a"]
    assert [row["id"] for row in result["schedule"]] == ["c"]
    assert result["schedule"][0]["start_seconds"] == 6
    state["jobs"]["a"]["status"] = "failed"
    result = execution.remaining_forecast(m, state, {}, 4)
    assert result["remaining_seconds"] == 0
    assert result["excluded_job_ids"] == ["a", "c"]
    assert result["complete_campaign_possible"] is False


def test_json_metadata_is_closed_to_duplicates_nonfinite_and_oversize(tmp_path):
    path = tmp_path / "metadata.json"
    for value in ('{"a":1,"a":2}', '{"a":NaN}'):
        path.write_text(value)
        with pytest.raises(ValueError):
            execution.read_json(path)
    with path.open("wb") as stream:
        stream.truncate(16 * 1024 * 1024 + 1)
    with pytest.raises(ValueError, match="16 MiB"):
        execution.read_json(path)
    victim = tmp_path / "victim"
    victim.write_text("unchanged")
    path.unlink()
    path.symlink_to(victim)
    execution.write_json(path, {"ok": True})
    assert victim.read_text() == "unchanged" and not path.is_symlink()
    assert execution.read_json(path) == {"ok": True}
    assert path.stat().st_mode & 0o777 == 0o600


def test_host_lock_rejects_collision_symlink_and_wrong_owner(monkeypatch, tmp_path):
    path = tmp_path / "lock"
    with execution.host_lock(path):
        with pytest.raises(BlockingIOError):
            with execution.host_lock(path):
                pass
    with execution.host_lock(path):
        pass
    link = tmp_path / "link"
    link.symlink_to(path)
    with pytest.raises(OSError):
        with execution.host_lock(link):
            pass
    monkeypatch.setattr(execution.os, "getuid", lambda: path.stat().st_uid + 1)
    with pytest.raises(ValueError, match="operator-owned"):
        with execution.host_lock(path):
            pass


def test_mount_and_output_hashes_bind_actual_files_and_refuse_escapes(tmp_path):
    source = tmp_path / "input"
    source.write_bytes(b"actual bytes")
    expected = "sha256:" + hashlib.sha256(source.read_bytes()).hexdigest()
    job = _job("a")
    job["container"]["mounts"] = [
        {"source": str(source), "target": "/input", "sha256": expected}
    ]
    assert execution.mount_hashes(job) == {str(source.resolve()): expected}
    allocation = scheduling.allocate(job, _manifest()["host"], [])
    command = execution.container_command(job, allocation, tmp_path, "name", 999, 987)
    assert f"type=bind,src={source.resolve()},dst=/input,readonly" in command
    source.write_text("changed")
    with pytest.raises(ValueError, match="digest mismatch"):
        execution.mount_hashes(job)
    job["container"]["mounts"][0]["source"] = str(tmp_path)
    with pytest.raises(ValueError, match="directory identity"):
        execution.mount_hashes(job)
    del job["container"]["mounts"][0]["sha256"]
    assert execution.mount_hashes(job) == {}
    output = tmp_path / "output"
    output.mkdir()
    (output / "ok").write_text("ok")
    assert execution.output_hashes(output, ["ok"]) == {
        "ok": "sha256:" + hashlib.sha256(b"ok").hexdigest()
    }
    (output / "link").symlink_to(source)
    for name in ("link", "../input", "."):
        with pytest.raises(ValueError):
            execution.output_hashes(output, [name])
    reserved = tmp_path / "reserved"
    reserved.symlink_to("/proc")
    with pytest.raises((ValueError, FileNotFoundError)):
        execution.resolved_mount(reserved)
    socket = tmp_path / "fake.sock"
    socket.write_text("not a mount")
    with pytest.raises(ValueError):
        execution.resolved_mount(socket)


def test_logs_replace_symlink_without_touching_victim_and_are_bounded(
    monkeypatch, tmp_path
):
    victim = tmp_path / "victim"
    victim.write_text("untouched")
    (tmp_path / "container.log").symlink_to(victim)
    engine = execution.DockerEngine()
    monkeypatch.setattr(
        engine,
        "command",
        lambda *args, **kwargs: SimpleNamespace(
            stdout="x" * (10 * 1024 * 1024), stderr="ERROR"
        ),
    )
    engine.logs("owned", tmp_path)
    result = tmp_path / "container.log"
    assert victim.read_text() == "untouched" and not result.is_symlink()
    assert result.stat().st_size == 10 * 1024 * 1024
    assert result.read_text().endswith("ERROR")


def _inspected(job, allocation, name):
    return {
        "Id": "a" * 64,
        "Image": job["container"]["image"],
        "HostConfig": {
            "DeviceRequests": [{"DeviceIDs": allocation["gpu_ids"]}]
            if allocation["gpu_ids"]
            else None,
            "CpusetCpus": ",".join(map(str, allocation["cpu_ids"])),
            "NanoCpus": len(allocation["cpu_ids"]) * 1000000000,
            "Memory": allocation["memory_mib"] * 1024 * 1024,
            "MemorySwap": allocation["memory_mib"] * 1024 * 1024,
            "NetworkMode": "none",
            "ReadonlyRootfs": True,
            "CapDrop": ["ALL"],
            "SecurityOpt": ["no-new-privileges:true"],
            "Privileged": False,
            "Devices": [],
            "Runtime": "runc",
            "PidsLimit": 2048,
            "ShmSize": min(8192, allocation["memory_mib"] // 4) * 1024 * 1024,
            "Tmpfs": {
                "/tmp": f"rw,nosuid,nodev,exec,size={min(8192, allocation['memory_mib'] // 4)}m"
            },
        },
        "Config": {
            "User": "999:987",
            "Labels": {"invarlock.campaign": name},
            "Entrypoint": job["container"]["command"][:1],
            "Cmd": job["container"]["command"][1:],
            "Env": [
                "HOME=/tmp",
                "HF_HUB_OFFLINE=1",
                "TRANSFORMERS_OFFLINE=1",
                f"OMP_NUM_THREADS={len(allocation['cpu_ids'])}",
                "CUDA_VISIBLE_DEVICES="
                + ",".join(map(str, range(len(allocation["gpu_ids"]))))
                if allocation["gpu_ids"]
                else "NVIDIA_VISIBLE_DEVICES=void",
            ],
        },
        "Mounts": [
            {"Destination": "/output", "RW": True},
            {"Destination": "/input", "RW": False},
        ],
        "State": {"Running": False, "ExitCode": 0},
    }


def _engine_with_inspect(monkeypatch, job, allocation, name="owned"):
    monkeypatch.setattr(execution.os, "getuid", lambda: 999)
    monkeypatch.setattr(execution.os, "getgid", lambda: 987)
    engine = execution.DockerEngine()
    observed = _inspected(job, allocation, name)
    commands = []

    def command(args, **kwargs):
        commands.append(args)
        if args[:2] == ["docker", "run"]:
            if observed["Mounts"] == [
                {"Destination": "/output", "RW": True},
                {"Destination": "/input", "RW": False},
            ]:
                observed["Mounts"] = [
                    {
                        "Type": "bind",
                        "Source": item.split("src=", 1)[1].split(",", 1)[0],
                        "Destination": item.split("dst=", 1)[1].split(",", 1)[0],
                        "RW": not item.endswith(",readonly"),
                    }
                    for item in args
                    if item.startswith("type=bind,")
                ]
        if args[:3] == ["docker", "image", "inspect"]:
            value = job["container"]["image"]
        elif args[:2] == ["docker", "inspect"]:
            value = json.dumps([observed])
        else:
            value = ""
        return SimpleNamespace(stdout=value, stderr="", returncode=0)

    monkeypatch.setattr(engine, "command", command)
    return engine, observed, commands


@pytest.mark.parametrize("gpu_count", [0, 1, 2])
def test_docker_start_verifies_pinned_image_and_observed_constraints(
    monkeypatch, tmp_path, gpu_count
):
    job = _job("a", gpus=gpu_count)
    allocation = scheduling.allocate(job, _manifest()["host"], [])
    engine, observed, commands = _engine_with_inspect(monkeypatch, job, allocation)
    assert engine.start(job, allocation, tmp_path, "owned") == observed
    launch = next(command for command in commands if command[:2] == ["docker", "run"])
    assert "--pull=never" in launch and "--network=none" in launch
    assert engine.poll("owned") == 0
    observed["State"]["Running"] = True
    assert engine.poll("owned") is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("Image", "sha256:" + "f" * 64),
        ("HostConfig.DeviceRequests", [{"DeviceIDs": []}]),
        ("HostConfig.CpusetCpus", "3"),
        ("HostConfig.NanoCpus", 1),
        ("HostConfig.Memory", 1),
        ("HostConfig.MemorySwap", 1),
        ("HostConfig.NetworkMode", "host"),
        ("HostConfig.ReadonlyRootfs", False),
        ("Config.User", "0:0"),
        ("HostConfig.CapDrop", []),
        ("HostConfig.SecurityOpt", []),
        ("HostConfig.Privileged", True),
        ("HostConfig.Devices", [{"PathOnHost": "/dev/sda"}]),
        ("Mounts", [{"Destination": "/input", "RW": True}]),
        ("Config.Labels", {}),
        ("Mounts", []),
        ("Config.Env", []),
        ("Config.Entrypoint", ["sh"]),
        ("Config.Cmd", ["other"]),
        ("HostConfig.PidsLimit", 0),
        ("HostConfig.ShmSize", 0),
        ("HostConfig.Tmpfs", {}),
    ],
)
def test_changed_docker_constraints_fail_closed(monkeypatch, tmp_path, field, value):
    job = _job("a")
    allocation = scheduling.allocate(job, _manifest()["host"], [])
    engine, observed, _ = _engine_with_inspect(monkeypatch, job, allocation)
    parts = field.split(".")
    target = observed
    for part in parts[:-1]:
        target = target[part]
    target[parts[-1]] = value
    with pytest.raises(ValueError, match="isolation"):
        engine.start(job, allocation, tmp_path, "owned")


def test_wrong_image_busy_gpu_inspect_ambiguity_and_cleanup_error(
    monkeypatch, tmp_path
):
    job = _job("a")
    allocation = scheduling.allocate(job, _manifest()["host"], [])
    engine = execution.DockerEngine()
    monkeypatch.setattr(
        engine,
        "command",
        lambda *args, **kwargs: SimpleNamespace(
            stdout="wrong", stderr="", returncode=0
        ),
    )
    with pytest.raises(ValueError, match="image identity"):
        engine.start(job, allocation, tmp_path, "owned")
    monkeypatch.setattr(
        engine,
        "command",
        lambda args, **kwargs: SimpleNamespace(
            stdout=job["container"]["image"] if args[0] == "docker" else GPUS[0],
            stderr="",
            returncode=0,
        ),
    )
    with pytest.raises(ValueError, match="unrelated"):
        engine.start(job, allocation, tmp_path, "owned")
    monkeypatch.setattr(
        engine,
        "command",
        lambda *args, **kwargs: SimpleNamespace(stdout="[]", stderr="", returncode=0),
    )
    with pytest.raises(ValueError, match="ambiguous"):
        engine.inspect("owned")
    monkeypatch.setattr(
        engine,
        "command",
        lambda *args, **kwargs: SimpleNamespace(
            stdout="", stderr="No such container", returncode=1
        ),
    )
    engine.remove("already-gone")
    monkeypatch.setattr(
        engine,
        "command",
        lambda *args, **kwargs: SimpleNamespace(
            stdout="", stderr="permission denied", returncode=1
        ),
    )
    with pytest.raises(ValueError, match="cleanup uncertain"):
        engine.remove("owned")


def test_docker_command_timeout_is_bounded_by_absolute_deadline(monkeypatch):
    engine = execution.DockerEngine()
    engine.deadline_monotonic = 12
    monkeypatch.setattr(execution.time, "monotonic", lambda: 10)
    calls = []
    monkeypatch.setattr(
        execution.subprocess, "run", lambda args, **kwargs: calls.append((args, kwargs))
    )
    engine.command(["docker", "version"])
    assert calls[0][1]["timeout"] == 2 and calls[0][1]["check"] is True
    engine.deadline_monotonic = 10
    with pytest.raises(ValueError, match="deadline"):
        engine.command(["docker", "version"])


@pytest.mark.parametrize(
    "issue",
    [
        None,
        "cpu_reserve",
        "memory_reserve",
        "missing_gpu",
        "heterogeneous",
        "busy",
        "unreconciled",
    ],
)
def test_actual_host_probe_checks_reserves_devices_and_foreign_work(monkeypatch, issue):
    host = _manifest()["host"]
    engine = execution.DockerEngine()
    monkeypatch.setattr(
        execution.os, "sched_getaffinity", lambda pid: set(range(12)), raising=False
    )
    original = Path.read_text
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda path, *args, **kwargs: (
            "MemTotal: 104857600 kB\n"
            if str(path) == "/proc/meminfo"
            else original(path, *args, **kwargs)
        ),
    )
    rows = [f"{gpu}, H100, 80000, 0{index}:00" for index, gpu in enumerate(GPUS)]
    if issue == "cpu_reserve":
        host["cpu_ids"] = list(range(5))
    elif issue == "memory_reserve":
        host["memory_mib"] = 100000
    elif issue == "missing_gpu":
        rows = rows[:1]
    elif issue == "heterogeneous":
        rows[1] = rows[1].replace("H100", "H200")

    def command(args, **kwargs):
        if args[0] == "docker":
            return SimpleNamespace(
                stdout="owned-but-unreconciled" if issue == "unreconciled" else "",
                stderr="",
                returncode=0,
            )
        return SimpleNamespace(
            stdout=(
                "\n".join(rows)
                if args[1].startswith("--query-gpu=")
                else GPUS[0]
                if issue == "busy"
                else ""
            ),
            stderr="",
            returncode=0,
        )

    monkeypatch.setattr(engine, "command", command)
    if issue is None:
        assert engine.probe(host)["memory_mib"] == 102400
    else:
        with pytest.raises(ValueError):
            engine.probe(host)


def test_overdue_remaining_estimate_is_unknown_with_timeout_reservation():
    m = _manifest([_job("a", seconds=1, timeout=10), _job("b", ["a"], seconds=2)])
    allocation = scheduling.allocate(m["jobs"][0], m["host"], [])
    result = execution.remaining_forecast(
        m,
        {"observations": [], "jobs": {"a": {"status": "running"}}},
        {"a": {"started": 0, "allocation": allocation}},
        3,
    )
    assert result["expected_remaining_seconds"] is None
    assert result["remaining_seconds"] == 9
    assert result["overdue_job_ids"] == ["a"]


def test_cli_forecast_run_exit_codes_and_signal_restoration(monkeypatch, tmp_path):
    manifest = tmp_path / "manifest.json"
    execution.write_json(manifest, _manifest())
    result = tmp_path / "forecast.json"
    assert (
        execution.main(
            ["forecast", "--manifest", str(manifest), "--output", str(result)]
        )
        == 0
    )
    assert execution.read_json(result)["scope"] == "full_manifest"
    handlers = {}
    restored = []

    def signal_handler(signum, handler):
        previous = handlers.get(signum, signal.SIG_DFL)
        handlers[signum] = handler
        restored.append((signum, handler))
        return previous

    monkeypatch.setattr(execution.signal, "signal", signal_handler)

    @contextlib.contextmanager
    def lock(path):
        assert path == Path("/tmp/invarlock-campaign-host.lock")
        yield

    monkeypatch.setattr(execution, "host_lock", lock)

    def run(manifest, output, *, cancelled):
        handlers[signal.SIGTERM](signal.SIGTERM, None)
        assert cancelled()
        return {"jobs": {"a": {"status": "cancelled"}}}

    monkeypatch.setattr(execution, "run", run)
    assert (
        execution.main(
            ["run", "--manifest", str(manifest), "--output", str(tmp_path / "run")]
        )
        == 1
    )
    assert all(
        handlers[sig] == signal.SIG_DFL for sig in (signal.SIGTERM, signal.SIGINT)
    )
    monkeypatch.setattr(
        execution,
        "run",
        lambda *args, **kwargs: {"jobs": {"a": {"status": "succeeded"}}},
    )
    assert (
        execution.main(
            ["run", "--manifest", str(manifest), "--output", str(tmp_path / "run")]
        )
        == 0
    )
    manifest.write_text('{"duplicate":1,"duplicate":2}')
    with pytest.raises(SystemExit) as error:
        execution.main(
            ["forecast", "--manifest", str(manifest), "--output", str(result)]
        )
    assert error.value.code == 2


def test_nonempty_output_without_state_refuses_implicit_restart(monkeypatch, tmp_path):
    (tmp_path / "prior-result").write_text("retained")
    with pytest.raises(ValueError, match="nonempty"):
        _run(monkeypatch, _manifest(), tmp_path)
    assert (tmp_path / "prior-result").read_text() == "retained"


def test_resume_failed_jobs_does_not_retry_them_implicitly(monkeypatch, tmp_path):
    clock = Clock()
    first = FakeEngine(clock, codes={"a": 2})
    m = _manifest([_job("a")])
    output = tmp_path / "run"
    state, _, _ = _run(monkeypatch, m, output, fake=first, clock=clock)
    assert state["jobs"]["a"]["status"] == "failed"
    state, engine, _ = _run(monkeypatch, m, output, clock=clock)
    assert not engine.starts and state["jobs"]["a"]["status"] == "failed"


@pytest.mark.parametrize("value", [None, [], "text", 1])
def test_nonobject_sentinel_fails_with_controlled_validation_error(tmp_path, value):
    job = _job("a")
    execution.write_json(tmp_path / "sentinel.json", value)
    record = {
        "status": "succeeded",
        "elapsed_seconds": 0.5,
        "output_hashes": {"sentinel.json": "unused"},
        "allocation": scheduling.allocate(job, _manifest()["host"], []),
    }
    with pytest.raises(ValueError):
        execution.observation(job, record, tmp_path)


def test_boolean_units_cannot_impersonate_single_unit_sentinel(tmp_path):
    job = _job("a")
    job["workload"]["units"] = 1
    execution.write_json(tmp_path / "sentinel.json", _sentinel(job, units=True))
    record = {
        "status": "succeeded",
        "elapsed_seconds": 0.5,
        "output_hashes": {"sentinel.json": "unused"},
        "allocation": scheduling.allocate(job, _manifest()["host"], []),
    }
    with pytest.raises(ValueError):
        execution.observation(job, record, tmp_path)


def test_dispatch_uses_calibrated_durations_after_sentinel(monkeypatch, tmp_path):
    sentinel = _job("sentinel", gpus=2)
    sentinel["workload"]["units"] = 1
    sentinel["outputs"] = ["sentinel.json"]
    many = _job("many", ["sentinel"], gpus=2, seconds=1)
    many["workload"]["units"] = 100
    few = _job("few", ["sentinel"], gpus=2, seconds=10)
    few["workload"]["units"] = 2
    clock = Clock()
    engine = FakeEngine(clock, payloads={"sentinel": _sentinel(sentinel)})
    state, _, _ = _run(
        monkeypatch,
        _manifest([sentinel, many, few]),
        tmp_path / "run",
        fake=engine,
        clock=clock,
    )
    assert all(record["status"] == "succeeded" for record in state["jobs"].values())
    assert [key for key, _, _ in engine.starts] == ["sentinel", "many", "few"]


def test_remaining_forecast_cannot_reallocate_active_gpu_or_exclusive_lease():
    for exclusive in (False, True):
        a = _job("a", seconds=10)
        a["resources"]["exclusive"] = exclusive
        b = _job("b", gpus=0 if exclusive else 2, seconds=4)
        manifest = _manifest([a, b])
        allocation = scheduling.allocate(a, manifest["host"], [])
        state = {"observations": [], "jobs": {"a": {"status": "running"}}}
        result = execution.remaining_forecast(
            manifest, state, {"a": {"started": 0, "allocation": allocation}}, 3
        )
        assert result["schedule"][0]["id"] == "b"
        assert result["schedule"][0]["start_seconds"] == 7
        assert result["remaining_seconds"] == 11


def test_cli_execution_exception_is_clean_and_restores_handlers(
    monkeypatch, tmp_path, capsys
):
    manifest = tmp_path / "manifest.json"
    execution.write_json(manifest, _manifest())
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}

    @contextlib.contextmanager
    def lock(path):
        yield

    def fail(*args, **kwargs):
        raise ValueError("injected isolation failure")

    monkeypatch.setattr(execution, "host_lock", lock)
    monkeypatch.setattr(execution, "run", fail)
    with pytest.raises(SystemExit) as error:
        execution.main(
            ["run", "--manifest", str(manifest), "--output", str(tmp_path / "run")]
        )
    assert error.value.code == 2
    assert "injected isolation failure" in capsys.readouterr().err
    assert all(signal.getsignal(sig) == handler for sig, handler in previous.items())


def test_log_timeout_cannot_consume_reserved_removal_budget(monkeypatch, tmp_path):
    clock = Clock()
    engine = FakeEngine(clock)
    observed = []

    def logs(name, output):
        observed.append(("logs", engine.deadline_monotonic))
        clock.now = engine.deadline_monotonic
        raise execution.subprocess.TimeoutExpired("docker logs", 1)

    original_remove = engine.remove

    def remove(name):
        observed.append(("remove", engine.deadline_monotonic))
        assert engine.deadline_monotonic > clock.now
        original_remove(name)

    engine.logs = logs
    engine.remove = remove
    state, _, _ = _run(
        monkeypatch, _manifest([_job("a")]), tmp_path, fake=engine, clock=clock
    )
    assert observed == [("logs", 5), ("remove", 100)]
    assert state["jobs"]["a"]["status"] == "succeeded"
    assert "log_error" in state["jobs"]["a"]


def test_cancellation_skips_logs_and_divides_cleanup_budget(monkeypatch, tmp_path):
    clock = Clock()
    engine = FakeEngine(clock, durations={"a": 100, "b": 100})
    removals = []
    original_remove = engine.remove

    def remove(name):
        removals.append(engine.deadline_monotonic)
        original_remove(name)

    engine.remove = remove
    engine.logs = lambda *args: pytest.fail("cancelled jobs must prioritize removal")
    state, _, _ = _run(
        monkeypatch,
        _manifest([_job("a"), _job("b")]),
        tmp_path,
        fake=engine,
        clock=clock,
        cancelled=lambda: clock.now >= 0.25,
    )
    assert removals == [50.125, 100]
    assert all(row["status"] == "cancelled" for row in state["jobs"].values())


def test_slow_logs_respect_the_other_active_job_deadline(monkeypatch, tmp_path):
    clock = Clock()
    engine = FakeEngine(clock, durations={"b": 100})
    deadlines = []

    def logs(name, output):
        deadlines.append(engine.deadline_monotonic)
        clock.now = engine.deadline_monotonic
        raise execution.subprocess.TimeoutExpired("docker logs", 1)

    engine.logs = logs
    state, _, _ = _run(
        monkeypatch,
        _manifest([_job("a", timeout=50), _job("b", timeout=2)]),
        tmp_path,
        fake=engine,
        clock=clock,
    )
    assert deadlines == [2]
    assert state["jobs"]["b"]["status"] == "cancelled"
    assert state["jobs"]["b"]["elapsed_seconds"] == 2
