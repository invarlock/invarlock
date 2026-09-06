"""Run a bounded local campaign with exclusive device leases and durable results."""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import signal
import stat
import subprocess
import time
import uuid
from functools import cache
from pathlib import Path

from examples.qualification import campaign_scheduling as scheduling


def digest(value):
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def read_json(path):
    if path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError("campaign metadata exceeds 16 MiB")

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    return json.loads(
        path.read_text(),
        object_pairs_hook=pairs,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError("nonfinite JSON")
        ),
    )


def write_json(path, value):
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        os.fchmod(stream.fileno(), 0o600)
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


@contextlib.contextmanager
def host_lock(path):
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
            raise ValueError("host lock must be an operator-owned regular file")
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        os.close(descriptor)


def container_command(job, allocation, output, name, uid, gid):
    memory = allocation["memory_mib"]
    command = [
        "docker",
        "run",
        "--detach",
        "--pull=never",
        "--name",
        name,
        "--label",
        "invarlock.campaign=" + name,
        "--network=none",
        "--read-only",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--pids-limit=2048",
        "--log-driver=json-file",
        "--log-opt=max-size=10m",
        "--log-opt=max-file=1",
        f"--cpus={len(allocation['cpu_ids'])}",
        "--cpuset-cpus=" + ",".join(map(str, allocation["cpu_ids"])),
        f"--memory={memory}m",
        f"--memory-swap={memory}m",
        f"--shm-size={min(8192, memory // 4)}m",
        f"--tmpfs=/tmp:rw,nosuid,nodev,exec,size={min(8192, memory // 4)}m",
        f"--user={uid}:{gid}",
    ]
    if allocation["gpu_ids"]:
        command.extend(
            [
                "--gpus",
                '"device=' + ",".join(allocation["gpu_ids"]) + '"',
                "--env",
                "CUDA_VISIBLE_DEVICES="
                + ",".join(map(str, range(len(allocation["gpu_ids"])))),
            ]
        )
    else:
        command.extend(["--runtime=runc", "--env", "NVIDIA_VISIBLE_DEVICES=void"])
    environment = {
        **job["container"]["environment"],
        "HOME": "/tmp",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "OMP_NUM_THREADS": str(len(allocation["cpu_ids"])),
    }
    for key, value in sorted(environment.items()):
        command.extend(["--env", key + "=" + value])
    for mount in job["container"]["mounts"]:
        source = resolved_mount(mount["source"])
        command.extend(
            ["--mount", f"type=bind,src={source},dst={mount['target']},readonly"]
        )
    command.extend(
        [
            "--mount",
            f"type=bind,src={output.resolve()},dst=/output",
            "--entrypoint",
            job["container"]["command"][0],
            job["container"]["image"],
            *job["container"]["command"][1:],
        ]
    )
    return command


def resolved_mount(source):
    path = Path(source).resolve(strict=True)
    if (
        path == Path("/")
        or path.suffix == ".sock"
        or any(
            path == Path(root) or Path(root) in path.parents
            for root in ("/proc", "/sys", "/dev", "/run", "/var/run", "/etc")
        )
        or not (path.is_file() or path.is_dir())
    ):
        raise ValueError("resolved mount is reserved or not a regular file/directory")
    return path


def mount_hashes(job):
    result = {}
    for mount in job["container"]["mounts"]:
        path = resolved_mount(mount["source"])
        if path.is_file():
            hashed = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    hashed.update(block)
            result[str(path)] = "sha256:" + hashed.hexdigest()
            if mount.get("sha256") and mount["sha256"] != result[str(path)]:
                raise ValueError("mounted source file digest mismatch")
        elif mount.get("sha256"):
            raise ValueError("directory identity requires its explicit validation job")
    return result


def output_hashes(output, paths):
    result = {}
    for name in paths:
        path = output / name
        resolved = path.resolve(strict=True)
        if (
            path.is_symlink()
            or not resolved.is_relative_to(output.resolve())
            or not resolved.is_file()
        ):
            raise ValueError("declared output must be a contained regular file")
        hashed = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                hashed.update(block)
        result[name] = "sha256:" + hashed.hexdigest()
    return result


def verify_bindings(job, allocation, output, observed, launched):
    expected = sorted(
        [
            *(
                (str(resolved_mount(m["source"])), m["target"], False)
                for m in job["container"]["mounts"]
            ),
            (str(output.resolve()), "/output", True),
        ]
    )
    actual = sorted(
        (m.get("Source"), m.get("Destination"), m.get("RW"))
        for m in observed["Mounts"]
        if m.get("Type") == "bind"
    )
    environment = dict(item.split("=", 1) for item in observed["Config"]["Env"])
    expected_environment = dict(
        launched[i + 1].split("=", 1)
        for i, value in enumerate(launched)
        if value == "--env"
    )
    memory = allocation["memory_mib"]
    if (
        actual != expected
        or any(
            m.get("Type") != "bind"
            and (m.get("Type"), m.get("Destination")) != ("tmpfs", "/tmp")
            for m in observed["Mounts"]
        )
        or any(environment.get(k) != v for k, v in expected_environment.items())
        or observed["Config"]["Entrypoint"] != job["container"]["command"][:1]
        or (observed["Config"]["Cmd"] or []) != job["container"]["command"][1:]
        or observed["HostConfig"]["PidsLimit"] != 2048
        or observed["HostConfig"]["ShmSize"] != min(8192, memory // 4) * 1024 * 1024
        or observed["HostConfig"]["Tmpfs"]
        != {"/tmp": f"rw,nosuid,nodev,exec,size={min(8192, memory // 4)}m"}
    ):
        raise ValueError(
            "observed container execution bindings differ from isolation request"
        )


class DockerEngine:
    """Only immutable, isolated containers may execute campaign jobs."""

    def command(self, arguments, *, check=True):
        remaining = getattr(self, "deadline_monotonic", float("inf")) - time.monotonic()
        if remaining <= 0:
            raise ValueError("absolute execution deadline exhausted")
        return subprocess.run(
            arguments,
            capture_output=True,
            text=True,
            timeout=min(30, remaining),
            check=check,
        )

    def probe(self, host):
        allowed = sorted(os.sched_getaffinity(0))
        memory = (
            int(
                next(
                    line.split()[1]
                    for line in Path("/proc/meminfo").read_text().splitlines()
                    if line.startswith("MemTotal:")
                )
            )
            // 1024
        )
        rows = (
            self.command(
                [
                    "nvidia-smi",
                    "--query-gpu=uuid,name,memory.total,pci.bus_id",
                    "--format=csv,noheader,nounits",
                ]
            )
            .stdout.strip()
            .splitlines()
        )
        gpu_rows = [[item.strip() for item in row.split(",")] for row in rows]
        if (
            not set(host["cpu_ids"]) <= set(allowed)
            or len(set(allowed) - set(host["cpu_ids"])) < 8
            or host["memory_mib"] > memory - 32768
        ):
            raise ValueError("declared host exceeds available CPUs or reserved memory")
        known = {row[0] for row in gpu_rows}
        if not set(host["gpu_ids"]) <= known:
            raise ValueError("declared GPU UUID is absent")
        selected = [row for row in gpu_rows if row[0] in host["gpu_ids"]]
        if selected and len({(row[1], row[2]) for row in selected}) != 1:
            raise ValueError("heterogeneous GPUs need separate calibrated hosts")
        retained = self.command(
            [
                "docker",
                "ps",
                "-a",
                "--filter",
                "label=invarlock.campaign",
                "--format",
                "{{.Names}}",
            ]
        )
        if retained.stdout.strip():
            raise ValueError(
                "unreconciled campaign containers remain; retained exact names: "
                + retained.stdout.strip()
            )
        busy = self.command(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"]
        ).stdout.splitlines()
        if set(map(str.strip, busy)) & set(host["gpu_ids"]):
            raise ValueError(
                "unrelated GPU process is active; no foreign process was stopped"
            )
        return {"allowed_cpu_ids": allowed, "memory_mib": memory, "gpus": gpu_rows}

    def inspect(self, name):
        values = json.loads(self.command(["docker", "inspect", name]).stdout)
        if len(values) != 1:
            raise ValueError("ambiguous container identity")
        return values[0]

    def start(self, job, allocation, output, name):
        image = job["container"]["image"]
        actual = self.command(
            ["docker", "image", "inspect", image, "--format", "{{.Id}}"]
        )
        if actual.stdout.strip() != image:
            raise ValueError("image identity mismatch")
        if allocation["gpu_ids"]:
            busy = self.command(
                ["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"]
            ).stdout.splitlines()
            if set(map(str.strip, busy)) & set(allocation["gpu_ids"]):
                raise ValueError("allocated GPU acquired by an unrelated process")
        launched = container_command(
            job, allocation, output, name, os.getuid(), os.getgid()
        )
        self.command(launched)
        observed = self.inspect(name)
        host = observed["HostConfig"]
        devices = [
            device
            for request in (host.get("DeviceRequests") or [])
            for device in (request.get("DeviceIDs") or [])
        ]
        if (
            observed["Image"] != image
            or sorted(devices) != sorted(allocation["gpu_ids"])
            or host["CpusetCpus"] != ",".join(map(str, allocation["cpu_ids"]))
            or host["NanoCpus"] != len(allocation["cpu_ids"]) * 1_000_000_000
            or host["Memory"] != allocation["memory_mib"] * 1024 * 1024
            or host["MemorySwap"] != host["Memory"]
            or host["NetworkMode"] != "none"
            or not host["ReadonlyRootfs"]
            or observed["Config"]["User"] != f"{os.getuid()}:{os.getgid()}"
            or host["CapDrop"] != ["ALL"]
            or not any(
                value.startswith("no-new-privileges") for value in host["SecurityOpt"]
            )
            or host["Privileged"]
            or host.get("Devices")
            or (not allocation["gpu_ids"] and host["Runtime"] != "runc")
            or any(
                mount["RW"]
                for mount in observed["Mounts"]
                if mount["Destination"] != "/output"
            )
            or observed["Config"]["Labels"].get("invarlock.campaign") != name
        ):
            raise ValueError("observed container isolation differs from allocation")
        verify_bindings(job, allocation, output, observed, launched)
        return observed

    def poll(self, name):
        state = self.inspect(name)["State"]
        return None if state["Running"] else state["ExitCode"]

    def remove(self, name):
        result = self.command(["docker", "rm", "--force", name], check=False)
        if result.returncode and "No such container" not in result.stderr:
            raise ValueError("container cleanup uncertain; stop admission: " + name)

    def logs(self, name, output):
        result = self.command(["docker", "logs", name], check=False)
        temporary = output / ("container-log-" + uuid.uuid4().hex + ".tmp")
        with temporary.open("x") as stream:
            os.fchmod(stream.fileno(), 0o600)
            stream.write((result.stdout + result.stderr)[-10 * 1024 * 1024 :])
        os.replace(temporary, output / "container.log")


def observation(job, record, output, jobs=None, records=None):
    path = output / "sentinel.json"
    if not path.exists() or "sentinel.json" not in record["output_hashes"]:
        return None
    value = read_json(path)
    if type(value) is not dict:
        raise ValueError("sentinel must be an object")
    keys = {
        "format",
        "workload_key",
        "units",
        "fixed_seconds",
        "complete",
        "semantic_ready",
    }
    if "observed_job_id" in value:
        target = value.pop("observed_job_id")
        if (
            target not in job["depends_on"]
            or jobs is None
            or records is None
            or records.get(target, {}).get("status") != "succeeded"
        ):
            raise ValueError("sentinel must validate a completed direct dependency")
        job, record = jobs[target], records[target]
    if (
        set(value) != keys
        or value["format"] != "invarlock/campaign-sentinel-v1"
        or value["workload_key"] != job["workload"]["key"]
        or type(value["units"]) is not int
        or value["units"] != job["workload"]["units"]
        or type(value["complete"]) is not bool
        or type(value["semantic_ready"]) is not bool
        or type(value["fixed_seconds"]) not in (int, float)
        or not 0 <= value["fixed_seconds"] <= record["elapsed_seconds"]
    ):
        raise ValueError("sentinel attribution or timing mismatch")
    return {key: value[key] for key in keys - {"format"}} | {
        "job_id": job["id"],
        "status": record["status"],
        "elapsed_seconds": record["elapsed_seconds"],
        "allocation": record["allocation"],
    }


def remaining_forecast(manifest, state, active, now_monotonic):
    """Forecast unfinished work while retaining actual active resource leases."""
    full = scheduling.forecast(manifest, state["observations"])
    details = {row["id"]: row for row in full["jobs"]}
    jobs = {job["id"]: job for job in manifest["jobs"]}
    complete = {
        key for key, record in state["jobs"].items() if record["status"] == "succeeded"
    }
    excluded = {
        key
        for key, record in state["jobs"].items()
        if record["status"] not in ("succeeded", "running")
    }
    while True:
        previous = set(excluded)
        excluded.update(
            key for key, job in jobs.items() if set(job["depends_on"]) & excluded
        )
        if previous == excluded:
            break
    pending = set(jobs) - complete - excluded - set(active)
    overdue = {
        key
        for key, item in active.items()
        if now_monotonic - item["started"] >= details[key]["seconds"]
    }
    running = {
        key: {
            "finish": max(
                0.001,
                (
                    jobs[key]["timeout_seconds"]
                    if key in overdue
                    else details[key]["seconds"]
                )
                - (now_monotonic - item["started"]),
            ),
            "allocation": item["allocation"],
        }
        for key, item in active.items()
    }
    time_now = 0.0
    schedule = []
    while pending or running:
        for key in sorted(pending, key=lambda k: (-details[k]["seconds"], k)):
            if not set(jobs[key]["depends_on"]) <= complete:
                continue
            allocation = scheduling.allocate(
                jobs[key],
                manifest["host"],
                [item["allocation"] for item in running.values()],
            )
            if allocation is not None:
                running[key] = {
                    "finish": time_now + details[key]["seconds"],
                    "allocation": allocation,
                }
                schedule.append(
                    {
                        "id": key,
                        "start_seconds": time_now,
                        "finish_seconds": running[key]["finish"],
                        "allocation": allocation,
                    }
                )
                pending.remove(key)
        if not running:
            raise ValueError("remaining dependency graph cannot make progress")
        time_now = min(item["finish"] for item in running.values())
        for key in [k for k, item in running.items() if item["finish"] <= time_now]:
            del running[key]
            complete.add(key)
    return {
        "format": "invarlock/campaign-remaining-v1",
        "manifest_digest": digest(manifest),
        "scope": "unfinished_admissible_work",
        "remaining_seconds": time_now,
        "expected_remaining_seconds": None if overdue else time_now,
        "overdue_job_ids": sorted(overdue),
        "overdue_policy": "Timeout remainder replaces an exhausted estimate; not an expected duration.",
        "estimated_remaining_cost_usd": time_now
        / 3600
        * manifest["host"]["hourly_cost_usd"],
        "active_job_ids": sorted(active),
        "excluded_job_ids": sorted(excluded),
        "schedule": schedule,
        "complete_campaign_possible": not excluded,
        "limit": "Estimate from matched sentinels or declared assumptions; active leases retained. Excluded jobs are not completed.",
    }


def recover_jobs(state, jobs, output, engine):
    for key, record in state["jobs"].items():
        if record["status"] in ("running", "cleanup_uncertain"):
            engine.remove(record["container_name"])
            record["status"] = "cancelled"
        elif record["status"] == "succeeded":
            if (
                output_hashes(output / "jobs" / key / "output", jobs[key]["outputs"])
                != record["output_hashes"]
                or mount_hashes(jobs[key]) != record["mounted_file_hashes"]
            ):
                raise ValueError("completed output changed; cannot resume")


def journal_transition(output, state, journal_last):
    transition = {key: row["status"] for key, row in state["jobs"].items()}
    transition_digest = digest(transition)
    if transition_digest != journal_last:
        descriptor = os.open(
            output / "events.jsonl",
            os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW,
            0o600,
        )
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise ValueError("journal must be a regular file")
            event = {
                "wall_epoch": state["last_wall_epoch"],
                "statuses": transition,
                "state_digest": digest(state),
            }
            os.write(descriptor, (json.dumps(event, sort_keys=True) + "\n").encode())
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        journal_last = transition_digest
    return journal_last


def load_state(output, identity, clock):
    state_path = output / "state.json"
    if not state_path.exists() and any(output.iterdir()):
        raise ValueError(
            "nonempty output without durable state; refuse implicit restart"
        )
    state = (
        read_json(state_path)
        if state_path.exists()
        else {
            "format": "invarlock/campaign-state-v1",
            "manifest_digest": identity,
            "last_wall_epoch": clock(),
            "jobs": {},
            "observations": [],
        }
    )
    if state["manifest_digest"] != identity or clock() + 1 < state["last_wall_epoch"]:
        raise ValueError("resume manifest changed or clock moved backwards")
    return state


def retain_logs(engine, record, directory, status, monotonic, hard_mono, deadline):
    if status != "cancelled" and monotonic() < hard_mono:
        engine.deadline_monotonic = deadline
        try:
            engine.logs(record["container_name"], directory.parent)
        except (ValueError, OSError, subprocess.SubprocessError) as error:
            record["log_error"] = str(error)


def run(
    manifest,
    output,
    *,
    engine=None,
    clock=time.time,
    monotonic=time.monotonic,
    sleep=time.sleep,
    cancelled=lambda: False,
):
    """Preserve completed outputs on resume; interrupted jobs require a new attempt."""
    if os.getuid() == 0:
        raise ValueError("campaign requires a non-root operator")
    manifest = scheduling.validate_manifest(manifest)
    engine = engine or DockerEngine()
    identity = digest(manifest)
    output.mkdir(mode=0o700, parents=True, exist_ok=True)
    state_path = output / "state.json"
    state = load_state(output, identity, clock)
    start_wall, start_mono = clock(), monotonic()
    stop_wall = manifest["host"]["deadline_epoch"] - manifest["host"]["reserve_seconds"]
    hard_mono = start_mono + max(0, stop_wall - start_wall)
    cleanup_deadline = start_mono + max(
        0, manifest["host"]["deadline_epoch"] - start_wall
    )
    engine.deadline_monotonic = cleanup_deadline
    jobs = {job["id"]: job for job in manifest["jobs"]}
    recover_jobs(state, jobs, output, engine)
    write_json(output / "manifest.json", manifest)
    active = {}
    engine.deadline_monotonic = hard_mono
    state["host_observation"] = engine.probe(manifest["host"])

    journal_last = None

    def persist():
        nonlocal journal_last
        state["last_wall_epoch"] = max(state["last_wall_epoch"], clock())
        state["elapsed_session_seconds"] = monotonic() - start_mono
        write_json(state_path, state)
        journal_last = journal_transition(output, state, journal_last)
        write_json(
            output / "forecast.json",
            scheduling.forecast(manifest, state["observations"]),
        )
        write_json(
            output / "remaining.json",
            remaining_forecast(manifest, state, active, monotonic()),
        )

    def normal_deadline():
        return min(
            [
                hard_mono,
                *(a["started"] + jobs[k]["timeout_seconds"] for k, a in active.items()),
            ]
        )

    def remove_active(key):
        # Divide the remaining cleanup budget so every lease gets an attempt.
        now = monotonic()
        engine.deadline_monotonic = now + max(0, cleanup_deadline - now) / len(active)
        try:
            engine.remove(state["jobs"][key]["container_name"])
        finally:
            engine.deadline_monotonic = hard_mono

    def finish(key, status, exit_code=None):
        record = state["jobs"][key]
        directory = output / "jobs" / key / "output"
        retain_logs(
            engine, record, directory, status, monotonic, hard_mono, normal_deadline()
        )
        remove_active(key)
        record.update(
            status=status,
            exit_code=exit_code,
            finished_epoch=clock(),
            elapsed_seconds=monotonic() - active[key]["started"],
        )
        if status == "succeeded":
            if record["mounted_file_hashes"] != mount_hashes(jobs[key]):
                raise ValueError("mounted source file changed during execution")
            record["output_hashes"] = output_hashes(directory, jobs[key]["outputs"])
            sample = observation(jobs[key], record, directory, jobs, state["jobs"])
            if sample is not None:
                sample["manifest_digest"] = identity
                state["observations"].append(sample)
        del active[key]
        persist()

    @cache
    def descendants(key):
        children = [j["id"] for j in jobs.values() if key in j["depends_on"]]
        return frozenset({key}).union(*(descendants(child) for child in children))

    def tail_timeout(key):
        return sum(
            jobs[child]["timeout_seconds"]
            for child in descendants(key)
            if state["jobs"].get(child, {}).get("status") != "succeeded"
        )

    try:
        while True:
            if cancelled() or clock() >= stop_wall or monotonic() >= hard_mono:
                for key in list(active):
                    finish(key, "cancelled")
                for key in jobs.keys() - state["jobs"].keys():
                    state["jobs"][key] = {"status": "deferred_budget"}
                break
            for key in list(active):
                job_deadline = min(
                    hard_mono, active[key]["started"] + jobs[key]["timeout_seconds"]
                )
                if monotonic() >= job_deadline:
                    finish(key, "cancelled")
                    continue
                engine.deadline_monotonic = normal_deadline()
                code = engine.poll(state["jobs"][key]["container_name"])
                if code is not None:
                    finish(key, "succeeded" if code == 0 else "failed", code)
                elif (
                    monotonic() - active[key]["started"] >= jobs[key]["timeout_seconds"]
                ):
                    finish(key, "cancelled")
            pending = [job for job in jobs.values() if job["id"] not in state["jobs"]]
            fitted = {
                row["id"]: row["seconds"]
                for row in scheduling.forecast(manifest, state["observations"])["jobs"]
            }
            for job in sorted(pending, key=lambda j: (-fitted[j["id"]], j["id"])):
                deps = [
                    state["jobs"].get(key, {}).get("status")
                    for key in job["depends_on"]
                ]
                if any(
                    status is not None and status not in ("succeeded", "running")
                    for status in deps
                ):
                    state["jobs"][job["id"]] = {"status": "blocked_dependency"}
                    continue
                if any(status != "succeeded" for status in deps):
                    continue
                if tail_timeout(job["id"]) > min(
                    stop_wall - clock(), hard_mono - monotonic()
                ):
                    state["jobs"][job["id"]] = {"status": "deferred_budget"}
                    continue
                allocation = scheduling.allocate(
                    job, manifest["host"], [a["allocation"] for a in active.values()]
                )
                if allocation is None:
                    continue
                key = job["id"]
                directory = output / "jobs" / key / "output"
                directory.mkdir(mode=0o700, parents=True, exist_ok=False)
                name = "invarlock-campaign-" + uuid.uuid4().hex
                state["jobs"][key] = {
                    "status": "running",
                    "container_name": name,
                    "allocation": allocation,
                    "started_epoch": clock(),
                    "mounted_file_hashes": mount_hashes(job),
                }
                active[key] = {"allocation": allocation, "started": monotonic()}
                persist()  # Record the exact cleanup target before contacting Docker.
                engine.deadline_monotonic = normal_deadline()
                if min(stop_wall - clock(), hard_mono - monotonic()) < tail_timeout(
                    key
                ):
                    finish(key, "cancelled")
                    continue
                observed = engine.start(job, allocation, directory, name)
                write_json(directory.parent / "container-inspect.json", observed)
            if not active and all(key in state["jobs"] for key in jobs):
                break
            persist()
            sleep(0.25)
    finally:
        cleanup_errors = []
        for key in list(active):
            try:
                remove_active(key)
                state["jobs"][key]["status"] = "cancelled"
                del active[key]
            except (ValueError, OSError, subprocess.SubprocessError) as error:
                state["jobs"][key]["status"] = "cleanup_uncertain"
                cleanup_errors.append(str(error))
        persist()
        if cleanup_errors:
            raise ValueError(
                "cleanup uncertain; no further admission: " + "; ".join(cleanup_errors)
            )
    return state


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("forecast", "run"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        manifest = scheduling.validate_manifest(read_json(args.manifest))
        if args.command == "forecast":
            write_json(args.output, scheduling.forecast(manifest))
            return 0
        stopped = False

        def stop(signum, frame):
            nonlocal stopped
            stopped = True

        previous = {
            sig: signal.signal(sig, stop) for sig in (signal.SIGTERM, signal.SIGINT)
        }
        try:
            with host_lock(Path("/tmp/invarlock-campaign-host.lock")):
                state = run(manifest, args.output, cancelled=lambda: stopped)
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
        return (
            0
            if all(value["status"] == "succeeded" for value in state["jobs"].values())
            else 1
        )
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        parser.exit(2, f"campaign execution: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
