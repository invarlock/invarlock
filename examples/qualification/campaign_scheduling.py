"""Bounded resource planning for immutable, offline campaign jobs.

This module performs no execution or filesystem access. Container mount sources
may be future dependency outputs; the executor must check their actual identity,
symlinks and file types before launch. Commands are argument arrays, never shell
command text interpreted by this module.

A workload key asserts equivalent model and role, corpus/profile/phase, token and
context policy, concurrency and source identity. It is an operator declaration,
not something this planner can infer or authenticate. Calibration additionally
requires the same pinned image and GPU/CPU/memory/I/O/exclusivity resource class.
GPU IDs may differ only under the caller's homogeneous-host assumption. Forecasts
are deterministic greedy schedules, not optimality or statistical guarantees.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import statistics
from pathlib import PurePosixPath

FORMAT = "invarlock/campaign-schedule-v1"
MAX_JOBS = 256
MAX_OBSERVATIONS = 4096
_SLUG = re.compile(r"[a-z0-9][a-z0-9-]{0,63}\Z")
_GPU = re.compile(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}\Z")
_IMAGE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_ENV = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,127}\Z")
_RESOURCES = {"gpus", "cpus", "memory_mib", "io_slots", "exclusive"}
_ALLOCATION = {"gpu_ids", "cpu_ids", "memory_mib", "io_slots", "exclusive"}
_HOST = {
    "gpu_ids",
    "cpu_ids",
    "memory_mib",
    "io_slots",
    "deadline_epoch",
    "reserve_seconds",
    "hourly_cost_usd",
}
_JOB = {
    "id",
    "depends_on",
    "resources",
    "timeout_seconds",
    "estimate_seconds",
    "workload",
    "container",
    "outputs",
}
_OBSERVATION = {
    "job_id",
    "workload_key",
    "units",
    "fixed_seconds",
    "elapsed_seconds",
    "status",
    "complete",
    "semantic_ready",
    "allocation",
}
_BLOCK_ENV = {
    "CUDA_VISIBLE_DEVICES",
    "NVIDIA_VISIBLE_DEVICES",
    "NVIDIA_DRIVER_CAPABILITIES",
    "LD_PRELOAD",
    "LD_LIBRARY_PATH",
    "LD_AUDIT",
    "PYTHONPATH",
    "PYTHONHOME",
    "PATH",
    "BASH_ENV",
    "ENV",
}


def _fail(message):
    raise ValueError(message)


def _object(value, keys, label, optional=()):
    if (
        type(value) is not dict
        or set(value) - keys - set(optional)
        or keys - set(value)
    ):
        _fail(f"{label} fields must match the closed schema")


def _number(value, label, *, integer=False, minimum=0, positive=False):
    kinds = (int,) if integer else (int, float)
    if (
        type(value) not in kinds
        or value > 10**12
        or value < minimum
        or not math.isfinite(value)
        or (positive and value == 0)
    ):
        _fail(f"{label} must be a bounded finite {'integer' if integer else 'number'}")


def _text(value, label, *, maximum=4096, pattern=None):
    if (
        not isinstance(value, str)
        or not value
        or len(value) > maximum
        or any(ord(c) < 32 or ord(c) == 127 for c in value)
        or (pattern is not None and not pattern.fullmatch(value))
    ):
        _fail(f"{label} must be a bounded safe string")


def _list(value, label, maximum, *, nonempty=False):
    if type(value) is not list or len(value) > maximum or (nonempty and not value):
        _fail(f"{label} must be a bounded list")


def _unique(value, label):
    if len(set(value)) != len(value):
        _fail(f"{label} contains duplicates")


def _ids(value, label, *, gpu):
    _list(value, label, 256 if gpu else 4096)
    for entry in value:
        if gpu:
            _text(entry, label, pattern=_GPU)
        else:
            _number(entry, label, integer=True)
    _unique(value, label)


def _path(value, label, *, absolute):
    _text(value, label)
    path = PurePosixPath(value)
    if (
        path.is_absolute() != absolute
        or "\\" in value
        or any(
            part in (".", "..", "") for part in value.split("/")[1 if absolute else 0 :]
        )
        or str(path) != value
    ):
        _fail(
            f"{label} must be a normalized {'absolute' if absolute else 'relative'} path"
        )
    return path


def _beneath(path, parent):
    return path == parent or parent in path.parents


def _host(host):
    _object(host, _HOST, "host")
    _ids(host["gpu_ids"], "host GPU IDs", gpu=True)
    _ids(host["cpu_ids"], "host CPU IDs", gpu=False)
    if not host["cpu_ids"]:
        _fail("host must have CPUs")
    for name in ("memory_mib", "io_slots", "deadline_epoch"):
        _number(host[name], f"host {name}", integer=True, positive=True)
    _number(host["reserve_seconds"], "host reserve_seconds", integer=True)
    _number(host["hourly_cost_usd"], "host hourly_cost_usd")


def _resources(resources, host):
    _object(resources, _RESOURCES, "resources")
    for name in ("gpus", "cpus", "memory_mib", "io_slots"):
        _number(
            resources[name], name, integer=True, positive=name in ("cpus", "memory_mib")
        )
    if type(resources["exclusive"]) is not bool:
        _fail("exclusive must be boolean")
    capacities = {
        "gpus": len(host["gpu_ids"]),
        "cpus": len(host["cpu_ids"]),
        "memory_mib": host["memory_mib"],
        "io_slots": host["io_slots"],
    }
    if any(resources[k] > capacity for k, capacity in capacities.items()):
        _fail("job resources cannot fit host")


def _job(job, host):
    _object(job, _JOB, "job")
    _text(job["id"], "job ID", pattern=_SLUG)
    _list(job["depends_on"], "dependencies", MAX_JOBS)
    for dependency in job["depends_on"]:
        _text(dependency, "dependency ID", pattern=_SLUG)
    _unique(job["depends_on"], "dependencies")
    _resources(job["resources"], host)
    _number(job["timeout_seconds"], "timeout_seconds", integer=True, positive=True)
    _number(job["estimate_seconds"], "estimate_seconds", positive=True)
    workload = job["workload"]
    _object(workload, {"key", "units", "fixed_seconds"}, "workload")
    _text(workload["key"], "workload key", maximum=1024)
    _number(workload["units"], "workload units", integer=True, positive=True)
    _number(workload["fixed_seconds"], "workload fixed_seconds")
    container = job["container"]
    _object(container, {"image", "command", "mounts", "environment"}, "container")
    _text(container["image"], "image", pattern=_IMAGE)
    _list(container["command"], "command", 128, nonempty=True)
    for argument in container["command"]:
        _text(argument, "command argument")
    _list(container["mounts"], "mounts", 64)
    targets = []
    for mount in container["mounts"]:
        _object(mount, {"source", "target"}, "mount", ("sha256",))
        if "sha256" in mount:
            _text(mount["sha256"], "mount sha256", pattern=_IMAGE)
        source = _path(mount["source"], "mount source", absolute=True)
        target = _path(mount["target"], "mount target", absolute=True)
        if "," in mount["source"] or "," in mount["target"]:
            _fail("mount paths cannot contain Docker option separators")
        if (
            source == PurePosixPath("/")
            or source.suffix == ".sock"
            or any(
                _beneath(source, PurePosixPath(p))
                for p in ("/proc", "/sys", "/dev", "/run", "/var/run", "/etc")
            )
        ):
            _fail("reserved or socket mount source")
        if target == PurePosixPath("/") or any(
            _beneath(target, PurePosixPath(p))
            for p in ("/output", "/tmp", "/proc", "/sys", "/dev", "/etc")
        ):
            _fail("reserved mount target")
        if any(_beneath(target, old) or _beneath(old, target) for old in targets):
            _fail("overlapping mount targets")
        targets.append(target)
    environment = container["environment"]
    if type(environment) is not dict or len(environment) > 128:
        _fail("environment must be a bounded mapping")
    for key, value in environment.items():
        _text(key, "environment key", pattern=_ENV)
        if key in _BLOCK_ENV or key.startswith(("LD_", "DYLD_")):
            _fail("reserved environment key")
        # Empty environment values are useful; control characters remain forbidden.
        if value != "":
            _text(value, "environment value")
    _list(job["outputs"], "outputs", 128)
    for output in job["outputs"]:
        _path(output, "output", absolute=False)
    _unique(job["outputs"], "outputs")


def validate_manifest(value) -> dict:
    """Return an independent validated copy; reject ambiguity before execution."""
    _object(value, {"format", "id", "host", "jobs"}, "manifest")
    if value["format"] != FORMAT:
        _fail("unsupported manifest format")
    _text(value["id"], "manifest ID", pattern=_SLUG)
    _host(value["host"])
    _list(value["jobs"], "jobs", MAX_JOBS, nonempty=True)
    for job in value["jobs"]:
        _job(job, value["host"])
    names = [job["id"] for job in value["jobs"]]
    _unique(names, "job IDs")
    known = set(names)
    for job in value["jobs"]:
        if job["id"] in job["depends_on"] or set(job["depends_on"]) - known:
            _fail("unknown or self dependency")
    pending = {job["id"]: set(job["depends_on"]) for job in value["jobs"]}
    done = set()
    while pending:
        ready = {name for name, dependencies in pending.items() if dependencies <= done}
        if not ready:
            _fail("dependency cycle")
        done.update(ready)
        for name in ready:
            del pending[name]
    return copy.deepcopy(value)


def manifest_digest(manifest) -> str:
    """Canonical manifest identity shared with the executor's retained copy."""
    data = json.dumps(
        validate_manifest(manifest),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _allocation(value, host):
    _object(value, _ALLOCATION, "allocation")
    _ids(value["gpu_ids"], "allocated GPU IDs", gpu=True)
    _ids(value["cpu_ids"], "allocated CPU IDs", gpu=False)
    if set(value["gpu_ids"]) - set(host["gpu_ids"]) or set(value["cpu_ids"]) - set(
        host["cpu_ids"]
    ):
        _fail("allocation IDs outside host")
    _resources(
        {
            "gpus": len(value["gpu_ids"]),
            "cpus": len(value["cpu_ids"]),
            **{k: value[k] for k in ("memory_mib", "io_slots", "exclusive")},
        },
        host,
    )


def _allocate(resources, host, active):
    if active and (resources["exclusive"] or any(a["exclusive"] for a in active)):
        return None
    used_gpus = {gpu for a in active for gpu in a["gpu_ids"]}
    used_cpus = {cpu for a in active for cpu in a["cpu_ids"]}
    gpus = [gpu for gpu in host["gpu_ids"] if gpu not in used_gpus]
    cpus = [cpu for cpu in host["cpu_ids"] if cpu not in used_cpus]
    if (
        len(gpus) < resources["gpus"]
        or len(cpus) < resources["cpus"]
        or sum(a["memory_mib"] for a in active) + resources["memory_mib"]
        > host["memory_mib"]
        or sum(a["io_slots"] for a in active) + resources["io_slots"] > host["io_slots"]
    ):
        return None
    return {
        "gpu_ids": gpus[: resources["gpus"]],
        "cpu_ids": cpus[: resources["cpus"]],
        "memory_mib": resources["memory_mib"],
        "io_slots": resources["io_slots"],
        "exclusive": resources["exclusive"],
    }


def allocate(job, host, active_allocations) -> dict | None:
    """Allocate atomically from host order, with no overlapping resource leases."""
    _host(host)
    _job(job, host)
    _list(active_allocations, "active allocations", MAX_JOBS)
    admitted = []
    for allocation in active_allocations:
        _allocation(allocation, host)
        resources = {
            "gpus": len(allocation["gpu_ids"]),
            "cpus": len(allocation["cpu_ids"]),
            **{k: allocation[k] for k in ("memory_mib", "io_slots", "exclusive")},
        }
        if _allocate(resources, host, admitted) is None or any(
            set(allocation[k]) & set(old[k])
            for old in admitted
            for k in ("gpu_ids", "cpu_ids")
        ):
            _fail("active allocations overlap or exceed host resources")
        admitted.append(allocation)
    return _allocate(job["resources"], host, admitted)


def _resource_class(resources):
    return tuple(
        resources[k] for k in ("gpus", "cpus", "memory_mib", "io_slots", "exclusive")
    )


def _key(job):
    return (
        job["workload"]["key"],
        job["container"]["image"],
        _resource_class(job["resources"]),
    )


def _observations(observations, jobs, host, digest):
    if (
        not isinstance(observations, (list, tuple))
        or len(observations) > MAX_OBSERVATIONS
    ):
        _fail("observations must be a bounded list or tuple")
    rates = {}
    excluded = []
    for index, observation in enumerate(observations):
        reason = None
        try:
            _object(observation, _OBSERVATION, "observation", ("manifest_digest",))
            _text(observation["job_id"], "observed job ID", pattern=_SLUG)
            _text(observation["workload_key"], "observed workload key", maximum=1024)
            _number(observation["units"], "observed units", integer=True, positive=True)
            for field in ("fixed_seconds", "elapsed_seconds"):
                _number(observation[field], field)
            if (
                observation["status"] not in ("succeeded", "failed", "cancelled")
                or type(observation["complete"]) is not bool
                or type(observation["semantic_ready"]) is not bool
            ):
                _fail("invalid observation status")
            _allocation(observation["allocation"], host)
            if "manifest_digest" in observation:
                _text(
                    observation["manifest_digest"],
                    "observed manifest digest",
                    pattern=_IMAGE,
                )
        except ValueError:
            reason = "invalid_observation"
        if reason is None:
            source = jobs.get(observation["job_id"])
            if source is None:
                reason = "unknown_job"
            elif observation.get("manifest_digest", digest) != digest:
                reason = "manifest_digest_mismatch"
            elif observation["status"] != "succeeded":
                reason = "status_not_succeeded"
            elif not observation["complete"]:
                reason = "incomplete"
            elif not observation["semantic_ready"]:
                reason = "semantic_not_ready"
            elif any(
                observation[a] != source["workload"][b]
                for a, b in (
                    ("workload_key", "key"),
                    ("units", "units"),
                )
            ):
                reason = "workload_mismatch"
            elif observation["elapsed_seconds"] < observation["fixed_seconds"]:
                reason = "elapsed_below_fixed_seconds"
            else:
                allocation = observation["allocation"]
                resources = {
                    "gpus": len(allocation["gpu_ids"]),
                    "cpus": len(allocation["cpu_ids"]),
                    **{
                        k: allocation[k]
                        for k in ("memory_mib", "io_slots", "exclusive")
                    },
                }
                if _resource_class(resources) != _resource_class(source["resources"]):
                    reason = "resource_class_mismatch"
                else:
                    rate = (
                        observation["elapsed_seconds"] - observation["fixed_seconds"]
                    ) / observation["units"]
                    rates.setdefault(_key(source), []).append(
                        (observation["fixed_seconds"], rate)
                    )
        if reason is not None:
            excluded.append(
                {
                    "index": index,
                    "job_id": observation.get("job_id")
                    if type(observation) is dict
                    and isinstance(observation.get("job_id"), str)
                    and len(observation["job_id"]) <= 64
                    else None,
                    "reason": reason,
                }
            )
    return rates, excluded


def _simulate(jobs, host, durations, priority):
    pending = set(jobs)
    done = set()
    active = {}
    schedule = {}
    now = 0.0
    while pending or active:
        ready = sorted(
            (name for name in pending if set(jobs[name]["depends_on"]) <= done),
            key=lambda name: (-priority[name], name),
        )
        for name in ready:
            allocation = _allocate(
                jobs[name]["resources"],
                host,
                [schedule[x]["allocation"] for x in active],
            )
            if allocation is not None:
                end = now + durations[name]
                schedule[name] = {
                    "start_seconds": now,
                    "finish_seconds": end,
                    "allocation": allocation,
                }
                active[name] = end
                pending.remove(name)
        # Manifest validation guarantees a finite DAG with individually fitting jobs.
        now = min(active.values())
        finished = {name for name, end in active.items() if end <= now}
        for name in finished:
            del active[name]
        done.update(finished)
    return now, schedule


def _lower_bound(jobs, host, durations):
    critical = {}
    while len(critical) < len(jobs):
        for name, job in jobs.items():
            if name not in critical and set(job["depends_on"]) <= set(critical):
                critical[name] = durations[name] + max(
                    (critical[d] for d in job["depends_on"]), default=0
                )
    exclusive = sum(
        durations[name] for name, job in jobs.items() if job["resources"]["exclusive"]
    )
    bounds = []
    for resource, capacity in (
        ("gpus", len(host["gpu_ids"])),
        ("cpus", len(host["cpu_ids"])),
        ("memory_mib", host["memory_mib"]),
        ("io_slots", host["io_slots"]),
    ):
        if capacity:
            work = sum(
                durations[name] * job["resources"][resource]
                for name, job in jobs.items()
                if not job["resources"]["exclusive"]
            )
            bounds.append(exclusive + work / capacity)
    return max(max(critical.values()), *bounds)


def forecast(manifest, observations=()) -> dict:
    """Simulate longest-estimate-first admission; use no implicit wall clock.

    Minimum/maximum scenarios use the observed rate extrema, not confidence
    bounds. The median schedule fixes their admission priority. Timeout excesses
    are reported without changing the estimate or granting execution authority.
    """
    manifest = validate_manifest(manifest)
    digest = manifest_digest(manifest)
    jobs = {job["id"]: job for job in manifest["jobs"]}
    host = manifest["host"]
    rates, excluded = _observations(observations, jobs, host, digest)
    details = {}
    for name, job in jobs.items():
        observed = rates.get(_key(job), [])
        seconds = job["estimate_seconds"]
        minimum = maximum = seconds
        fixed = fixed_min = fixed_max = job["workload"]["fixed_seconds"]
        if observed:
            fixed_values, variable_rates = zip(*observed, strict=True)
            fixed = statistics.median(fixed_values)
            fixed_min, fixed_max = min(fixed_values), max(fixed_values)
            units = job["workload"]["units"]
            seconds = fixed + statistics.median(variable_rates) * units
            minimum, maximum = (
                fixed_min + min(variable_rates) * units,
                fixed_max + max(variable_rates) * units,
            )
        details[name] = {
            "id": name,
            "seconds": seconds,
            "seconds_min": minimum,
            "seconds_max": maximum,
            "fixed_seconds": fixed,
            "fixed_seconds_min": fixed_min,
            "fixed_seconds_max": fixed_max,
            "fixed_source": "same_workload_sentinels" if observed else "manifest",
            "source": "same_workload_sentinels" if observed else "manifest",
            "observation_count": len(observed),
            "exceeds_timeout": seconds > job["timeout_seconds"],
        }
    durations = {name: row["seconds"] for name, row in details.items()}
    makespan, schedule = _simulate(jobs, host, durations, durations)
    scenarios = {}
    for label, field in (("minimum", "seconds_min"), ("maximum", "seconds_max")):
        duration, _ = _simulate(
            jobs, host, {name: row[field] for name, row in details.items()}, durations
        )
        scenarios[label] = {
            "makespan_seconds": duration,
            "estimated_host_cost_usd": duration / 3600 * host["hourly_cost_usd"],
        }
    return {
        "format": "invarlock/campaign-forecast-v1",
        "scope": "full_manifest",
        "manifest_digest": digest,
        "policy": "Deterministic dependency-ready longest-estimate-first; no optimality guarantee.",
        "makespan_seconds": makespan,
        "sequential_seconds": sum(durations.values()),
        "hardware_lower_bound_seconds": _lower_bound(jobs, host, durations),
        "estimated_host_cost_usd": makespan / 3600 * host["hourly_cost_usd"],
        "jobs": [details[name] | schedule[name] for name in sorted(jobs)],
        "scenarios": scenarios,
        "excluded_observations": excluded,
        "budget": {
            "deadline_epoch": host["deadline_epoch"],
            "reserve_seconds": host["reserve_seconds"],
            "required_with_reserve_seconds": makespan + host["reserve_seconds"],
            "available_seconds": None,
            "fits_deadline": None,
            "reason": "No wall-clock time supplied; executor must enforce deadline and reserve.",
        },
        "limitations": [
            "Rate extrema are scenarios, not confidence bounds.",
            "Whole-host cost excludes unmodeled idle time and uses the supplied hourly rate.",
            "Runtime observations do not establish model quality or qualification.",
        ],
    }
