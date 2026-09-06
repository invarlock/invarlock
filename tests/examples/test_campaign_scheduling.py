"""Resource accounting, dependency scheduling and sentinel calibration boundaries."""

import copy
import math

import pytest

from examples.qualification import campaign_scheduling as scheduling

GPU_IDS = [
    "GPU-00000000-0000-0000-0000-000000000001",
    "GPU-00000000-0000-0000-0000-000000000002",
]


def manifest():
    return {
        "format": "invarlock/campaign-schedule-v1",
        "id": "trial",
        "host": {
            "gpu_ids": GPU_IDS.copy(),
            "cpu_ids": [0, 1, 2, 3],
            "memory_mib": 1000,
            "io_slots": 2,
            "deadline_epoch": 2000000000,
            "reserve_seconds": 600,
            "hourly_cost_usd": 3.6,
        },
        "jobs": [job("a"), job("b"), job("c", depends_on=["a", "b"])],
    }


def job(
    name,
    *,
    depends_on=(),
    gpus=1,
    cpus=2,
    memory=400,
    io=1,
    exclusive=False,
    seconds=10,
):
    return {
        "id": name,
        "depends_on": list(depends_on),
        "resources": {
            "gpus": gpus,
            "cpus": cpus,
            "memory_mib": memory,
            "io_slots": io,
            "exclusive": exclusive,
        },
        "timeout_seconds": 100,
        "estimate_seconds": seconds,
        "workload": {"key": "attention-v1", "units": 10, "fixed_seconds": 2},
        "container": {
            "image": "sha256:" + "a" * 64,
            "command": ["python", "-m", "check"],
            "mounts": [{"source": "/srv/models/a", "target": "/models/a"}],
            "environment": {"MODE": "fixed"},
        },
        "outputs": ["sentinel.json"],
    }


def observation(m, *, job_id="a", elapsed=12):
    source = next(j for j in m["jobs"] if j["id"] == job_id)
    return {
        "job_id": job_id,
        "workload_key": source["workload"]["key"],
        "units": source["workload"]["units"],
        "fixed_seconds": source["workload"]["fixed_seconds"],
        "elapsed_seconds": elapsed,
        "status": "succeeded",
        "complete": True,
        "semantic_ready": True,
        "allocation": scheduling.allocate(source, m["host"], []),
        "manifest_digest": scheduling.manifest_digest(m),
    }


def test_dependency_parallelism_cost_and_lower_bound():
    m = manifest()
    out = scheduling.forecast(m)
    assert out["makespan_seconds"] == 20
    assert out["sequential_seconds"] == 30
    assert out["hardware_lower_bound_seconds"] == 20
    assert out["estimated_host_cost_usd"] == pytest.approx(0.02)
    assert [j["start_seconds"] for j in out["jobs"]] == [0, 0, 10]
    assert out["jobs"][0]["allocation"]["gpu_ids"] == [GPU_IDS[0]]
    assert out["jobs"][1]["allocation"]["gpu_ids"] == [GPU_IDS[1]]
    assert out["budget"]["fits_deadline"] is None
    assert out["budget"]["required_with_reserve_seconds"] == 620
    assert out == scheduling.forecast(copy.deepcopy(m))
    validated = scheduling.validate_manifest(m)
    validated["jobs"][0]["container"]["command"].append("changed")
    assert validated != m


@pytest.mark.parametrize(
    "resource,value",
    [
        ("memory_mib", 600),
        ("cpus", 3),
        ("io_slots", 2),
        ("gpus", 2),
        ("exclusive", True),
    ],
)
def test_each_shared_resource_can_force_serial_execution(resource, value):
    m = manifest()
    m["jobs"] = [job("a"), job("b")]
    for j in m["jobs"]:
        j["resources"][resource] = value
    out = scheduling.forecast(m)
    assert out["makespan_seconds"] == 20
    assert out["hardware_lower_bound_seconds"] <= 20


def test_atomic_tp2_and_cpu_only_overlap_and_longest_first():
    m = manifest()
    m["jobs"] = [
        job("short", seconds=2, cpus=1, memory=200, io=0),
        job("long", seconds=20, gpus=2, cpus=2, memory=600, io=0),
        job("cpu", seconds=5, gpus=0, cpus=1, memory=100, io=0),
    ]
    out = scheduling.forecast(m)
    indexed = {j["id"]: j for j in out["jobs"]}
    assert indexed["long"]["start_seconds"] == indexed["cpu"]["start_seconds"] == 0
    assert indexed["short"]["start_seconds"] == 20
    assert out["makespan_seconds"] == 22
    assert indexed["long"]["allocation"]["gpu_ids"] == GPU_IDS


def test_exclusive_jobs_isolate_and_allocations_are_stable():
    m = manifest()
    j = m["jobs"][0]
    allocation = scheduling.allocate(j, m["host"], [])
    assert allocation == {
        "gpu_ids": [GPU_IDS[0]],
        "cpu_ids": [0, 1],
        "memory_mib": 400,
        "io_slots": 1,
        "exclusive": False,
    }
    j["resources"]["exclusive"] = True
    assert scheduling.allocate(j, m["host"], [allocation]) is None
    exclusive = scheduling.allocate(j, m["host"], [])
    j["resources"]["exclusive"] = False
    assert scheduling.allocate(j, m["host"], [exclusive]) is None
    with pytest.raises(ValueError):
        scheduling.allocate(j, m["host"], [allocation, allocation])


def test_calibration_uses_median_and_retains_timeout_exceedance():
    m = manifest()
    m["jobs"][2]["workload"]["units"] = 100
    observations = [observation(m, elapsed=t) for t in [12, 22, 32]]
    result = scheduling.forecast(m, observations)
    c = next(j for j in result["jobs"] if j["id"] == "c")
    assert c["seconds"] == 202
    assert c["seconds_min"] == 102
    assert c["seconds_max"] == 302
    assert c["exceeds_timeout"] is True
    assert c["source"] == "same_workload_sentinels"
    assert c["observation_count"] == 3
    assert result["scenarios"]["minimum"]["makespan_seconds"] == 114
    assert result["scenarios"]["maximum"]["makespan_seconds"] == 334


@pytest.mark.parametrize(
    "field,value,reason",
    [
        ("status", "failed", "status"),
        ("status", "cancelled", "status"),
        ("complete", False, "incomplete"),
        ("semantic_ready", False, "semantic"),
        ("workload_key", "other", "workload"),
        ("units", 11, "workload"),
        ("job_id", "unknown", "unknown"),
        ("manifest_digest", "sha256:" + "0" * 64, "digest"),
        ("elapsed_seconds", 1, "fixed"),
        ("elapsed_seconds", math.nan, "invalid"),
        ("complete", 1, "invalid"),
        ("extra", 1, "invalid"),
    ],
)
def test_ineligible_observations_do_not_calibrate(field, value, reason):
    m = manifest()
    obs = observation(m)
    obs[field] = value
    result = scheduling.forecast(m, [obs])
    assert all(j["source"] == "manifest" for j in result["jobs"])
    assert reason in result["excluded_observations"][0]["reason"]


def test_calibration_never_crosses_resource_class_and_gpu_ids_can_differ():
    m = manifest()
    m["jobs"][1]["resources"]["gpus"] = 2
    obs = observation(m)
    obs["allocation"]["gpu_ids"] = [GPU_IDS[1]]
    out = scheduling.forecast(m, [obs])
    assert next(j for j in out["jobs"] if j["id"] == "b")["source"] == "manifest"
    obs["allocation"]["memory_mib"] += 1
    assert (
        "resource"
        in scheduling.forecast(m, [obs])["excluded_observations"][0]["reason"]
    )


@pytest.mark.parametrize(
    "path,value",
    [
        ((), []),
        (("extra",), True),
        (("host", "extra"), 1),
        (("host", "gpu_ids"), ["GPU-no"]),
        (("host", "cpu_ids"), [0, 0]),
        (("host", "cpu_ids"), [True]),
        (("host", "memory_mib"), 0),
        (("host", "hourly_cost_usd"), math.inf),
        (("host", "reserve_seconds"), -1),
        (("jobs", 0, "resources", "gpus"), 3),
        (("jobs", 0, "resources", "cpus"), True),
        (("jobs", 0, "resources", "exclusive"), 1),
        (("jobs", 0, "estimate_seconds"), 0),
        (("jobs", 0, "timeout_seconds"), 0),
        (("jobs", 0, "workload", "units"), 0),
        (("jobs", 0, "id"), "../bad"),
        (("jobs", 0, "container", "image"), "image:latest"),
        (("jobs", 0, "container", "command"), "sh -c unsafe"),
        (("jobs", 0, "container", "command"), [""]),
        (("jobs", 0, "container", "environment"), {"LD_PRELOAD": "/bad"}),
        (("jobs", 0, "container", "environment"), {"CUDA_VISIBLE_DEVICES": "all"}),
        (("jobs", 0, "container", "environment"), {"bad=key": "x"}),
        (("jobs", 0, "container", "environment"), {"OK": 2}),
        (("jobs", 0, "outputs"), ["../outside"]),
        (("jobs", 0, "outputs"), ["/absolute"]),
        (("jobs", 0, "outputs"), ["a", "a"]),
        (("jobs", 0, "container", "mounts", 0, "source"), "/"),
        (("jobs", 0, "container", "mounts", 0, "source"), "/var/run/docker.sock"),
        (("jobs", 0, "container", "mounts", 0, "source"), "/proc/1"),
        (("jobs", 0, "container", "mounts", 0, "target"), "/output/replace"),
        (("jobs", 0, "container", "mounts", 0, "target"), "/etc"),
        (("jobs", 0, "container", "mounts", 0, "target"), "/a/../b"),
    ],
)
def test_closed_manifest_and_unsafe_values_rejected(path, value):
    m = manifest()
    if not path:
        m = value
    else:
        target = m
        for part in path[:-1]:
            target = target[part]
        target[path[-1]] = value
    with pytest.raises(ValueError):
        scheduling.validate_manifest(m)


@pytest.mark.parametrize("dependencies", [["unknown"], ["a"], ["b", "b"]])
def test_dependency_errors_rejected(dependencies):
    m = manifest()
    m["jobs"][0]["depends_on"] = dependencies
    with pytest.raises(ValueError):
        scheduling.validate_manifest(m)


def test_cycle_duplicate_jobs_and_overlapping_mounts_rejected():
    m = manifest()
    m["jobs"][0]["depends_on"] = ["c"]
    with pytest.raises(ValueError):
        scheduling.validate_manifest(m)
    m = manifest()
    m["jobs"].append(copy.deepcopy(m["jobs"][0]))
    with pytest.raises(ValueError):
        scheduling.validate_manifest(m)
    m = manifest()
    m["jobs"][0]["container"]["mounts"].append(
        {"source": "/srv/other", "target": "/models/a/sub"}
    )
    with pytest.raises(ValueError):
        scheduling.validate_manifest(m)


def test_observed_fixed_cost_replaces_prior_without_cross_image_calibration():
    m = manifest()
    m["jobs"][2]["workload"]["units"] = 100
    m["jobs"][1]["container"]["image"] = "sha256:" + "b" * 64
    obs = observation(m, elapsed=25)
    obs["fixed_seconds"] = 5
    del obs["manifest_digest"]
    out = scheduling.forecast(m, [obs])
    target = next(j for j in out["jobs"] if j["id"] == "c")
    assert target["seconds"] == 205
    assert (
        target["fixed_seconds"]
        == target["fixed_seconds_min"]
        == target["fixed_seconds_max"]
        == 5
    )
    assert target["fixed_source"] == "same_workload_sentinels"
    assert next(j for j in out["jobs"] if j["id"] == "b")["source"] == "manifest"


def test_measured_fixed_and_variable_extrema_are_separate_scenarios():
    m = manifest()
    first, second = observation(m, elapsed=21), observation(m, elapsed=14)
    first["fixed_seconds"] = 1  # rate 2
    second["fixed_seconds"] = 4  # rate 1
    result = scheduling.forecast(m, [first, second])["jobs"][0]
    assert result["seconds"] == 17.5
    assert result["seconds_min"] == 11
    assert result["seconds_max"] == 24
    assert result["fixed_seconds"] == 2.5


def test_zero_duration_calibration_and_cpu_only_reversed_dependencies():
    m = manifest()
    m["host"]["gpu_ids"] = []
    for j in m["jobs"]:
        j["resources"]["gpus"] = 0
        j["container"]["environment"]["EMPTY"] = ""
    m["jobs"].reverse()
    obs = observation(m, elapsed=0)
    obs["fixed_seconds"] = 0
    result = scheduling.forecast(m, [obs])
    assert result["makespan_seconds"] == result["hardware_lower_bound_seconds"] == 0
    assert all(j["start_seconds"] == 0 for j in result["jobs"])


@pytest.mark.parametrize(
    "change",
    [
        "huge_integer",
        "empty_cpus",
        "empty_jobs",
        "many_jobs",
        "empty_command",
        "environment_type",
        "reserved_target",
        "extra_workload",
        "control_character",
        "nonabsolute_source",
        "backslash_output",
    ],
)
def test_additional_input_boundaries(change):
    m = manifest()
    if change == "huge_integer":
        m["host"]["memory_mib"] = 10**1000
    elif change == "empty_cpus":
        m["host"]["cpu_ids"] = []
    elif change == "empty_jobs":
        m["jobs"] = []
    elif change == "many_jobs":
        m["jobs"] *= 100
    elif change == "empty_command":
        m["jobs"][0]["container"]["command"] = []
    elif change == "environment_type":
        m["jobs"][0]["container"]["environment"] = []
    elif change == "reserved_target":
        m["jobs"][0]["container"]["mounts"][0]["target"] = "/tmp"
    elif change == "extra_workload":
        m["jobs"][0]["workload"]["extra"] = 0
    elif change == "control_character":
        m["jobs"][0]["container"]["command"] = ["bad\x00arg"]
    elif change == "nonabsolute_source":
        m["jobs"][0]["container"]["mounts"][0]["source"] = "relative"
    else:
        m["jobs"][0]["outputs"] = ["a\\b"]
    with pytest.raises(ValueError):
        scheduling.validate_manifest(m)


def test_invalid_observation_collection_and_allocations_are_bounded():
    m = manifest()
    for value in ({}, [None] * (scheduling.MAX_OBSERVATIONS + 1)):
        with pytest.raises(ValueError):
            scheduling.forecast(m, value)
    assert (
        scheduling.forecast(m, [None])["excluded_observations"][0]["reason"]
        == "invalid_observation"
    )
    obs = observation(m)
    obs["allocation"]["cpu_ids"] = [99]
    assert (
        scheduling.forecast(m, [obs])["excluded_observations"][0]["reason"]
        == "invalid_observation"
    )
    a = scheduling.allocate(m["jobs"][0], m["host"], [])
    a["exclusive"] = True
    b = copy.deepcopy(a)
    b["cpu_ids"] = [2, 3]
    b["gpu_ids"] = [GPU_IDS[1]]
    with pytest.raises(ValueError):
        scheduling.allocate(m["jobs"][0], m["host"], [a, b])


def test_varied_dags_never_oversubscribe_or_start_before_dependencies():
    import random

    rng = random.Random(1937)
    for _ in range(12):
        m = manifest()
        m["jobs"] = [
            job(
                f"job-{i}",
                depends_on=([f"job-{rng.randrange(i)}"] if i else []),
                gpus=rng.randrange(3),
                cpus=rng.randrange(1, 5),
                memory=rng.randrange(100, 701),
                io=rng.randrange(3),
                exclusive=rng.random() < 0.2,
                seconds=rng.randrange(1, 21),
            )
            for i in range(20)
        ]
        result = scheduling.forecast(m)
        rows = {row["id"]: row for row in result["jobs"]}
        for j in m["jobs"]:
            assert all(
                rows[d]["finish_seconds"] <= rows[j["id"]]["start_seconds"]
                for d in j["depends_on"]
            )
        boundaries = sorted(
            {
                row[k]
                for row in rows.values()
                for k in ("start_seconds", "finish_seconds")
            }
        )
        for left, right in zip(boundaries, boundaries[1:], strict=False):
            point = (left + right) / 2
            active = [
                row["allocation"]
                for row in rows.values()
                if row["start_seconds"] <= point < row["finish_seconds"]
            ]
            for resource in ("gpu_ids", "cpu_ids"):
                ids = [
                    identity
                    for allocation in active
                    for identity in allocation[resource]
                ]
                assert len(ids) == len(set(ids))
            assert sum(a["memory_mib"] for a in active) <= m["host"]["memory_mib"]
            assert sum(a["io_slots"] for a in active) <= m["host"]["io_slots"]
            assert not any(a["exclusive"] for a in active) or len(active) == 1
        assert (
            result["hardware_lower_bound_seconds"]
            <= result["makespan_seconds"]
            <= result["sequential_seconds"]
        )


def test_optional_mount_digest_is_closed_and_docker_separators_are_rejected():
    m = manifest()
    mount = m["jobs"][0]["container"]["mounts"][0]
    mount["sha256"] = "sha256:" + "c" * 64
    assert scheduling.validate_manifest(m)["jobs"][0]["container"]["mounts"][0] == mount
    assert scheduling.forecast(m)["scope"] == "full_manifest"
    for invalid in (None, "c" * 64, "sha256:" + "G" * 64):
        mount["sha256"] = invalid
        with pytest.raises(ValueError):
            scheduling.validate_manifest(m)
    del mount["sha256"]
    for field in ("source", "target"):
        previous = mount[field]
        mount[field] += ",readonly=false"
        with pytest.raises(ValueError):
            scheduling.validate_manifest(m)
        mount[field] = previous


def test_canonical_digest_matches_executor_encoding_for_unicode():
    import hashlib
    import json

    m = manifest()
    m["jobs"][0]["container"]["environment"]["DESCRIPTION"] = "café"
    raw = json.dumps(
        m, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    assert scheduling.manifest_digest(m) == "sha256:" + hashlib.sha256(raw).hexdigest()
    m["format"] = "invarlock/campaign-schedule-v2"
    with pytest.raises(ValueError, match="format"):
        scheduling.validate_manifest(m)
