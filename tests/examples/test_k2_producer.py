"""The capture worker performs bounded native calls and retains adverse outcomes."""

from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from examples.qualification import k2_campaign as campaign
from examples.qualification import k2_producer as capture_worker
from tests.examples.test_k2_campaign import _ready_plan


@pytest.fixture(autouse=True)
def nonroot_operator(monkeypatch):
    monkeypatch.setattr(
        capture_worker, "os", SimpleNamespace(**vars(capture_worker.os))
    )
    monkeypatch.setattr(capture_worker.os, "getuid", lambda: 1234)
    monkeypatch.setattr(capture_worker.os, "getgid", lambda: 2345)


def _transport(plan, role="baseline", *, fail=False, mutate=False):
    calls = []
    observations = 0

    def request(path, payload=None, **kwargs):
        nonlocal observations
        if path == "/server_info":
            observations += 1
            settings = campaign.expected_server_settings(plan, role)
            if mutate and observations > 1:
                settings["dtype"] = "float16"
            return settings
        if path == "/model_info":
            return {
                "model_path": f"/models/{role}",
                "tokenizer_path": f"/models/{role}",
                "model_type": "k2_horizon",
                "architectures": ["K2HorizonForCausalLM"],
                "served_model_name": "k2-campaign",
                "weight_version": plan["model"][role]["materialized"][
                    "artifact_digest"
                ],
            }
        assert path == "/v1/chat/completions"
        calls.append((copy.deepcopy(payload), kwargs))
        if fail:
            raise TimeoutError("test timeout")
        return {
            "model": "k2-campaign",
            "choices": [
                {"index": 0, "finish_reason": "stop", "message": {"content": "review"}}
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
        }

    return request, calls


def test_producer_calls_native_api_with_frozen_settings_and_keeps_raw_responses():
    plan = _ready_plan()
    request, calls = _transport(plan)
    capture = capture_worker.collect(
        plan, "baseline", phase="preflight", request=request
    )
    assert len(calls) == 24
    assert calls[0][0] == campaign.request_for(plan["preflight_cases"][0])
    assert calls[0][1]["timeout"] <= 120
    assert capture["rows"][0]["response"]["usage"]["completion_tokens"] == 2
    assert capture["resources"]["completion_tokens"] == 48
    assert capture_worker.preflight_summary(capture)["complete"] is True
    with pytest.raises(ValueError, match="plan or role"):
        campaign.project_capture(plan, capture)


def test_failed_native_requests_are_retained_and_fail_preflight():
    plan = _ready_plan()
    request, calls = _transport(plan, fail=True)
    capture = capture_worker.collect(
        plan, "baseline", phase="preflight", request=request
    )
    assert len(calls) == 24
    assert all(
        row["error"] == "native request failed: TimeoutError" for row in capture["rows"]
    )
    assert capture_worker.preflight_summary(capture)["complete"] is False


def test_budget_stops_requests_without_dropping_scheduled_cases():
    plan = _ready_plan()
    plan["budget"]["maximum_output_tokens"] = 1
    request, calls = _transport(plan)
    capture = capture_worker.collect(
        plan, "baseline", phase="decision", request=request
    )
    assert calls == []
    assert len(capture["rows"]) == 576
    assert capture["resources"]["requests_attempted"] == 0
    assert capture["resources"]["requests_per_second"] == 0
    runs = campaign.project_capture(plan, capture)
    assert all(row["error"] for run in runs.values() for row in run["records"])


def test_native_runtime_mutation_is_rejected():
    plan = _ready_plan()
    request, _ = _transport(plan, mutate=True)
    with pytest.raises(ValueError, match="runtime"):
        capture_worker.collect(plan, "baseline", phase="preflight", request=request)


def test_server_arguments_keep_bf16_native_loading_and_mova_routing():
    plan = campaign.select_plan("mova-36b-a4b")
    plan["model"]["baseline"]["materialized"] = {
        "artifact_digest": "sha256:" + "a" * 64
    }
    command = capture_worker.server_command(plan, "baseline")
    assert "--trust-remote-code" not in command
    assert "--quantization" not in command
    assert command[command.index("--dtype") + 1] == "bfloat16"
    assert command[command.index("--tp-size") + 1] == "2"
    assert json.loads(command[command.index("--json-model-override-args") + 1]) == {
        "xllm_source_router_gemm_partitions": 2
    }


def test_root_operator_rejected_before_plan_engine_or_output(tmp_path, monkeypatch):
    monkeypatch.setattr(capture_worker.os, "getuid", lambda: 0)
    monkeypatch.setattr(
        campaign, "read_json", lambda *args: pytest.fail("must not read plan")
    )
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("must not contact engine"),
    )
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="non-root operator with Docker access"):
        capture_worker.run_container(
            tmp_path / "missing-plan.json", "baseline", tmp_path, output, "preflight"
        )
    assert not output.exists()


def test_container_timeout_removes_only_its_named_worker(tmp_path, monkeypatch):
    monkeypatch.setattr(capture_worker.os, "getuid", lambda: 1234)
    monkeypatch.setattr(capture_worker.os, "getgid", lambda: 2345)
    plan = _ready_plan()
    plan_path = tmp_path / "plan.json"
    campaign.write_json(plan_path, plan)
    monkeypatch.setattr(
        campaign,
        "measure_snapshot",
        lambda *args: plan["model"]["baseline"]["materialized"],
    )
    calls = []

    def run(arguments, **kwargs):
        calls.append((arguments, kwargs))
        if arguments[1] == "image":
            return subprocess.CompletedProcess(
                arguments, 0, plan["runtime"]["image_digest"] + "\n", ""
            )
        if arguments[1] == "run":
            raise subprocess.TimeoutExpired(arguments, kwargs["timeout"])
        return subprocess.CompletedProcess(arguments, 0)

    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(subprocess.TimeoutExpired):
        capture_worker.run_container(
            plan_path, "baseline", tmp_path, tmp_path / "output", "preflight"
        )
    command = calls[1][0]
    assert "--network=none" in command and "--read-only" in command
    assert "--tmpfs=/tmp:rw,nosuid,nodev,exec,size=16g" in command
    assert "--pull=never" in command
    assert "--memory=280g" in command and "--cpus=32" in command
    assert command[command.index("--user") + 1] == "1234:2345"
    assert plan["runtime"]["image_digest"] in command
    name = command[command.index("--name") + 1]
    assert calls[-1][0] == ["docker", "rm", "--force", name]
    assert calls[1][1]["timeout"] == plan["budget"]["maximum_wall_seconds"]


def test_missing_preflight_cannot_start_decision_container(tmp_path, monkeypatch):
    plan = _ready_plan()
    plan_path = tmp_path / "plan.json"
    campaign.write_json(plan_path, plan)
    monkeypatch.setattr(
        campaign,
        "measure_snapshot",
        lambda *args: plan["model"]["baseline"]["materialized"],
    )
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("must not start container"),
    )
    with pytest.raises(ValueError, match="preflight"):
        capture_worker.run_container(
            plan_path, "baseline", tmp_path, tmp_path / "output", "decision"
        )


@pytest.mark.parametrize(
    "rows,tp",
    [
        (["NVIDIA A100, 80000, 580.159.03, Disabled"], 1),
        (["NVIDIA H200, 140000, 570.159.03, Disabled"], 1),
        (["NVIDIA H200, 70000, 580.159.03, Disabled"], 1),
        (["NVIDIA H200, 140000, 580.159.03, Disabled"], 2),
    ],
)
def test_hardware_preflight_rejects_wrong_device_memory_driver_or_count(rows, tp):
    with pytest.raises(ValueError, match="hardware"):
        capture_worker.validate_hardware(rows, tp)


def test_hardware_preflight_accepts_full_h200_devices():
    capture_worker.validate_hardware(
        ["NVIDIA H200, 143771, 580.178.04, Disabled"] * 2, 2
    )


@pytest.mark.parametrize(
    "name,memory",
    [
        ("NVIDIA H100 80GB HBM3", 81559),
        ("NVIDIA H100 80GB HBM3", 80000),
        ("NVIDIA H200", 135000),
    ],
)
def test_hardware_accepts_full_supported_devices_with_mig_disabled(name, memory):
    capture_worker.validate_hardware([f"{name}, {memory}, 580.159.03, Disabled"] * 2, 2)


@pytest.mark.parametrize(
    "rows",
    [
        ["NVIDIA H100 80GB HBM3, 79999, 580.159.03, Disabled"] * 2,
        ["NVIDIA H100 80GB HBM3 MIG 1g.10gb, 81559, 580.159.03, Disabled"] * 2,
        ["NVIDIA H100 80GB HBM3, 81559, 580.159.03, Enabled"] * 2,
        ["NVIDIA H100 80GB HBM3, 81559, 580.159.03, N/A"] * 2,
        ["NVIDIA H100 PCIe, 81559, 580.159.03, Disabled"] * 2,
        ["NVIDIA H100 NVL, 95830, 580.159.03, Disabled"] * 2,
        ["NVIDIA H200 unreviewed, 143771, 580.159.03, Disabled"] * 2,
        [
            "NVIDIA H200, 143771, 580.159.03, Disabled",
            "NVIDIA H100 80GB HBM3, 81559, 580.159.03, Disabled",
        ],
        ["NVIDIA H100 80GB HBM3, 81559, 580.159.03"],
    ],
)
def test_hardware_rejects_undersized_partitioned_unknown_or_mixed_devices(rows):
    with pytest.raises(ValueError, match="hardware"):
        capture_worker.validate_hardware(rows, min(2, len(rows)))


def test_plan_cli_writes_an_unqualified_plan(tmp_path):
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            __import__("sys").executable,
            "-m",
            "examples.qualification.k2_campaign",
            "plan",
            "--model",
            "7b",
            "--output",
            str(tmp_path / "plan.json"),
        ],
        cwd=root,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    plan = campaign.read_json(tmp_path / "plan.json")
    assert plan["model"]["repository"] == "IFM/K2-Horizon-7B"
    assert plan["status"] == "candidate_not_qualified"


@pytest.mark.parametrize("mutation", ["short_schedule", "request", "final_config"])
def test_decision_refuses_altered_preflight_before_container_start(
    tmp_path, monkeypatch, mutation
):
    plan = _ready_plan()
    campaign.write_json(tmp_path / "plan.json", plan)
    request, _ = _transport(plan)
    captured = capture_worker.collect(
        plan, "baseline", phase="preflight", request=request
    )
    if mutation == "short_schedule":
        captured["rows"] = captured["rows"][:1]
    elif mutation == "request":
        captured["rows"][0]["request"]["temperature"] = 1
    else:
        captured["final_native_server_info"]["trust_remote_code"] = True
    campaign.write_json(tmp_path / "preflight.json", captured)
    monkeypatch.setattr(
        campaign,
        "measure_snapshot",
        lambda *args: plan["model"]["baseline"]["materialized"],
    )
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail(
            "must reject preflight before container start"
        ),
    )
    with pytest.raises(ValueError):
        capture_worker.run_container(
            tmp_path / "plan.json",
            "baseline",
            tmp_path,
            tmp_path / "output",
            "decision",
            preflight=tmp_path / "preflight.json",
        )


@pytest.mark.parametrize(
    "response",
    [
        None,
        [],
        {"usage": None},
        {"usage": []},
        {
            "model": "k2-campaign",
            "choices": [{"index": 0, "finish_reason": "stop", "message": None}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    ],
)
def test_malformed_native_shape_is_retained_and_charged(tmp_path, response):
    plan = _ready_plan()
    plan["budget"]["maximum_output_tokens"] = 512
    normal, _ = _transport(plan)

    def request(path, payload=None, **kwargs):
        return response if payload is not None else normal(path, payload, **kwargs)

    captured = capture_worker.collect(
        plan, "baseline", phase="preflight", request=request
    )
    assert captured["rows"][0]["response"] == response
    assert campaign._answer(captured["rows"][0])[1] is not None
    assert captured["resources"]["charged_output_tokens"] == 512
    assert captured["resources"]["requests_attempted"] == 1
    assert all(row["error"] for row in captured["rows"][1:])
    assert not capture_worker.preflight_summary(captured)["complete"]


@pytest.mark.parametrize("field", ["startup_seconds", "elapsed_seconds"])
@pytest.mark.parametrize("value", [-1, float("inf"), float("nan"), None, "1", True])
def test_preflight_rejects_impossible_resource_duration(field, value):
    plan = _ready_plan()
    request, _ = _transport(plan)
    captured = capture_worker.collect(
        plan, "baseline", phase="preflight", request=request
    )
    captured["resources"][field] = value
    with pytest.raises(ValueError, match="duration"):
        capture_worker.preflight_summary(captured)


@pytest.mark.parametrize(
    "driver", ["580.126.20", "580.159.02", "590.99.99", "580", "580.159"]
)
def test_hardware_requires_a_verified_security_fixed_driver_branch(driver):
    with pytest.raises(ValueError, match="hardware"):
        capture_worker.validate_hardware(
            [f"NVIDIA H200, 143771, {driver}, Disabled"], 1
        )
