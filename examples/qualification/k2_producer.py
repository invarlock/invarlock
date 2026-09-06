"""Execute bounded K2 captures inside an explicitly pinned candidate image."""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

from examples.qualification import k2_campaign as campaign

BASE_URL = "http://127.0.0.1:30000"


def native_request(path, payload=None, *, timeout=120):
    """Contact only the campaign's loopback server; bound native response bytes."""
    request = urllib.request.Request(
        BASE_URL + path,
        data=None if payload is None else json.dumps(payload, allow_nan=False).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = response.read(1024 * 1024 + 1)
    if len(data) > 1024 * 1024:
        raise ValueError("native response exceeds 1 MiB")
    return json.loads(
        data,
        object_pairs_hook=campaign._object,
        parse_constant=lambda value: campaign._invalid("non-finite native JSON"),
    )


def collect(plan, role, *, phase, request=native_request, clock=time.monotonic):
    """Capture exact requests, native replies, errors, timing, and server settings."""
    campaign.require_ready(plan)
    if role not in campaign.ROLES or phase not in ("preflight", "decision"):
        raise ValueError("unknown role or phase")
    server_info, model_info = request("/server_info"), request("/model_info")
    observed = campaign.observe_server(plan, role, server_info, model_info)
    schedule = plan["preflight_cases"] if phase == "preflight" else plan["cases"]
    budget = plan["budget"]
    started, used_tokens, observed_tokens, rows = clock(), 0, 0, []
    attempted = 0
    for case in schedule:
        remaining = budget["maximum_wall_seconds"] - (clock() - started)
        payload = campaign.request_for(case)
        row = {
            "id": case["id"],
            "request": payload,
            "response": None,
            "latency_ms": 0.0,
            "error": None,
        }
        if (
            remaining <= 0
            or used_tokens + plan["runtime"]["maximum_new_tokens"]
            > budget["maximum_output_tokens"]
        ):
            row["error"] = "predeclared resource budget exhausted"
        else:
            attempted += 1
            before = clock()
            try:
                response = request(
                    "/v1/chat/completions", payload, timeout=min(120, remaining)
                )
                row["response"] = response
                usage = response.get("usage") if isinstance(response, dict) else None
                count = (
                    usage.get("completion_tokens") if isinstance(usage, dict) else None
                )
                observed_tokens += (
                    count if type(count) is int and 0 <= count <= 512 else 0
                )
                used_tokens += count if campaign._answer(row)[1] is None else 512
            except (OSError, ValueError, urllib.error.HTTPError) as error:
                row["error"] = f"native request failed: {type(error).__name__}"
                used_tokens += 512
            row["latency_ms"] = max(0.0, (clock() - before) * 1000)
        rows.append(row)
    final_server, final_model = request("/server_info"), request("/model_info")
    if server_info != final_server or model_info != final_model:
        # Full server_info can contain counters. Only immutable selected fields
        # control eligibility; both observations are retained for inspection.
        campaign.observe_server(plan, role, final_server, final_model)
    elapsed = max(0.0, clock() - started)
    return {
        "format": campaign.CAPTURE_FORMAT,
        "plan_digest": campaign.digest(plan),
        "role": role,
        "phase": phase,
        "runtime": observed,
        "native_server_info": server_info,
        "native_model_info": model_info,
        "final_native_server_info": final_server,
        "final_native_model_info": final_model,
        "rows": rows,
        "resources": {
            "elapsed_seconds": elapsed,
            "completion_tokens": observed_tokens,
            "charged_output_tokens": used_tokens,
            "requests_attempted": attempted,
            "requests_per_second": attempted / elapsed if elapsed else 0.0,
            "tokens_per_second": observed_tokens / elapsed if elapsed else 0.0,
        },
    }


def preflight_summary(capture):
    if capture.get("phase") != "preflight":
        raise ValueError("preflight capture required")
    for field in ("startup_seconds", "elapsed_seconds"):
        duration = capture["resources"].get(field, 0)
        if (
            isinstance(duration, bool)
            or not isinstance(duration, (int, float))
            or not math.isfinite(duration)
            or duration < 0
        ):
            raise ValueError(
                "preflight resource duration must be finite and nonnegative"
            )
    successful = [r for r in capture["rows"] if campaign._answer(r)[1] is None]
    latencies = sorted(r["latency_ms"] for r in successful)
    p95 = (
        latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))]
        if latencies
        else None
    )
    return {
        "format": "invarlock/k2-throughput-preflight-v1",
        "plan_digest": capture["plan_digest"],
        "role": capture["role"],
        "capture_digest": campaign.digest(capture),
        "complete": len(successful) == len(capture["rows"]),
        "p95_request_ms": p95,
        "estimated_decision_seconds": capture["resources"].get("startup_seconds", 0)
        + 576 * p95 / 1000 * 1.25
        if p95 is not None
        else None,
        "resources": capture["resources"],
        "limit": "Observed throughput estimate, not a cost commitment or model-quality result.",
    }


def server_command(plan, role):
    settings = campaign.expected_server_settings(plan, role)
    result = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--host",
        "127.0.0.1",
        "--port",
        "30000",
        "--served-model-name",
        "k2-campaign",
        "--random-seed",
        "20260905",
        "--weight-version",
        plan["model"][role]["materialized"]["artifact_digest"],
        "--mem-fraction-static",
        "0.7",
        "--disable-radix-cache",
        "--log-level",
        "warning",
    ]
    for key, value in settings.items():
        if key in ("trust_remote_code", "quantization"):
            continue
        result.extend(["--" + key.replace("_", "-"), str(value)])
    return result


def _drain(stream, path):
    """Drain all output without letting server logs consume unbounded disk."""
    remaining = 8 * 1024 * 1024
    with path.open("xb") as output:
        while block := stream.read(65536):
            if remaining:
                output.write(block[:remaining])
                remaining = max(0, remaining - len(block))


def validate_hardware(rows, tensor_parallel):
    minimum_memory = {"NVIDIA H100 80GB HBM3": 80000, "NVIDIA H200": 135000}
    try:
        if len(rows) < tensor_parallel:
            raise ValueError("insufficient GPUs")
        names = set()
        for row in rows:
            name, memory, driver, mig = [field.strip() for field in row.split(",")]
            names.add(name)
            driver_parts = tuple(int(part) for part in driver.split("."))
            if (
                name not in minimum_memory
                or int(memory) < minimum_memory[name]
                or mig != "Disabled"
                or len(driver_parts) != 3
                or driver_parts[0] != 580
                or driver_parts < (580, 159, 3)
            ):
                raise ValueError("unsupported GPU, memory, MIG mode, or driver")
        if len(names) != 1:
            raise ValueError("mixed GPU identities")
    except (TypeError, ValueError) as error:
        raise ValueError(
            "hardware differs from the candidate H100/H200 CUDA13 protocol"
        ) from error


def worker(plan, role, phase, output):
    campaign.require_ready(plan)
    hardware = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version,mig.mode.current",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    hardware_rows = hardware.stdout.strip().splitlines()
    validate_hardware(hardware_rows, plan["runtime"]["tensor_parallel"])
    environment = {
        **os.environ,
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HOME": "/tmp/huggingface",
        "HOME": "/tmp",
        "XDG_CACHE_HOME": "/tmp/cache",
    }
    process = subprocess.Popen(
        server_command(plan, role),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=environment,
        start_new_session=True,
    )
    assert process.stdout is not None
    drainer = threading.Thread(
        target=_drain, args=(process.stdout, output / "server.log"), daemon=True
    )
    drainer.start()
    started = time.monotonic()
    try:
        while True:
            if process.poll() is not None:
                raise ValueError(
                    "native server stopped before readiness; retained server.log"
                )
            if time.monotonic() - started > min(
                900, plan["budget"]["maximum_wall_seconds"]
            ):
                raise ValueError("native server startup budget exhausted")
            try:
                native_request("/model_info", timeout=2)
                break
            except (OSError, ValueError):
                time.sleep(0.25)
        captured = collect(plan, role, phase=phase)
        captured["resources"]["startup_seconds"] = (
            time.monotonic() - started - captured["resources"]["elapsed_seconds"]
        )
        captured["hardware"] = hardware_rows
        campaign.write_json(output / "capture.json", captured)
        if phase == "preflight":
            campaign.write_json(output / "preflight.json", preflight_summary(captured))
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
        drainer.join(timeout=5)


def run_container(
    plan_path, role, snapshot, output, phase, engine="docker", preflight=None
):
    uid, gid = os.getuid(), os.getgid()
    if uid == 0:
        raise ValueError("run as a non-root operator with Docker access")
    plan = campaign.read_json(plan_path)
    campaign.require_ready(plan)
    measured = campaign.measure_snapshot(snapshot, plan["model"][role]["files"])
    if measured != plan["model"][role]["materialized"]:
        raise ValueError("actual snapshot differs from frozen materialization")
    if phase == "decision":
        if preflight is None:
            raise ValueError("decision capture requires its retained preflight")
        previous = campaign.read_json(preflight)
        campaign.validate_capture(plan, previous, phase="preflight")
        summary = preflight_summary(previous)
        if (
            summary["plan_digest"] != campaign.digest(plan)
            or summary["role"] != role
            or not summary["complete"]
            or summary["estimated_decision_seconds"]
            > plan["budget"]["maximum_wall_seconds"]
        ):
            raise ValueError(
                "preflight is incomplete, mismatched, or exceeds the frozen budget"
            )
    image = plan["runtime"]["image_digest"]
    inspect = subprocess.run(
        [engine, "image", "inspect", image, "--format", "{{.Id}}"],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    if inspect.stdout.strip() != image:
        raise ValueError("local image identity differs from frozen image")
    output.mkdir(parents=True, exist_ok=False)
    name = "k2-campaign-" + uuid.uuid4().hex
    command = [
        engine,
        "run",
        "--rm",
        "--pull=never",
        "--name",
        name,
        "--network=none",
        "--read-only",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--shm-size=32g",
        "--tmpfs=/tmp:rw,nosuid,nodev,exec,size=16g",
        "--cpus=32",
        "--memory=280g",
        "--gpus=all",
        "--user",
        f"{uid}:{gid}",
        "--env",
        "CUDA_VISIBLE_DEVICES="
        + ("0,1" if plan["runtime"]["tensor_parallel"] == 2 else "0"),
        "--mount",
        f"type=bind,src={plan_path.resolve()},dst=/plan.json,readonly",
        "--mount",
        f"type=bind,src={snapshot.resolve()},dst=/models/{role},readonly",
        "--mount",
        f"type=bind,src={output.resolve()},dst=/output",
        "--entrypoint=python",
        image,
        "-m",
        "examples.qualification.k2_producer",
        "worker",
        "--plan",
        "/plan.json",
        "--role",
        role,
        "--phase",
        phase,
        "--output",
        "/output",
    ]
    try:
        subprocess.run(
            command, check=True, timeout=plan["budget"]["maximum_wall_seconds"]
        )
        if (
            campaign.measure_snapshot(snapshot, plan["model"][role]["files"])
            != measured
        ):
            raise ValueError("snapshot changed during native capture")
    finally:
        cleanup = subprocess.run(
            [engine, "rm", "--force", name],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        if cleanup.returncode and "No such container" not in cleanup.stderr:
            raise ValueError(
                f"container cleanup failed; stop the retained worker {name} before further execution"
            )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "worker"))
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--role", choices=campaign.ROLES, required=True)
    parser.add_argument("--phase", choices=("preflight", "decision"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--preflight", type=Path)
    parser.add_argument("--engine", default="docker")
    args = parser.parse_args(argv)
    try:
        if args.command == "worker":
            worker(campaign.read_json(args.plan), args.role, args.phase, args.output)
        else:
            if args.snapshot is None:
                raise ValueError("run requires --snapshot")
            run_container(
                args.plan,
                args.role,
                args.snapshot,
                args.output,
                args.phase,
                args.engine,
                args.preflight,
            )
        return 0
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        parser.exit(2, f"k2 capture worker: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
