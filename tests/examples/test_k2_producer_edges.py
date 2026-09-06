"""Exercise worker cleanup and command boundaries without a native runtime."""

from __future__ import annotations

import io
import json
import signal
import subprocess
from pathlib import Path

import pytest

from examples.qualification import k2_campaign as campaign
from examples.qualification import k2_producer as capture_worker
from tests.examples.test_k2_campaign import _ready_plan
from tests.examples.test_k2_producer import _transport


def test_native_transport_serializes_exact_request_and_bounds_responses(monkeypatch):
    seen = []

    def open_url(request, timeout):
        seen.append((request, timeout))
        return io.BytesIO(b'{"response":"ok"}')

    monkeypatch.setattr(capture_worker.urllib.request, "urlopen", open_url)
    assert capture_worker.native_request(
        "/v1/chat/completions", {"temperature": 0}, timeout=3
    ) == {"response": "ok"}
    request, timeout = seen[0]
    assert request.full_url == "http://127.0.0.1:30000/v1/chat/completions"
    assert json.loads(request.data) == {"temperature": 0}
    assert timeout == 3
    assert capture_worker.native_request("/model_info") == {"response": "ok"}
    monkeypatch.setattr(
        capture_worker.urllib.request,
        "urlopen",
        lambda *a, **k: io.BytesIO(b" " * (1024 * 1024 + 1)),
    )
    with pytest.raises(ValueError, match="exceeds"):
        capture_worker.native_request("/model_info")
    monkeypatch.setattr(
        capture_worker.urllib.request,
        "urlopen",
        lambda *a, **k: io.BytesIO(b'{"value":NaN}'),
    )
    with pytest.raises(ValueError, match="non-finite"):
        capture_worker.native_request("/model_info")


def test_unknown_collection_phase_and_non_preflight_summary_are_rejected():
    with pytest.raises(ValueError, match="phase"):
        capture_worker.collect(_ready_plan(), "baseline", phase="other")
    with pytest.raises(ValueError, match="preflight"):
        capture_worker.preflight_summary({"phase": "decision"})


def test_server_log_is_drained_with_bounded_retention(tmp_path):
    data = b"x" * (9 * 1024 * 1024)
    stream = io.BytesIO(data)
    capture_worker._drain(stream, tmp_path / "server.log")
    assert (tmp_path / "server.log").stat().st_size == 8 * 1024 * 1024
    assert stream.tell() == len(data)


@pytest.mark.parametrize(
    "mode,phase",
    [
        ("normal", "preflight"),
        ("normal", "decision"),
        ("stopped", "preflight"),
        ("startup_timeout", "preflight"),
        ("kill_timeout", "preflight"),
    ],
)
def test_worker_retains_capture_and_cleans_its_process_group(
    tmp_path, monkeypatch, mode, phase
):
    plan = _ready_plan()
    signals, process_calls = [], []

    class Process:
        pid = 654321
        stdout = io.BytesIO(b"native server diagnostic\n")
        waits = 0

        def poll(self):
            return 2 if mode == "stopped" else None

        def wait(self, timeout):
            self.waits += 1
            if mode == "kill_timeout" and self.waits == 1:
                raise subprocess.TimeoutExpired("server", timeout)
            return 0

    process = Process()

    def start(command, **kwargs):
        process_calls.append((command, kwargs))
        return process

    monkeypatch.setattr(capture_worker.subprocess, "Popen", start)
    monkeypatch.setattr(
        capture_worker.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command, 0, "NVIDIA H200, 143771, 580.178.04\n", ""
        ),
    )
    monkeypatch.setattr(
        capture_worker.os, "killpg", lambda pid, sig: signals.append((pid, sig))
    )
    request, _ = _transport(plan)
    original_collect = capture_worker.collect
    monkeypatch.setattr(
        capture_worker,
        "collect",
        lambda p, r, phase: original_collect(p, r, phase=phase, request=request),
    )
    attempts = 0

    def readiness(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ConnectionError("starting")
        return {}

    monkeypatch.setattr(capture_worker, "native_request", readiness)
    monkeypatch.setattr(capture_worker.time, "sleep", lambda duration: None)
    if mode == "startup_timeout":
        ticks = iter([0.0, 1000.0])
        monkeypatch.setattr(capture_worker.time, "monotonic", lambda: next(ticks))
    if mode in {"stopped", "startup_timeout"}:
        with pytest.raises(ValueError, match="stopped|startup"):
            capture_worker.worker(plan, "baseline", phase, tmp_path)
    else:
        capture_worker.worker(plan, "baseline", phase, tmp_path)
        captured = campaign.read_json(tmp_path / "capture.json")
        assert captured["hardware"] == ["NVIDIA H200, 143771, 580.178.04"]
        assert len(captured["rows"]) == (24 if phase == "preflight" else 576)
        assert (tmp_path / "preflight.json").exists() is (phase == "preflight")
    assert (tmp_path / "server.log").read_bytes() == b"native server diagnostic\n"
    assert process_calls[0][1]["env"]["HF_HUB_OFFLINE"] == "1"
    assert process_calls[0][1]["start_new_session"] is True
    if mode == "stopped":
        assert signals == []
    else:
        assert signals[0] == (process.pid, signal.SIGTERM)
    if mode == "kill_timeout":
        assert signals[-1] == (process.pid, signal.SIGKILL)


@pytest.mark.parametrize(
    "problem", ["materialization", "image", "preflight", "changed_after", "none"]
)
def test_container_requires_matching_materialization_image_and_preflight(
    tmp_path, monkeypatch, problem
):
    plan = _ready_plan()
    campaign.write_json(tmp_path / "plan.json", plan)
    request, _ = _transport(plan)
    preflight = capture_worker.collect(
        plan, "baseline", phase="preflight", request=request
    )
    if problem == "preflight":
        preflight["plan_digest"] = "wrong"
    campaign.write_json(tmp_path / "preflight.json", preflight)
    calls, measurements = [], 0

    def measure(*args):
        nonlocal measurements
        measurements += 1
        return (
            {}
            if problem == "materialization"
            or (problem == "changed_after" and measurements > 1)
            else plan["model"]["baseline"]["materialized"]
        )

    def run(command, **kwargs):
        calls.append(command)
        output = "wrong" if problem == "image" else plan["runtime"]["image_digest"]
        return subprocess.CompletedProcess(command, 0, output, "")

    monkeypatch.setattr(campaign, "measure_snapshot", measure)
    monkeypatch.setattr(capture_worker.subprocess, "run", run)
    if problem == "none":
        capture_worker.run_container(
            tmp_path / "plan.json",
            "baseline",
            tmp_path,
            tmp_path / "out",
            "decision",
            preflight=tmp_path / "preflight.json",
        )
        assert calls[-1][1:3] == ["rm", "--force"]
    else:
        with pytest.raises(ValueError):
            capture_worker.run_container(
                tmp_path / "plan.json",
                "baseline",
                tmp_path,
                tmp_path / "out",
                "decision",
                preflight=tmp_path / "preflight.json",
            )
        if problem in {"materialization", "preflight"}:
            assert not calls
        if problem == "changed_after":
            assert calls[-1][1:3] == ["rm", "--force"]


def test_producer_cli_runs_both_entrypoints_and_rejects_missing_snapshot(
    tmp_path, monkeypatch
):
    campaign.write_json(tmp_path / "plan.json", _ready_plan())
    calls = []
    monkeypatch.setattr(
        capture_worker, "worker", lambda *args: calls.append(("worker", args))
    )
    monkeypatch.setattr(
        capture_worker, "run_container", lambda *args: calls.append(("run", args))
    )
    args = [
        "--plan",
        str(tmp_path / "plan.json"),
        "--role",
        "baseline",
        "--phase",
        "preflight",
        "--output",
        str(tmp_path / "out"),
    ]
    assert capture_worker.main(["worker", *args]) == 0
    assert capture_worker.main(["run", *args, "--snapshot", str(tmp_path)]) == 0
    assert [c[0] for c in calls] == ["worker", "run"]
    with pytest.raises(SystemExit) as error:
        capture_worker.main(["run", *args])
    assert error.value.code == 2


def test_actual_producer_cli_refuses_unresolved_runtime(tmp_path):
    campaign.write_json(tmp_path / "draft.json", campaign.select_plan("0.9b"))
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            __import__("sys").executable,
            "-m",
            "examples.qualification.k2_producer",
            "worker",
            "--plan",
            str(tmp_path / "draft.json"),
            "--role",
            "baseline",
            "--phase",
            "preflight",
            "--output",
            str(tmp_path),
        ],
        cwd=root,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "unresolved runtime binding" in result.stderr
    assert not (tmp_path / "capture.json").exists()


def test_failed_container_cleanup_is_reported(tmp_path, monkeypatch):
    plan = _ready_plan()
    campaign.write_json(tmp_path / "plan.json", plan)
    monkeypatch.setattr(
        campaign,
        "measure_snapshot",
        lambda *args: plan["model"]["baseline"]["materialized"],
    )

    def run(command, **kwargs):
        return subprocess.CompletedProcess(
            command,
            1 if command[1] == "rm" else 0,
            plan["runtime"]["image_digest"],
            "daemon unavailable",
        )

    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(ValueError, match="cleanup failed"):
        capture_worker.run_container(
            tmp_path / "plan.json", "baseline", tmp_path, tmp_path / "out", "preflight"
        )
