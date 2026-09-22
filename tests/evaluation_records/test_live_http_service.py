"""Real loopback HTTP with synthetic worker observations; no model/provider calls."""

from __future__ import annotations

import base64
import copy
import hashlib
import importlib.util
import json
import socket
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

HERE = Path(__file__).resolve().parents[2] / "examples/integrations/evaluator-live"
SPEC = importlib.util.spec_from_file_location(
    "http_recipient_fixtures", Path(__file__).with_name("test_live_recipient.py")
)
FIXTURE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FIXTURE)
COMMON = FIXTURE.COMMON
HTTP = COMMON.module("http_service")
CAPTURE = COMMON.module("capture")
RECIPIENT = FIXTURE.LIVE
CAPABILITY = hashlib.sha256(b"private test-only HTTP capability").hexdigest()


def capability_file(path):
    path.write_text(CAPABILITY, encoding="ascii")
    path.chmod(0o600)
    return path


def unused_port():
    with socket.socket() as channel:
        channel.bind(("127.0.0.1", 0))
        return channel.getsockname()[1]


def setup(tmp_path, evaluator="inspect-ai"):
    original = tmp_path / "original"
    original.mkdir()
    protocol, _ = FIXTURE.fixture(original, "inspect-ai")
    protocol["evaluators"] = [evaluator]
    protocol["versions"] = {evaluator: COMMON.versions()[evaluator]}
    protocol["limits"] = {"max_requests": 2, "max_seconds": 30}
    protocol["http_services"] = {}
    for role in ("baseline", "subject"):
        protocol["models"][role].update(
            id=f"synthetic-{role}", revision=role + "-revision"
        )
        protocol["http_services"][role] = {
            "provider": "local-test",
            "service": "synthetic-http",
            "deployment": role,
            "endpoint": f"http://127.0.0.1:{unused_port()}/v1/tasks",
            "requested_model": "same-requested-alias",
            "helper_sha256": HTTP.sha((HERE / "http_service.py").read_bytes()),
        }
    return protocol, original


def observation(protocol, original, role, request):
    case = next(c for c in protocol["cases"] if c["id"] == request["case_id"])
    candidates = [
        COMMON.read(p) for p in (original / role / "tasks").glob("*.response.json")
    ]
    result = copy.deepcopy(
        next(v["result"] for v in candidates if v["request"]["case_id"] == case["id"])
    )
    execution = result["metadata"]["invarlock_model_execution"]
    execution.update(
        request=request,
        protocol_digest=request["protocol_digest"],
        model=protocol["models"][role],
    )
    return {"request": request, "result": result}


def native(evaluator, cases, task, workdir):
    rows = []
    for case in cases:
        result = task(case)
        metadata = {**case["metadata"], **result["metadata"]}
        if evaluator == "inspect-ai":
            row = {
                "id": case["id"],
                "input": case["input"],
                "target": case["expected"],
                "metadata": metadata,
                "output": {"choices": [{"message": {"content": result["output"]}}]},
            }
        elif evaluator == "langfuse":
            row = {
                "item": {
                    "id": case["id"],
                    "input": case["input"],
                    "expected_output": case["expected"],
                    "metadata": metadata,
                },
                "output": result["output"],
                "evaluations": [],
                "trace_id": "synthetic",
                "dataset_run_id": None,
            }
        elif evaluator == "lm-evaluation-harness":
            row = {
                "doc_id": case["id"],
                "doc": case,
                "target": case["expected"],
                "arguments": [[case["input"], {}]],
                "filtered_resps": [result["output"]],
                "metadata": metadata,
            }
        else:
            row = {
                "testIdx": len(rows),
                "promptIdx": 0,
                "prompt": {"raw": case["input"]},
                "testCase": {
                    "vars": {"prompt": case["input"], "case_id": case["id"]},
                    "metadata": {
                        **metadata,
                        "invarlock_id": case["id"],
                        "invarlock_expected": case["expected"],
                    },
                },
                "response": {"output": result["output"]},
                "success": True,
                "score": 1,
            }
        rows.append(row)
    if evaluator == "inspect-ai":
        return {"version": 2, "status": "success", "samples": rows}
    if evaluator == "langfuse":
        return {
            "name": "synthetic",
            "run_name": "synthetic-run",
            "experiment_id": "synthetic",
            "run_evaluations": [],
            "item_results": rows,
        }
    return rows


def captured(tmp_path, monkeypatch, evaluator="inspect-ai"):
    protocol, original = setup(tmp_path, evaluator)
    monkeypatch.setattr(CAPTURE, "local_environment", lambda *a, **k: None)
    monkeypatch.setattr(
        CAPTURE.importlib.metadata, "version", lambda _: protocol["versions"][evaluator]
    )
    package = tmp_path / "promptfoo"
    package.mkdir()
    COMMON.write(package / "package.json", {"version": protocol["versions"][evaluator]})
    monkeypatch.setenv("INVARLOCK_PROMPTFOO_PACKAGE", str(package))
    load = COMMON.module
    monkeypatch.setattr(
        COMMON,
        "module",
        lambda name: SimpleNamespace(run=native) if name == "harness" else load(name),
    )
    manifests = {}
    for role in ("baseline", "subject"):
        monkeypatch.setattr(
            COMMON.TaskClient,
            "exchange",
            lambda self, request, role=role: COMMON.encoded(
                observation(protocol, original, role, request)
            ),
        )
        server, clients = HTTP.make_server(
            protocol,
            role,
            "unused",
            tmp_path / (role + "-server"),
            capability=CAPABILITY,
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            # Earlier in-process CLI checks install the core's default deny policy.
            # Real captures use SDK-only processes; permit this test's loopback phase.
            from invarlock.security import temporarily_allow_network

            with temporarily_allow_network():
                manifests[role] = CAPTURE.capture(
                    protocol,
                    role,
                    evaluator,
                    None,
                    tmp_path / role,
                    http_capability_file=capability_file(
                        tmp_path / (role + "-capability")
                    ),
                )
            assert len(clients[evaluator].complete()) == 2
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)
    protocol_path = tmp_path / "protocol.json"
    COMMON.write(protocol_path, protocol)
    return (
        protocol,
        manifests,
        [
            protocol_path,
            COMMON.digest(protocol),
            tmp_path / "baseline",
            tmp_path / "subject",
            evaluator,
        ],
    )


@pytest.mark.parametrize("evaluator", sorted(HTTP.PRIORITY))
@pytest.mark.parametrize("route", ["native-json", "envelope"])
def test_real_http_preserves_raw_artifact_measurements_and_imports_hosted_nll(
    tmp_path, monkeypatch, evaluator, route
):
    protocol, manifests, args = captured(tmp_path, monkeypatch, evaluator)
    output = tmp_path / "recipient"
    _, runs = RECIPIENT.prepare(*args, route, output)
    for role, run in zip(("baseline", "subject"), runs, strict=True):
        descriptor = manifests[role]["service_identity"]
        assert run["artifact_digest"] is None
        assert run["service_identity"] == descriptor
        assert descriptor["observed_model"] == protocol["models"][role]["id"]
        assert descriptor["exposed_revision"] == protocol["models"][role]["revision"]
        assert "observation_window" not in protocol["http_services"][role]
        for record in run["records"]:
            assert record["likelihood"]["artifact_digest"] is None
            assert record["likelihood"]["service_identity_digest"] == COMMON.digest(
                descriptor
            )
            assert record["likelihood"]["logprob_sum"] == -1
        raw_sdk = (tmp_path / role / "native-original.json").read_bytes()
        assert b'"service_identity_digest"' not in raw_sdk
        assert (
            output / "captures" / role / "native-original.json"
        ).read_bytes() == raw_sdk
        task = COMMON.read(next((tmp_path / role / "tasks").glob("*.response.json")))
        assert (
            task["result"]["metadata"]["invarlock_likelihood"]["artifact_digest"]
            == protocol["models"][role]["artifact_digest"]
        )
    from invarlock.cli.app import app

    runner = CliRunner()

    def command(cwd, *args, allowed=(0,)):
        monkeypatch.chdir(cwd)
        result = runner.invoke(app, [*args, "--json"])
        assert result.exit_code in allowed, result.output
        return json.loads(result.stdout)

    monkeypatch.setattr(RECIPIENT, "command", command)
    result = RECIPIENT.journey(output)
    assert result["normalized_nll"]["verification"]["integrity_ok"] is True
    assert (output / "normalized_nll/verification.receipt.json").exists()


@pytest.mark.parametrize(
    "url",
    [
        "https://127.0.0.1:9000/v1/tasks",
        "http://localhost:9000/v1/tasks",
        "http://127.0.0.2:9000/v1/tasks",
        "http://127.0.0.1:9000/v1/tasks?x=1",
        "http://u@127.0.0.1:9000/v1/tasks",
        "http://127.0.0.1:80/v1/tasks",
        "http://127.0.0.1:9000/other",
    ],
)
def test_endpoint_cannot_enable_external_or_ambiguous_transport(url):
    with pytest.raises(ValueError):
        HTTP.endpoint(url)


@pytest.mark.parametrize(
    "fault",
    [
        "response",
        "prompt",
        "window",
        "window-forward",
        "fields",
        "result",
        "identity",
        "helper",
        "helper-fields",
        "native",
        "missing",
        "extra",
    ],
)
def test_recipient_refuses_http_observation_or_derivation_tampering(
    tmp_path, monkeypatch, fault
):
    _, _, args = captured(tmp_path, monkeypatch)
    root = tmp_path / "subject"
    if fault in {"response", "prompt", "result"}:
        suffix = "request" if fault == "prompt" else "response"
        path = next((root / "http").glob(f"*.{suffix}.json"))
        value = COMMON.read(path)
        field = suffix + "_body_base64"
        payload = COMMON.decode(base64.b64decode(value[field]))
        if fault == "response":
            payload["observed_model"] = "invented"
        elif fault == "result":
            payload["result"]["output"] = "changed"
        else:
            payload["input"] = "changed"
        value[field] = base64.b64encode(COMMON.encoded(payload)).decode()
    elif fault in {"window", "window-forward", "fields"}:
        path = next((root / "http").glob("*.response.json"))
        value = COMMON.read(path)
        if fault == "fields":
            value["unexpected"] = True
        else:
            value["ended_at"] = (
                "1900-01-01T00:00:00Z" if fault == "window" else "2099-01-01T00:00:00Z"
            )
    elif fault == "identity":
        path = root / "capture.json"
        value = COMMON.read(path)
        value["service_identity"]["deployment"] = "changed"
    elif fault == "helper":
        path = root / "capture.json"
        value = COMMON.read(path)
        value["driver_files"]["http_service"] = "sha256:" + "0" * 64
    elif fault == "helper-fields":
        path = root / "capture.json"
        value = COMMON.read(path)
        value["driver_files"] = None
    elif fault == "native":
        path = root / "native-original.json"
        value = COMMON.read(path)
        value["samples"][0]["output"]["choices"][0]["message"]["content"] = "changed"
    elif fault == "missing":
        next((root / "http").glob("*.response.json")).unlink()
    else:
        COMMON.write(root / "http/extra.json", {})
    if fault not in {"missing", "extra"}:
        path.write_bytes(COMMON.encoded(value))
    with pytest.raises((ValueError, FileNotFoundError)):
        RECIPIENT.prepare(*args, "native-json", tmp_path / "rejected")
    assert not (tmp_path / "rejected").exists()


def test_network_permission_is_limited_to_exact_http_peer(monkeypatch):
    network = COMMON.module("network")
    peer = ("127.0.0.1", 9001)
    hooks = []
    monkeypatch.setattr(network.sys, "addaudithook", hooks.append)
    monkeypatch.setattr(network.os, "environ", {})
    network.configure("inspect-ai", http_endpoint=peer)
    channel = SimpleNamespace(family=socket.AF_INET)
    hooks[0]("socket.connect", (channel, peer))
    hooks[0]("socket.getaddrinfo", (*peer, 0, 0, 0))
    for event, args in [
        ("socket.connect", (channel, ("127.0.0.1", 9002))),
        ("socket.connect", (channel, ("8.8.8.8", 9001))),
        ("socket.getaddrinfo", ("example.com", 9001)),
        ("socket.bind", (channel, peer)),
    ]:
        with pytest.raises(RuntimeError):
            hooks[0](event, args)


@pytest.mark.parametrize("fault", ["fields", "text", "pin", "sdk"])
def test_service_declaration_requires_exact_profile_and_source(tmp_path, fault):
    protocol, _ = setup(tmp_path)
    value = protocol["http_services"]["baseline"]
    if fault == "fields":
        value["extra"] = "unsupported"
    elif fault == "text":
        value["deployment"] = "bad\nheader"
    elif fault == "pin":
        value["helper_sha256"] = "sha256:" + "0" * 64
    else:
        protocol["evaluators"] = ["ragas"]
    with pytest.raises(ValueError):
        HTTP.service(protocol, "baseline")


@pytest.mark.parametrize("fault", ["fields", "evaluator", "case", "protocol"])
def test_http_task_requires_frozen_membership_before_execution(tmp_path, fault):
    protocol, _ = setup(tmp_path)
    request = {
        "evaluator": "inspect-ai",
        "case_id": "0",
        "protocol_digest": COMMON.digest({**protocol, "role": "baseline"}),
    }
    if fault == "fields":
        request["extra"] = True
    else:
        request[
            {
                "evaluator": "evaluator",
                "case": "case_id",
                "protocol": "protocol_digest",
            }[fault]
        ] = "changed"
    with pytest.raises(ValueError):
        HTTP.body(protocol, "baseline", request)


@pytest.mark.parametrize(
    "fault", ["fields", "model", "revision", "configuration", "worker"]
)
def test_response_identity_must_match_actual_worker(tmp_path, fault):
    protocol, original = setup(tmp_path)
    request = {
        "evaluator": "inspect-ai",
        "case_id": "0",
        "protocol_digest": COMMON.digest({**protocol, "role": "baseline"}),
    }
    value = {
        **observation(protocol, original, "baseline", request),
        "observed_model": "synthetic-baseline",
        "exposed_revision": "baseline-revision",
        "configuration": protocol["configuration"],
    }
    if fault == "fields":
        value.pop("configuration")
    elif fault == "worker":
        value["result"]["metadata"]["invarlock_model_execution"]["model"] = {}
    else:
        value[
            {
                "model": "observed_model",
                "revision": "exposed_revision",
                "configuration": "configuration",
            }[fault]
        ] = "changed"
    with pytest.raises(ValueError):
        HTTP.checked_response(value, request, protocol, "baseline")


def test_derived_binding_cannot_be_injected_by_sdk():
    with pytest.raises(ValueError, match="SDK cannot supply"):
        HTTP.bind_native({"metadata": {"invarlock_http_binding": {}}}, {})


def test_gateway_rejects_prompt_changes_and_duplicate_tasks_before_worker(
    tmp_path, monkeypatch
):
    from invarlock.security import temporarily_allow_network

    protocol, original = setup(tmp_path)
    calls = []

    def exchange(self, request):
        calls.append(request)
        return COMMON.encoded(observation(protocol, original, "baseline", request))

    monkeypatch.setattr(COMMON.TaskClient, "exchange", exchange)
    server, clients = HTTP.make_server(
        protocol, "baseline", "unused", tmp_path / "server", capability=CAPABILITY
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    request = {
        "evaluator": "inspect-ai",
        "case_id": "0",
        "protocol_digest": COMMON.digest({**protocol, "role": "baseline"}),
    }
    original_body = HTTP.body(protocol, "baseline", request)
    try:
        for fault in (
            "input",
            "reference",
            "configuration",
            "framing",
            None,
            "duplicate",
        ):
            value = copy.deepcopy(original_body)
            if fault in ("input", "reference", "configuration"):
                value[fault] = "changed"
            connection = HTTP.http.client.HTTPConnection(
                *server.server_address, timeout=5
            )
            try:
                with temporarily_allow_network():
                    connection.request(
                        "POST",
                        "/wrong" if fault == "framing" else HTTP.PATH,
                        COMMON.encoded(value),
                        {
                            "Authorization": "Bearer " + CAPABILITY,
                            "Content-Type": "application/json",
                        },
                    )
                    response = connection.getresponse()
                    response.read()
                assert response.status == (200 if fault is None else 400)
            finally:
                connection.close()
        assert len(calls) == 1 and len(clients["inspect-ai"].results) == 1
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_gateway_requires_private_capability_before_worker(tmp_path, monkeypatch):
    from invarlock.security import temporarily_allow_network

    protocol, original = setup(tmp_path)
    calls = []

    def exchange(self, request):
        calls.append(request)
        return COMMON.encoded(observation(protocol, original, "baseline", request))

    monkeypatch.setattr(COMMON.TaskClient, "exchange", exchange)
    server, clients = HTTP.make_server(
        protocol, "baseline", "unused", tmp_path / "server", capability=CAPABILITY
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    request = {
        "evaluator": "inspect-ai",
        "case_id": "0",
        "protocol_digest": COMMON.digest({**protocol, "role": "baseline"}),
    }
    body = COMMON.encoded(HTTP.body(protocol, "baseline", request))
    try:
        for authorization, expected in (
            (None, 403),
            ("Bearer " + "b" * HTTP.CAPABILITY_HEX_LENGTH, 403),
            ("Bearer é", 403),
            ("Bearer " + CAPABILITY, 200),
        ):
            connection = HTTP.http.client.HTTPConnection(*server.server_address)
            headers = {"Content-Type": "application/json"}
            if authorization is not None:
                headers["Authorization"] = authorization
            try:
                with temporarily_allow_network():
                    connection.request("POST", HTTP.PATH, body, headers)
                    response = connection.getresponse()
                    response.read()
                assert response.status == expected
            finally:
                connection.close()
            assert len(calls) == (1 if expected == 200 else 0)
        assert len(clients["inspect-ai"].results) == 1
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_http_capture_keeps_capability_out_of_retained_records(tmp_path, monkeypatch):
    captured(tmp_path, monkeypatch)
    for role in ("baseline", "subject"):
        for directory in (tmp_path / role, tmp_path / (role + "-server")):
            for path in directory.rglob("*"):
                if path.is_file():
                    assert CAPABILITY.encode() not in path.read_bytes(), path


def test_capability_file_requires_private_regular_file(tmp_path):
    private = capability_file(tmp_path / "private")
    assert HTTP.read_capability(private) == CAPABILITY
    private.write_text(CAPABILITY + "\n", encoding="ascii")
    assert HTTP.read_capability(private) == CAPABILITY
    private.write_text("A" * HTTP.CAPABILITY_HEX_LENGTH, encoding="ascii")
    with pytest.raises(ValueError, match="32 random bytes"):
        HTTP.read_capability(private)
    private.chmod(0o644)
    with pytest.raises(ValueError, match="owner-private"):
        HTTP.read_capability(private)
    private.chmod(0o600)
    link = tmp_path / "link"
    link.symlink_to(private)
    with pytest.raises(OSError):
        HTTP.read_capability(link)


def test_direct_http_replay_rejects_unrecognized_helper_source(tmp_path, monkeypatch):
    protocol, manifests, _ = captured(tmp_path, monkeypatch)
    altered = copy.deepcopy(protocol)
    altered["http_services"]["subject"]["helper_sha256"] = "sha256:" + "0" * 64
    manifest = copy.deepcopy(manifests["subject"])
    manifest["driver_files"]["http_service"] = "sha256:" + "0" * 64
    with pytest.raises(ValueError, match="independently frozen source"):
        HTTP.verify_capture(
            tmp_path / "subject",
            manifest,
            altered,
            "subject",
            "inspect-ai",
            {},
            {},
            lambda *args: None,
        )


@pytest.mark.parametrize("failure", ["status", "size", "disconnect"])
def test_failed_http_exchange_keeps_admission_and_never_retries(
    tmp_path, monkeypatch, failure
):
    protocol, _ = setup(tmp_path)
    response = SimpleNamespace(
        status=503 if failure == "status" else 200,
        read=lambda limit: b"x" * limit if failure == "size" else b"{}",
    )
    calls = []

    def getresponse():
        if failure == "disconnect":
            raise ConnectionResetError("synthetic transport loss")
        return response

    connection = SimpleNamespace(
        request=lambda *a, **k: calls.append("request"),
        getresponse=getresponse,
        close=lambda: calls.append("close"),
    )
    monkeypatch.setattr(HTTP.http.client, "HTTPConnection", lambda *a, **k: connection)
    task = HTTP.TaskClient(
        protocol["http_services"]["baseline"]["endpoint"],
        "inspect-ai",
        COMMON.digest({**protocol, "role": "baseline"}),
        protocol["cases"],
        tmp_path / "capture/tasks",
        protocol=protocol,
        role="baseline",
        capability=CAPABILITY,
    )
    with pytest.raises((ValueError, ConnectionResetError)):
        task(protocol["cases"][0])
    assert calls == ["request", "close"]
    with pytest.raises(FileExistsError):
        task(protocol["cases"][0])
    assert calls == ["request", "close"]
    assert len(list(task.output.glob("*.request.json"))) == 1
    assert not list(task.output.glob("*.response.json"))


@pytest.mark.parametrize("failure", [None, "deadline", "pin"])
def test_http_service_cli_preserves_completion_and_deadline(
    tmp_path, monkeypatch, failure
):
    protocol, _ = setup(tmp_path)
    source, output = tmp_path / "protocol.json", tmp_path / "server"
    COMMON.write(source, protocol)
    client = SimpleNamespace(results={})

    class Server:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            closed.append(True)

        def handle_request(self):
            client.results.update({c["id"]: {} for c in protocol["cases"]})

    closed = []

    def make(*args, **kwargs):
        output.mkdir()
        return Server(), {"inspect-ai": client}

    monkeypatch.setattr(HTTP, "make_server", make)
    moments = iter([0, 100] if failure == "deadline" else [0, 1])
    monkeypatch.setattr(HTTP.time, "monotonic", lambda: next(moments))
    monkeypatch.setattr(
        "sys.argv",
        [
            "http_service.py",
            "--protocol",
            str(source),
            "--protocol-sha256",
            "wrong" if failure == "pin" else COMMON.digest(protocol),
            "--role",
            "baseline",
            "--socket",
            "unused",
            "--output",
            str(output),
            "--capability-file",
            str(capability_file(tmp_path / "capability")),
        ],
    )
    if failure:
        with pytest.raises((ValueError, TimeoutError)):
            HTTP.main()
        assert not (output / "complete.json").exists()
    else:
        HTTP.main()
        assert COMMON.read(output / "complete.json") == {
            "status": "complete",
            "requests": 2,
        }
    assert closed == ([] if failure == "pin" else [True])


@pytest.mark.parametrize("invalid_seconds", [0, 86401, 1.5, True])
def test_http_service_cli_refuses_invalid_process_deadline(
    tmp_path, monkeypatch, invalid_seconds
):
    protocol, _ = setup(tmp_path)
    protocol["limits"]["max_seconds"] = invalid_seconds
    source = tmp_path / "protocol.json"
    COMMON.write(source, protocol)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "http_service.py",
            "--protocol",
            str(source),
            "--protocol-sha256",
            COMMON.digest(protocol),
            "--role",
            "baseline",
            "--socket",
            "unused",
            "--output",
            str(tmp_path / "server"),
            "--capability-file",
            str(capability_file(tmp_path / "capability")),
        ],
    )
    with pytest.raises(ValueError, match="bounded whole-process deadline"):
        HTTP.main()
    assert not (tmp_path / "server").exists()


def test_client_refuses_endpoint_drift_and_oversized_task_before_http(
    tmp_path, monkeypatch
):
    protocol, _ = setup(tmp_path)
    options = {"protocol": protocol, "role": "baseline", "capability": CAPABILITY}
    args = (
        "inspect-ai",
        COMMON.digest({**protocol, "role": "baseline"}),
        protocol["cases"],
        tmp_path / "capture/tasks",
    )
    with pytest.raises(ValueError, match="endpoint differs"):
        HTTP.TaskClient("http://127.0.0.1:1/v1/tasks", *args, **options)
    protocol["cases"][0]["input"] = "x" * COMMON.MAX_MESSAGE
    task = HTTP.TaskClient(
        protocol["http_services"]["baseline"]["endpoint"],
        "inspect-ai",
        COMMON.digest({**protocol, "role": "baseline"}),
        protocol["cases"],
        tmp_path / "capture/tasks",
        **options,
    )
    monkeypatch.setattr(
        HTTP.http.client,
        "HTTPConnection",
        lambda *a, **k: pytest.fail("oversized request cannot connect"),
    )
    with pytest.raises(ValueError, match="request exceeds"):
        task(protocol["cases"][0])
    assert len(list(task.output.glob("*.request.json"))) == 1
    assert not list(task.http.iterdir())


def test_absolute_deployment_deadline_interrupts_partial_http_body(tmp_path):
    import subprocess
    import sys
    import time

    from invarlock.security import temporarily_allow_network

    protocol, _ = setup(tmp_path)
    protocol["limits"]["max_seconds"] = 1
    source, output = tmp_path / "protocol.json", tmp_path / "server"
    COMMON.write(source, protocol)
    process = subprocess.Popen(
        [
            sys.executable,
            str(HERE / "http_service.py"),
            "--protocol",
            str(source),
            "--protocol-sha256",
            COMMON.digest(protocol),
            "--role",
            "baseline",
            "--socket",
            "unused",
            "--output",
            str(output),
            "--capability-file",
            str(capability_file(tmp_path / "capability")),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    channel = socket.socket()
    started = time.monotonic()
    try:
        while not (output / "ready.json").exists():
            assert process.poll() is None
            assert time.monotonic() - started < 4
            time.sleep(0.01)
        with temporarily_allow_network():
            channel.connect(
                HTTP.endpoint(protocol["http_services"]["baseline"]["endpoint"])
            )
            channel.sendall(
                b"POST /v1/tasks HTTP/1.1\r\nHost: 127.0.0.1\r\n"
                + ("Authorization: Bearer " + CAPABILITY + "\r\n").encode()
                + b"Content-Type: application/json\r\nContent-Length: 100\r\n\r\n{"
            )
        _, error = process.communicate(timeout=4)
        assert process.returncode != 0
        assert b"admitted window" in error
        assert time.monotonic() - started < 4
        assert not (output / "complete.json").exists()
        assert not list(output.rglob("*.request.json"))
    finally:
        channel.close()
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=5)


def test_http_journal_aggregate_limit_is_enforced(tmp_path, monkeypatch):
    protocol, manifests, _ = captured(tmp_path, monkeypatch)
    directory = tmp_path / "baseline"
    originals = {"native.json": (directory / "native.json").read_bytes()}
    results = {
        v["request"]["case_id"]: v["result"]
        for v in (COMMON.read(p) for p in (directory / "tasks").glob("*.response.json"))
    }
    monkeypatch.setattr(HTTP, "MAX_CAPTURE_BYTES", 1)

    def read(path, limit):
        # Native export read is independent; apply the small aggregate journal cap.
        return RECIPIENT.read(path, COMMON.MAX_MESSAGE * 2)

    with pytest.raises(ValueError, match="journal exceeds"):
        HTTP.verify_capture(
            directory,
            manifests["baseline"],
            protocol,
            "baseline",
            "inspect-ai",
            results,
            originals,
            read,
        )


def test_derived_export_is_recomputed_even_if_its_hash_is_updated(
    tmp_path, monkeypatch
):
    _, _, args = captured(tmp_path, monkeypatch)
    root = tmp_path / "subject"
    path = root / "native.json"
    native = COMMON.read(path)
    native["samples"][0]["metadata"]["invarlock_likelihood"]["logprob_sum"] = -9
    path.write_bytes(COMMON.encoded(native))
    manifest = COMMON.read(root / "capture.json")
    manifest["native_sha256"] = HTTP.sha(path.read_bytes())
    (root / "capture.json").write_bytes(COMMON.encoded(manifest))
    with pytest.raises(ValueError, match="native derivation"):
        RECIPIENT.prepare(*args, "native-json", tmp_path / "rejected")


@pytest.mark.parametrize(
    "peer",
    [["127.0.0.1", 9001], ("localhost", 9001), ("127.0.0.1", True), ("127.0.0.1", 80)],
)
def test_network_permission_refuses_unreviewed_peer(peer, monkeypatch):
    network = COMMON.module("network")
    monkeypatch.setattr(network.os, "environ", {})
    with pytest.raises(ValueError, match="exact loopback"):
        network.configure("inspect-ai", http_endpoint=peer)


@pytest.mark.parametrize("hosted", [False, True])
def test_capture_rejects_missing_or_irrelevant_socket_before_sdk_loading(
    tmp_path, hosted
):
    protocol = {"evaluators": ["inspect-ai"]}
    if hosted:
        protocol["http_services"] = {}
    with pytest.raises(
        ValueError, match="omit --socket" if hosted else "requires --socket"
    ):
        CAPTURE.capture(
            protocol,
            "baseline",
            "inspect-ai",
            "unused" if hosted else None,
            tmp_path / "output",
        )
    assert not (tmp_path / "output").exists()


def test_http_capture_cli_does_not_require_a_dummy_socket(
    tmp_path, monkeypatch, capsys
):
    protocol = {"http_services": {}, "evaluators": ["inspect-ai"]}
    source = tmp_path / "protocol.json"
    COMMON.write(source, protocol)
    calls = []
    monkeypatch.setattr(
        CAPTURE,
        "capture",
        lambda *args, **kwargs: calls.append((args, kwargs)) or {"ok": True},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "capture.py",
            "--protocol",
            str(source),
            "--protocol-sha256",
            COMMON.digest(protocol),
            "--role",
            "baseline",
            "--evaluator",
            "inspect-ai",
            "--output",
            str(tmp_path / "output"),
            "--http-capability-file",
            str(tmp_path / "capability"),
        ],
    )
    CAPTURE.main()
    assert calls[0][0][3] is None
    assert calls[0][1]["http_capability_file"] == tmp_path / "capability"
    assert json.loads(capsys.readouterr().out) == {"ok": True}
