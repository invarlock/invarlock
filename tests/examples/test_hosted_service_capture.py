"""HTTP simulation fixtures exercise collection boundaries, not model qualification."""

import base64
import importlib.util
import json
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "examples/hosted-service/capture.py"


def module():
    spec = importlib.util.spec_from_file_location("hosted_capture", SCRIPT)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_reject_duplicate_json_keys():
    with pytest.raises(ValueError, match="duplicate"):
        module().decode(b'{"model":"a","model":"b"}')


@pytest.mark.parametrize(
    "url",
    [
        "http://example.com/v1/chat/completions",
        "https://user:secret@example.com/v1/chat/completions",
        "https://example.com/v1/chat/completions?key=secret",
        "https://example.com/v1/chat/completions#secret",
        "file:///tmp/endpoint",
    ],
)
def test_reject_unsafe_endpoint(url):
    with pytest.raises(ValueError, match="endpoint"):
        module().endpoint(url)


def response(content="yes", model="simulation-model"):
    return {
        "model": model,
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": content},
            }
        ],
        "usage": {"completion_tokens": 1},
    }


@contextmanager
def server(responses):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            requests.append(
                {
                    "body": json.loads(
                        self.rfile.read(int(self.headers["Content-Length"]))
                    ),
                    "authorization": self.headers.get("Authorization"),
                }
            )
            item = responses[min(len(requests) - 1, len(responses) - 1)]
            if isinstance(item, float):
                time.sleep(item)
                return
            status, body = item
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, *args):
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=httpd.serve_forever, daemon=True)
    worker.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_port}/v1/chat/completions", requests
    finally:
        httpd.shutdown()
        worker.join()
        httpd.server_close()


def protocol(helper, url, count=2):
    service = {
        "provider": "local-http-simulation",
        "service": "chat",
        "deployment": "test",
        "endpoint": url,
        "model": "simulation-model",
        "expected_observed_model": "simulation-model",
        "exposed_revision": None,
    }
    return {
        "format": helper.FORMAT,
        "services": {
            "baseline": service,
            "subject": {**service, "deployment": "test-second-window"},
        },
        "environment": {"purpose": "HTTP simulation fixture"},
        "journey_source_digest": helper.digest(
            SCRIPT.with_name("journey.py").read_bytes()
        ),
        "collector_source_digest": helper.digest(SCRIPT.read_bytes()),
        "harness": {
            "name": "http-simulation-fixture",
            "version": "1",
            "source_digest": helper.digest(Path(__file__).read_bytes()),
        },
        "configuration": {
            "temperature": 0,
            "system_prompt": "Reply yes",
            "max_tokens": 8,
        },
        "limits": {
            "max_calls": count,
            "max_total_output_tokens": count * 8,
            "timeout_seconds": 2,
            "max_wall_seconds": 10,
            "max_response_bytes": 4096,
        },
        "cases": [
            {
                "id": f"case-{i}",
                "input": "Is this a simulation fixture?",
                "expected": "yes",
            }
            for i in range(count)
        ],
        "policy": {
            "format": "invarlock/comparison-policy-v1",
            "slices": [],
            "metrics": [
                {
                    "name": "quality",
                    "kind": "exact_match",
                    "configuration": {},
                    "direction": "higher",
                    "unit": "score",
                    "aggregation": "mean",
                    "minimum_count": count,
                    "maximum_regression": 0.7,
                    "maximum_interval_width": 2,
                    "subject_minimum": 0.1,
                }
            ],
        },
    }


def collect_fixture(tmp_path, helper, declaration, role="baseline", suffix=""):
    path = tmp_path / f"protocol{suffix}.json"
    pin = helper.write(path, declaration)
    destination = tmp_path / f"{role}{suffix}"
    capture_pin = helper.collect(path, pin, role, destination)
    return path, pin, destination / "capture.json", capture_pin


def test_collect_and_replay_actual_http_without_new_calls(tmp_path):
    helper = module()
    with server([(200, helper.encoded(response()))]) as (url, calls):
        values = collect_fixture(tmp_path, helper, protocol(helper, url))
        run = helper.export_run(*values, "baseline")
        assert len(calls) == 2
        assert all(call["authorization"] is None for call in calls)
        assert run["artifact_digest"] is None
        assert run["service_identity"]["observed_model"] == "simulation-model"
        assert run["source_digest"] == values[3]
        assert all(
            row["output"] == "yes" and row["error"] is None for row in run["records"]
        )
        raw = json.loads(values[2].read_bytes())
        assert (
            run["records"][0]["context"]["http_observation"] == raw["observations"][0]
        )
        helper.export_run(*values, "baseline")
        assert len(calls) == 2


@pytest.mark.parametrize(
    "status,body,error",
    [
        (429, b'{"error":"rate limit"}', "http_status_error"),
        (302, b"{}", "http_status_error"),
        (200, b"not json", "invalid_response"),
        (200, b"x" * 4097, "response_too_large"),
        (200, b"[]", "invalid_response"),
    ],
)
def test_http_failure_is_retained_without_retry(tmp_path, status, body, error):
    helper = module()
    with server([(status, body)]) as (url, calls):
        if status == 200:
            with pytest.raises(ValueError, match="missing_token_accounting"):
                collect_fixture(tmp_path, helper, protocol(helper, url, 1))
            assert len(calls) == 1
            retained = json.loads((tmp_path / "baseline/000000.json").read_bytes())
            assert base64.b64decode(retained["body_base64"]) == body[:4096]
            assert not (tmp_path / "baseline/capture.json").exists()
            return
        values = collect_fixture(tmp_path, helper, protocol(helper, url, 1))
        run = helper.export_run(*values, "baseline")
        assert len(calls) == 1
        row = run["records"][0]
        assert row["error"] == error and row["output"] is None
        assert row["context"]["http_observation"]["status"] == status
        assert (
            base64.b64decode(row["context"]["http_observation"]["body_base64"])
            == body[:4096]
        )


def test_deadline_is_absolute_and_retained(tmp_path):
    helper = module()
    with server([1.0]) as (url, calls):
        value = protocol(helper, url, 1)
        value["limits"]["timeout_seconds"] = 0.2
        before = time.monotonic()
        values = collect_fixture(tmp_path, helper, value)
        assert time.monotonic() - before < 0.8
        run = helper.export_run(*values, "baseline")
        assert len(calls) == 1
        assert run["records"][0]["error"] == "deadline_exceeded"


@pytest.mark.parametrize(
    "change",
    [
        lambda p: p["cases"].append(p["cases"][0]),
        lambda p: p["cases"][0].update(input={"not": "text"}),
        lambda p: p["limits"].update(max_total_output_tokens=1),
        lambda p: p["limits"].update(max_calls=True),
        lambda p: p["limits"].update(timeout_seconds=float("inf")),
        lambda p: p["configuration"].update(temperature=1),
        lambda p: p.update(collector_source_digest="not-a-digest"),
        lambda p: p["configuration"].update(unreviewed="flag"),
    ],
)
def test_invalid_predeclarations_never_contact_service(tmp_path, change):
    helper = module()
    with server([(200, helper.encoded(response()))]) as (url, calls):
        value = protocol(helper, url)
        change(value)
        with pytest.raises(ValueError):
            helper.validate_protocol(value)
        assert calls == []


@pytest.mark.parametrize(
    "change",
    [
        lambda p: p.update(model="another-model"),
        lambda p: p["choices"][0]["message"].update(content={"not": "text"}),
        lambda p: p["choices"][0]["message"].update(tool_calls=[{}]),
        lambda p: p["choices"][0].update(finish_reason="tool_calls"),
        lambda p: p["choices"][0].update(index=False),
        lambda p: p["choices"].append(p["choices"][0]),
        lambda p: p["usage"].update(completion_tokens=9),
    ],
)
def test_response_failures_do_not_become_answers(change):
    helper = module()
    value = response()
    change(value)
    result = helper.interpret(
        {
            "status": 200,
            "body_base64": base64.b64encode(helper.encoded(value)).decode(),
            "error": None,
        },
        {"expected_observed_model": "simulation-model", "exposed_revision": None},
        8,
    )
    assert result == (None, None, None, "invalid_response")


def test_symlinks_traversal_nonregular_and_overwrite_rejected(tmp_path):
    helper = module()
    regular = tmp_path / "ordinary.json"
    helper.write(regular, {})
    linked = tmp_path / "link.json"
    linked.symlink_to(regular)
    with pytest.raises(OSError):
        helper.read(linked)
    with pytest.raises(FileExistsError):
        helper.write(regular, {})
    with pytest.raises(ValueError, match="regular"):
        helper.read(tmp_path)
    with pytest.raises(ValueError, match="traversal"):
        helper.read(tmp_path / ".." / regular.name)
    parent_link = tmp_path / "directory-link"
    parent_link.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        helper.read(parent_link / regular.name)


def test_pin_and_source_drift_rejected_before_calls(tmp_path):
    helper = module()
    value = protocol(helper, "http://127.0.0.1:1/v1/chat/completions")
    path = tmp_path / "protocol.json"
    pin = helper.write(path, value)
    with pytest.raises(ValueError, match="protocol digest"):
        helper.collect(path, helper.digest(b"wrong"), "baseline", tmp_path / "capture")
    value["collector_source_digest"] = helper.digest(b"wrong")
    path2 = tmp_path / "changed.json"
    pin2 = helper.write(path2, value)
    with pytest.raises(ValueError, match="source"):
        helper.collect(path2, pin2, "baseline", tmp_path / "capture")
    with pytest.raises(ValueError, match="token"):
        helper.collect(
            path, pin, "baseline", tmp_path / "capture", "UNSET_SIMULATION_TOKEN"
        )
    assert not (tmp_path / "capture").exists()


def test_credentials_not_persisted_even_when_echoed(tmp_path, monkeypatch):
    helper = module()
    token = "ephemeral-simulation-credential"
    monkeypatch.setenv("SIMULATION_TOKEN", token)
    with server([(200, token.encode())]) as (url, calls):
        declaration = protocol(helper, url, 1)
        path = tmp_path / "protocol.json"
        pin = helper.write(path, declaration)
        destination = tmp_path / "capture"
        with pytest.raises(ValueError, match="missing_token_accounting"):
            helper.collect(path, pin, "baseline", destination, "SIMULATION_TOKEN")
        observation = json.loads((destination / "000000.json").read_bytes())
        assert calls[0]["authorization"] == "Bearer " + token
        assert observation["error"] == "credential_echo"
        assert all(token.encode() not in p.read_bytes() for p in destination.iterdir())


@pytest.mark.parametrize(
    "change",
    [
        lambda c: c.update(role="subject"),
        lambda c: c["observations"].pop(),
        lambda c: c["observations"][0]["request"].update(model="changed"),
        lambda c: c["observations"][0]["request"].update(n=True),
        lambda c: c["observations"][0].update(status=True),
        lambda c: c["observations"][0].update(error="ignored"),
        lambda c: c["observations"][0].update(body_base64=None),
        lambda c: c["observations"][0].update(elapsed_seconds=-1),
        lambda c: c["observations"][0].update(ended_at="2000-01-01T00:00:00Z"),
    ],
)
def test_capture_tampering_rejected(tmp_path, monkeypatch, change):
    helper = module()
    monkeypatch.setattr(
        helper,
        "bounded_request",
        lambda payload: {
            "status": 200,
            "body_base64": base64.b64encode(helper.encoded(response())).decode(),
            "error": None,
        },
    )
    values = collect_fixture(
        tmp_path, helper, protocol(helper, "http://127.0.0.1:1/v1/chat/completions")
    )
    captured = json.loads(values[2].read_bytes())
    change(captured)
    tampered = tmp_path / "changed.json"
    pin = helper.write(tampered, captured)
    with pytest.raises(ValueError):
        helper.export_run(values[0], values[1], tampered, pin, "baseline")


def test_nonfinite_json_worker_failures_and_connection_failure(monkeypatch):
    import subprocess

    helper = module()
    with pytest.raises(ValueError, match="non-finite"):
        helper.decode(b'{"number":NaN}')
    monkeypatch.setattr(
        helper.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(a, 1, b"private error"),
    )
    assert (
        helper.bounded_request({"timeout_seconds": 1})["error"]
        == "collector_worker_error"
    )

    class FailedConnection:
        def __init__(self, *args, **kwargs):
            pass

        def request(self, *args, **kwargs):
            raise OSError("secret-in-exception")

        def close(self):
            pass

    monkeypatch.setattr(helper.http.client, "HTTPConnection", FailedConnection)
    assert helper.http_request(
        {
            "endpoint": "http://127.0.0.1:1/v1/chat/completions",
            "timeout_seconds": 1,
            "token": None,
            "request": {},
        }
    ) == {"status": None, "body_base64": None, "error": "transport_error"}


def test_collect_missing_and_excess_token_usage_stops_schedule(tmp_path, monkeypatch):
    helper = module()
    for index, usage in enumerate((None, {"completion_tokens": 9})):
        data = response()
        data["usage"] = usage
        calls = []

        def request(payload, calls=calls, data=data):
            calls.append(payload)
            return {
                "status": 200,
                "body_base64": base64.b64encode(helper.encoded(data)).decode(),
                "error": None,
            }

        monkeypatch.setattr(helper, "bounded_request", request)
        with pytest.raises(ValueError, match="capture stopped"):
            collect_fixture(
                tmp_path,
                helper,
                protocol(helper, "http://127.0.0.1:1/v1/chat/completions"),
                suffix=str(index),
            )
        assert len(calls) == 1
        retained = json.loads((tmp_path / f"baseline{index}/000000.json").read_bytes())
        assert retained["error"] == (
            "missing_token_accounting"
            if usage is None
            else "output_token_limit_exceeded"
        )
        assert not (tmp_path / f"baseline{index}/capture.json").exists()


def test_main_collect_export_and_worker_dispatch(tmp_path, monkeypatch, capsys):
    import io
    import sys

    helper = module()
    declaration = protocol(helper, "http://127.0.0.1:1/v1/chat/completions", 1)
    path = tmp_path / "declaration.json"
    pin = helper.write(path, declaration)
    monkeypatch.setattr(
        helper,
        "bounded_request",
        lambda p: {
            "status": 200,
            "body_base64": base64.b64encode(helper.encoded(response())).decode(),
            "error": None,
        },
    )
    destination = tmp_path / "capture"
    common = [
        "--protocol",
        str(path),
        "--expected-protocol-sha256",
        pin,
        "--role",
        "baseline",
    ]
    monkeypatch.setattr(
        sys, "argv", ["capture.py", "collect", *common, "--output", str(destination)]
    )
    helper.main()
    capture_pin = capsys.readouterr().out.strip()
    exported = tmp_path / "run.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "capture.py",
            "export",
            *common,
            "--output",
            str(exported),
            "--capture",
            str(destination / "capture.json"),
            "--expected-capture-sha256",
            capture_pin,
        ],
    )
    helper.main()
    assert capsys.readouterr().out.strip() == helper.digest(exported.read_bytes())
    assert json.loads(exported.read_bytes())["artifact_digest"] is None
    monkeypatch.setattr(sys, "argv", ["capture.py", "_request"])
    monkeypatch.setattr(sys, "stdin", io.TextIOWrapper(io.BytesIO(b"{}")))
    result = io.BytesIO()
    monkeypatch.setattr(sys, "stdout", io.TextIOWrapper(result, write_through=True))
    monkeypatch.setattr(helper, "http_request", lambda payload: {"status": 503})
    helper.main()
    assert json.loads(result.getvalue()) == {"status": 503}


def test_whole_window_deadline_limits_last_request_and_preserves_partial_capture(
    tmp_path,
):
    helper = module()
    with server([1.0]) as (url, calls):
        declaration = protocol(helper, url)
        declaration["limits"].update(timeout_seconds=2, max_wall_seconds=0.2)
        before = time.monotonic()
        with pytest.raises(ValueError, match="window deadline"):
            collect_fixture(tmp_path, helper, declaration)
        assert time.monotonic() - before < 0.8
        assert len(calls) == 1
        assert (tmp_path / "baseline/000000.attempt.json").exists()
        retained = json.loads((tmp_path / "baseline/000000.json").read_bytes())
        assert retained["error"] == "deadline_exceeded"
        assert not (tmp_path / "baseline/000001.attempt.json").exists()
        assert not (tmp_path / "baseline/capture.json").exists()


@pytest.mark.parametrize("location", ["content", "key"])
def test_json_escaped_credentials_are_discarded_before_retention(
    tmp_path, monkeypatch, location
):
    helper = module()
    token = "synthetic-escaped-credential"
    monkeypatch.setenv("SIMULATION_TOKEN", token)
    data = response(token if location == "content" else "yes")
    if location == "key":
        data[token] = "echoed key"
    raw = helper.encoded(data).replace(token.encode(), b"\\u0073" + token[1:].encode())
    with server([(200, raw)]) as (url, calls):
        path = tmp_path / "protocol.json"
        pin = helper.write(path, protocol(helper, url, 1))
        output = tmp_path / "capture"
        with pytest.raises(ValueError, match="missing_token_accounting"):
            helper.collect(path, pin, "baseline", output, "SIMULATION_TOKEN")
        assert len(calls) == 1
        observation = json.loads((output / "000000.json").read_bytes())
        assert observation["error"] == "credential_echo"
        assert observation["body_base64"] is None
        assert not (output / "capture.json").exists()


@pytest.mark.parametrize("raw", [b"not JSON", b'{"a":1,"a":2}', b'{"x":NaN}'])
def test_credentials_require_inspectable_json_responses(tmp_path, monkeypatch, raw):
    helper = module()
    monkeypatch.setenv("SIMULATION_TOKEN", "synthetic-credential")
    with server([(503, raw)]) as (url, calls):
        path = tmp_path / "protocol.json"
        pin = helper.write(path, protocol(helper, url, 1))
        output = tmp_path / "capture"
        capture_pin = helper.collect(path, pin, "baseline", output, "SIMULATION_TOKEN")
        run = helper.export_run(
            path, pin, output / "capture.json", capture_pin, "baseline"
        )
        row = run["records"][0]
        assert len(calls) == 1
        assert row["error"] == "uninspectable_response"
        assert row["context"]["http_observation"]["body_base64"] is None


@pytest.mark.parametrize("location", ["input", "key", "invalid_policy"])
def test_protocol_credentials_are_checked_as_decoded_strings(
    tmp_path, monkeypatch, location
):
    helper = module()
    token = 'synthetic-quoted-"credential'
    monkeypatch.setenv("SIMULATION_TOKEN", token)
    declaration = protocol(helper, "http://127.0.0.1:1/v1/chat/completions", 1)
    if location == "input":
        declaration["cases"][0]["input"] = token
    elif location == "key":
        declaration["environment"][token] = "value"
    else:
        declaration["policy"] = {"invalid": token}
    assert token.encode() not in helper.encoded(declaration)
    path = tmp_path / "protocol.json"
    pin = helper.write(path, declaration)
    monkeypatch.setattr(helper, "bounded_request", lambda _: pytest.fail("made a call"))
    with pytest.raises(ValueError, match="credential occurs in protocol"):
        helper.collect(path, pin, "baseline", tmp_path / "capture", "SIMULATION_TOKEN")
    assert not (tmp_path / "capture").exists()


@pytest.mark.parametrize("field", ["model", "revision", "content"])
def test_unencodable_returned_text_is_a_response_error(field):
    helper = module()
    data = response()
    if field == "content":
        data["choices"][0]["message"]["content"] = "\ud800"
    else:
        data[field] = "\ud800"
    result = helper.interpret(
        {
            "status": 200,
            "body_base64": base64.b64encode(json.dumps(data).encode()).decode(),
            "error": None,
        },
        {"expected_observed_model": None, "exposed_revision": None},
        8,
    )
    assert result == (None, None, None, "invalid_response")


@pytest.mark.parametrize(
    "field",
    [
        "provider",
        "service",
        "deployment",
        "model",
        "expected_observed_model",
        "exposed_revision",
    ],
)
@pytest.mark.parametrize(
    "invalid",
    ["p" * 513, " \t", "bad\x00identity", "bad\x7fidentity"],
    ids=["oversized", "blank", "control", "delete"],
)
def test_unexportable_service_identity_stops_before_calls(
    tmp_path, monkeypatch, field, invalid
):
    helper = module()
    declaration = protocol(helper, "http://127.0.0.1:1/v1/chat/completions", 1)
    declaration["services"]["baseline"][field] = invalid
    path = tmp_path / "protocol.json"
    pin = helper.write(path, declaration)
    monkeypatch.setattr(helper, "bounded_request", lambda _: pytest.fail("made a call"))
    with pytest.raises(ValueError, match="identity"):
        helper.collect(path, pin, "baseline", tmp_path / "capture")
    assert not (tmp_path / "capture").exists()


@pytest.mark.parametrize("field", ["name", "version"])
def test_unexportable_harness_identity_rejected(field):
    helper = module()
    declaration = protocol(helper, "http://127.0.0.1:1/v1/chat/completions", 1)
    declaration["harness"][field] = "x" * 129
    with pytest.raises(ValueError, match="harness identity"):
        helper.validate_protocol(declaration)


@pytest.mark.parametrize(
    "invalid", ["x" * 129, " ", "bad\x00case"], ids=["oversized", "blank", "control"]
)
def test_unexportable_case_id_stops_before_calls(tmp_path, monkeypatch, invalid):
    helper = module()
    declaration = protocol(helper, "http://127.0.0.1:1/v1/chat/completions", 1)
    declaration["cases"][0]["id"] = invalid
    path = tmp_path / "protocol.json"
    pin = helper.write(path, declaration)
    monkeypatch.setattr(helper, "bounded_request", lambda _: pytest.fail("made a call"))
    with pytest.raises(ValueError, match="case ID"):
        helper.collect(path, pin, "baseline", tmp_path / "capture")
    assert not (tmp_path / "capture").exists()


@pytest.mark.parametrize(
    "change",
    [
        lambda p: p.update(policy={}),
        lambda p: p["policy"]["metrics"][0].update(configuration={"unsupported": True}),
        lambda p: p["policy"]["metrics"][0].update(kind="normalized_match"),
        lambda p: p["policy"].update(expected_case_set_digest="sha256:" + "0" * 64),
    ],
)
def test_unusable_policy_stops_before_calls(tmp_path, monkeypatch, change):
    helper = module()
    declaration = protocol(helper, "http://127.0.0.1:1/v1/chat/completions", 1)
    change(declaration)
    path = tmp_path / "protocol.json"
    pin = helper.write(path, declaration)
    monkeypatch.setattr(helper, "bounded_request", lambda _: pytest.fail("made a call"))
    with pytest.raises(ValueError):
        helper.collect(path, pin, "baseline", tmp_path / "capture")
    assert not (tmp_path / "capture").exists()


def test_planned_case_pin_uses_same_metadata_projection_as_export(
    tmp_path, monkeypatch
):
    from invarlock.engine import case_set_digest, freeze_case_set, validate_run_case_set

    helper = module()
    declaration = protocol(helper, "http://127.0.0.1:1/v1/chat/completions", 1)
    declaration["cases"][0]["id"] = "x" * 128
    declaration["harness"]["name"] = "h" * 128
    planned = case_set_digest(
        freeze_case_set([{**case, "metadata": {}} for case in declaration["cases"]])
    )
    declaration["policy"]["expected_case_set_digest"] = planned
    monkeypatch.setattr(
        helper,
        "bounded_request",
        lambda _: {
            "status": 200,
            "body_base64": base64.b64encode(helper.encoded(response())).decode(),
            "error": None,
        },
    )
    values = collect_fixture(tmp_path, helper, declaration)
    run = helper.export_run(*values, "baseline")
    validate_run_case_set(run, planned)


@pytest.mark.parametrize("bound", ["repeated-system-prompt", "responses-plus-records"])
def test_unexportable_byte_budgets_stop_before_calls(tmp_path, monkeypatch, bound):
    helper = module()
    count = 1000 if bound == "repeated-system-prompt" else 16
    declaration = protocol(helper, "http://127.0.0.1:1/v1/chat/completions", count)
    if bound == "repeated-system-prompt":
        declaration["configuration"]["system_prompt"] = "\x01" * 16384
    else:
        declaration["limits"]["max_response_bytes"] = 1024 * 1024
        for case in declaration["cases"]:
            case["input"] = "\x01" * 65536
    assert len(helper.encoded(declaration)) < helper.MAX_FILE_BYTES // 4
    path = tmp_path / "protocol.json"
    pin = helper.write(path, declaration)
    monkeypatch.setattr(helper, "bounded_request", lambda _: pytest.fail("made a call"))
    with pytest.raises(ValueError, match="export byte budget"):
        helper.collect(path, pin, "baseline", tmp_path / "capture")
    assert not (tmp_path / "capture").exists()


def test_identity_limit_and_unicode_are_compatible_with_canonical_export(
    tmp_path, monkeypatch
):
    helper = module()
    declaration = protocol(helper, "http://127.0.0.1:1/v1/chat/completions", 1)
    declaration["services"]["baseline"]["provider"] = "\u03bb" * 512
    monkeypatch.setattr(
        helper,
        "bounded_request",
        lambda _: {
            "status": 200,
            "body_base64": base64.b64encode(helper.encoded(response())).decode(),
            "error": None,
        },
    )
    values = collect_fixture(tmp_path, helper, declaration)
    run = helper.export_run(*values, "baseline")
    assert run["service_identity"]["provider"] == "\u03bb" * 512


@pytest.mark.parametrize(
    "raw,token,expected",
    [
        (b"not JSON", None, None),
        (b"synthetic-token", "synthetic-token", "credential_echo"),
        (b'{"a":[null,0,false,"public"]}', "synthetic-token", None),
        (
            b'{"a":[{"\\u0073ynthetic-token":"public"}]}',
            "synthetic-token",
            "credential_echo",
        ),
        (b'{"a":["\\u0073ynthetic-token"]}', "synthetic-token", "credential_echo"),
        (b'{"a":1,"a":2}', "synthetic-token", "uninspectable_response"),
        (b"[", "synthetic-token", "uninspectable_response"),
    ],
)
def test_credential_inspection_handles_nested_json_and_malformed_bytes(
    raw, token, expected
):
    assert module().credential_error(raw, token) == expected


@pytest.mark.parametrize(
    "expected,returned,error",
    [
        (None, None, None),
        (None, "returned-r1", None),
        ("expected-r1", "expected-r1", None),
        ("expected-r1", None, "invalid_response"),
        ("expected-r1", "different-r2", "invalid_response"),
        (None, "bad\x00revision", "invalid_response"),
    ],
)
def test_revision_is_observed_and_expected_revision_is_checked(
    tmp_path, monkeypatch, expected, returned, error
):
    helper = module()
    declaration = protocol(helper, "http://127.0.0.1:1/v1/chat/completions", 1)
    declaration["services"]["baseline"]["exposed_revision"] = expected
    data = response()
    if returned is not None:
        data["revision"] = returned
    monkeypatch.setattr(
        helper,
        "bounded_request",
        lambda _: {
            "status": 200,
            "body_base64": base64.b64encode(helper.encoded(data)).decode(),
            "error": None,
        },
    )
    values = collect_fixture(tmp_path, helper, declaration)
    run = helper.export_run(*values, "baseline")
    assert run["service_identity"]["exposed_revision"] == (
        returned if error is None else None
    )
    assert run["records"][0]["error"] == error


def test_returned_revision_changes_are_rejected(tmp_path, monkeypatch):
    helper = module()
    results = iter(("revision-1", "revision-2"))
    monkeypatch.setattr(
        helper,
        "bounded_request",
        lambda _: {
            "status": 200,
            "body_base64": base64.b64encode(
                helper.encoded({**response(), "revision": next(results)})
            ).decode(),
            "error": None,
        },
    )
    values = collect_fixture(
        tmp_path, helper, protocol(helper, "http://127.0.0.1:1/v1/chat/completions")
    )
    with pytest.raises(ValueError, match="revision changed"):
        helper.export_run(*values, "baseline")


@pytest.mark.parametrize(
    "change,message",
    [
        (
            lambda c: c["observation_window"].update(ended_at="2026-01-01T00:00:11Z"),
            "window duration",
        ),
        (
            lambda c: [o.update(elapsed_seconds=6) for o in c["observations"]],
            "duration budget",
        ),
        (
            lambda c: c["observations"][0].update(elapsed_seconds=2.1),
            "request duration",
        ),
    ],
)
def test_replay_rejects_timings_outside_declared_budgets(
    tmp_path, monkeypatch, change, message
):
    helper = module()
    monkeypatch.setattr(
        helper,
        "bounded_request",
        lambda _: {
            "status": 200,
            "body_base64": base64.b64encode(helper.encoded(response())).decode(),
            "error": None,
        },
    )
    values = collect_fixture(
        tmp_path, helper, protocol(helper, "http://127.0.0.1:1/v1/chat/completions")
    )
    capture = json.loads(values[2].read_bytes())
    capture["observation_window"] = {
        "started_at": "2026-01-01T00:00:00Z",
        "ended_at": "2026-01-01T00:00:02Z",
    }
    for index, observation in enumerate(capture["observations"]):
        observation.update(
            started_at=f"2026-01-01T00:00:0{index}Z",
            ended_at=f"2026-01-01T00:00:0{index + 1}Z",
            elapsed_seconds=1,
        )
    change(capture)
    changed = tmp_path / "changed.json"
    pin = helper.write(changed, capture)
    with pytest.raises(ValueError, match=message):
        helper.export_run(values[0], values[1], changed, pin, "baseline")


@pytest.mark.parametrize("cleanup,accepted", [(0.5, True), (1.1, False)])
def test_deadline_attempt_has_explicit_bounded_cleanup_tolerance(
    tmp_path, monkeypatch, cleanup, accepted
):
    helper = module()
    monkeypatch.setattr(
        helper,
        "bounded_request",
        lambda _: {"status": None, "body_base64": None, "error": "deadline_exceeded"},
    )
    values = collect_fixture(
        tmp_path, helper, protocol(helper, "http://127.0.0.1:1/v1/chat/completions", 1)
    )
    capture = json.loads(values[2].read_bytes())
    capture["observation_window"] = {
        "started_at": "2026-01-01T00:00:00Z",
        "ended_at": "2026-01-01T00:00:04Z",
    }
    capture["observations"][0].update(
        **capture["observation_window"], elapsed_seconds=2 + cleanup
    )
    changed = tmp_path / "changed.json"
    pin = helper.write(changed, capture)
    if accepted:
        run = helper.export_run(values[0], values[1], changed, pin, "baseline")
        assert run["records"][0]["error"] == "deadline_exceeded"
    else:
        with pytest.raises(ValueError, match="request duration"):
            helper.export_run(values[0], values[1], changed, pin, "baseline")
