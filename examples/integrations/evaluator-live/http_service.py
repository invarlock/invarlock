"""Bounded real loopback HTTP capture over independently retained model tasks.

The service exposes its observed worker identity and reference likelihood. It is
an example controlled deployment, not an opaque external-provider attestation.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import http.client
import signal
import time
from copy import deepcopy
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import common

PRIORITY = {"inspect-ai", "lm-evaluation-harness", "promptfoo", "langfuse"}
PATH = "/v1/tasks"
TIMEOUT = 180
MAX_CAPTURE_BYTES = 128 * 1024 * 1024


def sha(raw):
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def timestamp():
    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def endpoint(value):
    parsed = urlsplit(value)
    if (
        parsed.scheme != "http"
        or parsed.hostname != "127.0.0.1"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path != PATH
        or parsed.port is None
        or not 1024 <= parsed.port <= 65535
        or value != f"http://127.0.0.1:{parsed.port}{PATH}"
    ):
        raise ValueError("HTTP capture requires an exact literal loopback endpoint")
    return "127.0.0.1", parsed.port


def service(protocol, role):
    value = protocol["http_services"][role]
    if not isinstance(value, dict) or set(value) != {
        "provider",
        "service",
        "deployment",
        "endpoint",
        "requested_model",
        "helper_sha256",
    }:
        raise ValueError("HTTP service declaration has unexpected fields")
    endpoint(value["endpoint"])
    if any(
        not isinstance(v, str)
        or not v.strip()
        or len(v) > 512
        or any(ord(char) < 32 or ord(char) == 127 for char in v)
        for v in value.values()
    ):
        raise ValueError("HTTP service declaration contains invalid values")
    if value["helper_sha256"] != sha(Path(__file__).read_bytes()):
        raise ValueError("HTTP helper differs from its independently frozen source")
    if not set(protocol["evaluators"]) <= PRIORITY:
        raise ValueError("HTTP qualification is limited to its four declared SDKs")
    return value


def body(protocol, role, request):
    planned = {row["id"]: row for row in common.cases(protocol["cases"])}
    if (
        not isinstance(request, dict)
        or set(request) != {"evaluator", "case_id", "protocol_digest"}
        or request["evaluator"] not in protocol["evaluators"]
        or request["case_id"] not in planned
        or request["protocol_digest"] != common.digest({**protocol, "role": role})
    ):
        raise ValueError("HTTP request is outside the frozen schedule")
    case = planned[request["case_id"]]
    return {
        "request": request,
        "input": case["input"],
        "reference": case["expected"],
        "configuration": protocol["configuration"],
        "model": protocol["http_services"][role]["requested_model"],
    }


def checked_response(value, request, protocol, role):
    if not isinstance(value, dict) or set(value) != {
        "request",
        "result",
        "observed_model",
        "exposed_revision",
        "configuration",
    }:
        raise ValueError("HTTP response fields differ from the declared service")
    response = {key: value[key] for key in ("request", "result")}
    common.TaskClient.validate_response(response, request)
    execution = value["result"]["metadata"].get("invarlock_model_execution", {})
    model = protocol["models"][role]
    if (
        common.encoded(execution.get("model")) != common.encoded(model)
        or common.encoded(execution.get("configuration"))
        != common.encoded(protocol["configuration"])
        or execution.get("request") != request
        or execution.get("protocol_digest") != request["protocol_digest"]
        or value["observed_model"] != model["id"]
        or value["exposed_revision"] != model["revision"]
        or common.encoded(value["configuration"])
        != common.encoded(execution["configuration"])
    ):
        raise ValueError("HTTP observation differs from its actual worker execution")
    return response


class TaskClient(common.TaskClient):
    """Keep original HTTP request/response bodies alongside normal task journals."""

    def __init__(
        self, path, evaluator, protocol_digest, planned, output, *, protocol, role
    ):
        self.protocol, self.role = protocol, role
        self.service = service(protocol, role)
        if path != self.service["endpoint"]:
            raise ValueError("HTTP endpoint differs from the admitted deployment")
        self.host, self.port = endpoint(path)
        super().__init__(path, evaluator, protocol_digest, planned, output)
        self.http = self.output.parent / "http"
        self.http.mkdir(mode=0o700, exist_ok=False)

    def exchange(self, request):
        stem = common.digest(request).removeprefix("sha256:")
        raw_request = common.encoded(body(self.protocol, self.role, request))
        if len(raw_request) > common.MAX_MESSAGE:
            raise ValueError("HTTP request exceeds the byte cap")
        admission = {
            "endpoint": self.path,
            "method": "POST",
            "started_at": timestamp(),
            "request_body_base64": base64.b64encode(raw_request).decode(),
        }
        common.write(self.http / f"{stem}.request.json", admission)
        connection = http.client.HTTPConnection(self.host, self.port, timeout=TIMEOUT)
        try:
            connection.request(
                "POST",
                PATH,
                body=raw_request,
                headers={"Content-Type": "application/json"},
            )
            response = connection.getresponse()
            raw_response = response.read(common.MAX_MESSAGE + 1)
            retained = {
                "ended_at": timestamp(),
                "status": response.status,
                "response_body_base64": base64.b64encode(raw_response).decode(),
            }
            common.write(self.http / f"{stem}.response.json", retained)
        finally:
            connection.close()
        if response.status != 200 or len(raw_response) > common.MAX_MESSAGE:
            raise ValueError("HTTP service failed or exceeded the response cap")
        parsed = common.decode(raw_response)
        return common.encoded(
            checked_response(parsed, request, self.protocol, self.role)
        )

    def identity(self):
        self.complete()
        starts, ends = [], []
        for path in self.http.glob("*.request.json"):
            starts.append(common.read(path)["started_at"])
            ends.append(
                common.read(
                    path.with_name(path.name.replace(".request.", ".response."))
                )["ended_at"]
            )
        return identity(self.protocol, self.role, min(starts), max(ends))


def identity(protocol, role, started, ended):
    declared = service(protocol, role)
    return {
        "kind": "hosted_service",
        **{
            key: declared[key]
            for key in ("provider", "service", "deployment", "requested_model")
        },
        "observed_model": protocol["models"][role]["id"],
        "exposed_revision": protocol["models"][role]["revision"],
        "configuration": deepcopy(protocol["configuration"]),
        "configuration_digest": common.digest(protocol["configuration"]),
        "harness": {
            "name": "live-loopback-http",
            "version": "1",
            "source_digest": declared["helper_sha256"],
        },
        "observation_window": {"started_at": started, "ended_at": ended},
    }


def hosted_facts(facts, descriptor):
    return {
        **facts,
        "artifact_digest": None,
        "service_identity_digest": common.digest(descriptor),
        "configuration_digest": descriptor["configuration_digest"],
    }


def bind_native(native, descriptor):
    """Derive a service-bound export, preserving the exact SDK export separately."""
    if isinstance(native, list):
        return [bind_native(value, descriptor) for value in native]
    if not isinstance(native, dict):
        return native
    if "invarlock_http_binding" in native:
        raise ValueError("SDK cannot supply the HTTP derivation binding")
    value = {key: bind_native(child, descriptor) for key, child in native.items()}
    if "invarlock_likelihood" in native:
        original = native["invarlock_likelihood"]
        value["invarlock_likelihood"] = hosted_facts(original, descriptor)
        value["invarlock_http_binding"] = {
            "format": "invarlock/live-http-likelihood-v1",
            "original_likelihood": deepcopy(original),
            "service_identity": deepcopy(descriptor),
        }
    return value


def verify_capture(
    directory, manifest, protocol, role, evaluator, results, originals, read
):
    """Recompute every HTTP observation and final-window derivation offline."""
    directory = Path(directory)
    descriptor = manifest.get("service_identity")
    raw_native, original_raw = read(
        directory / "native-original.json", MAX_CAPTURE_BYTES
    )
    if manifest.get("native_original_sha256") != sha(original_raw):
        raise ValueError("original SDK export differs from its retained byte digest")
    originals["native-original.json"] = original_raw
    derived = bind_native(raw_native, descriptor)
    if common.encoded(derived) != originals["native.json"]:
        raise ValueError("HTTP native derivation differs from the original SDK export")
    expected_files, starts, ends = set(), [], []
    total_bytes = 0
    for case in protocol["cases"]:
        request = {
            "evaluator": evaluator,
            "case_id": case["id"],
            "protocol_digest": common.digest({**protocol, "role": role}),
        }
        stem = common.digest(request).removeprefix("sha256:")
        values = []
        for suffix in ("request", "response"):
            name = f"{stem}.{suffix}.json"
            expected_files.add(name)
            value, raw = read(directory / "http" / name, 2 * common.MAX_MESSAGE)
            originals["http/" + name] = raw
            total_bytes += len(raw)
            if total_bytes > MAX_CAPTURE_BYTES:
                raise ValueError("HTTP journal exceeds the capture byte limit")
            values.append(value)
        sent, received = values
        if set(sent) != {
            "endpoint",
            "method",
            "started_at",
            "request_body_base64",
        } or set(received) != {"ended_at", "status", "response_body_base64"}:
            raise ValueError("HTTP journal fields differ")
        actual_request = base64.b64decode(sent["request_body_base64"], validate=True)
        actual_response = base64.b64decode(
            received["response_body_base64"], validate=True
        )
        if (
            sent["endpoint"] != service(protocol, role)["endpoint"]
            or sent["method"] != "POST"
            or received["status"] != 200
            or actual_request != common.encoded(body(protocol, role, request))
            or len(actual_response) > common.MAX_MESSAGE
        ):
            raise ValueError("HTTP journal differs from its admitted request")
        parsed = checked_response(
            common.decode(actual_response), request, protocol, role
        )
        if common.encoded(parsed["result"]) != common.encoded(results[case["id"]]):
            raise ValueError("HTTP response differs from the original task result")
        start, end = sent["started_at"], received["ended_at"]
        if (
            not isinstance(start, str)
            or not isinstance(end, str)
            or not start.endswith("Z")
            or not end.endswith("Z")
            or datetime.fromisoformat(start) > datetime.fromisoformat(end)
        ):
            raise ValueError("HTTP observation window is invalid")
        starts.append(start)
        ends.append(end)
    if {p.name for p in (directory / "http").iterdir()} != expected_files:
        raise ValueError("HTTP journal differs from the complete schedule")
    if descriptor != identity(protocol, role, min(starts), max(ends)):
        raise ValueError("HTTP service descriptor differs from observed capture window")
    return descriptor


def make_server(protocol, role, socket_path, output):
    """Create one bounded single-request-at-a-time loopback deployment."""
    declared = service(protocol, role)
    output = Path(output)
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    clients = {
        name: common.TaskClient(
            socket_path,
            name,
            common.digest({**protocol, "role": role}),
            protocol["cases"],
            output / name,
        )
        for name in protocol["evaluators"]
    }
    planned = {row["id"]: row for row in protocol["cases"]}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def setup(self):
            super().setup()
            self.connection.settimeout(TIMEOUT)

        def do_POST(self):
            try:
                lengths = self.headers.get_all("Content-Length", [])
                if (
                    self.path != PATH
                    or self.headers.get("Transfer-Encoding")
                    or len(lengths) != 1
                    or not lengths[0].isdigit()
                    or not 0 < int(lengths[0]) <= common.MAX_MESSAGE
                    or self.headers.get("Content-Type") != "application/json"
                ):
                    raise ValueError(
                        "HTTP request framing is outside the declared profile"
                    )
                raw = self.rfile.read(int(lengths[0]))
                value = common.decode(raw)
                request = value["request"]
                if raw != common.encoded(body(protocol, role, request)):
                    raise ValueError("HTTP prompt, reference or configuration differs")
                result = clients[request["evaluator"]](planned[request["case_id"]])
                execution = result["metadata"]["invarlock_model_execution"]
                response = {
                    "request": request,
                    "result": result,
                    "observed_model": execution["model"]["id"],
                    "exposed_revision": execution["model"]["revision"],
                    "configuration": execution["configuration"],
                }
                checked_response(response, request, protocol, role)
                status = 200
            except (ValueError, KeyError, TypeError, OSError) as exc:
                status, response = 400, {"error": type(exc).__name__}
            raw = common.encoded(response)
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            try:
                self.wfile.write(raw)
            except (BrokenPipeError, ConnectionResetError):
                pass

    server = HTTPServer(endpoint(declared["endpoint"]), Handler)
    server.timeout = 1
    return server, clients


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--role", choices=("baseline", "subject"), required=True)
    parser.add_argument("--socket", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    protocol = common.read(args.protocol)
    if common.digest(protocol) != args.protocol_sha256:
        raise ValueError("HTTP protocol differs from its independent pin")
    seconds = protocol["limits"]["max_seconds"]
    if type(seconds) is not int or not 1 <= seconds <= 86400:
        raise ValueError("HTTP deployment requires a bounded whole-process deadline")
    server, clients = make_server(protocol, args.role, args.socket, args.output)
    deadline = time.monotonic() + seconds

    def expired(*_):
        # BaseException bypasses HTTPServer's per-request error recovery. A slow
        # header/body or blocked worker exchange must end the whole deployment.
        raise SystemExit("HTTP deployment exceeded its admitted window")

    previous = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        with server:
            common.write(
                args.output / "ready.json",
                {"endpoint": protocol["http_services"][args.role]["endpoint"]},
            )
            while sum(len(c.results) for c in clients.values()) < len(
                protocol["cases"]
            ) * len(clients):
                if time.monotonic() >= deadline:
                    raise TimeoutError("HTTP deployment exceeded its admitted window")
                server.handle_request()
            common.write(
                args.output / "complete.json",
                {
                    "status": "complete",
                    "requests": sum(len(c.results) for c in clients.values()),
                },
            )
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


if __name__ == "__main__":
    main()
