"""Bounded local transport for fresh model tasks in evaluator environments."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import socket
import threading
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
MAX_MESSAGE = 2 * 1024 * 1024


def encoded(value):
    def check(item):
        if isinstance(item, dict):
            if any(not isinstance(key, str) for key in item):
                raise ValueError("JSON object keys must be strings")
            for child in item.values():
                check(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                check(child)

    check(value)
    return (
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode()


def digest(value):
    return "sha256:" + hashlib.sha256(encoded(value)).hexdigest()


def decode(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON field")
            result[key] = value
        return result

    value = json.loads(raw, object_pairs_hook=pairs)
    encoded(value)
    return value


def read(path, limit=MAX_MESSAGE):
    with Path(path).open("rb") as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ValueError("input exceeds the declared byte limit")
    return decode(raw)


def write(path, value):
    with Path(path).open("xb") as stream:
        stream.write(encoded(value))


def module(name):
    spec = importlib.util.spec_from_file_location("live_" + name, HERE / (name + ".py"))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def versions():
    value = read(ROOT / "examples/evaluator-qualification/matrix.json")
    return {p["profile_id"]: p["upstream"]["version"] for p in value["profiles"]}


def cases(value):
    if not isinstance(value, list) or not 1 <= len(value) <= 2000:
        raise ValueError("campaign requires 1 to 2000 declared cases")
    seen = set()
    for row in value:
        if not isinstance(row, dict) or set(row) != {
            "id",
            "input",
            "expected",
            "metadata",
        }:
            raise ValueError("case fields differ from the declared profile")
        if (
            not isinstance(row["id"], str)
            or not row["id"]
            or row["id"] in seen
            or not isinstance(row["input"], str)
            or not row["input"]
            or not isinstance(row["expected"], str)
            or not row["expected"]
            or not isinstance(row["metadata"], dict)
        ):
            raise ValueError("case IDs, task text and references must be complete")
        if any(str(key).startswith("invarlock_") for key in row["metadata"]):
            raise ValueError("case metadata cannot supply reserved capture fields")
        seen.add(row["id"])
    encoded(value)
    return value


class TaskClient:
    """Call a separately loaded model using only an admitted frozen case ID."""

    def __init__(self, path, evaluator, protocol_digest, planned, output):
        self.path = str(path)
        self.evaluator = evaluator
        self.protocol_digest = protocol_digest
        self.planned = {row["id"]: deepcopy(row) for row in cases(planned)}
        self.output = Path(output)
        self.output.mkdir(mode=0o700, parents=True, exist_ok=False)
        self.results = {}
        self._lock = threading.Lock()

    def __call__(self, case):
        # SDKs may invoke cases concurrently; the model worker admits one at a time.
        with self._lock:
            return self._call(case)

    def _call(self, case):
        case_id = case.get("id")
        if case_id not in self.planned or encoded(case) != encoded(
            self.planned[case_id]
        ):
            raise ValueError("SDK task differs from its independently frozen case")
        if case_id in self.results:
            raise ValueError("SDK repeated a model task outside the declared schedule")
        request = {
            "evaluator": self.evaluator,
            "case_id": case_id,
            "protocol_digest": self.protocol_digest,
        }
        request_id = digest(request).removeprefix("sha256:")
        # Admission precedes the call. An interrupted attempt is not silently retried.
        write(self.output / (request_id + ".request.json"), request)
        raw = self.exchange(request)
        if not raw.endswith(b"\n") or len(raw) > MAX_MESSAGE:
            raise ValueError("model task returned an incomplete or oversized response")
        result = decode(raw)
        self.validate_response(result, request)
        row = result["result"]
        write(self.output / (request_id + ".response.json"), result)
        self.results[case_id] = deepcopy(row)
        return deepcopy(row)

    def exchange(self, request):
        """Exchange one request; alternate example transports retain this admission."""
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as channel:
            channel.settimeout(180)
            channel.connect(self.path)
            channel.sendall(encoded(request))
            with channel.makefile("rb") as stream:
                raw = stream.readline(MAX_MESSAGE + 1)
        return raw

    @staticmethod
    def validate_response(result, request):
        if (
            not isinstance(result, dict)
            or set(result) != {"request", "result"}
            or result["request"] != request
        ):
            raise ValueError("model task response does not match the admitted request")
        row = result["result"]
        if (
            not isinstance(row, dict)
            or not {"output", "metadata"} <= set(row)
            or set(row) - {"output", "metadata", "error"}
            or not isinstance(row.get("metadata"), dict)
            or (
                "error" in row
                and (not isinstance(row["error"], str) or not row["error"])
            )
            or (row.get("output") is not None and not isinstance(row["output"], str))
            or (
                row.get("output") is None
                and not row.get("error")
                and "invarlock_likelihood" not in row["metadata"]
            )
        ):
            raise ValueError("model task returned invalid observations")

    def complete(self):
        with self._lock:
            if set(self.results) != set(self.planned):
                raise ValueError("SDK omitted planned model tasks")
            return deepcopy(self.results)
