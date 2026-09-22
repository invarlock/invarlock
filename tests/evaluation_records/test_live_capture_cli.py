"""Production SDK CLI roundtrip with a synthetic task server; no model is called."""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
from contextlib import contextmanager, nullcontext
from pathlib import Path
from unittest.mock import patch

import pytest

HERE = Path(__file__).resolve().parents[2] / "examples/integrations/evaluator-live"
SPEC = importlib.util.spec_from_file_location("live_cli_common", HERE / "common.py")
COMMON = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COMMON)
PINS = COMMON.versions()
HTTP_CAPABILITY = "a" * 64

SDK_CLI = r"""
import importlib.abc, runpy, socket, sys
attempts = []
class NoRecipient(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "invarlock" or fullname.startswith("invarlock."):
            attempts.append(fullname)
            raise AssertionError("SDK capture must not import the recipient package")
sys.meta_path.insert(0, NoRecipient())
script = sys.argv[1]
sys.path.insert(0, str(__import__("pathlib").Path(script).parent))
sys.argv = sys.argv[1:]
runpy.run_path(script, run_name="__main__")
assert not attempts, attempts
assert not any(name == "invarlock" or name.startswith("invarlock.") for name in sys.modules)
# Emit only an audit event, never an actual outbound socket operation. This
# demonstrates that the production network guard remains active after the SDK.
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as channel:
    try:
        sys.audit("socket.connect", channel, ("192.0.2.1", 9))
    except RuntimeError as exc:
        assert "local callback transport" in str(exc)
    else:
        raise AssertionError("production SDK network guard was not installed")
"""

RECIPIENT = r"""
import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import common, bindings
recipient = common.module("recipient")
common.module("network").configure("judge-recipient")
identity = recipient.installed_identity()
from invarlock.engine import export_evaluator_result, load_run
capture = Path(sys.argv[2])
protocol = common.read(capture / "protocol.json")
manifest = common.read(capture / "capture.json")
evaluator = manifest["evaluator"]
payload = common.read(capture / "native.json", limit=128*1024*1024)
service_identity = manifest.get("service_identity")
if service_identity is not None:
    frozen_protocol = {key:value for key,value in protocol.items() if key != "role"}
    recipient.capture(capture, frozen_protocol, "baseline", evaluator)
run = load_run(capture / "native.json", adapter="evaluator-native-json",
    source={"name":evaluator,"version":manifest["version"]},
    run_id=payload["run_name"] if evaluator == "langfuse" else "synthetic-cli-contract",
    artifact_digest=None if service_identity else protocol["models"]["baseline"]["artifact_digest"],
    service_identity=service_identity,
    input_projection=bindings.projection(evaluator))
case = protocol["cases"][0]
response = common.read(next((capture / "tasks").glob("*.response.json")))
record = run["records"][0]
bindings.check_record(record,response["result"],case,evaluator,manifest["version"],
    service_identity=service_identity)
bound = bindings.bind_result(response["result"],case,evaluator,manifest["version"])
for name in ("invarlock_model_execution","invarlock_capture_binding"):
    assert recipient.contains(record,name,bound["metadata"][name])
routes = ["evaluator-native-json"]
if service_identity is not None:
    envelope = capture.parent / "recipient-envelope.json"
    native = {"format":"invarlock/langfuse-export-v1", "sdk_version":manifest["version"],
        "result":payload} if evaluator == "langfuse" else payload
    options = dict(run_id=run["run_id"], artifact_digest=None,
        service_identity=service_identity, input_projection=bindings.projection(evaluator))
    exported = export_evaluator_result(evaluator,native,envelope,
        expected_ids=[case["id"]],source_version=manifest["version"],**options)
    imported = load_run(envelope,adapter="evaluator-json",
        source={"name":evaluator,"version":manifest["version"]},**options)
    assert imported == exported
    assert imported["records"] == run["records"]
    assert imported["service_identity"] == run["service_identity"]
    routes.append("evaluator-json")
print(json.dumps({"run":run,"recipient":identity,"routes":routes}))
"""


def _selected(evaluator):
    selected = os.environ.get("INVARLOCK_LIVE_EVALUATOR")
    if selected is None:
        pytest.skip("select a pinned SDK environment with INVARLOCK_LIVE_EVALUATOR")
    if selected not in PINS:
        pytest.fail("selected live evaluator is not a maintained profile")
    if selected != evaluator:
        pytest.skip("another isolated evaluator environment is selected")
    recipient = os.environ.get("INVARLOCK_EVALUATOR_PARITY_PYTHON")
    if not recipient or not Path(recipient).is_file():
        pytest.fail(
            "selected live CLI test requires a separately installed recipient Python"
        )
    package = {
        "lm-evaluation-harness": "lm-eval",
        "hugging-face-evaluate": "evaluate",
        "openai-evals": "evals",
    }.get(evaluator, evaluator)
    try:
        actual = (
            COMMON.read(
                Path(os.environ["INVARLOCK_PROMPTFOO_PACKAGE"]) / "package.json"
            )["version"]
            if evaluator == "promptfoo"
            else importlib.metadata.version(package)
        )
    except (KeyError, OSError, importlib.metadata.PackageNotFoundError) as exc:
        pytest.fail(f"selected pinned evaluator is unavailable: {exc}")
    assert actual == PINS[evaluator], (package, actual, PINS[evaluator])
    return recipient


def _environment(directory):
    # No inherited API credentials are necessary for the synthetic local callback.
    environment = {
        key: value
        for key, value in os.environ.items()
        if key
        not in {
            "PYTHONPATH",
            "INVARLOCK_SIGNING_KEY",
            "HF_TOKEN",
            "HUGGING_FACE_HUB_TOKEN",
            "LANGFUSE_SECRET_KEY",
            "AWS_SECRET_ACCESS_KEY",
        }
        and not key.endswith(("_API_KEY", "_ACCESS_TOKEN"))
    }
    environment["XDG_DATA_HOME"] = str(directory / "data")
    environment["XDG_CACHE_HOME"] = str(directory / "cache")
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return environment


@contextmanager
def _http_gateway(protocol, address, directory):
    from invarlock.security import temporarily_allow_network

    with patch.dict(sys.modules, {"common": COMMON}):
        http = COMMON.module("http_service")
    with temporarily_allow_network():
        gateway, _ = http.make_server(
            protocol,
            "baseline",
            address,
            directory / "gateway",
            capability=HTTP_CAPABILITY,
        )
    failures = []
    stop = threading.Event()

    def serve():
        try:
            with temporarily_allow_network():
                while not stop.is_set():
                    gateway.handle_request()
        except Exception as exc:
            failures.append(exc)

    with gateway:
        thread = threading.Thread(target=serve, daemon=True)
        thread.start()
        try:
            yield
        finally:
            stop.set()
            thread.join(timeout=15)
        assert not thread.is_alive(), "HTTP gateway did not stop"
        assert not failures


@pytest.mark.parametrize("evaluator", PINS)
def test_actual_capture_cli_sdk_guard_and_installed_recipient(evaluator, tmp_path):
    _capture_cli(evaluator, tmp_path, http=False)


@pytest.mark.parametrize(
    "evaluator", ["lm-evaluation-harness", "inspect-ai", "promptfoo", "langfuse"]
)
def test_actual_http_capture_cli_sdk_guard_and_installed_recipient(evaluator, tmp_path):
    """Real SDK + HTTP transport, synthetic model facts, separate core recipient."""
    _capture_cli(evaluator, tmp_path, http=True)


def _capture_cli(evaluator, tmp_path, *, http):
    recipient_python = _selected(evaluator)
    directory = tmp_path.resolve()
    case = {
        "id": "synthetic-cli-case",
        "input": "Synthetic instruction: return alpha",
        "expected": "alpha",
        "metadata": {"family": "synthetic-cli-contract"},
    }
    model = {
        "artifact_digest": "sha256:" + "a" * 64,
        "tokenizer_digest": "sha256:" + "b" * 64,
    }
    configuration = {"max_new_tokens": 32}
    protocol = {
        "format": "invarlock/live-evaluator-protocol-v1",
        "cases": [case],
        "models": {"baseline": model},
        "configuration": configuration,
        "evaluators": [evaluator],
        "versions": {evaluator: PINS[evaluator]},
        "test_scope": "synthetic task server, no model execution",
    }
    if http:
        from invarlock.security import temporarily_allow_network

        with temporarily_allow_network(), socket.socket() as channel:
            channel.bind(("127.0.0.1", 0))
            port = channel.getsockname()[1]
        model.update(id="synthetic-cli-model", revision="synthetic-revision")
        configuration["max_length"] = 512
        protocol["limits"] = {"max_requests": 1, "max_seconds": 180}
        protocol["http_services"] = {
            "baseline": {
                "provider": "local-test",
                "service": "synthetic-cli-http",
                "deployment": "baseline",
                "endpoint": f"http://127.0.0.1:{port}/v1/tasks",
                "requested_model": "synthetic-cli-alias",
                "helper_sha256": "sha256:"
                + hashlib.sha256((HERE / "http_service.py").read_bytes()).hexdigest(),
            }
        }
        capability = directory / "http-capability"
        capability.write_text(HTTP_CAPABILITY, encoding="ascii")
        capability.chmod(0o600)
    COMMON.write(directory / "protocol.json", protocol)
    request = {
        "evaluator": evaluator,
        "case_id": case["id"],
        "protocol_digest": COMMON.digest({**protocol, "role": "baseline"}),
    }
    original = {
        "basis": "reference_continuation",
        "logprob_sum": -2.5,
        "token_count": 1,
        "utf8_byte_count": len(case["expected"].encode()),
        "input_digest": COMMON.digest(case["input"]),
        "reference_digest": COMMON.digest(case["expected"]),
        "artifact_digest": model["artifact_digest"],
        "source": {"name": "lm-eval", "version": "0.4.12"},
        "configuration_digest": COMMON.digest(configuration),
        "tokenizer_digest": model["tokenizer_digest"],
    }
    execution = {
        "request": request,
        "protocol_digest": request["protocol_digest"],
        "source": original["source"],
        "model": model,
        "configuration": configuration,
        "generation_result": ["alpha"],
        "likelihood_result": [-2.5, False],
        "test_scope": "synthetic observation, no model called",
    }
    if http:
        execution.update(
            generation_parameters={"until": [], "max_gen_toks": 32, "do_sample": False},
            tokenization={
                "context_token_ids": [1],
                "continuation_token_ids": [2],
                "joined_token_ids": [1, 2],
                "decoded_context": case["input"],
                "decoded_continuation": case["expected"],
                "decoded_joined": case["input"] + case["expected"],
            },
        )
    result = {
        "output": "alpha",
        "metadata": {
            "invarlock_model_execution": execution,
            "invarlock_likelihood": original,
        },
    }
    response = {"request": request, "result": result}
    observed, failures = [], []
    stop = threading.Event()
    # Use a short physical path: macOS bounds AF_UNIX addresses to 104 bytes.
    with tempfile.TemporaryDirectory(
        prefix="sdk-cli-", dir=Path("/tmp").resolve()
    ) as temporary:
        address = str(Path(temporary) / "task.sock")
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
            listener.bind(address)
            listener.listen(2)
            listener.settimeout(0.2)

            def serve():
                while not stop.is_set():
                    try:
                        channel, _ = listener.accept()
                    except TimeoutError:
                        continue
                    try:
                        with channel:
                            channel.settimeout(10)
                            with channel.makefile("rb") as stream:
                                raw = stream.readline(COMMON.MAX_MESSAGE + 1)
                            received = COMMON.decode(raw)
                            observed.append(received)
                            assert received == request
                            assert len(observed) == 1, (
                                "SDK duplicated the single admitted task"
                            )
                            channel.sendall(COMMON.encoded(response))
                    except Exception as exc:
                        failures.append(exc)

            server = threading.Thread(target=serve, daemon=True)
            server.start()
            try:
                with (
                    _http_gateway(protocol, address, directory)
                    if http
                    else nullcontext()
                ):
                    process = subprocess.run(
                        [
                            sys.executable,
                            "-I",
                            "-c",
                            SDK_CLI,
                            str(HERE / "capture.py"),
                            "--protocol",
                            str(directory / "protocol.json"),
                            "--protocol-sha256",
                            COMMON.digest(protocol),
                            "--role",
                            "baseline",
                            "--evaluator",
                            evaluator,
                            *([] if http else ["--socket", address]),
                            *(
                                ["--http-capability-file", str(capability)]
                                if http
                                else []
                            ),
                            "--output",
                            str(directory / "capture"),
                        ],
                        cwd=directory,
                        env=_environment(directory),
                        capture_output=True,
                        text=True,
                        timeout=180,
                    )
            finally:
                stop.set()
                server.join(timeout=12)
            assert not server.is_alive(), "synthetic task server did not stop"
    assert process.returncode == 0, process.stdout + process.stderr
    assert not failures
    assert observed == [request]
    capture = directory / "capture"
    manifest = COMMON.read(capture / "capture.json")
    assert manifest["status"] == "captured" and manifest["case_count"] == 1
    assert manifest["model"] == model
    assert manifest["protocol_digest"] == request["protocol_digest"]
    assert (
        manifest["native_sha256"]
        == "sha256:"
        + hashlib.sha256((capture / "native.json").read_bytes()).hexdigest()
    )
    assert "network" in manifest["driver_files"]
    ledger = list((capture / "tasks").glob("*.response.json"))
    assert len(ledger) == 1 and COMMON.read(ledger[0]) == response
    assert not (capture / "failure.json").exists()
    imported = subprocess.run(
        [recipient_python, "-I", "-c", RECIPIENT, str(HERE), str(capture)],
        cwd=directory,
        env=_environment(directory),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert imported.returncode == 0, imported.stdout + imported.stderr
    canonical = json.loads(imported.stdout)
    rows = canonical["run"]["records"]
    assert len(rows) == 1
    row = rows[0]
    assert (row["id"], row["input"], row["output"], row["expected"]) == (
        case["id"],
        case["input"],
        "alpha",
        case["expected"],
    )
    assert row["metadata"] == case["metadata"]
    assert row["error"] is None
    assert row["likelihood"]["logprob_sum"] == original["logprob_sum"]
    assert row["likelihood"]["token_count"] == original["token_count"]
    assert row["likelihood"]["utf8_byte_count"] == original["utf8_byte_count"]
    assert canonical["recipient"]["sdk_modules_absent"]
    if http:
        assert canonical["routes"] == ["evaluator-native-json", "evaluator-json"]
        descriptor = manifest["service_identity"]
        assert canonical["run"]["artifact_digest"] is None
        assert canonical["run"]["service_identity"] == descriptor
        assert descriptor["observed_model"] == model["id"]
        assert descriptor["exposed_revision"] == model["revision"]
        assert row["likelihood"]["artifact_digest"] is None
        assert row["likelihood"]["service_identity_digest"] == COMMON.digest(descriptor)
        assert (
            COMMON.read(ledger[0])["result"]["metadata"]["invarlock_likelihood"]
            == original
        )
        assert "http_service" in manifest["driver_files"]
        assert (capture / "native-original.json").is_file()
        assert len(list((capture / "http").glob("*.request.json"))) == 1
        assert len(list((capture / "http").glob("*.response.json"))) == 1
