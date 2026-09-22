"""Campaign helper behavior with synthetic transport/data, never model calls."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import socket
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parents[2] / "examples/integrations/evaluator-live"


@pytest.fixture
def modules():
    loaded = {}
    saved = {key: sys.modules.get(key) for key in ("common", "bindings")}
    try:
        for name in ("common", "bindings", "capture", "prepare"):
            spec = importlib.util.spec_from_file_location(
                f"campaign_test_{name}", HERE / f"{name}.py"
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            loaded[name] = module
            if name in saved:
                sys.modules[name] = module
        yield SimpleNamespace(**loaded)
    finally:
        for name, old in saved.items():
            if old is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old


def planned():
    return [
        {
            "id": "case-1",
            "input": "Original question",
            "expected": " Answer",
            "metadata": {"slice": "test"},
        }
    ]


@pytest.fixture
def exchange():
    if not hasattr(socket, "AF_UNIX"):
        pytest.skip("platform does not support Unix-domain transport")

    @contextmanager
    def start(respond):
        with tempfile.TemporaryDirectory(prefix="campaign-wire-") as directory:
            path = Path(directory) / "worker.sock"
            listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            listener.bind(str(path))
            listener.listen(1)
            listener.settimeout(5)
            requests, errors = [], []

            def serve():
                try:
                    with listener.accept()[0] as channel:
                        channel.settimeout(5)
                        with channel.makefile("rb") as stream:
                            request = json.loads(stream.readline(65536))
                        requests.append(request)
                        channel.sendall(respond(request))
                except Exception as exc:
                    errors.append(exc)

            thread = threading.Thread(target=serve, daemon=True)
            thread.start()
            try:
                yield path, requests
            finally:
                thread.join(timeout=6)
                listener.close()
                assert not thread.is_alive(), "test transport did not terminate"
                assert errors == []

    return start


def test_strict_json_bounds_and_exclusive_write(modules, tmp_path):
    common = modules.common
    with pytest.raises(ValueError, match="duplicate JSON"):
        common.decode(b'{"id":1,"id":2}')
    for value in ({"n": float("nan")}, {1: "coerced"}, {"nested": [{1: "coerced"}]}):
        with pytest.raises((ValueError, TypeError)):
            common.encoded(value)
    source = tmp_path / "input.json"
    common.write(source, {"text": "unchanged é"})
    before = source.read_bytes()
    assert common.read(source) == {"text": "unchanged é"}
    with pytest.raises(ValueError, match="byte limit"):
        common.read(source, limit=len(before) - 1)
    with pytest.raises(FileExistsError):
        common.write(source, {"text": "replacement"})
    assert source.read_bytes() == before
    assert common.digest({"b": 2, "a": 1}) == common.digest({"a": 1, "b": 2})


@pytest.mark.parametrize(
    "kind",
    [
        "empty",
        "duplicate",
        "extra",
        "missing",
        "null_input",
        "empty_reference",
        "metadata_keys",
    ],
)
def test_invalid_cases_fail_closed(modules, kind):
    cases = planned()
    if kind == "empty":
        cases = []
    elif kind == "duplicate":
        cases *= 2
    elif kind == "extra":
        cases[0]["unplanned"] = 1
    elif kind == "missing":
        del cases[0]["expected"]
    elif kind == "null_input":
        cases[0]["input"] = None
    elif kind == "empty_reference":
        cases[0]["expected"] = ""
    else:
        cases[0]["metadata"] = {2: "must not coerce"}
    with pytest.raises(ValueError):
        modules.common.cases(cases)


def test_client_freezes_schedule_before_any_transport(modules, monkeypatch, tmp_path):
    common = modules.common
    cases = planned()
    client = common.TaskClient(
        "unused", "deepeval", "digest", cases, tmp_path / "tasks"
    )
    cases[0]["input"] = "mutated after admission"
    monkeypatch.setattr(
        common.socket,
        "socket",
        lambda *args: pytest.fail("invalid case must not connect"),
    )
    with pytest.raises(ValueError, match="frozen case"):
        client(cases[0])
    assert not list(client.output.iterdir())
    with pytest.raises(ValueError, match="omitted"):
        client.complete()


def test_client_admission_response_binding_and_no_retry(modules, exchange, tmp_path):
    common = modules.common
    output = tmp_path / "tasks"
    original = {"output": "Fresh synthetic response", "metadata": {"trace": {"n": 1}}}

    def respond(request):
        admission = output / (
            common.digest(request).removeprefix("sha256:") + ".request.json"
        )
        assert common.read(admission) == request
        return common.encoded({"request": request, "result": original})

    with exchange(respond) as (path, requests):
        client = common.TaskClient(
            path, "deepeval", "approved-role-digest", planned(), output
        )
        result = client(planned()[0])
        result["metadata"]["trace"]["n"] = 99
        assert client.complete()["case-1"] == original
        complete = client.complete()
        complete["case-1"]["output"] = "tampered"
        assert client.complete()["case-1"] == original
        with pytest.raises(ValueError, match="repeated"):
            client(planned()[0])
    assert requests == [
        {
            "evaluator": "deepeval",
            "case_id": "case-1",
            "protocol_digest": "approved-role-digest",
        }
    ]
    assert len(list(output.glob("*.request.json"))) == 1
    assert len(list(output.glob("*.response.json"))) == 1


def test_eight_concurrent_sdk_tasks_use_serial_model_transport(modules, tmp_path):
    common = modules.common
    rows = [{**planned()[0], "id": f"case-{i}"} for i in range(8)]
    barrier = threading.Barrier(8)
    received, errors = [], []
    # Keep the socket name short on systems with a 104-byte sockaddr_un limit.
    with tempfile.TemporaryDirectory(prefix="eight-sdk-tasks-") as directory:
        path = str(Path(directory) / "worker.sock")
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
            listener.bind(path)
            listener.listen(1)
            listener.settimeout(5)
            client = common.TaskClient(
                path, "azure-ai-evaluation", "digest", rows, tmp_path / "tasks"
            )

            def serve():
                try:
                    for _ in rows:
                        with listener.accept()[0] as channel:
                            with channel.makefile("rb") as stream:
                                request = common.decode(stream.readline(65536))
                            received.append(request["case_id"])
                            # Real SDK bursts must not fill the one-slot backlog while
                            # a slower model task is executing.
                            time.sleep(0.02)
                            assert len(
                                list(client.output.glob("*.request.json"))
                            ) == len(received)
                            channel.sendall(
                                common.encoded(
                                    {
                                        "request": request,
                                        "result": {
                                            "output": request["case_id"],
                                            "metadata": {},
                                        },
                                    }
                                )
                            )
                except Exception as exc:
                    errors.append(exc)

            thread = threading.Thread(target=serve, daemon=True)
            thread.start()

            def call(row):
                barrier.wait(timeout=5)
                return client(row)

            with ThreadPoolExecutor(max_workers=8) as pool:
                results = list(pool.map(call, rows))
            thread.join(timeout=6)
            assert not thread.is_alive()
            assert not errors
    assert [row["output"] for row in results] == [row["id"] for row in rows]
    assert set(received) == set(client.complete()) == {row["id"] for row in rows}
    assert len(list(client.output.glob("*.request.json"))) == 8
    assert len(list(client.output.glob("*.response.json"))) == 8


@pytest.mark.parametrize(
    "kind",
    [
        "wrong_binding",
        "missing_output",
        "bad_error",
        "extra_result",
        "null_without_facts",
        "duplicate_json",
        "incomplete",
        "oversized",
    ],
)
def test_bad_response_keeps_admission_and_cannot_retry(
    modules, exchange, tmp_path, monkeypatch, kind
):
    common = modules.common
    if kind == "oversized":
        monkeypatch.setattr(common, "MAX_MESSAGE", 128)

    def respond(request):
        result = {"output": "A", "metadata": {}}
        response = {"request": copy.deepcopy(request), "result": result}
        if kind == "wrong_binding":
            response["request"]["case_id"] = "another"
        elif kind == "missing_output":
            result.pop("output")
            result["error"] = "missing output"
        elif kind == "bad_error":
            result["error"] = 4
        elif kind == "extra_result":
            result["unknown"] = True
        elif kind == "null_without_facts":
            result["output"] = None
        elif kind == "duplicate_json":
            return b'{"request":{},"request":{},"result":{}}\n'
        elif kind == "oversized":
            result["output"] = "a" * 256
        raw = common.encoded(response)
        return raw[:-1] if kind == "incomplete" else raw

    with exchange(respond) as (path, requests):
        client = common.TaskClient(
            path, "deepeval", "digest", planned(), tmp_path / "tasks"
        )
        with pytest.raises(ValueError):
            client(planned()[0])
        with pytest.raises(FileExistsError):
            client(planned()[0])
        with pytest.raises(ValueError, match="omitted"):
            client.complete()
    assert len(requests) == 1
    assert len(list(client.output.glob("*.request.json"))) == 1
    assert list(client.output.glob("*.response.json")) == []


def _capture_setup(modules, monkeypatch):
    common = modules.common
    monkeypatch.setattr(modules.capture, "local_environment", lambda evaluator: None)
    protocol = {
        "evaluators": ["deepeval"],
        "versions": {"deepeval": "4.1.3"},
        "cases": planned(),
        "models": {"baseline": {"artifact_digest": "test-model"}},
    }
    monkeypatch.setattr(
        modules.capture.importlib.metadata, "version", lambda package: "4.1.3"
    )

    class FakeClient:
        def __init__(self, path, evaluator, digest, cases, output):
            self.results = {}
            self.output = output
            output.mkdir()

        def __call__(self, case):
            result = {"output": "Synthetic task answer", "metadata": {}}
            self.results[case["id"]] = result
            common.write(self.output / "retained-response.json", result)
            return copy.deepcopy(result)

        def complete(self):
            if set(self.results) != {"case-1"}:
                raise ValueError("omitted planned model task")
            return self.results

    monkeypatch.setattr(common, "TaskClient", FakeClient)
    return protocol


def test_capture_manifest_authenticates_payload_and_driver_files(
    modules, monkeypatch, tmp_path
):
    protocol = _capture_setup(modules, monkeypatch)

    def run(evaluator, cases, task, workdir):
        return [{"id": case["id"], **task(case)} for case in cases]

    monkeypatch.setattr(
        modules.common, "module", lambda group: SimpleNamespace(run=run)
    )
    output = tmp_path / "capture"
    manifest = modules.capture.capture(
        protocol, "baseline", "deepeval", "unused", output
    )
    assert manifest["case_count"] == 1 and manifest["status"] == "captured"
    assert "pending" in manifest["qualification"]
    assert (
        manifest["native_sha256"]
        == "sha256:" + hashlib.sha256((output / "native.json").read_bytes()).hexdigest()
    )
    assert manifest["protocol_digest"] == modules.common.digest(
        {**protocol, "role": "baseline"}
    )
    assert modules.common.read(output / "capture.json") == manifest
    assert set(manifest["driver_files"]) == {
        "capture",
        "common",
        "bindings",
        "network",
        "scalar",
    }
    assert not (output / "failure.json").exists()
    with pytest.raises(FileExistsError):
        modules.capture.capture(protocol, "baseline", "deepeval", "unused", output)


@pytest.mark.parametrize("failure", ["driver", "omitted"])
def test_failed_capture_preserves_protocol_and_completed_tasks(
    modules, monkeypatch, tmp_path, failure
):
    protocol = _capture_setup(modules, monkeypatch)

    def run(evaluator, cases, task, workdir):
        if failure == "omitted":
            return []
        task(cases[0])
        raise RuntimeError("synthetic SDK failure after one task")

    monkeypatch.setattr(
        modules.common, "module", lambda group: SimpleNamespace(run=run)
    )
    output = tmp_path / "failed"
    with pytest.raises((ValueError, RuntimeError)):
        modules.capture.capture(protocol, "baseline", "deepeval", "unused", output)
    receipt = modules.common.read(output / "failure.json")
    assert receipt["status"] == "capture_failed"
    assert receipt["completed_case_ids"] == ([] if failure == "omitted" else ["case-1"])
    assert (output / "protocol.json").exists()
    assert (
        not (output / "capture.json").exists() and not (output / "native.json").exists()
    )
    assert (output / "tasks/retained-response.json").exists() == (failure == "driver")


@pytest.fixture
def corpus(modules, monkeypatch, tmp_path):
    common = modules.common
    root = tmp_path / "repository"
    monkeypatch.setattr(common, "ROOT", root)
    narrative = (
        root
        / "examples/integrations/evaluator_transaction/lambada_qwen35_deployment_400.jsonl"
    )
    narrative.parent.mkdir(parents=True)
    rows = [
        {
            "id": f"narrative-{index:03}",
            "prompt": f"Original narrative {index}",
            "expected": " end",
        }
        for index in range(30)
    ]
    narrative.write_text("".join(json.dumps(row) + "\n" for row in rows))
    squad = tmp_path / "squad.json"
    data = {
        "data": [
            {
                "title": f"Article {index}",
                "paragraphs": [
                    {
                        "context": "Original public-style context for testing. " * 8,
                        "qas": [
                            {
                                "id": f"answer-{index}",
                                "question": "What original span?",
                                "is_impossible": False,
                                "answers": [
                                    {"text": "Original", "answer_start": 0},
                                    {"text": "Alias", "answer_start": 0},
                                ],
                            },
                            {
                                "id": f"absent-{index}",
                                "question": "What absent fact?",
                                "is_impossible": True,
                                "answers": [],
                            },
                        ],
                    }
                ],
            }
            for index in range(30)
        ]
    }
    common.write(squad, data)
    return squad, data, narrative


def test_preparation_is_deterministic_and_preserves_source_text(modules, corpus):
    squad, original, narrative = corpus
    before = (squad.read_bytes(), narrative.read_bytes())
    sentinel, attribution = modules.prepare.source_cases(squad, "sentinel")
    repeated, same = modules.prepare.source_cases(squad, "sentinel")
    complete, _ = modules.prepare.source_cases(squad, "complete")
    assert sentinel == repeated and attribution == same
    assert len(sentinel) == 8 and len(complete) == 64
    assert {row["id"] for row in sentinel} <= {row["id"] for row in complete}
    for family, count in attribution["counts"].items():
        selected = [row for row in sentinel if row["metadata"]["family"] == family]
        assert len(selected) == count
        assert len({row["metadata"]["source_cluster_id"] for row in selected}) == count
    for row in sentinel:
        if row["metadata"]["dataset"] == "SQuAD2":
            assert original["data"][0]["paragraphs"][0]["context"] in row["input"]
            assert row["expected"] == (
                " Original"
                if row["metadata"]["family"] == "answerable"
                else " NO_ANSWER"
            )
    assert before == (squad.read_bytes(), narrative.read_bytes())
    assert (
        attribution["squad"]["sha256"]
        == "sha256:" + hashlib.sha256(squad.read_bytes()).hexdigest()
    )


def test_selection_independent_of_scores_and_input_order(modules):
    rows = [
        {"id": str(index), "metadata": {"source_cluster_id": str(index // 2)}}
        for index in range(20)
    ]
    selected = modules.prepare.select(rows, 5)
    reordered = copy.deepcopy(list(reversed(rows)))
    for row in reordered:
        row["measured_score"] = 1000 - int(row["id"])
    assert [row["id"] for row in selected] == [
        row["id"] for row in modules.prepare.select(reordered, 5)
    ]
    with pytest.raises(ValueError, match="distinct source groups"):
        modules.prepare.select(rows, 11)


def test_prepare_copies_model_pins_and_complete_schedule(modules, monkeypatch, corpus):
    squad, _, _ = corpus
    common = modules.common
    retained_path = (
        common.ROOT
        / "examples/captured-results/references/mistral-7b-likelihood/capture/baseline/protocol.json"
    )
    retained_path.parent.mkdir(parents=True)
    retained = {
        "models": {
            "baseline": {"artifact_digest": "a", "tokenizer_digest": "tokenizer-a"},
            "subject": {"artifact_digest": "b", "tokenizer_digest": "tokenizer-b"},
        },
        "source_implementations": {"source": "pinned"},
        "configuration": {"seed": 0, "device": "mps"},
    }
    common.write(retained_path, retained)
    monkeypatch.setattr(
        common, "versions", lambda: {"deepeval": "4.1.3", "ragas": "0.4.3"}
    )
    prepared = modules.prepare.prepare(squad, "sentinel", "cpu")
    assert prepared["models"] == retained["models"]
    assert prepared["source_implementations"] == retained["source_implementations"]
    assert prepared["configuration"]["device"] == "cpu"
    assert prepared["limits"] == {"max_requests": 16, "max_seconds": 21600}
    assert set(prepared["versions"]) == set(prepared["evaluators"])


@pytest.mark.parametrize(
    "evaluator", ["deepeval", "lm-evaluation-harness", "promptfoo"]
)
def test_capture_binding_retains_original_measurements(modules, evaluator):
    common, bindings = modules.common, modules.bindings
    case = planned()[0]
    model = {"artifact_digest": "artifact", "tokenizer_digest": "tokenizer"}
    configuration = {"seed": 0}
    facts = {
        "basis": "reference_continuation",
        "logprob_sum": -2.25,
        "token_count": 3,
        "utf8_byte_count": len(case["expected"].encode()),
        "source": {"name": "lm-eval", "version": "0.4.12"},
        "input_digest": common.digest(case["input"]),
        "reference_digest": common.digest(case["expected"]),
        "artifact_digest": "artifact",
        "tokenizer_digest": "tokenizer",
        "configuration_digest": common.digest(configuration),
    }
    result = {
        "output": "Actual captured text",
        "metadata": {
            "invarlock_likelihood": facts,
            "invarlock_model_execution": {
                "source": facts["source"],
                "model": model,
                "configuration": configuration,
            },
        },
    }
    untouched = copy.deepcopy(result)
    bound = bindings.bind_result(result, case, evaluator, "version")
    assert result == untouched
    metadata = bound["metadata"]
    assert metadata["invarlock_capture_binding"]["original_likelihood"] == facts
    changed = {
        key for key in facts if facts[key] != metadata["invarlock_likelihood"][key]
    }
    assert changed <= {"input_digest", "source"}
    assert metadata["invarlock_likelihood"]["source"] == {
        "name": evaluator,
        "version": "version",
    }
    invalid = copy.deepcopy(result)
    invalid["metadata"]["invarlock_likelihood"]["reference_digest"] = "changed"
    with pytest.raises(ValueError, match="execution"):
        bindings.bind_result(invalid, case, evaluator, "version")

    record = {
        **copy.deepcopy(case),
        "output": result["output"],
        "likelihood": metadata["invarlock_likelihood"],
        "context": {},
    }
    if bindings.projection(evaluator):
        record["context"]["input_projection"] = {
            "source": {"input": bindings.native_input(evaluator, case)}
        }
    bindings.check_record(record, result, case, evaluator, "version")
    changed_record = copy.deepcopy(record)
    changed_record["output"] = "changed answer"
    with pytest.raises(ValueError, match="original model task"):
        bindings.check_record(changed_record, result, case, evaluator, "version")
    changed_record = copy.deepcopy(record)
    changed_record["context"] = (
        {}
        if bindings.projection(evaluator)
        else {"input_projection": {"unexpected": True}}
    )
    with pytest.raises(ValueError, match="projection|transformation"):
        bindings.check_record(changed_record, result, case, evaluator, "version")
    if bindings.projection(evaluator):
        changed_record["context"] = {
            "input_projection": {"source": {"input": "changed wrapper"}}
        }
        with pytest.raises(ValueError, match="projection|nullable metadata expansion"):
            bindings.check_record(changed_record, result, case, evaluator, "version")
    injected = copy.deepcopy(result)
    injected["metadata"]["invarlock_capture_binding"] = {}
    with pytest.raises(ValueError, match="cannot supply"):
        bindings.bind_result(injected, case, evaluator, "version")


@pytest.mark.parametrize(
    "refusal", ["role", "evaluator", "missing_pin", "wrong_pin", "installed_version"]
)
def test_capture_refuses_unadmitted_profiles_before_creating_output(
    modules, monkeypatch, tmp_path, refusal
):
    protocol = _capture_setup(modules, monkeypatch)
    role, evaluator = "baseline", "deepeval"
    if refusal == "role":
        role = "unplanned"
    elif refusal == "evaluator":
        evaluator = "ragas"
    elif refusal == "missing_pin":
        protocol["versions"] = {}
    elif refusal == "wrong_pin":
        protocol["versions"]["deepeval"] = "0.0.0"
    else:
        monkeypatch.setattr(
            modules.capture.importlib.metadata, "version", lambda package: "0.0.0"
        )
    output = tmp_path / "must-not-exist"
    with pytest.raises(ValueError):
        modules.capture.capture(protocol, role, evaluator, "unused", output)
    assert not output.exists()


@pytest.mark.parametrize("hosted", [False, True])
def test_capture_refuses_mismatched_http_capability_before_sdk_loading(
    modules, tmp_path, hosted
):
    protocol = {"evaluators": ["inspect-ai"]}
    if hosted:
        protocol["http_services"] = {}
    output = tmp_path / "must-not-exist"
    with pytest.raises(
        ValueError,
        match="requires a private capability file"
        if hosted
        else "does not use an HTTP capability file",
    ):
        modules.capture.capture(
            protocol,
            "baseline",
            "inspect-ai",
            None if hosted else "unused",
            output,
            http_capability_file=None if hosted else tmp_path / "capability",
        )
    assert not output.exists()


def test_promptfoo_capture_checks_actual_package_pin(modules, monkeypatch, tmp_path):
    protocol = _capture_setup(modules, monkeypatch)
    version = modules.common.versions()["promptfoo"]
    protocol.update(evaluators=["promptfoo"], versions={"promptfoo": version})
    package = tmp_path / "promptfoo"
    package.mkdir()
    modules.common.write(package / "package.json", {"version": version})
    monkeypatch.setenv("INVARLOCK_PROMPTFOO_PACKAGE", str(package))
    monkeypatch.setattr(
        modules.common,
        "module",
        lambda group: SimpleNamespace(
            run=lambda evaluator, cases, task, workdir: [task(case) for case in cases]
        ),
    )
    result = modules.capture.capture(
        protocol, "baseline", "promptfoo", "unused", tmp_path / "capture"
    )
    assert result["version"] == version


def test_capture_cli_checks_digest_before_invoking_capture(
    modules, monkeypatch, tmp_path, capsys
):
    protocol = _capture_setup(modules, monkeypatch)
    source = tmp_path / "protocol.json"
    modules.common.write(source, protocol)
    calls = []
    monkeypatch.setattr(
        modules.capture,
        "capture",
        lambda *args, **kwargs: (
            calls.append((args, kwargs)) or {"status": "synthetic-test"}
        ),
    )
    arguments = [
        "capture.py",
        "--protocol",
        str(source),
        "--protocol-sha256",
        "wrong",
        "--role",
        "baseline",
        "--evaluator",
        "deepeval",
        "--socket",
        "unused",
        "--output",
        str(tmp_path / "output"),
    ]
    monkeypatch.setattr(sys, "argv", arguments)
    with pytest.raises(ValueError, match="approved digest"):
        modules.capture.main()
    assert calls == []
    arguments[4] = modules.common.digest(protocol)
    modules.capture.main()
    assert calls[0][0][:3] == (protocol, "baseline", "deepeval")
    assert calls[0][1] == {"http_capability_file": None}
    assert json.loads(capsys.readouterr().out) == {"status": "synthetic-test"}


def test_source_preparation_filters_ineligible_contexts_and_references(
    modules, corpus, monkeypatch
):
    squad, data, _ = corpus
    source = copy.deepcopy(data)
    source["data"][0]["paragraphs"][0]["context"] = "too short"
    source["data"][1]["paragraphs"][0]["qas"][0]["answers"][0]["text"] = ""
    source["data"][2]["paragraphs"][0]["qas"][0]["answers"][0]["text"] = "x" * 129
    squad.write_bytes(modules.common.encoded(source))
    observed = []
    select = modules.prepare.select

    def selecting(rows, count):
        observed.extend(row["id"] for row in rows)
        return select(rows, count)

    monkeypatch.setattr(modules.prepare, "select", selecting)
    modules.prepare.source_cases(squad, "sentinel")
    assert not {
        "squad2-answer-0",
        "squad2-absent-0",
        "squad2-answer-1",
        "squad2-answer-2",
    } & set(observed)
    source["data"][3]["paragraphs"][0]["qas"][0]["answers"] = []
    squad.write_bytes(modules.common.encoded(source))
    with pytest.raises(ValueError, match="no reference"):
        modules.prepare.source_cases(squad, "sentinel")


@pytest.mark.parametrize(
    ("stage", "device"), [("unknown", "cpu"), ("sentinel", "remote")]
)
def test_prepare_refuses_unknown_stage_or_device_before_reading(
    modules, monkeypatch, stage, device
):
    monkeypatch.setattr(
        modules.common,
        "read",
        lambda *args: pytest.fail("unadmitted preparation cannot read source data"),
    )
    with pytest.raises(ValueError, match="unsupported campaign"):
        modules.prepare.prepare("unused", stage, device)


def test_example_loader_and_prepare_cli_freeze_the_selected_arguments(
    modules, monkeypatch, tmp_path, capsys
):
    loaded = modules.common.module("prepare")
    assert loaded.SQUAD_URL == modules.prepare.SQUAD_URL
    calls = []
    value = {"prepared": "synthetic-test"}
    monkeypatch.setattr(
        modules.prepare, "prepare", lambda *args: calls.append(args) or value
    )
    output = tmp_path / "frozen.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare.py",
            "--squad",
            "source.json",
            "--stage",
            "complete",
            "--device",
            "cpu",
            "--output",
            str(output),
        ],
    )
    modules.prepare.main()
    assert calls == [(Path("source.json"), "complete", "cpu")]
    assert modules.common.read(output) == value
    assert capsys.readouterr().out.strip() == modules.common.digest(value)


def test_supervisor_cli_preserves_command_and_exit_status(
    modules, monkeypatch, tmp_path
):
    supervisor = modules.common.module("supervise")
    calls = []
    monkeypatch.setattr(
        supervisor, "run", lambda *args: calls.append(args) or {"exit_code": 124}
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "supervise.py",
            "--seconds",
            "10",
            "--output",
            str(tmp_path),
            "--",
            "python",
            "worker.py",
            "--execute",
        ],
    )
    with pytest.raises(SystemExit) as exc:
        supervisor.main()
    assert exc.value.code == 124
    assert calls == [(["python", "worker.py", "--execute"], 10, tmp_path)]


def test_frozen_cases_cannot_inject_captured_likelihood(modules):
    value = planned()
    value[0]["metadata"]["invarlock_likelihood"] = {"logprob_sum": -1}
    with pytest.raises(ValueError, match="reserved capture"):
        modules.common.cases(value)


def test_capture_configures_local_network_before_sdk_import(modules, monkeypatch):
    calls = []

    def load(name):
        calls.append(name)
        return SimpleNamespace(configure=lambda evaluator: calls.append(evaluator))

    monkeypatch.setattr(modules.common, "module", load)
    modules.capture.local_environment("inspect-ai")
    assert calls == ["network", "inspect-ai"]


@pytest.mark.parametrize("repetitions,reference_mode", [(1, "per_case"), (3, "none")])
def test_judge_proposal_bounds_validate_without_calls(
    modules, corpus, monkeypatch, repetitions, reference_mode
):
    from invarlock.judge_measurements.native_recipe import prepare_native_judge

    squad, _, _ = corpus
    rows, _ = modules.prepare.source_cases(squad, "sentinel")
    # Use the eight prepared records (with real source-group identities), not
    # the transport fixture's minimal case. The corpus itself is synthetic.
    monkeypatch.setattr(modules.common, "ROOT", HERE.parents[2])
    original = copy.deepcopy(rows)
    recipe = modules.prepare.judge_recipe(
        {"cases": rows}, repetitions=repetitions, reference_mode=reference_mode
    )
    calls = 2 * len(rows) * repetitions
    assert recipe["collection"]["max_calls"] == calls
    assert recipe["collection"]["max_output_tokens"] == 25000 * calls
    assert recipe["collection"]["max_cost_microusd"] == 31200 * calls
    assert recipe["plan"]["prompt"]["reference_mode"] == reference_mode
    assert recipe["plan"]["judge"]["config"]["reasoning_effort"] == "xhigh"
    units = {row["metadata"]["source_cluster_id"] for row in rows}
    assert recipe["analysis"]["minimum_units"] == len(units)
    schedule = {
        "records": [
            {
                "record_id": row["id"],
                "input_parts": [{"kind": "text", "text": row["input"]}],
                "expected_output": row["expected"],
            }
            for row in rows
        ]
    }
    preview = prepare_native_judge(modules.common.encoded(recipe), schedule)
    assert preview["planned_trials"] == calls
    assert preview["maximum_admitted_calls"] == calls
    assert preview["independent_units"] == len(units)
    assert rows == original


@pytest.mark.parametrize(
    "arguments",
    [{"repetitions": True}, {"repetitions": 2}, {"reference_mode": "invented"}],
)
def test_judge_proposal_refuses_unbounded_or_unknown_profiles(
    modules, monkeypatch, arguments
):
    monkeypatch.setattr(
        modules.common,
        "read",
        lambda *args: pytest.fail("invalid proposal cannot load recipes"),
    )
    with pytest.raises(ValueError):
        modules.prepare.judge_recipe({"cases": planned()}, **arguments)
