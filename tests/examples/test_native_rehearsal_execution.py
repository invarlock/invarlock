"""Exercise capture orchestration while replacing only optional SDK/model transports.

These are transport tests, not inference evidence. Real upstream/model execution
is the opt-in integration rehearsal, whose original native logs are retained.
"""

from __future__ import annotations

import asyncio
import importlib.util
import io
import json
import sys
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "examples/pipeline/native_rehearsal.py"


def load():
    spec = importlib.util.spec_from_file_location("capture_test", SCRIPT)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def install(monkeypatch, name, **exports):
    module = ModuleType(name)
    module.__path__ = []
    module.__dict__.update(exports)
    monkeypatch.setitem(sys.modules, name, module)
    return module


@pytest.fixture
def capture(tmp_path, monkeypatch):
    module = load()
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_bytes(b"{}")
    protocol_path = tmp_path / "protocol.json"
    expected = module.prepare(model, protocol_path, "fixture/no-inference", "a" * 40)
    protocol = json.loads(protocol_path.read_bytes())
    output = tmp_path / "capture"
    monkeypatch.setattr(module.os, "environ", dict(module.os.environ))
    versions = {
        "torch": "2.13.0+cpu",
        "transformers": "5.14.1",
        "inspect-ai": "0.3.254",
        "lm-eval": "0.4.12+invarlock.exactmatch.1",
    }
    monkeypatch.setattr(module.importlib.metadata, "version", versions.__getitem__)
    return SimpleNamespace(
        module=module,
        model=model,
        protocol_path=protocol_path,
        protocol=protocol,
        expected=expected,
        output=output,
        versions=versions,
    )


def fake_local(monkeypatch, module, *, eos=False):
    instances = []

    class Local:
        def __init__(self, model, protocol):
            self.calls = []
            self.tokenizer = SimpleNamespace(eos_token_id=9)
            instances.append(self)

        def generate(self, prompt):
            self.calls.append(
                {
                    "prompt": prompt,
                    "output": "transport fixture",
                    "token_ids": [9 if eos else 7],
                    "latency_ms": 2.0,
                    "completed_at_monotonic_ns": time.monotonic_ns(),
                    "error": None,
                }
            )
            return "transport fixture"

    monkeypatch.setattr(module, "LocalHF", Local)
    return instances


def inspect_transport(monkeypatch, capture, *, status="success", invalid=None):
    observations = []

    class API:
        def __init__(self, name):
            self.model_name = name

    class Model:
        def __init__(self, api, config):
            self.api, self.config = api, config

    class Output:
        @staticmethod
        def from_content(name, content, stop_reason):
            observations.append(stop_reason)
            return {"model": name, "choices": [{"message": {"content": content}}]}

    class Log:
        def __init__(self, data):
            self.data, self.status = data, data["status"]

        def model_dump_json(self, **kwargs):
            return json.dumps(self.data)

        def model_dump(self, **kwargs):
            return self.data

    def evaluate(task, *, model, **kwargs):
        assert kwargs["max_samples"] == 1 and kwargs["log_format"] == "json"
        assert model.config.max_connections == 1 and model.config.max_tokens == 8
        if invalid is not None:
            messages = [SimpleNamespace(content="text")]
            tools = []
            if invalid == "tools":
                tools = ["unsupported"]
            elif invalid == "multiple":
                messages *= 2
            else:
                messages[0].content = []
            asyncio.run(model.api.generate(messages, tools, None, model.config))
        rows = []
        for sample in task.dataset:
            prompt = task.solver[0].format(prompt=sample.input)
            answer = asyncio.run(
                model.api.generate(
                    [SimpleNamespace(content=prompt)], [], None, model.config
                )
            )
            rows.append(
                {
                    "id": sample.id,
                    "input": sample.input,
                    "target": sample.target,
                    "output": answer,
                }
            )
        return [Log({"status": status, "samples": rows})]

    install(monkeypatch, "inspect_ai", Task=SimpleNamespace, eval=evaluate)
    install(monkeypatch, "inspect_ai._util")
    install(
        monkeypatch,
        "inspect_ai._util.logger",
        init_logger=lambda level, trace_dir: observations.append(trace_dir.is_dir()),
    )
    install(monkeypatch, "inspect_ai.dataset", Sample=SimpleNamespace)
    install(
        monkeypatch,
        "inspect_ai.model",
        GenerateConfig=SimpleNamespace,
        Model=Model,
        ModelAPI=API,
        ModelOutput=Output,
        modelapi=lambda **kwargs: lambda cls: cls,
    )
    install(monkeypatch, "inspect_ai.scorer", match=lambda **kwargs: kwargs)
    install(
        monkeypatch,
        "inspect_ai.solver",
        generate=lambda: "generate",
        prompt_template=lambda value: value,
    )
    return observations


@pytest.mark.parametrize("eos", [False, True])
def test_inspect_orchestrates_all_predeclared_tasks_and_persists_native_logs(
    capture, monkeypatch, eos
):
    fake_local(monkeypatch, capture.module, eos=eos)
    seen = inspect_transport(monkeypatch, capture)
    result = capture.module.capture(
        "inspect",
        capture.model,
        capture.protocol_path,
        capture.expected,
        capture.output,
        "unused",
    )
    assert result == capture.module.digest(
        (capture.output / "capture.json").read_bytes()
    )
    assert seen.count("stop" if eos else "max_tokens") == 24
    assert len(list(capture.output.glob("*-calls.json"))) == 6
    manifest = json.loads((capture.output / "capture.json").read_bytes())
    assert (
        manifest["first_result_elapsed_seconds"] >= 0
        and manifest["process_cpu_seconds"] >= 0
    )
    assert manifest["script_sha256"] == capture.module.digest(SCRIPT.read_bytes())
    assert (
        json.loads((capture.output / "capture-script.json").read_bytes())["text"]
        == SCRIPT.read_text()
    )


@pytest.mark.parametrize("invalid", ["tools", "multiple", "nontext"])
def test_inspect_rejects_unsupported_model_requests(capture, monkeypatch, invalid):
    fake_local(monkeypatch, capture.module)
    inspect_transport(monkeypatch, capture, invalid=invalid)
    with pytest.raises(ValueError, match="one text message"):
        capture.module.capture(
            "inspect",
            capture.model,
            capture.protocol_path,
            capture.expected,
            capture.output,
            "unused",
        )
    assert not (capture.output / "capture.json").exists()


def test_inspect_incomplete_task_is_retained_but_not_published(capture, monkeypatch):
    fake_local(monkeypatch, capture.module)
    inspect_transport(monkeypatch, capture, status="error")
    with pytest.raises(ValueError, match="retained incomplete log"):
        capture.module.capture(
            "inspect",
            capture.model,
            capture.protocol_path,
            capture.expected,
            capture.output,
            "unused",
        )
    assert len(list(capture.output.glob("*-incomplete.json"))) == 1
    assert not (capture.output / "capture.json").exists()


def lm_transport(monkeypatch, capture):
    events = []

    class HFLM:
        def __init__(self, **kwargs):
            assert (
                kwargs["device"] == "cpu"
                and kwargs["dtype"] == "float32"
                and kwargs["trust_remote_code"] is False
            )

        def generate_until(self, requests, disable_tqdm):
            assert disable_tqdm is True and len(requests) == 1
            events.append(requests[0].args)
            return ["transport fixture"]

    def evaluate(*, model, tasks, task_manager, **kwargs):
        assert (
            kwargs["log_samples"] is True
            and kwargs["torch_random_seed"] == 42
            and kwargs["bootstrap_iters"] == 0
        )
        assert task_manager.include_defaults is False
        task = tasks[0]
        data = json.loads(
            Path(task["dataset_kwargs"]["data_files"]["test"]).read_bytes()
        )
        requests = [
            SimpleNamespace(
                args=(
                    task["doc_to_text"].replace("{{input}}", case["input"]),
                    task["generation_kwargs"],
                )
            )
            for case in data
        ]
        results = model.generate_until(requests)
        rows = [
            {
                "doc_id": i,
                "doc": case,
                "target": case["expected"],
                "arguments": [request.args],
                "filtered_resps": [answer],
            }
            for i, (case, request, answer) in enumerate(
                zip(data, requests, results, strict=True)
            )
        ]
        return {
            "samples": {task["task"]: rows},
            "results": {task["task"]: {"exact_match,none": 0}},
        }

    install(monkeypatch, "lm_eval", evaluator=SimpleNamespace(simple_evaluate=evaluate))
    install(monkeypatch, "lm_eval.api")
    install(monkeypatch, "lm_eval.api.metrics", exact_match_hf_evaluate=lambda: None)
    install(monkeypatch, "lm_eval.models")
    install(monkeypatch, "lm_eval.models.huggingface", HFLM=HFLM)
    install(monkeypatch, "lm_eval.tasks", TaskManager=SimpleNamespace)
    return events


def test_lm_uses_generate_until_and_preserves_actual_request_and_sample_records(
    capture, monkeypatch
):
    events = lm_transport(monkeypatch, capture)
    capture.module.capture(
        "lm-eval",
        capture.model,
        capture.protocol_path,
        capture.expected,
        capture.output,
        "unused",
    )
    assert len(events) == 24
    assert all(
        args[1] == {"until": [], "max_gen_toks": 8, "do_sample": False}
        for args in events
    )
    assert len(list(capture.output.glob("*-aggregate.json"))) == 6
    assert all(
        len(path.read_bytes().splitlines()) == 4
        for path in capture.output.glob("*.jsonl")
    )


def promptfoo_transport(monkeypatch, capture, *, exit_code=100):
    module = capture.module
    events = []
    server_ref = []

    class Server:
        server_port = 31234

        def __init__(self, address, handler):
            assert address == ("127.0.0.1", 0)
            self.handler = handler
            server_ref.append(self)

        def serve_forever(self):
            events.append("serve")

        def shutdown(self):
            events.append("shutdown")

        def server_close(self):
            events.append("closed")

    class Thread:
        def __init__(self, target, daemon):
            self.target = target
            assert daemon is True

        def start(self):
            self.target()

        def join(self, timeout):
            events.append(("joined", timeout))

    def request(body, length=None):
        handler = server_ref[0].handler.__new__(server_ref[0].handler)
        handler.headers = {
            "Content-Length": str(len(body) if length is None else length)
        }
        handler.rfile, handler.wfile = io.BytesIO(body), io.BytesIO()
        handler.send_response = lambda code: events.append(code)
        handler.send_error = lambda code: events.append(code)
        handler.send_header = lambda *args: None
        handler.end_headers = lambda: None
        handler.do_POST()
        handler.log_message("ignored")
        return handler.wfile.getvalue()

    def run(command, **kwargs):
        if command == ["fixture-promptfoo", "--version"]:
            return SimpleNamespace(stdout="0.121.19")
        if command == ["node", "--version"]:
            return SimpleNamespace(stdout="v22.18.0")
        assert kwargs["env"]["PROMPTFOO_CONFIG_DIR"] == str(
            capture.output / "promptfoo-state"
        )
        config = json.loads(Path(command[command.index("--config") + 1]).read_bytes())
        assert (
            "--no-cache" in command
            and command[command.index("--max-concurrency") + 1] == "1"
        )
        assert request(b"{}", 65537) == b""
        outputs = []
        for case in config["tests"]:
            rendered = config["prompts"][0].replace("{{input}}", case["vars"]["input"])
            outputs.append(
                json.loads(request(json.dumps({"prompt": rendered}).encode()))
            )
        Path(command[command.index("--output") + 1]).write_text(
            "".join(json.dumps(row) + "\n" for row in outputs)
        )
        return SimpleNamespace(
            args=command,
            returncode=exit_code,
            stdout="fixture log",
            stderr="fixture failure" if exit_code == 2 else "",
        )

    monkeypatch.setattr(module, "HTTPServer", Server)
    monkeypatch.setattr(module.threading, "Thread", Thread)
    monkeypatch.setattr(module.subprocess, "run", run)
    return events


def test_promptfoo_calls_local_http_endpoint_with_transport_generated_text(
    capture, monkeypatch
):
    local = fake_local(monkeypatch, capture.module)
    events = promptfoo_transport(monkeypatch, capture)
    capture.module.capture(
        "promptfoo",
        capture.model,
        capture.protocol_path,
        capture.expected,
        capture.output,
        "fixture-promptfoo",
    )
    assert events.count(200) == 24 and events.count(413) == 6
    assert events[-3:] == ["shutdown", ("joined", 5), "closed"]
    assert local[0].calls and len(list(capture.output.glob("*-command.json"))) == 6
    assert (
        json.loads((capture.output / "capture.json").read_bytes())["node_version"]
        == "v22.18.0"
    )


def test_promptfoo_failure_still_closes_the_local_listener(capture, monkeypatch):
    fake_local(monkeypatch, capture.module)
    events = promptfoo_transport(monkeypatch, capture, exit_code=2)
    with pytest.raises(ValueError, match="Promptfoo execution failed"):
        capture.module.capture(
            "promptfoo",
            capture.model,
            capture.protocol_path,
            capture.expected,
            capture.output,
            "fixture-promptfoo",
        )
    assert events[-3:] == ["shutdown", ("joined", 5), "closed"]
    assert not (capture.output / "capture.json").exists()


def test_capture_rejects_runtime_version_drift_before_model_loading(capture):
    capture.versions["transformers"] = "0.0.0"
    with pytest.raises(ValueError, match="unexpected transformers"):
        capture.module.capture(
            "inspect",
            capture.model,
            capture.protocol_path,
            capture.expected,
            capture.output,
            "unused",
        )


def test_capture_detects_model_mutation_before_publication(capture, monkeypatch):
    def changed(*args):
        (capture.model / "config.json").write_bytes(b'{"changed":true}')

    monkeypatch.setattr(capture.module, "inspect_capture", changed)
    with pytest.raises(ValueError, match="changed during capture"):
        capture.module.capture(
            "inspect",
            capture.model,
            capture.protocol_path,
            capture.expected,
            capture.output,
            "unused",
        )
    assert not (capture.output / "capture.json").exists()


def test_local_hf_enforces_offline_no_remote_code_and_records_generation(
    capture, monkeypatch
):
    module = capture.module
    events = []

    class Tensor:
        shape = (1, 2)

        def __getitem__(self, key):
            assert key == (0, slice(2, None, None))
            return SimpleNamespace(tolist=lambda: [7, 8])

    class Tokenizer:
        eos_token_id = 9

        def __call__(self, prompt, return_tensors):
            assert return_tensors == "pt"
            return {"input_ids": Tensor()}

        def decode(self, ids, **kwargs):
            assert ids == [7, 8] and kwargs == {
                "skip_special_tokens": True,
                "clean_up_tokenization_spaces": False,
            }
            return " preserved whitespace "

    class Model:
        def eval(self):
            events.append("eval")
            return self

        def generate(self, **kwargs):
            assert (
                kwargs["do_sample"] is False
                and kwargs["max_new_tokens"] == 8
                and kwargs["pad_token_id"] == 9
            )
            return Tensor()

    def tokenizer(path, **kwargs):
        assert path == capture.model and kwargs == {
            "local_files_only": True,
            "trust_remote_code": False,
        }
        return Tokenizer()

    def model(path, **kwargs):
        assert kwargs == {
            "local_files_only": True,
            "trust_remote_code": False,
            "weights_only": True,
            "dtype": "float32",
        }
        return Model()

    class Inference:
        def __enter__(self):
            events.append("inference")

        def __exit__(self, *args):
            events.append("done")

    install(
        monkeypatch,
        "torch",
        set_num_threads=lambda n: events.append(("threads", n)),
        manual_seed=lambda n: events.append(("seed", n)),
        float32="float32",
        inference_mode=Inference,
    )
    install(
        monkeypatch,
        "transformers",
        AutoTokenizer=SimpleNamespace(from_pretrained=tokenizer),
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=model),
    )
    local = module.LocalHF(capture.model, capture.protocol)
    assert local.generate("prompt") == " preserved whitespace "
    assert events == [("threads", 1), ("seed", 42), "eval", "inference", "done"]
    assert local.calls[0]["token_ids"] == [7, 8] and local.calls[0]["error"] is None
    assert local.calls[0]["latency_ms"] >= 0


def test_cli_prepare_and_capture_arguments_are_explicit(capture, monkeypatch, capsys):
    module = capture.module
    path = capture.output.with_suffix(".json")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "prepare",
            "--model",
            str(capture.model),
            "--model-id",
            "fixture/no-inference",
            "--revision",
            "a" * 40,
            "--output",
            str(path),
        ],
    )
    module.main()
    assert capsys.readouterr().out.strip() == module.digest(path.read_bytes())
    monkeypatch.setattr(module, "capture", lambda *args: args[3])
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "capture",
            "--evaluator",
            "inspect",
            "--model",
            str(capture.model),
            "--protocol",
            str(path),
            "--expected-protocol",
            capture.expected,
            "--output",
            str(capture.output),
        ],
    )
    module.main()
    assert capsys.readouterr().out.strip() == capture.expected


def test_preparation_rejects_empty_model_and_unknown_protocol(tmp_path):
    module = load()
    with pytest.raises(ValueError, match="empty"):
        module.model_files(tmp_path)
    path = tmp_path / "protocol.json"
    path.write_bytes(b"{}")
    with pytest.raises(ValueError, match="unsupported rehearsal protocol"):
        module.load_protocol(path, module.digest(path.read_bytes()))
