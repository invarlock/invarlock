"""Run a predeclared, tiny CPU model through three real evaluator interfaces.

This is an integration rehearsal, not a model-quality or runtime qualification.
Use prepare before capture, then retain the protocol's independently recorded hash.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

VERSIONS = {
    "inspect": ("inspect-ai", "0.3.254"),
    "lm-eval": ("lm-eval", "0.4.12+invarlock.exactmatch.1"),
    "promptfoo": ("promptfoo", "0.121.19"),
}
PROMPTS = {
    "baseline": "Question: {input}\nAnswer:",
    "candidate": "Return only the requested answer, with no explanation.\nQuestion: {input}\nAnswer:",
}
CASES = {
    "classification": [
        ("Classify sentiment as positive or negative: I loved the meal.", "positive"),
        (
            "Classify sentiment as positive or negative: The service was awful.",
            "negative",
        ),
        (
            "Classify sentiment as positive or negative: The staff were kind.",
            "positive",
        ),
        (
            "Classify sentiment as positive or negative: It broke immediately.",
            "negative",
        ),
    ],
    "extraction": [
        (
            f'Extract status as a JSON object with only the status key: {{"status":"{status}","ticket":{i}}}',
            json.dumps({"status": status}, separators=(",", ":")),
        )
        for i, status in enumerate(("queued", "ready", "failed", "done"))
    ],
    "numeric": [
        (f"Compute {left} + {right}. Reply with the number only.", str(left + right))
        for left, right in ((2, 2), (3, 5), (7, 1), (10, 4))
    ],
}


def encoded(value):
    return (
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
        + b"\n"
    )


def digest(data):
    return "sha256:" + hashlib.sha256(data).hexdigest()


def write(path: Path, value):
    with path.open("xb") as stream:
        stream.write(encoded(value))


def model_files(root: Path):
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(
                "copy the selected local model files; symbolic links are unsupported"
            )
        if path.is_file():
            result[path.relative_to(root).as_posix()] = {
                "sha256": digest(path.read_bytes()),
                "bytes": path.stat().st_size,
            }
    if not result:
        raise ValueError("the local model directory is empty")
    return result


def prepare(model: Path, output: Path, model_id: str, revision: str):
    protocol = {
        "format": "invarlock/example-native-rehearsal-v1",
        "scope": "Tiny CPU integration rehearsal; authored cases do not establish generalization, K2 qualification or useful model quality.",
        "model": {"id": model_id, "revision": revision, "files": model_files(model)},
        "runtime": {
            "torch": "2.13.0",
            "transformers": "5.14.1",
            "dtype": "float32",
            "device": "cpu",
            "threads": 1,
        },
        "generation": {"seed": 42, "max_new_tokens": 8, "do_sample": False},
        "prompts": PROMPTS,
        "cases": {
            kind: [
                {"id": str(i), "input": prompt, "expected": expected}
                for i, (prompt, expected) in enumerate(cases)
            ]
            for kind, cases in CASES.items()
        },
        "evaluators": {
            name: {"package": package, "version": version}
            for name, (package, version) in VERSIONS.items()
        },
        "policy": {
            "candidate_minimum": 0.75,
            "minimum_count": 4,
            "maximum_regression": 0.1,
            "maximum_interval_width": 1.0,
            "latency_candidate_maximum_ms": 10000,
            "latency_maximum_regression_ms": 10000,
            "latency_maximum_interval_width_ms": 20000,
        },
        "scoring": "Native scorers are Inspect match(exact, ignore_case=False), LM exact_match and Promptfoo equals. The recipient requires their observed scores to agree with strict literal equality; normalization-dependent native results require a different mapping. Recipient quality additionally exercises JSON-field and numeric-tolerance scoring on the corresponding task groups. Malformed generated JSON/numbers score zero.",
        "latency": "Recorded wall time of one actual model call, including tokenization and decoding. LM measurements also include the HFLM wrapper. No cross-evaluator latency ranking is supported.",
        "unsupported": [
            "tool calls and non-text completions",
            "remote model code",
            "multiple targets or epochs",
            "production quality claims",
            "GPU execution",
        ],
    }
    write(output, protocol)
    return digest(output.read_bytes())


def load_protocol(path: Path, expected: str, model: Path | None = None):
    raw = path.read_bytes()
    if digest(raw) != expected:
        raise ValueError("predeclared protocol digest mismatch")
    protocol = json.loads(raw)
    if protocol.get("format") != "invarlock/example-native-rehearsal-v1":
        raise ValueError("unsupported rehearsal protocol")
    if model is not None and model_files(model) != protocol["model"]["files"]:
        raise ValueError("local model differs from the predeclared model")
    return protocol


class LocalHF:
    def __init__(self, model: Path, protocol):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.protocol = protocol
        torch.set_num_threads(1)
        torch.manual_seed(protocol["generation"]["seed"])
        self.tokenizer = AutoTokenizer.from_pretrained(
            model, local_files_only=True, trust_remote_code=False
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model,
            local_files_only=True,
            trust_remote_code=False,
            weights_only=True,
            dtype=torch.float32,
        ).eval()
        self.calls = []

    def generate(self, prompt):
        started = time.perf_counter()
        tokens = self.tokenizer(prompt, return_tensors="pt")
        with self.torch.inference_mode():
            generated = self.model.generate(
                **tokens,
                max_new_tokens=self.protocol["generation"]["max_new_tokens"],
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        token_ids = generated[0, tokens["input_ids"].shape[1] :].tolist()
        output = self.tokenizer.decode(
            token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        self.calls.append(
            {
                "prompt": prompt,
                "output": output,
                "token_ids": token_ids,
                "latency_ms": (time.perf_counter() - started) * 1000,
                "completed_at_monotonic_ns": time.monotonic_ns(),
                "error": None,
            }
        )
        return output


def inspect_capture(model: Path, protocol, output: Path):
    from inspect_ai import Task, eval
    from inspect_ai._util.logger import init_logger
    from inspect_ai.dataset import Sample
    from inspect_ai.model import GenerateConfig, Model, ModelAPI, ModelOutput, modelapi
    from inspect_ai.scorer import match
    from inspect_ai.solver import generate, prompt_template

    trace_root = output / "inspect-traces"
    trace_root.mkdir()
    init_logger("ERROR", trace_dir=trace_root)
    local = LocalHF(model, protocol)

    @modelapi(name="native_cpu_rehearsal")
    class API(ModelAPI):
        async def generate(self, input, tools, tool_choice, config):
            if tools or len(input) != 1 or not isinstance(input[0].content, str):
                raise ValueError(
                    "this local provider supports one text message without tools"
                )
            answer = local.generate(input[0].content)
            token_ids = local.calls[-1]["token_ids"]
            reason = (
                "stop"
                if token_ids and token_ids[-1] == local.tokenizer.eos_token_id
                else "max_tokens"
            )
            return ModelOutput.from_content(self.model_name, answer, stop_reason=reason)

    instance = Model(
        API(protocol["model"]["id"]),
        GenerateConfig(max_tokens=8, temperature=0, max_connections=1),
    )
    for kind, cases in protocol["cases"].items():
        for side, template in protocol["prompts"].items():
            local.calls = []
            task = Task(
                name=f"{kind}_{side}",
                dataset=[
                    Sample(id=case["id"], input=case["input"], target=case["expected"])
                    for case in cases
                ],
                solver=[
                    prompt_template(template.replace("{input}", "{prompt}")),
                    generate(),
                ],
                scorer=match(location="exact", ignore_case=False),
            )
            logs = eval(
                task,
                model=instance,
                log_format="json",
                log_dir=str(output / "inspect-logs"),
                display="none",
                log_realtime=False,
                max_samples=1,
            )
            if len(logs) != 1 or logs[0].status != "success":
                write(
                    output / f"{kind}-{side}-incomplete.json",
                    [item.model_dump(mode="json", exclude_none=True) for item in logs],
                )
                raise ValueError(
                    "Inspect did not finish the declared task; see the retained incomplete log"
                )
            write(
                output / f"{kind}-{side}.json",
                json.loads(logs[0].model_dump_json(exclude_none=True)),
            )
            write(output / f"{kind}-{side}-calls.json", local.calls)


def lm_capture(model: Path, protocol, output: Path):
    from lm_eval import evaluator
    from lm_eval.api.metrics import exact_match_hf_evaluate
    from lm_eval.models.huggingface import HFLM
    from lm_eval.tasks import TaskManager

    class MeasuredHF(HFLM):
        calls = None

        def generate_until(self, requests, disable_tqdm=False):
            outputs = []
            for request in requests:
                started = time.perf_counter()
                result = super().generate_until([request], disable_tqdm=True)
                self.calls.append(
                    {
                        "prompt": request.args[0],
                        "output": result[0],
                        "latency_ms": (time.perf_counter() - started) * 1000,
                        "completed_at_monotonic_ns": time.monotonic_ns(),
                        "error": None,
                    }
                )
                outputs.extend(result)
            return outputs

    local = MeasuredHF(
        pretrained=str(model),
        backend="causal",
        dtype="float32",
        device="cpu",
        batch_size=1,
        trust_remote_code=False,
    )
    for kind, cases in protocol["cases"].items():
        data = output / f"{kind}-dataset.json"
        write(data, cases)
        for side, template in protocol["prompts"].items():
            local.calls = []
            name = f"{kind}_{side}"
            task = {
                "task": name,
                "dataset_path": "json",
                "dataset_kwargs": {"data_files": {"test": str(data)}},
                "test_split": "test",
                "output_type": "generate_until",
                "doc_to_text": template.replace("{input}", "{{input}}"),
                "doc_to_target": "{{expected}}",
                "generation_kwargs": {
                    "until": [],
                    "max_gen_toks": protocol["generation"]["max_new_tokens"],
                    "do_sample": False,
                },
                "metric_list": [
                    {
                        "metric": exact_match_hf_evaluate,
                        "aggregation": "mean",
                        "higher_is_better": True,
                    }
                ],
            }
            result = evaluator.simple_evaluate(
                model=local,
                tasks=[task],
                task_manager=TaskManager(include_defaults=False),
                batch_size=1,
                bootstrap_iters=0,
                log_samples=True,
                random_seed=42,
                numpy_random_seed=42,
                torch_random_seed=42,
            )
            with (output / f"{kind}-{side}.jsonl").open("xb") as stream:
                for row in result["samples"][name]:
                    stream.write(encoded(row))
            write(output / f"{kind}-{side}-aggregate.json", result["results"][name])
            write(output / f"{kind}-{side}-calls.json", local.calls)


def promptfoo_capture(model: Path, protocol, output: Path, executable: str):
    local = LocalHF(model, protocol)

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers["Content-Length"])
            if length > 65536:
                self.send_error(413)
                return
            prompt = json.loads(self.rfile.read(length))["prompt"]
            body = encoded({"output": local.generate(prompt)})
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        for kind, cases in protocol["cases"].items():
            for side, template in protocol["prompts"].items():
                local.calls = []
                config = {
                    "description": "Actual tiny CPU inference integration rehearsal",
                    "prompts": [template.replace("{input}", "{{input}}")],
                    "providers": [
                        {
                            "id": f"http://127.0.0.1:{server.server_port}",
                            "config": {
                                "method": "POST",
                                "headers": {"Content-Type": "application/json"},
                                "body": {"prompt": "{{prompt}}"},
                                "responseParser": "json.output",
                            },
                        }
                    ],
                    "tests": [
                        {
                            "vars": {"input": case["input"]},
                            "metadata": {"invarlock_expected": case["expected"]},
                            "assert": [{"type": "equals", "value": case["expected"]}],
                        }
                        for case in cases
                    ],
                }
                config_path = output / f"{kind}-{side}-config.json"
                write(config_path, config)
                result = subprocess.run(
                    [
                        executable,
                        "eval",
                        "--config",
                        str(config_path),
                        "--output",
                        str(output / f"{kind}-{side}.jsonl"),
                        "--no-cache",
                        "--max-concurrency",
                        "1",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=120,
                    env=dict(
                        os.environ,
                        PROMPTFOO_CONFIG_DIR=str(output / "promptfoo-state"),
                        PROMPTFOO_DISABLE_TELEMETRY="1",
                        PROMPTFOO_DISABLE_UPDATE="1",
                        NODE_OPTIONS="--max-old-space-size=1024",
                    ),
                )
                write(
                    output / f"{kind}-{side}-command.json",
                    {
                        "arguments": result.args,
                        "returncode": result.returncode,
                        "stdout": result.stdout,
                        "stderr": result.stderr,
                    },
                )
                if result.returncode not in (0, 100):
                    raise ValueError(f"Promptfoo execution failed: {result.stderr}")
                write(output / f"{kind}-{side}-calls.json", local.calls)
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def capture(evaluator, model, protocol_path, expected_protocol, output, promptfoo):
    import resource

    started = time.perf_counter()
    monotonic_started = time.monotonic_ns()
    cpu_started = time.process_time()
    protocol = load_protocol(protocol_path, expected_protocol, model)
    output.mkdir()
    os.environ.update(
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false",
        HF_HOME=str(output / "hf-cache"),
        HF_DATASETS_CACHE=str(output / "datasets-cache"),
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        PROMPTFOO_DISABLE_TELEMETRY="1",
        PROMPTFOO_DISABLE_UPDATE="1",
        PROMPTFOO_CONFIG_DIR=str(output / "promptfoo-state"),
    )
    package, version = VERSIONS[evaluator]
    actual = (
        subprocess.run(
            [promptfoo, "--version"],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        ).stdout.strip()
        if evaluator == "promptfoo"
        else importlib.metadata.version(package)
    )
    if actual != version:
        raise ValueError(f"expected {package} {version}, found {actual}")
    for name in ("torch", "transformers"):
        if (
            importlib.metadata.version(name).removesuffix("+cpu")
            != protocol["runtime"][name]
        ):
            raise ValueError(f"unexpected {name} runtime version")
    if evaluator == "inspect":
        inspect_capture(model, protocol, output)
    elif evaluator == "lm-eval":
        lm_capture(model, protocol, output)
    else:
        promptfoo_capture(model, protocol, output, promptfoo)
    if model_files(model) != protocol["model"]["files"]:
        raise ValueError("model files changed during capture")
    write(
        output / "capture-script.json",
        {"filename": Path(__file__).name, "text": Path(__file__).read_text()},
    )
    files = {
        p.name: digest(p.read_bytes()) for p in sorted(output.iterdir()) if p.is_file()
    }
    first_completed = min(
        row["completed_at_monotonic_ns"]
        for path in output.glob("*-calls.json")
        for row in json.loads(path.read_bytes())
    )
    manifest = {
        "format": "invarlock/example-native-capture-v1",
        "evaluator": evaluator,
        "version": actual,
        "protocol_sha256": expected_protocol,
        "files": files,
        "elapsed_seconds": time.perf_counter() - started,
        "first_result_elapsed_seconds": (first_completed - monotonic_started) / 1e9,
        "process_cpu_seconds": time.process_time() - cpu_started,
        "peak_resident_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * (1 if platform.system() == "Darwin" else 1024),
        "host_logical_cpu_count": os.cpu_count(),
        "requested_cpu_threads": 1,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            dist.metadata["Name"]: dist.version
            for dist in importlib.metadata.distributions()
        },
        "script_sha256": digest(Path(__file__).read_bytes()),
    }
    if evaluator == "promptfoo":
        manifest["node_version"] = subprocess.run(
            ["node", "--version"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
    write(output / "capture.json", manifest)
    return digest((output / "capture.json").read_bytes())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_args = sub.add_parser("prepare")
    prepare_args.add_argument("--model", type=Path, required=True)
    prepare_args.add_argument("--model-id", required=True)
    prepare_args.add_argument("--revision", required=True)
    prepare_args.add_argument("--output", type=Path, required=True)
    capture_args = sub.add_parser("capture")
    capture_args.add_argument("--evaluator", choices=VERSIONS, required=True)
    capture_args.add_argument("--model", type=Path, required=True)
    capture_args.add_argument("--protocol", type=Path, required=True)
    capture_args.add_argument("--expected-protocol", required=True)
    capture_args.add_argument("--output", type=Path, required=True)
    capture_args.add_argument("--promptfoo", default="promptfoo")
    args = parser.parse_args()
    if args.command == "prepare":
        print(prepare(args.model, args.output, args.model_id, args.revision))
    else:
        print(
            capture(
                args.evaluator,
                args.model,
                args.protocol,
                args.expected_protocol,
                args.output,
                args.promptfoo,
            )
        )


if __name__ == "__main__":
    main()
