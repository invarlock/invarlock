"""Serve fresh, admitted local model tasks from one persistent Harness model.

Preflight authenticates local files and tokenization without loading weights.
Execution requires an explicit flag, never downloads model files, and records
each admission durably before generation or reference-continuation likelihood.
An admitted pair is never retried, including after interrupted inference.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
import re
import signal
import socket
import stat
import sys
import time
from pathlib import Path

import common

SOURCE = {"name": "lm-eval", "version": "0.4.12"}
MAX_SECONDS = 86400
MAX_MODEL_BYTES = 20 * 1024**3
TOKENIZER_FILES = {
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "merges.txt",
    "vocab.json",
}


def comparison_helpers():
    path = common.ROOT / "examples/captured-results/harness_model_comparison.py"
    spec = importlib.util.spec_from_file_location("live_model_comparison_helpers", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_protocol(protocol, role):
    if role not in {"baseline", "subject"} or not isinstance(protocol, dict):
        raise ValueError("worker requires a declared baseline or subject role")
    common.cases(protocol["cases"])
    evaluators, versions = protocol["evaluators"], protocol["versions"]
    maintained = common.versions()
    if (
        not isinstance(evaluators, list)
        or not evaluators
        or len(evaluators) != len(set(evaluators))
        or not isinstance(versions, dict)
        or set(versions) != set(evaluators)
        or any(
            name not in maintained or versions[name] != maintained[name]
            for name in evaluators
        )
    ):
        raise ValueError("evaluators must have unique maintained version pins")
    configuration = protocol["configuration"]
    required = {"device", "dtype", "batch_size", "max_length", "max_new_tokens", "seed"}
    if not isinstance(configuration, dict) or not required <= set(configuration):
        raise ValueError(
            "configuration must explicitly declare model execution settings"
        )
    if configuration["device"] not in {"cpu", "mps", "cuda"}:
        raise ValueError("worker supports one declared local CPU, MPS, or CUDA device")
    if configuration["dtype"] not in {"float16", "float32", "bfloat16"}:
        raise ValueError("worker requires an explicit floating point dtype")
    for key, expected in {
        "batch_size": 1,
        "max_new_tokens": 32,
        "seed": 0,
    }.items():
        if type(configuration[key]) is not int or configuration[key] != expected:
            raise ValueError(f"configuration requires {key}={expected}")
    if type(configuration["max_length"]) is not int or configuration[
        "max_length"
    ] not in {512, 1024}:
        raise ValueError("configuration requires a declared 512- or 1024-token context")
    fixed = {
        "backend": "causal",
        "softmax_dtype": "float32",
        "logits_cache": False,
        "result_cache": False,
        "truncation": False,
        "add_bos_token": False,
        "use_fast_tokenizer": True,
        "trust_remote_code": False,
        "local_files_only": True,
        "parallelize": False,
        "chat_template": False,
        "deterministic_algorithms": True,
        "attn_implementation": "eager",
        "use_safetensors": True,
        "async_weight_loading": False,
        "maximum_model_file_bytes": MAX_MODEL_BYTES,
        "mps_memory_fraction": 0.42,
        "torch_threads": 4,
        "separate_model_instances": True,
    }
    if set(configuration) - required - set(fixed) - {"timeout_seconds"}:
        raise ValueError("configuration contains unsupported model execution settings")
    for key in set(configuration) & set(fixed):
        if common.encoded(configuration[key]) != common.encoded(fixed[key]):
            raise ValueError(f"configuration cannot enable {key}")
    limits = protocol["limits"]
    if (
        not isinstance(limits, dict)
        or set(limits) != {"max_requests", "max_seconds"}
        or type(limits["max_requests"]) is not int
        or limits["max_requests"] != len(protocol["cases"]) * len(evaluators)
        or type(limits["max_seconds"]) is not int
        or not 1 <= limits["max_seconds"] <= MAX_SECONDS
    ):
        raise ValueError(
            "limits must bind the complete schedule and a finite wall-clock cap"
        )
    if "timeout_seconds" in configuration and (
        type(configuration["timeout_seconds"]) is not int
        or configuration["timeout_seconds"] <= 0
    ):
        raise ValueError("legacy timeout_seconds must be a positive integer")
    if (
        "source_implementations" in protocol
        and protocol["source_implementations"] != comparison_helpers().UPSTREAM_HASHES
    ):
        raise ValueError(
            "Harness source pins differ from the maintained implementation"
        )
    models = protocol["models"]
    if not isinstance(models, dict) or set(models) != {"baseline", "subject"}:
        raise ValueError("both model roles require independent artifact identities")
    supported_models = comparison_helpers().MODELS
    for model_role, model in models.items():
        if not isinstance(model, dict) or set(model) != {
            "id",
            "revision",
            "artifact_digest",
            "tokenizer_digest",
            "files",
        }:
            raise ValueError("model must declare its complete pinned file inventory")
        if (
            not isinstance(model["id"], str)
            or not model["id"]
            or not isinstance(model["revision"], str)
            or not re.fullmatch(r"[0-9a-f]{40}", model["revision"])
        ):
            raise ValueError("model identity requires a full revision pin")
        if (model["id"], model["revision"]) != supported_models[model_role]:
            raise ValueError(
                "worker requires the declared pinned Mistral model profile"
            )
        files = model["files"]
        if not isinstance(files, list) or not files:
            raise ValueError("model requires a nonempty file inventory")
        names = []
        for fact in files:
            if not isinstance(fact, dict) or set(fact) != {
                "path",
                "byte_size",
                "sha256",
            }:
                raise ValueError("invalid model file fact")
            name = fact["path"]
            if not isinstance(name, str) or not (
                name
                in TOKENIZER_FILES
                | {
                    "config.json",
                    "generation_config.json",
                    "model.safetensors.index.json",
                }
                or re.fullmatch(r"model(?:-[0-9]{5}-of-[0-9]{5})?\.safetensors", name)
            ):
                raise ValueError("model inventory contains an unsupported file")
            if (
                type(fact["byte_size"]) is not int
                or fact["byte_size"] <= 0
                or not isinstance(fact["sha256"], str)
                or not re.fullmatch(r"sha256:[0-9a-f]{64}", fact["sha256"])
            ):
                raise ValueError("model files require bounded sizes and SHA-256 pins")
            names.append(name)
        tokenizer_files = [fact for fact in files if fact["path"] in TOKENIZER_FILES]
        if (
            names != sorted(set(names))
            or "config.json" not in names
            or not any(name.endswith(".safetensors") for name in names)
            or not tokenizer_files
            or sum(fact["byte_size"] for fact in files) > MAX_MODEL_BYTES
            or common.digest(files) != model["artifact_digest"]
            or common.digest(tokenizer_files) != model["tokenizer_digest"]
        ):
            raise ValueError(
                "model or tokenizer inventory is incomplete or has a digest mismatch"
            )
    common.encoded(protocol)
    # Historical configurations may carry a prior timeout. The live campaign's
    # explicit limits govern both preflight and execution.
    return {**fixed, **configuration, "timeout_seconds": limits["max_seconds"]}


def inventory(model_dir, model):
    """Authenticate every local model byte; permit normal pinned HF snapshot links."""
    helpers = comparison_helpers()
    if not model_dir.is_dir() or {path.name for path in model_dir.iterdir()} != {
        fact["path"] for fact in model["files"]
    }:
        raise ValueError("model directory must contain exactly its pinned files")
    total, observed = 0, []
    for expected in model["files"]:
        fact = helpers.file_fact(
            model_dir / expected["path"], expected["path"], MAX_MODEL_BYTES - total
        )
        if fact != expected:
            raise ValueError("local model file differs from its independent pin")
        total += fact["byte_size"]
        observed.append(fact)
    config = common.read(model_dir / "config.json")
    if (
        not isinstance(config, dict)
        or "auto_map" in config
        or config.get("model_type") != "mistral"
        or config.get("architectures") != ["MistralForCausalLM"]
    ):
        raise ValueError(
            "model requires the declared built-in Mistral architecture without remote code"
        )
    if (model_dir / "tokenizer_config.json").exists():
        tokenizer_config = common.read(model_dir / "tokenizer_config.json")
        if "auto_map" in tokenizer_config:
            raise ValueError("tokenizer remote code is outside the local profile")
    return observed


def dependencies():
    helpers = comparison_helpers()
    return helpers.dependencies({"source_implementations": helpers.UPSTREAM_HASHES})


def offline():
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        os.environ[key] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    os.environ["HF_DEACTIVATE_ASYNC_LOAD"] = "1"
    # Deterministic CUDA matrix operations require this before Torch initializes.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    sys.dont_write_bytecode = True

    def forbid_network(event, args):
        if event == "socket.getaddrinfo" or (
            event
            in {"socket.connect", "socket.sendto", "socket.sendmsg", "socket.bind"}
            and args[0].family != socket.AF_UNIX
        ):
            raise RuntimeError("network access is forbidden in the local model worker")

    sys.addaudithook(forbid_network)


def tokenizations(model, cases, configuration):
    helpers = comparison_helpers()
    helpers.CONFIGURATION = {
        **helpers.CONFIGURATION,
        "max_length": configuration["max_length"],
    }
    tokens = helpers.tokenize(model, cases)
    for row in tokens:
        if (
            len(row["context_token_ids"]) + configuration["max_new_tokens"]
            > configuration["max_length"]
        ):
            raise ValueError("generation would truncate the frozen input")
    return tokens


def load_model(model_dir, configuration):
    import torch
    from lm_eval.models.huggingface import HFLM

    device = configuration["device"]
    if device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("declared MPS device is unavailable")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("declared CUDA device is unavailable")
    torch.manual_seed(configuration["seed"])
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(4)
    if device == "mps":
        torch.mps.set_per_process_memory_fraction(0.42)
    keys = {
        "backend",
        "device",
        "dtype",
        "softmax_dtype",
        "batch_size",
        "max_length",
        "logits_cache",
        "truncation",
        "add_bos_token",
        "trust_remote_code",
        "local_files_only",
        "parallelize",
        "use_fast_tokenizer",
        "attn_implementation",
        "use_safetensors",
    }
    model = HFLM(pretrained=str(model_dir), **{key: configuration[key] for key in keys})
    if (
        model.backend != "causal"
        or model.max_length != configuration["max_length"]
        or model.logits_cache
        or model.cache_hook.dbdict is not None
        or model.softmax_dtype != torch.float32
        or model.enable_thinking
        or any(
            parameter.device.type != device
            or parameter.dtype != getattr(torch, configuration["dtype"])
            for parameter in model.model.parameters()
        )
    ):
        raise ValueError(
            "loaded model differs from its device, precision, or cache profile"
        )
    return model


def durable_write(path, value):
    common.write(path, value)
    with Path(path).open("rb") as stream:
        os.fsync(stream.fileno())
    descriptor = os.open(Path(path).parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class Worker:
    """One serial schedule with durable admission and no implicit retry."""

    def __init__(
        self,
        protocol,
        role,
        model,
        tokens,
        output,
        *,
        started=None,
        clock=time.monotonic,
        instance=None,
    ):
        self.protocol, self.role, self.model = protocol, role, model
        self.clock = clock
        self.started = clock() if started is None else started
        self.protocol_digest = common.digest({**protocol, "role": role})
        self.cases = {
            case["id"]: (index, case, token)
            for index, (case, token) in enumerate(
                zip(protocol["cases"], tokens, strict=True)
            )
        }
        self.output = Path(output)
        self.admissions = self.output / "requests"
        self.admissions.mkdir(mode=0o700, exist_ok=False)
        self.seen = set()
        self.instance = instance

    def remaining(self):
        return self.protocol["limits"]["max_seconds"] - (self.clock() - self.started)

    def check_time(self):
        if self.remaining() <= 0:
            raise TimeoutError("worker exceeded its declared wall-clock cap")

    def handle(self, request):
        self.check_time()
        if not isinstance(request, dict) or set(request) != {
            "evaluator",
            "case_id",
            "protocol_digest",
        }:
            raise ValueError("request fields differ from the frozen protocol")
        if request["protocol_digest"] != self.protocol_digest:
            raise ValueError("request protocol digest differs from the admitted role")
        if (
            request["evaluator"] not in self.protocol["evaluators"]
            or request["case_id"] not in self.cases
        ):
            raise ValueError("request is outside the frozen evaluator/case schedule")
        pair = request["evaluator"], request["case_id"]
        if pair in self.seen:
            raise ValueError("request pair was already admitted; retries are forbidden")
        if len(self.seen) >= self.protocol["limits"]["max_requests"]:
            raise ValueError("request count exceeds its declared cap")
        request_id = common.digest(request).removeprefix("sha256:")
        durable_write(self.admissions / f"{request_id}.request.json", request)
        self.seen.add(pair)
        index, case, tokens = self.cases[request["case_id"]]
        metadata = {
            "invarlock_model_execution": {
                "protocol_digest": self.protocol_digest,
                "request": request,
                "model": self.protocol["models"][self.role],
                "source": SOURCE,
                "configuration": self.protocol["configuration"],
                "tokenization": tokens,
                "generation_parameters": {
                    "until": [],
                    "max_gen_toks": self.protocol["configuration"]["max_new_tokens"],
                    "do_sample": False,
                },
            }
        }
        result = {"output": None, "metadata": metadata}
        try:
            self.check_time()
            generation = self.model.generate_until(
                [
                    self._instance(
                        "generate_until",
                        case,
                        (
                            case["input"],
                            metadata["invarlock_model_execution"][
                                "generation_parameters"
                            ],
                        ),
                        index,
                    )
                ],
                disable_tqdm=True,
            )
            if (
                not isinstance(generation, list)
                or len(generation) != 1
                or not isinstance(generation[0], str)
            ):
                raise ValueError(
                    "Harness generation did not return exactly one text output"
                )
            result["output"] = generation[0]
            metadata["invarlock_model_execution"]["generation_result"] = generation
            self.check_time()
            likelihood = self.model.loglikelihood(
                [
                    self._instance(
                        "loglikelihood", case, (case["input"], case["expected"]), index
                    )
                ],
                disable_tqdm=True,
            )
            if not isinstance(likelihood, list) or len(likelihood) != 1:
                raise ValueError("Harness likelihood did not return exactly one result")
            record = comparison_helpers().capture_record(
                case,
                likelihood[0],
                tokens,
                self.protocol["models"][self.role],
                common.digest(self.protocol["configuration"]),
                index,
            )
            metadata["invarlock_likelihood"] = record["likelihood"]
            metadata["invarlock_model_execution"]["likelihood_result"] = list(
                likelihood[0]
            )
            self.check_time()
        except Exception as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
        response = {"request": request, "result": result}
        durable_write(self.admissions / f"{request_id}.response.json", response)
        return response

    def _instance(self, kind, case, arguments, index):
        constructor = self.instance
        if constructor is None:
            from lm_eval.api.instance import Instance

            constructor = Instance
        return constructor(request_type=kind, doc=case, arguments=arguments, idx=index)


def private_socket(path):
    path = Path(path)
    if len(os.fsencode(path)) > 100:
        raise ValueError("Unix socket path exceeds the portable byte limit")
    if not path.parent.exists():
        path.parent.mkdir(mode=0o700, parents=True)
    info = path.parent.lstat()
    if (
        not stat.S_ISDIR(info.st_mode)
        or stat.S_IMODE(info.st_mode) != 0o700
        or info.st_uid != os.getuid()
    ):
        raise ValueError(
            "Unix socket directory must be owned by this user with mode 0700"
        )
    if path.exists() or path.is_symlink():
        raise ValueError("Unix socket already exists; it will not be replaced")
    channel = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        channel.bind(str(path))
        path.chmod(0o600)
        channel.listen(1)
    except BaseException:
        channel.close()
        raise
    return channel


def serve(worker, path):
    with private_socket(path) as listener:
        durable_write(
            worker.output / "ready.json",
            {"status": "ready", "protocol_digest": worker.protocol_digest},
        )
        try:
            while len(worker.seen) < worker.protocol["limits"]["max_requests"]:
                worker.check_time()
                listener.settimeout(min(1.0, worker.remaining()))
                try:
                    channel, _ = listener.accept()
                except TimeoutError:
                    continue
                with channel:
                    channel.settimeout(min(10.0, worker.remaining()))
                    request = None
                    try:
                        with channel.makefile("rb") as stream:
                            raw = stream.readline(common.MAX_MESSAGE + 1)
                        if not raw.endswith(b"\n") or len(raw) > common.MAX_MESSAGE:
                            raise ValueError(
                                "request is incomplete or exceeds its byte cap"
                            )
                        request = json.loads(raw)
                        response = worker.handle(request)
                    except (TimeoutError, ValueError, TypeError, KeyError) as exc:
                        response = {
                            "request": request,
                            "rejected": f"{type(exc).__name__}: {exc}",
                        }
                    try:
                        channel.sendall(common.encoded(response))
                    except (BrokenPipeError, ConnectionResetError):
                        pass  # Durable admission/result remains; never rerun lost responses.
        finally:
            Path(path).unlink(missing_ok=True)


def prepare(protocol, role, model_dir, output):
    configuration = validate_protocol(protocol, role)
    model_dir, output = Path(model_dir), Path(output)
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    durable_write(output / "protocol.json", {**protocol, "role": role})
    observed = inventory(model_dir, protocol["models"][role])
    packages, implementations = dependencies()
    tokenizer = comparison_helpers().tokenizer_only(model_dir)
    tokens = tokenizations(tokenizer, protocol["cases"], configuration)
    helpers = comparison_helpers()
    implementation_files = [
        helpers.file_fact(Path(path), name)
        for name, path in (
            ("model_worker.py", __file__),
            ("common.py", common.__file__),
            ("harness_model_comparison.py", helpers.__file__),
        )
    ]
    durable_write(output / "tokenizations.json", tokens)
    durable_write(
        output / "manifest.json",
        {
            "format": "invarlock/live-model-worker-v1",
            "role": role,
            "protocol_digest": common.digest({**protocol, "role": role}),
            "files": observed,
            "packages": packages,
            "source": SOURCE,
            "source_implementations": implementations,
            "capture_implementations": implementation_files,
            "effective_configuration": configuration,
            "runtime_environment": {
                "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG")
            },
        },
    )
    return configuration, tokens


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--role", choices=("baseline", "subject"), required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--socket", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    protocol = common.read(args.protocol)
    if common.digest(protocol) != args.protocol_sha256:
        raise ValueError("protocol differs from its independently approved digest")
    validate_protocol(protocol, args.role)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(
            "worker output already exists; interrupted runs are never resumed"
        )
    offline()
    started = time.monotonic()

    def deadline(signum, frame):
        raise TimeoutError("worker exceeded its declared wall-clock cap")

    prior = signal.signal(signal.SIGALRM, deadline)
    signal.setitimer(signal.ITIMER_REAL, protocol["limits"]["max_seconds"])
    try:
        configuration, tokens = prepare(
            protocol, args.role, args.model_dir, args.output
        )
        if args.preflight:
            summary = {
                "status": "tokenization_only",
                "model_loaded": False,
                "inference_calls": 0,
                "case_count": len(tokens),
            }
            durable_write(args.output / "preflight.json", summary)
        else:
            model = load_model(args.model_dir, configuration)
            if tokenizations(model, protocol["cases"], configuration) != tokens:
                raise ValueError("loaded model tokenization differs from preflight")
            worker = Worker(
                protocol, args.role, model, tokens, args.output, started=started
            )
            serve(worker, args.socket)
            worker.check_time()
            inventory(args.model_dir, protocol["models"][args.role])
            summary = {
                "status": "complete",
                "admitted_requests": len(worker.seen),
                "elapsed_seconds": time.monotonic() - started,
            }
            durable_write(args.output / "complete.json", summary)
        print(common.encoded(summary).decode(), end="")
        return 0
    except BaseException as exc:
        if args.output.is_dir() and not (args.output / "failure.json").exists():
            durable_write(
                args.output / "failure.json",
                {
                    "status": "failed",
                    "exception_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, prior)


if __name__ == "__main__":
    raise SystemExit(main())
