"""Compare pinned local Mistral checkpoints with unmodified LM Harness 0.4.12.

The independently pinned protocol fixes all 400 cases before scoring. Preflight
loads only tokenizers. Execution retains each native result without retries or
selection. Captured source hashes do not attest to model or provider execution.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import re
import signal
import stat
import sys
import time
from pathlib import Path
from types import MethodType, SimpleNamespace

SOURCE = {"name": "lm-eval", "version": "0.4.12"}
MODELS = {
    "baseline": (
        "mistralai/Mistral-7B-v0.1",
        "27d67f1b5f57dc0953326b2601d68371d40ea8da",
    ),
    "subject": (
        "mistralai/Mistral-7B-Instruct-v0.1",
        "ec5deb64f2c6e6fa90c1abf74a91d5c93a9669ca",
    ),
}
UPSTREAM_HASHES = {
    "lm_eval.models.huggingface": "d039cf6cb8d4bb1f8244d9dca2eab8fe90e99d0261099fe93ee29b79219aab2f",
    "lm_eval.api.model": "4b98863889c17ffa7a7d391e83264fde0aa76f6883ef2e863065728b126508f8",
    "lm_eval.api.metrics": "7292215bb68b9cf650023939b2c0e0a9a6f48c77b39600a4069c796c33384b94",
}
CONFIGURATION = {
    "backend": "causal",
    "device": "mps",
    "dtype": "float16",
    "softmax_dtype": "float32",
    "batch_size": 1,
    "max_length": 512,
    "logits_cache": False,
    "result_cache": False,
    "truncation": False,
    "add_bos_token": False,
    "use_fast_tokenizer": True,
    "trust_remote_code": False,
    "local_files_only": True,
    "parallelize": False,
    "seed": 0,
    "torch_threads": 4,
    "deterministic_algorithms": True,
    "maximum_model_file_bytes": 20 * 1024**3,
    "timeout_seconds": 3600,
    "separate_model_instances": True,
    "chat_template": False,
    "mps_memory_fraction": 0.42,
    "attn_implementation": "eager",
    "use_safetensors": True,
    "async_weight_loading": False,
}
TOKENIZER_FILES = {
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
}
PACKAGES = (
    "lm-eval",
    "transformers",
    "torch",
    "tokenizers",
    "safetensors",
    "huggingface-hub",
    "accelerate",
    "sentencepiece",
)


def canonical_bytes(value):
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def digest(value):
    return "sha256:" + hashlib.sha256(canonical_bytes(value)).hexdigest()


def bytes_digest(raw):
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def file_fact(path, name, maximum=20 * 1024**3):
    """Hash a regular file in bounded chunks, including staged snapshot symlinks."""
    size, hasher = 0, hashlib.sha256()
    with path.open("rb") as handle:
        if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
            raise ValueError("model input must resolve to a regular file")
        while chunk := handle.read(min(1024 * 1024, maximum - size + 1)):
            size += len(chunk)
            if size > maximum:
                raise ValueError("model files exceed the fixed byte allowance")
            hasher.update(chunk)
    return {"path": name, "byte_size": size, "sha256": "sha256:" + hasher.hexdigest()}


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError("duplicate JSON field")
        result[key] = value
    return result


def read_json(path, maximum=16 * 1024 * 1024):
    with path.open("rb") as handle:
        raw = handle.read(maximum + 1)
    if len(raw) > maximum:
        raise ValueError("JSON input exceeds its byte allowance")
    value = json.loads(raw, object_pairs_hook=_pairs)
    canonical_bytes(value)  # Refuse non-finite values even in unused metadata.
    return value, raw


def exact(value, fields, label):
    if not isinstance(value, dict) or set(value) != set(fields):
        raise ValueError(f"{label} must contain exactly the declared fields")


def validate_protocol(protocol, raw, expected_hash):
    if bytes_digest(raw) != expected_hash:
        raise ValueError("protocol differs from its independent byte pin")
    exact(
        protocol,
        {
            "format",
            "models",
            "cases",
            "configuration",
            "policy",
            "source",
            "source_implementations",
            "dataset",
            "capture_script_sha256",
        },
        "protocol",
    )
    if protocol["format"] != "invarlock/harness-model-comparison-v1":
        raise ValueError("unsupported comparison protocol")
    if protocol["source"] != SOURCE:
        raise ValueError("unmodified lm-eval 0.4.12 is required")
    if canonical_bytes(protocol["configuration"]) != canonical_bytes(CONFIGURATION):
        raise ValueError("configuration differs from the fixed MPS profile")
    if (
        protocol["capture_script_sha256"]
        != file_fact(Path(__file__), "script")["sha256"]
    ):
        raise ValueError("capture script differs from the approved protocol")
    for name in ("policy", "dataset"):
        if not isinstance(protocol[name], dict) or not protocol[name]:
            raise ValueError(f"protocol requires explicit {name}")
    exact(protocol["models"], MODELS, "models")
    for role, model in protocol["models"].items():
        exact(
            model,
            {"id", "revision", "files", "artifact_digest", "tokenizer_digest"},
            "model",
        )
        if (model["id"], model["revision"]) != MODELS[role]:
            raise ValueError(
                "model identity differs from the fixed distinct checkpoints"
            )
        files = model["files"]
        if not isinstance(files, list) or not files:
            raise ValueError("model requires a complete file inventory")
        names = []
        for fact in files:
            exact(fact, {"path", "byte_size", "sha256"}, "file fact")
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
                raise ValueError(
                    "only pinned model configuration, tokenizer and safetensors files are allowed"
                )
            if type(fact["byte_size"]) is not int or fact["byte_size"] <= 0:
                raise ValueError("model file byte count must be a positive integer")
            if not isinstance(fact["sha256"], str) or not re.fullmatch(
                r"sha256:[0-9a-f]{64}", fact["sha256"]
            ):
                raise ValueError("model file requires a SHA-256 byte pin")
            names.append(name)
        if (
            names != sorted(set(names))
            or "config.json" not in names
            or not any(name.endswith(".safetensors") for name in names)
        ):
            raise ValueError(
                "model inventory must be sorted, unique and contain config and safetensors"
            )
        tokenizer_files = [fact for fact in files if fact["path"] in TOKENIZER_FILES]
        if (
            not tokenizer_files
            or model["artifact_digest"] != digest(files)
            or model["tokenizer_digest"] != digest(tokenizer_files)
        ):
            raise ValueError("model or tokenizer inventory digest mismatch")
    if (
        protocol["models"]["baseline"]["artifact_digest"]
        == protocol["models"]["subject"]["artifact_digest"]
    ):
        raise ValueError("comparison requires distinct model artifacts")
    cases = protocol["cases"]
    if not isinstance(cases, list) or len(cases) != 400:
        raise ValueError("comparison requires the complete 400-case protocol")
    ids = set()
    for case in cases:
        exact(case, {"id", "input", "expected", "metadata"}, "case")
        if not isinstance(case["id"], str) or not case["id"] or case["id"] in ids:
            raise ValueError("case identities must be nonempty and unique")
        ids.add(case["id"])
        context, reference = case["input"], case["expected"]
        if not isinstance(context, str) or not context or context != context.rstrip():
            raise ValueError("context must be nonempty without trailing whitespace")
        if (
            not isinstance(reference, str)
            or not reference
            or not isinstance(case["metadata"], dict)
        ):
            raise ValueError("case requires original string reference and metadata")
        if len((context + reference).encode("utf-8")) > 65536:
            raise ValueError("case exceeds the fixed text allowance")

    expected_policy = {
        "format": "invarlock/comparison-policy-v1",
        "expected_case_set_digest": digest(
            {
                "format": "invarlock/evaluation-case-set-v1",
                "cases": sorted(cases, key=lambda case: case["id"]),
            }
        ),
        "metrics": [
            {
                "name": "reference_nll",
                "kind": "normalized_nll_per_utf8_byte",
                "direction": "lower",
                "unit": "nats_per_utf8_byte",
                "aggregation": "mean",
                "minimum_count": 400,
                "ratio_max": 1.05,
                "maximum_interval_width": 0.10,
                "configuration": {
                    "configuration_digest": digest(protocol["configuration"]),
                    "baseline_tokenizer_digest": protocol["models"]["baseline"][
                        "tokenizer_digest"
                    ],
                    "subject_tokenizer_digest": protocol["models"]["subject"][
                        "tokenizer_digest"
                    ],
                },
            }
        ],
        "slices": [],
    }
    if canonical_bytes(protocol["policy"]) != canonical_bytes(expected_policy):
        raise ValueError(
            "policy differs from the predeclared case/configuration/ratio bindings"
        )


def inventory(model_path, model):
    files = model["files"]
    if {path.name for path in model_path.iterdir()} != {fact["path"] for fact in files}:
        raise ValueError("snapshot must contain exactly its pinned files")
    total, observed = 0, []
    for expected in files:
        fact = file_fact(
            model_path / expected["path"],
            expected["path"],
            CONFIGURATION["maximum_model_file_bytes"] - total,
        )
        total += fact["byte_size"]
        if fact != expected:
            raise ValueError("snapshot file differs from its independent byte pin")
        observed.append(fact)
    config, _ = read_json(model_path / "config.json")
    if (
        config.get("model_type") != "mistral"
        or config.get("architectures") != ["MistralForCausalLM"]
        or "auto_map" in config
    ):
        raise ValueError(
            "snapshot must contain the ordinary Mistral causal configuration"
        )
    return observed


def dependencies(protocol):
    packages = {name: importlib.metadata.version(name) for name in PACKAGES}
    if packages["lm-eval"] != SOURCE["version"]:
        raise ValueError("unmodified lm-eval 0.4.12 is required")
    facts = []
    for name, expected in UPSTREAM_HASHES.items():
        module = importlib.import_module(name)
        fact = file_fact(Path(module.__file__), name.replace(".", "/") + ".py")
        if fact["sha256"] != "sha256:" + expected:
            raise ValueError("installed Harness source differs from upstream 0.4.12")
        facts.append(fact)
    if protocol["source_implementations"] != UPSTREAM_HASHES:
        raise ValueError("Harness source inventory differs from the protocol")
    runtime_facts = []
    for name in (
        "lm_eval.api.instance",
        "transformers.models.mistral.modeling_mistral",
        "transformers.tokenization_utils_base",
    ):
        module = importlib.import_module(name)
        runtime_facts.append(
            file_fact(Path(module.__file__), name.replace(".", "/") + ".py")
        )
    return packages, facts + runtime_facts


def block_network(event, _args):
    if event in {"socket.connect", "socket.getaddrinfo", "socket.sendto"}:
        raise RuntimeError("network access is forbidden during the comparison")


def offline():
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        os.environ[name] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    os.environ["HF_DEACTIVATE_ASYNC_LOAD"] = "1"
    sys.dont_write_bytecode = True
    sys.addaudithook(block_network)


def deadline(_signum, _frame):
    raise TimeoutError("comparison role exceeded its fixed wall-clock cap")


def tokenizer_only(model_path):
    from lm_eval.models.huggingface import HFLM
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), use_fast=True, trust_remote_code=False, local_files_only=True
    )
    context = SimpleNamespace(
        backend="causal",
        tokenizer=tokenizer,
        add_bos_token=False,
        prefix_token_id=tokenizer.bos_token_id
        if tokenizer.bos_token_id is not None
        else tokenizer.eos_token_id,
    )
    context.tok_encode = MethodType(HFLM.tok_encode, context)
    context._encode_pair = MethodType(HFLM._encode_pair, context)
    return context


def tokenize(lm, cases):
    results = []
    for case in cases:
        left, right = lm._encode_pair(case["input"], case["expected"])
        joined = lm.tok_encode(case["input"] + case["expected"])
        for tokens in (left, right, joined):
            if not tokens or any(
                type(token) is not int or token < 0 for token in tokens
            ):
                raise ValueError("token sequences must contain nonnegative integer IDs")
        if lm.tok_encode(case["input"]) != left or left + right != joined:
            raise ValueError(
                "Harness changed the original context/continuation boundary"
            )
        options = {"skip_special_tokens": False, "clean_up_tokenization_spaces": False}
        decoded_context = lm.tokenizer.decode(left, **options)
        decoded_joined = lm.tokenizer.decode(joined, **options)
        if (
            decoded_context != case["input"]
            or decoded_joined != case["input"] + case["expected"]
        ):
            raise ValueError(
                "tokenizer must round-trip original context and joined text exactly"
            )
        decoded = decoded_joined[len(decoded_context) :]
        if len(joined) > CONFIGURATION["max_length"] + 1:
            raise ValueError("request would require Harness context truncation")
        results.append(
            {
                "context_token_ids": left,
                "continuation_token_ids": right,
                "joined_token_ids": joined,
                "decoded_context": decoded_context,
                "decoded_joined": decoded_joined,
                "decoded_continuation": decoded,
            }
        )
    return results


def capture_record(case, result, tokens, model, configuration_digest, index):
    if not isinstance(result, (list, tuple)) or len(result) != 2:
        raise ValueError("Harness must return a native likelihood/greedy pair")
    score, greedy = result
    if (
        type(score) not in (float, int)
        or not math.isfinite(score)
        or score > 0
        or type(greedy) is not bool
    ):
        raise ValueError("invalid native likelihood/greedy result")
    return {
        **case,
        "output": None,
        "likelihood": {
            "basis": "reference_continuation",
            "logprob_sum": score,
            "token_count": len(tokens["continuation_token_ids"]),
            "utf8_byte_count": len(case["expected"].encode("utf-8")),
            "input_digest": digest(case["input"]),
            "reference_digest": digest(case["expected"]),
            "artifact_digest": model["artifact_digest"],
            "configuration_digest": configuration_digest,
            "tokenizer_digest": model["tokenizer_digest"],
            "source": SOURCE,
        },
        "context": {
            "model_id": model["id"],
            "model_revision": model["revision"],
            "harness": {
                **tokens,
                "request_index": index,
                "result": list(result),
                "is_greedy": greedy,
                "input_utf8_byte_count": len(case["input"].encode("utf-8")),
            },
        },
    }


def write_new(path, value):
    with path.open("xb") as handle:
        handle.write(canonical_bytes(value))
        handle.flush()
        os.fsync(handle.fileno())
    path.chmod(0o444)


def load_model(model_path):
    import torch
    from lm_eval.models.huggingface import HFLM

    if not torch.backends.mps.is_available():
        raise ValueError("the fixed comparison requires Apple MPS")
    torch.set_num_threads(CONFIGURATION["torch_threads"])
    torch.manual_seed(CONFIGURATION["seed"])
    torch.use_deterministic_algorithms(True)
    torch.mps.set_per_process_memory_fraction(CONFIGURATION["mps_memory_fraction"])
    keys = (
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
    )
    lm = HFLM(pretrained=str(model_path), **{key: CONFIGURATION[key] for key in keys})
    if (
        lm.backend != "causal"
        or lm.max_length != CONFIGURATION["max_length"]
        or lm.logits_cache
        or lm.cache_hook.dbdict is not None
        or lm.softmax_dtype != torch.float32
        or lm.enable_thinking
        or any(
            parameter.device.type != "mps" or parameter.dtype != torch.float16
            for parameter in lm.model.parameters()
        )
    ):
        raise ValueError("Harness did not honor the fixed MPS/cache/context settings")
    return lm


def execute_role(protocol, raw_protocol, model_path, output, role, preflight=False):
    if output.exists() or output.is_symlink():
        raise FileExistsError("role output must not already exist")
    previous = signal.signal(signal.SIGALRM, deadline)
    signal.alarm(CONFIGURATION["timeout_seconds"])
    started, started_ns = time.monotonic(), time.time_ns()
    records = []
    lm = None
    try:
        model = protocol["models"][role]
        inventory(model_path, model)
        packages, implementation_facts = dependencies(protocol)
        manifest = {
            "format": "invarlock/harness-model-comparison-manifest-v1",
            "protocol_sha256": bytes_digest(raw_protocol),
            "role": role,
            "source": SOURCE,
            "model": model,
            "configuration": protocol["configuration"],
            "configuration_digest": digest(protocol["configuration"]),
            "source_implementations": implementation_facts,
            "packages": packages,
            "capture_script_sha256": protocol["capture_script_sha256"],
        }
        output.mkdir(mode=0o700, parents=True, exist_ok=False)
        with (output / "protocol.json").open("xb") as handle:
            handle.write(raw_protocol)
        write_new(output / "manifest.json", manifest)
        tokens = tokenize(tokenizer_only(model_path), protocol["cases"])
        write_new(output / "tokenizations.json", tokens)
        if preflight:
            result = {
                "status": "tokenization_only",
                "case_count": len(tokens),
                "model_loaded": False,
                "inference_calls": 0,
            }
            write_new(output / "preflight.json", result)
            return result
        lm = load_model(model_path)
        if tokenize(lm, protocol["cases"]) != tokens:
            raise ValueError("loaded model tokenization differs from preflight")
        import torch
        from lm_eval.api.instance import Instance

        progress = output / "progress"
        progress.mkdir(mode=0o700)
        for index, (case, tokenization) in enumerate(
            zip(protocol["cases"], tokens, strict=True)
        ):
            request = Instance(
                request_type="loglikelihood",
                doc=case,
                arguments=(case["input"], case["expected"]),
                idx=index,
            )
            # Durable admission distinguishes an unattempted slot from an interrupted call.
            write_new(
                progress / f"{index:06}.attempt.json",
                {
                    "request_index": index,
                    "case_id": case["id"],
                    "protocol_sha256": bytes_digest(raw_protocol),
                },
            )
            with torch.inference_mode():
                results = lm.loglikelihood([request], disable_tqdm=True)
            if not isinstance(results, list) or len(results) != 1:
                raise ValueError("Harness omitted or added batch results")
            row = capture_record(
                case,
                results[0],
                tokenization,
                model,
                manifest["configuration_digest"],
                index,
            )
            write_new(progress / f"{index:06}.json", row)
            records.append(row)
        inventory(model_path, model)
        result = {
            "format": "invarlock/harness-model-comparison-results-v1",
            "source": SOURCE,
            "protocol_sha256": bytes_digest(raw_protocol),
            "role": role,
            "records": records,
            "metadata": {
                "status": "complete",
                "harness_loglikelihood_call_count": len(records),
                "case_count": len(records),
                "timings": {
                    "wall_seconds": time.monotonic() - started,
                    "started_unix_ns": started_ns,
                    "finished_unix_ns": time.time_ns(),
                },
            },
        }
        write_new(output / "raw-results.json", result)
        return result
    except BaseException as exc:
        if output.is_dir():
            write_new(
                output / "failure.json",
                {
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "completed_cases": len(records),
                    "role": role,
                },
            )
        raise
    finally:
        del lm
        gc.collect()
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("protocol", "baseline-model", "subject-model", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--role", choices=tuple(MODELS))
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args(argv)
    offline()
    protocol, raw = read_json(args.protocol)
    validate_protocol(protocol, raw, args.protocol_sha256)
    for role in (args.role,) if args.role else MODELS:
        result = execute_role(
            protocol,
            raw,
            getattr(args, role + "_model"),
            args.output / role,
            role,
            args.preflight,
        )
        print(
            json.dumps(
                {
                    "role": role,
                    "status": result.get(
                        "status", result.get("metadata", {}).get("status")
                    ),
                    "output": str(args.output / role),
                }
            )
        )


if __name__ == "__main__":
    main()
