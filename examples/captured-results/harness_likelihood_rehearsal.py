"""Capture real, offline LM Harness likelihoods on six authored compatibility cases.

Requires unmodified lm-eval 0.4.12 and a complete local sshleifer/tiny-gpt2
snapshot. No InvarLock dependency, generation, result cache, or network calls.
Example: python harness_likelihood_rehearsal.py --model ./tiny-gpt2 --output ./raw
These small authored cases establish API compatibility, not model quality.
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
import signal
import sys
import time
from pathlib import Path

SOURCE = {"name": "lm-eval", "version": "0.4.12"}
MODEL_IDENTITY = {
    "id": "sshleifer/tiny-gpt2",
    "revision": "5f91d94bd9cd7190a9f3216ff93cd1dd95f2c7be",
}
MODEL_FILES = (
    "config.json",
    "merges.txt",
    "pytorch_model.bin",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "vocab.json",
)
MODEL_HASHES = {
    "config.json": "77a9e9830c3abeba929f5c61e0e97c398f98b4481ad75516c0be5bf038b340b0",
    "merges.txt": "1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5",
    "pytorch_model.bin": "b706b24034032bdfe765ded5ab6403d201d295a995b790cb24c74becca5c04e6",
    "special_tokens_map.json": "c0b3c279b6ecdb71996a86ffb4d4ab94dfdb5df95f00bac9515688faef2ff5dd",
    "tokenizer_config.json": "5e04eb606e3a1583530a42e36c2a6b6615c86f34fe77e44d9ddeb43ff940931f",
    "vocab.json": "03087853bc70c618b66e7c7a43e787d2db4c469416beac9a483e53dad1f72f27",
}
TOKENIZER_FILES = tuple(
    name for name in MODEL_FILES if name not in {"config.json", "pytorch_model.bin"}
)
UPSTREAM_HASHES = {
    "lm_eval.models.huggingface": "d039cf6cb8d4bb1f8244d9dca2eab8fe90e99d0261099fe93ee29b79219aab2f",
    "lm_eval.api.model": "4b98863889c17ffa7a7d391e83264fde0aa76f6883ef2e863065728b126508f8",
    "lm_eval.api.metrics": "7292215bb68b9cf650023939b2c0e0a9a6f48c77b39600a4069c796c33384b94",
}
CASES = [
    {"id": "capital", "input": "The capital of France is", "expected": " Paris"},
    {"id": "arithmetic", "input": "Two plus two equals", "expected": " four"},
    {"id": "color", "input": "The clear daytime sky looks", "expected": " blue"},
    {"id": "sequence", "input": "The next number after seven is", "expected": " eight"},
    {"id": "unicode", "input": "We met at the", "expected": " café"},
    {"id": "sentence", "input": "A short greeting is", "expected": " hello world"},
]
CONFIGURATION = {
    "backend": "causal",
    "device": "cpu",
    "dtype": "float32",
    "softmax_dtype": "float32",
    "batch_size": 1,
    "max_length": 128,
    "logits_cache": False,
    "result_cache": False,
    "truncation": False,
    "add_bos_token": False,
    "use_fast_tokenizer": True,
    "trust_remote_code": False,
    "local_files_only": True,
    "parallelize": False,
    "seed": 0,
    "torch_threads": 1,
    "deterministic_algorithms": True,
    "maximum_model_file_bytes": 64 * 1024 * 1024,
    "timeout_seconds": 300,
    "separate_model_instances": True,
}


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


def file_fact(path, name):
    payload = path.read_bytes()
    return {
        "path": name,
        "byte_size": len(payload),
        "sha256": "sha256:" + hashlib.sha256(payload).hexdigest(),
    }


def inventory(model):
    if {path.name for path in model.iterdir()} != set(MODEL_FILES):
        raise ValueError(
            "model directory must contain exactly the pinned snapshot files"
        )
    facts = []
    total = 0
    for name in MODEL_FILES:
        path = model / name
        total += path.stat().st_size
        if total > CONFIGURATION["maximum_model_file_bytes"]:
            raise ValueError("model files exceed the fixed local rehearsal size cap")
        fact = file_fact(path, name)
        if fact["sha256"] != "sha256:" + MODEL_HASHES[name]:
            raise ValueError("model file differs from the pinned snapshot")
        facts.append(fact)
    config = json.loads((model / "config.json").read_bytes())
    if config.get("model_type") != "gpt2" or any(
        config.get(key) != value
        for key, value in {
            "n_embd": 2,
            "n_head": 2,
            "n_layer": 2,
            "n_positions": 1024,
            "vocab_size": 50257,
        }.items()
    ):
        raise ValueError("this bounded rehearsal requires the tiny GPT-2 configuration")
    return facts


def validate_case(case):
    if not isinstance(case.get("id"), str) or not case["id"]:
        raise ValueError("case ID must be a nonempty string")
    context, reference = case.get("input"), case.get("expected")
    if not isinstance(context, str) or not context or context != context.rstrip():
        raise ValueError("context must be nonempty with no trailing whitespace")
    if not isinstance(reference, str) or not reference:
        raise ValueError("reference must be a nonempty string")
    if len(context.encode("utf-8")) + len(reference.encode("utf-8")) > 4096:
        raise ValueError("case exceeds the fixed text size cap")


def validate_encoding(
    context_ids, continuation_ids, joined_ids, decoded, reference, max_length
):
    for ids in (context_ids, continuation_ids, joined_ids):
        if not ids or any(type(token) is not int or token < 0 for token in ids):
            raise ValueError("token sequences must contain nonnegative integer IDs")
    if joined_ids != context_ids + continuation_ids:
        raise ValueError(
            "context/continuation token boundary does not match the original joined text"
        )
    if decoded != reference:
        raise ValueError("decoded continuation differs from the original reference")
    if len(joined_ids) > max_length + 1:
        raise ValueError("request would require Harness context truncation")


def capture_record(case, result, tokenization, manifest, index):
    if not isinstance(result, (list, tuple)) or len(result) != 2:
        raise ValueError("Harness must return its native likelihood/greedy pair")
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
            "token_count": len(tokenization["continuation_token_ids"]),
            "utf8_byte_count": len(case["expected"].encode("utf-8")),
            "input_digest": digest(case["input"]),
            "reference_digest": digest(case["expected"]),
            "artifact_digest": manifest["model"]["artifact_digest"],
            "configuration_digest": manifest["configuration_digest"],
            "tokenizer_digest": manifest["tokenizer"]["tokenizer_digest"],
            "source": SOURCE,
        },
        "context": {
            "harness": {
                **tokenization,
                "request_index": index,
                "result": list(result),
                "is_greedy": greedy,
                "input_utf8_byte_count": len(case["input"].encode("utf-8")),
            }
        },
    }


def block_network(event, _args):
    if event in {"socket.connect", "socket.getaddrinfo", "socket.sendto"}:
        raise RuntimeError("network access is forbidden during the local rehearsal")


def deadline(_signum, _frame):
    raise TimeoutError("local rehearsal exceeded its fixed wall-clock cap")


def run(model, output):
    if output.exists():
        raise FileExistsError("output directory must not already exist")
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        os.environ[name] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    sys.dont_write_bytecode = True
    sys.addaudithook(block_network)
    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(CONFIGURATION["timeout_seconds"])
    if importlib.metadata.version("lm-eval") != SOURCE["version"]:
        raise ValueError("unmodified lm-eval 0.4.12 is required")
    implementations = []
    for module_name, expected_hash in UPSTREAM_HASHES.items():
        module = importlib.import_module(module_name)
        fact = file_fact(Path(module.__file__), module_name.replace(".", "/") + ".py")
        if fact["sha256"] != "sha256:" + expected_hash:
            raise ValueError(
                "installed Harness implementation does not match upstream 0.4.12"
            )
        implementations.append(fact)
    import torch
    from lm_eval.api.instance import Instance
    from lm_eval.models.huggingface import HFLM

    for module_name in (
        "lm_eval.api.instance",
        "transformers.models.gpt2.modeling_gpt2",
        "transformers.tokenization_utils_base",
        "transformers.tokenization_utils_tokenizers",
    ):
        module = importlib.import_module(module_name)
        implementations.append(
            file_fact(Path(module.__file__), module_name.replace(".", "/") + ".py")
        )
    torch.set_num_threads(CONFIGURATION["torch_threads"])
    torch.use_deterministic_algorithms(True)
    files = inventory(model)
    tokenizer_files = [item for item in files if item["path"] in TOKENIZER_FILES]
    for case in CASES:
        validate_case(case)
    manifest = {
        "format": "invarlock/harness-likelihood-manifest-v1",
        "source": SOURCE,
        "model_identity": MODEL_IDENTITY,
        "model": {"files": files, "artifact_digest": digest(files)},
        "tokenizer": {
            "files": tokenizer_files,
            "tokenizer_digest": digest(tokenizer_files),
        },
        "configuration": CONFIGURATION,
        "configuration_digest": digest(CONFIGURATION),
        "cases": CASES,
        "source_implementations": implementations,
        "capture_script": file_fact(
            Path(__file__), "examples/captured-results/harness_likelihood_rehearsal.py"
        ),
        "packages": {
            name: importlib.metadata.version(name)
            for name in (
                "lm-eval",
                "transformers",
                "torch",
                "tokenizers",
                "safetensors",
                "huggingface-hub",
                "accelerate",
            )
        },
        "case_scope": "authored compatibility cases; not benchmark or model-quality evidence",
    }
    frozen_manifest = canonical_bytes(manifest)
    runs, timings = {}, {}
    for role in ("baseline", "subject"):
        started = time.monotonic()
        started_unix_ns = time.time_ns()
        torch.manual_seed(CONFIGURATION["seed"])
        lm = HFLM(
            pretrained=str(model),
            backend="causal",
            device="cpu",
            dtype="float32",
            softmax_dtype="float32",
            batch_size=1,
            max_length=CONFIGURATION["max_length"],
            logits_cache=False,
            truncation=False,
            add_bos_token=False,
            trust_remote_code=False,
            local_files_only=True,
            parallelize=False,
            use_fast_tokenizer=True,
        )
        if (
            lm.backend != "causal"
            or lm.max_length != CONFIGURATION["max_length"]
            or lm.logits_cache
            or lm.cache_hook.dbdict is not None
            or any(
                parameter.device.type != "cpu" for parameter in lm.model.parameters()
            )
        ):
            raise ValueError(
                "Harness did not honor the fixed CPU/cache/context settings"
            )
        tokenizations, requests = [], []
        for index, case in enumerate(CASES):
            context_ids, continuation_ids = lm._encode_pair(
                case["input"], case["expected"]
            )
            joined_ids = lm.tok_encode(case["input"] + case["expected"])
            if lm.tok_encode(case["input"]) != context_ids:
                raise ValueError("Harness altered the original context tokenization")
            decoded = lm.tokenizer.decode(
                continuation_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            validate_encoding(
                context_ids,
                continuation_ids,
                joined_ids,
                decoded,
                case["expected"],
                lm.max_length,
            )
            tokenizations.append(
                {
                    "context_token_ids": context_ids,
                    "continuation_token_ids": continuation_ids,
                    "joined_token_ids": joined_ids,
                    "decoded_continuation": decoded,
                }
            )
            requests.append(
                Instance(
                    request_type="loglikelihood",
                    doc=case,
                    arguments=(case["input"], case["expected"]),
                    idx=index,
                )
            )
        with torch.inference_mode():
            results = lm.loglikelihood(requests, disable_tqdm=True)
        if len(results) != len(CASES):
            raise ValueError("Harness omitted or added results")
        runs[role] = {
            "records": [
                capture_record(case, result, tokens, manifest, index)
                for index, (case, result, tokens) in enumerate(
                    zip(CASES, results, tokenizations, strict=True)
                )
            ]
        }
        timings[role] = {
            "wall_seconds": time.monotonic() - started,
            "started_unix_ns": started_unix_ns,
            "finished_unix_ns": time.time_ns(),
        }
        del lm
        gc.collect()
    if inventory(model) != files or canonical_bytes(manifest) != frozen_manifest:
        raise ValueError("source facts changed during the rehearsal")
    raw = {
        "format": "invarlock/harness-likelihood-results-v1",
        "source": SOURCE,
        "runs": runs,
        "metadata": {
            "timings": timings,
            "harness_loglikelihood_call_count": 2,
            "case_count_per_call": len(CASES),
        },
    }
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    for name, payload in (
        ("manifest.json", frozen_manifest),
        ("raw-results.json", canonical_bytes(raw)),
    ):
        path = output / name
        with path.open("xb") as handle:
            handle.write(payload)
        path.chmod(0o444)
    signal.alarm(0)
    return manifest, raw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    manifest, raw = run(args.model.resolve(), args.output.resolve())
    print(
        json.dumps(
            {
                "source": SOURCE,
                "artifact_digest": manifest["model"]["artifact_digest"],
                "case_count": len(CASES),
                "runs": list(raw["runs"]),
            }
        )
    )


if __name__ == "__main__":
    main()
