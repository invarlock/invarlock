"""Closed configuration for the maintained evaluator transaction."""

from __future__ import annotations

import os
from typing import Any

MAX_GENERATION_TOKENS = 1
BATCH_SIZE = 8
SEED = 20_260_716
MINIMUM_SIDE_ACCURACY = 0.20
DATASET_NAME = "qwen3-0.6b-base-to-post-trained"
DATASET_SHA256 = "d80e81ba17fb93b9b8a46f9817f9841f5f9c2858c9d703b3ce28847b2eaeb57c"
TOKENIZER_PADDING_SIDE = "left"
TOKENIZER_ADD_SPECIAL_TOKENS = True
TOKENIZER_CLEAN_UP_SPACES = False
PAD_TOKEN_POLICY = "eos_if_missing"
MODEL_USE_CACHE = False
TORCH_NUM_THREADS = 1
INSPECT_RAW_CHAT_TEMPLATE = '{{ messages[0]["content"] }}'
RECORD_COUNT = 102
MAX_WORKER_ARTIFACT_BYTES = 64 * 1024 * 1024
PER_RECORD_TIMEOUT_SECONDS = 300
WORKER_TIMEOUT_SECONDS = min(
    PER_RECORD_TIMEOUT_SECONDS * (RECORD_COUNT + 2), 24 * 60 * 60
)

EVALUATORS: dict[str, dict[str, str]] = {
    "inspect-ai": {
        "distribution": "inspect-ai",
        "version": "0.3.254",
        "entrypoint": "inspect_ai.eval -> inspect_ai.scorer.match",
        "lock": "requirements/workflows/inspect-ai-runtime-py312.txt",
        "container_lock": "/opt/invarlock/evaluator-locks/inspect-ai-runtime-requirements.txt",
    },
    "openai-evals": {
        "distribution": "evals",
        "version": "3.0.1.post1",
        "entrypoint": "evals.elsuite.basic.match.Match",
        "lock": "requirements/workflows/openai-evals-runtime-py312.txt",
        "container_lock": "/opt/invarlock/evaluator-locks/openai-evals-runtime-requirements.txt",
    },
}

EXPECTED_MODEL_ARTIFACTS = {
    "baseline": {
        "path": "models/baseline",
        "model_id": "Qwen/Qwen3-0.6B-Base",
        "locator": "hf://Qwen/Qwen3-0.6B-Base@da87bfb608c14b7cf20ba1ce41287e8de496c0cd",
    },
    "subject": {
        "path": "models/subject",
        "model_id": "Qwen/Qwen3-0.6B",
        "locator": "hf://Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca",
    },
}
EXPECTED_MODEL_TREE_DIGESTS = {
    "baseline": "sha256:eddb974cecb32ecf6bfaec2a19ecfbb32c73be9f7c38c7b54d551cd8ef66bd75",
    "subject": "sha256:f97b7ac0717847938aed654bf671a93a28cf13413e37d29040ebad85564f6346",
}
EXPECTED_TOKENIZER_DIGESTS = {
    "baseline": "c5f0898f912c7d953302779f61c86026b3cea05561a9520b6209e82b9d650581",
    "subject": "ddf5fc73d604adf713f3d2fa98a9229c9dc05abb0881b33e636d15a5616dcd02",
}

RUN_FIELDS = {
    "format",
    "role",
    "evaluator",
    "evaluator_version",
    "task_config",
    "task_config_sha256",
    "execution_config",
    "execution_config_sha256",
    "samples",
    "samples_sha256",
    "model_tree_sha256",
    "dataset_sha256",
    "evaluator_lock_sha256",
    "runtime_image_digest",
    "record_count",
    "stable_id_field",
}
SAMPLE_FIELDS = {
    "record_id",
    "prompt",
    "target",
    "output",
    "input_sha256",
    "target_sha256",
    "output_sha256",
    "reported_score",
    "score_detail",
    "status",
}


class BridgeError(ValueError):
    """The upstream output cannot support verifier replay."""


def evaluator_id() -> str:
    value = os.environ.get("INVARLOCK_EVALUATOR")
    if value not in EVALUATORS:
        raise BridgeError("INVARLOCK_EVALUATOR is not a maintained evaluator")
    return value


def task_config(dataset: str, selected: str | None = None) -> dict[str, Any]:
    name = selected or evaluator_id()
    if name not in EVALUATORS:
        raise BridgeError(f"unsupported evaluator: {name}")
    return {
        "evaluator": name,
        "dataset": dataset,
        "metric": "exact_match",
        "scorer": (
            {
                "location": "exact",
                "ignore_case": False,
                "ignore_punctuation": True,
            }
            if name == "inspect-ai"
            else {"native": "prefix", "transaction": "exact"}
        ),
        "completion_boundary": (
            "target_leading_whitespace" if name == "inspect-ai" else "native"
        ),
        "generation": {
            "do_sample": False,
            "max_new_tokens": MAX_GENERATION_TOKENS,
            "stop": ["\n"],
        },
    }


def execution_config(selected: str | None = None) -> dict[str, Any]:
    name = selected or evaluator_id()
    package = EVALUATORS[name]
    return {
        "batch_size": BATCH_SIZE,
        "device": "cpu",
        "do_sample": False,
        "dtype": "float32",
        "evaluator": name,
        "evaluator_distribution": package["distribution"],
        "evaluator_entrypoint": package["entrypoint"],
        "evaluator_version": package["version"],
        "max_generation_tokens": MAX_GENERATION_TOKENS,
        "model_use_cache": MODEL_USE_CACHE,
        "pad_token_policy": PAD_TOKEN_POLICY,
        "prompt_rendering": (
            {
                "mode": "custom_chat_template",
                "template": INSPECT_RAW_CHAT_TEMPLATE,
            }
            if name == "inspect-ai"
            else {"mode": "completion_function_raw_text"}
        ),
        "seed": SEED,
        "tokenizer_add_special_tokens": TOKENIZER_ADD_SPECIAL_TOKENS,
        "tokenizer_clean_up_tokenization_spaces": TOKENIZER_CLEAN_UP_SPACES,
        "tokenizer_padding_side": TOKENIZER_PADDING_SIDE,
        "torch_num_threads": TORCH_NUM_THREADS,
        "trust_remote_code": False,
    }


__all__ = [
    "BATCH_SIZE",
    "BridgeError",
    "DATASET_NAME",
    "DATASET_SHA256",
    "EVALUATORS",
    "EXPECTED_MODEL_ARTIFACTS",
    "EXPECTED_MODEL_TREE_DIGESTS",
    "EXPECTED_TOKENIZER_DIGESTS",
    "INSPECT_RAW_CHAT_TEMPLATE",
    "MAX_GENERATION_TOKENS",
    "MAX_WORKER_ARTIFACT_BYTES",
    "MINIMUM_SIDE_ACCURACY",
    "MODEL_USE_CACHE",
    "PAD_TOKEN_POLICY",
    "PER_RECORD_TIMEOUT_SECONDS",
    "RECORD_COUNT",
    "RUN_FIELDS",
    "SAMPLE_FIELDS",
    "SEED",
    "TOKENIZER_ADD_SPECIAL_TOKENS",
    "TOKENIZER_CLEAN_UP_SPACES",
    "TOKENIZER_PADDING_SIDE",
    "TORCH_NUM_THREADS",
    "WORKER_TIMEOUT_SECONDS",
    "evaluator_id",
    "execution_config",
    "task_config",
]
