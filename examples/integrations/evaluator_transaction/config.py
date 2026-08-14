"""Closed configuration for the maintained evaluator transaction."""

from __future__ import annotations

import os
from typing import Any

from .corpora import CorpusProfile, corpus_profile
from .model_profiles import ModelProfile, model_profile

QUICK_CORPUS = corpus_profile("quick")
QUICK_MODELS = model_profile("quick")
MAX_GENERATION_TOKENS = 1
BATCH_SIZE = QUICK_MODELS.batch_size
SEED = 20_260_716
MINIMUM_SIDE_ACCURACY = QUICK_CORPUS.minimum_side_accuracy
DATASET_NAME = QUICK_CORPUS.dataset_name
DATASET_SHA256 = QUICK_CORPUS.dataset_sha256
TOKENIZER_PADDING_SIDE = "left"
TOKENIZER_ADD_SPECIAL_TOKENS = True
TOKENIZER_CLEAN_UP_SPACES = False
PAD_TOKEN_POLICY = "eos_if_missing"
MODEL_USE_CACHE = False
TORCH_NUM_THREADS = QUICK_MODELS.torch_num_threads
INSPECT_RAW_CHAT_TEMPLATE = '{{ messages[0]["content"] }}'
RECORD_COUNT = QUICK_CORPUS.record_count
MAX_WORKER_ARTIFACT_BYTES = 64 * 1024 * 1024
MAX_PROVENANCE_BYTES = 768 * 1024
PER_RECORD_TIMEOUT_SECONDS = 300
WORKER_TIMEOUT_SECONDS = min(
    PER_RECORD_TIMEOUT_SECONDS * (RECORD_COUNT + 2), 24 * 60 * 60
)


def worker_timeout_seconds(profile: CorpusProfile) -> int:
    return min(PER_RECORD_TIMEOUT_SECONDS * (profile.record_count + 2), 24 * 60 * 60)


EVALUATORS: dict[str, dict[str, str]] = {
    "inspect-ai": {
        "distribution": "inspect-ai",
        "version": "0.3.254",
        "entrypoint": "inspect_ai.eval -> inspect_ai.scorer.match",
        "lock": "requirements/workflows/inspect-ai-runtime-py312.txt",
        "cuda_lock": "requirements/workflows/inspect-ai-runtime-py312-cu129.txt",
        "container_lock": "/opt/invarlock/evaluator-locks/inspect-ai-runtime-requirements.txt",
    },
    "openai-evals": {
        "distribution": "evals",
        "version": "3.0.1.post1",
        "entrypoint": "evals.elsuite.basic.match.Match",
        "lock": "requirements/workflows/openai-evals-runtime-py312.txt",
        "cuda_lock": "requirements/workflows/openai-evals-runtime-py312-cu129.txt",
        "container_lock": "/opt/invarlock/evaluator-locks/openai-evals-runtime-requirements.txt",
    },
}


def model_artifacts(profile: ModelProfile) -> dict[str, dict[str, str]]:
    return {
        snapshot.role: {
            "path": f"models/{snapshot.role}",
            "model_id": snapshot.repository,
            "locator": snapshot.locator,
        }
        for snapshot in profile.snapshots
    }


EXPECTED_MODEL_ARTIFACTS = model_artifacts(QUICK_MODELS)
EXPECTED_MODEL_TREE_DIGESTS = {
    snapshot.role: snapshot.checkpoint_tree_sha256
    for snapshot in QUICK_MODELS.snapshots
    if snapshot.checkpoint_tree_sha256 is not None
}
EXPECTED_TOKENIZER_DIGESTS = {
    snapshot.role: snapshot.tokenizer_contract_sha256
    for snapshot in QUICK_MODELS.snapshots
    if snapshot.tokenizer_contract_sha256 is not None
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


def execution_config(
    selected: str | None = None, profile: CorpusProfile | None = None
) -> dict[str, Any]:
    name = selected or evaluator_id()
    package = EVALUATORS[name]
    selected_corpus = profile or corpus_profile(
        os.environ.get("INVARLOCK_CORPUS_PROFILE", "quick")
    )
    selected_models = model_profile(selected_corpus.key)
    return {
        "batch_size": selected_models.batch_size,
        "device": selected_models.device,
        "do_sample": False,
        "dtype": selected_models.dtype,
        "evaluator": name,
        "evaluator_distribution": package["distribution"],
        "evaluator_entrypoint": package["entrypoint"],
        "evaluator_version": package["version"],
        "max_generation_tokens": MAX_GENERATION_TOKENS,
        "model_profile": selected_models.profile_id,
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
        "torch_num_threads": selected_models.torch_num_threads,
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
    "MAX_PROVENANCE_BYTES",
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
    "worker_timeout_seconds",
    "evaluator_id",
    "execution_config",
    "model_artifacts",
    "task_config",
]
