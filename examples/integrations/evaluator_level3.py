#!/usr/bin/env python3
"""Run one pinned evaluator over two Qwen3 sides and sign the result."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

import yaml
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

try:
    from examples.integrations.trust_material import (
        create_trust_material,
        read_external_file,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - flat-script compatibility
    if not exc.name or not exc.name.startswith("examples"):
        raise
    from trust_material import (  # type: ignore[no-redef]
        create_trust_material,
        read_external_file,
    )
try:
    from examples.integrations.launch import inspect_level3_image
except ModuleNotFoundError as exc:  # pragma: no cover - flat-script compatibility
    if not exc.name or not exc.name.startswith("examples"):
        raise
    from launch import inspect_level3_image  # type: ignore[no-redef]
from invarlock import __version__ as INVARLOCK_VERSION

try:
    from examples.integrations.bounded_command import run_bounded_command
except ModuleNotFoundError as exc:  # pragma: no cover - flat-script compatibility
    if not exc.name or not exc.name.startswith("examples"):
        raise
    from bounded_command import run_bounded_command  # type: ignore[no-redef]
from invarlock.core.checkpoint_identity import checkpoint_tree_sha256
from invarlock.core.runtime_provider import (
    ModelRuntimeSpec,
    RuntimeBackendIdentity,
    RuntimeDeviceFacts,
    RuntimeExecutionSettings,
    RuntimeProviderPluginIdentity,
    canonical_runtime_behavioral_schedule_json,
    load_runtime_behavioral_schedule,
)
from invarlock.core.schedule_preparation import (
    LocalDatasetRequest,
    prepare_local_evaluation_schedule_bytes,
)
from invarlock.evaluation_oci import OciEvaluationError
from invarlock.evidence_pack_contract import canonical_json_bytes, sha256_digest
from invarlock.evidence_pack_integrity import public_key_fingerprint
from invarlock.runtime_import_authoring import (
    load_external_scoring_records_jsonl,
    write_runtime_import_paired_records,
    write_runtime_import_side,
)
from invarlock.runtime_providers.hf_transformers import HFTransformersProvider

try:
    from examples.integrations.evaluator_transaction.worker import (
        run_evaluator_worker,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - flat-script compatibility
    if not exc.name or not exc.name.startswith("examples"):
        raise
    try:
        from evaluator_transaction.worker import run_evaluator_worker
    except ModuleNotFoundError as nested_exc:
        if nested_exc.name not in {
            "evaluator_transaction",
            "evaluator_transaction.worker",
        }:
            raise
        from evaluator_transaction_worker import (  # type: ignore[no-redef]
            run_evaluator_worker,
        )

MAX_GENERATION_TOKENS = 1
BATCH_SIZE = 8
SEED = 20_260_716
MINIMUM_SIDE_ACCURACY = 0.20
DATASET_NAME = "qwen3-0.6b-base-to-post-trained"
DATASET_SHA256 = "d80e81ba17fb93b9b8a46f9817f9841f5f9c2858c9d703b3ce28847b2eaeb57c"
IMAGE_ID = re.compile(r"^sha256:[0-9a-f]{64}$")
SOURCE_COMMIT = re.compile(r"^[0-9a-f]{40}$")
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
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
        "lock": "requirements/workflows/inspect-ai-level3-py312.txt",
        "container_lock": "/opt/invarlock/evaluator-locks/inspect-ai-level3-requirements.txt",
    },
    "openai-evals": {
        "distribution": "evals",
        "version": "3.0.1.post1",
        "entrypoint": "evals.elsuite.basic.match.Match",
        "lock": "requirements/workflows/openai-evals-level3-py312.txt",
        "container_lock": "/opt/invarlock/evaluator-locks/openai-evals-level3-requirements.txt",
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


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_regular_file(
    path: Path,
    *,
    label: str,
    max_bytes: int = MAX_WORKER_ARTIFACT_BYTES,
) -> bytes:
    """Read one regular file through a stable, no-follow file descriptor."""

    nofollow = getattr(os, "O_NOFOLLOW", None)
    if not isinstance(nofollow, int):
        raise BridgeError("secure evaluator artifact loading is unavailable")
    try:
        descriptor = os.open(path, os.O_RDONLY | nofollow)
    except OSError as exc:
        raise BridgeError(
            f"{label} could not be opened without following links"
        ) from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise BridgeError(f"{label} must be a regular file")
        if before.st_size > max_bytes:
            raise BridgeError(f"{label} exceeds its size limit")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, max_bytes + 1 - total))
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise BridgeError(f"{label} exceeds its size limit")
            chunks.append(chunk)
        after = os.fstat(descriptor)
        identity = lambda value: (  # noqa: E731 - compact stable projection
            value.st_dev,
            value.st_ino,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )
        if identity(before) != identity(after):
            raise BridgeError(f"{label} changed while being read")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def evaluator_lock_digest(selected: str, *, container: bool = False) -> str:
    package = EVALUATORS[selected]
    path = Path(package["container_lock"] if container else package["lock"])
    if not container:
        path = REPOSITORY_ROOT / path
    return (
        f"sha256:{digest(_read_regular_file(path, label=f'{selected} evaluator lock'))}"
    )


def _runtime_image_from_environment() -> str:
    value = os.environ.get("INVARLOCK_RUNTIME_IMAGE_ID", "")
    if IMAGE_ID.fullmatch(value) is None:
        raise BridgeError("the worker must receive the inspected runtime image digest")
    return value


def _external_ed25519_key(path: Path, *, label: str) -> ed25519.Ed25519PrivateKey:
    try:
        payload = read_external_file(path, label=label)
        key = serialization.load_pem_private_key(payload, password=None)
    except (TypeError, ValueError) as exc:
        raise BridgeError(f"{label} is not an Ed25519 private key") from exc
    if not isinstance(key, ed25519.Ed25519PrivateKey):
        raise BridgeError(f"{label} is not an Ed25519 private key")
    return key


def _external_ed25519_public_key(path: Path, *, label: str) -> ed25519.Ed25519PublicKey:
    try:
        payload = read_external_file(path, label=label)
        key = serialization.load_pem_public_key(payload)
    except (OSError, TypeError, ValueError) as exc:
        raise BridgeError(f"{label} is not an Ed25519 public key") from exc
    if not isinstance(key, ed25519.Ed25519PublicKey):
        raise BridgeError(f"{label} is not an Ed25519 public key")
    return key


def _require_distinct_signers(
    evidence_key: ed25519.Ed25519PrivateKey,
    verifier_key: ed25519.Ed25519PrivateKey,
    builder_key: ed25519.Ed25519PublicKey | None = None,
) -> None:
    fingerprints = [
        public_key_fingerprint(evidence_key.public_key()),
        public_key_fingerprint(verifier_key.public_key()),
    ]
    if builder_key is not None:
        fingerprints.append(public_key_fingerprint(builder_key))
    if len(fingerprints) != len(set(fingerprints)):
        raise BridgeError(
            "evidence, verifier, and builder signing keys must be distinct"
        )


def _inspect_runtime_image(
    engine: str,
    image: str,
    selected: str,
    lock_digest: str,
    *,
    source_commit: str,
    base_image_id: str,
    build_attestation: Path,
    builder_public_key: ed25519.Ed25519PublicKey,
) -> None:
    if engine not in {"docker", "podman"}:
        raise BridgeError("container engine must be docker or podman")
    if SOURCE_COMMIT.fullmatch(source_commit) is None:
        raise BridgeError("source commit must be a full lowercase Git commit")
    if IMAGE_ID.fullmatch(base_image_id) is None:
        raise BridgeError("base image identity must be an immutable image digest")
    try:
        inspect_level3_image(
            engine=engine,
            image=image,
            repository=REPOSITORY_ROOT,
            attestation_path=build_attestation,
            evaluator=selected,
            evaluator_version=EVALUATORS[selected]["version"],
            lock_sha256=lock_digest,
            expected_entrypoint=(
                "python",
                "/opt/invarlock/examples/evaluator-level3.py",
                "worker",
            ),
            source_commit=source_commit,
            base_image_id=base_image_id,
            builder_public_key=builder_public_key,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise BridgeError(
            "Level 3 build attestation did not authenticate the image"
        ) from exc


def evaluator_id() -> str:
    value = os.environ.get("INVARLOCK_EVALUATOR")
    if value not in EVALUATORS:
        raise BridgeError("INVARLOCK_EVALUATOR is not a maintained Level 3 evaluator")
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


def _restore_inspect_causal_boundary(completion: str, target: str) -> str:
    """Restore the causal token boundary removed by Inspect's HF decoder.

    Inspect AI 0.3.254 uses ``tokenizer.batch_decode`` for HF completions;
    that path removes the leading whitespace carried by the first causal BPE
    token.  The fixed corpus makes that boundary part of the target and the
    core exact-match contract is byte-exact, so the bridge restores only the
    authenticated target's leading whitespace.  The native Inspect score is
    still checked against the restored output during adaptation.
    """

    prefix_length = len(target) - len(target.lstrip())
    prefix = target[:prefix_length]
    if prefix and not completion.startswith(prefix):
        return prefix + completion
    return completion


def _records(dataset_bytes: bytes) -> list[dict[str, str]]:
    values = [json.loads(line) for line in dataset_bytes.splitlines()]
    if len(values) != 102 or any(
        not isinstance(value, dict)
        or set(value) != {"expected", "id", "prompt"}
        or any(not isinstance(value[key], str) or not value[key] for key in value)
        for value in values
    ):
        raise BridgeError("the Level 3 corpus must contain 102 complete records")
    if len({value["id"] for value in values}) != len(values):
        raise BridgeError("the Level 3 corpus IDs are not unique")
    return cast(list[dict[str, str]], values)


class _HfGreedyGenerator:
    """The pinned local model adapter used by both native evaluator runners."""

    def __init__(self, model_path: Path) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise BridgeError(
                "the Level 3 image lacks the Hugging Face runtime"
            ) from exc

        self._torch = torch
        execution = execution_config()
        if (model_path / "generation_config.json").exists() or (
            model_path / "generation_config.json"
        ).is_symlink():
            raise BridgeError("model snapshot must not provide generation defaults")
        torch.manual_seed(execution["seed"])
        torch.set_num_threads(execution["torch_num_threads"])
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, local_files_only=True, trust_remote_code=False
        )
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise BridgeError("the tokenizer has neither a pad nor EOS token")
            if execution["pad_token_policy"] != PAD_TOKEN_POLICY:
                raise BridgeError("unsupported pad-token policy")
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = execution["tokenizer_padding_side"]
        self._tokenizer = tokenizer
        self._model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            dtype=torch.float32,
            trust_remote_code=False,
        ).eval()

    def generate(self, prompts: list[str]) -> list[str]:
        execution = execution_config()
        output: list[str] = []
        with self._torch.inference_mode():
            for offset in range(0, len(prompts), BATCH_SIZE):
                batch = prompts[offset : offset + BATCH_SIZE]
                encoded = self._tokenizer(
                    batch,
                    add_special_tokens=execution["tokenizer_add_special_tokens"],
                    padding=True,
                    return_tensors="pt",
                )
                generated = self._model.generate(
                    **encoded,
                    do_sample=execution["do_sample"],
                    max_new_tokens=execution["max_generation_tokens"],
                    pad_token_id=self._tokenizer.pad_token_id,
                    use_cache=execution["model_use_cache"],
                )
                continuation = generated[:, encoded["input_ids"].shape[1] :]
                output.extend(
                    self._tokenizer.decode(
                        tokens,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=execution[
                            "tokenizer_clean_up_tokenization_spaces"
                        ],
                    ).split("\n", 1)[0]
                    for tokens in continuation
                )
        return output

    def close(self) -> None:
        del self._model


def _generate(model_path: Path, dataset_bytes: bytes) -> list[dict[str, str]]:
    """Generate one greedy token per record for compatibility and diagnostics."""

    records = _records(dataset_bytes)
    generator = _HfGreedyGenerator(model_path)
    outputs = generator.generate([record["prompt"] for record in records])
    generator.close()
    if len(outputs) != len(records):
        raise BridgeError("the model adapter returned an incomplete result")
    output = [
        {**record, "output": text}
        for record, text in zip(records, outputs, strict=True)
    ]
    return output


def _run_inspect_ai(
    model_path: Path, dataset_bytes: bytes
) -> tuple[list[dict[str, str]], list[tuple[float, dict[str, Any]]]]:
    """Run an Inspect Task, including its model adapter and scorer."""

    from inspect_ai import Task
    from inspect_ai import eval as inspect_eval
    from inspect_ai.dataset import MemoryDataset, Sample
    from inspect_ai.scorer import match
    from inspect_ai.solver import generate

    records = _records(dataset_bytes)
    task = Task(
        dataset=MemoryDataset(
            [
                Sample(input=item["prompt"], target=item["expected"], id=item["id"])
                for item in records
            ]
        ),
        solver=generate(),
        scorer=match(location="exact", ignore_case=False),
        name="invarlock-level3",
    )
    logs = inspect_eval(
        task,
        model="hf/invarlock",
        model_args={
            "model_path": str(model_path),
            "device": "cpu",
            "batch_size": BATCH_SIZE,
            "do_sample": False,
            "chat_template": INSPECT_RAW_CHAT_TEMPLATE,
            "trust_remote_code": False,
            "enable_thinking": False,
            "tokenizer_call_args": {"add_special_tokens": True},
        },
        display="none",
        log_dir="/tmp/invarlock-inspect-logs",
        log_samples=True,
        log_realtime=False,
        log_model_api=False,
        score=True,
        run_samples=True,
        sample_shuffle=False,
        epochs=1,
        fail_on_error=True,
        continue_on_fail=False,
        max_connections=BATCH_SIZE,
        max_samples=BATCH_SIZE,
        max_tokens=MAX_GENERATION_TOKENS,
        stop_seqs=["\n"],
        seed=SEED,
        log_level="error",
    )
    if len(logs) != 1 or logs[0].status != "success" or logs[0].samples is None:
        raise BridgeError("Inspect AI did not produce one successful sample log")
    by_id = {str(sample.id): sample for sample in logs[0].samples}
    generated: list[dict[str, str]] = []
    scored: list[tuple[float, dict[str, Any]]] = []
    for record in records:
        sample = by_id.get(record["id"])
        if (
            sample is None
            or sample.input != record["prompt"]
            or str(sample.target) != record["expected"]
        ):
            raise BridgeError(
                "Inspect AI changed the Level 3 sample identity or target"
            )
        native_completion = sample.output.completion
        if not isinstance(native_completion, str):
            raise BridgeError("Inspect AI returned a non-text completion")
        score = sample.scores.get("match")
        if score is None:
            raise BridgeError("Inspect AI did not return its match score")
        value = str(score.value)
        generated.append(
            {
                **record,
                "output": _restore_inspect_causal_boundary(
                    native_completion, record["expected"]
                ),
            }
        )
        scored.append(
            (
                1.0 if value == "C" else 0.0,
                {
                    "answer": score.answer,
                    "explanation": score.explanation,
                    "value": value,
                },
            )
        )
    return generated, scored


class _OpenAICompletionResult:
    def __init__(self, completion: str) -> None:
        self._completion = completion

    def get_completions(self) -> list[str]:
        return [self._completion]


class _OpenAIHfCompletionFn:
    def __init__(self, generator: _HfGreedyGenerator) -> None:
        self._generator = generator

    def __call__(self, *, prompt: str, **_: Any) -> _OpenAICompletionResult:
        if not isinstance(prompt, str):
            raise BridgeError("OpenAI Evals supplied a non-text prompt")
        completions = self._generator.generate([prompt])
        if len(completions) != 1:
            raise BridgeError(
                "the OpenAI Evals model adapter returned an invalid result"
            )
        return _OpenAICompletionResult(completions[0])


def _openai_event_to_sample(
    record: dict[str, str], data: Any
) -> tuple[str, float, dict[str, Any]]:
    if not isinstance(data, dict) or data.get("expected") != record["expected"]:
        raise BridgeError("OpenAI Evals changed the Level 3 sample identity or target")
    completion = data.get("sampled")
    if not isinstance(completion, str):
        raise BridgeError("OpenAI Evals returned a non-text completion")
    correct = data.get("correct")
    if not isinstance(correct, bool):
        raise BridgeError("OpenAI Evals returned an invalid match result")
    native_correct = completion.startswith(record["expected"])
    if correct != native_correct:
        raise BridgeError("OpenAI Evals returned an inconsistent native match result")
    transaction_correct = completion == record["expected"]
    return (
        completion,
        1.0 if transaction_correct else 0.0,
        {
            "picked": data.get("picked"),
            "native_correct": correct,
            "transaction_correct": transaction_correct,
        },
    )


def _run_openai_evals(
    model_path: Path, dataset_bytes: bytes
) -> tuple[list[dict[str, str]], list[tuple[float, dict[str, Any]]]]:
    """Run the upstream OpenAI Evals basic.Match evaluator."""

    os.environ.setdefault("OPENAI_API_KEY", "unused")
    from evals.elsuite.basic.match import Match
    from evals.record import DummyRecorder, RunSpec

    records = _records(dataset_bytes)
    previous = {
        name: os.environ.get(name)
        for name in ("EVALS_SEQUENTIAL", "EVALS_THREADS", "EVALS_SHOW_EVAL_PROGRESS")
    }
    generator = _HfGreedyGenerator(model_path)
    try:
        with tempfile.TemporaryDirectory(prefix="invarlock-openai-evals-") as temp_dir:
            dataset_path = Path(temp_dir) / "samples.jsonl"
            dataset_path.write_bytes(
                b"".join(
                    canonical_json_bytes(
                        {"input": item["prompt"], "ideal": item["expected"]}
                    )
                    for item in records
                )
            )
            os.environ["EVALS_SEQUENTIAL"] = "1"
            os.environ["EVALS_THREADS"] = "1"
            os.environ["EVALS_SHOW_EVAL_PROGRESS"] = "0"
            evaluation = Match(
                completion_fns=[_OpenAIHfCompletionFn(generator)],
                samples_jsonl=str(dataset_path),
                eval_registry_path=temp_dir,
                name="invarlock-level3.default",
                seed=SEED,
                max_tokens=MAX_GENERATION_TOKENS,
                num_few_shot=0,
            )
            recorder = DummyRecorder(
                RunSpec(
                    completion_fns=["invarlock/hf"],
                    eval_name="invarlock-level3.default",
                    base_eval="basic.match",
                    split="default",
                    run_config={},
                    created_by="invarlock",
                ),
                log=False,
            )
            evaluation.eval_all_samples(
                recorder, evaluation.get_samples(), show_progress=False
            )
            events = recorder.get_events("match")
            if len(events) != len(records):
                raise BridgeError(
                    "OpenAI Evals did not return one match event per record"
                )
            by_index: dict[int, Any] = {}
            for event in events:
                sample_id = str(event.data.get("sample_id", event.sample_id))
                suffix = sample_id.rsplit(".", 1)[-1]
                if not suffix.isdigit() or int(suffix) in by_index:
                    raise BridgeError(
                        "OpenAI Evals returned ambiguous sample identities"
                    )
                by_index[int(suffix)] = event.data
            generated = []
            scored = []
            for index, record in enumerate(records):
                data = by_index.get(index)
                completion, score, detail = _openai_event_to_sample(record, data)
                generated.append({**record, "output": completion})
                scored.append((score, detail))
            return generated, scored
    finally:
        generator.close()
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _run_upstream_evaluator(
    model_path: Path, dataset_bytes: bytes, selected: str
) -> tuple[list[dict[str, str]], list[tuple[float, dict[str, Any]]]]:
    if selected == "inspect-ai":
        return _run_inspect_ai(model_path, dataset_bytes)
    if selected == "openai-evals":
        return _run_openai_evals(model_path, dataset_bytes)
    raise BridgeError(f"unsupported evaluator: {selected}")


def _run_local_cli(command: list[str]) -> str:
    """Run one completion CLI with bounded diagnostics and a hard deadline."""
    try:
        completed = run_bounded_command(
            command,
            capture_output=True,
            check=True,
            timeout_seconds=WORKER_TIMEOUT_SECONDS,
            stdout_limit=4 * 1024 * 1024,
            stderr_limit=256 * 1024,
            label="Level 3 completion command",
        )
    except subprocess.CalledProcessError as exc:
        diagnostic = (exc.stderr or exc.output or "").strip()
        raise BridgeError(
            diagnostic or f"command exited with status {exc.returncode}"
        ) from exc
    except RuntimeError as exc:
        raise BridgeError(str(exc)) from exc
    return completed.stdout or ""


def worker(role: str, model: Path, dataset: Path, output: Path) -> None:
    selected = evaluator_id()
    package = EVALUATORS[selected]
    observed = importlib.metadata.version(package["distribution"])
    if observed != package["version"]:
        raise BridgeError(
            f"the runtime must contain {package['distribution']} {package['version']}"
        )
    runtime_image_digest = _runtime_image_from_environment()
    lock_digest = evaluator_lock_digest(selected, container=True)
    if os.environ.get("INVARLOCK_EVALUATOR_LOCK_SHA256") != lock_digest:
        raise BridgeError("the evaluator lock does not match the inspected image label")
    if (
        output.exists()
        or output.is_symlink()
        or not model.is_dir()
        or dataset.is_symlink()
        or not dataset.is_file()
    ):
        raise BridgeError("worker inputs must exist and output must be new")
    output.mkdir(parents=True)
    model_digest = checkpoint_tree_sha256(model)
    dataset_bytes = _read_regular_file(dataset, label="worker dataset")
    dataset_digest = digest(dataset_bytes)
    config = task_config("/records.jsonl", selected)
    config_path = output / "task.json"
    config_path.write_bytes(canonical_json_bytes(config))
    config_digest = digest(_read_regular_file(config_path, label="task configuration"))
    generated, scored = _run_upstream_evaluator(model, dataset_bytes, selected)
    if len(generated) != len(scored):
        raise BridgeError("upstream scorer did not return one result per record")
    samples: list[dict[str, Any]] = []
    for sample, (score, detail) in zip(generated, scored, strict=True):
        samples.append(
            {
                "record_id": sample["id"],
                "prompt": sample["prompt"],
                "target": sample["expected"],
                "output": sample["output"],
                "input_sha256": digest(sample["prompt"].encode()),
                "target_sha256": digest(sample["expected"].encode()),
                "output_sha256": digest(sample["output"].encode()),
                "reported_score": float(score),
                "score_detail": detail,
                "status": "ok",
            }
        )
    samples_path = output / "samples.jsonl"
    samples_path.write_bytes(b"".join(canonical_json_bytes(item) for item in samples))
    if (
        checkpoint_tree_sha256(model) != model_digest
        or digest(_read_regular_file(dataset, label="worker dataset")) != dataset_digest
    ):
        raise BridgeError("model or dataset changed during evaluator execution")
    execution = execution_config(selected)
    manifest = {
        "format": "invarlock/evaluator-level3-run-v1",
        "role": role,
        "evaluator": selected,
        "evaluator_version": package["version"],
        "task_config": config,
        "task_config_sha256": digest(canonical_json_bytes(config)),
        "execution_config": execution,
        "execution_config_sha256": digest(canonical_json_bytes(execution)),
        "samples": samples_path.name,
        "samples_sha256": digest(_read_regular_file(samples_path, label="samples")),
        "model_tree_sha256": model_digest,
        "dataset_sha256": dataset_digest,
        "evaluator_lock_sha256": lock_digest,
        "runtime_image_digest": runtime_image_digest,
        "record_count": len(samples),
        "stable_id_field": "record_id",
    }
    if config_digest != manifest["task_config_sha256"]:
        raise BridgeError("task configuration digest was not canonicalized")
    (output / "run-manifest.json").write_bytes(canonical_json_bytes(manifest))


def mount_source(path: Path, *, label: str) -> str:
    """Return a stable, representable host path for a restricted bind mount."""

    if path.is_symlink():
        raise BridgeError(f"{label} must not be a symlink")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise BridgeError(f"{label} could not be resolved") from exc
    rendered = str(resolved)
    if any(character in rendered for character in (",", "\n", "\r", "\x00")):
        raise BridgeError(f"{label} cannot be represented in an OCI mount")
    return rendered


def _run_verified_worker(
    *,
    engine: str,
    image: str,
    selected: str,
    role: str,
    prepared: Path,
    output: Path,
    lock_digest: str,
) -> None:
    """Re-run the evaluator in the inspected image for this transaction.

    Prepared evaluator output is deliberately not an authority.  Completion
    creates a fresh output directory and only consumes the run produced by the
    inspected image during this call.
    """

    if engine not in {"docker", "podman"}:
        raise BridgeError("container engine must be docker or podman")
    if role not in {"baseline", "subject"}:
        raise BridgeError("evaluator worker role is invalid")
    model = prepared / f"evaluation/models/{role}"
    dataset = prepared / "evaluation/inputs/records.jsonl"
    if not model.is_dir() or model.is_symlink() or not dataset.is_file():
        raise BridgeError("prepared evaluator inputs are missing or unsafe")
    if dataset.is_symlink() or output.exists() or output.is_symlink():
        raise BridgeError("transaction evaluator output must be new")
    output.parent.mkdir(parents=True, exist_ok=True)
    output_name = output.name
    if output_name in {"", ".", ".."} or "/" in output_name:
        raise BridgeError("transaction evaluator output name is invalid")
    try:
        completed = run_evaluator_worker(
            engine=engine,
            image=image,
            entrypoint=(
                "python",
                "/opt/invarlock/examples/evaluator-level3.py",
                "worker",
            ),
            worker_arguments=(
                "--role",
                role,
                "--model",
                "/model",
                "--dataset",
                "/records.jsonl",
                "--output",
                f"/outputs/{output_name}",
            ),
            model_source=model,
            dataset_source=dataset,
            output=output,
            environment={
                "INVARLOCK_RUNTIME_IMAGE_ID": image,
                "INVARLOCK_EVALUATOR_LOCK_SHA256": lock_digest,
            },
            timeout_seconds=WORKER_TIMEOUT_SECONDS,
        )
    except OciEvaluationError as exc:
        raise BridgeError(
            f"{selected} worker control failed for {role}: {exc}"
        ) from exc
    if completed.returncode:
        raise BridgeError(
            completed.stderr.strip()
            or completed.stdout.strip()
            or f"{selected} worker failed for {role}"
        )


def load_run(path: Path, role: str, selected: str) -> tuple[dict[str, Any], bytes]:
    try:
        run = json.loads(_read_regular_file(path, label=f"{role} run manifest"))
    except (BridgeError, OSError, json.JSONDecodeError) as exc:
        raise BridgeError(f"{role} run provenance is missing") from exc
    if not isinstance(run, dict) or set(run) != RUN_FIELDS:
        raise BridgeError(f"{role} run provenance is incomplete")
    if (
        run["format"] != "invarlock/evaluator-level3-run-v1"
        or run["role"] != role
        or run["evaluator"] != selected
        or run["evaluator_version"] != EVALUATORS[selected]["version"]
        or run["stable_id_field"] != "record_id"
        or not isinstance(run["samples"], str)
        or not isinstance(run["samples_sha256"], str)
        or not isinstance(run["record_count"], int)
        or isinstance(run["record_count"], bool)
        or run["record_count"] != 102
        or IMAGE_ID.fullmatch(run["model_tree_sha256"]) is None
        or IMAGE_ID.fullmatch(f"sha256:{run['dataset_sha256']}") is None
        or IMAGE_ID.fullmatch(run["evaluator_lock_sha256"]) is None
        or IMAGE_ID.fullmatch(run["runtime_image_digest"]) is None
        or run["evaluator_lock_sha256"]
        != evaluator_lock_digest(selected, container=False)
        or run["task_config"] != task_config("/records.jsonl", selected)
        or run["task_config_sha256"] != digest(canonical_json_bytes(run["task_config"]))
        or run["execution_config"] != execution_config(selected)
        or run["execution_config_sha256"]
        != digest(canonical_json_bytes(run["execution_config"]))
    ):
        raise BridgeError(f"{role} run provenance is invalid")
    if run["samples"] != "samples.jsonl":
        raise BridgeError(f"{role} per-record output path is not canonical")
    samples = path.parent / run["samples"]
    try:
        sample_bytes = _read_regular_file(samples, label=f"{role} evaluator samples")
    except BridgeError:
        raise
    if (
        digest(sample_bytes) != run["samples_sha256"]
        or len(sample_bytes.splitlines()) != run["record_count"]
    ):
        raise BridgeError(f"{role} per-record output was tampered")
    return cast(dict[str, Any], run), sample_bytes


def load_canonical_samples(sample_bytes: bytes, *, role: str) -> list[dict[str, Any]]:
    """Decode the exact upstream JSONL snapshot for signed provenance."""

    samples: list[dict[str, Any]] = []
    for index, raw in enumerate(sample_bytes.splitlines(), 1):
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise BridgeError(f"{role} evaluator sample {index} is not JSON") from exc
        if (
            not isinstance(value, dict)
            or canonical_json_bytes(value, newline=False) != raw
        ):
            raise BridgeError(f"{role} evaluator sample {index} is not canonical JSON")
        samples.append(value)
    return samples


def adapt(samples: Path | bytes, schedule: Any, destination: Path) -> None:
    sample_bytes = (
        _read_regular_file(samples, label="evaluator samples")
        if isinstance(samples, Path)
        else samples
    )
    if not isinstance(sample_bytes, bytes):
        raise BridgeError("evaluator samples must be bytes or a regular file")
    lines = sample_bytes.splitlines()
    if len(lines) != len(schedule.records):
        raise BridgeError("one evaluator sample is required for every schedule record")
    output: list[dict[str, object]] = []
    for index, (raw, expected) in enumerate(
        zip(lines, schedule.records, strict=True), 1
    ):
        sample = json.loads(raw)
        if not isinstance(sample, dict) or not SAMPLE_FIELDS.issubset(sample):
            raise BridgeError(f"sample {index} lacks complete per-record facts")
        if any(
            not isinstance(sample[field], str)
            for field in ("record_id", "prompt", "target", "output", "status")
        ):
            raise BridgeError(f"sample {index} has invalid text facts")
        if (
            sample["record_id"] != expected.record_id
            or sample["prompt"] != expected.input_parts[0].text
            or sample["target"] != expected.expected_output
            or sample["input_sha256"] != digest(sample["prompt"].encode())
            or sample["target_sha256"] != digest(sample["target"].encode())
            or sample["output_sha256"] != digest(sample["output"].encode())
            or sample["status"] != "ok"
        ):
            raise BridgeError(
                f"sample {index} authenticated inputs or output are invalid"
            )
        reported = sample["reported_score"]
        if isinstance(reported, bool) or reported not in (0.0, 1.0):
            raise BridgeError(f"sample {index} has an invalid evaluator score")
        recomputed = 1.0 if sample["output"] == sample["target"] else 0.0
        if float(reported) != recomputed:
            raise BridgeError(f"sample {index} evaluator score disagrees with replay")
        output.append(
            {
                "record_id": expected.record_id,
                "input_sha256": expected.input_sha256,
                "status": "ok",
                "output_text": sample["output"],
                "output_sha256": sample["output_sha256"],
            }
        )
    destination.write_bytes(b"".join(canonical_json_bytes(item) for item in output))
    load_external_scoring_records_jsonl(destination, schedule=schedule)


def imported(role: str) -> dict[str, str]:
    root = f"imports/{role}"
    return {
        key: f"{root}/{name}"
        for key, name in {
            "identity": "model-artifact.identity.json",
            "receipt": "runtime-provider.receipt.json",
            "observation": "runtime-scoring.observation.json",
            "run_report": "report.json",
            "runtime_manifest": "runtime.manifest.json",
            "runtime_config": "run.yaml",
        }.items()
    }


def validate_completed_outputs(evidence: Path, receipt: Path, report: Path) -> None:
    try:
        evaluation = json.loads(
            _read_regular_file(
                evidence / "reports/evaluation.report.json",
                label="evaluation report",
            )
        )
        signed = json.loads(_read_regular_file(receipt, label="verification receipt"))
    except (BridgeError, OSError, json.JSONDecodeError) as exc:
        raise BridgeError(
            "the completed transaction is missing verified outputs"
        ) from exc
    statement = signed.get("statement") if isinstance(signed, dict) else None
    verdict = statement.get("verdict") if isinstance(statement, dict) else None
    if (
        not isinstance(evaluation, dict)
        or evaluation.get("verdict") != "pass"
        or evaluation.get("metric") != "exact_match"
        or not isinstance(verdict, dict)
        or verdict.get("ok") is not True
        or verdict.get("integrity_ok") is not True
        or verdict.get("policy_verdict") != "pass"
        or not report.is_file()
    ):
        raise BridgeError("the completed transaction did not verify a passing result")


def _validate_completion_paths(
    root: Path,
    *,
    build_attestation: Path,
    trust_root: Path,
    key_paths: tuple[tuple[Path, str], ...],
) -> None:
    if build_attestation.is_symlink():
        raise BridgeError("build attestation must not be a symlink")
    try:
        build_attestation.relative_to(root)
    except ValueError:
        pass
    else:
        raise BridgeError("build attestation must remain outside the transaction")
    if trust_root.exists() or trust_root.is_symlink():
        raise BridgeError("trust root must be new and outside the transaction")
    try:
        trust_root.relative_to(root)
    except ValueError:
        pass
    else:
        raise BridgeError("trust root must remain outside the transaction")
    for key_path, label in key_paths:
        try:
            key_path.relative_to(root)
        except ValueError:
            pass
        else:
            raise BridgeError(f"{label} must remain outside the transaction")


def complete(
    root: Path,
    prepared: Path,
    image: str,
    selected: str,
    *,
    container_engine: str = "docker",
    evidence_signing_key: Path | None = None,
    verifier_signing_key: Path | None = None,
    trust_root: Path | None = None,
    source_commit: str | None = None,
    base_image_id: str | None = None,
    build_attestation: Path | None = None,
    builder_public_key: Path | None = None,
) -> tuple[Path, Path, Path]:
    """Author strict import inputs and execute evaluate, verify, and report."""

    if (
        selected not in EVALUATORS
        or IMAGE_ID.fullmatch(image) is None
        or evidence_signing_key is None
        or verifier_signing_key is None
        or trust_root is None
        or source_commit is None
        or base_image_id is None
        or build_attestation is None
        or builder_public_key is None
    ):
        raise BridgeError(
            "an inspected image and caller-owned evidence, verifier, and trust roots "
            "builder public key, and build attestation are required"
        )
    if root.exists() or root.is_symlink():
        raise BridgeError("transaction workspace must be new")
    _validate_completion_paths(
        root,
        build_attestation=build_attestation,
        trust_root=trust_root,
        key_paths=(
            (evidence_signing_key, "evidence signing key"),
            (verifier_signing_key, "verifier signing key"),
            (builder_public_key, "builder public key"),
        ),
    )
    evidence_key = _external_ed25519_key(
        evidence_signing_key, label="evidence signing key"
    )
    verifier_key = _external_ed25519_key(
        verifier_signing_key, label="verifier signing key"
    )
    builder_key = _external_ed25519_public_key(
        builder_public_key, label="builder public key"
    )
    _require_distinct_signers(evidence_key, verifier_key, builder_key)
    lock_digest = evaluator_lock_digest(selected)
    _inspect_runtime_image(
        container_engine,
        image,
        selected,
        lock_digest,
        source_commit=source_commit,
        base_image_id=base_image_id,
        build_attestation=build_attestation,
        builder_public_key=builder_key,
    )
    prepared_image = (
        _read_regular_file(
            prepared / "runtime-image-id.txt", label="prepared runtime image identity"
        )
        .decode("ascii")
        .strip()
    )
    if prepared_image != image:
        raise BridgeError("prepared runtime image identity does not match inspection")
    try:
        request0 = yaml.safe_load(
            _read_regular_file(
                prepared / "evaluation/request.yaml", label="prepared request"
            )
        )
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise BridgeError("prepared request could not be read as YAML") from exc
    comparison0 = request0.get("comparison") if isinstance(request0, dict) else None
    if (
        not isinstance(comparison0, dict)
        or set(comparison0)
        != {"baseline", "subject", "dataset", "policy", "task", "metric"}
        or comparison0.get("metric") != "exact_match"
        or comparison0.get("policy") != "inputs/acceptance.json"
        or comparison0.get("task") != "text_causal"
    ):
        raise BridgeError("prepared request is not the fixed exact-match transaction")
    dataset0 = prepared / "evaluation/inputs/records.jsonl"
    dataset = comparison0.get("dataset")
    if not isinstance(dataset, dict):
        raise BridgeError("prepared request lacks the authenticated dataset")
    raw_dataset = _read_regular_file(dataset0, label="prepared dataset")
    expected_dataset = {
        "path": "inputs/records.jsonl",
        "sha256": DATASET_SHA256,
        "format": "jsonl",
        "name": DATASET_NAME,
        "split": "validation",
        "input_field": "prompt",
        "expected_output_field": "expected",
        "id_field": "id",
    }
    if dataset != expected_dataset or dataset["sha256"] != digest(raw_dataset):
        raise BridgeError("prepared dataset does not match the request digest")
    schedule = prepare_local_evaluation_schedule_bytes(
        LocalDatasetRequest(
            path=dataset0,
            sha256=digest(raw_dataset),
            format="jsonl",
            name=dataset["name"],
            split=dataset["split"],
            input_field=dataset["input_field"],
            expected_output_field=dataset["expected_output_field"],
            id_field=dataset["id_field"],
        ),
        raw_dataset,
    )
    root.mkdir(parents=True)
    (root / "inputs").mkdir()
    (root / "imports").mkdir()
    (root / "upstream").mkdir()
    (root / "verifier").mkdir()
    schedule_path = root / "inputs/schedule.json"
    schedule_path.write_bytes(canonical_runtime_behavioral_schedule_json(schedule))
    schedule = load_runtime_behavioral_schedule(schedule_path)
    policy = _read_regular_file(
        prepared / "evaluation/inputs/acceptance.json", label="prepared policy"
    )
    expected_policy = {
        "resolved_policy": {
            "metrics": {
                "exact_match": {
                    "delta_min_pp": -20.0,
                    "maximum_interval_width_pp": 20.0,
                    "minimum_record_count": 102,
                    "minimum_side_accuracy": MINIMUM_SIDE_ACCURACY,
                }
            }
        }
    }
    if json.loads(policy) != expected_policy:
        raise BridgeError("prepared exact-match policy is not the fixed example policy")
    for role in ("baseline", "subject"):
        output = root / f"upstream/{role}/result"
        _run_verified_worker(
            engine=container_engine,
            image=image,
            selected=selected,
            role=role,
            prepared=prepared,
            output=output,
            lock_digest=lock_digest,
        )
    runs = {
        role: load_run(
            root / f"upstream/{role}/result/run-manifest.json", role, selected
        )
        for role in ("baseline", "subject")
    }
    if (
        runs["baseline"][0]["task_config_sha256"]
        != runs["subject"][0]["task_config_sha256"]
        or runs["baseline"][0]["execution_config_sha256"]
        != runs["subject"][0]["execution_config_sha256"]
    ):
        raise BridgeError(
            "baseline and subject used different evaluator configurations"
        )
    if any(
        run["runtime_image_digest"] != image
        or run["evaluator_lock_sha256"] != lock_digest
        or run["dataset_sha256"] != dataset["sha256"]
        for run, _samples in runs.values()
    ):
        raise BridgeError(
            "fresh worker runs are not bound to the inspected image, lock, and dataset"
        )
    (root / "inputs/acceptance.json").write_bytes(policy)
    provenance = canonical_json_bytes(
        {
            "format": "invarlock/evaluator-level3-provenance-v2",
            "evaluator": selected,
            "evaluator_lock_sha256": lock_digest,
            "runtime_image_digest": image,
            "source_commit": source_commit,
            "base_image_id": base_image_id,
            "task_config": runs["baseline"][0]["task_config"],
            "task_config_sha256": runs["baseline"][0]["task_config_sha256"],
            "execution_config": runs["baseline"][0]["execution_config"],
            "execution_config_sha256": runs["baseline"][0]["execution_config_sha256"],
            "runs": {
                role: {
                    "manifest": runs[role][0],
                    "samples": load_canonical_samples(runs[role][1], role=role),
                }
                for role in ("baseline", "subject")
            },
        }
    )
    (root / "inputs/evaluator-provenance.json").write_bytes(provenance)
    provider = HFTransformersProvider()
    sides: dict[str, Any] = {}
    anchors: dict[str, str] = {}
    for role in ("baseline", "subject"):
        records_path = root / f"imports/{role}-records.jsonl"
        adapt(runs[role][1], schedule, records_path)
        original = comparison0[role]
        if (
            not isinstance(original, dict)
            or original.get("artifact") != EXPECTED_MODEL_ARTIFACTS[role]
            or not isinstance(original.get("runtime"), dict)
            or original["runtime"].get("provider") != "hf_transformers"
            or set(original["runtime"]) != {"provider", "settings"}
        ):
            raise BridgeError(f"{role} is not the canonical pinned Qwen3 model")
        settings = original["runtime"]["settings"]
        if not isinstance(settings, dict):
            raise BridgeError(f"{role} runtime settings are not canonical")
        expected_settings = {
            "batch_size": BATCH_SIZE,
            "checkpoint_tree_sha256": EXPECTED_MODEL_TREE_DIGESTS[role],
            "context_length": 64,
            "max_output_tokens": MAX_GENERATION_TOKENS,
            "offline": True,
            "seed": SEED,
            "timeout_seconds": PER_RECORD_TIMEOUT_SECONDS,
            "tokenizer_metadata_sha256": EXPECTED_TOKENIZER_DIGESTS[role],
        }
        if set(settings) != set(expected_settings) or any(
            settings.get(key) != value for key, value in expected_settings.items()
        ):
            raise BridgeError(f"{role} runtime settings are not canonical")
        checkpoint = prepared / f"evaluation/models/{role}"
        if (checkpoint / "generation_config.json").exists() or (
            checkpoint / "generation_config.json"
        ).is_symlink():
            raise BridgeError(
                f"{role} snapshot does not leave generation defaults to the task"
            )
        try:
            from transformers import AutoTokenizer

            from invarlock.runtime_providers.hf_transformers import (
                hf_tokenizer_contract_sha256,
            )

            tokenizer = AutoTokenizer.from_pretrained(
                checkpoint, local_files_only=True, trust_remote_code=False
            )
            observed_tokenizer_digest = hf_tokenizer_contract_sha256(tokenizer)
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            raise BridgeError(
                f"{role} tokenizer identity could not be authenticated"
            ) from exc
        if settings["tokenizer_metadata_sha256"] != observed_tokenizer_digest:
            raise BridgeError(
                f"{role} tokenizer identity does not match the checkpoint"
            )
        identity = provider.authenticate_artifact(
            ModelRuntimeSpec(
                "hf_transformers", original["artifact"]["model_id"], settings
            ),
            checkpoint,
        )
        if runs[role][0]["model_tree_sha256"] != settings.get("checkpoint_tree_sha256"):
            raise BridgeError(f"{role} run used a different authenticated checkpoint")
        execution = runs[role][0]["execution_config"]
        if (
            settings.get("seed") != execution["seed"]
            or settings.get("batch_size") != execution["batch_size"]
            or settings.get("max_output_tokens") != execution["max_generation_tokens"]
        ):
            raise BridgeError(
                f"{role} runtime settings do not match evaluator execution"
            )
        side = write_runtime_import_side(
            root / f"imports/{role}",
            role=cast(Literal["baseline", "subject"], role),
            schedule=schedule,
            policy_digest=sha256_digest(policy),
            artifact_identity=identity,
            records=load_external_scoring_records_jsonl(
                records_path, schedule=schedule
            ),
            plugin=RuntimeProviderPluginIdentity(
                "hf_transformers", "invarlock", INVARLOCK_VERSION
            ),
            backend=RuntimeBackendIdentity(
                f"{selected}-hf",
                EVALUATORS[selected]["version"],
                digest(provenance),
                None,
                runs[role][0]["task_config_sha256"],
            ),
            capabilities=provider.capabilities(),
            execution_settings=RuntimeExecutionSettings(
                settings["seed"],
                settings["context_length"],
                settings["batch_size"],
                settings["max_output_tokens"],
                settings["timeout_seconds"],
                False,
            ),
            device=RuntimeDeviceFacts(
                str(execution["device"]), f"container-{execution['device']}"
            ),
            runtime_image_ref=image,
            runtime_image_digest=image,
            generated_at_utc=datetime.now(tz=UTC).isoformat(),
        )
        sides[role] = side
        anchors[role] = sha256_digest(side.provider_evidence.artifact_identity_bytes)
    write_runtime_import_paired_records(
        root / "imports/paired-records.json",
        schedule=schedule,
        metric="exact_match",
        baseline=sides["baseline"],
        subject=sides["subject"],
    )

    def side_descriptor(role: str) -> dict[str, Any]:
        original = comparison0[role]
        return {
            "artifact": {
                key: original["artifact"][key] for key in ("model_id", "locator")
            },
            "runtime": original["runtime"],
        }

    request = {
        "format_version": "invarlock/evaluation-request-v1",
        "comparison": {
            "baseline": side_descriptor("baseline"),
            "subject": side_descriptor("subject"),
            "dataset": "inputs/schedule.json",
            "policy": "inputs/acceptance.json",
            "task": "text_causal",
            "metric": "exact_match",
        },
        "execution": {
            "mode": "import",
            "records": "imports/paired-records.json",
            "schedule": "inputs/schedule.json",
            "baseline": imported("baseline"),
            "subject": imported("subject"),
        },
        "observations": [
            {
                "id": f"{selected}-provenance",
                "kind": "evaluator_provenance",
                "scope": "comparison",
                "path": "inputs/evaluator-provenance.json",
            }
        ],
        "output": {"evidence": "evidence"},
    }
    request_path = root / "request.yaml"
    request_path.write_text(yaml.safe_dump(request, sort_keys=False))
    evidence_fingerprint = public_key_fingerprint(evidence_key.public_key())
    trust = create_trust_material(
        transaction_root=root,
        evidence_key=evidence_signing_key,
        verifier_key_bytes=read_external_file(
            verifier_signing_key, label="verifier signing key"
        ),
        evidence_fingerprint=evidence_fingerprint,
        verifier_fingerprint=public_key_fingerprint(verifier_key.public_key()),
        trust_root=trust_root,
        policy_bytes=policy,
        verifier_identity=f"invarlock-example/{selected}-verifier",
        anchors={
            "baseline_artifact_digest": anchors["baseline"],
            "subject_artifact_digest": anchors["subject"],
            "schedule_digest": f"sha256:{schedule.schedule_sha256}",
            "baseline_runtime_digest": image,
            "subject_runtime_digest": image,
            "evidence_signer_fingerprint": evidence_fingerprint,
        },
    )
    trust_path = trust.trusted_inputs
    evidence, receipt, report = (
        root / "evidence",
        root / "verifier/verification.receipt.json",
        root / "comparison-report.html",
    )
    commands = [
        [
            "evaluate",
            str(request_path),
            "--signing-key",
            str(evidence_signing_key),
            "--json",
        ],
        [
            "verify",
            str(evidence),
            "--trust-profile",
            str(trust_path),
            "--receipt",
            str(receipt),
            "--json",
        ],
        ["report", str(evidence), "--html", str(report)],
    ]
    for arguments in commands:
        _run_local_cli([sys.executable, "-m", "invarlock", *arguments])
    validate_completed_outputs(evidence, receipt, report)
    return evidence, receipt, report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    worker_parser = subparsers.add_parser("worker")
    worker_parser.add_argument("--role", choices=("baseline", "subject"), required=True)
    worker_parser.add_argument("--model", type=Path, required=True)
    worker_parser.add_argument("--dataset", type=Path, required=True)
    worker_parser.add_argument("--output", type=Path, required=True)
    bridge_parser = subparsers.add_parser("complete")
    bridge_parser.add_argument("--workspace", type=Path, required=True)
    bridge_parser.add_argument("--prepared", type=Path, required=True)
    bridge_parser.add_argument("--runtime-image", required=True)
    bridge_parser.add_argument("--evaluator", choices=tuple(EVALUATORS), required=True)
    bridge_parser.add_argument(
        "--container-engine", choices=("docker", "podman"), default="docker"
    )
    bridge_parser.add_argument("--evidence-signing-key", type=Path, required=True)
    bridge_parser.add_argument("--verifier-signing-key", type=Path, required=True)
    bridge_parser.add_argument("--trust-root", type=Path, required=True)
    bridge_parser.add_argument("--builder-public-key", type=Path, required=True)
    bridge_parser.add_argument("--source-commit", required=True)
    bridge_parser.add_argument("--base-image-id", required=True)
    bridge_parser.add_argument("--build-attestation", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "worker":
            worker(args.role, args.model, args.dataset, args.output)
        else:
            evidence, receipt, report = complete(
                args.workspace.resolve(),
                args.prepared.resolve(),
                args.runtime_image,
                args.evaluator,
                container_engine=args.container_engine,
                evidence_signing_key=args.evidence_signing_key.resolve(),
                verifier_signing_key=args.verifier_signing_key.resolve(),
                trust_root=args.trust_root.expanduser().absolute(),
                source_commit=args.source_commit,
                base_image_id=args.base_image_id,
                builder_public_key=args.builder_public_key.resolve(),
                build_attestation=(
                    args.build_attestation.resolve()
                    if args.build_attestation is not None
                    else None
                ),
            )
            print(f"Evidence: {evidence}\nReceipt: {receipt}\nReport: {report}")
    except (BridgeError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"FAIL {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
