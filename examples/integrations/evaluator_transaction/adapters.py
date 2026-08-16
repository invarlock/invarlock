"""Pinned native evaluator adapters used by the maintained transaction."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, cast

from invarlock.evidence_pack_contract import canonical_json_bytes

from .config import (
    INSPECT_RAW_CHAT_TEMPLATE,
    MAX_GENERATION_TOKENS,
    PAD_TOKEN_POLICY,
    SEED,
    BridgeError,
    execution_config,
)
from .corpora import profile_for_dataset


def _restore_inspect_causal_boundary(completion: str, target: str) -> str:
    """Restore the target-leading whitespace removed by Inspect's HF decoder."""

    prefix_length = len(target) - len(target.lstrip())
    prefix = target[:prefix_length]
    if prefix and not completion.startswith(prefix):
        return prefix + completion
    return completion


def _records(dataset_bytes: bytes) -> list[dict[str, str]]:
    try:
        profile_for_dataset(dataset_bytes)
        values = [json.loads(line) for line in dataset_bytes.splitlines()]
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise BridgeError(str(exc)) from exc
    return cast(list[dict[str, str]], values)


class _HfGreedyGenerator:
    """The pinned local model adapter used by both native evaluator runners."""

    def __init__(self, model_path: Path) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise BridgeError(
                "the evaluator image lacks the Hugging Face runtime"
            ) from exc

        self._torch = torch
        execution = execution_config()
        if (model_path / "generation_config.json").exists() or (
            model_path / "generation_config.json"
        ).is_symlink():
            raise BridgeError("model snapshot must not provide generation defaults")
        torch.manual_seed(execution["seed"])
        torch.set_num_threads(execution["torch_num_threads"])
        dtype = getattr(torch, execution["dtype"], None)
        if dtype is None:
            raise BridgeError("the evaluator dtype is unavailable")
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
        self._model = (
            AutoModelForCausalLM.from_pretrained(
                model_path,
                local_files_only=True,
                dtype=dtype,
                trust_remote_code=False,
            )
            .to(execution["device"])
            .eval()
        )

    def generate(self, prompts: list[str]) -> list[str]:
        execution = execution_config()
        output: list[str] = []
        with self._torch.inference_mode():
            batch_size = execution["batch_size"]
            for offset in range(0, len(prompts), batch_size):
                batch = prompts[offset : offset + batch_size]
                encoded = self._tokenizer(
                    batch,
                    add_special_tokens=execution["tokenizer_add_special_tokens"],
                    padding=True,
                    return_tensors="pt",
                )
                encoded = {
                    key: value.to(execution["device"]) for key, value in encoded.items()
                }
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
        if self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()


def _generate(model_path: Path, dataset_bytes: bytes) -> list[dict[str, str]]:
    """Generate one greedy token per record for compatibility diagnostics."""

    records = _records(dataset_bytes)
    generator = _HfGreedyGenerator(model_path)
    outputs = generator.generate([record["prompt"] for record in records])
    generator.close()
    if len(outputs) != len(records):
        raise BridgeError("the model adapter returned an incomplete result")
    return [
        {**record, "output": text}
        for record, text in zip(records, outputs, strict=True)
    ]


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
    execution = execution_config()
    task = Task(
        dataset=MemoryDataset(
            [
                Sample(input=item["prompt"], target=item["expected"], id=item["id"])
                for item in records
            ]
        ),
        solver=generate(),
        scorer=match(location="exact", ignore_case=False),
        name="invarlock-evaluator-transaction",
    )
    logs = inspect_eval(
        task,
        model="hf/invarlock",
        model_args={
            "model_path": str(model_path),
            "device": execution["device"],
            "batch_size": execution["batch_size"],
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
        max_connections=execution["batch_size"],
        max_samples=execution["batch_size"],
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
                "Inspect AI changed the evaluator sample identity or target"
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
        raise BridgeError(
            "OpenAI Evals changed the evaluator sample identity or target"
        )
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
                name="invarlock-evaluator-transaction.default",
                seed=SEED,
                max_tokens=MAX_GENERATION_TOKENS,
                num_few_shot=0,
            )
            recorder = DummyRecorder(
                RunSpec(
                    completion_fns=["invarlock/hf"],
                    eval_name="invarlock-evaluator-transaction.default",
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


__all__ = [
    "_HfGreedyGenerator",
    "_OpenAICompletionResult",
    "_OpenAIHfCompletionFn",
    "_generate",
    "_openai_event_to_sample",
    "_records",
    "_restore_inspect_causal_boundary",
    "_run_inspect_ai",
    "_run_openai_evals",
    "_run_upstream_evaluator",
]
