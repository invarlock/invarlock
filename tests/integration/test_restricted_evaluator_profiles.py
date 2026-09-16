"""Opt-in installed-profile checks with local, generated tiny HF weights."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("INVARLOCK_RUN_RESTRICTED_EVALUATOR_SMOKE") != "1",
        reason="requires the separately installed restricted evaluator wheels",
    ),
]


@pytest.fixture
def tiny_model(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.syspath_prepend(str(ROOT))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
    monkeypatch.setenv("TOKENIZERS_PARALLELISM", "false")
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf-home"))
    monkeypatch.setenv("INVARLOCK_CORPUS_PROFILE", "quick")
    for name in ("nltk", "rouge_score", "sqlitedict"):
        assert importlib.util.find_spec(name) is None, f"unexpected package: {name}"

    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    torch.manual_seed(42)
    torch.set_num_threads(1)
    tokenizer = Tokenizer(
        WordLevel({"[UNK]": 0, "[EOS]": 1, "red": 2, "blue": 3}, unk_token="[UNK]")
    )
    tokenizer.pre_tokenizer = Whitespace()
    wrapped = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        eos_token="[EOS]",
        pad_token="[EOS]",
    )
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=4,
            n_positions=512,
            n_embd=8,
            n_layer=1,
            n_head=1,
            bos_token_id=1,
            eos_token_id=1,
            pad_token_id=1,
        )
    ).eval()
    destination = tmp_path / "tiny-model"
    model.save_pretrained(destination)
    wrapped.save_pretrained(destination)
    (destination / "generation_config.json").unlink()
    return destination


def test_lm_hf_generation_and_exact_match_without_nltk(
    tiny_model: Path, tmp_path: Path
) -> None:
    assert importlib.metadata.version("lm-eval") == "0.4.12+invarlock.exactmatch.1"
    from lm_eval import evaluator
    from lm_eval.api.model import CachingLM
    from lm_eval.tasks import TaskManager

    dataset = tmp_path / "samples.jsonl"
    dataset.write_text(
        '{"prompt":"red","expected":"red"}\n{"prompt":"blue","expected":"red"}\n'
    )
    task = {
        "task": "restricted_profile_smoke",
        "dataset_path": "json",
        "dataset_kwargs": {"data_files": {"test": str(dataset)}},
        "test_split": "test",
        "output_type": "generate_until",
        "doc_to_text": "{{prompt}}",
        "doc_to_target": "{{expected}}",
        "generation_kwargs": {"do_sample": False, "max_gen_toks": 1, "until": ["\n"]},
        "metric_list": [
            {"metric": "exact_match", "aggregation": "mean", "higher_is_better": True}
        ],
    }
    result = evaluator.simple_evaluate(
        model="hf",
        model_args={
            "pretrained": str(tiny_model),
            "backend": "causal",
            "dtype": "float32",
            "trust_remote_code": False,
        },
        tasks=[task],
        task_manager=TaskManager(include_defaults=False),
        device="cpu",
        batch_size=1,
        bootstrap_iters=0,
        log_samples=True,
        random_seed=42,
        numpy_random_seed=42,
        torch_random_seed=42,
    )
    samples = result["samples"]["restricted_profile_smoke"]
    assert len(samples) == 2
    outputs = [row["resps"][0][0] for row in samples]
    assert all(isinstance(output, str) for output in outputs)
    expected_score = sum(output == "red" for output in outputs) / 2
    assert (
        result["results"]["restricted_profile_smoke"]["exact_match,none"]
        == expected_score
    )
    with pytest.raises(RuntimeError, match="response caching is unavailable"):
        CachingLM(None, str(tmp_path / "forbidden-cache.sqlite"))
    with pytest.raises(ModuleNotFoundError, match="rouge_score"):
        importlib.import_module("lm_eval.tasks.truthfulqa.utils")
    print(
        json.dumps(
            {"evaluator": "lm-eval", "outputs": outputs, "exact_match": expected_score}
        )
    )


def test_openai_native_match_runs_hf_without_nltk(
    tiny_model: Path, monkeypatch
) -> None:
    monkeypatch.setenv("INVARLOCK_EVALUATOR", "openai-evals")
    assert importlib.metadata.version("evals") == "3.0.1.post1+invarlock.match.1"
    from examples.integrations.evaluator_transaction.adapters import _run_openai_evals
    from examples.integrations.evaluator_transaction.corpora import (
        quick_records,
        records_jsonl,
    )

    data = records_jsonl(quick_records())
    generated, scores = _run_openai_evals(tiny_model, data)
    assert len(generated) == len(scores) == 102
    for row, (score, detail) in zip(generated, scores, strict=True):
        assert isinstance(row["output"], str)
        assert score == float(row["output"] == row["expected"])
        assert detail["native_correct"] == row["output"].startswith(row["expected"])
    with pytest.raises(ModuleNotFoundError, match="nltk"):
        importlib.import_module("evals.elsuite.skill_acquisition.utils")
    print(
        json.dumps(
            {
                "evaluator": "openai-evals",
                "count": len(scores),
                "exact_match": sum(x[0] for x in scores) / len(scores),
                "first_output": generated[0]["output"],
            }
        )
    )
