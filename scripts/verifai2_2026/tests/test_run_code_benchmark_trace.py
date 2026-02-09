from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts.verifai2_2026 import run_code_benchmark_trace


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=True) + "\n" for r in rows), encoding="utf-8"
    )


def test_skip_generation_requires_existing_completions(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    tasks = tmp_path / "tasks.jsonl"
    _write_jsonl(tasks, [{"id": "a", "prompt": "x", "tests": "assert True"}])
    out_dir = tmp_path / "out"

    rc = run_code_benchmark_trace.main(
        [
            "--tasks",
            str(tasks),
            "--out-dir",
            str(out_dir),
            "--verifier-name",
            "mbpp",
            "--model-id",
            "m",
            "--model-revision",
            "r",
            "--model",
            "m",
            "--tokenizer-id",
            "t",
            "--tokenizer-revision",
            "tr",
            "--decoding-method",
            "greedy",
            "--temperature",
            "0",
            "--top-p",
            "1",
            "--top-k",
            "0",
            "--max-new-tokens",
            "4",
            "--seed",
            "1",
            "--harness-name",
            "h",
            "--skip-generation",
        ]
    )
    assert rc == 2
    assert "missing completions file" in capsys.readouterr().err


def test_happy_path_calls_both_stages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tasks = tmp_path / "tasks.jsonl"
    _write_jsonl(tasks, [{"id": "a", "prompt": "x", "tests": "assert True"}])
    out_dir = tmp_path / "out"

    called: dict[str, list[list[str]]] = {"gen": [], "trace": []}

    def _fake_gen(argv: list[str] | None = None) -> int:  # noqa: ANN001
        assert argv is not None
        called["gen"].append(list(argv))
        # Create the expected completions file.
        p = Path(argv[argv.index("--out") + 1])
        _write_jsonl(p, [{"id": "a", "attempt_id": 0, "completion": ""}])
        return 0

    def _fake_trace(argv: list[str] | None = None) -> int:  # noqa: ANN001
        assert argv is not None
        called["trace"].append(list(argv))
        # Create required outputs: prompt_set/cases/trace
        out_prompt = Path(argv[argv.index("--prompt-set-out") + 1])
        out_cases = Path(argv[argv.index("--cases-out") + 1])
        out_trace = Path(argv[argv.index("--trace-out") + 1])
        out_prompt.write_text(
            json.dumps(
                {
                    "items": [{"id": "a", "sha256": "0" * 64}],
                    "dataset": {"name": "x", "split": "test", "revision": "unknown"},
                    "digest_sha256": "0" * 64,
                    "mode": "hash_only",
                },
                ensure_ascii=True,
            ),
            encoding="utf-8",
        )
        _write_jsonl(out_cases, [{"id": "a", "verdict": "pass"}])
        out_trace.write_text(
            json.dumps(
                {
                    "schema_version": "verifier_trace.v1",
                    "verifier": {
                        "name": "mbpp",
                        "kind": "code_execution",
                        "harness": {"name": "h", "version": ""},
                    },
                    "trace_contract": {
                        "prompt_set": {
                            "mode": "hash_only",
                            "dataset": {
                                "name": "x",
                                "split": "test",
                                "revision": "unknown",
                            },
                            "items": [{"id": "a", "sha256": "0" * 64}],
                            "digest_sha256": "0" * 64,
                        },
                        "model": {"id": "m", "revision": "r"},
                        "tokenizer": {"id": "t", "revision": "tr"},
                        "decoding": {
                            "method": "greedy",
                            "max_new_tokens": 4,
                            "temperature": 0.0,
                            "top_p": 1.0,
                            "seed": 1,
                        },
                    },
                    "results": {
                        "metric": {"name": "pass@1", "value": 1.0},
                        "cases": [{"id": "a", "verdict": "pass"}],
                    },
                },
                ensure_ascii=True,
            ),
            encoding="utf-8",
        )
        return 0

    monkeypatch.setattr(
        run_code_benchmark_trace.generate_completions, "main", _fake_gen
    )
    monkeypatch.setattr(
        run_code_benchmark_trace.run_verifier_trace_pipeline, "main", _fake_trace
    )

    rc = run_code_benchmark_trace.main(
        [
            "--tasks",
            str(tasks),
            "--out-dir",
            str(out_dir),
            "--verifier-name",
            "mbpp",
            "--model-id",
            "m",
            "--model-revision",
            "r",
            "--model",
            "m",
            "--model-load-revision",
            "",
            "--tokenizer-id",
            "t",
            "--tokenizer-revision",
            "tr",
            "--decoding-method",
            "sample",
            "--temperature",
            "0.2",
            "--top-p",
            "0.95",
            "--top-k",
            "0",
            "--max-new-tokens",
            "4",
            "--seed",
            "42",
            "--num-samples",
            "2",
            "--num-beams",
            "0",
            "--harness-name",
            "h",
            "--limit",
            "1",
        ]
    )
    assert rc == 0
    assert called["gen"]
    assert called["trace"]
    assert (out_dir / "completions.jsonl").exists()
    assert (out_dir / "verifier_trace.v1.json").exists()


def test_generation_failure_short_circuits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tasks = tmp_path / "tasks.jsonl"
    _write_jsonl(tasks, [{"id": "a", "prompt": "x", "tests": "assert True"}])
    out_dir = tmp_path / "out"

    monkeypatch.setattr(
        run_code_benchmark_trace.generate_completions, "main", lambda _a=None: 2
    )  # noqa: E731

    rc = run_code_benchmark_trace.main(
        [
            "--tasks",
            str(tasks),
            "--out-dir",
            str(out_dir),
            "--verifier-name",
            "mbpp",
            "--model-id",
            "m",
            "--model-revision",
            "r",
            "--model",
            "m",
            "--tokenizer-id",
            "t",
            "--tokenizer-revision",
            "tr",
            "--decoding-method",
            "greedy",
            "--temperature",
            "0",
            "--top-p",
            "1",
            "--top-k",
            "0",
            "--max-new-tokens",
            "4",
            "--seed",
            "1",
            "--harness-name",
            "h",
        ]
    )
    assert rc == 2


def test_skip_generation_reuses_existing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tasks = tmp_path / "tasks.jsonl"
    _write_jsonl(tasks, [{"id": "a", "prompt": "x", "tests": "assert True"}])
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out_dir / "completions.jsonl", [{"id": "a", "completion": ""}])

    monkeypatch.setattr(
        run_code_benchmark_trace.generate_completions, "main", lambda _a=None: 1
    )  # noqa: E731
    monkeypatch.setattr(
        run_code_benchmark_trace.run_verifier_trace_pipeline, "main", lambda _a=None: 0
    )  # noqa: E731

    rc = run_code_benchmark_trace.main(
        [
            "--tasks",
            str(tasks),
            "--out-dir",
            str(out_dir),
            "--verifier-name",
            "mbpp",
            "--model-id",
            "m",
            "--model-revision",
            "r",
            "--model",
            "m",
            "--tokenizer-id",
            "t",
            "--tokenizer-revision",
            "tr",
            "--decoding-method",
            "greedy",
            "--temperature",
            "0",
            "--top-p",
            "1",
            "--top-k",
            "0",
            "--max-new-tokens",
            "4",
            "--seed",
            "1",
            "--harness-name",
            "h",
            "--skip-generation",
        ]
    )
    assert rc == 0
