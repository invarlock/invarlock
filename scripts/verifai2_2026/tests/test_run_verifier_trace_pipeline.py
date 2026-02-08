from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.verifai2_2026 import run_verifier_trace_pipeline


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=True) + "\n" for r in rows), encoding="utf-8"
    )


def test_read_jsonl_ids(tmp_path: Path) -> None:
    p = tmp_path / "x.jsonl"
    p.write_text("\n", encoding="utf-8")
    with p.open("a", encoding="utf-8") as f:
        for row in [{"id": "a"}, {}, {"id": 1}, {"id": "b"}]:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")
    assert run_verifier_trace_pipeline._read_jsonl_ids(p, id_field="id") == ["a", "b"]


def test_validate_decoding_errors() -> None:
    assert (
        run_verifier_trace_pipeline._validate_decoding(
            method="greedy", temperature=0.0, top_p=1.0, top_k=0, num_samples=0, k=0
        )
        == []
    )
    assert (
        run_verifier_trace_pipeline._validate_decoding(
            method="sample", temperature=0.7, top_p=0.9, top_k=50, num_samples=3, k=2
        )
        == []
    )
    errs = run_verifier_trace_pipeline._validate_decoding(
        method="greedy", temperature=1.0, top_p=1.0, top_k=0, num_samples=2, k=3
    )
    assert any("temperature" in e for e in errs)
    assert any("num_samples" in e for e in errs)
    errs2 = run_verifier_trace_pipeline._validate_decoding(
        method="greedy", temperature=0.0, top_p=0.5, top_k=1, num_samples=0, k=0
    )
    assert any("top_p" in e for e in errs2)
    assert any("top_k" in e for e in errs2)


def _common_args(*, tmp_path: Path) -> list[str]:
    return [
        "--prompt-set-out",
        str(tmp_path / "prompt_set.json"),
        "--cases-out",
        str(tmp_path / "cases.jsonl"),
        "--trace-out",
        str(tmp_path / "trace.json"),
        "--verifier-name",
        "humaneval",
        "--verifier-kind",
        "code_execution",
        "--harness-name",
        "h",
        "--harness-version",
        "1",
        "--model-id",
        "m",
        "--model-revision",
        "r",
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
        "8",
        "--seed",
        "0",
    ]


def test_main_requires_prompts_or_tasks(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = run_verifier_trace_pipeline.main(_common_args(tmp_path=tmp_path))
    assert rc == 2
    assert "--prompts or --tasks is required" in capsys.readouterr().err


def test_main_decoding_validation_failure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    prompts = tmp_path / "prompts.jsonl"
    _write_jsonl(prompts, [{"id": "a", "prompt": "x"}])
    rc = run_verifier_trace_pipeline.main(
        [
            "--prompts",
            str(prompts),
            *(_common_args(tmp_path=tmp_path)),
            "--temperature",
            "1.0",
        ]
    )
    assert rc == 2
    assert "requires temperature=0.0" in capsys.readouterr().err


def test_main_prompt_set_failure(tmp_path: Path) -> None:
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text("", encoding="utf-8")
    rc = run_verifier_trace_pipeline.main(
        [
            "--prompts",
            str(prompts),
            *(_common_args(tmp_path=tmp_path)),
        ]
    )
    assert rc == 2


def test_main_code_tests_backend_missing_inputs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    tasks = tmp_path / "tasks.jsonl"
    _write_jsonl(tasks, [{"id": "a", "prompt": "", "tests": "assert True"}])
    rc = run_verifier_trace_pipeline.main(
        ["--tasks", str(tasks), *(_common_args(tmp_path=tmp_path))]
    )
    assert rc == 2
    assert "required for backend=code_tests" in capsys.readouterr().err


def test_main_harness_backend_missing_results(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    prompts = tmp_path / "prompts.jsonl"
    _write_jsonl(prompts, [{"id": "a", "prompt": "x"}])
    rc = run_verifier_trace_pipeline.main(
        [
            "--backend",
            "harness_jsonl",
            "--prompts",
            str(prompts),
            *(_common_args(tmp_path=tmp_path)),
        ]
    )
    assert rc == 2
    assert "required for backend=harness_jsonl" in capsys.readouterr().err


def test_main_cases_backend_missing_cases(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    prompts = tmp_path / "prompts.jsonl"
    _write_jsonl(prompts, [{"id": "a", "prompt": "x"}])
    rc = run_verifier_trace_pipeline.main(
        [
            "--backend",
            "cases_jsonl",
            "--prompts",
            str(prompts),
            *(_common_args(tmp_path=tmp_path)),
        ]
    )
    assert rc == 2
    assert "required for backend=cases_jsonl" in capsys.readouterr().err


def test_main_cases_backend_no_ids(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    prompts = tmp_path / "prompts.jsonl"
    _write_jsonl(prompts, [{"id": "a", "prompt": "x"}])
    cases = tmp_path / "cases_in.jsonl"
    _write_jsonl(cases, [{}])

    rc = run_verifier_trace_pipeline.main(
        [
            "--backend",
            "cases_jsonl",
            "--prompts",
            str(prompts),
            "--cases",
            str(cases),
            *(_common_args(tmp_path=tmp_path)),
        ]
    )
    assert rc == 2
    assert "produced no ids" in capsys.readouterr().err


def test_main_success_code_tests_backend(tmp_path: Path) -> None:
    tasks = tmp_path / "tasks.jsonl"
    completions = tmp_path / "completions.jsonl"
    _write_jsonl(
        tasks,
        [
            {"id": "a", "prompt": "", "tests": "x = 1\nassert x == 1\n"},
        ],
    )
    _write_jsonl(completions, [{"id": "a", "completion": ""}])

    rc = run_verifier_trace_pipeline.main(
        [
            "--backend",
            "code_tests",
            "--tasks",
            str(tasks),
            "--completions",
            str(completions),
            *(_common_args(tmp_path=tmp_path)),
        ]
    )
    assert rc == 0
    trace = json.loads((tmp_path / "trace.json").read_text(encoding="utf-8"))
    assert trace["schema_version"] == "verifier_trace.v1"
    assert trace["results"]["summary"]["n_total"] == 1
    assert trace["results"]["cases"][0]["verdict"] == "pass"


def test_main_code_tests_backend_propagates_verifier_failure(tmp_path: Path) -> None:
    tasks = tmp_path / "tasks.jsonl"
    completions = tmp_path / "completions.jsonl"
    _write_jsonl(tasks, [{"id": "a", "prompt": "", "tests": "assert True"}])
    completions.write_text("", encoding="utf-8")

    rc = run_verifier_trace_pipeline.main(
        [
            "--backend",
            "code_tests",
            "--tasks",
            str(tasks),
            "--completions",
            str(completions),
            *(_common_args(tmp_path=tmp_path)),
        ]
    )
    assert rc == 2


def test_main_success_harness_jsonl_backend_and_non_strict(tmp_path: Path) -> None:
    prompts = tmp_path / "prompts.jsonl"
    _write_jsonl(prompts, [{"id": "a", "prompt": "PROMPT"}])
    harness = tmp_path / "harness.jsonl"
    _write_jsonl(harness, [{"id": "a", "verdict": "nope", "completion": "print(1)"}])

    rc = run_verifier_trace_pipeline.main(
        [
            "--backend",
            "harness_jsonl",
            "--prompts",
            str(prompts),
            "--harness-results",
            str(harness),
            "--harness-non-strict",
            *(_common_args(tmp_path=tmp_path)),
        ]
    )
    assert rc == 0
    trace = json.loads((tmp_path / "trace.json").read_text(encoding="utf-8"))
    assert trace["results"]["cases"][0]["verdict"] == "error"


def test_main_harness_backend_propagates_ingest_failure(tmp_path: Path) -> None:
    prompts = tmp_path / "prompts.jsonl"
    _write_jsonl(prompts, [{"id": "a", "prompt": "PROMPT"}])
    harness = tmp_path / "harness.jsonl"
    harness.write_text("", encoding="utf-8")

    rc = run_verifier_trace_pipeline.main(
        [
            "--backend",
            "harness_jsonl",
            "--prompts",
            str(prompts),
            "--harness-results",
            str(harness),
            *(_common_args(tmp_path=tmp_path)),
        ]
    )
    assert rc == 2


def test_main_success_cases_jsonl_backend_copies(tmp_path: Path) -> None:
    prompts = tmp_path / "prompts.jsonl"
    _write_jsonl(prompts, [{"id": "a", "prompt": "x"}])
    cases = tmp_path / "cases_in.jsonl"
    _write_jsonl(cases, [{"id": "a", "verdict": "pass"}])

    rc = run_verifier_trace_pipeline.main(
        [
            "--backend",
            "cases_jsonl",
            "--prompts",
            str(prompts),
            "--cases",
            str(cases),
            *(_common_args(tmp_path=tmp_path)),
        ]
    )
    assert rc == 0
    assert (tmp_path / "cases.jsonl").read_text(encoding="utf-8") == cases.read_text(
        encoding="utf-8"
    )


def test_main_success_cases_jsonl_backend_no_copy_when_same_path(
    tmp_path: Path,
) -> None:
    prompts = tmp_path / "prompts.jsonl"
    _write_jsonl(prompts, [{"id": "a", "prompt": "x"}])
    cases_out = tmp_path / "cases.jsonl"
    _write_jsonl(cases_out, [{"id": "a", "verdict": "pass"}])

    rc = run_verifier_trace_pipeline.main(
        [
            "--backend",
            "cases_jsonl",
            "--prompts",
            str(prompts),
            "--cases",
            str(cases_out),
            *(_common_args(tmp_path=tmp_path)),
        ]
    )
    assert rc == 0
