from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.verifai2_2026 import run_matrix


def _write_json(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=True) + "\n" for r in rows), encoding="utf-8"
    )


def test_parse_flag_value() -> None:
    assert run_matrix._parse_flag_value(["--x", "1"], "--x") == "1"
    assert run_matrix._parse_flag_value(["--x"], "--x") is None
    assert run_matrix._parse_flag_value([], "--x") is None


def test_main_plan_validation_errors(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    p = tmp_path / "plan.json"
    p.write_text("[]\n", encoding="utf-8")
    assert run_matrix.main(["--plan", str(p)]) == 2
    assert "plan must be a JSON object" in capsys.readouterr().err

    _write_json(
        p,
        {
            "schema_version": "x",
            "jobs": [{"job_id": "j", "trace_argv": ["--trace-out", "x"]}],
        },
    )
    assert run_matrix.main(["--plan", str(p)]) == 2
    assert "schema_version" in capsys.readouterr().err

    _write_json(p, {"schema_version": "verifai2_matrix_plan.v1", "jobs": []})
    assert run_matrix.main(["--plan", str(p)]) == 2
    assert "plan.jobs must be a non-empty array" in capsys.readouterr().err


def test_main_job_validation_errors(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    p = tmp_path / "plan.json"
    _write_json(p, {"schema_version": "verifai2_matrix_plan.v1", "jobs": ["x"]})
    assert run_matrix.main(["--plan", str(p)]) == 2
    assert "job must be a JSON object" in capsys.readouterr().err

    _write_json(p, {"schema_version": "verifai2_matrix_plan.v1", "jobs": [{}]})
    assert run_matrix.main(["--plan", str(p)]) == 2
    assert "job.job_id is required" in capsys.readouterr().err

    _write_json(
        p,
        {
            "schema_version": "verifai2_matrix_plan.v1",
            "jobs": [{"job_id": "j", "trace_argv": "nope"}],
        },
    )
    assert run_matrix.main(["--plan", str(p)]) == 2
    assert "trace_argv" in capsys.readouterr().err

    _write_json(
        p,
        {
            "schema_version": "verifai2_matrix_plan.v1",
            "jobs": [{"job_id": "j", "trace_argv": ["--x", 1]}],
        },
    )
    assert run_matrix.main(["--plan", str(p)]) == 2
    assert "trace_argv" in capsys.readouterr().err

    _write_json(
        p,
        {
            "schema_version": "verifai2_matrix_plan.v1",
            "jobs": [{"job_id": "j", "trace_argv": ["--no-trace-out", "x"]}],
        },
    )
    assert run_matrix.main(["--plan", str(p)]) == 2
    assert "--trace-out" in capsys.readouterr().err


def test_main_dry_run_success(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    p = tmp_path / "plan.json"
    _write_json(
        p,
        {
            "schema_version": "verifai2_matrix_plan.v1",
            "jobs": [{"job_id": "j", "trace_argv": ["--trace-out", "x.json"]}],
        },
    )
    assert run_matrix.main(["--plan", str(p)]) == 0
    assert "[job j]" in capsys.readouterr().out


def test_main_dry_run_with_artifact_validate_branch(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    p = tmp_path / "plan.json"
    _write_json(
        p,
        {
            "schema_version": "verifai2_matrix_plan.v1",
            "jobs": [
                {
                    "job_id": "j",
                    "trace_argv": ["--trace-out", "x.json"],
                    "artifact": {
                        "evaluation_report": "e",
                        "out": "o",
                        "validate": True,
                        "check_files": False,
                    },
                }
            ],
        },
    )
    assert run_matrix.main(["--plan", str(p)]) == 0
    out = capsys.readouterr().out
    assert "pilot_assemble_artifact.py" in out
    assert "schema_verify.py" in out


def test_main_dry_run_artifact_validate_false_multi_job_hits_loop_backedge(
    tmp_path: Path,
) -> None:
    # Coverage: ensure the `validate=False` branch is taken and the loop continues
    # to a second job (covers the arc 152->70 in run_matrix.py).
    p = tmp_path / "plan.json"
    _write_json(
        p,
        {
            "schema_version": "verifai2_matrix_plan.v1",
            "jobs": [
                {
                    "job_id": "j1",
                    "trace_argv": ["--trace-out", "x.json"],
                    "artifact": {
                        "evaluation_report": "e",
                        "out": "o",
                        "validate": False,
                    },
                },
                {
                    "job_id": "j2",
                    "trace_argv": ["--trace-out", "y.json"],
                },
            ],
        },
    )
    assert run_matrix.main(["--plan", str(p)]) == 0


def test_main_execute_success_with_artifact_and_validate(tmp_path: Path) -> None:
    tasks = tmp_path / "tasks.jsonl"
    completions = tmp_path / "completions.jsonl"
    _write_jsonl(tasks, [{"id": "a", "prompt": "", "tests": "x=1\nassert x==1\n"}])
    _write_jsonl(completions, [{"id": "a", "completion": ""}])

    eval_report = tmp_path / "evaluation.report.json"
    _write_json(eval_report, {"hello": "world"})
    verify_json = tmp_path / "verify.json"
    _write_json(verify_json, {"profile": "ci", "ok": True, "errors": []})

    prompt_set_out = tmp_path / "prompt_set.json"
    cases_out = tmp_path / "cases.jsonl"
    trace_out = tmp_path / "trace.json"
    artifact_out = tmp_path / "artifact.json"

    trace_argv = [
        "--backend",
        "code_tests",
        "--tasks",
        str(tasks),
        "--completions",
        str(completions),
        "--prompt-set-out",
        str(prompt_set_out),
        "--cases-out",
        str(cases_out),
        "--trace-out",
        str(trace_out),
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

    plan = {
        "schema_version": "verifai2_matrix_plan.v1",
        "jobs": [
            {
                "job_id": "j",
                "trace_argv": trace_argv,
                "artifact": {
                    "evaluation_report": str(eval_report),
                    "out": str(artifact_out),
                    "verify_json": str(verify_json),
                    "embed_evaluation_report": True,
                    "invarlock_version": "0.0",
                    "git_commit": "deadbeef",
                    "validate": True,
                    "check_files": False,
                    "schema_root": "research/verifai2_2026/specs",
                },
            }
        ],
    }
    plan_path = tmp_path / "plan.json"
    _write_json(plan_path, plan)

    rc = run_matrix.main(["--plan", str(plan_path), "--execute"])
    assert rc == 0
    assert trace_out.exists()
    assert artifact_out.exists()


def test_main_execute_trace_failure_returns_rc(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _fail(argv: list[str] | None = None) -> int:  # noqa: ANN001
        return 5

    monkeypatch.setattr(run_matrix.run_verifier_trace_pipeline, "main", _fail)
    p = tmp_path / "plan.json"
    _write_json(
        p,
        {
            "schema_version": "verifai2_matrix_plan.v1",
            "jobs": [{"job_id": "j", "trace_argv": ["--trace-out", "x.json"]}],
        },
    )
    assert run_matrix.main(["--plan", str(p), "--execute"]) == 5


def test_main_execute_continue_on_error_returns_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _fail(argv: list[str] | None = None) -> int:  # noqa: ANN001
        return 3

    monkeypatch.setattr(run_matrix.run_verifier_trace_pipeline, "main", _fail)
    p = tmp_path / "plan.json"
    _write_json(
        p,
        {
            "schema_version": "verifai2_matrix_plan.v1",
            "jobs": [{"job_id": "j", "trace_argv": ["--trace-out", "x.json"]}],
        },
    )
    assert run_matrix.main(["--plan", str(p), "--execute", "--continue-on-error"]) == 2


def test_main_artifact_validation_and_failure_branches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # artifact must be dict
    p = tmp_path / "plan.json"
    _write_json(
        p,
        {
            "schema_version": "verifai2_matrix_plan.v1",
            "jobs": [
                {"job_id": "j", "trace_argv": ["--trace-out", "x.json"], "artifact": []}
            ],
        },
    )
    assert run_matrix.main(["--plan", str(p)]) == 2

    # missing required artifact fields
    _write_json(
        p,
        {
            "schema_version": "verifai2_matrix_plan.v1",
            "jobs": [
                {
                    "job_id": "j",
                    "trace_argv": ["--trace-out", "x.json"],
                    "artifact": {"out": "x"},
                }
            ],
        },
    )
    assert run_matrix.main(["--plan", str(p)]) == 2

    # invalid verifier_traces list
    _write_json(
        p,
        {
            "schema_version": "verifai2_matrix_plan.v1",
            "jobs": [
                {
                    "job_id": "j",
                    "trace_argv": ["--trace-out", "x.json"],
                    "artifact": {
                        "evaluation_report": "e",
                        "out": "o",
                        "verifier_traces": [],
                    },
                }
            ],
        },
    )
    assert run_matrix.main(["--plan", str(p)]) == 2

    # Assembly failure with continue-on-error sets overall failure.
    monkeypatch.setattr(
        run_matrix.run_verifier_trace_pipeline, "main", lambda argv=None: 0
    )
    monkeypatch.setattr(run_matrix.pilot_assemble_artifact, "main", lambda argv=None: 4)
    _write_json(
        p,
        {
            "schema_version": "verifai2_matrix_plan.v1",
            "jobs": [
                {
                    "job_id": "j",
                    "trace_argv": ["--trace-out", "x.json"],
                    "artifact": {
                        "evaluation_report": "e",
                        "out": "o",
                        "validate": False,
                    },
                }
            ],
        },
    )
    assert run_matrix.main(["--plan", str(p), "--execute", "--continue-on-error"]) == 2

    # schema_verify failure returns rc when not continuing.
    monkeypatch.setattr(run_matrix.pilot_assemble_artifact, "main", lambda argv=None: 0)
    monkeypatch.setattr(run_matrix.schema_verify, "main", lambda argv=None: 7)
    _write_json(
        p,
        {
            "schema_version": "verifai2_matrix_plan.v1",
            "jobs": [
                {
                    "job_id": "j",
                    "trace_argv": ["--trace-out", "x.json"],
                    "artifact": {
                        "evaluation_report": "e",
                        "out": "o",
                        "validate": True,
                        "check_files": True,
                        "schema_root": "root",
                    },
                }
            ],
        },
    )
    assert run_matrix.main(["--plan", str(p), "--execute"]) == 7


def test_main_artifact_assembly_failure_returns_rc_when_not_continuing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        run_matrix.run_verifier_trace_pipeline, "main", lambda argv=None: 0
    )
    monkeypatch.setattr(run_matrix.pilot_assemble_artifact, "main", lambda argv=None: 4)
    plan = {
        "schema_version": "verifai2_matrix_plan.v1",
        "jobs": [
            {
                "job_id": "j",
                "trace_argv": ["--trace-out", "x.json"],
                "artifact": {"evaluation_report": "e", "out": "o", "validate": False},
            }
        ],
    }
    plan_path = tmp_path / "plan.json"
    _write_json(plan_path, plan)
    assert run_matrix.main(["--plan", str(plan_path), "--execute"]) == 4


def test_main_schema_verify_failure_continue_on_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        run_matrix.run_verifier_trace_pipeline, "main", lambda argv=None: 0
    )
    monkeypatch.setattr(run_matrix.pilot_assemble_artifact, "main", lambda argv=None: 0)
    monkeypatch.setattr(run_matrix.schema_verify, "main", lambda argv=None: 7)
    plan = {
        "schema_version": "verifai2_matrix_plan.v1",
        "jobs": [
            {
                "job_id": "j",
                "trace_argv": ["--trace-out", "x.json"],
                "artifact": {"evaluation_report": "e", "out": "o", "validate": True},
            }
        ],
    }
    plan_path = tmp_path / "plan.json"
    _write_json(plan_path, plan)
    assert (
        run_matrix.main(["--plan", str(plan_path), "--execute", "--continue-on-error"])
        == 2
    )
