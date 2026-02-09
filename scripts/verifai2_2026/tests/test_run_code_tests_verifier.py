from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.verifai2_2026 import run_code_tests_verifier


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=True) + "\n" for r in rows), encoding="utf-8"
    )


def test_extract_error_type() -> None:
    stderr = "Traceback (most recent call last):\n\n  ...\nValueError: boom\n"
    assert run_code_tests_verifier._extract_error_type(stderr) == "ValueError"
    # Exercises: skip blank lines, ignore non-matching lines, return None.
    assert run_code_tests_verifier._extract_error_type("\n\n---\n") is None


def test_read_jsonl_skips_blank_and_rejects_invalid(tmp_path: Path) -> None:
    p = tmp_path / "x.jsonl"
    p.write_text("\n" + json.dumps({"a": 1}) + "\n", encoding="utf-8")
    assert len(run_code_tests_verifier._read_jsonl(p)) == 1

    p2 = tmp_path / "bad.jsonl"
    p2.write_text("{\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"Invalid JSONL"):
        run_code_tests_verifier._read_jsonl(p2)

    p3 = tmp_path / "bad2.jsonl"
    p3.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"Expected JSON object"):
        run_code_tests_verifier._read_jsonl(p3)


def test_main_no_completions_returns_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    tasks = tmp_path / "tasks.jsonl"
    _write_jsonl(tasks, [{"id": "a", "prompt": "", "tests": "assert True"}])
    completions = tmp_path / "completions.jsonl"
    completions.write_text("", encoding="utf-8")
    out_cases = tmp_path / "cases.jsonl"

    rc = run_code_tests_verifier.main(
        [
            "--tasks",
            str(tasks),
            "--completions",
            str(completions),
            "--out-cases",
            str(out_cases),
        ]
    )
    assert rc == 2
    assert "No completions found." in capsys.readouterr().err


def test_main_missing_task_writes_error_case(tmp_path: Path) -> None:
    tasks = tmp_path / "tasks.jsonl"
    _write_jsonl(tasks, [{"id": "known", "prompt": "", "tests": "assert True"}])
    completions = tmp_path / "completions.jsonl"
    _write_jsonl(completions, [{"id": "missing", "completion": "x=1"}])
    out_cases = tmp_path / "cases.jsonl"

    rc = run_code_tests_verifier.main(
        [
            "--tasks",
            str(tasks),
            "--completions",
            str(completions),
            "--out-cases",
            str(out_cases),
        ]
    )
    assert rc == 0
    rows = [
        json.loads(line) for line in out_cases.read_text(encoding="utf-8").splitlines()
    ]
    assert rows[0]["verdict"] == "error"
    assert rows[0]["error_type"] == "missing_task"


def test_main_rejects_task_missing_id_and_completion_missing_id(tmp_path: Path) -> None:
    tasks = tmp_path / "tasks.jsonl"
    _write_jsonl(tasks, [{"prompt": "", "tests": "assert True"}])
    completions = tmp_path / "completions.jsonl"
    _write_jsonl(completions, [{"id": "x", "completion": ""}])
    out_cases = tmp_path / "cases.jsonl"

    with pytest.raises(ValueError, match=r"Task missing id field"):
        run_code_tests_verifier.main(
            [
                "--tasks",
                str(tasks),
                "--completions",
                str(completions),
                "--out-cases",
                str(out_cases),
            ]
        )

    tasks2 = tmp_path / "tasks2.jsonl"
    _write_jsonl(tasks2, [{"id": "x", "prompt": "", "tests": "assert True"}])
    completions2 = tmp_path / "completions2.jsonl"
    _write_jsonl(completions2, [{"completion": ""}])
    with pytest.raises(ValueError, match=r"Completion missing id field"):
        run_code_tests_verifier.main(
            [
                "--tasks",
                str(tasks2),
                "--completions",
                str(completions2),
                "--out-cases",
                str(out_cases),
            ]
        )


def test_main_pass_and_fail_and_bad_attempt_id(tmp_path: Path) -> None:
    tasks = tmp_path / "tasks.jsonl"
    _write_jsonl(
        tasks,
        [
            {"id": "p", "prompt": "", "tests": "x = 1\nassert x == 1\n"},
            {"id": "f", "prompt": "", "tests": "raise ValueError('boom')\n"},
        ],
    )
    completions = tmp_path / "completions.jsonl"
    _write_jsonl(
        completions,
        [
            {"id": "p", "attempt_id": "nope", "completion": ""},
            {"id": "f", "attempt_id": 1, "completion": ""},
        ],
    )
    out_cases = tmp_path / "cases.jsonl"

    rc = run_code_tests_verifier.main(
        [
            "--tasks",
            str(tasks),
            "--completions",
            str(completions),
            "--out-cases",
            str(out_cases),
            "--python",
            sys.executable,
            "--timeout-s",
            "2",
        ]
    )
    assert rc == 0
    rows = [
        json.loads(line) for line in out_cases.read_text(encoding="utf-8").splitlines()
    ]
    assert [r["id"] for r in rows] == ["p", "f"]
    assert rows[0]["verdict"] == "pass"
    assert rows[0]["attempt_id"] == 0  # bad attempt_id coerced
    assert rows[1]["verdict"] == "fail"
    assert rows[1]["error_type"] == "ValueError"
    assert "message_excerpt" in rows[1]


def test_main_rejects_missing_fields(tmp_path: Path) -> None:
    tasks = tmp_path / "tasks.jsonl"
    _write_jsonl(tasks, [{"id": "a", "prompt": "", "tests": "assert True"}])
    completions = tmp_path / "completions.jsonl"
    _write_jsonl(completions, [{"id": "a", "completion": None}])
    out_cases = tmp_path / "cases.jsonl"

    with pytest.raises(ValueError, match=r"Completion missing text field"):
        run_code_tests_verifier.main(
            [
                "--tasks",
                str(tasks),
                "--completions",
                str(completions),
                "--out-cases",
                str(out_cases),
            ]
        )


def test_main_rejects_task_missing_prompt_or_tests(tmp_path: Path) -> None:
    tasks = tmp_path / "tasks.jsonl"
    _write_jsonl(tasks, [{"id": "a", "prompt": None, "tests": "assert True"}])
    completions = tmp_path / "completions.jsonl"
    _write_jsonl(completions, [{"id": "a", "completion": ""}])
    out_cases = tmp_path / "cases.jsonl"

    with pytest.raises(ValueError, match=r"Task fields missing"):
        run_code_tests_verifier.main(
            [
                "--tasks",
                str(tasks),
                "--completions",
                str(completions),
                "--out-cases",
                str(out_cases),
            ]
        )


def test_main_jobs_must_be_ge_1(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    tasks = tmp_path / "tasks.jsonl"
    _write_jsonl(tasks, [{"id": "a", "prompt": "", "tests": "assert True"}])
    completions = tmp_path / "completions.jsonl"
    _write_jsonl(completions, [{"id": "a", "completion": ""}])
    out_cases = tmp_path / "cases.jsonl"

    rc = run_code_tests_verifier.main(
        [
            "--tasks",
            str(tasks),
            "--completions",
            str(completions),
            "--out-cases",
            str(out_cases),
            "--jobs",
            "0",
        ]
    )
    assert rc == 2
    assert "--jobs must be >= 1" in capsys.readouterr().err


def test_main_jobs_gt_1_uses_thread_pool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _Proc:
        def __init__(self, *, ok: bool) -> None:
            self.returncode = 0 if ok else 1
            self.stderr = "" if ok else "ValueError: boom\n"

    calls: list[str] = []

    def _fake_run(argv, **kwargs):  # noqa: ANN001
        # Called from multiple threads; list append is atomic enough here.
        calls.append("x")
        # Alternate pass/fail by call count to exercise both branches.
        return _Proc(ok=(len(calls) % 2 == 1))

    monkeypatch.setattr(run_code_tests_verifier.subprocess, "run", _fake_run)

    tasks = tmp_path / "tasks.jsonl"
    _write_jsonl(
        tasks,
        [
            {"id": "a", "prompt": "", "tests": "assert True"},
            {"id": "b", "prompt": "", "tests": "assert True"},
        ],
    )
    completions = tmp_path / "completions.jsonl"
    _write_jsonl(
        completions,
        [
            {"id": "a", "completion": ""},
            {"id": "b", "completion": ""},
        ],
    )
    out_cases = tmp_path / "cases.jsonl"

    rc = run_code_tests_verifier.main(
        [
            "--tasks",
            str(tasks),
            "--completions",
            str(completions),
            "--out-cases",
            str(out_cases),
            "--jobs",
            "2",
        ]
    )
    assert rc == 0
    assert len(calls) == 2
    rows = [
        json.loads(line) for line in out_cases.read_text(encoding="utf-8").splitlines()
    ]
    assert {r["verdict"] for r in rows} == {"pass", "fail"}


def test_timeout_includes_message_excerpt_when_available(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _fake_run(*args, **kwargs):  # noqa: ANN001
        raise subprocess.TimeoutExpired(cmd="x", timeout=0.1, stderr="ERR\n")

    monkeypatch.setattr(run_code_tests_verifier.subprocess, "run", _fake_run)

    tasks = tmp_path / "tasks.jsonl"
    _write_jsonl(
        tasks, [{"id": "t", "prompt": "", "tests": "import time; time.sleep(1)"}]
    )
    completions = tmp_path / "completions.jsonl"
    _write_jsonl(completions, [{"id": "t", "completion": ""}])
    out_cases = tmp_path / "cases.jsonl"

    rc = run_code_tests_verifier.main(
        [
            "--tasks",
            str(tasks),
            "--completions",
            str(completions),
            "--out-cases",
            str(out_cases),
        ]
    )
    assert rc == 0
    row = json.loads(out_cases.read_text(encoding="utf-8").splitlines()[0])
    assert row["verdict"] == "timeout"
    assert row["message_excerpt"].startswith("ERR")


def test_timeout_with_non_str_stderr_hits_type_branch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _FakeTimeout(subprocess.TimeoutExpired):
        pass

    def _fake_run(*args, **kwargs):  # noqa: ANN001
        raise _FakeTimeout(cmd="x", timeout=0.1, stderr=b"bytes")

    monkeypatch.setattr(run_code_tests_verifier.subprocess, "run", _fake_run)

    tasks = tmp_path / "tasks.jsonl"
    _write_jsonl(
        tasks, [{"id": "t", "prompt": "", "tests": "import time; time.sleep(1)"}]
    )
    completions = tmp_path / "completions.jsonl"
    _write_jsonl(completions, [{"id": "t", "completion": ""}])
    out_cases = tmp_path / "cases.jsonl"

    rc = run_code_tests_verifier.main(
        [
            "--tasks",
            str(tasks),
            "--completions",
            str(completions),
            "--out-cases",
            str(out_cases),
        ]
    )
    assert rc == 0
    row = json.loads(out_cases.read_text(encoding="utf-8").splitlines()[0])
    assert row["verdict"] == "timeout"
    assert "message_excerpt" not in row  # msg is empty when stderr is non-str


def test_non_str_proc_stderr_hits_type_branch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _Proc:
        def __init__(self) -> None:
            self.returncode = 1
            self.stderr = b"bytes"

    def _fake_run(*args, **kwargs):  # noqa: ANN001
        return _Proc()

    monkeypatch.setattr(run_code_tests_verifier.subprocess, "run", _fake_run)

    tasks = tmp_path / "tasks.jsonl"
    _write_jsonl(tasks, [{"id": "a", "prompt": "", "tests": "assert True"}])
    completions = tmp_path / "completions.jsonl"
    _write_jsonl(completions, [{"id": "a", "completion": ""}])
    out_cases = tmp_path / "cases.jsonl"

    rc = run_code_tests_verifier.main(
        [
            "--tasks",
            str(tasks),
            "--completions",
            str(completions),
            "--out-cases",
            str(out_cases),
        ]
    )
    assert rc == 0
    row = json.loads(out_cases.read_text(encoding="utf-8").splitlines()[0])
    assert row["verdict"] == "fail"
    assert "stderr_sha256" not in row
