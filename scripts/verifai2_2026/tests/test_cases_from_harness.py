from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.verifai2_2026 import cases_from_harness


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=True) + "\n" for r in rows), encoding="utf-8"
    )


def test_read_jsonl_skips_blank_and_errors(tmp_path: Path) -> None:
    p = tmp_path / "x.jsonl"
    p.write_text("\n" + json.dumps({"id": "a"}) + "\n", encoding="utf-8")
    assert len(cases_from_harness._read_jsonl(p)) == 1

    p2 = tmp_path / "bad.jsonl"
    p2.write_text("{\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"Invalid JSONL"):
        cases_from_harness._read_jsonl(p2)

    p3 = tmp_path / "bad2.jsonl"
    p3.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"Expected JSON object"):
        cases_from_harness._read_jsonl(p3)


def test_extract_error_type() -> None:
    assert cases_from_harness._extract_error_type("ValueError: boom\n") == "ValueError"
    assert cases_from_harness._extract_error_type("\n\n") is None
    # Non-empty non-matching lines should be ignored.
    assert cases_from_harness._extract_error_type("###\nno_colon\n") is None


def test_coerce_bool() -> None:
    assert cases_from_harness._coerce_bool(True) is True
    assert cases_from_harness._coerce_bool(False) is False
    assert cases_from_harness._coerce_bool(0) is False
    assert cases_from_harness._coerce_bool(2) is True
    assert cases_from_harness._coerce_bool("YES") is True
    assert cases_from_harness._coerce_bool("no") is False
    assert cases_from_harness._coerce_bool("maybe") is None


def test_infer_verdict_strict_and_non_strict() -> None:
    rec = {"verdict": "passed"}
    assert (
        cases_from_harness._infer_verdict(
            rec,
            verdict_field="verdict",
            passed_field="passed",
            status_field="status",
            strict=True,
        )
        == "pass"
    )
    rec2 = {"verdict": "wrong"}
    assert (
        cases_from_harness._infer_verdict(
            rec2,
            verdict_field="verdict",
            passed_field="passed",
            status_field="status",
            strict=True,
        )
        == "fail"
    )

    with pytest.raises(ValueError, match=r"Unknown verdict"):
        cases_from_harness._infer_verdict(
            {"verdict": "nope"},
            verdict_field="verdict",
            passed_field="passed",
            status_field="status",
            strict=True,
        )

    assert (
        cases_from_harness._infer_verdict(
            {"verdict": "nope"},
            verdict_field="verdict",
            passed_field="passed",
            status_field="status",
            strict=False,
        )
        == "error"
    )

    assert (
        cases_from_harness._infer_verdict(
            {"passed": "true"},
            verdict_field="verdict",
            passed_field="passed",
            status_field="status",
            strict=True,
        )
        == "pass"
    )

    assert (
        cases_from_harness._infer_verdict(
            {"status": "timeout"},
            verdict_field="verdict",
            passed_field="passed",
            status_field="status",
            strict=True,
        )
        == "timeout"
    )
    assert (
        cases_from_harness._infer_verdict(
            {"status": "failed"},
            verdict_field="verdict",
            passed_field="passed",
            status_field="status",
            strict=True,
        )
        == "fail"
    )
    # Status contains the timeout substring but isn't a canonical enum value.
    assert (
        cases_from_harness._infer_verdict(
            {"status": "TimeoutExpired"},
            verdict_field="verdict",
            passed_field="passed",
            status_field="status",
            strict=True,
        )
        == "timeout"
    )
    assert (
        cases_from_harness._infer_verdict(
            {"status": "Exception: x"},
            verdict_field="verdict",
            passed_field="passed",
            status_field="status",
            strict=True,
        )
        == "error"
    )
    assert (
        cases_from_harness._infer_verdict(
            {"verdict": "timeout"},
            verdict_field="verdict",
            passed_field="passed",
            status_field="status",
            strict=True,
        )
        == "timeout"
    )

    with pytest.raises(ValueError, match=r"Unable to infer verdict"):
        cases_from_harness._infer_verdict(
            {},
            verdict_field="verdict",
            passed_field="passed",
            status_field="status",
            strict=True,
        )
    assert (
        cases_from_harness._infer_verdict(
            {"status": "unknown"},
            verdict_field="verdict",
            passed_field="passed",
            status_field="status",
            strict=False,
        )
        == "error"
    )


def test_normalize_record_happy_path_and_branches(tmp_path: Path) -> None:
    rec = {
        "task_id": "t",
        "completion": "print(1)",
        "stderr": "ValueError: boom",
        "passed": False,
        "attempt_id": "7",
        "wall_time_s": "1.5",
        "failing_test_ids": ["x"],
        "message": "MSG",
    }
    out = cases_from_harness.normalize_record(
        rec,
        id_fields=["id", "task_id"],
        attempt_fields=["attempt_id"],
        completion_fields=["completion"],
        stderr_fields=["stderr"],
        wall_time_fields=["wall_time_s"],
        error_type_fields=["error_type"],
        failing_tests_fields=["failing_test_ids"],
        message_fields=["message_excerpt", "message"],
        verdict_field="verdict",
        passed_field="passed",
        status_field="status",
        max_message_chars=2,
        include_output_text=True,
        strict=True,
    )
    assert out["id"] == "t"
    assert out["attempt_id"] == 7
    assert out["verdict"] == "fail"
    assert out["wall_time_s"] == 1.5
    assert out["error_type"] == "ValueError"
    assert out["message_excerpt"] == "MS"
    assert out["failing_test_ids"] == ["x"]
    assert out["output"] == "print(1)"

    # wall_time parse failure and attempt_id parse failure branches.
    rec2 = {"id": "x", "status": "ok", "attempt_id": "nope", "wall_time_s": "bad"}
    out2 = cases_from_harness.normalize_record(
        rec2,
        id_fields=["id"],
        attempt_fields=["attempt_id"],
        completion_fields=["completion"],
        stderr_fields=["stderr"],
        wall_time_fields=["wall_time_s"],
        error_type_fields=["error_type"],
        failing_tests_fields=["failing_test_ids"],
        message_fields=["message_excerpt"],
        verdict_field="verdict",
        passed_field="passed",
        status_field="status",
        max_message_chars=400,
        include_output_text=False,
        strict=True,
    )
    assert out2["verdict"] == "pass"
    assert "attempt_id" not in out2
    assert "wall_time_s" not in out2

    # stderr present but no message -> message is taken from stderr.
    rec3 = {"id": "z", "status": "error", "stderr": "BoomError: x"}
    out3 = cases_from_harness.normalize_record(
        rec3,
        id_fields=["id"],
        attempt_fields=["attempt_id"],
        completion_fields=["completion"],
        stderr_fields=["stderr"],
        wall_time_fields=["wall_time_s"],
        error_type_fields=["error_type"],
        failing_tests_fields=["failing_test_ids"],
        message_fields=["message_excerpt"],
        verdict_field="verdict",
        passed_field="passed",
        status_field="status",
        max_message_chars=400,
        include_output_text=False,
        strict=True,
    )
    assert out3["message_excerpt"] == "BoomError: x"

    # missing id branch
    with pytest.raises(ValueError, match=r"missing id"):
        cases_from_harness.normalize_record(
            {},
            id_fields=["id"],
            attempt_fields=["attempt_id"],
            completion_fields=["completion"],
            stderr_fields=["stderr"],
            wall_time_fields=["wall_time_s"],
            error_type_fields=["error_type"],
            failing_tests_fields=["failing_test_ids"],
            message_fields=["message_excerpt"],
            verdict_field="verdict",
            passed_field="passed",
            status_field="status",
            max_message_chars=400,
            include_output_text=False,
            strict=True,
        )


def test_main_writes_and_non_strict(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    in_path = tmp_path / "in.jsonl"
    _write_jsonl(in_path, [{"id": "a", "verdict": "nope"}])
    out_path = tmp_path / "cases.jsonl"

    rc = cases_from_harness.main(
        ["--in", str(in_path), "--out", str(out_path), "--non-strict"]
    )
    assert rc == 0
    assert capsys.readouterr().out.strip() == str(out_path)
    rows = [
        json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines()
    ]
    assert rows[0]["verdict"] == "error"


def test_main_no_records_returns_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    in_path = tmp_path / "empty.jsonl"
    in_path.write_text("", encoding="utf-8")
    out_path = tmp_path / "cases.jsonl"

    rc = cases_from_harness.main(["--in", str(in_path), "--out", str(out_path)])
    assert rc == 2
    assert "No records found." in capsys.readouterr().err
