from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.verifai2_2026 import verifier_trace_from_cases


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_json(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=True) + "\n" for r in rows), encoding="utf-8"
    )


def _prompt_set(*, ids: list[str], mode: str = "hash_only") -> dict:
    items = [{"id": i, "sha256": _sha256_hex(i.encode("utf-8"))} for i in ids]
    dataset = {"name": "local", "split": "test", "revision": "rev"}
    ps = {"mode": mode, "dataset": dataset, "items": items}
    ps["digest_sha256"] = verifier_trace_from_cases._compute_prompt_set_digest(ps)
    return ps


def test_compute_prompt_set_digest_ignores_embedded_text_and_non_dicts() -> None:
    ps = {
        "dataset": ["not-a-dict"],
        "items": [{"id": "a", "sha256": "x"}, "skip-me"],
    }
    d = verifier_trace_from_cases._compute_prompt_set_digest(ps)
    assert isinstance(d, str) and len(d) == 64


def test_read_jsonl_skips_blank_and_rejects_invalid(tmp_path: Path) -> None:
    p = tmp_path / "x.jsonl"
    p.write_text(
        "\n" + json.dumps({"id": "a", "verdict": "pass"}) + "\n", encoding="utf-8"
    )
    assert len(verifier_trace_from_cases._read_jsonl(p)) == 1

    p2 = tmp_path / "bad.jsonl"
    p2.write_text("{\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"Invalid JSONL"):
        verifier_trace_from_cases._read_jsonl(p2)

    p3 = tmp_path / "bad2.jsonl"
    p3.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"Expected JSON object"):
        verifier_trace_from_cases._read_jsonl(p3)


def test_compute_prompt_set_digest_items_not_list_branch() -> None:
    d = verifier_trace_from_cases._compute_prompt_set_digest(
        {"dataset": {}, "items": "nope"}
    )
    assert isinstance(d, str) and len(d) == 64


def test_normalize_case_record_errors() -> None:
    with pytest.raises(ValueError, match=r"missing id"):
        verifier_trace_from_cases._normalize_case_record({"verdict": "pass"})

    with pytest.raises(ValueError, match=r"invalid verdict"):
        verifier_trace_from_cases._normalize_case_record({"id": "x", "verdict": "nope"})


def test_normalize_case_record_hashes_and_counterexample() -> None:
    rec = verifier_trace_from_cases._normalize_case_record(
        {
            "id": "x",
            "verdict": "fail",
            "wall_time_s": 1.5,
            "output": "OUT",
            "stderr": "ERR",
            "failing_test_ids": ["t1", "t2"],
            "message_excerpt": "boom",
        }
    )
    assert rec["output_sha256"] == _sha256_hex(b"OUT")
    assert rec["stderr_sha256"] == _sha256_hex(b"ERR")
    assert rec["counterexample"]["message_sha256"] == _sha256_hex(b"boom")

    # Explicit sha fields take precedence; wall_time parse failure is ignored.
    rec2 = verifier_trace_from_cases._normalize_case_record(
        {
            "id": "y",
            "verdict": "error",
            "wall_time_s": "bad",
            "output_sha256": "0" * 64,
            "stderr_sha256": "1" * 64,
            "error_type": "ValueError",
        }
    )
    assert rec2["output_sha256"] == "0" * 64
    assert rec2["stderr_sha256"] == "1" * 64
    assert "wall_time_s" not in rec2

    rec3 = verifier_trace_from_cases._normalize_case_record(
        {
            "id": "z",
            "verdict": "fail",
            "failing_test_ids": ["ok", 1],
            "message_excerpt": "",
        }
    )
    assert rec3["counterexample"] == {}


def test_aggregate_verdict_precedence() -> None:
    assert verifier_trace_from_cases._aggregate_verdict(["skipped"]) == "skipped"
    assert verifier_trace_from_cases._aggregate_verdict(["error"]) == "error"
    assert verifier_trace_from_cases._aggregate_verdict(["timeout"]) == "timeout"
    assert verifier_trace_from_cases._aggregate_verdict(["fail", "error"]) == "fail"
    assert verifier_trace_from_cases._aggregate_verdict(["pass", "fail"]) == "pass"


def test_main_rejects_prompt_set_not_object(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ps = tmp_path / "ps.json"
    ps.write_text("[]\n", encoding="utf-8")
    cases = tmp_path / "cases.jsonl"
    _write_jsonl(cases, [{"id": "a", "verdict": "pass"}])
    out = tmp_path / "trace.json"

    rc = verifier_trace_from_cases.main(
        [
            "--prompt-set",
            str(ps),
            "--cases",
            str(cases),
            "--out",
            str(out),
            "--verifier-name",
            "humeval",
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
            "--max-new-tokens",
            "8",
            "--seed",
            "0",
        ]
    )
    assert rc == 2
    assert "prompt_set must be a JSON object" in capsys.readouterr().err


def test_main_rejects_digest_mismatch(tmp_path: Path) -> None:
    ps_obj = _prompt_set(ids=["a"])
    ps_obj["digest_sha256"] = "0" * 64
    ps = tmp_path / "ps.json"
    _write_json(ps, ps_obj)

    cases = tmp_path / "cases.jsonl"
    _write_jsonl(cases, [{"id": "a", "verdict": "pass"}])
    out = tmp_path / "trace.json"

    rc = verifier_trace_from_cases.main(
        [
            "--prompt-set",
            str(ps),
            "--cases",
            str(cases),
            "--out",
            str(out),
            "--verifier-name",
            "humeval",
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
            "--max-new-tokens",
            "8",
            "--seed",
            "0",
        ]
    )
    assert rc == 2


def test_main_rejects_items_missing_or_invalid(tmp_path: Path) -> None:
    ps_obj = _prompt_set(ids=["a"])
    ps_obj["items"] = []
    ps_obj["digest_sha256"] = verifier_trace_from_cases._compute_prompt_set_digest(
        ps_obj
    )
    ps = tmp_path / "ps.json"
    _write_json(ps, ps_obj)

    cases = tmp_path / "cases.jsonl"
    _write_jsonl(cases, [{"id": "a", "verdict": "pass"}])
    out = tmp_path / "trace.json"

    rc = verifier_trace_from_cases.main(
        [
            "--prompt-set",
            str(ps),
            "--cases",
            str(cases),
            "--out",
            str(out),
            "--verifier-name",
            "humeval",
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
            "--max-new-tokens",
            "8",
            "--seed",
            "0",
        ]
    )
    assert rc == 2

    ps_obj = _prompt_set(ids=["a"])
    ps_obj["items"] = [{"id": 123, "sha256": "0" * 64}]
    ps_obj["digest_sha256"] = verifier_trace_from_cases._compute_prompt_set_digest(
        ps_obj
    )
    _write_json(ps, ps_obj)
    rc2 = verifier_trace_from_cases.main(
        [
            "--prompt-set",
            str(ps),
            "--cases",
            str(cases),
            "--out",
            str(out),
            "--verifier-name",
            "humeval",
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
            "--max-new-tokens",
            "8",
            "--seed",
            "0",
        ]
    )
    assert rc2 == 2


def test_main_harness_identity_incomplete(tmp_path: Path) -> None:
    ps = tmp_path / "ps.json"
    _write_json(ps, _prompt_set(ids=["a"]))
    cases = tmp_path / "cases.jsonl"
    _write_jsonl(cases, [{"id": "a", "verdict": "pass"}])
    out = tmp_path / "trace.json"

    rc = verifier_trace_from_cases.main(
        [
            "--prompt-set",
            str(ps),
            "--cases",
            str(cases),
            "--out",
            str(out),
            "--verifier-name",
            "humeval",
            "--verifier-kind",
            "code_execution",
            "--harness-name",
            "h",
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
            "--max-new-tokens",
            "8",
            "--seed",
            "0",
        ]
    )
    assert rc == 2


def test_main_success_pass1_and_missing_results(tmp_path: Path) -> None:
    ps = tmp_path / "ps.json"
    _write_json(ps, _prompt_set(ids=["a", "b"]))
    cases = tmp_path / "cases.jsonl"
    _write_jsonl(
        cases,
        [
            {
                "id": "a",
                "verdict": "pass",
                "output": "ok",
                "stderr": "",
            }
        ],
    )
    out = tmp_path / "trace.json"

    rc = verifier_trace_from_cases.main(
        [
            "--prompt-set",
            str(ps),
            "--cases",
            str(cases),
            "--out",
            str(out),
            "--verifier-name",
            "humeval",
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
            "--max-new-tokens",
            "8",
            "--seed",
            "0",
        ]
    )
    assert rc == 0

    trace = json.loads(out.read_text(encoding="utf-8"))
    assert trace["verifier"]["sandbox"]["network_enabled"] is False
    assert [c["id"] for c in trace["results"]["cases"]] == ["a", "b"]
    assert trace["results"]["cases"][1]["error_type"] == "missing_result"
    assert trace["results"]["summary"]["n_total"] == 2


def test_main_success_multi_attempts_non_code_kind(tmp_path: Path) -> None:
    ps = tmp_path / "ps.json"
    _write_json(ps, _prompt_set(ids=["x"]))
    cases = tmp_path / "cases.jsonl"
    _write_jsonl(
        cases,
        [
            {"id": "x", "attempt_id": 1, "verdict": "fail", "stderr": "ValueError: x"},
            {"id": "x", "attempt_id": "nope", "verdict": "pass", "stderr": ""},
        ],
    )
    out = tmp_path / "trace.json"
    cfg = tmp_path / "cfg.json"
    cfg.write_text('{"x": 1}\n', encoding="utf-8")

    rc = verifier_trace_from_cases.main(
        [
            "--prompt-set",
            str(ps),
            "--cases",
            str(cases),
            "--out",
            str(out),
            "--verifier-name",
            "solver",
            "--verifier-kind",
            "smt_solver",
            "--harness-name",
            "h",
            "--harness-git-commit",
            "deadbeef",
            "--harness-container-image",
            "img@sha256:abc",
            "--harness-config",
            str(cfg),
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
            "--max-new-tokens",
            "8",
            "--seed",
            "0",
            "--k",
            "2",
            "--num-samples",
            "3",
            "--metric-name",
            "pass@2",
        ]
    )
    assert rc == 0

    trace = json.loads(out.read_text(encoding="utf-8"))
    assert "sandbox" not in trace["verifier"]
    assert trace["verifier"]["harness"]["container_image"] == "img@sha256:abc"
    assert trace["verifier"]["harness"]["config_digest_sha256"] == _sha256_hex(
        cfg.read_bytes()
    )
    case = trace["results"]["cases"][0]
    assert case["verdict"] == "pass"
    assert len(case["attempts"]) == 2
    assert trace["results"]["summary"]["k"] == 2
    assert trace["results"]["summary"]["n_samples_per_case"] == 3
    assert trace["trace_contract"]["decoding"]["num_samples"] == 3
