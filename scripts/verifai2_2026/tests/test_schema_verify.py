from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.verifai2_2026 import schema_verify, verifier_trace_from_cases


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_json(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=True) + "\n" for r in rows), encoding="utf-8"
    )


def _prompt_set(ids: list[str]) -> dict:
    items = [{"id": i, "sha256": _sha256_hex(i.encode("utf-8"))} for i in ids]
    dataset = {"name": "local", "split": "test", "revision": "rev"}
    ps = {"mode": "hash_only", "dataset": dataset, "items": items}
    ps["digest_sha256"] = verifier_trace_from_cases._compute_prompt_set_digest(ps)
    return ps


def _make_valid_trace(tmp_path: Path) -> dict:
    ps_path = tmp_path / "prompt_set.json"
    cases_path = tmp_path / "cases.jsonl"
    out_path = tmp_path / "trace.json"
    _write_json(ps_path, _prompt_set(["a"]))
    _write_jsonl(cases_path, [{"id": "a", "verdict": "pass", "output": "ok"}])
    rc = verifier_trace_from_cases.main(
        [
            "--prompt-set",
            str(ps_path),
            "--cases",
            str(cases_path),
            "--out",
            str(out_path),
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
    return json.loads(out_path.read_text(encoding="utf-8"))


def test_validate_artifact_ok_and_main_ok(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    eval_report = tmp_path / "evaluation.report.json"
    _write_json(eval_report, {"hello": "world"})

    trace = _make_valid_trace(tmp_path)
    capsys.readouterr()  # verifier_trace_from_cases prints the output path
    artifact = {
        "schema_version": "verifier_carrying_artifact.v1",
        "guard_evidence": {
            "invarlock": {
                "evaluation_report": {
                    "path": str(eval_report),
                    "sha256": _sha256_hex(eval_report.read_bytes()),
                }
            }
        },
        "verifier_traces": [trace],
        "provenance": {
            "created_at": "2026-02-08T00:00:00+00:00",
            "tooling": {
                "invarlock_version": "0.0",
                "schema_verify_version": "0.0",
                "git_commit": "deadbeef",
            },
        },
    }
    art_path = tmp_path / "artifact.json"
    _write_json(art_path, artifact)

    errs = schema_verify.validate_artifact(
        art_path,
        schema_root=Path("research/verifai2_2026/specs"),
        check_files=True,
    )
    assert errs == []

    rc = schema_verify.main(
        [
            str(art_path),
            "--schema-root",
            "research/verifai2_2026/specs",
            "--check-files",
        ]
    )
    assert rc == 0
    assert capsys.readouterr().out.strip() == "OK"


def test_main_prints_errors_and_returns_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    art_path = tmp_path / "bad.json"
    _write_json(art_path, {"schema_version": "verifier_carrying_artifact.v1"})

    rc = schema_verify.main(
        [str(art_path), "--schema-root", "research/verifai2_2026/specs"]
    )
    assert rc == 2
    assert "schema:" in capsys.readouterr().err


def test_validate_artifact_not_a_dict_skips_cross_checks(tmp_path: Path) -> None:
    art_path = tmp_path / "not_obj.json"
    _write_json(art_path, [])
    errs = schema_verify.validate_artifact(
        art_path, schema_root=Path("research/verifai2_2026/specs"), check_files=False
    )
    assert any(e.startswith("schema:") for e in errs)


def test_check_file_sha256_missing_mismatch_and_ok(tmp_path: Path) -> None:
    missing = schema_verify._check_file_sha256(tmp_path / "nope", "0" * 64)
    assert "does not exist" in (missing or "")

    p = tmp_path / "x.txt"
    p.write_text("x", encoding="utf-8")
    bad = schema_verify._check_file_sha256(p, "0" * 64)
    assert "sha256 mismatch" in (bad or "")

    ok = schema_verify._check_file_sha256(p, _sha256_hex(p.read_bytes()))
    assert ok is None


def test_compute_prompt_set_digest_handles_non_dicts_and_non_lists() -> None:
    d1 = schema_verify._compute_prompt_set_digest({"dataset": "x", "items": "y"})
    d2 = schema_verify._compute_prompt_set_digest({"dataset": {}, "items": []})
    assert isinstance(d1, str) and len(d1) == 64
    assert isinstance(d2, str) and len(d2) == 64


def test_validate_prompt_set_branches() -> None:
    assert (
        schema_verify._validate_prompt_set({"trace_contract": {"prompt_set": [1]}})
        == []
    )

    # Embedded mode but items is not a list: skip embedded checks, still do digest.
    trace_items_not_list = {
        "trace_contract": {
            "prompt_set": {
                "mode": "embedded",
                "dataset": {"name": "d", "split": "s", "revision": "r"},
                "items": "nope",
                "digest_sha256": "0" * 64,
            }
        }
    }
    assert any(
        "digest_sha256 mismatch" in e
        for e in schema_verify._validate_prompt_set(trace_items_not_list)
    )

    trace_missing_text = {
        "trace_contract": {
            "prompt_set": {
                "mode": "embedded",
                "dataset": {"name": "d", "split": "s", "revision": "r"},
                "items": [{"id": "a", "sha256": "0" * 64}],
                "digest_sha256": "0" * 64,
            }
        }
    }
    errs_missing = schema_verify._validate_prompt_set(trace_missing_text)
    assert any("missing text" in e for e in errs_missing)

    trace = {
        "trace_contract": {
            "prompt_set": {
                "mode": "embedded",
                "dataset": {"name": "d", "split": "s", "revision": "r"},
                "items": [
                    {"id": "a", "sha256": _sha256_hex(b"ok"), "text": "ok"},
                    {"id": "b", "sha256": "0" * 64, "text": "hi"},
                    "skip",
                ],
                "digest_sha256": "0" * 64,
            }
        }
    }
    errs = schema_verify._validate_prompt_set(trace)
    # b sha mismatch, but a sha matches (covers both branches).
    assert any("sha256 mismatch" in e for e in errs)
    assert any("digest_sha256 mismatch" in e for e in errs)

    # Digest missing (non-str) should skip digest validation entirely.
    trace_no_digest = {
        "trace_contract": {
            "prompt_set": {
                "mode": "hash_only",
                "dataset": {"name": "d", "split": "s", "revision": "r"},
                "items": [{"id": "a", "sha256": "0" * 64}],
            }
        }
    }
    assert schema_verify._validate_prompt_set(trace_no_digest) == []


def test_validate_results_consistency_branches() -> None:
    assert (
        schema_verify._validate_results_consistency(
            {"results": {"summary": {}, "cases": {}}}
        )
        == []
    )
    assert (
        schema_verify._validate_results_consistency(
            {"results": {"summary": {"n_total": "x"}, "cases": []}}
        )
        == []
    )

    errs = schema_verify._validate_results_consistency(
        {
            "results": {
                "summary": {"n_total": 3, "n_pass": 2, "pass_rate": 1.0},
                "cases": [{"verdict": "pass"}, {"verdict": "fail"}],
            }
        }
    )
    assert any("n_total" in e for e in errs)
    assert any("n_pass" in e for e in errs)


def test_validate_case_ids_match_prompt_set_branches() -> None:
    assert (
        schema_verify._validate_case_ids_match_prompt_set(
            {"trace_contract": {"prompt_set": []}}
        )
        == []
    )
    assert (
        schema_verify._validate_case_ids_match_prompt_set(
            {"trace_contract": {"prompt_set": {"items": "nope"}}}
        )
        == []
    )
    assert (
        schema_verify._validate_case_ids_match_prompt_set(
            {"trace_contract": {"prompt_set": {"items": []}}}
        )
        == []
    )

    errs = schema_verify._validate_case_ids_match_prompt_set(
        {
            "trace_contract": {
                "prompt_set": {"items": [{"id": "a"}, {"id": 1}, {"id": "b"}]}
            },
            "results": {"cases": [{"id": "a"}, {"id": "a"}, {"id": 2}]},
        }
    )
    assert any("do not exactly match" in e for e in errs)
    assert any("duplicate ids" in e for e in errs)


def test_validate_attempts_consistency_branches() -> None:
    # Type guards.
    assert (
        schema_verify._validate_attempts_consistency(
            {"results": {"summary": {}, "cases": {}}}
        )
        == []
    )

    # k > num_samples and mismatch between summary and decoding.
    trace = {
        "trace_contract": {"decoding": {"num_samples": 2}},
        "results": {
            "summary": {"k": 3, "n_samples_per_case": 1},
            "cases": [{"id": "a", "verdict": "pass"}],
        },
    }
    errs = schema_verify._validate_attempts_consistency(trace)
    assert any("results.summary.k" in e for e in errs)
    assert any("n_samples_per_case does not match" in e for e in errs)

    # Bad ints become None, attempts structural errors.
    trace2 = {
        "trace_contract": {"decoding": {"num_samples": "bad"}},
        "results": {
            "summary": {"k": "bad", "n_samples_per_case": "bad"},
            "cases": [
                {"id": "a", "verdict": "pass", "attempts": []},
                {"id": "b", "verdict": "pass", "attempts": "nope"},
            ],
        },
    }
    errs2 = schema_verify._validate_attempts_consistency(trace2)
    assert any("attempts must be a non-empty array" in e for e in errs2)

    # Duplicate attempt_ids and verdict consistency.
    trace3 = {
        "results": {
            "summary": {"n_samples_per_case": 2},
            "cases": [
                "skip",
                {
                    "id": "a",
                    "verdict": "pass",
                    "attempts": [
                        {"attempt_id": 0, "verdict": "fail"},
                        {"attempt_id": 0, "verdict": "fail"},
                        "skip",
                    ],
                },
                {
                    "id": "b",
                    "verdict": "fail",
                    "attempts": [
                        {"attempt_id": "x", "verdict": "pass"},
                        {"attempt_id": 1, "verdict": "fail"},
                    ],
                },
            ],
        }
    }
    errs3 = schema_verify._validate_attempts_consistency(trace3)
    assert any("duplicate attempt_id" in e for e in errs3)
    assert any("verdict=pass but no attempt passed" in e for e in errs3)
    assert any("at least one attempt passed" in e for e in errs3)


def test_validate_artifact_appends_cross_block_errors_and_trace_type_guard(
    tmp_path: Path,
) -> None:
    eval_report = tmp_path / "evaluation.report.json"
    _write_json(eval_report, {"x": 1})

    trace = {
        "schema_version": "verifier_trace.v1",
        "verifier": {
            "name": "humeval",
            "kind": "code_execution",
            "harness": {"name": "h", "version": "1"},
            "sandbox": {
                "network_enabled": False,
                "timeout_s": 1.0,
                "cpu_limit": 1,
                "mem_limit_mb": 128,
                "wall_limit_s": 1.0,
            },
        },
        "trace_contract": {
            "prompt_set": {
                "mode": "embedded",
                "dataset": {"name": "d", "split": "s", "revision": "r"},
                "items": [{"id": "a", "sha256": "0" * 64, "text": "hello"}],
                "digest_sha256": "0" * 64,
            },
            "model": {"id": "m", "revision": "r"},
            "tokenizer": {"id": "t", "revision": "tr"},
            "decoding": {
                "method": "greedy",
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": 0,
                "max_new_tokens": 8,
                "seed": 0,
            },
        },
        "results": {
            "summary": {
                "n_total": 2,
                "n_pass": 0,
                "pass_rate": 0.0,
                "metric_name": "pass@1",
            },
            "cases": [
                {
                    "id": "b",
                    "verdict": "pass",
                    "attempts": [{"attempt_id": 0, "verdict": "fail"}],
                }
            ],
        },
    }

    artifact = {
        "schema_version": "verifier_carrying_artifact.v1",
        "guard_evidence": {
            "invarlock": {
                "evaluation_report": {
                    "path": str(eval_report),
                    "sha256": _sha256_hex(eval_report.read_bytes()),
                }
            }
        },
        "verifier_traces": [trace, "skip"],
        "provenance": {
            "created_at": "2026-02-08T00:00:00+00:00",
            "tooling": {
                "invarlock_version": "0.0",
                "schema_verify_version": "0.0",
                "git_commit": "deadbeef",
            },
        },
    }
    art_path = tmp_path / "artifact.json"
    _write_json(art_path, artifact)

    errs = schema_verify.validate_artifact(
        art_path, schema_root=Path("research/verifai2_2026/specs"), check_files=False
    )
    # These errors must come from the validate_artifact() cross-check append loop.
    assert any(e.startswith("trace[0]:") for e in errs)


def test_validate_file_refs_only_runs_when_check_files_true(tmp_path: Path) -> None:
    artifact = {
        "guard_evidence": {
            "invarlock": {
                "evaluation_report": {
                    "path": str(tmp_path / "missing.json"),
                    "sha256": "0" * 64,
                }
            }
        }
    }
    assert schema_verify._validate_file_refs(artifact, check_files=False) == []
    errs = schema_verify._validate_file_refs(artifact, check_files=True)
    assert any("does not exist" in e for e in errs)
