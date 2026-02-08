from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.verifai2_2026 import pilot_assemble_artifact


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_json(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=True) + "\n", encoding="utf-8")


def test_main_embeds_evaluation_and_verify_legacy(tmp_path: Path) -> None:
    eval_report = tmp_path / "evaluation.report.json"
    _write_json(eval_report, {"hello": "world"})

    trace1 = tmp_path / "trace1.json"
    _write_json(trace1, {"schema_version": "verifier_trace.v1"})
    trace2 = tmp_path / "trace2.json"
    _write_json(trace2, {"schema_version": "verifier_trace.v1"})

    verify_json = tmp_path / "verify.json"
    _write_json(verify_json, {"profile": "ci", "ok": True, "errors": []})

    out = tmp_path / "artifact.json"
    rc = pilot_assemble_artifact.main(
        [
            "--evaluation-report",
            str(eval_report),
            "--verifier-trace",
            str(trace1),
            "--verifier-trace",
            str(trace2),
            "--out",
            str(out),
            "--embed-evaluation-report",
            "--verify-json",
            str(verify_json),
            "--invarlock-version",
            "0.0",
            "--git-commit",
            "abc",
        ]
    )
    assert rc == 0

    art = json.loads(out.read_text(encoding="utf-8"))
    eref = art["guard_evidence"]["invarlock"]["evaluation_report"]
    assert eref["sha256"] == _sha256_hex(eval_report.read_bytes())
    assert eref["embedded"]["hello"] == "world"
    assert len(art["verifier_traces"]) == 2
    assert art["guard_evidence"]["invarlock"]["verify"]["ok"] is True
    assert art["guard_evidence"]["invarlock"]["verify"]["profile"] == "ci"


def test_main_embeds_evaluation_and_verify_legacy_non_list_errors(
    tmp_path: Path,
) -> None:
    eval_report = tmp_path / "evaluation.report.json"
    _write_json(eval_report, {"hello": "world"})

    trace = tmp_path / "trace.json"
    _write_json(trace, {"schema_version": "verifier_trace.v1"})

    verify_json = tmp_path / "verify.json"
    _write_json(verify_json, {"profile": "nope", "ok": False, "errors": "boom"})

    out = tmp_path / "artifact.json"
    rc = pilot_assemble_artifact.main(
        [
            "--evaluation-report",
            str(eval_report),
            "--verifier-trace",
            str(trace),
            "--out",
            str(out),
            "--verify-json",
            str(verify_json),
            "--verify-profile",
            "dev",
        ]
    )
    assert rc == 0

    art = json.loads(out.read_text(encoding="utf-8"))
    v = art["guard_evidence"]["invarlock"]["verify"]
    assert v["profile"] == "dev"
    assert v["ok"] is False
    assert v["errors"] == ["boom"]


def test_main_embeds_evaluation_and_verify_legacy_defaults_profile_to_ci(
    tmp_path: Path,
) -> None:
    eval_report = tmp_path / "evaluation.report.json"
    _write_json(eval_report, {"x": 1})

    trace = tmp_path / "trace.json"
    _write_json(trace, {"schema_version": "verifier_trace.v1"})

    verify_json = tmp_path / "verify.json"
    _write_json(verify_json, {"profile": "nope", "ok": True, "errors": []})

    out = tmp_path / "artifact.json"
    rc = pilot_assemble_artifact.main(
        [
            "--evaluation-report",
            str(eval_report),
            "--verifier-trace",
            str(trace),
            "--out",
            str(out),
            "--verify-json",
            str(verify_json),
        ]
    )
    assert rc == 0

    art = json.loads(out.read_text(encoding="utf-8"))
    v = art["guard_evidence"]["invarlock"]["verify"]
    assert v["profile"] == "ci"
    assert v["ok"] is True
    assert v["errors"] == []


def test_main_embeds_evaluation_and_verify_v1(tmp_path: Path) -> None:
    eval_report = tmp_path / "evaluation.report.json"
    _write_json(eval_report, {"hello": "world"})

    trace = tmp_path / "trace.json"
    _write_json(trace, {"schema_version": "verifier_trace.v1"})

    verify_json = tmp_path / "verify.json"
    _write_json(
        verify_json,
        {
            "format_version": "verify-v1",
            "summary": {"ok": False, "reason": "failed"},
            "results": [
                {"id": "ratio", "ok": False, "reason": "mismatch"},
                "not-a-dict",
                {"id": "other", "ok": True, "reason": "ok"},
            ],
        },
    )

    out = tmp_path / "artifact.json"
    rc = pilot_assemble_artifact.main(
        [
            "--evaluation-report",
            str(eval_report),
            "--verifier-trace",
            str(trace),
            "--out",
            str(out),
            "--verify-json",
            str(verify_json),
            "--verify-profile",
            "dev",
        ]
    )
    assert rc == 0

    art = json.loads(out.read_text(encoding="utf-8"))
    v = art["guard_evidence"]["invarlock"]["verify"]
    assert v["profile"] == "dev"
    assert v["ok"] is False
    assert v["errors"] == ["ratio:mismatch"]


def test_main_embeds_evaluation_and_verify_v1_fallback_error(tmp_path: Path) -> None:
    eval_report = tmp_path / "evaluation.report.json"
    _write_json(eval_report, {"x": 1})
    trace = tmp_path / "trace.json"
    _write_json(trace, {"schema_version": "verifier_trace.v1"})

    verify_json = tmp_path / "verify.json"
    _write_json(
        verify_json,
        {
            "format_version": "verify-v1",
            "summary": "not-a-dict",
            "results": "not-a-list",
        },
    )

    out = tmp_path / "artifact.json"
    rc = pilot_assemble_artifact.main(
        [
            "--evaluation-report",
            str(eval_report),
            "--verifier-trace",
            str(trace),
            "--verify-json",
            str(verify_json),
            "--verify-profile",
            "ci",
            "--out",
            str(out),
        ]
    )
    assert rc == 0

    art = json.loads(out.read_text(encoding="utf-8"))
    v = art["guard_evidence"]["invarlock"]["verify"]
    assert v["profile"] == "ci"
    assert v["ok"] is False
    assert v["errors"] == ["failed"]


def test_main_verify_json_must_be_object(tmp_path: Path) -> None:
    eval_report = tmp_path / "evaluation.report.json"
    _write_json(eval_report, {"x": 1})
    trace = tmp_path / "trace.json"
    _write_json(trace, {"schema_version": "verifier_trace.v1"})
    verify_json = tmp_path / "verify.json"
    _write_json(verify_json, ["not", "an", "object"])
    out = tmp_path / "artifact.json"

    try:
        pilot_assemble_artifact.main(
            [
                "--evaluation-report",
                str(eval_report),
                "--verifier-trace",
                str(trace),
                "--verify-json",
                str(verify_json),
                "--out",
                str(out),
            ]
        )
        raise AssertionError("expected TypeError")
    except TypeError as e:
        assert "JSON object" in str(e)


def test_main_no_embed_no_verify(tmp_path: Path) -> None:
    eval_report = tmp_path / "evaluation.report.json"
    _write_json(eval_report, {"x": 1})
    trace = tmp_path / "trace.json"
    _write_json(trace, {"schema_version": "verifier_trace.v1"})
    out = tmp_path / "artifact.json"

    rc = pilot_assemble_artifact.main(
        [
            "--evaluation-report",
            str(eval_report),
            "--verifier-trace",
            str(trace),
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    art = json.loads(out.read_text(encoding="utf-8"))
    eref = art["guard_evidence"]["invarlock"]["evaluation_report"]
    assert "embedded" not in eref
    assert "verify" not in art["guard_evidence"]["invarlock"]
