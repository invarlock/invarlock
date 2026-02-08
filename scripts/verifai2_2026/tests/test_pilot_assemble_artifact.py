from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.verifai2_2026 import pilot_assemble_artifact


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_json(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=True) + "\n", encoding="utf-8")


def test_main_embeds_evaluation_and_verify(tmp_path: Path) -> None:
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
