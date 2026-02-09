from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.verifai2_2026 import make_prompt_set, verifier_trace_from_cases


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=True) + "\n" for r in rows), encoding="utf-8"
    )


def test_read_jsonl_rejects_invalid_json(tmp_path: Path) -> None:
    p = tmp_path / "bad.jsonl"
    p.write_text("{\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"Invalid JSONL"):
        make_prompt_set._read_jsonl(p)


def test_read_jsonl_rejects_non_object(tmp_path: Path) -> None:
    p = tmp_path / "bad2.jsonl"
    p.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"Expected JSON object"):
        make_prompt_set._read_jsonl(p)


def test_read_jsonl_skips_blank_lines(tmp_path: Path) -> None:
    p = tmp_path / "ok.jsonl"
    p.write_text("\n" + json.dumps({"id": "a", "text": "x"}) + "\n", encoding="utf-8")
    rows = make_prompt_set._read_jsonl(p)
    assert len(rows) == 1


def test_format_prompt_missing_text_field() -> None:
    with pytest.raises(KeyError, match=r"Missing text field"):
        make_prompt_set._format_prompt("{text}", {"id": "x"}, text_field="prompt")


def test_format_prompt_template_missing_key() -> None:
    with pytest.raises(KeyError, match=r"Template references missing key"):
        make_prompt_set._format_prompt(
            "{missing}", {"id": "x", "text": "hi"}, text_field="text"
        )


def test_main_no_records_returns_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    in_path = tmp_path / "empty.jsonl"
    in_path.write_text("", encoding="utf-8")
    out_path = tmp_path / "prompt_set.json"

    rc = make_prompt_set.main(["--in", str(in_path), "--out", str(out_path)])
    assert rc == 2
    assert "No records found." in capsys.readouterr().err


def test_main_hash_only_limit_and_revision_default(tmp_path: Path) -> None:
    in_path = tmp_path / "tasks.jsonl"
    _write_jsonl(
        in_path,
        [
            {"id": "a", "text": "print(1)"},
            {"id": "b", "text": "print(2)"},
        ],
    )
    sel = tmp_path / "select.py"
    sel.write_text("print('select')\n", encoding="utf-8")

    out_path = tmp_path / "prompt_set.json"
    rc = make_prompt_set.main(
        [
            "--in",
            str(in_path),
            "--out",
            str(out_path),
            "--limit",
            "1",
            "--dataset-name",
            "local_jsonl",
            "--dataset-config",
            "code_canary",
            "--dataset-manifest-sha256",
            _sha256_hex(b"manifest"),
            "--selection-script",
            str(sel),
        ]
    )
    assert rc == 0

    obj = json.loads(out_path.read_text(encoding="utf-8"))
    assert obj["mode"] == "hash_only"
    assert obj["dataset"]["name"] == "local_jsonl"
    assert obj["dataset"]["config"] == "code_canary"
    assert obj["dataset"]["manifest_sha256"] == _sha256_hex(b"manifest")
    # Default dataset revision: sha256 of input file bytes.
    assert obj["dataset"]["revision"] == _sha256_hex(in_path.read_bytes())

    assert obj["selection_script_sha256"] == _sha256_hex(sel.read_bytes())

    # --limit trimmed to 1 item.
    assert [it["id"] for it in obj["items"]] == ["a"]
    assert "text" not in obj["items"][0]

    # Digest is computed per verifier_trace_contract.md: it must ignore embedded
    # prompt text and non-identifier dataset metadata like manifest_sha256.
    expected_digest = verifier_trace_from_cases._compute_prompt_set_digest(obj)
    assert obj["digest_sha256"] == expected_digest


def test_main_embedded_and_revision_provided(tmp_path: Path) -> None:
    in_path = tmp_path / "tasks.jsonl"
    _write_jsonl(in_path, [{"id": "x", "text": "hello"}])
    out_path = tmp_path / "prompt_set.json"

    rc = make_prompt_set.main(
        [
            "--in",
            str(in_path),
            "--out",
            str(out_path),
            "--mode",
            "embedded",
            "--dataset-revision",
            "rev1",
        ]
    )
    assert rc == 0
    obj = json.loads(out_path.read_text(encoding="utf-8"))
    assert obj["mode"] == "embedded"
    assert obj["dataset"]["revision"] == "rev1"
    assert obj["items"][0]["text"] == "hello"


def test_main_duplicate_id_raises(tmp_path: Path) -> None:
    in_path = tmp_path / "tasks.jsonl"
    _write_jsonl(in_path, [{"id": "dup", "text": "a"}, {"id": "dup", "text": "b"}])
    out_path = tmp_path / "prompt_set.json"

    with pytest.raises(ValueError, match=r"Duplicate id"):
        make_prompt_set.main(["--in", str(in_path), "--out", str(out_path)])


def test_main_missing_id_field_raises(tmp_path: Path) -> None:
    in_path = tmp_path / "tasks.jsonl"
    _write_jsonl(in_path, [{"text": "a"}])
    out_path = tmp_path / "prompt_set.json"

    with pytest.raises(ValueError, match=r"Missing/invalid id field"):
        make_prompt_set.main(["--in", str(in_path), "--out", str(out_path)])
