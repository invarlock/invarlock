from __future__ import annotations

import base64
import json
import re
from pathlib import Path

import pytest

from invarlock.evidence_pack_json import parse_json_bytes

ROOT = Path(__file__).resolve().parents[2]
SCANNED_ROOTS = ("docs", "examples", "src", "tests")
SCANNED_FILES = ("CHANGELOG.md", "Makefile", "README.md", "SECURITY.md", "SUPPORT.md")
TEXT_SUFFIXES = {
    ".cue",
    ".json",
    ".md",
    ".pem",
    ".py",
    ".rego",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
REMOVED_ROLE = b"pro" + b"ducer"
ROLE_WORD = re.compile(
    rb"(?<![A-Za-z0-9_])" + REMOVED_ROLE + rb"(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
ENCODED_ROLE = base64.b64encode(REMOVED_ROLE).rstrip(b"=")


def _project_authored_payload(path: Path) -> bytes:
    """Exclude only third-party benchmark prose from terminology checks."""

    if "signed-transactions" not in path.parts or path.parts[-3:] != (
        "evidence",
        "schedule",
        "runtime-behavioral-schedule.json",
    ):
        return path.read_bytes()
    value = parse_json_bytes(path.read_bytes(), label="retained schedule")
    records = value.get("records") if isinstance(value, dict) else None
    if not isinstance(records, list):
        raise AssertionError(f"retained schedule records are invalid: {path}")
    for record in records:
        parts = record.get("input_parts") if isinstance(record, dict) else None
        if not isinstance(parts, list):
            raise AssertionError(f"retained schedule input parts are invalid: {path}")
        for part in parts:
            if not isinstance(part, dict) or not isinstance(part.get("text"), str):
                raise AssertionError(f"retained schedule text part is invalid: {path}")
            part["text"] = ""
    return json.dumps(value, sort_keys=True).encode()


def _contains_removed_role(path: Path) -> bool:
    payload = _project_authored_payload(path)
    return ROLE_WORD.search(payload) is not None or ENCODED_ROLE in payload


def test_release_tree_uses_explicit_trust_and_evaluation_roles() -> None:
    files = [ROOT / name for name in SCANNED_FILES]
    files.extend(
        path
        for root in SCANNED_ROOTS
        for path in (ROOT / root).rglob("*")
        if path.is_file() and path.suffix in TEXT_SUFFIXES
    )
    findings = [
        path.relative_to(ROOT).as_posix()
        for path in files
        if _contains_removed_role(path)
    ]
    names = [
        path.relative_to(ROOT).as_posix()
        for root in SCANNED_ROOTS
        for path in (ROOT / root).rglob("*")
        if path.is_file() and ROLE_WORD.search(path.name.encode())
    ]

    assert findings == []
    assert names == []


def test_retained_schedule_scan_excludes_only_benchmark_prompt_text(
    tmp_path: Path,
) -> None:
    schedule = (
        tmp_path / "signed-transactions/example/evidence/schedule/"
        "runtime-behavioral-schedule.json"
    )
    schedule.parent.mkdir(parents=True)
    value = {
        "format_version": "fixed",
        "records": [
            {"input_parts": [{"kind": "text", "text": REMOVED_ROLE.decode("ascii")}]}
        ],
    }
    schedule.write_text(json.dumps(value), encoding="utf-8")

    assert ROLE_WORD.search(_project_authored_payload(schedule)) is None

    value["format_version"] = REMOVED_ROLE.decode("ascii")
    schedule.write_text(json.dumps(value), encoding="utf-8")
    assert ROLE_WORD.search(_project_authored_payload(schedule)) is not None


@pytest.mark.parametrize("records", [None, [{}], [{"input_parts": [None]}]])
def test_retained_schedule_scan_rejects_malformed_benchmark_structure(
    records: object,
    tmp_path: Path,
) -> None:
    schedule = (
        tmp_path / "signed-transactions/example/evidence/schedule/"
        "runtime-behavioral-schedule.json"
    )
    schedule.parent.mkdir(parents=True)
    schedule.write_text(json.dumps({"records": records}), encoding="utf-8")

    with pytest.raises(AssertionError, match="retained schedule"):
        _project_authored_payload(schedule)
