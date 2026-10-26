"""Whole evidence-set report snapshots and rendered output parity."""

import dataclasses
import json
import os
import subprocess
from pathlib import Path

import pytest

from invarlock.evidence_sets.reporting import (
    build_evidence_set_view,
    render_evidence_set,
)
from invarlock.report_presentation import render_html, render_markdown
from tests.evidence_sets.test_verification import fixture, write

ROOT = Path(__file__).resolve().parents[2]


def native(request):
    target = Path(os.environ.get("CARGO_TARGET_DIR", ROOT / "native/target"))
    binary = target / "debug/examples/attestation"
    result = subprocess.run(
        [str(binary)], input=json.dumps(request) + "\n", text=True, capture_output=True
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


@pytest.mark.parametrize(
    "mode",
    [
        "pass",
        "incomplete",
        "minimum",
        "advisory",
        "unsigned",
        "cases",
        "pin",
        "statement",
        "nested",
        "case-id",
    ],
)
def test_combined_view_snapshot_and_rendering(mode, tmp_path):
    directory = tmp_path.resolve()
    options = (
        {"incomplete": True}
        if mode == "incomplete"
        else {"captured_minimum": 20}
        if mode == "minimum"
        else {"role": "advisory"}
        if mode == "advisory"
        else {}
    )
    if mode == "cases":

        def change(baseline, subject):
            for run in (baseline, subject):
                run["records"][0]["input"] = "different case"

        options["captured_change"] = change
    root, _ = fixture(directory, **options)
    index_path = root / "evidence-set.json"
    if mode == "unsigned":
        path = root / "judge/envelope.json"
        envelope = json.loads(path.read_bytes())
        envelope.update(signature=None, signer=None)
        path.chmod(0o600)
        write(path, envelope)
        from invarlock.captured_contracts import sha

        index = json.loads(index_path.read_bytes())
        index["members"]["judge"]["statement_sha256"] = sha(path.read_bytes())
        write(index_path, index)
    if mode == "pin":
        index = json.loads(index_path.read_bytes())
        index["members"]["judge"]["statement_sha256"] = "sha256:" + "f" * 64
        write(index_path, index)
    if mode == "statement":
        path = root / "judge/envelope.json"
        path.chmod(0o600)
        path.write_bytes(path.read_bytes() + b" ")
    if mode == "nested":
        (root / "judge").chmod(0o700)
        write(root / "judge/evidence-set.json", {})
    case_ids = ("case-1",) if mode == "case-id" else ()
    try:
        view, facts, digest = build_evidence_set_view(root, case_ids=case_ids)
        expected = {
            "ok": {
                "view": dataclasses.asdict(view),
                "facts": facts,
                "index_sha256": digest,
            }
        }
        expected = json.loads(json.dumps(expected))
    except (ValueError, OSError, KeyError, TypeError) as error:
        expected = {"error": str(error)}
    actual = native(
        {"operation": "set_view", "path": str(root), "case_ids": list(case_ids)}
    )
    assert actual == expected
    if "error" in expected:
        return
    value = actual["ok"]["view"]
    assert native({"operation": "markdown", "view": value}) == {
        "ok": render_markdown(view)
    }
    assert native(
        {"operation": "markdown", "view": value, "include_details": True}
    ) == {"ok": render_markdown(view, include_details=True)}
    assert native({"operation": "html", "view": value}) == {"ok": render_html(view)}
    junit = directory / "reference.xml"
    render_evidence_set(root, junit_path=junit, case_ids=case_ids)
    assert native({"operation": "set_junit", "view": value}) == {
        "ok": junit.read_text()
    }


@pytest.mark.parametrize(
    "mode",
    [
        "files",
        "nested-output",
        "collision",
        "inside",
        "duplicate",
        "overlap",
        "symlink-parent",
    ],
)
def test_report_publication_destinations_and_bytes(mode, tmp_path):
    from invarlock.evidence_reporting import EvidenceReportError

    directory = tmp_path.resolve()
    root, _ = fixture(directory)
    target = directory / "outputs"
    target.mkdir()
    outputs = {
        "html_path": target / "report.html",
        "markdown_path": target / "report.md",
        "junit_path": target / "report.xml",
    }
    if mode == "nested-output":
        outputs["html_path"] = target / "new/deep/report.html"
    if mode == "collision":
        outputs["markdown_path"].write_text("preserve foreign output")
    if mode == "inside":
        outputs["html_path"] = root / "new/report.html"
    if mode == "duplicate":
        outputs["markdown_path"] = outputs["html_path"]
    if mode == "overlap":
        outputs["markdown_path"] = outputs["html_path"] / "child"
    if mode == "symlink-parent":
        (target / "link").symlink_to(directory, target_is_directory=True)
        outputs["html_path"] = target / "link/report.html"
    original = {p: p.read_bytes() for p in outputs.values() if p.is_file()}
    rendered = {}
    try:
        report = render_evidence_set(root, **outputs, explain=True)
        expected = {
            "text": report.text,
            "payload": json.loads(report.as_json()),
            "status": 0,
        }
        rendered = {p: p.read_bytes() for p in outputs.values()}
        for path in rendered:
            path.unlink()
    except EvidenceReportError as error:
        expected = {"text": "", "payload": error.payload, "status": 2}
    actual = native(
        {
            "operation": "set_render",
            "evidence_path": str(root),
            "explain": True,
            **{name: str(path) for name, path in outputs.items()},
        }
    )
    assert actual == {"ok": expected}
    for path, payload in rendered.items():
        assert path.read_bytes() == payload
    for path, payload in original.items():
        assert path.read_bytes() == payload
