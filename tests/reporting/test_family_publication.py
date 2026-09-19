"""Caller-level publication guarantees for judge and composed evidence reports."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock
from xml.etree.ElementTree import fromstring

import pytest
from typer.testing import CliRunner

from invarlock import captured_contracts, engine
from invarlock.cli.app import app
from invarlock.evidence_sets import reporting as set_reporting
from invarlock.judge_measurements import reporting as judge_reporting
from tests.evidence_sets.test_verification import fixture as publish_set
from tests.judge_measurements.test_evidence_acceptance import _publish as publish_judge


@pytest.fixture(params=["judge", "evidence_set"])
def report_family(request, tmp_path):
    if request.param == "judge":
        publication, _ = publish_judge(tmp_path)
        return publication.path, judge_reporting, "_snapshot", "judge-evidence"
    root, _ = publish_set(tmp_path)
    return root, set_reporting, "build_evidence_set_view", "evidence-set"


def _bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def _outputs(tmp_path: Path) -> dict[str, Path]:
    return {
        "html": tmp_path / "report.html",
        "markdown": tmp_path / "report.md",
        "junit": tmp_path / "report.xml",
    }


@pytest.mark.parametrize(
    "mode", ["terminal", "json", "html", "markdown", "junit", "all"]
)
def test_family_cli_renders_only_requested_files(
    report_family, tmp_path, monkeypatch, mode
):
    root, owner, snapshot_name, contract = report_family
    before = _bytes(root)
    outputs = _outputs(tmp_path)
    selected = list(outputs) if mode == "all" else [mode] if mode in outputs else []
    html = Mock(wraps=owner.render_html)
    junit = Mock(wraps=owner.Element)
    snapshot = Mock(wraps=getattr(owner, snapshot_name))
    monkeypatch.setattr(owner, "render_html", html)
    monkeypatch.setattr(owner, "Element", junit)
    monkeypatch.setattr(owner, snapshot_name, snapshot)
    args = ["report", str(root)]
    for name in selected:
        args.extend([f"--{name}", str(outputs[name])])
    if mode != "terminal":
        args.append("--json")

    result = CliRunner().invoke(app, args)

    assert result.exit_code == 0, result.output
    assert result.stderr == ""
    snapshot.assert_called_once()
    assert html.call_count == int("html" in selected)
    assert junit.call_count == int("junit" in selected)
    for name, destination in outputs.items():
        assert destination.is_file() == (name in selected)
        if name in selected:
            assert destination.read_bytes()
    if "junit" in selected:
        assert fromstring(outputs["junit"].read_bytes()).tag == "testsuite"
    if mode == "terminal":
        assert result.stdout.strip()
    else:
        payload = json.loads(result.stdout)
        assert payload["format_version"] == f"invarlock/{contract}-report-v1"
        assert payload["ok"] is True
        assert payload["errors"] == []
        assert (
            payload["requested_outputs"]
            == payload["written_outputs"]
            == {name: str(outputs[name]) for name in selected}
        )
        if contract == "evidence-set":
            assert payload["recipient_acceptance"] == "not_performed"
        else:
            assert payload["assurance"]["recipient_acceptance"] == "not_performed"
    assert _bytes(root) == before


@pytest.mark.parametrize("failed_output", ["html", "markdown", "junit"])
@pytest.mark.parametrize("consumer", ["api", "cli_json", "cli_terminal"])
def test_family_partial_write_preserves_completed_outputs_and_contract(
    report_family, tmp_path, monkeypatch, failed_output, consumer
):
    root, _, _, contract = report_family
    before = _bytes(root)
    outputs = _outputs(tmp_path)
    writer = captured_contracts.atomic_write
    attempted = []

    def write(path, raw):
        attempted.append(path)
        if path == outputs[failed_output]:
            raise OSError("report destination unavailable")
        writer(path, raw)

    monkeypatch.setattr(captured_contracts, "atomic_write", write)
    failed_index = list(outputs).index(failed_output)
    written = {name: str(path) for name, path in list(outputs.items())[:failed_index]}
    if consumer == "api":
        with pytest.raises(engine.EvidenceReportError) as caught:
            engine.render_evidence(
                root, **{f"{name}_path": path for name, path in outputs.items()}
            )
        assert caught.value.exit_code == 2
        assert caught.value.failed_output == failed_output
        assert caught.value.written_outputs == written
        payload = caught.value.payload
    else:
        args = ["report", str(root)]
        for name, path in outputs.items():
            args.extend([f"--{name}", str(path)])
        if consumer == "cli_json":
            args.append("--json")
        result = CliRunner().invoke(app, args)
        assert result.exit_code == 2, result.output
        assert result.stderr == ""
        if consumer == "cli_json":
            payload = json.loads(result.stdout)
        else:
            assert "report destination unavailable" in result.stdout
            assert f"Failed output: {failed_output}" in result.stdout
            for name in outputs:
                assert (f"Written {name}:" in result.stdout) == (name in written)
            payload = None
    if payload is not None:
        assert payload == {
            "format_version": f"invarlock/{contract}-report-v1",
            "kind": "judge" if contract == "judge-evidence" else "evidence_set",
            "ok": False,
            "requested_outputs": {name: str(path) for name, path in outputs.items()},
            "written_outputs": written,
            "failed_output": failed_output,
            "errors": ["report destination unavailable"],
        }
    assert attempted == list(outputs.values())[: failed_index + 1]
    for name, path in outputs.items():
        assert path.is_file() == (name in written)
        if name in written:
            assert path.read_bytes()
    assert _bytes(root) == before


def test_family_render_failure_publishes_nothing(report_family, tmp_path, monkeypatch):
    root, owner, _, contract = report_family
    before = _bytes(root)
    outputs = _outputs(tmp_path)
    html = Mock(wraps=owner.render_html)
    monkeypatch.setattr(owner, "render_html", html)
    monkeypatch.setattr(
        owner, "tostring", Mock(side_effect=ValueError("cannot encode JUnit"))
    )
    writer = Mock(side_effect=AssertionError("render failure reached publication"))
    monkeypatch.setattr(captured_contracts, "atomic_write", writer)

    with pytest.raises(
        engine.EvidenceReportError, match="cannot encode JUnit"
    ) as caught:
        engine.render_evidence(
            root, **{f"{name}_path": path for name, path in outputs.items()}
        )

    html.assert_called_once()
    writer.assert_not_called()
    assert caught.value.exit_code == 2
    assert caught.value.failed_output is None
    assert caught.value.written_outputs == {}
    assert caught.value.payload["format_version"] == f"invarlock/{contract}-report-v1"
    assert not any(path.exists() for path in outputs.values())
    assert _bytes(root) == before
