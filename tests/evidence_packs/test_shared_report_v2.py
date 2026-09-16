"""V2 output accounting on both evidence families, with unchanged native v1."""

import json
from itertools import permutations
from unittest.mock import Mock
from xml.etree.ElementTree import fromstring

import pytest
from typer.testing import CliRunner

from invarlock import captured_contracts, captured_reporting, engine
from invarlock.cli.app import app
from tests.core.test_captured_sdk_omissions import _inputs
from tests.evidence_packs.test_evidence_reporting import _evidence


def _pack(tmp_path, captured):
    if not captured:
        return _evidence(tmp_path)[0]
    _inputs(tmp_path)
    return engine.evaluate_request_file(
        tmp_path / "request.json", signing_key_path=None, unsigned=True
    ).evidence_path


@pytest.mark.parametrize("captured", [False, True])
def test_shared_report_outputs_and_native_v1_matrix(tmp_path, captured):
    pack = _pack(tmp_path, captured)
    before = {
        p.relative_to(pack): p.read_bytes() for p in pack.rglob("*") if p.is_file()
    }
    default = engine.render_evidence(pack)
    assert isinstance(
        default, engine.EvidenceReportV2 if captured else engine.EvidenceReport
    )
    html = engine.render_evidence(pack, html_path=tmp_path / "only.html")
    assert isinstance(
        html, engine.EvidenceReportV2 if captured else engine.EvidenceReport
    )
    paths = {
        "html": tmp_path / "report.html",
        "markdown": tmp_path / "report.md",
        "junit": tmp_path / "report.xml",
    }
    result = engine.render_evidence(
        pack,
        html_path=paths["html"],
        markdown_path=paths["markdown"],
        junit_path=paths["junit"],
    )
    assert isinstance(result, engine.EvidenceReportV2)
    payload = json.loads(result.as_json())
    assert payload["kind"] == ("captured" if captured else "runtime")
    assert (
        payload["requested_outputs"]
        == payload["written_outputs"]
        == {k: str(v) for k, v in paths.items()}
    )
    assert payload["failed_output"] is None
    assert payload["errors"] == []
    assert fromstring(paths["junit"].read_bytes()).tag == "testsuite"
    assert before == {
        p.relative_to(pack): p.read_bytes() for p in pack.rglob("*") if p.is_file()
    }
    cli = CliRunner().invoke(
        app, ["report", str(pack), "--markdown", str(tmp_path / "cli.md"), "--json"]
    )
    assert cli.exit_code == 0, cli.output
    assert json.loads(cli.stdout)["written_outputs"] == {
        "markdown": str(tmp_path / "cli.md")
    }


@pytest.mark.parametrize("captured", [False, True])
def test_duplicate_destinations_reject_before_writing(tmp_path, captured):
    pack = _pack(tmp_path, captured)
    destination = tmp_path / "same"
    with pytest.raises(engine.EvidenceReportError, match="collide") as caught:
        engine.render_evidence(pack, html_path=destination, markdown_path=destination)
    assert caught.value.written_outputs == {}
    assert not destination.exists()


@pytest.mark.parametrize("captured", [False, True])
@pytest.mark.parametrize(
    ("ancestor", "descendant"), list(permutations(("html", "markdown", "junit"), 2))
)
def test_overlapping_destinations_reject_before_writing(
    tmp_path, monkeypatch, captured, ancestor, descendant
):
    pack = _pack(tmp_path, captured)
    output_root = tmp_path / "outputs"
    paths = {name: output_root / name for name in ("html", "markdown", "junit")}
    paths[ancestor] = output_root / "rendered"
    paths[descendant] = output_root / "rendered" / "summary.md"
    writer = Mock(wraps=captured_contracts.atomic_write)
    monkeypatch.setattr(captured_contracts, "atomic_write", writer)

    with pytest.raises(engine.EvidenceReportError, match="collide") as caught:
        engine.render_evidence(
            pack, **{f"{name}_path": path for name, path in paths.items()}
        )

    writer.assert_not_called()
    assert caught.value.written_outputs == {}
    assert caught.value.failed_output is None
    assert not output_root.exists()


@pytest.mark.parametrize("captured", [False, True])
def test_second_output_failure_keeps_first_and_stops_publication(
    tmp_path, monkeypatch, captured
):
    pack = _pack(tmp_path, captured)
    writer = captured_contracts.atomic_write

    def write(path, raw):
        if path.suffix == ".md":
            raise OSError("second destination unavailable")
        writer(path, raw)

    monkeypatch.setattr(captured_contracts, "atomic_write", write)
    with pytest.raises(engine.EvidenceReportError) as caught:
        engine.render_evidence(
            pack,
            html_path=tmp_path / "out.html",
            markdown_path=tmp_path / "out.md",
            junit_path=tmp_path / "out.xml",
        )
    assert caught.value.exit_code == 2
    assert caught.value.failed_output == "markdown"
    assert caught.value.written_outputs == {"html": str(tmp_path / "out.html")}
    assert (tmp_path / "out.html").is_file()
    assert not (tmp_path / "out.md").exists()
    assert not (tmp_path / "out.xml").exists()


def test_malformed_captured_payload_cannot_reach_view_or_native_fallback(
    tmp_path, monkeypatch
):
    pack = _pack(tmp_path, True)
    policy = pack / "inputs/policy.json"
    policy.chmod(0o600)
    policy.write_text("{}")
    forbidden = Mock(
        side_effect=AssertionError("invalid evidence reached presentation")
    )
    monkeypatch.setattr(captured_reporting, "_view", forbidden)
    monkeypatch.setattr(
        "invarlock.evidence_reporting._render_native_evidence", forbidden
    )
    with pytest.raises(engine.EvidenceReportError):
        engine.render_evidence(pack, markdown_path=tmp_path / "out.md")
    forbidden.assert_not_called()
    assert not (tmp_path / "out.md").exists()


def test_native_missing_anchors_remain_type_error(tmp_path):
    pack = _pack(tmp_path, False)
    with pytest.raises(TypeError, match="anchors"):
        engine.verify_evidence(
            pack,
            policy_path=tmp_path / "policy.json",
            expected_signer="sha256:" + "a" * 64,
        )
