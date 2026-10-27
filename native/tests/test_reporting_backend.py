"""Public API and CLI report parity with Python presentation disabled after setup."""

import json
import os

import pytest
from typer.testing import CliRunner

pytestmark = pytest.mark.skipif(
    not os.environ.get("INVARLOCK_NATIVE_REPORT_BINDINGS"),
    reason="native reporting extension qualification is opt-in",
)


@pytest.mark.parametrize("family", ["captured", "native", "judge", "evidence_set"])
def test_native_report_facade_and_cli(family, tmp_path, monkeypatch):
    from invarlock.cli.app import app
    from invarlock.evidence_reporting import EvidenceReportError, render_evidence
    from tests.evidence_packs.test_evidence_pack import _publish
    from tests.evidence_sets.test_verification import fixture

    directory = tmp_path.resolve()
    if family == "native":
        evidence, *_ = _publish(directory)
    else:
        root, _ = fixture(directory)
        evidence = (
            root
            if family == "evidence_set"
            else root / ("deterministic" if family == "captured" else "judge")
        )
    expected = render_evidence(evidence, explain=True)
    plain = render_evidence(evidence)
    occupied = directory / "occupied.html"
    occupied.write_text("retained")
    with pytest.raises(EvidenceReportError) as failure:
        render_evidence(evidence, html_path=occupied)

    def forbidden(*args, **kwargs):
        raise AssertionError("native reporting called Python replay or presentation")

    import invarlock.captured_reporting as captured
    import invarlock.evidence_reporting as reporting
    import invarlock.evidence_sets.reporting as sets
    import invarlock.judge_measurements.evidence as judge_evidence
    import invarlock.judge_measurements.reporting as judge
    import invarlock.report_presentation as presentation

    for module, names in [
        (captured, ["_load", "_view"]),
        (judge, ["_snapshot", "_view"]),
        (
            reporting,
            ["_render_native_evidence", "render_report_html", "render_report_markdown"],
        ),
        (sets, ["build_evidence_set_view"]),
        (presentation, ["render_html", "render_markdown"]),
        (judge_evidence, ["replay_judge_evidence"]),
    ]:
        for name in names:
            monkeypatch.setattr(module, name, forbidden)
    actual = render_evidence(evidence, backend="rust", explain=True)
    assert type(actual) is type(expected)
    assert actual.text == expected.text
    assert json.loads(actual.as_json()) == json.loads(expected.as_json())
    with pytest.raises(EvidenceReportError) as native_failure:
        render_evidence(evidence, html_path=occupied, backend="rust")
    assert native_failure.value.payload == failure.value.payload
    assert native_failure.value.exit_code == failure.value.exit_code
    result = CliRunner().invoke(
        app, ["report", str(evidence), "--backend", "rust", "--json"]
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == json.loads(plain.as_json())


def test_client_handoff_requires_implemented_backend(tmp_path):
    from invarlock.cli.app import app

    destination = tmp_path / "handoff"
    result = CliRunner().invoke(
        app,
        ["report", str(tmp_path), "--package", str(destination), "--backend", "rust"],
    )
    assert result.exit_code == 2
    assert "Client handoff packages require --backend python" in result.output
    assert not destination.exists()
