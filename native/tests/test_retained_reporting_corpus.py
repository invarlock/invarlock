"""Report parity on retained published evidence, with immutable inputs.

This exercises historical retained facts; it is not a new live execution or a
recipient trust decision. Native judge report loading separately replays judge
measurements; native/captured report loading authenticates and checks stored facts.
"""

import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
CORPUS = sorted(
    {
        p.parent.parent.relative_to(ROOT).as_posix()
        for base in [
            ROOT / "examples",
            ROOT / "public_evidence",
            ROOT / "tests/fixtures/compatibility",
        ]
        for p in base.rglob("evaluation.report.json")
    }
)
pytestmark = pytest.mark.skipif(
    not os.environ.get("INVARLOCK_NATIVE_REPORT_BINDINGS"),
    reason="native reporting extension qualification is opt-in",
)


@pytest.mark.parametrize("relative", CORPUS)
def test_retained_report_parity(relative, tmp_path):
    from invarlock.evidence_reporting import EvidenceReportError, render_evidence

    evidence = ROOT / relative
    outputs = {
        "html_path": tmp_path / "report.html",
        "markdown_path": tmp_path / "report.md",
        "junit_path": tmp_path / "report.xml",
    }
    try:
        expected = render_evidence(evidence, explain=True, **outputs)
    except EvidenceReportError as error:
        with pytest.raises(EvidenceReportError) as actual:
            render_evidence(evidence, backend="rust", explain=True, **outputs)
        assert str(actual.value) == str(error)
        assert actual.value.payload == error.payload
        return
    expected_bytes = {key: path.read_bytes() for key, path in outputs.items()}
    for path in outputs.values():
        path.unlink()
    actual = render_evidence(evidence, backend="rust", explain=True, **outputs)
    assert actual.text == expected.text
    assert json.loads(actual.as_json()) == json.loads(expected.as_json())
    for key, path in outputs.items():
        assert path.read_bytes() == expected_bytes[key]
