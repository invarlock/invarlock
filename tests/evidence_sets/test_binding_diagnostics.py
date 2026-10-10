"""Rejected compositions identify their conflicting frozen inputs."""

import hashlib

import pytest

from invarlock.evidence_reporting import EvidenceReportError, render_evidence
from invarlock.evidence_sets.contracts import EvidenceSetError
from invarlock.evidence_sets.verification import _require_shared_inputs
from tests.evidence_sets.test_verification import fixture

FIELDS = (
    "baseline_run_sha256",
    "subject_run_sha256",
    "case_set_sha256",
    "subject_artifact_sha256",
    "subject_service_identity_sha256",
)


@pytest.mark.parametrize("field", FIELDS)
@pytest.mark.parametrize("change", ["different", "missing", "null"])
def test_binding_diagnostic_preserves_exact_equality(field, change):
    expected = dict.fromkeys(FIELDS, "sha256:" + "a" * 64)
    observed = dict(reversed(list(expected.items())))
    _require_shared_inputs(observed, expected, message="mismatch")
    if change == "missing":
        del observed[field]
    else:
        observed[field] = None if change == "null" else "sha256:" + "b" * 64
    with pytest.raises(EvidenceSetError) as caught:
        _require_shared_inputs(observed, expected, message="mismatch")
    error = str(caught.value)
    assert field in error and expected[field] in error
    assert "expected" in error and "observed" in error
    assert ("<missing>" in error) == (change == "missing")
    assert ("None" in error) == (change == "null")


def test_hosted_null_is_not_missing_and_unexpected_fields_still_reject():
    with pytest.raises(EvidenceSetError, match="expected None, observed <missing>"):
        _require_shared_inputs(
            {}, {"subject_artifact_sha256": None}, message="mismatch"
        )
    with pytest.raises(EvidenceSetError, match="unexpected shared input fields"):
        _require_shared_inputs({"extra": None}, {}, message="mismatch")
    with pytest.raises(EvidenceSetError, match="expected <missing>, observed None"):
        _require_shared_inputs(
            {"subject_artifact_sha256": None}, {}, message="mismatch"
        )


@pytest.mark.parametrize("side", ["baseline", "subject"])
def test_report_identifies_different_frozen_answers_without_writing(tmp_path, side):
    def change(baseline, subject):
        run = baseline if side == "baseline" else subject
        run["records"][0]["output"] = "different retained answer"

    root, _ = fixture(tmp_path, captured_change=change)

    def snapshot():
        return {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in root.rglob("*")
            if path.is_file()
        }

    before = snapshot()
    output = tmp_path / "rejected.html"
    with pytest.raises(EvidenceReportError) as caught:
        render_evidence(root, html_path=output)
    error = str(caught.value)
    assert f"component runs or case sets differ: {side}_run_sha256" in error
    assert "expected 'sha256:" in error and "observed 'sha256:" in error
    assert not output.exists()
    assert snapshot() == before
