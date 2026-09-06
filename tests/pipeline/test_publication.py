"""Publication must not expose partial results or mutate caller-owned evidence."""

import pytest

from invarlock.pipeline import contracts, create_evidence
from invarlock.pipeline.templates import example_project


def test_evidence_is_a_snapshot_of_caller_owned_inputs():
    baseline, candidate, policy = example_project("classification")
    evidence = create_evidence(baseline, candidate, policy)
    candidate["records"][0]["output"] = "changed later"
    policy["metrics"][0]["maximum_regression"] = 1
    assert evidence["candidate"]["records"][0]["output"] != "changed later"
    assert evidence["policy"]["metrics"][0]["maximum_regression"] != 1


def test_failed_directory_write_is_never_published(tmp_path, monkeypatch):
    def fail_write(*args):
        raise OSError("simulated full disk")

    monkeypatch.setattr(contracts, "write_new", fail_write)
    with pytest.raises(OSError, match="full disk"):
        contracts.write_directory(tmp_path / "result", {"evidence.json": b"{}"})
    assert list(tmp_path.iterdir()) == []


def test_failed_temporary_file_creation_preserves_destination(tmp_path, monkeypatch):
    import tempfile

    destination = tmp_path / "evidence.json"
    destination.write_bytes(b"previous evidence")

    def no_space(**kwargs):
        raise OSError("no space for temporary file")

    monkeypatch.setattr(tempfile, "mkstemp", no_space)
    with pytest.raises(contracts.PipelineError, match="no space"):
        contracts.write_new(destination, b"replacement")
    assert destination.read_bytes() == b"previous evidence"
    assert list(tmp_path.iterdir()) == [destination]


def test_complete_directory_does_not_replace_even_an_empty_destination(tmp_path):
    destination = tmp_path / "result"
    destination.mkdir()
    with pytest.raises(OSError):
        contracts.write_directory(destination, {"comparison.json": b"{}"})
    assert list(destination.iterdir()) == []
    assert list(tmp_path.iterdir()) == [destination]


def test_contract_resource_limits_apply_to_sdk_inputs(monkeypatch):
    baseline, _, _ = example_project("classification")
    monkeypatch.setattr(contracts, "MAX_INPUT_BYTES", 16)
    with pytest.raises(contracts.PipelineError, match="byte limit"):
        contracts.validate(baseline, "run")
