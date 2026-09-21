"""A capture cannot silently omit expected cases or overwrite original exports."""

import json

import pytest

from invarlock.engine import EvaluationRecordsError, load_run
from invarlock.evaluation_records import adapters

SOURCE = {"name": "capture", "version": "1"}
IDENTITY = "sha256:" + "a" * 64


def raw_rows():
    return b'{"id":"one","input":"Q","expected":"A","output":"A"}\n'


def publish(tmp_path, raw=None, **overrides):
    options = {
        "adapter": "jsonl",
        "source": SOURCE,
        "run_id": "run",
        "artifact_digest": IDENTITY,
        "expected_ids": ["one"],
        **overrides,
    }
    return adapters.write_evaluator_export(
        raw_rows() if raw is None else raw, tmp_path / "export.jsonl", **options
    )


def test_lossless_export_is_independently_importable(tmp_path):
    run = publish(tmp_path)
    path = tmp_path / "export.jsonl"
    assert path.read_bytes() == raw_rows()
    assert (
        load_run(
            path, adapter="jsonl", source=SOURCE, run_id="run", artifact_digest=IDENTITY
        )
        == run
    )
    with pytest.raises(FileExistsError):
        publish(tmp_path)
    assert path.read_bytes() == raw_rows()


@pytest.mark.parametrize(
    "ids", [[], ["missing"], ["one", "missing"], ["one", "one"], [1], "one"]
)
def test_invalid_or_missing_schedule_never_publishes(tmp_path, ids):
    with pytest.raises(EvaluationRecordsError):
        publish(tmp_path, expected_ids=ids)
    assert not (tmp_path / "export.jsonl").exists()


@pytest.mark.parametrize(
    "raw",
    [
        b"",
        b"{}",
        raw_rows() * 2,
        b'{"id":"one","id":"two"}',
        b'{"cost":NaN}',
        "not bytes",
    ],
)
def test_malformed_capture_never_publishes(tmp_path, raw):
    with pytest.raises(EvaluationRecordsError):
        publish(tmp_path, raw)
    assert not (tmp_path / "export.jsonl").exists()


def test_failed_record_is_preserved(tmp_path):
    row = json.loads(raw_rows())
    row.update(output=None, error="upstream_error")
    run = publish(tmp_path, json.dumps(row).encode())
    assert run["records"][0]["error"] == "upstream_error"
    assert run["records"][0]["output"] is None


def test_size_limits_precede_publication(tmp_path, monkeypatch):
    monkeypatch.setattr(adapters, "MAX_INPUT_BYTES", 1)
    with pytest.raises(EvaluationRecordsError):
        publish(tmp_path)
    assert not (tmp_path / "export.jsonl").exists()
