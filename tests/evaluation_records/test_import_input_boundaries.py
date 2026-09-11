"""Native imports retain explicit identities and reject replaced artifact inputs."""

import errno
import hashlib
import json
import os

import pytest

from invarlock.engine import EvaluationRecordsError, physical_file_digest
from invarlock.evaluation_records import adapters

_ROW = {"id": "one", "input": "q", "expected": "yes", "output": "yes"}
_OPTIONS = {
    "adapter": "jsonl",
    "source": {"name": "fixture", "version": "1"},
    "run_id": "one",
    "artifact_digest": "sha256:" + "a" * 64,
}


def test_existing_export_rejects_unknown_adapter(tmp_path):
    path = tmp_path / "export.jsonl"
    path.write_text(json.dumps(_ROW) + "\n")
    with pytest.raises(EvaluationRecordsError, match="unsupported adapter 'guess'"):
        adapters.load_run(path, adapter="guess")


@pytest.mark.parametrize("missing", ["source", "run_id", "artifact_digest"])
def test_native_import_requires_each_pipeline_identity(tmp_path, missing):
    path = tmp_path / "export.jsonl"
    path.write_text(json.dumps(_ROW) + "\n")
    options = {key: value for key, value in _OPTIONS.items() if key != missing}
    with pytest.raises(EvaluationRecordsError, match="native import requires"):
        adapters.load_run(path, **options)


@pytest.mark.parametrize("prompt", [None, {}, {"raw": ["unrendered", "parts"]}])
def test_promptfoo_import_requires_a_rendered_text_prompt(tmp_path, prompt):
    path = tmp_path / "promptfoo.jsonl"
    row = {
        "testIdx": 0,
        "promptIdx": 0,
        "testCase": {"vars": {"question": "q"}},
        "prompt": prompt,
        "response": {"output": "yes"},
    }
    path.write_text(json.dumps(row) + "\n")
    options = {**_OPTIONS, "adapter": "promptfoo-jsonl"}
    with pytest.raises(EvaluationRecordsError, match="actual rendered prompt"):
        adapters.load_run(path, **options)


def test_snapshot_parser_enforces_exact_byte_limit(monkeypatch):
    raw = (json.dumps(_ROW) + "\n").encode()
    monkeypatch.setattr(adapters, "MAX_INPUT_BYTES", len(raw))
    run = adapters._parse_run_bytes(raw, **_OPTIONS)
    assert run["records"][0]["output"] == "yes"
    assert run["source_digest"] == "sha256:" + hashlib.sha256(raw).hexdigest()
    monkeypatch.setattr(adapters, "MAX_INPUT_BYTES", len(raw) - 1)
    with pytest.raises(EvaluationRecordsError, match="export exceeds its byte limit"):
        adapters._parse_run_bytes(raw, **_OPTIONS)


def test_physical_digest_rejects_replacement_between_stat_and_open(
    tmp_path, monkeypatch
):
    artifact = tmp_path / "model.bin"
    artifact.write_bytes(b"original artifact")
    replacement = tmp_path / "replacement.bin"
    replacement.write_bytes(b"replacement artifact")
    original_open = os.open
    opened = []

    def replace_before_open(name, flags, *args, **kwargs):
        if name == artifact.name:
            replacement.replace(artifact)
        descriptor = original_open(name, flags, *args, **kwargs)
        opened.append(descriptor)
        return descriptor

    monkeypatch.setattr(os, "open", replace_before_open)
    with pytest.raises(EvaluationRecordsError, match="artifact changed during hashing"):
        physical_file_digest(artifact)
    assert artifact.read_bytes() == b"replacement artifact"
    assert opened
    for descriptor in opened:
        with pytest.raises(OSError) as error:
            os.fstat(descriptor)
        assert error.value.errno == errno.EBADF
