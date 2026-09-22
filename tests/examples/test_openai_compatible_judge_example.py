"""Endpoint setup freezes real contracts without server or inference calls."""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from invarlock.judge_measurements.contracts import render_judge_request
from invarlock.judge_measurements.workflow import load_judge_request

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "openai_compatible_judge_example",
    ROOT / "examples/openai-compatible-judge/prepare.py",
)
EXAMPLE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXAMPLE)
FIXTURE = ROOT / "examples/native-local-judge"


def prepare(workspace, **kwargs):
    options = {
        "service": "vllm",
        "base_url": "http://127.0.0.1:8000/v1",
        "model": "judge-model",
    }
    options.update(kwargs)
    return EXAMPLE.prepare(
        workspace,
        FIXTURE / "baseline_run.json",
        FIXTURE / "subject_run.json",
        **options,
    )


@pytest.mark.parametrize(
    "service", ["vllm", "ollama", "lm_studio", "openai_compatible"]
)
@pytest.mark.parametrize("reference_mode", ["per_case", "none"])
def test_services_freeze_valid_v3_requests_without_local_artifact(
    tmp_path, service, reference_mode
):
    result = prepare(tmp_path, service=service, reference_mode=reference_mode)
    request = load_judge_request(tmp_path / "request.json")
    assert request.integration == "openai-compatible-judge"
    assert request.model is None
    assert result["expected_trials"] == 2
    plan = json.loads((tmp_path / "plan.json").read_bytes())
    assert plan["judge"]["model_identity"] == {
        "kind": "hosted_api",
        "weights_sha256": None,
    }
    assert plan["judge"]["provider"] == "openai_compatible"
    original = json.loads((FIXTURE / "baseline_run.json").read_bytes())
    assert json.loads((tmp_path / "baseline_run.json").read_bytes()) == original
    row = original["records"][0]
    wire = render_judge_request(
        plan,
        input_text=row["input"],
        answer_text=row["output"],
        reference_text=row["expected"],
    )
    rendered = json.loads(json.loads(wire)["messages"][-1]["content"])
    assert ("reference" in rendered) == (reference_mode == "per_case")
    collection = json.loads((tmp_path / "collection.json").read_bytes())
    expected_identity = {
        "service": service,
        "endpoint_sha256": hashlib.sha256(b"http://127.0.0.1:8000/v1/").hexdigest(),
    }
    if service == "lm_studio":
        expected_identity["response_format"] = "json_schema"
        assert collection["response_format"] == "json_schema"
    else:
        assert "response_format" not in collection
    assert plan["judge"]["service_identity"] == expected_identity
    assert collection["max_calls"] == 2
    assert collection["max_output_tokens"] == 256
    assert collection["max_input_bytes"] == 131072
    assert not (tmp_path / "judge-work").exists()
    assert not (tmp_path / "evidence").exists()


@pytest.mark.parametrize(
    "base_url",
    [
        "http://user:secret@localhost:8000/v1",
        "http://localhost:8000/v1?token=x",
        "file:///v1",
    ],
)
def test_unsafe_endpoint_rejects_before_writes(tmp_path, base_url):
    with pytest.raises(ValueError):
        prepare(tmp_path, base_url=base_url)
    assert not list(tmp_path.iterdir())


def test_explicit_authentication_and_budget_contain_no_credentials(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("INVARLOCK_OPENAI_COMPATIBLE_API_KEY", "not-a-real-credential")
    prepare(tmp_path, authentication="bearer_env", max_input_bytes=200000)
    collection = json.loads((tmp_path / "collection.json").read_bytes())
    assert collection["authentication"] == "bearer_env"
    assert collection["max_input_bytes"] == 200000
    assert all(
        b"not-a-real-credential" not in path.read_bytes() for path in tmp_path.iterdir()
    )


def test_partial_write_is_preserved_and_cannot_be_overwritten(tmp_path, monkeypatch):
    original = Path.open

    def failing(path, *args, **kwargs):
        if path == tmp_path / "plan.json" and args and args[0] == "x":
            raise OSError("storage failed")
        return original(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", failing)
        with pytest.raises(OSError, match="storage failed"):
            prepare(tmp_path)
    retained = (tmp_path / "baseline_run.json").read_bytes()
    with pytest.raises(ValueError, match="must be new"):
        prepare(tmp_path)
    assert (tmp_path / "baseline_run.json").read_bytes() == retained


def test_cli_prepares_new_workspace(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare.py",
            "--workspace",
            str(tmp_path),
            "--baseline",
            str(FIXTURE / "baseline_run.json"),
            "--subject",
            str(FIXTURE / "subject_run.json"),
            "--service",
            "ollama",
            "--base-url",
            "http://127.0.0.1:11434/v1",
            "--model",
            "judge-model",
        ],
    )
    EXAMPLE.main()
    assert json.loads(capsys.readouterr().out)["service"] == "ollama"
