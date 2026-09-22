"""Local-judge setup authenticates synthetic fixture files without inference."""

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest

from invarlock.core.runtime_provider import ModelRuntimeSpec, artifact_identity_sha256
from invarlock.engine import checkpoint_tree_sha256
from invarlock.judge_measurements.contracts import validate_measurement_plan
from invarlock.judge_measurements.workflow import load_judge_request
from invarlock.runtime_providers.gguf_identity import read_gguf_artifact_identity
from tests.runtime_providers.test_gguf_identity import _fixture

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "native_local_judge_example", ROOT / "examples/native-local-judge/prepare.py"
)
EXAMPLE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXAMPLE)


def model_fixture(workspace, provider):
    artifacts = workspace / "artifacts"
    artifacts.mkdir()
    settings = {
        "seed": 0,
        "context_length": 4096,
        "batch_size": 1,
        "max_output_tokens": 128,
        "timeout_seconds": 300,
    }
    if provider == "hf_transformers":
        artifact = artifacts / "judge"
        artifact.mkdir()
        (artifact / "config.json").write_text('{"model_type":"gpt2"}')
        model_id = "local-judge"
        settings.update(
            offline=True,
            checkpoint_tree_sha256=checkpoint_tree_sha256(artifact).removeprefix(
                "sha256:"
            ),
            tokenizer_metadata_sha256="a" * 64,
        )
    else:
        artifact = artifacts / "judge.gguf"
        artifact.write_bytes(_fixture())
        identity = read_gguf_artifact_identity(artifact)
        model_id = identity.artifact_name
        settings.update(
            artifact_sha256=identity.sha256,
            artifact_byte_length=identity.byte_length,
            gguf_metadata_sha256=identity.gguf_metadata_sha256,
            tensor_inventory_sha256=identity.tensor_inventory_sha256,
            tokenizer_metadata_sha256=identity.tokenizer_metadata_sha256,
            backend_binary_sha256="b" * 64,
            backend_source_sha256="c" * 64,
            backend_version="version: 4242 (test) built with TestCompiler for TestOS",
            cpu_threads=4,
            prompt_batch_size=512,
            prompt_microbatch_size=512,
        )
    model = {
        "artifact": {
            "path": artifact.relative_to(workspace).as_posix(),
            "model_id": model_id,
            "locator": "local://judge",
        },
        "runtime": {"provider": provider, "settings": settings},
    }
    path = workspace / "model.json"
    path.write_text(json.dumps(model))
    return path, model, artifact


@pytest.mark.parametrize("provider", ["hf_transformers", "llama_cpp"])
def test_both_profiles_freeze_valid_request_and_complete_identity(tmp_path, provider):
    path, model, artifact = model_fixture(tmp_path, provider)
    result = EXAMPLE.prepare(tmp_path, path)
    resolver = {
        "hf_transformers": EXAMPLE.HFTransformersProvider,
        "llama_cpp": EXAMPLE.LlamaCppProvider,
    }
    request = load_judge_request(
        tmp_path / "request.json", provider_resolver=lambda name: resolver[name]()
    )
    assert request.integration == "runtime-provider-judge"
    assert request.model.runtime.provider == provider
    plan = json.loads((tmp_path / "plan.json").read_bytes())
    validate_measurement_plan(plan)
    spec = ModelRuntimeSpec(
        provider_name=provider,
        model_id=model["artifact"]["model_id"],
        settings=model["runtime"]["settings"],
    )
    identity = resolver[provider]().authenticate_artifact(spec, artifact)
    assert plan["judge"]["model_identity"] == {
        "kind": "local_weights",
        "weights_sha256": artifact_identity_sha256(identity),
    }
    assert result["expected_trials"] == 2
    assert plan["judge"]["config"]["reasoning_effort"] is None
    assert plan["schedule"]["retry_on"] == []
    assert json.loads((tmp_path / "collection.json").read_bytes()) == {
        "profile": "runtime-provider-text-frozen-answer-v1",
        "max_calls": 2,
        "max_output_tokens": 256,
    }
    assert not (tmp_path / "judge-work").exists()
    assert not (tmp_path / "evidence").exists()


@pytest.mark.parametrize("provider", ["hf_transformers", "llama_cpp"])
def test_changed_model_bytes_reject_before_any_generated_files(tmp_path, provider):
    path, _, artifact = model_fixture(tmp_path, provider)
    altered = artifact / "config.json" if artifact.is_dir() else artifact
    raw = bytearray(altered.read_bytes())
    raw[-1] ^= 1
    altered.write_bytes(raw)
    with pytest.raises(ValueError, match="(digest|identity)"):
        EXAMPLE.prepare(tmp_path, path)
    assert {p.name for p in tmp_path.iterdir()} == {"model.json", "artifacts"}


@pytest.mark.parametrize(
    "reference", [".", "../outside", "/absolute", "artifacts//judge"]
)
def test_noncanonical_artifact_reference_rejected(tmp_path, reference):
    path, model, _ = model_fixture(tmp_path, "hf_transformers")
    model["artifact"]["path"] = reference
    path.write_text(json.dumps(model))
    with pytest.raises(ValueError, match="beneath"):
        EXAMPLE.prepare(tmp_path, path)


def test_artifact_symlink_and_unknown_provider_rejected(tmp_path):
    path, model, artifact = model_fixture(tmp_path, "hf_transformers")
    link = tmp_path / "alias"
    link.symlink_to(artifact, target_is_directory=True)
    model["artifact"]["path"] = "alias"
    path.write_text(json.dumps(model))
    with pytest.raises(ValueError, match="symlink"):
        EXAMPLE.prepare(tmp_path, path)
    model["runtime"]["provider"] = "unqualified"
    path.write_text(json.dumps(model))
    with pytest.raises(ValueError, match="choose"):
        EXAMPLE.prepare(tmp_path, path)


def test_existing_and_partial_outputs_are_preserved(tmp_path, monkeypatch):
    path, _, _ = model_fixture(tmp_path, "hf_transformers")
    original = Path.open

    def interrupted(self, *args, **kwargs):
        if self == tmp_path / "plan.json" and args == ("x",):
            raise OSError("interrupted preparation")
        return original(self, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", interrupted)
        with pytest.raises(OSError, match="interrupted"):
            EXAMPLE.prepare(tmp_path, path)
    before = {p.name: p.read_bytes() for p in tmp_path.glob("*.json")}
    assert {"baseline_run.json", "subject_run.json"} <= before.keys()
    assert "request.json" not in before
    with pytest.raises(ValueError, match="must be new"):
        EXAMPLE.prepare(tmp_path, path)
    assert before == {p.name: p.read_bytes() for p in tmp_path.glob("*.json")}


def test_cli_reports_identity_and_refuses_completed_setup(
    tmp_path, monkeypatch, capsys
):
    path, _, _ = model_fixture(tmp_path, "hf_transformers")
    monkeypatch.setattr(
        sys, "argv", ["prepare.py", "--workspace", str(tmp_path), "--model", str(path)]
    )
    EXAMPLE.main()
    result = json.loads(capsys.readouterr().out)
    assert len(result["judge_artifact_identity_sha256"]) == 64
    assert result["provider"] == "hf_transformers"
    original = (tmp_path / "plan.json").read_bytes()
    with pytest.raises(ValueError, match="must be new"):
        EXAMPLE.main()
    assert (tmp_path / "plan.json").read_bytes() == original


def test_edited_cases_and_repetitions_reserve_the_complete_schedule(
    tmp_path, monkeypatch
):
    path, _, _ = model_fixture(tmp_path, "hf_transformers")
    starter = tmp_path / "starter"
    shutil.copytree(EXAMPLE.HERE, starter, ignore=shutil.ignore_patterns("__pycache__"))
    for name in ("baseline_run.json", "subject_run.json"):
        value = json.loads((starter / name).read_bytes())
        value["records"].append({**value["records"][0], "id": "case-2"})
        (starter / name).write_text(json.dumps(value))
    recipe = json.loads((starter / "recipe-template.json").read_bytes())
    recipe["plan"]["sampling"]["case_units"].append(
        {"case_id": "case-2", "unit_id": "unit-2"}
    )
    recipe["plan"]["schedule"]["repetitions"] = 3
    (starter / "recipe-template.json").write_text(json.dumps(recipe))
    monkeypatch.setattr(EXAMPLE, "HERE", starter)
    assert EXAMPLE.prepare(tmp_path, path)["expected_trials"] == 12
    collection = json.loads((tmp_path / "collection.json").read_bytes())
    assert collection["max_calls"] == 12
    assert collection["max_output_tokens"] == 12 * 128
