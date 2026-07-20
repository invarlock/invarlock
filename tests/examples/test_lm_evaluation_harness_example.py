from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import threading
from pathlib import Path
from types import ModuleType

import pytest
import yaml

from invarlock.core.checkpoint_identity import checkpoint_tree_sha256
from invarlock.core.runtime_provider import (
    build_runtime_behavioral_schedule_from_material,
)

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples" / "integrations" / "lm-evaluation-harness" / "example.py"
DOCKERFILE = EXAMPLE.with_name("Dockerfile")
LAUNCHER = EXAMPLE.with_name("launch.py")
MODEL_INPUTS = EXAMPLE.with_name("model_inputs.py")


def _module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("lm_eval_example", EXAMPLE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _launcher_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("lm_eval_launcher", LAUNCHER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _model_inputs_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("lm_eval_model_inputs", MODEL_INPUTS)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _schedule():
    return build_runtime_behavioral_schedule_from_material(
        dataset_identity={
            "provider": "local",
            "dataset_name": "bridge-test",
            "config_name": None,
            "revision": "a" * 40,
            "split": "test",
        },
        records=[
            {
                "record_id": "stable-1",
                "input_text": "Prompt",
                "expected_output": "Answer",
            }
        ],
    )


def _sample(*, record_id: str = "stable-1", prompt: str = "Prompt") -> dict:
    doc = {"expected": "Answer", "id": record_id, "prompt": "Prompt"}
    return {
        "doc_id": 0,
        "doc": doc,
        "target": "Answer",
        "arguments": {
            "gen_args_0": {
                "arg_0": prompt,
                "arg_1": {
                    "do_sample": False,
                    "max_gen_toks": 1,
                    "until": ["\n"],
                },
            }
        },
        "resps": [["Answer"]],
        "filtered_resps": ["Answer"],
        "filter": "none",
        "metrics": ["exact_match"],
        "doc_hash": _digest(
            json.dumps(doc, indent=2, default=str, ensure_ascii=False).encode()
        ),
        "prompt_hash": _digest(prompt.encode()),
        "target_hash": _digest(b"Answer"),
        "exact_match": 1.0,
    }


def _write_jsonl(path: Path, value: object) -> None:
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _write_run(tmp_path: Path, module: ModuleType) -> Path:
    run_dir = tmp_path / "baseline"
    run_dir.mkdir()
    samples = run_dir / "samples.jsonl"
    _write_jsonl(samples, _sample())
    config = module.task_config("/records.jsonl")
    manifest = {
        "format": "invarlock/lm-evaluation-harness-run-v1",
        "role": "baseline",
        "harness_version": module.VERSION,
        "task_config": config,
        "task_config_sha256": module.digest(module.canonical_json_bytes(config)),
        "execution_config": module.execution_config(),
        "execution_config_sha256": module.digest(
            module.canonical_json_bytes(module.execution_config())
        ),
        "samples": "samples.jsonl",
        "samples_sha256": _digest(samples.read_bytes()),
        "model_tree_sha256": "sha256:" + ("a" * 64),
        "dataset_sha256": "b" * 64,
        "record_count": 1,
        "stable_id_field": "id",
    }
    manifest_path = run_dir / "run-manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def _write_full_run(
    prepared: Path, module: ModuleType, role: str, records: list[dict[str, str]]
) -> None:
    run_dir = prepared / "harness" / role
    run_dir.mkdir(parents=True)
    samples = run_dir / "samples.jsonl"
    with samples.open("w", encoding="utf-8") as handle:
        for record in records:
            prompt = record["prompt"]
            target = record["expected"]
            response = target
            doc = {
                "expected": target,
                "id": record["id"],
                "prompt": prompt,
            }
            sample = {
                "doc_id": 0,
                "doc": doc,
                "target": target,
                "arguments": {
                    "gen_args_0": {
                        "arg_0": prompt,
                        "arg_1": module.task_config("/records.jsonl")[
                            "generation_kwargs"
                        ],
                    }
                },
                "resps": [[response]],
                "filtered_resps": [response],
                "filter": "none",
                "metrics": ["exact_match"],
                "doc_hash": _digest(
                    json.dumps(doc, indent=2, default=str, ensure_ascii=False).encode()
                ),
                "prompt_hash": _digest(prompt.encode()),
                "target_hash": _digest(target.encode()),
                "exact_match": 1.0,
            }
            handle.write(json.dumps(sample) + "\n")
    config = module.task_config("/records.jsonl")
    manifest = {
        "format": "invarlock/lm-evaluation-harness-run-v1",
        "role": role,
        "harness_version": module.VERSION,
        "task_config": config,
        "task_config_sha256": module.digest(module.canonical_json_bytes(config)),
        "execution_config": module.execution_config(),
        "execution_config_sha256": module.digest(
            module.canonical_json_bytes(module.execution_config())
        ),
        "samples": "samples.jsonl",
        "samples_sha256": _digest(samples.read_bytes()),
        "model_tree_sha256": module.checkpoint_tree_sha256(
            prepared / f"evaluation/models/{role}"
        ),
        "dataset_sha256": _digest(
            (prepared / "evaluation/inputs/records.jsonl").read_bytes()
        ),
        "record_count": len(records),
        "stable_id_field": "id",
    }
    (run_dir / "run-manifest.json").write_bytes(module.canonical_json_bytes(manifest))


def _prepare_test_transaction(
    prepared: Path, module: ModuleType, image: str
) -> list[dict[str, str]]:
    inputs = prepared / "evaluation/inputs"
    inputs.mkdir(parents=True)
    settings: dict[str, dict[str, object]] = {}
    for role in ("baseline", "subject"):
        checkpoint = prepared / f"evaluation/models/{role}"
        checkpoint.mkdir(parents=True)
        (checkpoint / "config.json").write_text(
            '{"model_type":"gpt2"}\n', encoding="utf-8"
        )
        (checkpoint / "model.safetensors").write_bytes(role.encode("ascii"))
        (checkpoint / "tokenizer.json").write_text(
            '{"model":{"type":"WordLevel","vocab":{"[UNK]":0}}}\n',
            encoding="utf-8",
        )
        settings[role] = {
            "batch_size": module.HARNESS_BATCH_SIZE,
            "checkpoint_tree_sha256": checkpoint_tree_sha256(checkpoint),
            "context_length": 64,
            "max_output_tokens": module.MAX_GENERATION_TOKENS,
            "offline": True,
            "seed": 20_260_716,
            "timeout_seconds": 300,
            "tokenizer_metadata_sha256": "b" * 64,
        }
    records = [
        {
            "id": f"harness-fixture-{index:02d}",
            "prompt": "Prompt",
            "expected": "Answer",
        }
        for index in range(102)
    ]
    records_path = inputs / "records.jsonl"
    records_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    policy = {
        "resolved_policy": {
            "metrics": {
                "exact_match": {
                    "delta_min_pp": -20.0,
                    "maximum_interval_width_pp": 20.0,
                    "minimum_record_count": 102,
                }
            }
        }
    }
    (inputs / "acceptance.json").write_text(json.dumps(policy), encoding="utf-8")

    def side(role: str) -> dict[str, object]:
        return {
            "artifact": {
                "path": f"models/{role}",
                "model_id": f"test/{role}",
                "locator": f"generated://test/{role}",
            },
            "runtime": {"provider": "hf_transformers", "settings": settings[role]},
        }

    request = {
        "format_version": "invarlock/evaluation-request-v1",
        "comparison": {
            "baseline": side("baseline"),
            "subject": side("subject"),
            "dataset": {
                "path": "inputs/records.jsonl",
                "sha256": hashlib.sha256(records_path.read_bytes()).hexdigest(),
                "format": "jsonl",
                "name": module.DATASET_NAME,
                "split": "validation",
                "input_field": "prompt",
                "expected_output_field": "expected",
                "id_field": "id",
            },
            "policy": "inputs/acceptance.json",
            "task": "text_causal",
            "metric": "exact_match",
        },
        "execution": {"mode": "run"},
        "output": {"evidence": "evidence"},
    }
    (prepared / "evaluation/request.yaml").write_text(
        yaml.safe_dump(request, sort_keys=False), encoding="utf-8"
    )
    (prepared / "runtime-image-id.txt").write_text(image + "\n", encoding="ascii")
    return records


def test_adapter_writes_schedule_bound_records(tmp_path: Path) -> None:
    module = _module()
    samples = tmp_path / "samples.jsonl"
    destination = tmp_path / "records.jsonl"
    _write_jsonl(samples, _sample())

    module.adapt(samples, _schedule(), destination)

    record = json.loads(destination.read_text(encoding="utf-8"))
    assert record["record_id"] == "stable-1"
    assert record["output_text"] == "Answer"
    assert "exact_match" not in record


def test_worker_runs_the_pinned_harness_and_binds_official_samples(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}\n", encoding="utf-8")
    dataset = tmp_path / "records.jsonl"
    dataset.write_text("{}\n", encoding="utf-8")
    output = tmp_path / "output"
    commands: list[list[str]] = []
    monkeypatch.setattr(
        module.importlib.metadata, "version", lambda _name: module.VERSION
    )

    def run(command: list[str], *, check: bool, text: bool) -> object:
        assert check is False and text is True
        commands.append(command)
        raw = Path(command[command.index("--output_path") + 1])
        raw.mkdir(parents=True)
        (raw / "samples_task.jsonl").write_text('{"sample":1}\n', encoding="utf-8")
        return type("Result", (), {"returncode": 0})()

    monkeypatch.setattr(module.subprocess, "run", run)
    module.worker("baseline", model, dataset, output)

    command = commands[0]
    assert command[:4] == [sys.executable, "-m", "lm_eval", "run"]
    assert command[command.index("--device") + 1] == "cpu"
    assert command[command.index("--batch_size") + 1] == "8"
    assert command[command.index("--seed") + 1] == "20260716"
    assert command[command.index("--model") + 1] == "hf"
    assert "backend=causal" in command[command.index("--model_args") + 1]
    assert "dtype=float32" in command[command.index("--model_args") + 1]
    assert "trust_remote_code=False" in command[command.index("--model_args") + 1]
    assert "--log_samples" in command
    manifest = json.loads((output / "run-manifest.json").read_text(encoding="utf-8"))
    assert manifest["harness_version"] == module.VERSION
    assert manifest["task_config"] == module.task_config(str(dataset))
    assert manifest["task_config_sha256"] == module.digest(
        module.canonical_json_bytes(module.task_config(str(dataset)))
    )
    assert manifest["model_tree_sha256"] == module.checkpoint_tree_sha256(model)
    assert manifest["dataset_sha256"] == _digest(dataset.read_bytes())
    assert manifest["samples_sha256"] == _digest(
        (output / "samples.jsonl").read_bytes()
    )
    assert manifest["execution_config"] == module.execution_config()
    assert manifest["execution_config_sha256"] == module.digest(
        module.canonical_json_bytes(module.execution_config())
    )

    monkeypatch.setattr(module.importlib.metadata, "version", lambda _name: "0")
    with pytest.raises(module.BridgeError, match="must contain"):
        module.worker("baseline", model, dataset, tmp_path / "wrong-version")
    monkeypatch.setattr(
        module.importlib.metadata, "version", lambda _name: module.VERSION
    )
    with pytest.raises(module.BridgeError, match="inputs must exist"):
        module.worker("baseline", model, dataset, output)

    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *_args, **_kwargs: type("Result", (), {"returncode": 3})(),
    )
    with pytest.raises(module.BridgeError, match="execution failed"):
        module.worker("baseline", model, dataset, tmp_path / "failed-run")

    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *_args, **_kwargs: type("Result", (), {"returncode": 0})(),
    )
    with pytest.raises(module.BridgeError, match="one per-record file"):
        module.worker("baseline", model, dataset, tmp_path / "missing-samples")


@pytest.mark.parametrize("target", ["model", "dataset", "task"])
def test_worker_rejects_inputs_changed_during_harness_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, target: str
) -> None:
    module = _module()
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}\n", encoding="utf-8")
    dataset = tmp_path / "records.jsonl"
    dataset.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        module.importlib.metadata, "version", lambda _name: module.VERSION
    )

    def run(command: list[str], *, check: bool, text: bool) -> object:
        assert check is False and text is True
        if target == "model":
            (model / "config.json").write_text('{"changed":true}\n', encoding="utf-8")
        elif target == "dataset":
            dataset.write_text('{"changed":true}\n', encoding="utf-8")
        else:
            Path(command[command.index("--tasks") + 1]).write_text(
                "task: changed\n", encoding="utf-8"
            )
        return type("Result", (), {"returncode": 0})()

    monkeypatch.setattr(module.subprocess, "run", run)
    with pytest.raises(module.BridgeError, match="inputs changed during execution"):
        module.worker("baseline", model, dataset, tmp_path / "output")


def test_worker_rejects_checkpoint_generation_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}\n", encoding="utf-8")
    (model / "generation_config.json").write_text(
        '{"max_new_tokens":2048}\n', encoding="utf-8"
    )
    dataset = tmp_path / "records.jsonl"
    dataset.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        module.importlib.metadata, "version", lambda _name: module.VERSION
    )

    with pytest.raises(module.BridgeError, match="generation defaults"):
        module.worker("baseline", model, dataset, tmp_path / "output")


def test_adapter_rejects_aggregate_only_results(tmp_path: Path) -> None:
    module = _module()
    samples = tmp_path / "samples.jsonl"
    _write_jsonl(samples, {"results": {"exact_match": 1.0}})

    with pytest.raises(module.BridgeError, match="aggregate-only"):
        module.adapt(samples, _schedule(), tmp_path / "records.jsonl")


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("wrong-count", "one Harness sample"),
        ("not-object", "is not an object"),
        ("missing-facts", "lacks per-record facts"),
        ("wrong-prompt", "prompt does not match"),
        ("missing-response", "lacks one model response"),
    ],
)
def test_adapter_rejects_incomplete_per_record_output(
    tmp_path: Path, case: str, message: str
) -> None:
    module = _module()
    samples = tmp_path / "samples.jsonl"
    if case == "wrong-count":
        samples.write_bytes(b"")
    elif case == "not-object":
        _write_jsonl(samples, "not-an-object")
    elif case == "missing-facts":
        _write_jsonl(samples, {})
    else:
        sample = _sample(prompt="different" if case == "wrong-prompt" else "Prompt")
        if case == "missing-response":
            sample["filtered_resps"] = []
        _write_jsonl(samples, sample)

    with pytest.raises(module.BridgeError, match=message):
        module.adapt(samples, _schedule(), tmp_path / "records.jsonl")


def test_adapter_rejects_unstable_ids(tmp_path: Path) -> None:
    module = _module()
    samples = tmp_path / "samples.jsonl"
    _write_jsonl(samples, _sample(record_id="0"))

    with pytest.raises(module.BridgeError, match="stable, ordered ID"):
        module.adapt(samples, _schedule(), tmp_path / "records.jsonl")


def test_adapter_rejects_authenticated_input_tampering(tmp_path: Path) -> None:
    module = _module()
    samples = tmp_path / "samples.jsonl"
    value = _sample()
    value["target"] = "changed"
    _write_jsonl(samples, value)

    with pytest.raises(module.BridgeError, match="authenticated inputs"):
        module.adapt(samples, _schedule(), tmp_path / "records.jsonl")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_gen_toks", 2),
        ("do_sample", True),
        ("until", []),
    ],
)
def test_adapter_rejects_generation_setting_drift(
    tmp_path: Path, field: str, value: object
) -> None:
    module = _module()
    samples = tmp_path / "samples.jsonl"
    sample = _sample()
    sample["arguments"]["gen_args_0"]["arg_1"][field] = value
    _write_jsonl(samples, sample)

    with pytest.raises(module.BridgeError, match="generation settings"):
        module.adapt(samples, _schedule(), tmp_path / "records.jsonl")


def test_run_manifest_rejects_output_tampering(tmp_path: Path) -> None:
    module = _module()
    manifest = _write_run(tmp_path, module)
    with (manifest.parent / "samples.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("{}\n")

    with pytest.raises(module.BridgeError, match="tampered"):
        module.load_run(manifest, "baseline")


def test_run_manifest_requires_provenance(tmp_path: Path) -> None:
    module = _module()

    with pytest.raises(module.BridgeError, match="provenance is missing"):
        module.load_run(tmp_path / "missing.json", "baseline")


def test_run_manifest_rejects_incomplete_provenance(tmp_path: Path) -> None:
    module = _module()
    manifest = _write_run(tmp_path, module)
    value = json.loads(manifest.read_text(encoding="utf-8"))
    del value["execution_config"]
    manifest.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(module.BridgeError, match="provenance is incomplete"):
        module.load_run(manifest, "baseline")


def test_run_manifest_requires_the_expected_task_configuration(tmp_path: Path) -> None:
    module = _module()
    manifest = _write_run(tmp_path, module)
    value = json.loads(manifest.read_text(encoding="utf-8"))
    value["task_config"]["generation_kwargs"]["do_sample"] = True
    value["task_config_sha256"] = module.digest(
        module.canonical_json_bytes(value["task_config"])
    )
    manifest.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(module.BridgeError, match="provenance is invalid"):
        module.load_run(manifest, "baseline")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("batch_size", 1),
        ("device", "cuda"),
        ("dtype", "float16"),
        ("max_generation_tokens", 2),
        ("seed", 0),
        ("checkpoint_generation_config", "included"),
        ("harness_backend", "other"),
        ("harness_model", "other"),
        ("trust_remote_code", True),
    ],
)
def test_run_manifest_rejects_execution_profile_drift(
    tmp_path: Path, field: str, value: object
) -> None:
    module = _module()
    manifest = _write_run(tmp_path, module)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["execution_config"][field] = value
    payload["execution_config_sha256"] = module.digest(
        module.canonical_json_bytes(payload["execution_config"])
    )
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(module.BridgeError, match="provenance is invalid"):
        module.load_run(manifest, "baseline")


def test_complete_requires_immutable_runtime_identity(tmp_path: Path) -> None:
    module = _module()

    with pytest.raises(module.BridgeError, match="immutable local sha256"):
        module.complete(tmp_path / "transaction", tmp_path / "prepared", "latest")


def test_complete_rejects_existing_workspace_and_mismatched_harness_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    image = "sha256:" + ("d" * 64)
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(module.BridgeError, match="workspace must be new"):
        module.complete(existing, tmp_path / "prepared", image)

    monkeypatch.setattr(
        module,
        "load_run",
        lambda _path, role: (
            {
                "execution_config_sha256": "shared-execution",
                "task_config_sha256": f"different-{role}",
            },
            tmp_path / f"{role}-samples.jsonl",
        ),
    )
    with pytest.raises(module.BridgeError, match="different Harness configurations"):
        module.complete(tmp_path / "new-transaction", tmp_path / "prepared", image)


def test_bridge_main_dispatches_worker_complete_and_reports_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _module()
    observed: list[str] = []
    monkeypatch.setattr(
        module,
        "worker",
        lambda *_args: observed.append("worker"),
    )
    assert (
        module.main(
            [
                "worker",
                "--role",
                "baseline",
                "--model",
                str(tmp_path / "model"),
                "--dataset",
                str(tmp_path / "records.jsonl"),
                "--output",
                str(tmp_path / "output"),
            ]
        )
        == 0
    )
    monkeypatch.setattr(
        module,
        "complete",
        lambda *_args: (
            tmp_path / "evidence",
            tmp_path / "receipt.json",
            tmp_path / "report.html",
        ),
    )
    assert (
        module.main(
            [
                "complete",
                "--workspace",
                str(tmp_path / "transaction"),
                "--prepared",
                str(tmp_path / "prepared"),
                "--runtime-image",
                "sha256:" + "a" * 64,
            ]
        )
        == 0
    )
    assert observed == ["worker"]
    assert "Evidence:" in capsys.readouterr().out

    monkeypatch.setattr(
        module,
        "worker",
        lambda *_args: (_ for _ in ()).throw(module.BridgeError("bad run")),
    )
    assert (
        module.main(
            [
                "worker",
                "--role",
                "subject",
                "--model",
                str(tmp_path / "model"),
                "--dataset",
                str(tmp_path / "records.jsonl"),
                "--output",
                str(tmp_path / "output"),
            ]
        )
        == 2
    )
    assert "FAIL bad run" in capsys.readouterr().err


def test_complete_replays_real_import_transaction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()

    def missing_distribution_version(package: str) -> str:
        raise module.importlib.metadata.PackageNotFoundError(package)

    monkeypatch.setattr(
        module.importlib.metadata,
        "version",
        missing_distribution_version,
    )
    prepared = tmp_path / "prepared"
    image = "sha256:" + ("d" * 64)
    records = _prepare_test_transaction(prepared, module, image)
    for role in ("baseline", "subject"):
        _write_full_run(prepared, module, role, records)

    evidence, receipt, report = module.complete(
        tmp_path / "transaction", prepared, image
    )

    assert evidence.is_dir()
    statement = json.loads(receipt.read_text(encoding="utf-8"))["statement"]
    assert statement["verdict"]["ok"] is True
    assert statement["verdict"]["policy_verdict"] == "pass"
    assert report.is_file()
    observation = json.loads(
        (evidence / "observations/lm-evaluation-harness-provenance.json").read_text(
            encoding="utf-8"
        )
    )
    provenance = observation["payload"]
    assert provenance["execution_config"] == module.execution_config()
    assert provenance["execution_config_sha256"] == module.digest(
        module.canonical_json_bytes(module.execution_config())
    )
    assert provenance["task_config"]["generation_kwargs"] == {
        "do_sample": False,
        "max_gen_toks": module.MAX_GENERATION_TOKENS,
        "until": ["\n"],
    }
    for role in ("baseline", "subject"):
        provider_receipt = json.loads(
            (evidence / f"providers/{role}/runtime-provider.receipt.json").read_text(
                encoding="utf-8"
            )
        )
        assert provider_receipt["execution_settings"] == {
            "allow_network": False,
            "batch_size": module.HARNESS_BATCH_SIZE,
            "context_length": 64,
            "max_output_tokens": module.MAX_GENERATION_TOKENS,
            "seed": module.HARNESS_SEED,
            "timeout_seconds": 300,
        }
        assert provider_receipt["device"]["device_kind"] == "cpu"
        assert provider_receipt["plugin"] == {
            "distribution": "invarlock",
            "distribution_version": module.INVARLOCK_VERSION,
            "name": "hf_transformers",
            "provider_abi": "1",
        }
        assert provider_receipt["backend"]["source_sha256"] == module.digest(
            module.canonical_json_bytes(provenance)
        )
        assert (
            provider_receipt["backend"]["build_sha256"]
            == provenance["task_config_sha256"]
        )


@pytest.mark.parametrize("tamper", ["dataset", "policy"])
def test_complete_rejects_tampered_prepared_acceptance_inputs(
    tmp_path: Path, tamper: str
) -> None:
    module = _module()
    prepared = tmp_path / "prepared"
    image = "sha256:" + ("d" * 64)
    records = _prepare_test_transaction(prepared, module, image)
    records_path = prepared / "evaluation/inputs/records.jsonl"
    for role in ("baseline", "subject"):
        _write_full_run(prepared, module, role, records)

    if tamper == "dataset":
        with records_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(records[0]) + "\n")
        error = "dataset does not match"
    else:
        policy_path = prepared / "evaluation/inputs/acceptance.json"
        policy = json.loads(policy_path.read_text(encoding="utf-8"))
        policy["resolved_policy"]["metrics"]["exact_match"]["delta_min_pp"] = -1.0
        policy_path.write_text(json.dumps(policy), encoding="utf-8")
        error = "policy is not the fixed"

    with pytest.raises(module.BridgeError, match=error):
        module.complete(tmp_path / f"transaction-{tamper}", prepared, image)


@pytest.mark.parametrize("tamper", ["checkpoint", "policy-pointer", "dataset-name"])
def test_complete_rejects_changed_prepared_identity(
    tmp_path: Path, tamper: str
) -> None:
    module = _module()
    prepared = tmp_path / "prepared"
    image = "sha256:" + ("d" * 64)
    records = _prepare_test_transaction(prepared, module, image)
    for role in ("baseline", "subject"):
        _write_full_run(prepared, module, role, records)

    if tamper == "checkpoint":
        checkpoint = prepared / "evaluation/models/subject/config.json"
        checkpoint.write_text(checkpoint.read_text() + "\n", encoding="utf-8")
        error = "checkpoint.*does not match"
    else:
        request_path = prepared / "evaluation/request.yaml"
        request = yaml.safe_load(request_path.read_text(encoding="utf-8"))
        if tamper == "policy-pointer":
            request["comparison"]["policy"] = "inputs/other.json"
            error = "policy path"
        else:
            request["comparison"]["dataset"]["name"] = "other"
            error = "dataset descriptor"
        request_path.write_text(yaml.safe_dump(request), encoding="utf-8")

    with pytest.raises((module.BridgeError, ValueError), match=error):
        module.complete(tmp_path / f"transaction-{tamper}", prepared, image)


def test_launcher_runs_workers_in_restricted_inspected_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _launcher_module()
    from examples.integrations import launch as shared_launch

    base_id = "sha256:" + ("a" * 64)
    final_id = "sha256:" + ("b" * 64)
    commands: list[list[str]] = []
    monkeypatch.setattr(
        shared_launch, "_require_committed_checkout", lambda _root: "c" * 40
    )
    monkeypatch.setattr(
        shared_launch,
        "_runtime_image",
        lambda **_kwargs: (base_id, base_id),
    )

    def fake_run(
        command: list[str], *, cwd: Path, stdin_path: Path | None = None
    ) -> str:
        commands.append(command)
        if command[:3] == ["docker", "image", "inspect"]:
            if "org.invarlock.example.base-image-id" in " ".join(command):
                return base_id
            return (
                base_id
                if command[-1].startswith("invarlock-example-runtime:")
                else final_id
            )
        if "model_inputs.py" in " ".join(command):
            prepared = tmp_path / "journey/prepared"
            for role in ("baseline", "subject"):
                (prepared / f"evaluation/models/{role}").mkdir(parents=True)
            (prepared / "evaluation/inputs").mkdir(parents=True)
            (prepared / "evaluation/inputs/records.jsonl").write_text("{}\n")
        if "example.py" in " ".join(command):
            return "transaction complete"
        if command[:2] == ["docker", "build"]:
            assert stdin_path == tmp_path / "journey/build/harness-source.tar"
            assert command[-1] == "-"
        return ""

    monkeypatch.setattr(module, "run", fake_run)
    assert module.main(["--workspace", str(tmp_path / "journey")]) == 0
    workers = [
        command for command in commands if command[:3] == ["docker", "run", "--rm"]
    ]
    assert len(workers) == 2
    for command in workers:
        assert command[command.index("--network") + 1] == "none"
        assert "--pull=never" in command
        assert "--read-only" in command
        assert "--cap-drop=ALL" in command
        assert "no-new-privileges" in command
        assert command[command.index("--user") + 1]
        assert command[command.index("--tmpfs") + 1].startswith("/tmp:rw,noexec")
        assert command[command.index("--mount") + 1].endswith("dst=/model,readonly")
        assert final_id in command
    build = next(command for command in commands if command[:2] == ["docker", "build"])
    assert "BASE_IMAGE=invarlock-example-runtime:cccccccccccc" in build
    assert f"BASE_IMAGE_ID={base_id}" in build
    assert f"BASE_IMAGE={base_id}" not in build
    base_inspections = [
        command
        for command in commands
        if command[:3] == ["docker", "image", "inspect"]
        and command[-1] == "invarlock-example-runtime:cccccccccccc"
    ]
    assert len(base_inspections) == 2
    assert any(
        "org.invarlock.example.base-image-id" in " ".join(command)
        for command in commands
    )
    preparation = next(
        command for command in commands if "model_inputs.py" in " ".join(command)
    )
    assert preparation[preparation.index("--runtime-image") + 1] == final_id


def test_model_inputs_pin_public_qwen3_snapshots_and_closed_records() -> None:
    module = _model_inputs_module()

    assert [
        (snapshot.role, snapshot.repository, snapshot.revision)
        for snapshot in module.SNAPSHOTS
    ] == [
        (
            "baseline",
            "Qwen/Qwen3-0.6B-Base",
            "da87bfb608c14b7cf20ba1ce41287e8de496c0cd",
        ),
        (
            "subject",
            "Qwen/Qwen3-0.6B",
            "c1899de289a04d12100db370d81485cdf75e47ca",
        ),
    ]
    assert all(
        len(item.sha256) == 64 and item.byte_length > 0
        for snapshot in module.SNAPSHOTS
        for item in snapshot.files
    )
    assert all(
        item.name != "generation_config.json"
        for snapshot in module.SNAPSHOTS
        for item in snapshot.files
    )
    records = module._records()
    assert len(records) == 102
    assert len({record["id"] for record in records}) == 102
    assert all(record["id"].startswith("causal-cloze-") for record in records)


def test_model_input_download_accepts_only_the_pinned_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _model_inputs_module()
    payload = b"authenticated snapshot file"
    item = module.SnapshotFile(
        "model.bin", len(payload), hashlib.sha256(payload).hexdigest()
    )
    snapshot = module.Snapshot("baseline", "Qwen/example", "a" * 40, (item,))

    class Response:
        def __init__(self) -> None:
            self.payload = payload

        def __enter__(self) -> Response:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self, _size: int) -> bytes:
            value, self.payload = self.payload, b""
            return value

    monkeypatch.setattr(
        module.urllib.request, "urlopen", lambda *_args, **_kwargs: Response()
    )
    destination = tmp_path / item.name
    module._download(destination, snapshot, item)
    assert destination.read_bytes() == payload

    bad = module.SnapshotFile("bad.bin", len(payload), "0" * 64)
    monkeypatch.setattr(
        module.urllib.request, "urlopen", lambda *_args, **_kwargs: Response()
    )
    with pytest.raises(RuntimeError, match="is not pinned"):
        module._download(tmp_path / bad.name, snapshot, bad)
    assert not (tmp_path / "bad.bin.partial").exists()


@pytest.mark.parametrize(
    ("chunk", "byte_length", "message"),
    [
        ("not-bytes", 9, "did not return bytes"),
        (b"too large", 1, "exceeds its pinned size"),
    ],
)
def test_model_input_download_rejects_invalid_streams_and_removes_partial_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    chunk: object,
    byte_length: int,
    message: str,
) -> None:
    module = _model_inputs_module()
    item = module.SnapshotFile("model.bin", byte_length, "0" * 64)
    snapshot = module.Snapshot("baseline", "Qwen/example", "a" * 40, (item,))

    class Response:
        def __init__(self) -> None:
            self.chunk = chunk

        def __enter__(self) -> Response:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self, _size: int) -> object:
            value, self.chunk = self.chunk, b""
            return value

    monkeypatch.setattr(
        module.urllib.request, "urlopen", lambda *_args, **_kwargs: Response()
    )
    destination = tmp_path / item.name

    with pytest.raises(RuntimeError, match=message):
        module._download(destination, snapshot, item)

    assert not destination.exists()
    assert not destination.with_suffix(".bin.partial").exists()


def test_model_input_snapshot_staging_validates_qwen3_and_cleans_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _model_inputs_module()
    item = module.SnapshotFile("config.json", 1, "0" * 64)

    def stage(payload: str) -> tuple[Path, object]:
        snapshot = module.Snapshot("baseline", "Qwen/example", "a" * 40, (item,))
        root = tmp_path / ("valid" if "qwen3" in payload else "invalid")
        root.mkdir()

        def download(destination: Path, *_args: object) -> None:
            destination.write_text(payload, encoding="utf-8")

        monkeypatch.setattr(module, "_download", download)
        return root, snapshot

    root, snapshot = stage('{"model_type":"qwen3"}')
    destination = module.stage_snapshot(root, snapshot)
    assert destination == root / "baseline"
    assert (
        json.loads((destination / "config.json").read_text())["model_type"] == "qwen3"
    )

    root, snapshot = stage('{"model_type":"other"}')
    with pytest.raises(RuntimeError, match="is not a Qwen3 checkpoint"):
        module.stage_snapshot(root, snapshot)
    assert not (root / "baseline").exists()


def test_model_input_snapshot_pair_is_staged_concurrently(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _model_inputs_module()
    snapshots = (
        module.Snapshot("baseline", "Qwen/base", "a" * 40, ()),
        module.Snapshot("subject", "Qwen/subject", "b" * 40, ()),
    )
    observed: list[str] = []
    rendezvous = threading.Barrier(2, timeout=5)

    def stage(root: Path, snapshot: object) -> Path:
        observed.append(snapshot.role)
        rendezvous.wait()
        return root / snapshot.role

    monkeypatch.setattr(module, "SNAPSHOTS", snapshots)
    monkeypatch.setattr(module, "stage_snapshot", stage)

    staged = module.stage_snapshots(tmp_path / "models")

    assert staged == {
        "baseline": tmp_path / "models/baseline",
        "subject": tmp_path / "models/subject",
    }
    assert sorted(observed) == ["baseline", "subject"]


@pytest.mark.parametrize(
    "records",
    [
        [],
        ["not-an-object"] * 102,
        [{"id": "same", "prompt": "Prompt", "expected": "Answer"}] * 102,
    ],
)
def test_model_input_records_reject_bad_count_shape_and_duplicate_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    records: list[object],
) -> None:
    module = _model_inputs_module()
    records_path = tmp_path / "records.json"
    records_path.write_text(json.dumps(records), encoding="utf-8")
    monkeypatch.setattr(module, "RECORDS", records_path)

    with pytest.raises(RuntimeError, match="records|record IDs"):
        module._records()


def test_model_input_authoring_binds_qwen3_ids_and_fixed_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _model_inputs_module()
    root = tmp_path / "prepared"
    model_root = tmp_path / "models"
    models: dict[str, Path] = {}
    for snapshot in module.SNAPSHOTS:
        path = model_root / snapshot.role
        path.mkdir(parents=True)
        (path / "config.json").write_text('{"model_type":"qwen3"}\n')
        models[snapshot.role] = path

    monkeypatch.setattr(module, "stage_snapshots", lambda _root: models)

    class BoundTokenizer:
        last_text = ""

        def __call__(self, text: str, **_kwargs: object) -> dict[str, list[int]]:
            self.last_text = text
            return {"input_ids": list(range(min(len(text.split()), 4)))}

        def decode(self, _ids: list[int], **_kwargs: object) -> str:
            return self.last_text

    monkeypatch.setattr(
        module.AutoTokenizer,
        "from_pretrained",
        lambda *_args, **_kwargs: BoundTokenizer(),
    )
    monkeypatch.setattr(module, "checkpoint_tree_sha256", lambda _path: "a" * 64)
    monkeypatch.setattr(
        module, "hf_tokenizer_contract_sha256", lambda _tokenizer: "b" * 64
    )
    image = "sha256:" + "c" * 64
    module.prepare(root, image)

    request = yaml.safe_load(
        (root / "evaluation/request.yaml").read_text(encoding="utf-8")
    )
    comparison = request["comparison"]
    assert comparison["dataset"]["name"] == module.DATASET_NAME
    assert comparison["baseline"]["artifact"]["locator"].startswith(
        "hf://Qwen/Qwen3-0.6B-Base@"
    )
    assert comparison["subject"]["artifact"]["locator"].startswith(
        "hf://Qwen/Qwen3-0.6B@"
    )
    assert comparison["baseline"]["runtime"]["settings"]["batch_size"] == 8
    assert comparison["subject"]["runtime"]["settings"]["batch_size"] == 8
    assert (
        comparison["baseline"]["runtime"]["settings"]["max_output_tokens"]
        == module.MAX_GENERATION_TOKENS
    )
    policy = json.loads(
        (root / "evaluation/inputs/acceptance.json").read_text(encoding="utf-8")
    )
    assert policy["resolved_policy"]["metrics"]["exact_match"] == {
        "delta_min_pp": -20.0,
        "maximum_interval_width_pp": 20.0,
        "minimum_record_count": 102,
    }


def test_model_input_authoring_rejects_existing_workspace_and_unsafe_targets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _model_inputs_module()
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(RuntimeError, match="workspace must be new"):
        module.prepare(existing, "sha256:" + "a" * 64)

    model_root = tmp_path / "models"
    models: dict[str, Path] = {}
    for snapshot in module.SNAPSHOTS:
        checkpoint = model_root / snapshot.role
        checkpoint.mkdir(parents=True)
        (checkpoint / "config.json").write_text(
            '{"model_type":"qwen3"}\n', encoding="utf-8"
        )
        models[snapshot.role] = checkpoint
    monkeypatch.setattr(module, "stage_snapshots", lambda _root: models)

    class OversizeTokenizer:
        def __call__(self, _text: str, **_kwargs: object) -> dict[str, list[int]]:
            return {"input_ids": [1, 2, 3, 4, 5]}

        def decode(self, _ids: list[int], **_kwargs: object) -> str:
            return "different"

    monkeypatch.setattr(
        module.AutoTokenizer,
        "from_pretrained",
        lambda *_args, **_kwargs: OversizeTokenizer(),
    )
    with pytest.raises(RuntimeError, match="lossless generation bound"):
        module.prepare(tmp_path / "oversize", "sha256:" + "b" * 64)


def test_model_input_main_resolves_workspace_and_dispatches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _model_inputs_module()
    workspace = tmp_path / "parent/../prepared"
    image = "sha256:" + "c" * 64
    observed: list[tuple[Path, str]] = []
    monkeypatch.setattr(
        module.argparse.ArgumentParser,
        "parse_args",
        lambda _parser: module.argparse.Namespace(
            workspace=workspace, runtime_image=image
        ),
    )
    monkeypatch.setattr(
        module,
        "prepare",
        lambda root, runtime_image: observed.append((root, runtime_image)),
    )

    assert module.main() == 0
    assert observed == [(workspace.expanduser().resolve(), image)]


def test_completed_outputs_require_passing_report_and_receipt(tmp_path: Path) -> None:
    module = _module()
    evidence = tmp_path / "evidence"
    reports = evidence / "reports"
    reports.mkdir(parents=True)
    receipt = tmp_path / "receipt.json"
    rendered = tmp_path / "report.html"
    rendered.write_text("<html></html>", encoding="utf-8")
    (reports / "evaluation.report.json").write_text(
        json.dumps(
            {
                "baseline": {"mean_score": 0.8},
                "comparison": {"value": 10.0},
                "metric": "exact_match",
                "subject": {"mean_score": 0.9},
                "verdict": "pass",
            }
        ),
        encoding="utf-8",
    )
    receipt.write_text(
        json.dumps(
            {
                "signature": {},
                "statement": {
                    "verdict": {
                        "integrity_ok": True,
                        "ok": True,
                        "policy_verdict": "pass",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    module.validate_completed_outputs(evidence, receipt, rendered)

    value = json.loads(receipt.read_text(encoding="utf-8"))
    value["statement"]["verdict"]["policy_verdict"] = "fail"
    receipt.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(module.BridgeError, match="did not verify a passing"):
        module.validate_completed_outputs(evidence, receipt, rendered)

    value["statement"]["verdict"]["policy_verdict"] = "pass"
    receipt.write_text(json.dumps(value), encoding="utf-8")
    evaluation = json.loads(
        (reports / "evaluation.report.json").read_text(encoding="utf-8")
    )
    evaluation["baseline"]["mean_score"] = 0.19
    (reports / "evaluation.report.json").write_text(
        json.dumps(evaluation), encoding="utf-8"
    )
    with pytest.raises(module.BridgeError, match="did not verify a passing"):
        module.validate_completed_outputs(evidence, receipt, rendered)

    evaluation["baseline"]["mean_score"] = 0.8
    evaluation["comparison"]["value"] = True
    (reports / "evaluation.report.json").write_text(
        json.dumps(evaluation), encoding="utf-8"
    )
    with pytest.raises(module.BridgeError, match="did not verify a passing"):
        module.validate_completed_outputs(evidence, receipt, rendered)

    receipt.unlink()
    with pytest.raises(module.BridgeError, match="missing verified outputs"):
        module.validate_completed_outputs(evidence, receipt, rendered)


def test_completed_outputs_reject_non_object_json(tmp_path: Path) -> None:
    module = _module()
    evidence = tmp_path / "evidence"
    reports = evidence / "reports"
    reports.mkdir(parents=True)
    (reports / "evaluation.report.json").write_text("[]", encoding="utf-8")
    receipt = tmp_path / "receipt.json"
    receipt.write_text("{}", encoding="utf-8")
    rendered = tmp_path / "report.html"
    rendered.write_text("<html></html>", encoding="utf-8")

    with pytest.raises(module.BridgeError, match="returned invalid outputs"):
        module.validate_completed_outputs(evidence, receipt, rendered)


def test_launcher_rejects_unsafe_mount_paths() -> None:
    module = _launcher_module()

    with pytest.raises(ValueError, match="OCI mount"):
        module.mount_source(Path("bad,path"))


def test_launcher_command_runner_and_existing_workspace_fail_closed(
    tmp_path: Path,
) -> None:
    module = _launcher_module()

    assert module.run([sys.executable, "-c", "print('ok')"], cwd=tmp_path) == "ok"
    input_file = tmp_path / "stdin.bin"
    input_file.write_bytes(b"exact source bytes")
    assert (
        module.run(
            [
                sys.executable,
                "-c",
                "import sys; print(len(sys.stdin.buffer.read()))",
            ],
            cwd=tmp_path,
            stdin_path=input_file,
        )
        == "18"
    )
    with pytest.raises(RuntimeError, match="failed"):
        module.run(
            [sys.executable, "-c", "import sys; print('failed'); sys.exit(3)"],
            cwd=tmp_path,
        )
    existing = tmp_path / "existing"
    existing.mkdir()
    assert module.main(["--workspace", str(existing)]) == 2


def test_launcher_canonicalizes_default_workspace_before_source_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _launcher_module()
    from examples.integrations import launch as shared_launch

    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    observed: list[Path] = []
    monkeypatch.setattr(module.tempfile, "mkdtemp", lambda **_kwargs: str(alias))
    monkeypatch.setattr(
        shared_launch, "_require_committed_checkout", lambda _root: "c" * 40
    )

    def stop(**arguments: object) -> tuple[str, str]:
        observed.append(Path(arguments["build_root"]))
        raise RuntimeError("stop after canonical workspace check")

    monkeypatch.setattr(shared_launch, "_runtime_image", stop)
    assert module.main([]) == 2
    assert observed == [real.resolve() / "build"]


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        ("after-build", "changed after source-bound build"),
        ("during-build", "changed during Harness image build"),
        ("mutable-image", "did not return an immutable ID"),
        ("wrong-base-label", "does not bind the inspected base image ID"),
    ],
)
def test_launcher_rejects_image_identity_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
    message: str,
) -> None:
    module = _launcher_module()
    from examples.integrations import launch as shared_launch

    base_id = "sha256:" + "a" * 64
    final_id = "sha256:" + "b" * 64
    monkeypatch.setattr(
        shared_launch, "_require_committed_checkout", lambda _root: "c" * 40
    )
    monkeypatch.setattr(
        shared_launch, "_runtime_image", lambda **_kwargs: (base_id, base_id)
    )
    base_inspections = 0

    def run(command: list[str], *, cwd: Path, stdin_path: Path | None = None) -> str:
        nonlocal base_inspections
        if command[:3] != ["docker", "image", "inspect"]:
            return ""
        if "org.invarlock.example.base-image-id" in " ".join(command):
            return "sha256:" + "d" * 64 if failure == "wrong-base-label" else base_id
        if command[-1].startswith("invarlock-example-runtime:"):
            base_inspections += 1
            if failure == "after-build" and base_inspections == 1:
                return "sha256:" + "d" * 64
            if failure == "during-build" and base_inspections == 2:
                return "sha256:" + "d" * 64
            return base_id
        if failure == "mutable-image":
            return "mutable-tag"
        return final_id

    monkeypatch.setattr(module, "run", run)
    assert module.main(["--workspace", str(tmp_path / failure)]) == 2


def test_container_recipe_pins_the_harness_and_real_worker() -> None:
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")

    lock = ROOT / "requirements/workflows/lm-evaluation-harness-py312.txt"

    assert "lm-evaluation-harness-py312.txt" in dockerfile
    assert "--require-hashes" in dockerfile
    assert "download.pytorch.org/whl/cu128" not in dockerfile
    assert 'pip install --no-compile "lm_eval' not in dockerfile
    assert "lm-eval==0.4.12" in lock.read_text(encoding="utf-8")
    assert "--hash=sha256:" in lock.read_text(encoding="utf-8")
    assert "org.invarlock.example.base-image-id" in dockerfile
    assert "lm-evaluation-harness-example.py" in dockerfile
