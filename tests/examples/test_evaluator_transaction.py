from __future__ import annotations

import hashlib
import importlib
import json
import subprocess
import sys
import types
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import yaml
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

from examples.integrations.evaluator_transaction.build_attestation import (
    make_evaluator_build_attestation,
    sign_evaluator_build_attestation,
    write_evaluator_build_attestation,
)
from invarlock.core.runtime_provider import (
    build_runtime_behavioral_schedule_from_material,
)

ROOT = Path(__file__).resolve().parents[2]


def _module() -> ModuleType:
    return importlib.reload(
        importlib.import_module(
            "examples.integrations.evaluator_transaction.transaction"
        )
    )


def _launcher_module() -> ModuleType:
    return importlib.reload(
        importlib.import_module("examples.integrations.evaluator_transaction.launcher")
    )


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


def _sample(module: ModuleType, *, score: float = 1.0) -> dict[str, object]:
    output = "Answer"
    return {
        "record_id": "stable-1",
        "prompt": "Prompt",
        "target": "Answer",
        "output": output,
        "input_sha256": module.digest(b"Prompt"),
        "target_sha256": module.digest(b"Answer"),
        "output_sha256": module.digest(output.encode()),
        "reported_score": score,
        "score_detail": {"score": score},
        "status": "ok",
    }


@pytest.mark.parametrize("evaluator", ["inspect-ai", "openai-evals"])
def test_evaluator_profiles_bind_distinct_upstream_entrypoints(evaluator: str) -> None:
    module = _module()
    config = module.execution_config(evaluator)
    assert config["evaluator"] == evaluator
    assert config["evaluator_entrypoint"] == module.EVALUATORS[evaluator]["entrypoint"]
    assert config["max_generation_tokens"] == 1
    assert config["trust_remote_code"] is False


@pytest.mark.parametrize("evaluator", ["inspect-ai", "openai-evals"])
def test_evaluator_transaction_worker_images_include_flat_script_dependencies(
    evaluator: str,
) -> None:
    dockerfile = (ROOT / "examples/integrations" / evaluator / "Dockerfile").read_text(
        encoding="utf-8"
    )

    for helper in (
        "trust_material.py",
        "local_registry.py",
        "launch.py",
    ):
        assert f"COPY examples/integrations/{helper}" in dockerfile
        assert f"/opt/invarlock/examples/{helper}" in dockerfile
    assert "COPY examples/integrations/evaluator_transaction" in dockerfile
    assert "/opt/invarlock/examples/evaluator_transaction" in dockerfile


def test_evaluator_transaction_dataset_digest_matches_the_staging_writer() -> None:
    module = _module()
    records = json.loads(
        (ROOT / "examples/integrations/lm-evaluation-harness/records.json").read_text(
            encoding="utf-8"
        )
    )
    staged_bytes = b"".join(
        (json.dumps(record, sort_keys=True) + "\n").encode() for record in records
    )
    assert (
        module.corpus_profile("quick").dataset_sha256
        == hashlib.sha256(staged_bytes).hexdigest()
    )


def test_inspect_bridge_restores_the_authenticated_causal_boundary() -> None:
    module = _module()

    assert (
        module.adapters._restore_inspect_causal_boundary("Paris", " Paris") == " Paris"
    )
    assert module.adapters._restore_inspect_causal_boundary("Paris", "Paris") == "Paris"
    assert (
        module.adapters._restore_inspect_causal_boundary(" Paris", " Paris") == " Paris"
    )


def _records_bytes(module: ModuleType, *, duplicate: bool = False) -> bytes:
    records = json.loads(
        (ROOT / "examples/integrations/lm-evaluation-harness/records.json").read_text(
            encoding="utf-8"
        )
    )
    if duplicate:
        records[-1]["id"] = records[0]["id"]
    return b"".join(
        (json.dumps(record, sort_keys=True) + "\n").encode() for record in records
    )


def test_evaluator_transaction_record_loader_accepts_only_pinned_corpora() -> None:
    module = _module()
    assert len(module.adapters._records(_records_bytes(module))) == 102
    with pytest.raises(module.BridgeError, match="not a pinned evaluator corpus"):
        module.adapters._records(module.canonical_json_bytes({"id": "only-one"}))
    with pytest.raises(module.BridgeError, match="not a pinned evaluator corpus"):
        module.adapters._records(_records_bytes(module, duplicate=True))


def test_evaluator_transaction_worker_and_cli_command_wrappers_are_bounded(
    tmp_path: Path,
) -> None:
    del tmp_path
    module = _module()
    with pytest.raises(module.BridgeError, match="stdout limit exceeded"):
        module._run_local_cli([sys.executable, "-c", "print('x' * (5 * 1024 * 1024))"])
    module.WORKER_TIMEOUT_SECONDS = 1
    with pytest.raises(module.BridgeError, match="timed out"):
        module._run_local_cli([sys.executable, "-c", "import time; time.sleep(2)"])


def test_inspect_runner_binds_each_sample_to_native_output_and_score(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    monkeypatch.setenv("INVARLOCK_EVALUATOR", "inspect-ai")
    records = [json.loads(line) for line in _records_bytes(module).splitlines()]

    class FakeSample:
        def __init__(self, item: dict[str, str]) -> None:
            self.id = item["id"]
            self.input = item["prompt"]
            self.target = item["expected"]
            self.output = SimpleNamespace(completion=item["expected"])
            self.scores = {
                "match": SimpleNamespace(
                    value="C", answer=item["expected"], explanation="exact"
                )
            }

    inspect_ai = types.ModuleType("inspect_ai")
    inspect_ai.Task = lambda **kwargs: SimpleNamespace(**kwargs)  # type: ignore[attr-defined]
    inspect_ai.eval = lambda *_args, **_kwargs: [
        SimpleNamespace(
            status="success",
            samples=[FakeSample(item) for item in records],
        )
    ]
    dataset = types.ModuleType("inspect_ai.dataset")
    dataset.MemoryDataset = lambda values: values
    dataset.Sample = lambda **kwargs: SimpleNamespace(**kwargs)
    scorer = types.ModuleType("inspect_ai.scorer")
    scorer.match = lambda **kwargs: kwargs
    solver = types.ModuleType("inspect_ai.solver")
    solver.generate = lambda: "generate"
    monkeypatch.setitem(sys.modules, "inspect_ai", inspect_ai)
    monkeypatch.setitem(sys.modules, "inspect_ai.dataset", dataset)
    monkeypatch.setitem(sys.modules, "inspect_ai.scorer", scorer)
    monkeypatch.setitem(sys.modules, "inspect_ai.solver", solver)

    generated, scored = module.adapters._run_inspect_ai(
        Path("/model"), _records_bytes(module)
    )
    assert generated[0]["output"] == records[0]["expected"]
    assert generated[-1]["id"] == records[-1]["id"]
    assert scored[0][0] == 1.0
    assert scored[0][1]["value"] == "C"


def test_openai_runner_binds_event_identity_and_restores_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    records = [json.loads(line) for line in _records_bytes(module).splitlines()]
    samples = [{"input": item["prompt"], "ideal": item["expected"]} for item in records]
    targets = iter(item["expected"] for item in records)

    class FakeGenerator:
        def __init__(self, _path: Path) -> None:
            self.closed = False

        def generate(self, prompts: list[str]) -> list[str]:
            return [next(targets) for _ in prompts]

        def close(self) -> None:
            self.closed = True

    class FakeMatch:
        def __init__(self, *, completion_fns: list[object], **_kwargs: object) -> None:
            self.completion_fns = completion_fns

        def get_samples(self) -> list[dict[str, str]]:
            return samples

        def eval_all_samples(
            self, recorder: object, samples: list[dict[str, str]], **_kwargs: object
        ) -> None:
            for index, item in enumerate(samples):
                result = self.completion_fns[0](prompt=item["input"])
                completion = result.get_completions()[0]
                recorder.events.append(
                    SimpleNamespace(
                        sample_id=f"sample.{index}",
                        data={
                            "sample_id": f"sample.{index}",
                            "expected": item["ideal"],
                            "sampled": completion,
                            "correct": True,
                            "picked": completion,
                        },
                    )
                )

    class FakeRecorder:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            self.events: list[object] = []

        def get_events(self, _name: str) -> list[object]:
            return self.events

    evals = types.ModuleType("evals")
    elsuite = types.ModuleType("evals.elsuite")
    basic = types.ModuleType("evals.elsuite.basic")
    match = types.ModuleType("evals.elsuite.basic.match")
    match.Match = FakeMatch
    record = types.ModuleType("evals.record")
    record.DummyRecorder = FakeRecorder
    record.RunSpec = lambda **kwargs: kwargs
    monkeypatch.setitem(sys.modules, "evals", evals)
    monkeypatch.setitem(sys.modules, "evals.elsuite", elsuite)
    monkeypatch.setitem(sys.modules, "evals.elsuite.basic", basic)
    monkeypatch.setitem(sys.modules, "evals.elsuite.basic.match", match)
    monkeypatch.setitem(sys.modules, "evals.record", record)
    monkeypatch.setattr(module.adapters, "_HfGreedyGenerator", FakeGenerator)
    monkeypatch.setenv("EVALS_SEQUENTIAL", "before")
    monkeypatch.delenv("EVALS_THREADS", raising=False)
    monkeypatch.delenv("EVALS_SHOW_EVAL_PROGRESS", raising=False)

    generated, scored = module.adapters._run_openai_evals(
        Path("/model"), _records_bytes(module)
    )
    assert generated[0]["output"] == records[0]["expected"]
    assert scored[-1][0] == 1.0
    assert module.os.environ["EVALS_SEQUENTIAL"] == "before"
    assert "EVALS_THREADS" not in module.os.environ


def test_openai_completion_callback_rejects_invalid_adapter_results() -> None:
    module = _module()

    class Generator:
        def __init__(self, result: list[str]) -> None:
            self.result = result

        def generate(self, _prompts: list[str]) -> list[str]:
            return self.result

    callback = module.adapters._OpenAIHfCompletionFn(Generator(["Answer"]))
    assert callback(prompt="Prompt").get_completions() == ["Answer"]
    with pytest.raises(module.BridgeError, match="non-text prompt"):
        callback(prompt=1)  # type: ignore[arg-type]
    with pytest.raises(module.BridgeError, match="invalid result"):
        module.adapters._OpenAIHfCompletionFn(Generator([]))(prompt="Prompt")


def test_evaluator_transaction_complete_replays_and_authenticates_a_fully_stubbed_transaction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    prepared = tmp_path / "prepared"
    image = "sha256:" + "1" * 64
    records = json.loads(
        (ROOT / "examples/integrations/lm-evaluation-harness/records.json").read_text(
            encoding="utf-8"
        )
    )
    raw_dataset = b"".join(
        (json.dumps(record, sort_keys=True) + "\n").encode() for record in records
    )
    quick = module.corpus_profile("quick")
    models = module.model_profile("quick")
    tree_digests = {
        snapshot.role: snapshot.checkpoint_tree_sha256 for snapshot in models.snapshots
    }
    tokenizer_digests = {
        snapshot.role: snapshot.tokenizer_contract_sha256
        for snapshot in models.snapshots
    }
    assert module.digest(raw_dataset) == quick.dataset_sha256
    dataset_path = prepared / "evaluation/inputs/records.jsonl"
    dataset_path.parent.mkdir(parents=True)
    dataset_path.write_bytes(raw_dataset)
    policy = {
        "resolved_policy": {
            "metrics": {
                "exact_match": {
                    "delta_min_pp": -20.0,
                    "maximum_interval_width_pp": 20.0,
                    "minimum_record_count": 102,
                    "minimum_side_accuracy": quick.minimum_side_accuracy,
                }
            }
        }
    }
    policy_path = prepared / "evaluation/inputs/acceptance.json"
    policy_path.write_bytes(module.canonical_json_bytes(policy))
    (prepared / "evaluation/inputs/corpus-profile.json").write_bytes(
        module.canonical_json_bytes(
            module.corpus_provenance(module.corpus_profile("quick"))
        )
    )
    for role in ("baseline", "subject"):
        (prepared / f"evaluation/models/{role}").mkdir(parents=True)

    expected_settings = {
        "batch_size": models.batch_size,
        "checkpoint_tree_sha256": tree_digests["baseline"],
        "context_length": 64,
        "max_output_tokens": module.MAX_GENERATION_TOKENS,
        "offline": True,
        "seed": module.SEED,
        "timeout_seconds": module.PER_RECORD_TIMEOUT_SECONDS,
        "tokenizer_metadata_sha256": tokenizer_digests["baseline"],
    }
    comparisons: dict[str, object] = {}
    for role in ("baseline", "subject"):
        settings = {
            **expected_settings,
            "checkpoint_tree_sha256": tree_digests[role],
            "tokenizer_metadata_sha256": tokenizer_digests[role],
        }
        comparisons[role] = {
            "artifact": module.model_artifacts(models)[role],
            "runtime": {"provider": "hf_transformers", "settings": settings},
        }
    request = {
        "comparison": {
            "baseline": comparisons["baseline"],
            "subject": comparisons["subject"],
            "dataset": {
                "path": "inputs/records.jsonl",
                "sha256": quick.dataset_sha256,
                "format": "jsonl",
                "name": quick.dataset_name,
                "split": "validation",
                "input_field": "prompt",
                "expected_output_field": "expected",
                "id_field": "id",
            },
            "policy": "inputs/acceptance.json",
            "task": "text_causal",
            "metric": "exact_match",
        }
    }
    prepared_request = prepared / "evaluation/request.yaml"
    prepared_request.parent.mkdir(parents=True, exist_ok=True)
    prepared_request.write_text(yaml.safe_dump(request), encoding="utf-8")
    (prepared / "runtime-image-id.txt").write_text(image + "\n", encoding="ascii")

    lock_digest = module.evaluator_lock_digest("inspect-ai")

    def write_run(role: str, output: Path) -> None:
        output.mkdir(parents=True)
        samples = []
        for record in records:
            samples.append(
                {
                    "record_id": record["id"],
                    "prompt": record["prompt"],
                    "target": record["expected"],
                    "output": record["expected"],
                    "input_sha256": module.digest(record["prompt"].encode()),
                    "target_sha256": module.digest(record["expected"].encode()),
                    "output_sha256": module.digest(record["expected"].encode()),
                    "reported_score": 1.0,
                    "score_detail": {"score": 1.0},
                    "status": "ok",
                }
            )
        sample_bytes = b"".join(module.canonical_json_bytes(item) for item in samples)
        (output / "samples.jsonl").write_bytes(sample_bytes)
        manifest = {
            "format": "invarlock/evaluator-run-v1",
            "role": role,
            "evaluator": "inspect-ai",
            "evaluator_version": module.EVALUATORS["inspect-ai"]["version"],
            "task_config": module.task_config("/records.jsonl", "inspect-ai"),
            "task_config_sha256": module.digest(
                module.canonical_json_bytes(
                    module.task_config("/records.jsonl", "inspect-ai")
                )
            ),
            "execution_config": module.execution_config("inspect-ai"),
            "execution_config_sha256": module.digest(
                module.canonical_json_bytes(module.execution_config("inspect-ai"))
            ),
            "samples": "samples.jsonl",
            "samples_sha256": module.digest(sample_bytes),
            "model_tree_sha256": tree_digests[role],
            "dataset_sha256": quick.dataset_sha256,
            "evaluator_lock_sha256": lock_digest,
            "runtime_image_digest": image,
            "record_count": 102,
            "stable_id_field": "record_id",
        }
        (output / "run-manifest.json").write_bytes(
            module.canonical_json_bytes(manifest)
        )

    monkeypatch.setattr(module, "_inspect_runtime_image", lambda *_a, **_k: None)
    monkeypatch.setattr(
        module,
        "checkpoint_tree_sha256",
        lambda path: tree_digests["subject" if "subject" in str(path) else "baseline"],
    )
    monkeypatch.setattr(
        module,
        "_run_verified_worker",
        lambda **kwargs: write_run(kwargs["role"], kwargs["output"]),
    )
    monkeypatch.setattr(
        module,
        "_run_local_cli",
        lambda _command: "ok",
    )
    monkeypatch.setattr(module, "validate_completed_outputs", lambda *_args: None)
    monkeypatch.setattr(
        module,
        "load_external_scoring_records_jsonl",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        module,
        "write_runtime_import_side",
        lambda *args, **kwargs: SimpleNamespace(
            provider_evidence=SimpleNamespace(artifact_identity_bytes=b"identity")
        ),
    )
    monkeypatch.setattr(
        module, "write_runtime_import_paired_records", lambda *_a, **_k: None
    )

    class FakeTokenizer:
        def __init__(self, digest_value: str) -> None:
            self.digest = digest_value

    transformers = types.ModuleType("transformers")
    transformers.AutoTokenizer = SimpleNamespace(
        from_pretrained=lambda path, **_kwargs: FakeTokenizer(
            tokenizer_digests["subject" if "subject" in str(path) else "baseline"]
        )
    )
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setattr(
        "invarlock.runtime_providers.hf_transformers.hf_tokenizer_contract_sha256",
        lambda tokenizer: tokenizer.digest,
    )

    class FakeProvider:
        def authenticate_artifact(self, _spec: object, checkpoint: Path) -> object:
            return SimpleNamespace(artifact_identity_bytes=str(checkpoint).encode())

        def capabilities(self) -> dict[str, object]:
            return {}

    monkeypatch.setattr(module, "HFTransformersProvider", FakeProvider)
    evidence_key = tmp_path / "evidence.pem"
    verifier_key = tmp_path / "verifier.pem"
    builder_key = tmp_path / "builder.pem"
    for path in (evidence_key, verifier_key, builder_key):
        key = ed25519.Ed25519PrivateKey.generate()
        path.write_bytes(
            key.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.PKCS8,
                serialization.NoEncryption(),
            )
        )
    builder_private = serialization.load_pem_private_key(
        builder_key.read_bytes(), password=None
    )
    assert isinstance(builder_private, ed25519.Ed25519PrivateKey)
    builder_public = tmp_path / "builder-public.pem"
    builder_public.write_bytes(
        builder_private.public_key().public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    attestation = tmp_path / "attestation.json"
    attestation.write_text("{}\n", encoding="utf-8")

    evidence, receipt, report = module.complete(
        tmp_path / "transaction",
        prepared,
        image,
        "inspect-ai",
        container_engine="docker",
        evidence_signing_key=evidence_key,
        verifier_signing_key=verifier_key,
        trust_root=tmp_path / "trust-root",
        source_commit="a" * 40,
        base_image_id="sha256:" + "2" * 64,
        build_attestation=attestation,
        builder_public_key=builder_public,
    )
    assert evidence == tmp_path / "transaction/evidence"
    assert receipt == tmp_path / "transaction/verifier/verification.receipt.json"
    assert report == tmp_path / "transaction/comparison-report.html"


def test_openai_evals_event_shape_is_bound_to_the_upstream_record() -> None:
    module = _module()
    record = {"id": "stable-1", "prompt": "Prompt", "expected": "Answer"}

    completion, score, detail = module.adapters._openai_event_to_sample(
        record,
        {
            "expected": "Answer",
            "sampled": "Answer",
            "correct": True,
            "picked": "Answer",
        },
    )
    assert (completion, score, detail) == (
        "Answer",
        1.0,
        {
            "picked": "Answer",
            "native_correct": True,
            "transaction_correct": True,
        },
    )

    completion, score, detail = module.adapters._openai_event_to_sample(
        record,
        {
            "expected": "Answer",
            "sampled": "Answer with extra text",
            "correct": True,
            "picked": "Answer",
        },
    )
    assert completion == "Answer with extra text"
    assert score == 0.0
    assert detail == {
        "picked": "Answer",
        "native_correct": True,
        "transaction_correct": False,
    }

    with pytest.raises(module.BridgeError, match="identity or target"):
        module.adapters._openai_event_to_sample(
            record,
            {"expected": ["Answer"], "sampled": "Answer", "correct": True},
        )

    with pytest.raises(module.BridgeError, match="inconsistent native match"):
        module.adapters._openai_event_to_sample(
            record,
            {"expected": "Answer", "sampled": "Answer", "correct": False},
        )


def test_child_image_config_rejects_uncontracted_runtime_changes() -> None:
    from examples.integrations.launch import (
        _require_child_image_config,
        _require_child_image_layers,
    )

    base = {
        "Cmd": ["python"],
        "Env": ["SAFE=1"],
        "Labels": {"base": "1"},
        "User": "65532:65532",
        "WorkingDir": "/opt/invarlock",
    }
    child = {
        **base,
        "User": "0:0",
        "Env": ["SAFE=1", "ALLOWED=1"],
        "Labels": {"base": "1", "allowed": "1"},
    }

    with pytest.raises(RuntimeError, match="'User'"):
        _require_child_image_config(
            base,
            child,
            allowed_environment={"ALLOWED"},
            allowed_labels={"allowed"},
        )
    _require_child_image_config(
        base,
        {**base, "WorkingDir": "/opt/invarlock/examples"},
        allowed_environment=set(),
        allowed_labels=set(),
        expected_working_directory="/opt/invarlock/examples",
    )
    with pytest.raises(RuntimeError, match="working directory"):
        _require_child_image_config(
            base,
            {**base, "WorkingDir": "/tmp"},
            allowed_environment=set(),
            allowed_labels=set(),
            expected_working_directory="/opt/invarlock/examples",
        )

    base_layers = ("sha256:" + "a" * 64, "sha256:" + "b" * 64)
    _require_child_image_layers(
        base_layers,
        (*base_layers, "sha256:" + "c" * 64, "sha256:" + "d" * 64),
    )
    with pytest.raises(RuntimeError, match="derive from the authenticated base"):
        _require_child_image_layers(base_layers, base_layers)
    with pytest.raises(RuntimeError, match="derive from the authenticated base"):
        _require_child_image_layers(
            base_layers,
            (base_layers[0], "sha256:" + "c" * 64, base_layers[1]),
        )


def test_adapter_emits_strict_runtime_import_records(tmp_path: Path) -> None:
    module = _module()
    samples = tmp_path / "samples.jsonl"
    samples.write_bytes(module.canonical_json_bytes(_sample(module)))
    destination = tmp_path / "records.jsonl"

    module.adapt(samples, _schedule(), destination)

    record = json.loads(destination.read_text(encoding="utf-8"))
    assert record == {
        "input_sha256": _schedule().records[0].input_sha256,
        "output_sha256": hashlib.sha256(b"Answer").hexdigest(),
        "output_text": "Answer",
        "record_id": "stable-1",
        "status": "ok",
    }


def test_adapter_rejects_evaluator_score_disagreement(tmp_path: Path) -> None:
    module = _module()
    samples = tmp_path / "samples.jsonl"
    samples.write_bytes(module.canonical_json_bytes(_sample(module, score=0.0)))

    with pytest.raises(module.BridgeError, match="disagrees with replay"):
        module.adapt(samples, _schedule(), tmp_path / "records.jsonl")


def test_runtime_image_inspection_requires_and_dispatches_attestation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    image = "sha256:" + ("d" * 64)
    lock = "sha256:" + ("e" * 64)
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        module,
        "inspect_evaluator_image",
        lambda **kwargs: captured.update(kwargs),
    )

    module._inspect_runtime_image(
        "docker",
        image,
        "inspect-ai",
        lock,
        source_commit="c" * 40,
        base_image_id="sha256:" + "a" * 64,
        build_attestation=tmp_path / "attestation.json",
        builder_public_key=module.ed25519.Ed25519PrivateKey.generate().public_key(),
    )

    assert captured["image"] == image
    assert captured["lock_sha256"] == lock
    assert captured["expected_entrypoint"] == (
        "python",
        "-m",
        "evaluator_transaction.cli",
        "worker",
    )


def test_shared_image_inspection_rechecks_the_signed_oci_observation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from examples.integrations import launch as shared_launch

    image = "sha256:" + "a" * 64
    base = "sha256:" + "b" * 64
    layer = "sha256:" + "c" * 64
    source = "sha256:" + "d" * 64
    lock = "sha256:" + "e" * 64
    commit = "f" * 40
    entrypoint = ("python", "-m", "evaluator_transaction.cli", "worker")
    config = {
        "Entrypoint": list(entrypoint),
        "Labels": {
            "org.invarlock.example.base-image-id": base,
            "org.invarlock.example.evaluator": "inspect-ai",
            "org.invarlock.example.evaluator-version": "0.3.254",
            "org.invarlock.example.evaluator-lock-sha256": lock,
            "org.invarlock.example.source-commit": commit,
            "org.invarlock.example.source-bundle-sha256": source,
        },
    }
    signing_key = ed25519.Ed25519PrivateKey.generate()
    attestation = make_evaluator_build_attestation(
        evaluator="inspect-ai",
        evaluator_version="0.3.254",
        runtime_image_id=image,
        base_image_id=base,
        source_commit=commit,
        source_bundle_sha256=source,
        lock_sha256=lock,
        entrypoint=entrypoint,
        base_layers=[base],
        image_layers=[base, layer],
        config=config,
    )
    attestation_path = tmp_path / "attestation.json"
    write_evaluator_build_attestation(
        attestation_path, sign_evaluator_build_attestation(attestation, signing_key)
    )

    def fake_run(
        command: list[str], *, cwd: Path, capture_output: bool = False
    ) -> subprocess.CompletedProcess[str]:
        del cwd, capture_output
        template = command[4]
        if template == "{{.Id}}":
            stdout = image
        elif "RootFS.Layers" in template:
            stdout = json.dumps([base, layer] if command[-1] == image else [base])
        elif template == "{{json .Config}}":
            stdout = json.dumps(config)
        else:
            raise AssertionError(command)
        return subprocess.CompletedProcess(command, 0, stdout, "")

    monkeypatch.setattr(shared_launch, "_run", fake_run)
    observed = shared_launch.inspect_evaluator_image(
        engine="docker",
        image=image,
        repository=tmp_path,
        attestation_path=attestation_path,
        evaluator="inspect-ai",
        evaluator_version="0.3.254",
        lock_sha256=lock,
        expected_entrypoint=entrypoint,
        source_commit=commit,
        base_image_id=base,
        builder_public_key=signing_key.public_key(),
    )

    assert observed["statement"] == attestation


def test_workers_mount_role_private_output_parents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    prepared = tmp_path / "prepared"
    for role in ("baseline", "subject"):
        (prepared / f"evaluation/models/{role}").mkdir(parents=True)
    (prepared / "evaluation/inputs").mkdir(parents=True)
    (prepared / "evaluation/inputs/records.jsonl").write_text("{}\n", encoding="utf-8")
    commands: list[dict[str, object]] = []

    def fake_worker(**kwargs: object) -> SimpleNamespace:
        commands.append(kwargs)
        assert kwargs["timeout_seconds"] == module.WORKER_TIMEOUT_SECONDS
        return SimpleNamespace(returncode=0, stderr="", stdout="")

    monkeypatch.setattr(
        module,
        "run_evaluator_worker",
        fake_worker,
    )

    for role in ("baseline", "subject"):
        module._run_verified_worker(
            engine="docker",
            image="sha256:" + "a" * 64,
            selected="inspect-ai",
            role=role,
            prepared=prepared,
            output=tmp_path / f"upstream/{role}/result",
            lock_digest="sha256:" + "b" * 64,
        )

    assert [call["worker_arguments"][1] for call in commands] == [
        "baseline",
        "subject",
    ]
    assert all(call["output"].name == "result" for call in commands)


def test_launcher_returns_the_verified_child_image_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launcher = _launcher_module()
    from examples.integrations import launch as shared_launch

    base_id = "sha256:" + "a" * 64
    child_id = "sha256:" + "b" * 64
    commit = "c" * 40
    base_layers = ["sha256:" + "1" * 64]
    child_layers = base_layers + ["sha256:" + digit * 64 for digit in "23456789"]
    base_config = {
        "ArgsEscaped": False,
        "Cmd": ["/bin/sh"],
        "Entrypoint": None,
        "Env": ["BASE_SETTING=preserved"],
        "Labels": {"base-setting": "preserved"},
        "OnBuild": None,
        "Shell": ["/bin/sh"],
        "StopSignal": "SIGTERM",
        "User": "",
        "WorkingDir": "/",
    }
    child_config = {
        **base_config,
        "WorkingDir": "/opt/invarlock/examples",
        "Entrypoint": [
            "python",
            "-m",
            "evaluator_transaction.cli",
            "worker",
        ],
        "Env": [
            "BASE_SETTING=preserved",
            "INVARLOCK_EVALUATOR=inspect-ai",
            "INVARLOCK_EVALUATOR_LOCK_SHA256=" + "sha256:" + "f" * 64,
        ],
        "Labels": {
            "base-setting": "preserved",
            "org.invarlock.example.base-image-id": base_id,
            "org.invarlock.example.evaluator": "inspect-ai",
            "org.invarlock.example.evaluator-version": "0.3.254",
            "org.invarlock.example.evaluator-lock-sha256": "sha256:" + "f" * 64,
            "org.invarlock.example.evaluator-runtime": "cpu",
            "org.invarlock.example.source-commit": commit,
            "org.invarlock.example.source-bundle-sha256": "sha256:"
            + hashlib.sha256(b"test-source").hexdigest(),
        },
    }
    lock_digest = (
        "sha256:"
        + hashlib.sha256((ROOT / launcher.LOCKS["inspect-ai"]).read_bytes()).hexdigest()
    )
    child_config["Env"][2] = "INVARLOCK_EVALUATOR_LOCK_SHA256=" + lock_digest
    child_config["Labels"]["org.invarlock.example.evaluator-lock-sha256"] = lock_digest
    calls: list[list[str]] = []
    monkeypatch.setattr(shared_launch, "_require_committed_checkout", lambda _r: commit)
    monkeypatch.setattr(
        shared_launch, "_runtime_image", lambda **_kwargs: (base_id, base_id)
    )

    def fake_run(
        command: list[str], *, cwd: Path, stdin_path: Path | None = None
    ) -> str:
        calls.append(command)
        if command[:2] == ["git", "archive"]:
            output = next(
                Path(value.removeprefix("--output="))
                for value in command
                if value.startswith("--output=")
            )
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(b"test-source")
            return ""
        if command[:3] == ["docker", "image", "inspect"]:
            template = command[4]
            if "RootFS.Layers" in template:
                return json.dumps(
                    base_layers if command[-1] == base_id else child_layers
                )
            if template == "{{json .Config}}":
                return json.dumps(
                    base_config if command[-1] == base_id else child_config
                )
            if template == "{{.Id}}":
                return base_id if "example-runtime" in command[-1] else child_id
            if "base-image-id" in template:
                return base_id
            if "evaluator-lock-sha256" in template:
                return lock_digest
            if "org.invarlock.example.evaluator" in template:
                return "inspect-ai"
        if command[:2] == ["docker", "build"]:
            Path(command[command.index("--iidfile") + 1]).write_text(
                child_id + "\n", encoding="ascii"
            )
        return ""

    monkeypatch.setattr(launcher, "run", fake_run)
    result = launcher._build_image(
        "inspect-ai",
        launcher.REPOSITORY,
        tmp_path / "build",
        "docker",
        builder_signing_key=launcher.ed25519.Ed25519PrivateKey.generate(),
        cleanup_tags=[],
    )

    build = next(command for command in calls if command[:2] == ["docker", "build"])
    assert "--pull=false" in build
    assert "--iidfile" in build
    assert result == (child_id, commit, base_id)
    build_tag = build[build.index("--tag") + 1]
    assert build_tag.startswith("invarlock-inspect-ai-evaluator:" + commit[:12])
    assert any(
        command[-1] == build_tag
        for command in calls
        if command[:3] == ["docker", "image", "inspect"]
    )


def test_launcher_cleanup_removes_only_tags_that_still_name_owned_images(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launcher = _launcher_module()
    calls: list[list[str]] = []

    def fake_run(
        command: list[str], *, cwd: Path, stdin_path: Path | None = None
    ) -> str:
        calls.append(command)
        if command[:3] == ["docker", "image", "inspect"]:
            return {
                "invarlock-base:owned": "sha256:" + "a" * 64,
                "invarlock-child:owned": "sha256:" + "b" * 64,
            }[command[-1]]
        return ""

    monkeypatch.setattr(launcher, "run", fake_run)
    launcher.remove_owned_image_tags(
        launcher.run,
        "docker",
        ROOT,
        [
            launcher.OwnedImageTag("invarlock-base:owned", "sha256:" + "a" * 64),
            launcher.OwnedImageTag("invarlock-base:owned", "sha256:" + "a" * 64),
            launcher.OwnedImageTag("invarlock-child:owned", "sha256:" + "b" * 64),
        ],
    )

    assert calls == [
        ["docker", "image", "inspect", "--format", "{{.Id}}", "invarlock-base:owned"],
        ["docker", "image", "rm", "invarlock-base:owned"],
        ["docker", "image", "inspect", "--format", "{{.Id}}", "invarlock-child:owned"],
        ["docker", "image", "rm", "invarlock-child:owned"],
    ]


def test_launcher_cleanup_refuses_a_reassigned_tag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launcher = _launcher_module()
    calls: list[list[str]] = []

    def fake_run(
        command: list[str], *, cwd: Path, stdin_path: Path | None = None
    ) -> str:
        calls.append(command)
        return "sha256:" + "b" * 64

    with pytest.raises(RuntimeError, match="ownership changed"):
        launcher.remove_owned_image_tags(
            fake_run,
            "docker",
            ROOT,
            [launcher.OwnedImageTag("invarlock-base:owned", "sha256:" + "a" * 64)],
        )

    assert not any(command[:3] == ["docker", "image", "rm"] for command in calls)


def test_launcher_cleanup_refuses_conflicting_ownership_records() -> None:
    launcher = _launcher_module()
    calls: list[list[str]] = []

    def fake_run(command: list[str], *, cwd: Path) -> str:
        calls.append(command)
        return ""

    with pytest.raises(RuntimeError, match="conflicting owned image identities"):
        launcher.remove_owned_image_tags(
            fake_run,
            "docker",
            ROOT,
            [
                launcher.OwnedImageTag("shared", "sha256:" + "a" * 64),
                launcher.OwnedImageTag("shared", "sha256:" + "b" * 64),
            ],
        )

    assert calls == []


def test_launcher_records_only_a_tag_created_for_the_built_image() -> None:
    launcher = _launcher_module()

    def missing_tag(*_args: object, **_kwargs: object) -> str:
        raise RuntimeError("No such image")

    with pytest.raises(RuntimeError, match="temporary image tag was not created"):
        launcher.record_owned_image_tag(
            missing_tag,
            "docker",
            "invarlock-base:owned",
            "sha256:" + "a" * 64,
            ROOT,
        )

    with pytest.raises(RuntimeError, match="does not name the image built"):
        launcher.record_owned_image_tag(
            lambda *_args, **_kwargs: "sha256:" + "b" * 64,
            "docker",
            "invarlock-base:owned",
            "sha256:" + "a" * 64,
            ROOT,
        )


def test_launcher_cleanup_ignores_absent_tags_and_reports_engine_failures() -> None:
    launcher = _launcher_module()
    calls: list[list[str]] = []
    image_id = "sha256:" + "a" * 64

    def fake_run(command: list[str], *, cwd: Path) -> str:
        calls.append(command)
        tag = command[-1]
        if command[:3] == ["docker", "image", "inspect"]:
            if tag == "absent":
                raise RuntimeError("No such object")
            if tag == "inspect-failed":
                raise RuntimeError("daemon unavailable")
            return image_id
        if tag == "remove-failed":
            raise RuntimeError("permission denied")
        return ""

    with pytest.raises(RuntimeError) as raised:
        launcher.remove_owned_image_tags(
            fake_run,
            "docker",
            ROOT,
            [
                launcher.OwnedImageTag("absent", image_id),
                launcher.OwnedImageTag("inspect-failed", image_id),
                launcher.OwnedImageTag("remove-failed", image_id),
            ],
        )

    diagnostic = str(raised.value)
    assert "inspect-failed: daemon unavailable" in diagnostic
    assert "remove-failed: permission denied" in diagnostic
    assert ["docker", "image", "rm", "absent"] not in calls
    assert ["docker", "image", "rm", "inspect-failed"] not in calls


def test_evaluator_transaction_launcher_bounds_child_output_and_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launcher = _launcher_module()
    monkeypatch.setattr(launcher, "_COMMAND_STDOUT_LIMIT", 16)
    with pytest.raises(RuntimeError, match="stdout limit exceeded"):
        launcher.run(
            [
                sys.executable,
                "-c",
                "import sys; sys.stdout.write('x' * 100)",
            ],
            cwd=tmp_path,
        )

    monkeypatch.setattr(launcher, "_COMMAND_TIMEOUT_SECONDS", 1)
    with pytest.raises(RuntimeError, match="timed out"):
        launcher.run(
            [sys.executable, "-c", "import time; time.sleep(2)"],
            cwd=tmp_path,
        )


def test_completion_requires_distinct_evidence_and_verifier_signers() -> None:
    module = _module()
    key = module.ed25519.Ed25519PrivateKey.generate()

    with pytest.raises(module.BridgeError, match="must be distinct"):
        module._require_distinct_signers(key, key)


def test_worker_load_run_returns_the_verified_sample_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    monkeypatch.setenv("INVARLOCK_EVALUATOR", "inspect-ai")
    monkeypatch.setenv("INVARLOCK_RUNTIME_IMAGE_ID", "sha256:" + ("b" * 64))
    monkeypatch.setenv("INVARLOCK_EVALUATOR_LOCK_SHA256", "sha256:" + ("c" * 64))
    monkeypatch.setattr(
        module,
        "evaluator_lock_digest",
        lambda _selected, *, container=False, profile=None: "sha256:" + ("c" * 64),
    )
    monkeypatch.setattr(module.importlib.metadata, "version", lambda _name: "0.3.254")
    monkeypatch.setattr(
        module, "checkpoint_tree_sha256", lambda _path: "sha256:" + ("a" * 64)
    )
    dataset = tmp_path / "records.jsonl"
    dataset.write_text(
        "".join(
            json.dumps({"expected": "Answer", "id": f"id-{i}", "prompt": "Prompt"})
            + "\n"
            for i in range(102)
        ),
        encoding="utf-8",
    )
    generated = [
        {
            "id": f"id-{i}",
            "prompt": "Prompt",
            "expected": "Answer",
            "output": "Answer",
        }
        for i in range(102)
    ]
    monkeypatch.setattr(
        module.adapters,
        "_run_upstream_evaluator",
        lambda _model, _dataset, _evaluator: (generated, [(1.0, {})] * 102),
    )
    model = tmp_path / "model"
    model.mkdir()
    output = tmp_path / "output"
    module.worker("baseline", model, dataset, output)

    manifest, snapshot = module.load_run(
        output / "run-manifest.json", "baseline", "inspect-ai"
    )
    original = (output / "samples.jsonl").read_bytes()
    assert snapshot == original
    (output / "samples.jsonl").write_bytes(b"tampered\n")
    assert snapshot == original
    assert module.digest(snapshot) == manifest["samples_sha256"]


def test_worker_binds_model_dataset_and_upstream_output_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    monkeypatch.setenv("INVARLOCK_EVALUATOR", "inspect-ai")
    monkeypatch.setenv("INVARLOCK_RUNTIME_IMAGE_ID", "sha256:" + ("b" * 64))
    monkeypatch.setenv("INVARLOCK_EVALUATOR_LOCK_SHA256", "sha256:" + ("c" * 64))
    monkeypatch.setattr(
        module,
        "evaluator_lock_digest",
        lambda _selected, *, container=False, profile=None: "sha256:" + ("c" * 64),
    )
    monkeypatch.setattr(
        module.importlib.metadata,
        "version",
        lambda _name: "0.3.254",
    )
    monkeypatch.setattr(
        module,
        "checkpoint_tree_sha256",
        lambda _path: "sha256:" + ("a" * 64),
    )
    model = tmp_path / "model"
    model.mkdir()
    dataset = tmp_path / "records.jsonl"
    dataset.write_text(
        "".join(
            json.dumps({"expected": "Answer", "id": f"id-{i}", "prompt": "Prompt"})
            + "\n"
            for i in range(102)
        ),
        encoding="utf-8",
    )
    generated = [
        {"id": f"id-{i}", "prompt": "Prompt", "expected": "Answer", "output": "Answer"}
        for i in range(102)
    ]
    monkeypatch.setattr(
        module.adapters,
        "_run_upstream_evaluator",
        lambda _model, _dataset, _evaluator: (generated, [(1.0, {})] * 102),
    )

    output = tmp_path / "output"
    module.worker("baseline", model, dataset, output)

    manifest = json.loads((output / "run-manifest.json").read_text(encoding="utf-8"))
    assert manifest["format"] == "invarlock/evaluator-run-v1"
    assert manifest["evaluator"] == "inspect-ai"
    assert manifest["record_count"] == 102
    assert manifest["samples_sha256"] == module.digest(
        (output / "samples.jsonl").read_bytes()
    )


def test_worker_rejects_a_dataset_outside_its_declared_corpus_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    monkeypatch.setenv("INVARLOCK_EVALUATOR", "inspect-ai")
    monkeypatch.setenv("INVARLOCK_RUNTIME_IMAGE_ID", "sha256:" + "a" * 64)
    monkeypatch.setenv("INVARLOCK_EVALUATOR_LOCK_SHA256", "sha256:" + "b" * 64)
    monkeypatch.setenv("INVARLOCK_CORPUS_PROFILE", "quick")
    monkeypatch.setattr(module.importlib.metadata, "version", lambda _name: "0.3.254")
    monkeypatch.setattr(
        module, "evaluator_lock_digest", lambda *_args, **_kwargs: "sha256:" + "b" * 64
    )
    monkeypatch.setattr(
        module, "checkpoint_tree_sha256", lambda _path: "sha256:" + "c" * 64
    )
    model = tmp_path / "model"
    model.mkdir()
    dataset = tmp_path / "records.jsonl"
    dataset.write_text("{}\n", encoding="utf-8")

    with pytest.raises(module.BridgeError, match="pinned evaluator corpus"):
        module.worker("baseline", model, dataset, tmp_path / "output")


def test_worker_rejects_symlinked_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    monkeypatch.setenv("INVARLOCK_EVALUATOR", "inspect-ai")
    monkeypatch.setenv("INVARLOCK_RUNTIME_IMAGE_ID", "sha256:" + ("b" * 64))
    monkeypatch.setenv("INVARLOCK_EVALUATOR_LOCK_SHA256", "sha256:" + ("c" * 64))
    monkeypatch.setattr(
        module,
        "evaluator_lock_digest",
        lambda _selected, *, container=False, profile=None: "sha256:" + ("c" * 64),
    )
    monkeypatch.setattr(module.importlib.metadata, "version", lambda _name: "0.3.254")
    model = tmp_path / "model"
    model.mkdir()
    real_dataset = tmp_path / "real-records.jsonl"
    real_dataset.write_text("{}\n", encoding="utf-8")
    dataset = tmp_path / "records.jsonl"
    dataset.symlink_to(real_dataset)

    with pytest.raises(module.BridgeError, match="inputs must exist"):
        module.worker("baseline", model, dataset, tmp_path / "output")


def test_evaluator_transaction_boundary_helpers_reject_untrusted_files_and_identity_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    original_nofollow = module.os.O_NOFOLLOW
    regular = tmp_path / "regular"
    regular.write_bytes(b"payload")
    assert module._read_regular_file(regular, label="regular") == b"payload"

    with pytest.raises(module.BridgeError, match="regular file"):
        module._read_regular_file(tmp_path, label="directory")
    oversized = tmp_path / "oversized"
    oversized.write_bytes(b"12345")
    with pytest.raises(module.BridgeError, match="size limit"):
        module._read_regular_file(oversized, label="oversized", max_bytes=4)
    link = tmp_path / "link"
    link.symlink_to(regular)
    with pytest.raises(module.BridgeError, match="opened without following"):
        module._read_regular_file(link, label="link")
    monkeypatch.setattr(module.os, "O_NOFOLLOW", None)
    with pytest.raises(module.BridgeError, match="unavailable"):
        module._read_regular_file(regular, label="regular")
    monkeypatch.setattr(module.os, "O_NOFOLLOW", original_nofollow)

    monkeypatch.delenv("INVARLOCK_RUNTIME_IMAGE_ID", raising=False)
    with pytest.raises(module.BridgeError, match="runtime image digest"):
        module._runtime_image_from_environment()
    monkeypatch.setenv("INVARLOCK_EVALUATOR", "not-maintained")
    with pytest.raises(module.BridgeError, match="maintained"):
        module.evaluator_id()
    with pytest.raises(module.BridgeError, match="unsupported evaluator"):
        module.task_config("records.jsonl", "not-maintained")

    invalid_private = tmp_path / "invalid-private.pem"
    invalid_private.write_bytes(b"bad")
    with pytest.raises(module.BridgeError, match="private key"):
        module._external_ed25519_key(invalid_private, label="private")
    invalid_public = tmp_path / "invalid-public.pem"
    invalid_public.write_bytes(b"bad")
    with pytest.raises(module.BridgeError, match="public key"):
        module._external_ed25519_public_key(invalid_public, label="public")
    private = ed25519.Ed25519PrivateKey.generate()
    private_path = tmp_path / "private.pem"
    private_path.write_bytes(
        private.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    public_path = tmp_path / "public.pem"
    public_path.write_bytes(
        private.public_key().public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    assert isinstance(
        module._external_ed25519_key(private_path, label="private"),
        ed25519.Ed25519PrivateKey,
    )
    assert isinstance(
        module._external_ed25519_public_key(public_path, label="public"),
        ed25519.Ed25519PublicKey,
    )


def test_evaluator_transaction_cli_preserves_symlink_for_nofollow_rejection(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _module()
    real_key = tmp_path / "real-evidence.pem"
    real_key.write_bytes(
        ed25519.Ed25519PrivateKey.generate().private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    key_link = tmp_path / "evidence.pem"
    key_link.symlink_to(real_key)
    prepared = tmp_path / "prepared"
    prepared.mkdir()

    result = module.main(
        [
            "complete",
            "--workspace",
            str(tmp_path / "transaction"),
            "--prepared",
            str(prepared),
            "--runtime-image",
            "sha256:" + "a" * 64,
            "--evaluator",
            "inspect-ai",
            "--evidence-signing-key",
            str(key_link),
            "--verifier-signing-key",
            str(tmp_path / "verifier.pem"),
            "--trust-root",
            str(tmp_path / "trust"),
            "--builder-public-key",
            str(tmp_path / "builder.pem"),
            "--source-commit",
            "b" * 40,
            "--base-image-id",
            "sha256:" + "c" * 64,
            "--build-attestation",
            str(tmp_path / "build-attestation.json"),
        ]
    )

    assert result == 2
    assert (
        "evidence signing key is not an Ed25519 private key" in capsys.readouterr().err
    )


def test_evaluator_transaction_image_and_mount_validation_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    key = ed25519.Ed25519PrivateKey.generate().public_key()
    kwargs = {
        "image": "sha256:" + "a" * 64,
        "selected": "inspect-ai",
        "lock_digest": "sha256:" + "b" * 64,
        "source_commit": "c" * 40,
        "base_image_id": "sha256:" + "d" * 64,
        "build_attestation": tmp_path / "attestation.json",
        "builder_public_key": key,
    }
    for field, value, message in (
        ("engine", "containerd", "docker or podman"),
        ("source_commit", "bad", "full lowercase"),
        ("base_image_id", "latest", "immutable"),
    ):
        with pytest.raises(module.BridgeError, match=message):
            call = {**kwargs, field: value} if field != "engine" else kwargs
            module._inspect_runtime_image(
                value if field == "engine" else "docker", **call
            )
    monkeypatch.setattr(
        module,
        "inspect_evaluator_image",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("bad image")),
    )
    with pytest.raises(module.BridgeError, match="did not authenticate"):
        module._inspect_runtime_image("docker", **kwargs)

    missing = tmp_path / "missing"
    with pytest.raises(module.BridgeError, match="could not be resolved"):
        module.mount_source(missing, label="missing")
    mount_link = tmp_path / "mount-link"
    regular = tmp_path / "mount-source"
    regular.write_text("x", encoding="utf-8")
    mount_link.symlink_to(regular)
    with pytest.raises(module.BridgeError, match="must not be a symlink"):
        module.mount_source(mount_link, label="link")


def test_hf_generator_and_compatibility_adapter_are_exercised_without_model_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    monkeypatch.setenv("INVARLOCK_EVALUATOR", "inspect-ai")

    class FakeTensor:
        shape = (1, 1)

        def to(self, _device: str) -> FakeTensor:
            return self

    class FakeTokenizer:
        pad_token_id = None
        eos_token_id = 2
        eos_token = "<eos>"

        def __call__(self, prompts: list[str], **_kwargs: object) -> dict[str, object]:
            assert prompts
            return {"input_ids": FakeTensor()}

        def decode(self, _tokens: object, **_kwargs: object) -> str:
            return "Answer\nignored"

    class FakeModel:
        def to(self, _device: str) -> FakeModel:
            return self

        def eval(self) -> FakeModel:
            return self

        def generate(self, **_kwargs: object) -> object:
            class Generated:
                def __getitem__(self, _key: object) -> list[list[int]]:
                    return [[1]]

            return Generated()

    class Inference:
        def __enter__(self) -> None:
            return None

        def __exit__(self, *_args: object) -> None:
            return None

    torch = types.ModuleType("torch")
    torch.float32 = object()
    torch.manual_seed = lambda _seed: None
    torch.set_num_threads = lambda _count: None
    torch.inference_mode = lambda: Inference()
    torch.cuda = SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None)
    transformers = types.ModuleType("transformers")
    transformers.AutoTokenizer = SimpleNamespace(
        from_pretrained=lambda *_a, **_k: FakeTokenizer()
    )
    transformers.AutoModelForCausalLM = SimpleNamespace(
        from_pretrained=lambda *_a, **_k: FakeModel()
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    generator = module.adapters._HfGreedyGenerator(tmp_path / "model")
    assert generator._tokenizer.pad_token == "<eos>"
    assert generator.generate(["Prompt"]) == ["Answer"]
    generator.close()

    class FakeGenerator:
        def __init__(self, _path: Path) -> None:
            pass

        def generate(self, prompts: list[str]) -> list[str]:
            return ["Answer" for _ in prompts]

        def close(self) -> None:
            pass

    monkeypatch.setattr(module.adapters, "_HfGreedyGenerator", FakeGenerator)
    generated = module.adapters._generate(tmp_path / "model", _records_bytes(module))
    assert len(generated) == 102
    assert generated[0]["output"] == "Answer"


def test_evaluator_transaction_runner_rejects_malformed_native_results_and_restores_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    record = {"id": "stable-1", "prompt": "Prompt", "expected": "Answer"}
    for data, message in (
        ({"expected": "Answer", "sampled": 1, "correct": True}, "non-text"),
        (
            {"expected": "Answer", "sampled": "Answer", "correct": "yes"},
            "invalid match",
        ),
        (None, "identity or target"),
    ):
        with pytest.raises(module.BridgeError, match=message):
            module.adapters._openai_event_to_sample(record, data)

    class FakeGenerator:
        def __init__(self, _path: Path) -> None:
            pass

        def close(self) -> None:
            pass

    class FakeMatch:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def get_samples(self) -> list[object]:
            return []

        def eval_all_samples(
            self, _recorder: object, _samples: object, **_kwargs: object
        ) -> None:
            return None

    class FakeRecorder:
        def __init__(self, *_a: object, **_k: object) -> None:
            pass

        def get_events(self, _name: str) -> list[object]:
            return []

    for name, value in {
        "evals.elsuite.basic.match": SimpleNamespace(Match=FakeMatch),
        "evals.record": SimpleNamespace(
            DummyRecorder=FakeRecorder, RunSpec=lambda **k: k
        ),
    }.items():
        monkeypatch.setitem(sys.modules, name, value)
    monkeypatch.setattr(module.adapters, "_HfGreedyGenerator", FakeGenerator)
    monkeypatch.setenv("EVALS_SEQUENTIAL", "previous")
    with pytest.raises(module.BridgeError, match="one match event"):
        module.adapters._run_openai_evals(Path("/model"), _records_bytes(module))
    assert module.os.environ["EVALS_SEQUENTIAL"] == "previous"


def test_evaluator_transaction_local_cli_and_worker_validation_errors_are_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    with pytest.raises(module.BridgeError, match="could not start"):
        module._run_local_cli([str(tmp_path / "does-not-exist")])
    with pytest.raises(module.BridgeError, match="failed"):
        module._run_local_cli(
            [sys.executable, "-c", "import sys; print('failed'); sys.exit(3)"]
        )

    monkeypatch.setenv("INVARLOCK_EVALUATOR", "inspect-ai")
    monkeypatch.setattr(module.importlib.metadata, "version", lambda _name: "wrong")
    with pytest.raises(module.BridgeError, match="must contain"):
        module.worker("baseline", tmp_path, tmp_path / "records", tmp_path / "output")
    monkeypatch.setattr(module.importlib.metadata, "version", lambda _name: "0.3.254")
    monkeypatch.setenv("INVARLOCK_RUNTIME_IMAGE_ID", "latest")
    with pytest.raises(module.BridgeError, match="runtime image digest"):
        module.worker("baseline", tmp_path, tmp_path / "records", tmp_path / "output")


def test_evaluator_transaction_adaptation_and_canonical_sample_boundaries(
    tmp_path: Path,
) -> None:
    module = _module()
    schedule = _schedule()
    valid = _sample(module)
    for raw, message in (
        (b"not-json\n", "not JSON"),
        (b'{ "record_id": "stable-1"}\n', "not canonical"),
    ):
        with pytest.raises(module.BridgeError, match=message):
            module.load_canonical_samples(raw, role="baseline")
    with pytest.raises(module.BridgeError, match="every schedule record"):
        module.adapt(b"", schedule, tmp_path / "records.jsonl")
    for field, value, message in (
        ("record_id", 1, "invalid text"),
        ("reported_score", True, "invalid evaluator score"),
        ("reported_score", 0.0, "disagrees with replay"),
        ("input_sha256", "bad", "authenticated inputs"),
    ):
        sample = dict(valid)
        sample[field] = value
        with pytest.raises(module.BridgeError, match=message):
            module.adapt(
                module.canonical_json_bytes(sample),
                schedule,
                tmp_path / "records.jsonl",
            )
    incomplete = dict(valid)
    incomplete.pop("score_detail")
    with pytest.raises(module.BridgeError, match="complete per-record"):
        module.adapt(
            module.canonical_json_bytes(incomplete),
            schedule,
            tmp_path / "records.jsonl",
        )


def test_evaluator_transaction_completion_output_and_path_guards(
    tmp_path: Path,
) -> None:
    module = _module()
    root = tmp_path / "transaction"
    root.mkdir()
    attestation = tmp_path / "attestation.json"
    trust_root = tmp_path / "trust-root"
    key = tmp_path / "key.pem"
    module._validate_completion_paths(
        root,
        build_attestation=attestation,
        trust_root=trust_root,
        key_paths=((key, "signing key"),),
    )
    with pytest.raises(module.BridgeError, match="attestation.*outside"):
        module._validate_completion_paths(
            root,
            build_attestation=root / "attestation.json",
            trust_root=trust_root,
            key_paths=((key, "signing key"),),
        )
    trust_root.mkdir()
    with pytest.raises(module.BridgeError, match="trust root must be new"):
        module._validate_completion_paths(
            root,
            build_attestation=attestation,
            trust_root=trust_root,
            key_paths=((key, "signing key"),),
        )

    evidence = root / "evidence/reports"
    evidence.mkdir(parents=True)
    receipt = root / "receipt.json"
    report = root / "report.html"
    (evidence / "evaluation.report.json").write_text(
        json.dumps({"verdict": "pass", "metric": "exact_match"}), encoding="utf-8"
    )
    receipt.write_text(
        json.dumps(
            {
                "statement": {
                    "verdict": {
                        "ok": True,
                        "integrity_ok": True,
                        "policy_verdict": "pass",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    report.write_text("<html>", encoding="utf-8")
    module.validate_completed_outputs(root / "evidence", receipt, report)
    report.unlink()
    with pytest.raises(module.BridgeError, match="passing result"):
        module.validate_completed_outputs(root / "evidence", receipt, report)


def test_evaluator_transaction_main_dispatches_and_reports_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _module()
    monkeypatch.setattr(module, "worker", lambda *_args: None)
    assert (
        module.main(
            [
                "worker",
                "--role",
                "baseline",
                "--model",
                str(tmp_path),
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
        "worker",
        lambda *_args: (_ for _ in ()).throw(module.BridgeError("bad worker")),
    )
    assert (
        module.main(
            [
                "worker",
                "--role",
                "baseline",
                "--model",
                str(tmp_path),
                "--dataset",
                str(tmp_path / "records.jsonl"),
                "--output",
                str(tmp_path / "output"),
            ]
        )
        == 2
    )
    assert "FAIL bad worker" in capsys.readouterr().err


def test_evaluator_transaction_launcher_entrypoint_covers_success_and_cleanup_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    launcher = _launcher_module()
    key = ed25519.Ed25519PrivateKey.generate()
    args = [
        "--workspace",
        str(tmp_path / "workspace"),
        "--evidence-signing-key",
        str(tmp_path / "evidence.pem"),
        "--verifier-signing-key",
        str(tmp_path / "verifier.pem"),
        "--trust-root",
        str(tmp_path / "trust-root"),
        "--builder-signing-key",
        str(tmp_path / "builder.pem"),
        "--builder-public-key",
        str(tmp_path / "builder-public.pem"),
    ]
    existing = tmp_path / "existing"
    existing.mkdir()
    existing_args = list(args)
    existing_args[1] = str(existing)
    assert launcher.main("inspect-ai", existing_args) == 2

    linked = tmp_path / "linked"
    linked.symlink_to(tmp_path / "missing-workspace", target_is_directory=True)
    linked_args = list(args)
    linked_args[1] = str(linked)
    assert launcher.main("inspect-ai", linked_args) == 2

    cleanup: list[list[str]] = []
    commands: list[list[str]] = []
    monkeypatch.setattr(launcher, "load_builder_signing_key", lambda _path: key)
    monkeypatch.setattr(
        launcher, "load_builder_public_key", lambda _path: key.public_key()
    )
    monkeypatch.setattr(launcher, "require_builder_key_pair", lambda *_args: None)
    monkeypatch.setattr(
        launcher,
        "_build_image",
        lambda _evaluator, _repository, _build, _engine, **kwargs: (
            kwargs["cleanup_tags"].append(
                launcher.OwnedImageTag("owned-tag", "sha256:" + "a" * 64)
            )
            or ("sha256:" + "a" * 64, "c" * 40, "sha256:" + "b" * 64)
        ),
    )
    monkeypatch.setattr(
        launcher,
        "run",
        lambda command, **_kwargs: (
            commands.append(command)
            or (
                "Evidence: complete"
                if "examples.integrations.evaluator_transaction.cli"
                in " ".join(command)
                else ""
            )
        ),
    )
    monkeypatch.setattr(
        launcher,
        "remove_owned_image_tags",
        lambda _run, _engine, _repo, tags: cleanup.append(list(tags)),
    )
    result = launcher.main("inspect-ai", ["--container-engine", "podman", *args])
    assert result == 0
    assert cleanup == [[launcher.OwnedImageTag("owned-tag", "sha256:" + "a" * 64)]]
    assert any("model_inputs.py" in " ".join(command) for command in commands)
    assert "Complete integration workspace" in capsys.readouterr().out


def test_evaluator_transaction_launcher_reports_build_and_cleanup_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launcher = _launcher_module()
    key = ed25519.Ed25519PrivateKey.generate()
    common = [
        "--workspace",
        str(tmp_path / "workspace"),
        "--evidence-signing-key",
        str(tmp_path / "evidence.pem"),
        "--verifier-signing-key",
        str(tmp_path / "verifier.pem"),
        "--trust-root",
        str(tmp_path / "trust-root"),
        "--builder-signing-key",
        str(tmp_path / "builder.pem"),
        "--builder-public-key",
        str(tmp_path / "builder-public.pem"),
    ]
    monkeypatch.setattr(launcher, "load_builder_signing_key", lambda _path: key)
    monkeypatch.setattr(
        launcher, "load_builder_public_key", lambda _path: key.public_key()
    )
    monkeypatch.setattr(launcher, "require_builder_key_pair", lambda *_args: None)
    monkeypatch.setattr(
        launcher,
        "_build_image",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("build failed")),
    )
    monkeypatch.setattr(launcher, "remove_owned_image_tags", lambda *_args: None)
    assert launcher.main("inspect-ai", common) == 2

    workspace = tmp_path / "cleanup-workspace"
    monkeypatch.setattr(
        launcher,
        "_build_image",
        lambda *_args, **_kwargs: (
            "sha256:" + "a" * 64,
            "c" * 40,
            "sha256:" + "b" * 64,
        ),
    )
    monkeypatch.setattr(
        launcher,
        "run",
        lambda command, **_kwargs: (
            "done"
            if "examples.integrations.evaluator_transaction.cli" in " ".join(command)
            else ""
        ),
    )
    monkeypatch.setattr(
        launcher,
        "remove_owned_image_tags",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("cleanup failed")),
    )
    cleanup_args = list(common)
    cleanup_args[1] = str(workspace)
    assert launcher.main("inspect-ai", cleanup_args) == 2


def test_evaluator_transaction_launcher_command_and_inspection_helpers_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launcher = _launcher_module()
    assert launcher.run([sys.executable, "-c", "print('ok')"], cwd=tmp_path) == "ok"
    source = tmp_path / "source"
    source.write_bytes(b"input")
    assert (
        launcher.run(
            [
                sys.executable,
                "-c",
                "import sys; print(sys.stdin.buffer.read().decode())",
            ],
            cwd=tmp_path,
            stdin_path=source,
        )
        == "input"
    )
    with pytest.raises(RuntimeError, match="could not start"):
        launcher.run([str(tmp_path / "missing")], cwd=tmp_path)
    with pytest.raises(RuntimeError, match="status 3"):
        launcher.run([sys.executable, "-c", "import sys; sys.exit(3)"], cwd=tmp_path)
    with pytest.raises(ValueError, match="OCI mount"):
        launcher.mount_source(Path("bad,path"))

    monkeypatch.setattr(launcher, "run", lambda *_args, **_kwargs: "not-json")
    with pytest.raises(RuntimeError, match="not JSON"):
        launcher._image_layers("docker", "image", tmp_path)
    monkeypatch.setattr(launcher, "run", lambda *_args, **_kwargs: "{}")
    with pytest.raises(RuntimeError, match="inspection is invalid"):
        launcher._image_layers("docker", "image", tmp_path)


def test_evaluator_transaction_native_runner_and_worker_failure_boundaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    monkeypatch.setenv("INVARLOCK_EVALUATOR", "inspect-ai")
    records = [json.loads(line) for line in _records_bytes(module).splitlines()]

    class Sample:
        def __init__(
            self, item: dict[str, str], *, completion: object = "Answer"
        ) -> None:
            self.id = item["id"]
            self.input = item["prompt"]
            self.target = item["expected"]
            self.output = SimpleNamespace(completion=completion)
            self.scores = {
                "match": SimpleNamespace(
                    value="C", answer="Answer", explanation="exact"
                )
            }

    inspect_ai = types.ModuleType("inspect_ai")
    inspect_ai.Task = lambda **kwargs: SimpleNamespace(**kwargs)
    dataset = types.ModuleType("inspect_ai.dataset")
    dataset.MemoryDataset = lambda values: values
    dataset.Sample = lambda **kwargs: SimpleNamespace(**kwargs)
    scorer = types.ModuleType("inspect_ai.scorer")
    scorer.match = lambda **kwargs: kwargs
    solver = types.ModuleType("inspect_ai.solver")
    solver.generate = lambda: "generate"
    monkeypatch.setitem(sys.modules, "inspect_ai", inspect_ai)
    monkeypatch.setitem(sys.modules, "inspect_ai.dataset", dataset)
    monkeypatch.setitem(sys.modules, "inspect_ai.scorer", scorer)
    monkeypatch.setitem(sys.modules, "inspect_ai.solver", solver)

    logs: list[object] = []
    inspect_ai.eval = lambda *_args, **_kwargs: logs  # type: ignore[attr-defined]

    def run_logs(samples: object, status: str = "success") -> None:
        logs[:] = [SimpleNamespace(status=status, samples=samples)]
        module.adapters._run_inspect_ai(Path("/model"), _records_bytes(module))

    with pytest.raises(module.BridgeError, match="one successful sample log"):
        run_logs(None, status="failed")
    with pytest.raises(module.BridgeError, match="identity or target"):
        run_logs([Sample(records[0], completion="Answer")])
    with pytest.raises(module.BridgeError, match="non-text completion"):
        run_logs([Sample(item, completion=1) for item in records])
    no_scores = [Sample(item) for item in records]
    for sample in no_scores:
        sample.scores = {}
    with pytest.raises(module.BridgeError, match="match score"):
        run_logs(no_scores)

    class FakeGenerator:
        def __init__(self, _path: Path) -> None:
            pass

        def close(self) -> None:
            pass

    class FakeRecorder:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            self.events: list[object] = []

        def get_events(self, _name: str) -> list[object]:
            return self.events

    class FakeMatch:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def get_samples(self) -> list[dict[str, str]]:
            return [
                {"input": item["prompt"], "ideal": item["expected"]} for item in records
            ]

        def eval_all_samples(
            self, recorder: FakeRecorder, _samples: object, **_kwargs: object
        ) -> None:
            recorder.events.extend(
                SimpleNamespace(sample_id="sample.0", data={"sample_id": "sample.0"})
                for _ in records
            )

    evals = types.ModuleType("evals")
    elsuite = types.ModuleType("evals.elsuite")
    basic = types.ModuleType("evals.elsuite.basic")
    match = types.ModuleType("evals.elsuite.basic.match")
    match.Match = FakeMatch
    record = types.ModuleType("evals.record")
    record.DummyRecorder = FakeRecorder
    record.RunSpec = lambda **kwargs: kwargs
    for name, value in {
        "evals": evals,
        "evals.elsuite": elsuite,
        "evals.elsuite.basic": basic,
        "evals.elsuite.basic.match": match,
        "evals.record": record,
    }.items():
        monkeypatch.setitem(sys.modules, name, value)
    monkeypatch.setattr(module.adapters, "_HfGreedyGenerator", FakeGenerator)
    with pytest.raises(module.BridgeError, match="ambiguous"):
        module.adapters._run_openai_evals(Path("/model"), _records_bytes(module))

    assert module._run_local_cli([sys.executable, "-c", "print('ok')"]) == "ok\n"
    mount_target = tmp_path / "mount-target"
    mount_target.write_text("x", encoding="utf-8")
    mount_link = tmp_path / "mount-link"
    mount_link.symlink_to(mount_target)
    with pytest.raises(module.BridgeError, match="must not be a symlink"):
        module.mount_source(mount_link, label="mount")


def test_evaluator_transaction_worker_and_main_complete_guards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _module()
    monkeypatch.setenv("INVARLOCK_EVALUATOR", "inspect-ai")
    monkeypatch.setenv("INVARLOCK_RUNTIME_IMAGE_ID", "sha256:" + "a" * 64)
    monkeypatch.setattr(
        module, "evaluator_lock_digest", lambda *_args, **_kwargs: "sha256:" + "b" * 64
    )
    monkeypatch.setattr(module.importlib.metadata, "version", lambda _name: "0.3.254")
    monkeypatch.setenv("INVARLOCK_EVALUATOR_LOCK_SHA256", "wrong")
    with pytest.raises(module.BridgeError, match="lock"):
        module.worker("baseline", tmp_path, tmp_path / "records", tmp_path / "output")

    with pytest.raises(module.BridgeError, match="engine"):
        module._run_verified_worker(
            engine="runc",
            image="sha256:" + "a" * 64,
            selected="inspect-ai",
            role="baseline",
            prepared=tmp_path,
            output=tmp_path / "output",
            lock_digest="sha256:" + "b" * 64,
        )
    with pytest.raises(module.BridgeError, match="role"):
        module._run_verified_worker(
            engine="docker",
            image="sha256:" + "a" * 64,
            selected="inspect-ai",
            role="other",
            prepared=tmp_path,
            output=tmp_path / "output",
            lock_digest="sha256:" + "b" * 64,
        )
    with pytest.raises(module.BridgeError, match="missing or unsafe"):
        module._run_verified_worker(
            engine="docker",
            image="sha256:" + "a" * 64,
            selected="inspect-ai",
            role="baseline",
            prepared=tmp_path,
            output=tmp_path / "output",
            lock_digest="sha256:" + "b" * 64,
        )

    monkeypatch.setattr(
        module,
        "complete",
        lambda *_args, **_kwargs: (
            tmp_path / "evidence",
            tmp_path / "receipt",
            tmp_path / "report",
        ),
    )
    arguments = [
        "complete",
        "--workspace",
        str(tmp_path / "transaction"),
        "--prepared",
        str(tmp_path / "prepared"),
        "--runtime-image",
        "sha256:" + "a" * 64,
        "--evaluator",
        "inspect-ai",
        "--evidence-signing-key",
        str(tmp_path / "evidence.pem"),
        "--verifier-signing-key",
        str(tmp_path / "verifier.pem"),
        "--trust-root",
        str(tmp_path / "trust-root"),
        "--builder-public-key",
        str(tmp_path / "builder-public.pem"),
        "--source-commit",
        "c" * 40,
        "--base-image-id",
        "sha256:" + "b" * 64,
        "--build-attestation",
        str(tmp_path / "attestation.json"),
    ]
    assert module.main(arguments) == 0
    assert "Evidence:" in capsys.readouterr().out


def test_evaluator_transaction_remaining_artifact_and_worker_guards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    private = ed25519.Ed25519PrivateKey.generate()
    private_path = tmp_path / "private.pem"
    private_path.write_bytes(
        private.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    public_path = tmp_path / "public.pem"
    public_path.write_bytes(
        private.public_key().public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    with pytest.raises(module.BridgeError, match="private key"):
        module._external_ed25519_key(public_path, label="private")
    with pytest.raises(module.BridgeError, match="public key"):
        module._external_ed25519_public_key(private_path, label="public")

    real_fstat = module.os.fstat
    changed = tmp_path / "changed"
    changed.write_bytes(b"x")
    descriptor = module.os.open(changed, module.os.O_RDONLY)
    facts = module.os.fstat(descriptor)
    module.os.close(descriptor)
    calls = iter(
        [
            facts,
            SimpleNamespace(
                st_mode=facts.st_mode,
                st_ino=facts.st_ino,
                st_dev=facts.st_dev,
                st_size=facts.st_size,
                st_mtime_ns=facts.st_mtime_ns,
                st_ctime_ns=facts.st_ctime_ns + 1,
            ),
        ]
    )
    monkeypatch.setattr(module.os, "fstat", lambda _fd: next(calls))
    with pytest.raises(module.BridgeError, match="changed while"):
        module._read_regular_file(changed, label="changed")
    monkeypatch.setattr(module.os, "fstat", real_fstat)

    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(module.BridgeError, match="lacks the Hugging Face"):
        module.adapters._HfGreedyGenerator(tmp_path / "model")

    class ShortGenerator:
        def __init__(self, _path: Path) -> None:
            pass

        def generate(self, _prompts: list[str]) -> list[str]:
            return []

        def close(self) -> None:
            pass

    monkeypatch.setattr(module.adapters, "_HfGreedyGenerator", ShortGenerator)
    with pytest.raises(module.BridgeError, match="incomplete result"):
        module.adapters._generate(tmp_path / "model", _records_bytes(module))

    with pytest.raises(module.BridgeError, match="inspected image"):
        module.complete(
            tmp_path / "transaction", tmp_path / "prepared", "latest", "inspect-ai"
        )
    with pytest.raises(module.BridgeError, match="caller-owned"):
        module.complete(
            tmp_path / "transaction-2",
            tmp_path / "prepared",
            "sha256:" + "a" * 64,
            "inspect-ai",
        )
    real_prepared = tmp_path / "real-prepared"
    real_prepared.mkdir()
    linked_prepared = tmp_path / "linked-prepared"
    linked_prepared.symlink_to(real_prepared, target_is_directory=True)
    with pytest.raises(module.BridgeError, match="prepared workspace"):
        module.complete(
            tmp_path / "transaction-3",
            linked_prepared,
            "sha256:" + "a" * 64,
            "inspect-ai",
            evidence_signing_key=tmp_path / "evidence.pem",
            verifier_signing_key=tmp_path / "verifier.pem",
            trust_root=tmp_path / "trust-root",
            source_commit="c" * 40,
            base_image_id="sha256:" + "b" * 64,
            build_attestation=tmp_path / "attestation.json",
            builder_public_key=tmp_path / "builder-public.pem",
        )

    prepared = tmp_path / "prepared-workers"
    (prepared / "evaluation/models/baseline").mkdir(parents=True)
    (prepared / "evaluation/inputs").mkdir(parents=True)
    dataset = prepared / "evaluation/inputs/records.jsonl"
    dataset.write_text("{}\n", encoding="utf-8")
    with pytest.raises(module.BridgeError, match="must be new"):
        output = tmp_path / "existing-output"
        output.mkdir()
        module._run_verified_worker(
            engine="docker",
            image="sha256:" + "a" * 64,
            selected="inspect-ai",
            role="baseline",
            prepared=prepared,
            output=output,
            lock_digest="sha256:" + "b" * 64,
        )

    class InvalidOutput:
        name = ".."
        parent = SimpleNamespace(mkdir=lambda **_kwargs: None)

        def exists(self) -> bool:
            return False

        def is_symlink(self) -> bool:
            return False

    with pytest.raises(module.BridgeError, match="output name"):
        module._run_verified_worker(
            engine="docker",
            image="sha256:" + "a" * 64,
            selected="inspect-ai",
            role="baseline",
            prepared=prepared,
            output=InvalidOutput(),  # type: ignore[arg-type]
            lock_digest="sha256:" + "b" * 64,
        )


def test_evaluator_transaction_lock_and_worker_contract_mismatch_guards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    monkeypatch.setattr(module, "_read_regular_file", lambda *_args, **_kwargs: b"lock")
    assert module.evaluator_lock_digest("inspect-ai", container=True) == (
        "sha256:" + module.digest(b"lock")
    )

    model = tmp_path / "model"
    model.mkdir()
    dataset = tmp_path / "records.jsonl"
    dataset.write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("INVARLOCK_EVALUATOR", "inspect-ai")
    monkeypatch.setenv("INVARLOCK_RUNTIME_IMAGE_ID", "sha256:" + "a" * 64)
    monkeypatch.setenv("INVARLOCK_EVALUATOR_LOCK_SHA256", "sha256:" + "b" * 64)
    monkeypatch.setattr(module.importlib.metadata, "version", lambda _name: "0.3.254")
    monkeypatch.setattr(
        module, "evaluator_lock_digest", lambda *_args, **_kwargs: "sha256:" + "b" * 64
    )
    monkeypatch.setattr(
        module, "checkpoint_tree_sha256", lambda _path: "sha256:" + "c" * 64
    )
    monkeypatch.setattr(
        module.adapters, "_run_upstream_evaluator", lambda *_args: ([], [(1.0, {})])
    )
    with pytest.raises(module.BridgeError, match="one result per record"):
        module.worker("baseline", model, dataset, tmp_path / "output")

    generation = model / "generation_config.json"
    generation.write_text("{}", encoding="utf-8")
    fake_torch = types.ModuleType("torch")
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoModelForCausalLM = object()
    fake_transformers.AutoTokenizer = object()
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    with pytest.raises(module.BridgeError, match="generation defaults"):
        module.adapters._HfGreedyGenerator(model)
