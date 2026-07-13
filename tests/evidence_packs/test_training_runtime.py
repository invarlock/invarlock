from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
from typing import Any

import pytest
import torch

from scripts.evidence_packs.python.editing import training_artifact_verifier as verifier
from scripts.evidence_packs.python.editing import training_runtime as runtime
from scripts.evidence_packs.python.editing import training_runtime_provider as provider
from scripts.evidence_packs.python.editing.training_contract import (
    LoraTrainingProfile,
    file_sha256,
    load_training_profile,
)
from scripts.evidence_packs.python.editing.training_receipt import (
    require_valid_training_receipt,
    with_receipt_digest,
)
from tests.evidence_packs._support_training_runtime import (
    FakeAutoModel,
    FakeAutoTokenizer,
    FakeLoraConfig,
    FakePeftModel,
    NoOpOptimizer,
    RecordingAdamW,
    TinyCausalLM,
    TinyTokenizer,
    reset_training_fakes,
)
from tests.evidence_packs._support_training_runtime import (
    fake_peft_dependencies as _fake_peft_dependencies,
)


@pytest.fixture(autouse=True)
def reset_fakes() -> None:
    reset_training_fakes()


@pytest.fixture
def fake_runtime(monkeypatch: pytest.MonkeyPatch) -> runtime.RuntimeDependencies:
    dependencies = runtime.RuntimeDependencies(
        torch=torch,
        auto_model=FakeAutoModel,
        auto_tokenizer=FakeAutoTokenizer,
        optimizer_cls=RecordingAdamW,
        transformers_version="5.12.0",
    )
    monkeypatch.setattr(runtime, "_load_runtime_dependencies", lambda: dependencies)
    return dependencies


def test_local_training_enforces_offline_mode_for_entire_lifecycle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed: dict[str, str | None] = {}
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)

    def execute(*_args, **_kwargs):
        observed.update(
            {
                "hub": os.environ.get("HF_HUB_OFFLINE"),
                "transformers": os.environ.get("TRANSFORMERS_OFFLINE"),
            }
        )
        return object()

    monkeypatch.setattr(runtime, "_run_training_profile", execute)
    result = runtime.run_training_profile(
        load_training_profile("tiny_gpt2_full_ft_v1"),
        tmp_path / "subject",
        local_files_only=True,
    )

    assert result is not None
    assert observed == {"hub": "1", "transformers": "1"}
    assert os.environ.get("HF_HUB_OFFLINE") is None
    assert os.environ.get("TRANSFORMERS_OFFLINE") is None


@pytest.mark.parametrize(
    ("payload", "message"),
    (
        ([], "binding is malformed"),
        (
            {"provider": {}, "provider_sha256": "sha256:" + "0" * 64},
            "coordinates are missing",
        ),
        (
            {"provider": {"kind": "fixture"}, "provider_sha256": "sha256:" + "0" * 64},
            "digest mismatch",
        ),
    ),
)
def test_acceptance_dataset_provider_binding_rejects_malformed_or_forged_inputs(
    monkeypatch: pytest.MonkeyPatch, payload: object, message: str
) -> None:
    monkeypatch.setenv(
        "INVARLOCK_ACCEPTANCE_DATASET_PROVIDER_SNAPSHOT_JSON", json.dumps(payload)
    )

    with pytest.raises(runtime.TrainingRuntimeError, match=message):
        provider.dataset_provider_binding(load_training_profile("tiny_gpt2_full_ft_v1"))


def test_directory_identity_rejects_non_directory_publication_target(
    tmp_path: Path,
) -> None:
    target = tmp_path / "not-a-directory"
    target.write_text("not a directory", encoding="utf-8")

    with pytest.raises(runtime.TrainingRuntimeError, match="non-symlink directory"):
        runtime._directory_identity(target, label="publication target")


def test_full_fine_tune_runs_optimizer_saves_reloads_and_validates_receipt(
    tmp_path: Path, fake_runtime: runtime.RuntimeDependencies
) -> None:
    del fake_runtime
    profile = load_training_profile("tiny_gpt2_full_ft_v1")
    output = tmp_path / "subject"

    result = runtime.run_training_profile(profile, output, local_files_only=True)

    receipt = json.loads(result.receipt_path.read_text())
    assert result.subject_dir == output
    assert require_valid_training_receipt(receipt, profile=profile) == receipt
    # Publication, artifact replay, and the independent optimizer rerun all
    # consume the same bounded training profile before acceptance.
    assert RecordingAdamW.completed_steps == 2 * profile.steps
    assert RecordingAdamW.constructions == 2 * [
        {
            "lr": profile.optimizer.learning_rate,
            "betas": profile.optimizer.betas,
            "eps": profile.optimizer.eps,
            "weight_decay": profile.optimizer.weight_decay,
        }
    ]
    assert receipt["training"]["completed_steps"] == profile.steps
    assert len(receipt["training"]["losses"]) == profile.steps
    assert receipt["changes"]["changed_tensors"] == 3
    assert (
        receipt["hashes"]["post_training_state_sha256"]
        == receipt["hashes"]["reloaded_subject_state_sha256"]
    )
    assert (
        runtime.directory_sha256(output, exclude=frozenset({"training_receipt.json"}))
        == receipt["hashes"]["subject_tree_sha256"]
    )
    assert (
        len(TinyTokenizer.calls)
        == 3 * profile.steps * profile.gradient_accumulation_steps
    )
    assert all(
        len(call["texts"]) == profile.micro_batch_size for call in TinyTokenizer.calls
    )
    assert FakeAutoModel.source_calls[0][1]["revision"] == profile.model_revision
    assert FakeAutoTokenizer.source_calls[0][1]["revision"] == profile.model_revision


def test_profile_pinned_toolchain_mismatch_cannot_train(
    tmp_path: Path,
    fake_runtime: runtime.RuntimeDependencies,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mismatched = runtime.RuntimeDependencies(
        **{**fake_runtime.__dict__, "transformers_version": "5.13.0"}
    )
    monkeypatch.setattr(runtime, "_load_runtime_dependencies", lambda: mismatched)
    output = tmp_path / "toolchain-mismatch"

    with pytest.raises(
        runtime.TrainingRuntimeError,
        match="toolchain does not match the immutable profile",
    ):
        runtime.run_training_profile(
            load_training_profile("tiny_gpt2_full_ft_v1"),
            output,
            local_files_only=True,
        )

    assert not output.exists()


def test_lora_trains_only_adapters_serializes_merges_and_reloads(
    tmp_path: Path,
    fake_runtime: runtime.RuntimeDependencies,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del fake_runtime
    monkeypatch.setattr(runtime, "_load_peft_dependencies", _fake_peft_dependencies)
    profile = load_training_profile("tiny_gpt2_lora_v1")
    assert isinstance(profile, LoraTrainingProfile)

    result = runtime.run_training_profile(
        profile, tmp_path / "lora", local_files_only=True
    )

    receipt = result.receipt
    assert require_valid_training_receipt(receipt, profile=profile) == receipt
    assert (
        receipt["lora"]["initial_adapter_state_sha256"]
        != receipt["lora"]["trained_adapter_state_sha256"]
    )
    assert (
        receipt["lora"]["serialized_adapter_state_sha256"]
        == receipt["lora"]["trained_adapter_state_sha256"]
    )
    assert (
        receipt["lora"]["base_state_before_adapter_sha256"]
        == receipt["lora"]["base_state_after_training_sha256"]
    )
    assert receipt["lora"]["adapter_modules_before_merge"] == 1
    assert receipt["lora"]["adapter_modules_after_merge"] == 0
    assert (
        receipt["lora"]["merged_state_sha256"]
        != receipt["hashes"]["baseline_state_sha256"]
    )
    assert (result.subject_dir / "adapter" / "adapter_model.pt").is_file()
    assert FakeLoraConfig.last_options["target_modules"] == list(
        profile.lora.target_modules
    )
    assert RecordingAdamW.completed_steps == 2 * profile.steps


@pytest.mark.parametrize(
    "profile_id",
    ["tiny_gpt2_full_ft_v1", "tiny_gpt2_lora_v1"],
)
def test_independent_artifact_verifier_recomputes_published_evidence(
    profile_id: str,
    tmp_path: Path,
    fake_runtime: runtime.RuntimeDependencies,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del fake_runtime
    if profile_id.endswith("lora_v1"):
        monkeypatch.setattr(runtime, "_load_peft_dependencies", _fake_peft_dependencies)
    profile = load_training_profile(profile_id)
    result = runtime.run_training_profile(
        profile, tmp_path / profile_id, local_files_only=True
    )

    verified = runtime.verify_training_artifact(
        profile, result.subject_dir, local_files_only=True
    )

    assert verified == result.receipt


def test_artifact_verifier_rejects_structurally_valid_fabricated_hash(
    tmp_path: Path,
    fake_runtime: runtime.RuntimeDependencies,
) -> None:
    del fake_runtime
    profile = load_training_profile("tiny_gpt2_full_ft_v1")
    result = runtime.run_training_profile(
        profile, tmp_path / "fine-tune", local_files_only=True
    )
    receipt = dict(result.receipt)
    receipt["hashes"] = dict(receipt["hashes"])
    fabricated = "sha256:" + "1" * 64
    receipt["hashes"]["baseline_state_sha256"] = fabricated
    receipt["hashes"]["pre_training_state_sha256"] = fabricated
    receipt = with_receipt_digest(receipt)
    result.receipt_path.write_text(json.dumps(receipt, sort_keys=True) + "\n")
    assert require_valid_training_receipt(receipt, profile=profile) == receipt

    with pytest.raises(runtime.TrainingRuntimeError, match="baseline model state"):
        runtime.verify_training_artifact(
            profile, result.subject_dir, local_files_only=True
        )


def test_artifact_verifier_rejects_tampered_reload_inference_digest(
    tmp_path: Path,
    fake_runtime: runtime.RuntimeDependencies,
) -> None:
    del fake_runtime
    profile = load_training_profile("tiny_gpt2_full_ft_v1")
    result = runtime.run_training_profile(
        profile, tmp_path / "fine-tune", local_files_only=True
    )
    receipt = dict(result.receipt)
    receipt["reload_smoke"] = dict(receipt["reload_smoke"])
    receipt["reload_smoke"]["logits_sha256"] = "sha256:" + "1" * 64
    receipt = with_receipt_digest(receipt)
    result.receipt_path.write_text(json.dumps(receipt, sort_keys=True) + "\n")
    assert require_valid_training_receipt(receipt, profile=profile) == receipt

    with pytest.raises(
        runtime.TrainingRuntimeError, match="inference evidence mismatch"
    ):
        runtime.verify_training_artifact(
            profile, result.subject_dir, local_files_only=True
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("lora_dropout", 0.5),
        ("use_dora", True),
        ("modules_to_save", ["lm_head"]),
        ("rank_pattern", {"projection": 1}),
    ],
)
def test_artifact_verifier_rejects_rebound_serialized_lora_config(
    field: str,
    value: Any,
    tmp_path: Path,
    fake_runtime: runtime.RuntimeDependencies,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del fake_runtime
    monkeypatch.setattr(runtime, "_load_peft_dependencies", _fake_peft_dependencies)
    profile = load_training_profile("tiny_gpt2_lora_v1")
    result = runtime.run_training_profile(
        profile, tmp_path / "lora", local_files_only=True
    )
    config_path = result.subject_dir / "adapter" / "adapter_config.json"
    config = json.loads(config_path.read_text())
    config[field] = value
    config_path.write_text(json.dumps(config, sort_keys=True) + "\n")
    receipt = json.loads(result.receipt_path.read_text())
    receipt["lora"]["serialized_adapter_config_sha256"] = file_sha256(config_path)
    receipt["lora"]["adapter_tree_sha256"] = runtime.directory_sha256(
        result.subject_dir / "adapter"
    )
    receipt["hashes"]["subject_tree_sha256"] = runtime.directory_sha256(
        result.subject_dir, exclude=frozenset({"training_receipt.json"})
    )
    receipt = with_receipt_digest(receipt)
    result.receipt_path.write_text(json.dumps(receipt, sort_keys=True) + "\n")

    with pytest.raises(
        runtime.TrainingRuntimeError,
        match="serialized LoRA configuration does not match",
    ):
        runtime.verify_training_artifact(
            profile, result.subject_dir, local_files_only=True
        )


def test_atomic_publication_never_replaces_concurrent_empty_directory(
    tmp_path: Path,
) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "subject.txt").write_text("trained")
    output = tmp_path / "subject"
    output.mkdir()
    (output / "owner.txt").write_text("concurrent")

    with pytest.raises(runtime.TrainingRuntimeError, match="refusing to replace"):
        runtime._publish_directory_no_replace(staging, output)

    assert (output / "owner.txt").read_text() == "concurrent"
    assert (staging / "subject.txt").read_text() == "trained"


def test_peft_adapter_construction_cannot_mutate_pristine_base(
    tmp_path: Path,
    fake_runtime: runtime.RuntimeDependencies,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del fake_runtime
    dependencies = _fake_peft_dependencies()

    def mutating_get_peft_model(
        model: TinyCausalLM, config: FakeLoraConfig
    ) -> FakePeftModel:
        with torch.no_grad():
            model.projection.weight.add_(1.0)
        return FakePeftModel(model, config)

    monkeypatch.setattr(
        runtime,
        "_load_peft_dependencies",
        lambda: runtime.PeftDependencies(
            **{**dependencies.__dict__, "get_peft_model": mutating_get_peft_model}
        ),
    )
    output = tmp_path / "mutated-on-adapter-construction"

    with pytest.raises(
        runtime.TrainingRuntimeError, match="construction mutated the pristine base"
    ):
        runtime.run_training_profile(
            load_training_profile("tiny_gpt2_lora_v1"),
            output,
            local_files_only=True,
        )

    assert not output.exists()


def test_stale_serialized_adapter_cannot_publish(
    tmp_path: Path,
    fake_runtime: runtime.RuntimeDependencies,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del fake_runtime
    dependencies = _fake_peft_dependencies()

    class StaleSerializedPeftModel(FakePeftModel):
        @classmethod
        def from_pretrained(
            cls,
            base_model: TinyCausalLM,
            path: Path,
            *,
            is_trainable: bool,
            local_files_only: bool,
            config: FakeLoraConfig | None = None,
        ) -> FakePeftModel:
            model = super().from_pretrained(
                base_model,
                path,
                is_trainable=is_trainable,
                local_files_only=local_files_only,
                config=config,
            )
            with torch.no_grad():
                model.lora_A.zero_()
                model.lora_B.zero_()
            return model

    monkeypatch.setattr(
        runtime,
        "_load_peft_dependencies",
        lambda: runtime.PeftDependencies(
            **{**dependencies.__dict__, "peft_model_cls": StaleSerializedPeftModel}
        ),
    )
    output = tmp_path / "stale-adapter"

    with pytest.raises(
        runtime.TrainingRuntimeError,
        match="serialized LoRA adapter state does not match",
    ):
        runtime.run_training_profile(
            load_training_profile("tiny_gpt2_lora_v1"),
            output,
            local_files_only=True,
        )

    assert not output.exists()


def test_noop_optimizer_cannot_publish_flag_only_evidence(
    tmp_path: Path,
    fake_runtime: runtime.RuntimeDependencies,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = load_training_profile("tiny_gpt2_full_ft_v1")
    dependencies = runtime.RuntimeDependencies(
        **{**fake_runtime.__dict__, "optimizer_cls": NoOpOptimizer}
    )
    monkeypatch.setattr(runtime, "_load_runtime_dependencies", lambda: dependencies)
    output = tmp_path / "unchanged"

    with pytest.raises(
        runtime.TrainingRuntimeError, match="did not change model tensors"
    ):
        runtime.run_training_profile(profile, output, local_files_only=True)

    assert not output.exists()


def test_reload_mismatch_cannot_publish_subject(
    tmp_path: Path, fake_runtime: runtime.RuntimeDependencies
) -> None:
    del fake_runtime
    FakeAutoModel.reload_baseline = True
    profile = load_training_profile("tiny_gpt2_full_ft_v1")
    output = tmp_path / "bad-reload"

    with pytest.raises(runtime.TrainingRuntimeError, match="reload state-hash"):
        runtime.run_training_profile(profile, output, local_files_only=True)

    assert not output.exists()


def test_receipt_validation_failure_prevents_publication(
    tmp_path: Path,
    fake_runtime: runtime.RuntimeDependencies,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del fake_runtime
    profile = load_training_profile("tiny_gpt2_full_ft_v1")
    output = tmp_path / "invalid-receipt"

    def reject(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise ValueError("receipt rejected")

    monkeypatch.setattr(runtime, "validate_training_receipt", reject)
    with pytest.raises(ValueError, match="receipt rejected"):
        runtime.run_training_profile(profile, output, local_files_only=True)

    assert not output.exists()


def test_peft_dependency_error_is_actionable(monkeypatch: pytest.MonkeyPatch) -> None:
    original = importlib.import_module

    def import_without_peft(name: str, package: str | None = None) -> Any:
        if name == "peft":
            raise ImportError("missing")
        return original(name, package)

    monkeypatch.setattr(runtime.importlib, "import_module", import_without_peft)
    with pytest.raises(runtime.TrainingRuntimeError, match="optional `peft` package"):
        runtime._load_peft_dependencies()


def test_peft_runtime_rejects_missing_pinned_state_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = importlib.import_module
    imports: list[str] = []

    def import_without_pinned_api(name: str, package: str | None = None) -> Any:
        imports.append(name)
        if name == "peft":
            return type("Peft", (), {"__version__": "0.19.1"})()
        if name == "peft.utils.save_and_load":
            return type(
                "LegacyPeftStateShim",
                (),
                {"get_peft_model_state_dict": lambda *_args, **_kwargs: {}},
            )()
        return original(name, package)

    monkeypatch.setattr(runtime.importlib, "import_module", import_without_pinned_api)
    with pytest.raises(
        runtime.TrainingRuntimeError,
        match="pinned PEFT runtime lacks get_peft_model_state_dict",
    ):
        runtime._load_peft_dependencies()
    assert imports == ["peft"]


@pytest.mark.integration
@pytest.mark.parametrize(
    "profile_id",
    ["tiny_gpt2_full_ft_v1", "tiny_gpt2_lora_v1"],
)
def test_cached_tiny_gpt2_real_training_runtime_smoke(
    profile_id: str,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    required = os.environ.get("INVARLOCK_REQUIRE_REAL_TRAINING") == "1"
    if profile_id.endswith("lora_v1"):
        if required:
            try:
                importlib.import_module("peft")
            except Exception as exc:
                pytest.fail(f"required PEFT training dependency is unusable: {exc}")
        else:
            pytest.importorskip("peft")
    if required:
        try:
            transformers = importlib.import_module("transformers")
        except Exception as exc:
            pytest.fail(f"required Transformers dependency is unusable: {exc}")
    else:
        transformers = pytest.importorskip("transformers")
    profile = load_training_profile(profile_id)
    local_files_only = not required
    try:
        transformers.AutoTokenizer.from_pretrained(
            profile.model_id,
            revision=profile.model_revision,
            local_files_only=local_files_only,
            trust_remote_code=False,
        )
        transformers.AutoModelForCausalLM.from_pretrained(
            profile.model_id,
            revision=profile.model_revision,
            local_files_only=local_files_only,
            trust_remote_code=False,
        )
    except OSError:
        if required:
            pytest.fail("required pinned tiny-gpt2 training model is unavailable")
        pytest.skip("pinned tiny-gpt2 revision is not cached")

    proofs: list[dict[str, object]] = []
    original_execution_proof = verifier._independent_optimizer_execution_proof

    def record_execution_proof(*args: object, **kwargs: object) -> dict[str, object]:
        proof = original_execution_proof(*args, **kwargs)
        proofs.append(proof)
        return proof

    monkeypatch.setattr(
        verifier, "_independent_optimizer_execution_proof", record_execution_proof
    )
    result = runtime.run_training_profile(
        profile, tmp_path / profile_id, local_files_only=local_files_only
    )

    assert require_valid_training_receipt(result.receipt, profile=profile)
    assert proofs == [
        {
            "schema": "invarlock/independent-optimizer-execution-proof-v1",
            "profile_id": profile.profile_id,
            "profile_sha256": profile.profile_sha256,
            "edit_type": profile.edit_type,
            "receipt_sha256": result.receipt["receipt_sha256"],
            "subject_tree_sha256": result.receipt["hashes"]["subject_tree_sha256"],
            "post_training_state_sha256": result.receipt["hashes"][
                "post_training_state_sha256"
            ],
            "runtime": result.receipt["runtime"],
            "completed_steps": profile.steps,
        }
    ]
    assert "loss_type=None" not in caplog.text
    assert result.receipt["model"]["baseline_load"]["diagnostics"][
        "unexpected_keys"
    ] == list(profile.model_load.expected_unexpected_keys)
