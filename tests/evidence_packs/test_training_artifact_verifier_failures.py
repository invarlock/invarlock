from __future__ import annotations

import copy
import json
import os
import shutil
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from invarlock.training_evidence_contracts.common import canonical_json_sha256
from scripts.evidence_packs.python.editing import training_artifact_verifier as verifier
from scripts.evidence_packs.python.editing import training_runtime as runtime
from scripts.evidence_packs.python.editing.training_contract import (
    LoraTrainingProfile,
    file_sha256,
    load_training_profile,
)
from scripts.evidence_packs.python.editing.training_receipt import with_receipt_digest
from tests.evidence_packs._support_training_runtime import (
    FakeAutoModel,
    FakeAutoTokenizer,
    FakeLoraConfig,
    RecordingAdamW,
    TinyTokenizer,
)
from tests.evidence_packs._support_training_runtime import (
    fake_peft_dependencies as _fake_peft_dependencies,
)


@pytest.fixture
def fake_runtime(monkeypatch: pytest.MonkeyPatch) -> runtime.RuntimeDependencies:
    FakeAutoModel.reload_baseline = False
    FakeAutoModel.source_state = None
    dependencies = runtime.RuntimeDependencies(
        torch=torch,
        auto_model=FakeAutoModel,
        auto_tokenizer=FakeAutoTokenizer,
        optimizer_cls=RecordingAdamW,
        transformers_version="5.12.0",
    )
    monkeypatch.setattr(runtime, "_load_runtime_dependencies", lambda: dependencies)
    return dependencies


def test_local_artifact_replay_enforces_offline_mode_for_entire_lifecycle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed: dict[str, str | None] = {}
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)

    def replay(*_args, **_kwargs):
        observed.update(
            {
                "hub": os.environ.get("HF_HUB_OFFLINE"),
                "transformers": os.environ.get("TRANSFORMERS_OFFLINE"),
            }
        )
        return {"verified": True}

    monkeypatch.setattr(verifier, "_verify_training_artifact", replay)
    result = verifier.verify_training_artifact(
        load_training_profile("tiny_gpt2_full_ft_v1"),
        tmp_path / "subject",
        local_files_only=True,
    )

    assert result == {"verified": True}
    assert observed == {"hub": "1", "transformers": "1"}
    assert os.environ.get("HF_HUB_OFFLINE") is None
    assert os.environ.get("TRANSFORMERS_OFFLINE") is None


def test_artifact_config_normalization_and_mapping_errors() -> None:
    assert verifier._normalize_lora_config(SimpleNamespace(value="enum")) == "enum"
    assert verifier._normalize_lora_config((1, "two")) == [1, "two"]
    assert verifier._normalize_lora_config({3, 1}) == [1, 3]
    with pytest.raises(runtime.TrainingRuntimeError, match="non-string field"):
        verifier._normalize_lora_config({1: "bad"})
    with pytest.raises(runtime.TrainingRuntimeError, match="unsupported value type"):
        verifier._normalize_lora_config(object())
    with pytest.raises(runtime.TrainingRuntimeError, match="does not expose"):
        verifier._config_mapping(object(), label="config")
    with pytest.raises(runtime.TrainingRuntimeError, match="did not return a mapping"):
        verifier._config_mapping(SimpleNamespace(to_dict=lambda: []), label="config")


def test_artifact_serialized_and_loaded_lora_config_tampering(tmp_path: Path) -> None:
    profile = load_training_profile("tiny_gpt2_lora_v1")
    assert isinstance(profile, LoraTrainingProfile)

    class RichConfig(FakeLoraConfig):
        def to_dict(self) -> dict[str, Any]:
            value = super().to_dict()
            value.update({"inference_mode": False, "base_model_name_or_path": None})
            return value

    deps = replace(_fake_peft_dependencies(), lora_config_cls=RichConfig)
    expected = verifier._expected_serialized_lora_config(profile, deps)
    assert expected["inference_mode"] is True
    assert expected["base_model_name_or_path"] == profile.model_id

    adapter = tmp_path / "adapter"
    adapter.mkdir()
    with pytest.raises(runtime.TrainingRuntimeError, match="unable to read serialized"):
        verifier._require_serialized_lora_config_file(
            adapter, profile, deps, expected_sha256="sha256:" + "0" * 64
        )
    (adapter / "adapter_config.json").write_text('{"r":1,"r":2}', encoding="utf-8")
    with pytest.raises(runtime.TrainingRuntimeError, match="unable to read serialized"):
        verifier._require_serialized_lora_config_file(
            adapter,
            profile,
            deps,
            expected_sha256=file_sha256(adapter / "adapter_config.json"),
        )
    (adapter / "adapter_config.json").write_text("[]", encoding="utf-8")
    with pytest.raises(runtime.TrainingRuntimeError, match="must be a JSON object"):
        verifier._require_serialized_lora_config_file(
            adapter,
            profile,
            deps,
            expected_sha256=file_sha256(adapter / "adapter_config.json"),
        )
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(runtime.TrainingRuntimeError, match="does not match"):
        verifier._require_serialized_lora_config_file(
            adapter,
            profile,
            deps,
            expected_sha256=file_sha256(adapter / "adapter_config.json"),
        )

    with pytest.raises(runtime.TrainingRuntimeError, match="exactly one"):
        verifier._require_loaded_lora_config(SimpleNamespace(peft_config={}), expected)
    wrong = RichConfig(
        r=profile.lora.rank,
        lora_alpha=profile.lora.alpha,
        lora_dropout=profile.lora.dropout,
        target_modules=list(profile.lora.target_modules),
        bias=profile.lora.bias,
        task_type=profile.lora.task_type,
        fan_in_fan_out=profile.lora.fan_in_fan_out,
    )
    wrong.r += 1
    with pytest.raises(runtime.TrainingRuntimeError, match="does not match its pinned"):
        verifier._require_loaded_lora_config(
            SimpleNamespace(peft_config={"default": wrong}), expected
        )


def test_artifact_verifier_rejects_missing_subject_and_receipt(tmp_path: Path) -> None:
    profile = load_training_profile("tiny_gpt2_full_ft_v1")
    with pytest.raises(runtime.TrainingRuntimeError, match="not a directory"):
        verifier.verify_training_artifact(profile, tmp_path / "missing")

    subject = tmp_path / "subject"
    subject.mkdir()
    with pytest.raises(
        runtime.TrainingRuntimeError, match="unable to read training receipt"
    ):
        verifier.verify_training_artifact(profile, subject)

    (subject / "training_receipt.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown schema"):
        verifier.verify_training_artifact(profile, subject)

    (subject / "training_receipt.json").write_text(
        '{"schema":"first","schema":"second"}', encoding="utf-8"
    )
    with pytest.raises(
        runtime.TrainingRuntimeError, match="unable to read training receipt"
    ):
        verifier.verify_training_artifact(profile, subject)


def test_artifact_verifier_rejects_top_level_subject_symlink(
    tmp_path: Path,
    fake_runtime: runtime.RuntimeDependencies,
) -> None:
    del fake_runtime
    profile = load_training_profile("tiny_gpt2_full_ft_v1")
    result = runtime.run_training_profile(profile, tmp_path / "subject")
    linked = tmp_path / "linked-subject"
    linked.symlink_to(result.subject_dir, target_is_directory=True)

    with pytest.raises(runtime.TrainingRuntimeError, match="non-symlink directory"):
        verifier.verify_training_artifact(profile, linked)


def test_subject_resolution_fails_closed_on_resolution_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    subject = tmp_path / "subject"
    subject.mkdir()

    def fail_resolve(_path: Path, *, strict: bool = False) -> Path:
        del strict
        raise OSError("subject disappeared")

    monkeypatch.setattr(Path, "resolve", fail_resolve)
    with pytest.raises(
        runtime.TrainingRuntimeError, match="changed while being resolved"
    ):
        verifier._subject_directory(subject)


def test_subject_resolution_fails_closed_on_identity_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    subject = tmp_path / "subject"
    subject.mkdir()
    identities = iter(((1, 1), (1, 2), (1, 1)))
    monkeypatch.setattr(verifier, "_directory_identity", lambda _stat: next(identities))

    with pytest.raises(
        runtime.TrainingRuntimeError, match="changed while being resolved"
    ):
        verifier._subject_directory(subject)


def test_subject_identity_recheck_fails_closed_when_subject_disappears(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing"

    with pytest.raises(runtime.TrainingRuntimeError, match="changed before replay"):
        verifier._require_subject_identity(missing, (1, 1), phase="before replay")


@pytest.mark.parametrize("replace", [False, True])
def test_artifact_verifier_rejects_receipt_mutation_or_replacement_during_run(
    replace: bool,
    tmp_path: Path,
    fake_runtime: runtime.RuntimeDependencies,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del fake_runtime
    profile = load_training_profile("tiny_gpt2_full_ft_v1")
    result = runtime.run_training_profile(profile, tmp_path / "subject")
    receipt_path = result.receipt_path
    original_toolchain = verifier.runtime._toolchain
    mutated = False

    def mutate_late(*args: Any, **kwargs: Any) -> dict[str, str]:
        nonlocal mutated
        observed = original_toolchain(*args, **kwargs)
        if not mutated:
            mutated = True
            raw = receipt_path.read_bytes()
            if replace:
                replacement = receipt_path.with_name(".replacement-receipt")
                replacement.write_bytes(raw)
                os.replace(replacement, receipt_path)
            else:
                receipt_path.write_bytes(raw + b" ")
        return observed

    monkeypatch.setattr(verifier.runtime, "_toolchain", mutate_late)
    with pytest.raises(
        runtime.TrainingRuntimeError,
        match="receipt changed during artifact verification",
    ):
        verifier.verify_training_artifact(profile, result.subject_dir)


def test_artifact_verifier_rejects_subject_directory_replacement_after_entry(
    tmp_path: Path,
    fake_runtime: runtime.RuntimeDependencies,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del fake_runtime
    profile = load_training_profile("tiny_gpt2_full_ft_v1")
    result = runtime.run_training_profile(profile, tmp_path / "subject")
    replacement = tmp_path / "replacement"
    displaced = tmp_path / "displaced"
    shutil.copytree(result.subject_dir, replacement)
    original_snapshot = verifier.runtime._receipt_file_snapshot
    replaced = False

    def replace_root_then_read(path: Path, *, label: str) -> Any:
        nonlocal replaced
        if not replaced:
            replaced = True
            result.subject_dir.rename(displaced)
            replacement.rename(result.subject_dir)
        return original_snapshot(path, label=label)

    monkeypatch.setattr(
        verifier.runtime, "_receipt_file_snapshot", replace_root_then_read
    )
    with pytest.raises(
        runtime.TrainingRuntimeError,
        match="training subject changed while opening its receipt",
    ):
        verifier.verify_training_artifact(profile, result.subject_dir)


def test_artifact_verifier_rejects_tampered_changed_count_and_delta(
    tmp_path: Path,
    fake_runtime: runtime.RuntimeDependencies,
) -> None:
    del fake_runtime
    profile = load_training_profile("tiny_gpt2_full_ft_v1")
    result = runtime.run_training_profile(profile, tmp_path / "subject")

    receipt = json.loads(result.receipt_path.read_text())
    receipt["changes"]["changed_tensors"] += 1
    receipt = with_receipt_digest(receipt)
    result.receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(runtime.TrainingRuntimeError, match="changed tensor count"):
        verifier.verify_training_artifact(profile, result.subject_dir)

    receipt["changes"]["changed_tensors"] -= 1
    receipt["changes"]["max_abs_delta"] *= 2
    receipt = with_receipt_digest(receipt)
    result.receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(runtime.TrainingRuntimeError, match="maximum tensor delta"):
        verifier.verify_training_artifact(profile, result.subject_dir)


@pytest.mark.parametrize("profile_id", ("tiny_gpt2_lora_v1", "tiny_gpt2_full_ft_v1"))
@pytest.mark.parametrize(
    ("field", "message"),
    (
        ("changed_params", "changed parameter count"),
        ("total_params", "total parameter count"),
    ),
)
def test_artifact_verifier_independently_recomputes_parameter_counts(
    tmp_path: Path,
    fake_runtime: runtime.RuntimeDependencies,
    monkeypatch: pytest.MonkeyPatch,
    profile_id: str,
    field: str,
    message: str,
) -> None:
    del fake_runtime
    if profile_id == "tiny_gpt2_lora_v1":
        monkeypatch.setattr(runtime, "_load_peft_dependencies", _fake_peft_dependencies)
    profile = load_training_profile(profile_id)
    result = runtime.run_training_profile(profile, tmp_path / profile_id)
    receipt = json.loads(result.receipt_path.read_text())
    receipt["changes"][field] += -1 if field == "changed_params" else 1
    receipt = with_receipt_digest(receipt)
    result.receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    with pytest.raises(runtime.TrainingRuntimeError, match=message):
        verifier.verify_training_artifact(profile, result.subject_dir)


def test_artifact_verifier_rejects_rebound_baseline_load_diagnostics(
    tmp_path: Path,
    fake_runtime: runtime.RuntimeDependencies,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del fake_runtime
    profile = load_training_profile("tiny_gpt2_full_ft_v1")
    result = runtime.run_training_profile(profile, tmp_path / "subject")
    original_load = runtime._load_profile_baseline

    def forged_diagnostics(*args: Any, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        model, diagnostics = original_load(*args, **kwargs)
        changed = dict(diagnostics)
        changed["unexpected_keys"] = [*changed["unexpected_keys"], "forged.key"]
        return model, changed

    monkeypatch.setattr(runtime, "_load_profile_baseline", forged_diagnostics)

    with pytest.raises(
        runtime.TrainingRuntimeError,
        match="baseline loading diagnostics do not match",
    ):
        verifier.verify_training_artifact(profile, result.subject_dir)


def test_artifact_verifier_rejects_coherently_forged_base_manifest(
    tmp_path: Path,
    fake_runtime: runtime.RuntimeDependencies,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del fake_runtime
    monkeypatch.setattr(runtime, "_load_peft_dependencies", _fake_peft_dependencies)
    profile = load_training_profile("tiny_gpt2_lora_v1")
    result = runtime.run_training_profile(profile, tmp_path / "subject")
    receipt = json.loads(result.receipt_path.read_text())
    forged = "sha256:" + "a" * 64
    receipt["lora"]["base_state_manifest_sha256"] = forged
    receipt["lora"]["base_state_manifest_before_adapter_sha256"] = forged
    receipt["lora"]["base_state_manifest_after_training_sha256"] = forged
    receipt = with_receipt_digest(receipt)
    result.receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    with pytest.raises(runtime.TrainingRuntimeError, match="baseline state manifest"):
        verifier.verify_training_artifact(profile, result.subject_dir)


def test_artifact_verifier_rejects_rebound_token_count_and_toolchain(
    tmp_path: Path,
    fake_runtime: runtime.RuntimeDependencies,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del fake_runtime
    profile = load_training_profile("tiny_gpt2_full_ft_v1")
    result = runtime.run_training_profile(profile, tmp_path / "subject")
    receipt = json.loads(result.receipt_path.read_text())

    receipt["training_data"]["token_count"] += 1
    receipt = with_receipt_digest(receipt)
    result.receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(runtime.TrainingRuntimeError, match="token count"):
        verifier.verify_training_artifact(profile, result.subject_dir)

    receipt["training_data"]["token_count"] -= 1
    receipt = with_receipt_digest(receipt)
    result.receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    monkeypatch.setattr(
        verifier.runtime,
        "_toolchain",
        lambda *_args: {
            "python": profile.toolchain.python,
            "torch": "different",
            "transformers": profile.toolchain.transformers,
        },
    )
    monkeypatch.setattr(
        verifier.runtime, "_require_expected_toolchain", lambda *_a: None
    )
    with pytest.raises(runtime.TrainingRuntimeError, match="toolchain does not match"):
        verifier.verify_training_artifact(profile, result.subject_dir)


def test_artifact_verifier_rejects_tokenizer_without_pad_or_eos(
    tmp_path: Path,
    fake_runtime: runtime.RuntimeDependencies,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del fake_runtime
    profile = load_training_profile("tiny_gpt2_full_ft_v1")
    result = runtime.run_training_profile(profile, tmp_path / "subject")

    class NoSpecialTokens(TinyTokenizer):
        pad_token_id = None
        eos_token = None

    monkeypatch.setattr(
        FakeAutoTokenizer,
        "from_pretrained",
        classmethod(lambda cls, source, **options: NoSpecialTokens()),
    )
    with pytest.raises(
        runtime.TrainingRuntimeError, match="neither a pad token nor an EOS"
    ):
        verifier.verify_training_artifact(profile, result.subject_dir)


def test_artifact_verifier_accepts_model_without_config_attribute(
    tmp_path: Path,
    fake_runtime: runtime.RuntimeDependencies,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del fake_runtime
    profile = load_training_profile("tiny_gpt2_full_ft_v1")
    result = runtime.run_training_profile(profile, tmp_path / "subject")
    original = FakeAutoModel.from_pretrained.__func__

    def without_config(cls: type[FakeAutoModel], source: Any, **options: Any) -> Any:
        loaded = original(cls, source, **options)
        model, diagnostics = loaded if isinstance(loaded, tuple) else (loaded, None)
        del model.config
        return (model, diagnostics) if diagnostics is not None else model

    monkeypatch.setattr(FakeAutoModel, "from_pretrained", classmethod(without_config))
    assert (
        verifier.verify_training_artifact(profile, result.subject_dir) == result.receipt
    )


def test_optimizer_execution_proof_uses_the_receipt_runtime_image_identity(
    tmp_path: Path,
    fake_runtime: runtime.RuntimeDependencies,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A verifier rerun must not inherit a later environment-image change."""

    del fake_runtime
    original_image = "sha256:" + "a" * 64
    monkeypatch.setenv("INVARLOCK_EXPECTED_RUNTIME_IMAGE_DIGEST", original_image)
    profile = load_training_profile("tiny_gpt2_full_ft_v1")
    result = runtime.run_training_profile(profile, tmp_path / "subject")
    assert result.receipt["runtime"]["container_image_digest"] == original_image

    monkeypatch.setenv("INVARLOCK_EXPECTED_RUNTIME_IMAGE_DIGEST", "sha256:" + "b" * 64)

    assert (
        verifier.verify_training_artifact(profile, result.subject_dir) == result.receipt
    )


@pytest.mark.parametrize("profile_id", ("tiny_gpt2_full_ft_v1", "tiny_gpt2_lora_v1"))
def test_artifact_verifier_rejects_recomputed_forged_optimizer_receipt(
    tmp_path: Path,
    fake_runtime: runtime.RuntimeDependencies,
    monkeypatch: pytest.MonkeyPatch,
    profile_id: str,
) -> None:
    """A canonical receipt digest cannot replace the independent rerun."""

    del fake_runtime
    if profile_id == "tiny_gpt2_lora_v1":
        monkeypatch.setattr(runtime, "_load_peft_dependencies", _fake_peft_dependencies)
    profile = load_training_profile(profile_id)
    result = runtime._run_training_profile(
        profile,
        tmp_path / profile_id,
        local_files_only=True,
        verify_artifact=False,
    )
    forged = copy.deepcopy(result.receipt)
    forged_losses = [float(loss) + 0.125 for loss in forged["training"]["losses"]]
    forged["training"]["losses"] = forged_losses
    forged["training"]["initial_loss"] = forged_losses[0]
    forged["training"]["final_loss"] = forged_losses[-1]
    forged = with_receipt_digest(forged)
    result.receipt_path.write_text(json.dumps(forged), encoding="utf-8")

    with pytest.raises(
        runtime.TrainingRuntimeError,
        match="independent optimizer execution proof does not match",
    ):
        verifier.verify_training_artifact(profile, result.subject_dir)


def test_artifact_verifier_rejects_self_consistent_forged_dataset_provider(
    tmp_path: Path,
    fake_runtime: runtime.RuntimeDependencies,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The artifact verifier uses policy identity, not receipt self-hashes."""

    del fake_runtime
    profile = load_training_profile("tiny_gpt2_full_ft_v1")
    result = runtime._run_training_profile(
        profile,
        tmp_path / "subject",
        local_files_only=True,
        verify_artifact=False,
    )
    forged = copy.deepcopy(result.receipt)
    provider = {"kind": "forged-provider", "revision": "f" * 40}
    forged["dataset_provider"] = {
        "provider": provider,
        "provider_sha256": canonical_json_sha256(provider),
    }
    forged = with_receipt_digest(forged)
    result.receipt_path.write_text(json.dumps(forged), encoding="utf-8")
    monkeypatch.setenv(
        "INVARLOCK_ACCEPTANCE_DATASET_PROVIDER_SNAPSHOT_JSON",
        json.dumps(forged["dataset_provider"]),
    )

    with pytest.raises(
        runtime.TrainingRuntimeError,
        match="dataset provider does not match the immutable provider policy",
    ):
        verifier.verify_training_artifact(profile, result.subject_dir)


def test_optimizer_execution_proof_rejects_missing_or_no_step_rerun(
    tmp_path: Path,
    fake_runtime: runtime.RuntimeDependencies,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del fake_runtime
    profile = load_training_profile("tiny_gpt2_full_ft_v1")
    result = runtime._run_training_profile(
        profile,
        tmp_path / "subject",
        local_files_only=True,
        verify_artifact=False,
    )

    monkeypatch.setattr(
        runtime, "_run_training_profile", lambda *_args, **_kwargs: None
    )
    with pytest.raises(
        runtime.TrainingRuntimeError,
        match="optimizer execution proof is unavailable",
    ):
        verifier._independent_optimizer_execution_proof(
            profile,
            result.receipt,
            repo_root=Path.cwd(),
            local_files_only=True,
        )

    def no_step_rerun(*_args: object, **_kwargs: object) -> runtime.TrainingRunResult:
        receipt = copy.deepcopy(result.receipt)
        receipt["training"]["completed_steps"] = 0
        return runtime.TrainingRunResult(
            subject_dir=result.subject_dir,
            receipt_path=result.receipt_path,
            receipt=receipt,
        )

    monkeypatch.setattr(runtime, "_run_training_profile", no_step_rerun)
    with pytest.raises(
        runtime.TrainingRuntimeError,
        match="optimizer execution proof contains no completed steps",
    ):
        verifier._independent_optimizer_execution_proof(
            profile,
            result.receipt,
            repo_root=Path.cwd(),
            local_files_only=True,
        )


def test_optimizer_execution_proof_rejects_nonmapping_runtime_facts(
    tmp_path: Path,
    fake_runtime: runtime.RuntimeDependencies,
) -> None:
    """A receipt must retain mapping-shaped runtime facts for the sealed rerun."""

    del fake_runtime
    profile = load_training_profile("tiny_gpt2_full_ft_v1")
    result = runtime._run_training_profile(
        profile,
        tmp_path / "subject",
        local_files_only=True,
        verify_artifact=False,
    )
    malformed = copy.deepcopy(result.receipt)
    malformed["runtime"] = None

    with pytest.raises(runtime.TrainingRuntimeError, match="runtime facts"):
        verifier._independent_optimizer_execution_proof(
            profile,
            malformed,
            repo_root=Path.cwd(),
            local_files_only=True,
        )
