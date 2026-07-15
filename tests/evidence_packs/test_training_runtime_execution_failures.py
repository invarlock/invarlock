from __future__ import annotations

import os
import shutil
import struct
import weakref
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch

from invarlock.training_state_evidence import (
    HASH_DOMAIN,
    TrainingStateEvidenceError,
    tensor_content_sha256,
)
from scripts.evidence_packs.python.editing import training_runtime as runtime
from scripts.evidence_packs.python.editing.training_contract import (
    load_training_profile,
)
from tests.evidence_packs._support_training_runtime import (
    FakeAutoModel,
    FakeAutoTokenizer,
    FakePeftModel,
    NoOpOptimizer,
    RecordingAdamW,
    TinyCausalLM,
    TinyTokenizer,
    pin_fake_training_toolchain,
)
from tests.evidence_packs._support_training_runtime import (
    fake_peft_dependencies as _fake_peft_dependencies,
)


@pytest.fixture
def fake_runtime(monkeypatch: pytest.MonkeyPatch) -> runtime.RuntimeDependencies:
    pin_fake_training_toolchain(monkeypatch)
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


def test_run_refuses_existing_output_before_loading_dependencies(
    tmp_path: Path,
) -> None:
    output = tmp_path / "subject"
    output.mkdir()
    with pytest.raises(runtime.TrainingRuntimeError, match="refusing to replace"):
        runtime.run_training_profile(
            load_training_profile("tiny_gpt2_full_ft_v1"), output
        )


def test_run_sets_pad_from_eos_and_handles_models_without_config(
    tmp_path: Path,
    fake_runtime: runtime.RuntimeDependencies,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del fake_runtime

    class EosOnlyTokenizer(TinyTokenizer):
        pad_token_id = None
        eos_token = "<eos>"

    monkeypatch.setattr(
        FakeAutoTokenizer,
        "from_pretrained",
        classmethod(lambda cls, source, **options: EosOnlyTokenizer()),
    )
    original = FakeAutoModel.from_pretrained.__func__

    def without_config(cls: type[FakeAutoModel], source: Any, **options: Any) -> Any:
        loaded = original(cls, source, **options)
        model, diagnostics = loaded if isinstance(loaded, tuple) else (loaded, None)
        del model.config
        return (model, diagnostics) if diagnostics is not None else model

    monkeypatch.setattr(FakeAutoModel, "from_pretrained", classmethod(without_config))
    result = runtime.run_training_profile(
        load_training_profile("tiny_gpt2_full_ft_v1"), tmp_path / "subject"
    )
    assert result.receipt["reload_smoke"]["passed"] is True
    assert result.receipt["reload_smoke"]["inference_performed"] is True
    assert result.receipt["reload_smoke"]["all_logits_finite"] is True


def test_lora_run_handles_serialized_base_without_config(
    tmp_path: Path,
    fake_runtime: runtime.RuntimeDependencies,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del fake_runtime
    original = FakeAutoModel.from_pretrained.__func__

    def without_config(cls: type[FakeAutoModel], source: Any, **options: Any) -> Any:
        loaded = original(cls, source, **options)
        model, diagnostics = loaded if isinstance(loaded, tuple) else (loaded, None)
        del model.config
        return (model, diagnostics) if diagnostics is not None else model

    monkeypatch.setattr(FakeAutoModel, "from_pretrained", classmethod(without_config))
    monkeypatch.setattr(runtime, "_load_peft_dependencies", _fake_peft_dependencies)
    result = runtime.run_training_profile(
        load_training_profile("tiny_gpt2_lora_v1"), tmp_path / "subject"
    )
    assert result.receipt["lora"]["adapter_merge_performed"] is True


def test_lora_uses_streaming_state_evidence_without_full_snapshots(
    tmp_path: Path,
    fake_runtime: runtime.RuntimeDependencies,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del fake_runtime
    monkeypatch.setattr(runtime, "_load_peft_dependencies", _fake_peft_dependencies)

    def reject_full_snapshot(_model: Any) -> dict[str, torch.Tensor]:
        raise AssertionError("LoRA must not retain a full duplicate state dict")

    monkeypatch.setattr(runtime, "_snapshot", reject_full_snapshot)
    monkeypatch.setattr(
        runtime,
        "verify_training_artifact",
        lambda _profile, _subject, **_options: {},
    )

    result = runtime.run_training_profile(
        load_training_profile("tiny_gpt2_lora_v1"), tmp_path / "subject"
    )

    assert result.receipt["lora"]["state_evidence_policy"] == (
        "streaming-per-tensor-digests-v1"
    )
    assert result.receipt["lora"]["merge_scope_exact"] is True


def test_lora_adapter_save_honors_offline_boundary(
    tmp_path: Path,
    fake_runtime: runtime.RuntimeDependencies,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del fake_runtime
    monkeypatch.setattr(runtime, "_load_peft_dependencies", _fake_peft_dependencies)
    original = FakePeftModel.save_pretrained
    offline_save_calls = 0

    def require_offline(self: FakePeftModel, *args: Any, **kwargs: Any) -> None:
        nonlocal offline_save_calls
        assert os.environ.get("HF_HUB_OFFLINE") == "1"
        assert os.environ.get("TRANSFORMERS_OFFLINE") == "1"
        offline_save_calls += 1
        original(self, *args, **kwargs)

    monkeypatch.setattr(FakePeftModel, "save_pretrained", require_offline)
    runtime.run_training_profile(
        load_training_profile("tiny_gpt2_lora_v1"), tmp_path / "subject"
    )
    # The independent optimizer replay serializes a fresh adapter too.  The
    # callback above asserts the offline variables for every save, not merely
    # the first producer-side serialization.
    assert offline_save_calls == 2


def test_lora_releases_post_training_live_state_before_serialized_reload(
    tmp_path: Path,
    fake_runtime: runtime.RuntimeDependencies,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del fake_runtime
    monkeypatch.setattr(runtime, "_load_peft_dependencies", _fake_peft_dependencies)
    original_state = runtime._peft_base_state
    original_load = FakeAutoModel.from_pretrained.__func__
    state_calls = 0
    retained: weakref.ReferenceType[dict[str, torch.Tensor]] | None = None

    class WeakState(dict[str, torch.Tensor]):
        pass

    def tracked_state(model: Any) -> dict[str, torch.Tensor]:
        nonlocal state_calls, retained
        state_calls += 1
        value = WeakState(original_state(model))
        if state_calls == 3:
            retained = weakref.ref(value)
        return value

    load_calls = 0

    def checked_load(cls: type[FakeAutoModel], source: Any, **options: Any) -> Any:
        nonlocal load_calls
        load_calls += 1
        if load_calls == 2:
            assert retained is not None and retained() is None
        return original_load(cls, source, **options)

    monkeypatch.setattr(runtime, "_peft_base_state", tracked_state)
    monkeypatch.setattr(FakeAutoModel, "from_pretrained", classmethod(checked_load))
    runtime.run_training_profile(
        load_training_profile("tiny_gpt2_lora_v1"), tmp_path / "subject"
    )
    assert load_calls >= 2
    assert retained is not None and retained() is None


def test_lora_releases_all_merged_model_aliases_before_final_reload(
    tmp_path: Path,
    fake_runtime: runtime.RuntimeDependencies,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del fake_runtime
    deps = _fake_peft_dependencies()
    original_get = deps.get_peft_model
    original_adapter_load = FakePeftModel.from_pretrained.__func__
    original_model_load = FakeAutoModel.from_pretrained.__func__
    refs: list[weakref.ReferenceType[Any]] = []

    def tracked_get(model: Any, config: Any) -> FakePeftModel:
        wrapper = original_get(model, config)
        refs.extend(
            (weakref.ref(wrapper), weakref.ref(model), weakref.ref(wrapper.lora_A))
        )
        return wrapper

    def tracked_adapter_load(
        cls: type[FakePeftModel], model: Any, path: Path, **options: Any
    ) -> FakePeftModel:
        wrapper = original_adapter_load(cls, model, path, **options)
        refs.extend(
            (
                weakref.ref(wrapper),
                weakref.ref(model),
                weakref.ref(wrapper.base_model.projection.weight),
            )
        )
        return wrapper

    load_calls = 0

    def checked_model_load(
        cls: type[FakeAutoModel], source: Any, **options: Any
    ) -> Any:
        nonlocal load_calls
        load_calls += 1
        if load_calls == 3:
            assert refs and all(reference() is None for reference in refs)
        return original_model_load(cls, source, **options)

    monkeypatch.setattr(
        runtime,
        "_load_peft_dependencies",
        lambda: replace(deps, get_peft_model=tracked_get),
    )
    monkeypatch.setattr(
        FakePeftModel, "from_pretrained", classmethod(tracked_adapter_load)
    )
    monkeypatch.setattr(
        FakeAutoModel, "from_pretrained", classmethod(checked_model_load)
    )
    runtime.run_training_profile(
        load_training_profile("tiny_gpt2_lora_v1"), tmp_path / "subject"
    )
    assert load_calls >= 3
    assert refs and all(reference() is None for reference in refs)


@pytest.mark.parametrize(
    ("profile_id", "final_reload_call"),
    (("tiny_gpt2_full_ft_v1", 2), ("tiny_gpt2_lora_v1", 3)),
)
def test_run_releases_final_reload_and_parameter_aliases_before_verifier(
    profile_id: str,
    final_reload_call: int,
    tmp_path: Path,
    fake_runtime: runtime.RuntimeDependencies,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del fake_runtime
    if profile_id.endswith("lora_v1"):
        monkeypatch.setattr(runtime, "_load_peft_dependencies", _fake_peft_dependencies)
    original_load = FakeAutoModel.from_pretrained.__func__
    references: list[weakref.ReferenceType[Any]] = []
    load_calls = 0

    def tracked_load(cls: type[FakeAutoModel], source: Any, **options: Any) -> Any:
        nonlocal load_calls
        load_calls += 1
        loaded = original_load(cls, source, **options)
        model = loaded[0] if isinstance(loaded, tuple) else loaded
        if load_calls == final_reload_call:
            references.extend(
                (weakref.ref(model), weakref.ref(model.projection.weight))
            )
        return loaded

    def require_released(_profile: Any, _subject: Path, **_options: Any) -> dict:
        assert references and all(reference() is None for reference in references)
        return {}

    monkeypatch.setattr(FakeAutoModel, "from_pretrained", classmethod(tracked_load))
    monkeypatch.setattr(runtime, "verify_training_artifact", require_released)

    runtime.run_training_profile(
        load_training_profile(profile_id), tmp_path / "subject"
    )
    assert references and all(reference() is None for reference in references)


def test_run_rehashes_staging_after_verifier_returns(
    tmp_path: Path,
    fake_runtime: runtime.RuntimeDependencies,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del fake_runtime

    def mutate_after_verify(
        _profile: Any, subject: Path, **_options: Any
    ) -> dict[str, Any]:
        (subject / "config.json").write_text("mutated", encoding="utf-8")
        return {}

    monkeypatch.setattr(runtime, "verify_training_artifact", mutate_after_verify)
    with pytest.raises(
        runtime.TrainingRuntimeError, match="changed after final verification"
    ):
        runtime.run_training_profile(
            load_training_profile("tiny_gpt2_full_ft_v1"), tmp_path / "subject"
        )


@pytest.mark.parametrize("replace", [False, True])
def test_run_rejects_receipt_mutation_or_replacement_during_verification(
    replace: bool,
    tmp_path: Path,
    fake_runtime: runtime.RuntimeDependencies,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del fake_runtime

    def mutate_after_verify(
        _profile: Any, subject: Path, **_options: Any
    ) -> dict[str, Any]:
        receipt = subject / "training_receipt.json"
        raw = receipt.read_bytes()
        if replace:
            replacement = subject / ".replacement-receipt"
            replacement.write_bytes(raw)
            os.replace(replacement, receipt)
        else:
            receipt.write_bytes(raw + b" ")
        return {}

    monkeypatch.setattr(runtime, "verify_training_artifact", mutate_after_verify)
    with pytest.raises(
        runtime.TrainingRuntimeError, match="receipt changed during final verification"
    ):
        runtime.run_training_profile(
            load_training_profile("tiny_gpt2_full_ft_v1"), tmp_path / "subject"
        )


def test_run_rebinds_receipt_immediately_before_publication(
    tmp_path: Path,
    fake_runtime: runtime.RuntimeDependencies,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del fake_runtime
    original = runtime._require_unchanged_receipt

    def mutate_at_prepublication(path: Path, expected: Any, *, phase: str) -> Any:
        if phase == "immediately before publication":
            path.write_bytes(path.read_bytes() + b" ")
        return original(path, expected, phase=phase)

    monkeypatch.setattr(runtime, "_require_unchanged_receipt", mutate_at_prepublication)
    with pytest.raises(
        runtime.TrainingRuntimeError,
        match="receipt changed immediately before publication",
    ):
        runtime.run_training_profile(
            load_training_profile("tiny_gpt2_full_ft_v1"), tmp_path / "subject"
        )


def test_run_rejects_artifact_mutation_inside_publication_boundary(
    tmp_path: Path,
    fake_runtime: runtime.RuntimeDependencies,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del fake_runtime
    original_publish = runtime._publish_directory_no_replace

    def mutate_then_publish(staging: Path, output: Path) -> None:
        (staging / "config.json").write_text("mutated", encoding="utf-8")
        original_publish(staging, output)

    output = tmp_path / "subject"
    monkeypatch.setattr(runtime, "_publish_directory_no_replace", mutate_then_publish)
    # Exercise the outer publication boundary directly: artifact replay has
    # its own fresh publication and must not consume this mutation hook.
    monkeypatch.setattr(runtime, "verify_training_artifact", lambda *_a, **_k: {})
    with pytest.raises(
        runtime.TrainingRuntimeError,
        match="published training subject does not match the verified artifact tree",
    ):
        runtime.run_training_profile(
            load_training_profile("tiny_gpt2_full_ft_v1"), output
        )
    assert not output.exists()
    quarantines = list(tmp_path.glob(".subject.rejected-*"))
    assert len(quarantines) == 1
    shutil.rmtree(quarantines[0])


def test_failed_publication_never_deletes_a_swapped_quarantine_path(
    tmp_path: Path,
    fake_runtime: runtime.RuntimeDependencies,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del fake_runtime
    original_publish = runtime._publish_directory_no_replace
    original_identity = runtime._directory_identity
    expected_quarantine = tmp_path / "expected-quarantine"
    replacement_marker = "replacement must survive"
    swapped = False

    def mutate_then_publish(staging: Path, output: Path) -> None:
        (staging / "config.json").write_text("mutated", encoding="utf-8")
        original_publish(staging, output)

    def swap_before_identity(path: Path, *, label: str) -> tuple[int, int]:
        nonlocal swapped
        if label == "rejected training subject" and not swapped:
            swapped = True
            path.rename(expected_quarantine)
            path.mkdir()
            (path / "marker").write_text(replacement_marker, encoding="utf-8")
        return original_identity(path, label=label)

    output = tmp_path / "subject"
    monkeypatch.setattr(runtime, "_publish_directory_no_replace", mutate_then_publish)
    monkeypatch.setattr(runtime, "_directory_identity", swap_before_identity)
    # Keep the injected swap scoped to the rejected outer publication.
    monkeypatch.setattr(runtime, "verify_training_artifact", lambda *_a, **_k: {})
    with pytest.raises(
        runtime.TrainingRuntimeError,
        match="identity changed while being quarantined",
    ):
        runtime.run_training_profile(
            load_training_profile("tiny_gpt2_full_ft_v1"), output
        )

    assert not output.exists()
    replacements = list(tmp_path.glob(".subject.rejected-*"))
    assert len(replacements) == 1
    assert (replacements[0] / "marker").read_text(encoding="utf-8") == (
        replacement_marker
    )
    assert expected_quarantine.is_dir()
    shutil.rmtree(replacements[0])
    shutil.rmtree(expected_quarantine)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16])
def test_chunked_tensor_content_hash_matches_legacy_bytes(dtype: torch.dtype) -> None:
    tensor = torch.arange(33, dtype=torch.float32).to(dtype).reshape(3, 11)
    expected = (
        "sha256:" + runtime.sha256(runtime._tensor_bytes(tensor, torch)).hexdigest()
    )
    assert tensor_content_sha256(tensor, torch=torch) == expected


def test_chunked_tensor_content_hash_rejects_noncontiguous_state() -> None:
    tensor = torch.arange(33, dtype=torch.float32).reshape(3, 11).T
    with pytest.raises(TrainingStateEvidenceError, match="requires contiguous"):
        tensor_content_sha256(tensor, torch=torch)


def test_chunked_state_hash_matches_legacy_encoding() -> None:
    state = {"weight": torch.arange(41, dtype=torch.float32).reshape(1, 41)}
    expected = runtime.sha256(HASH_DOMAIN + b"tensor-state\0")
    tensor = state["weight"]
    for value in (
        b"weight",
        str(tensor.dtype).encode("ascii"),
        runtime.canonical_json_bytes(list(tensor.shape)),
        runtime._tensor_bytes(tensor, torch),
    ):
        expected.update(struct.pack(">Q", len(value)))
        expected.update(value)
    assert (
        runtime.tensor_state_sha256(state, torch=torch)
        == "sha256:" + expected.hexdigest()
    )


def test_chunked_lora_delta_matches_full_delta_encoding() -> None:
    before = {
        "changed": torch.arange(29, dtype=torch.float32),
        "frozen": torch.arange(17, dtype=torch.float32),
    }
    after = {name: tensor.clone() for name, tensor in before.items()}
    after["changed"][5:9] += 0.25
    expected = runtime._delta_evidence(before, after, torch=torch)
    observed = runtime._streaming_lora_delta_evidence(
        baseline_manifest=runtime._state_manifest(before, torch=torch),
        baseline_targets={"changed": before["changed"]},
        after=after,
        torch=torch,
    )
    assert observed == expected


def test_run_rejects_tokenizer_without_pad_or_eos(
    tmp_path: Path,
    fake_runtime: runtime.RuntimeDependencies,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del fake_runtime

    class NoSpecialTokenizer(TinyTokenizer):
        pad_token_id = None
        eos_token = None

    monkeypatch.setattr(
        FakeAutoTokenizer,
        "from_pretrained",
        classmethod(lambda cls, source, **options: NoSpecialTokenizer()),
    )
    with pytest.raises(
        runtime.TrainingRuntimeError, match="neither a pad token nor an EOS"
    ):
        runtime.run_training_profile(
            load_training_profile("tiny_gpt2_full_ft_v1"), tmp_path / "subject"
        )


def test_run_rejects_frozen_full_fine_tune_parameters(
    tmp_path: Path,
    fake_runtime: runtime.RuntimeDependencies,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del fake_runtime
    original = FakeAutoModel.from_pretrained.__func__

    def frozen(cls: type[FakeAutoModel], source: Any, **options: Any) -> TinyCausalLM:
        loaded = original(cls, source, **options)
        model = loaded[0] if isinstance(loaded, tuple) else loaded
        for parameter in model.parameters():
            parameter.requires_grad = False
        return loaded

    monkeypatch.setattr(FakeAutoModel, "from_pretrained", classmethod(frozen))
    with pytest.raises(runtime.TrainingRuntimeError, match="every model parameter"):
        runtime.run_training_profile(
            load_training_profile("tiny_gpt2_full_ft_v1"), tmp_path / "subject"
        )


def test_run_rejects_non_lora_trainable_parameters(
    tmp_path: Path,
    fake_runtime: runtime.RuntimeDependencies,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del fake_runtime

    class BadTrainableName(FakePeftModel):
        def named_parameters(self, *args: Any, **kwargs: Any):
            for name, parameter in super().named_parameters(*args, **kwargs):
                if name == "lora_A":
                    yield "adapter_A", parameter
                else:
                    yield name, parameter

    deps = _fake_peft_dependencies()
    monkeypatch.setattr(
        runtime,
        "_load_peft_dependencies",
        lambda: replace(
            deps,
            get_peft_model=lambda model, config: BadTrainableName(model, config),
        ),
    )
    with pytest.raises(runtime.TrainingRuntimeError, match="only LoRA adapter"):
        runtime.run_training_profile(
            load_training_profile("tiny_gpt2_lora_v1"), tmp_path / "subject"
        )


def test_run_rejects_mutated_frozen_lora_base(
    tmp_path: Path,
    fake_runtime: runtime.RuntimeDependencies,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del fake_runtime
    real_state = runtime._peft_base_state
    calls = 0

    def mutated_second(model: Any) -> dict[str, torch.Tensor]:
        nonlocal calls
        calls += 1
        snapshot = dict(real_state(model))
        if calls == 2:
            first = next(iter(snapshot))
            snapshot[first] = snapshot[first].detach().clone().add_(1.0)
        return snapshot

    monkeypatch.setattr(runtime, "_load_peft_dependencies", _fake_peft_dependencies)
    monkeypatch.setattr(runtime, "_peft_base_state", mutated_second)
    with pytest.raises(
        runtime.TrainingRuntimeError,
        match="PEFT adapter construction base state changed",
    ):
        runtime.run_training_profile(
            load_training_profile("tiny_gpt2_lora_v1"), tmp_path / "subject"
        )


def test_run_rejects_unchanged_adapter_and_missing_adapter_modules(
    tmp_path: Path,
    fake_runtime: runtime.RuntimeDependencies,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = load_training_profile("tiny_gpt2_lora_v1")
    no_op = runtime.RuntimeDependencies(
        **{
            **fake_runtime.__dict__,
            "optimizer_cls": NoOpOptimizer,
        }
    )
    monkeypatch.setattr(runtime, "_load_runtime_dependencies", lambda: no_op)
    monkeypatch.setattr(runtime, "_load_peft_dependencies", _fake_peft_dependencies)
    with pytest.raises(runtime.TrainingRuntimeError, match="did not change adapter"):
        runtime.run_training_profile(profile, tmp_path / "unchanged")

    monkeypatch.setattr(runtime, "_load_runtime_dependencies", lambda: fake_runtime)
    monkeypatch.setattr(runtime, "_adapter_module_count", lambda _state: 0)
    with pytest.raises(runtime.TrainingRuntimeError, match="no LoRA adapter modules"):
        runtime.run_training_profile(profile, tmp_path / "missing-modules")


def test_run_rejects_merge_that_leaves_adapter_modules(
    tmp_path: Path,
    fake_runtime: runtime.RuntimeDependencies,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del fake_runtime
    deps = _fake_peft_dependencies()
    counts = iter((1, 1))
    monkeypatch.setattr(runtime, "_load_peft_dependencies", lambda: deps)
    monkeypatch.setattr(runtime, "_adapter_module_count", lambda _state: next(counts))
    with pytest.raises(runtime.TrainingRuntimeError, match="left LoRA adapter modules"):
        runtime.run_training_profile(
            load_training_profile("tiny_gpt2_lora_v1"), tmp_path / "subject"
        )


def test_run_rejects_incomplete_fine_tune_state_change(
    tmp_path: Path,
    fake_runtime: runtime.RuntimeDependencies,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del fake_runtime
    original = runtime._delta_evidence

    def incomplete(before: Any, after: Any, *, torch: Any):
        digest, count, maximum, names = original(before, after, torch=torch)
        return digest, count, maximum, {next(iter(names))}

    monkeypatch.setattr(runtime, "_delta_evidence", incomplete)
    with pytest.raises(
        runtime.TrainingRuntimeError, match="left trainable tensors unchanged"
    ):
        runtime.run_training_profile(
            load_training_profile("tiny_gpt2_full_ft_v1"), tmp_path / "subject"
        )


def test_run_rejects_state_hash_collision_and_copied_artifact_tree(
    tmp_path: Path,
    fake_runtime: runtime.RuntimeDependencies,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del fake_runtime
    real_tensor_hash = runtime.tensor_state_sha256
    hashes: list[str] = []

    def collide(state: Any, *, torch: Any) -> str:
        value = real_tensor_hash(state, torch=torch)
        hashes.append(value)
        return hashes[0] if len(hashes) == 3 else value

    monkeypatch.setattr(runtime, "tensor_state_sha256", collide)
    with pytest.raises(runtime.TrainingRuntimeError, match="state equals the baseline"):
        runtime.run_training_profile(
            load_training_profile("tiny_gpt2_full_ft_v1"), tmp_path / "collision"
        )

    monkeypatch.setattr(runtime, "tensor_state_sha256", real_tensor_hash)
    real_directory_hash = runtime.directory_sha256
    calls: list[str] = []

    def copied_tree(path: Path, *, exclude: frozenset[str] = frozenset()) -> str:
        value = real_directory_hash(path, exclude=exclude)
        calls.append(value)
        return calls[1] if len(calls) == 3 else value

    monkeypatch.setattr(runtime, "directory_sha256", copied_tree)
    with pytest.raises(runtime.TrainingRuntimeError, match="artifact tree equals"):
        runtime.run_training_profile(
            load_training_profile("tiny_gpt2_full_ft_v1"), tmp_path / "copied-tree"
        )


def test_run_rejects_artifact_mutation_while_writing_receipt(
    tmp_path: Path,
    fake_runtime: runtime.RuntimeDependencies,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del fake_runtime
    real_directory_hash = runtime.directory_sha256
    calls = 0

    def changed(path: Path, *, exclude: frozenset[str] = frozenset()) -> str:
        nonlocal calls
        calls += 1
        value = real_directory_hash(path, exclude=exclude)
        return "sha256:" + "0" * 64 if calls == 4 else value

    monkeypatch.setattr(runtime, "directory_sha256", changed)
    with pytest.raises(runtime.TrainingRuntimeError, match="changed while writing"):
        runtime.run_training_profile(
            load_training_profile("tiny_gpt2_full_ft_v1"), tmp_path / "subject"
        )


def test_run_cleans_staging_when_baseline_temp_creation_fails(
    tmp_path: Path,
    fake_runtime: runtime.RuntimeDependencies,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del fake_runtime
    real_mkdtemp = runtime.tempfile.mkdtemp
    calls = 0

    def fail_second(*args: Any, **kwargs: Any) -> str:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("baseline temp failed")
        return real_mkdtemp(*args, **kwargs)

    monkeypatch.setattr(runtime.tempfile, "mkdtemp", fail_second)
    with pytest.raises(OSError, match="baseline temp failed"):
        runtime.run_training_profile(
            load_training_profile("tiny_gpt2_full_ft_v1"), tmp_path / "subject"
        )
    assert not list(tmp_path.glob(".subject.*"))
