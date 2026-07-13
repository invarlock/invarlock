from __future__ import annotations

import errno
import importlib
import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from invarlock.peft_runtime import (
    PeftRuntimeError,
    normalize_base_key,
    peft_base_snapshot,
)
from scripts.evidence_packs.python.editing import training_runtime as runtime
from scripts.evidence_packs.python.editing.training_contract import (
    file_sha256,
    load_training_profile,
)
from tests.evidence_packs._support_training_runtime import (
    FakeAutoModel,
    FakeAutoTokenizer,
    FakeLoraConfig,
    FakePeftModel,
    RecordingAdamW,
)
from tests.evidence_packs._support_training_runtime_branch_contracts import (
    _FakeFunction,
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


def test_runtime_dependency_loaders_fail_closed_without_peft_compatibility_shim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = importlib.import_module

    def missing_torch(name: str, package: str | None = None) -> Any:
        if name == "torch":
            raise ImportError("missing")
        return original(name, package)

    monkeypatch.setattr(runtime.importlib, "import_module", missing_torch)
    with pytest.raises(runtime.TrainingRuntimeError, match="requires the torch"):
        runtime._load_runtime_dependencies()

    def missing_transformers(name: str, package: str | None = None) -> Any:
        if name == "transformers":
            raise ImportError("missing")
        return original(name, package)

    monkeypatch.setattr(runtime.importlib, "import_module", missing_transformers)
    with pytest.raises(runtime.TrainingRuntimeError, match="requires the torch"):
        runtime._load_runtime_dependencies()

    peft = SimpleNamespace(
        LoraConfig=FakeLoraConfig,
        get_peft_model=lambda model, config: FakePeftModel(model, config),
        get_peft_model_state_dict=None,
        PeftModel=FakePeftModel,
        __version__="0.19.1",
    )

    imported_modules: list[str] = []

    def fallback_peft(name: str, package: str | None = None) -> Any:
        imported_modules.append(name)
        if name == "peft":
            return peft
        if name == "peft.utils.save_and_load":
            pytest.fail("the pinned PEFT API must not use a compatibility shim")
        return original(name, package)

    monkeypatch.setattr(runtime.importlib, "import_module", fallback_peft)
    with pytest.raises(
        runtime.TrainingRuntimeError,
        match="pinned PEFT runtime lacks get_peft_model_state_dict",
    ):
        runtime._load_peft_dependencies()
    assert imported_modules == ["peft"]

    monkeypatch.setattr(runtime.importlib, "import_module", original)
    loaded = runtime._load_runtime_dependencies()
    assert loaded.torch is torch
    assert loaded.transformers_version


def test_profile_device_determinism_and_toolchain_failures_are_actionable() -> None:
    profile = load_training_profile("tiny_gpt2_full_ft_v1")
    with pytest.raises(runtime.TrainingRuntimeError, match="profile_sha256"):
        runtime._validate_profile(
            replace(profile, profile_sha256="sha256:" + "0" * 64),
            repo_root=Path.cwd(),
        )

    class TorchWithoutCuda:
        float32 = "float32"
        backends = SimpleNamespace(mps=None)

        @staticmethod
        def device(value: str) -> str:
            return value

    cuda_profile = replace(profile, device="cuda")
    with pytest.raises(runtime.TrainingRuntimeError, match="CUDA is unavailable"):
        runtime._device_and_dtype(
            SimpleNamespace(
                **TorchWithoutCuda.__dict__,
                cuda=SimpleNamespace(is_available=lambda: False),
            ),
            cuda_profile,
        )
    with pytest.raises(runtime.TrainingRuntimeError, match="MPS is unavailable"):
        runtime._device_and_dtype(TorchWithoutCuda, replace(profile, device="mps"))
    with pytest.raises(runtime.TrainingRuntimeError, match="unsupported torch dtype"):
        runtime._device_and_dtype(TorchWithoutCuda, replace(profile, dtype="missing"))
    assert runtime._device_and_dtype(TorchWithoutCuda, profile) == ("cpu", "float32")

    calls: list[Any] = []
    no_cuda = SimpleNamespace(
        manual_seed=lambda seed: calls.append(("seed", seed)),
        use_deterministic_algorithms=lambda enabled: calls.append(("det", enabled)),
        backends=SimpleNamespace(cudnn=None),
    )
    runtime._configure_determinism(no_cuda, profile)
    assert ("seed", profile.seed) in calls
    assert ("det", True) in calls

    deps = runtime.RuntimeDependencies(
        torch=SimpleNamespace(__version__="0.0.0"),
        auto_model=None,
        auto_tokenizer=None,
        optimizer_cls=None,
        transformers_version="0.0.0",
    )
    with pytest.raises(runtime.TrainingRuntimeError, match="torch=.*transformers"):
        runtime._require_expected_toolchain(profile, deps, None)


def test_training_rows_reject_digest_json_text_and_count_tampering(
    tmp_path: Path,
) -> None:
    profile = load_training_profile("tiny_gpt2_full_ft_v1")
    data = tmp_path / "rows.jsonl"

    data.write_text('{"text":"ok"}\n', encoding="utf-8")
    wrong_digest = replace(
        profile,
        training_data=replace(
            profile.training_data, path=data.name, sha256="sha256:" + "0" * 64, rows=1
        ),
    )
    with pytest.raises(runtime.TrainingRuntimeError, match="digest changed"):
        runtime._load_rows(wrong_digest, repo_root=tmp_path)

    data.write_text("{\n", encoding="utf-8")
    invalid_json = replace(
        profile,
        training_data=replace(
            profile.training_data,
            path=data.name,
            sha256=file_sha256(data),
            rows=1,
        ),
    )
    with pytest.raises(runtime.TrainingRuntimeError, match="not valid JSON"):
        runtime._load_rows(invalid_json, repo_root=tmp_path)

    data.write_text("[]\n", encoding="utf-8")
    missing_text = replace(
        profile,
        training_data=replace(
            profile.training_data,
            path=data.name,
            sha256=file_sha256(data),
            rows=1,
        ),
    )
    with pytest.raises(runtime.TrainingRuntimeError, match="lacks the configured"):
        runtime._load_rows(missing_text, repo_root=tmp_path)

    data.write_text('{"text":"one"}\n', encoding="utf-8")
    wrong_count = replace(
        profile,
        training_data=replace(
            profile.training_data,
            path=data.name,
            sha256=file_sha256(data),
            rows=2,
        ),
    )
    with pytest.raises(runtime.TrainingRuntimeError, match="row count changed"):
        runtime._load_rows(wrong_count, repo_root=tmp_path)


def test_tensor_directory_delta_and_fixture_guards_reject_unsafe_artifacts(
    tmp_path: Path,
) -> None:
    # bfloat16 uses the raw-byte fallback because NumPy cannot expose it directly.
    assert runtime._tensor_bytes(torch.tensor([1], dtype=torch.bfloat16), torch)

    tree = tmp_path / "tree"
    tree.mkdir()
    (tree / "file").write_text("value")
    (tree / "skip").write_text("ignored")
    (tree / "dir").mkdir()
    first = runtime.directory_sha256(tree, exclude=frozenset({"skip"}))
    (tree / "skip").write_text("changed")
    assert runtime.directory_sha256(tree, exclude=frozenset({"skip"})) == first
    (tree / "link").symlink_to(tree / "file")
    with pytest.raises(runtime.TrainingRuntimeError, match="contains a symlink"):
        runtime.directory_sha256(tree)

    before = {"weight": torch.zeros(2)}
    with pytest.raises(runtime.TrainingRuntimeError, match="state keys differ"):
        runtime._delta_evidence(before, {"bias": torch.zeros(2)}, torch=torch)
    with pytest.raises(runtime.TrainingRuntimeError, match="shape changed"):
        runtime._delta_evidence(before, {"weight": torch.zeros(3)}, torch=torch)
    with pytest.raises(runtime.TrainingRuntimeError, match="non-finite"):
        runtime._delta_evidence(
            before, {"weight": torch.tensor([float("nan"), 0.0])}, torch=torch
        )
    digest, changed, maximum, names = runtime._delta_evidence(
        before, {"weight": torch.ones(2)}, torch=torch
    )
    assert digest.startswith("sha256:")
    assert (changed, maximum, names) == (1, 1.0, {"weight"})
    _, changed, maximum, names = runtime._delta_evidence(
        {"empty": torch.empty(0)}, {"empty": torch.empty(0)}, torch=torch
    )
    assert (changed, maximum, names) == (0, 0.0, set())

    huge = SimpleNamespace(
        parameters=lambda: [SimpleNamespace(numel=lambda: 100_000_001)]
    )
    with pytest.raises(runtime.TrainingRuntimeError, match="limited to tiny"):
        runtime._require_fixture_sized_model(huge)


@pytest.mark.parametrize("failure", ["nonfinite", "nondeterministic"])
def test_reload_inference_smoke_fails_closed_on_invalid_logits(
    failure: str,
    fake_runtime: runtime.RuntimeDependencies,
) -> None:
    class InvalidReload(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
            del attention_mask
            self.calls += 1
            if failure == "nonfinite":
                logits = torch.full((*input_ids.shape, 3), float("nan"))
            else:
                logits = torch.full((*input_ids.shape, 3), float(self.calls))
            return SimpleNamespace(logits=logits)

    batch = {
        "input_ids": torch.ones((1, 2), dtype=torch.long),
        "attention_mask": torch.ones((1, 2), dtype=torch.long),
        "labels": torch.ones((1, 2), dtype=torch.long),
    }
    expected = "non-finite logits" if failure == "nonfinite" else "repeat-deterministic"
    with pytest.raises(runtime.TrainingRuntimeError, match=expected):
        runtime._reload_forward_smoke(
            InvalidReload(), batch, deps=fake_runtime, device=torch.device("cpu")
        )


def test_peft_base_snapshot_rejects_ambiguous_and_incomplete_state() -> None:
    class Model:
        @staticmethod
        def state_dict() -> dict[str, torch.Tensor]:
            return {
                "base_model.model.weight": torch.ones(1),
                "base_model.weight": torch.ones(1),
                "lora_A": torch.ones(1),
                "modules_to_save.copy": torch.ones(1),
            }

    with pytest.raises(PeftRuntimeError, match="ambiguous PEFT"):
        peft_base_snapshot(Model(), {"weight": torch.ones(1)})

    class MissingModel:
        @staticmethod
        def state_dict() -> dict[str, torch.Tensor]:
            return {"base_model.model.extra": torch.ones(1)}

    with pytest.raises(PeftRuntimeError, match="missing=.*extra"):
        peft_base_snapshot(MissingModel(), {"weight": torch.ones(1)})

    assert normalize_base_key("base_model.model.layer.base_layer.weight") == (
        "layer.weight"
    )
    assert normalize_base_key("base_model.layer.weight") == "layer.weight"
    assert normalize_base_key("layer.weight") == "layer.weight"


def test_offline_environment_restores_absent_and_existing_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "original")
    with runtime._hf_offline_if(True):
        assert os.environ["HF_HUB_OFFLINE"] == "1"
        assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
    assert "HF_HUB_OFFLINE" not in os.environ
    assert os.environ["TRANSFORMERS_OFFLINE"] == "original"

    with runtime._hf_offline_if(False):
        assert "HF_HUB_OFFLINE" not in os.environ


def test_atomic_publication_platform_failures_preserve_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    output = tmp_path / "output"

    monkeypatch.setattr(runtime.platform, "system", lambda: "Linux")
    monkeypatch.setattr(runtime.ctypes, "CDLL", lambda *_a, **_k: SimpleNamespace())
    with pytest.raises(runtime.TrainingRuntimeError, match="unavailable on this Linux"):
        runtime._publish_directory_no_replace(staging, output)
    assert staging.is_dir()

    function = _FakeFunction(-1)
    monkeypatch.setattr(
        runtime.ctypes, "CDLL", lambda *_a, **_k: SimpleNamespace(renameat2=function)
    )
    monkeypatch.setattr(runtime.ctypes, "get_errno", lambda: errno.EIO)
    with pytest.raises(runtime.TrainingRuntimeError, match="Input/output error"):
        runtime._publish_directory_no_replace(staging, output)

    monkeypatch.setattr(runtime.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(runtime.ctypes, "CDLL", lambda *_a, **_k: SimpleNamespace())
    with pytest.raises(runtime.TrainingRuntimeError, match="unavailable on this macOS"):
        runtime._publish_directory_no_replace(staging, output)


def test_batch_preparation_rejects_missing_ids_and_wrong_shapes() -> None:
    profile = load_training_profile("tiny_gpt2_full_ft_v1")

    with pytest.raises(runtime.TrainingRuntimeError, match="did not return input_ids"):
        runtime._prepare_batches(lambda *_a, **_k: {}, ["row"], profile, torch=torch)

    def wrong_shape(*_args: Any, **_kwargs: Any) -> dict[str, torch.Tensor]:
        return {
            "input_ids": torch.zeros((1, 1), dtype=torch.long),
            "attention_mask": torch.zeros((1, 2), dtype=torch.long),
        }

    with pytest.raises(runtime.TrainingRuntimeError, match="exact profile batch shape"):
        runtime._prepare_batches(wrong_shape, ["row"], profile, torch=torch)

    def ids_only(texts: list[str], **options: Any) -> dict[str, torch.Tensor]:
        return {
            "input_ids": torch.ones(
                (len(texts), options["max_length"]), dtype=torch.long
            )
        }

    batches, token_count, digest = runtime._prepare_batches(
        ids_only, ["row"], profile, torch=torch
    )
    assert token_count == (
        profile.steps * profile.micro_batch_size * profile.max_sequence_length
    )
    assert batches[0]["labels"].equal(batches[0]["input_ids"])
    assert digest.startswith("sha256:")


def test_train_rejects_nonfinite_loss_for_mapping_output() -> None:
    profile = load_training_profile("tiny_gpt2_full_ft_v1")
    parameter = torch.nn.Parameter(torch.tensor(1.0))

    class Model:
        @staticmethod
        def train() -> None:
            return None

        def __call__(self, **_batch: Any) -> dict[str, torch.Tensor]:
            return {"loss": parameter * torch.tensor(float("nan"))}

    deps = runtime.RuntimeDependencies(
        torch=torch,
        auto_model=None,
        auto_tokenizer=None,
        optimizer_cls=torch.optim.AdamW,
        transformers_version="5.12.0",
    )
    batches = [
        {"input_ids": torch.ones(1, dtype=torch.long)}
        for _ in range(profile.steps * profile.gradient_accumulation_steps)
    ]
    with pytest.raises(runtime.TrainingRuntimeError, match="non-finite loss"):
        runtime._train(Model(), [parameter], batches, profile, deps=deps, device="cpu")


def test_train_rejects_mutable_profile_that_changes_requested_step_count() -> None:
    base = load_training_profile("tiny_gpt2_full_ft_v1")
    parameter = torch.nn.Parameter(torch.tensor(1.0))

    class MutableSteps:
        optimizer = base.optimizer
        gradient_accumulation_steps = 1
        calls = 0

        @property
        def steps(self) -> int:
            self.calls += 1
            return 1 if self.calls == 1 else 2

    class Model:
        @staticmethod
        def train() -> None:
            return None

        def __call__(self, **_batch: Any) -> dict[str, torch.Tensor]:
            return {"loss": parameter.square()}

    deps = runtime.RuntimeDependencies(
        torch=torch,
        auto_model=None,
        auto_tokenizer=None,
        optimizer_cls=torch.optim.AdamW,
        transformers_version="5.12.0",
    )
    profile_like: Any = MutableSteps()
    with pytest.raises(runtime.TrainingRuntimeError, match="complete every requested"):
        runtime._train(
            Model(),
            [parameter],
            [{"input_ids": torch.ones(1)}],
            profile_like,
            deps=deps,
            device="cpu",
        )
