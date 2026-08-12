from __future__ import annotations

import builtins
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from examples.integrations import launch, local_registry, qwen3_profile, trust_material
from examples.integrations import run as integration

ZERO_DIGEST = "sha256:" + ("0" * 64)
REAL_QWEN3_LOADER = qwen3_profile.load_model_and_tokenizer


@pytest.fixture(autouse=True)
def tiny_qwen3_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep focused tests offline while production uses the pinned checkpoint."""

    def load_model_and_tokenizer(
        *, torch: object, transformers: object
    ) -> tuple[object, object]:
        import tokenizers

        torch.manual_seed(integration._SEED)  # type: ignore[attr-defined]
        vocabulary = {
            "<pad>": 0,
            "<bos>": 1,
            "<eos>": 2,
            "<unk>": 3,
            "alpha": 4,
            "beta": 5,
            "target": 6,
            "other": 7,
        }
        backend = tokenizers.Tokenizer(
            tokenizers.models.WordLevel(vocabulary, unk_token="<unk>")
        )
        backend.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
        tokenizer = transformers.PreTrainedTokenizerFast(  # type: ignore[attr-defined]
            tokenizer_object=backend,
            bos_token="<bos>",
            eos_token="<eos>",
            pad_token="<pad>",
            unk_token="<unk>",
        )
        config = transformers.Qwen3Config(  # type: ignore[attr-defined]
            vocab_size=8,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            max_position_embeddings=64,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
            use_cache=False,
            tie_word_embeddings=False,
        )
        return transformers.Qwen3ForCausalLM(config), tokenizer  # type: ignore[attr-defined]

    monkeypatch.setattr(
        qwen3_profile, "load_model_and_tokenizer", load_model_and_tokenizer
    )


def test_hf_preparation_creates_closed_distinct_transaction(tmp_path: Path) -> None:
    paths, anchors = integration._prepare_workspace(
        tmp_path / "hf",
        integration="hf-transformers",
        runtime_image_digest=ZERO_DIGEST,
    )

    request = yaml.safe_load(paths.request.read_text(encoding="utf-8"))
    records = (
        (paths.evaluation / "inputs/records.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    assert len(records) == 50
    parsed_records = [json.loads(record) for record in records]
    assert len({record["prompt"] for record in parsed_records}) == 50
    assert {record["expected"] for record in parsed_records} == {" target"}
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        paths.evaluation / "models/baseline", local_files_only=True
    )
    for record in parsed_records:
        prompt_ids = tokenizer(record["prompt"], add_special_tokens=True).input_ids
        target_ids = tokenizer(record["expected"], add_special_tokens=False).input_ids
        decoded_prompt = tokenizer.decode(
            prompt_ids,
            clean_up_tokenization_spaces=False,
            skip_special_tokens=False,
        )
        decoded_continuation = tokenizer.decode(
            prompt_ids + target_ids,
            clean_up_tokenization_spaces=False,
            skip_special_tokens=False,
        )
        assert decoded_continuation == decoded_prompt + record["expected"]
    assert request["comparison"]["dataset"]["name"] == "hf-transformers-smoke"
    assert request["comparison"]["baseline"]["runtime"]["provider"] == "hf_transformers"
    assert request["comparison"]["subject"]["runtime"]["provider"] == "hf_transformers"
    baseline_locator = request["comparison"]["baseline"]["artifact"]["locator"]
    subject_locator = request["comparison"]["subject"]["artifact"]["locator"]
    assert baseline_locator.startswith(
        f"hf://{qwen3_profile.MODEL_ID}@{qwen3_profile.MODEL_REVISION}#"
    )
    assert subject_locator.startswith(
        "generated://invarlock-example/hf-transformers-subject@sha256:"
    )
    summary = json.loads(
        (paths.evaluation / "inputs/subject-transformation.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["source_model_id"] == qwen3_profile.MODEL_ID
    assert summary["source_model_revision"] == qwen3_profile.MODEL_REVISION
    assert summary["method"] == "causal-output-row-fit"
    assert anchors["baseline_artifact_digest"] != anchors["subject_artifact_digest"]
    assert anchors["baseline_runtime_digest"] == ZERO_DIGEST
    assert paths.evidence_key.stat().st_mode & 0o777 == 0o600
    assert paths.verifier_key.stat().st_mode & 0o777 == 0o600
    for checkpoint in ("baseline", "subject"):
        model_root = paths.evaluation / "models" / checkpoint
        assert model_root.stat().st_mode & 0o005 == 0o005
        assert all(
            path.stat().st_mode & (0o005 if path.is_dir() else 0o004)
            for path in model_root.rglob("*")
        )


def test_hf_preparation_uses_caller_owned_external_trust_material(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence_key = tmp_path / "evidence.pem"
    verifier_key = tmp_path / "verifier.pem"
    integration._write_private_key(evidence_key)
    integration._write_private_key(verifier_key)
    trust_root = tmp_path / "trust" / "hf"
    trust_root.parent.mkdir()

    paths, _anchors = integration._prepare_workspace(
        tmp_path / "external",
        integration="hf-transformers",
        runtime_image_digest=ZERO_DIGEST,
        evidence_signing_key=evidence_key,
        verifier_signing_key=verifier_key,
        trust_root=trust_root,
        ephemeral_trust_root=False,
    )

    assert paths.evidence_key == evidence_key.resolve()
    assert paths.verifier_key == trust_root / "verifier.pem"
    assert paths.trusted_inputs == trust_root / "trusted-inputs.json"
    assert paths.independent_policy == trust_root / "policy/acceptance.json"
    assert not (paths.root / "keys").exists()
    assert not (paths.root / "verifier").exists()
    assert json.loads(paths.independent_policy.read_text()) == json.loads(
        (paths.evaluation / "inputs/acceptance.json").read_text()
    )

    with pytest.raises(ValueError, match="trust root must be new"):
        integration._prepare_workspace(
            tmp_path / "existing-trust-rejected",
            integration="hf-transformers",
            runtime_image_digest=ZERO_DIGEST,
            evidence_signing_key=evidence_key,
            verifier_signing_key=verifier_key,
            trust_root=trust_root,
            ephemeral_trust_root=False,
        )

    monkeypatch.setattr(
        integration,
        "create_trust_material",
        lambda **_kwargs: SimpleNamespace(
            trusted_inputs=tmp_path / "unexpected-trusted-inputs.json"
        ),
    )
    mismatched_trust_root = tmp_path / "trust" / "mismatched"
    with pytest.raises(ValueError, match="unexpected root"):
        integration._prepare_workspace(
            tmp_path / "mismatched-trust-material",
            integration="hf-transformers",
            runtime_image_digest=ZERO_DIGEST,
            evidence_signing_key=evidence_key,
            verifier_signing_key=verifier_key,
            trust_root=mismatched_trust_root,
            ephemeral_trust_root=False,
        )
    assert not (tmp_path / "mismatched-trust-material").exists()


def test_external_trust_material_rejects_intermediate_symlink_aliases(
    tmp_path: Path,
) -> None:
    transaction = tmp_path / "transaction"
    transaction.mkdir()
    inside = transaction / "inside"
    inside.mkdir()
    key_inside = inside / "key.pem"
    integration._write_private_key(key_inside)
    key_alias = tmp_path / "key-alias"
    key_alias.symlink_to(inside, target_is_directory=True)

    with pytest.raises(ValueError, match="Ed25519 private key"):
        trust_material.load_external_key(
            key_alias / key_inside.name,
            transaction_root=transaction,
            label="evidence signing key",
        )

    trust_alias = tmp_path / "trust-alias"
    trust_alias.symlink_to(inside, target_is_directory=True)
    with pytest.raises(ValueError, match="without symlinks"):
        trust_material.create_trust_material(
            transaction_root=transaction,
            evidence_key=tmp_path / "evidence.pem",
            verifier_key_bytes=b"verifier",
            evidence_fingerprint="evidence",
            verifier_fingerprint="verifier",
            trust_root=trust_alias / "profile",
            policy_bytes=b"{}\n",
            verifier_identity="test/verifier",
            anchors={},
        )
    assert not (inside / "profile").exists()


def test_hf_preparation_can_author_an_exact_match_transaction(tmp_path: Path) -> None:
    paths, _anchors = integration._prepare_workspace(
        tmp_path / "hf-exact-match",
        integration="hf-transformers",
        runtime_image_digest=ZERO_DIGEST,
        metric="exact_match",
    )

    request = yaml.safe_load(paths.request.read_text(encoding="utf-8"))
    records = [
        json.loads(line)
        for line in (paths.evaluation / "inputs/records.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    policy = json.loads(
        (paths.evaluation / "inputs/acceptance.json").read_text(encoding="utf-8")
    )
    assert {record["expected"] for record in records} == {"target"}
    from transformers import AutoModelForCausalLM, AutoTokenizer

    generated_outputs: dict[str, set[str]] = {}
    for role in ("baseline", "subject"):
        checkpoint = paths.evaluation / "models" / role
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(checkpoint, local_files_only=True)
        encoded = tokenizer(
            [record["prompt"] for record in records],
            add_special_tokens=False,
            padding=True,
            return_tensors="pt",
        )
        generated = model.generate(
            **encoded,
            do_sample=False,
            max_new_tokens=1,
            pad_token_id=tokenizer.pad_token_id,
        )
        generated_outputs[role] = set(
            tokenizer.batch_decode(
                generated[:, -1:],
                clean_up_tokenization_spaces=False,
                skip_special_tokens=True,
            )
        )
    assert generated_outputs["baseline"] != {"target"}
    assert generated_outputs["subject"] == {"target"}
    assert request["comparison"]["metric"] == "exact_match"
    assert policy == {
        "resolved_policy": {
            "metrics": {
                "exact_match": {
                    "delta_min_pp": 0.0,
                    "maximum_interval_width_pp": 20.0,
                    "minimum_record_count": 50,
                }
            }
        }
    }


def test_peft_preparation_trains_serializes_reloads_and_merges(tmp_path: Path) -> None:
    paths, anchors = integration._prepare_workspace(
        tmp_path / "peft",
        integration="peft-lora",
        runtime_image_digest=ZERO_DIGEST,
    )

    summary = json.loads(
        (paths.evaluation / "inputs/subject-transformation.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["library"] == "peft"
    assert summary["library_version"] == "0.20.0"
    assert summary["source_model_id"] == qwen3_profile.MODEL_ID
    assert summary["source_model_revision"] == qwen3_profile.MODEL_REVISION
    assert summary["target_modules"] == ["q_proj", "v_proj"]
    assert summary["training_record_count"] == 50
    assert summary["training_steps"] == 12
    assert summary["final_loss"] < summary["initial_loss"]
    assert (paths.root / "upstream/peft-adapter/adapter_model.safetensors").is_file()
    request = yaml.safe_load(paths.request.read_text(encoding="utf-8"))
    assert request["observations"] == [
        {
            "id": "peft-lora-subject-transformation",
            "kind": "artifact_transformation",
            "scope": "subject",
            "path": "inputs/subject-transformation.json",
        }
    ]
    assert anchors["baseline_artifact_digest"] != anchors["subject_artifact_digest"]


def test_torchao_preparation_quantizes_and_materializes_subject(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tiny_loader = qwen3_profile.load_model_and_tokenizer

    def load_bfloat16_model(
        *, torch: object, transformers: object
    ) -> tuple[object, object]:
        model, tokenizer = tiny_loader(torch=torch, transformers=transformers)
        return model.to(dtype=torch.bfloat16), tokenizer  # type: ignore[attr-defined]

    monkeypatch.setattr(qwen3_profile, "load_model_and_tokenizer", load_bfloat16_model)
    paths, anchors = integration._prepare_workspace(
        tmp_path / "torchao",
        integration="torchao-int8",
        runtime_image_digest=ZERO_DIGEST,
    )

    summary = json.loads(
        (paths.evaluation / "inputs/subject-transformation.json").read_text(
            encoding="utf-8"
        )
    )
    policy = json.loads(
        (paths.evaluation / "inputs/acceptance.json").read_text(encoding="utf-8")
    )
    assert summary["library"] == "torchao"
    assert summary["library_version"] == "0.17.0"
    assert summary["source_model_id"] == qwen3_profile.MODEL_ID
    assert summary["source_model_revision"] == qwen3_profile.MODEL_REVISION
    assert summary["quantization"] == {
        "configuration": "Int8WeightOnlyConfig(version=2)",
        "excluded_modules": ["lm_head"],
        "materialization": "dequantize-dense-state-v1",
        "selected_module_type": "torch.nn.Linear",
    }
    assert len(summary["quantized_tensors"]) >= 7
    assert "lm_head.weight" not in summary["quantized_tensors"]
    assert summary["quantized_tensor_count"] == len(summary["quantized_tensors"])
    import torch as pytorch
    from transformers import AutoModelForCausalLM

    persisted = AutoModelForCausalLM.from_pretrained(
        paths.evaluation / "models/subject", local_files_only=True
    ).state_dict()
    commitment = hashlib.sha256()
    for name in summary["quantized_tensors"]:
        tensor = persisted[name].contiguous()
        descriptor = integration.canonical_json_bytes(
            {
                "dtype": str(tensor.dtype),
                "name": name,
                "shape": list(tensor.shape),
            }
        )
        payload = tensor.view(pytorch.uint8).numpy().tobytes(order="C")
        commitment.update(len(descriptor).to_bytes(8, "big"))
        commitment.update(descriptor)
        commitment.update(len(payload).to_bytes(8, "big"))
        commitment.update(payload)
    assert summary["dequantized_tensor_state_sha256"] == (
        "sha256:" + commitment.hexdigest()
    )
    assert summary["dequantized_tensor_state_loaded_exact"] is True
    assert summary["dequantized_tensor_state_save_reload_exact"] is True
    observation = summary["live_kernel_observation"]
    assert observation["authority"] == "observation"
    assert observation["device"] == "cpu"
    assert observation["record_count"] == 50
    assert 0 <= observation["top1_agreement_count"] <= 50
    assert math.isfinite(observation["max_abs_next_token_logit_delta"])
    assert math.isfinite(observation["mean_abs_next_token_logit_delta"])
    assert (
        observation["max_abs_next_token_logit_delta"]
        >= (observation["mean_abs_next_token_logit_delta"])
    )
    assert observation["mean_abs_next_token_logit_delta"] >= 0.0
    request = yaml.safe_load(paths.request.read_text(encoding="utf-8"))
    assert request["observations"][0]["id"] == ("torchao-int8-subject-transformation")
    assert (
        policy["resolved_policy"]["metrics"]["normalized_nll_per_utf8_byte"][
            "ratio_max"
        ]
        == 1.01
    )
    assert anchors["baseline_artifact_digest"] != anchors["subject_artifact_digest"]


def test_torchao_preparation_rejects_nonfinite_materialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tiny_loader = qwen3_profile.load_model_and_tokenizer

    def load_nonfinite_model(
        *, torch: object, transformers: object
    ) -> tuple[object, object]:
        model, tokenizer = tiny_loader(torch=torch, transformers=transformers)
        with torch.no_grad():  # type: ignore[attr-defined]
            model.model.embed_tokens.weight[0, 0] = float("nan")
        return model, tokenizer

    monkeypatch.setattr(qwen3_profile, "load_model_and_tokenizer", load_nonfinite_model)
    with pytest.raises(RuntimeError, match="produced a non-finite tensor"):
        integration._prepare_workspace(
            tmp_path / "torchao-nonfinite",
            integration="torchao-int8",
            runtime_image_digest=ZERO_DIGEST,
        )


@pytest.mark.parametrize(
    ("case", "message"),
    (
        ("tokenizer", "baseline and subject tokenizers do not match"),
        ("identical", "subject is identical to its baseline"),
    ),
)
def test_hf_checkpoint_authoring_rejects_identity_inconsistencies(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    message: str,
) -> None:
    paths = integration._paths(tmp_path / case)
    monkeypatch.setattr(
        qwen3_profile,
        "load_model_and_tokenizer",
        lambda **_kwargs: (object(), object()),
    )
    tokenizer_digests = iter(
        ("tokenizer-a", "tokenizer-b" if case == "tokenizer" else "tokenizer-a")
    )
    monkeypatch.setattr(
        qwen3_profile,
        "save_checkpoint",
        lambda *_args: next(tokenizer_digests),
    )
    checkpoint_digests = iter(
        ("checkpoint-a", "checkpoint-a" if case == "identical" else "checkpoint-b")
    )
    monkeypatch.setattr(
        integration,
        "checkpoint_tree_sha256",
        lambda *_args: next(checkpoint_digests),
    )
    monkeypatch.setattr(
        integration,
        "_target_row_derivative",
        lambda *_args, **_kwargs: (7, 1.0),
    )

    with pytest.raises(RuntimeError, match=message):
        integration._create_hf_checkpoints(paths, expected_output=" target")


@pytest.mark.parametrize(
    ("case", "message"),
    (
        ("no-trainable", "did not expose trainable"),
        ("initial", "non-finite initial loss"),
        ("step", "non-finite loss"),
        ("final", "non-finite final loss"),
        ("no-improvement", "did not improve"),
        ("tokenizer", "baseline and subject tokenizers do not match"),
        ("identical", "subject is identical to its baseline"),
    ),
)
def test_peft_checkpoint_authoring_rejects_training_and_identity_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    message: str,
) -> None:
    import peft
    import torch
    import transformers

    parameter = torch.nn.Parameter(torch.ones(1))

    class BaselineModel:
        def parameters(self):
            return iter((parameter,))

        def to(self, _device: object) -> BaselineModel:
            return self

    if case == "initial":
        losses: list[object] = [None]
    elif case == "step":
        losses = [torch.tensor(1.0, requires_grad=True), None]
    elif case == "final":
        losses = (
            [torch.tensor(1.0, requires_grad=True)]
            + [torch.tensor(0.5, requires_grad=True) for _ in range(12)]
            + [None]
        )
    else:
        final_loss = 1.0 if case == "no-improvement" else 0.25
        losses = (
            [torch.tensor(1.0, requires_grad=True)]
            + [torch.tensor(0.5, requires_grad=True) for _ in range(12)]
            + [torch.tensor(final_loss, requires_grad=True)]
        )

    class TrainingModel(BaselineModel):
        def __init__(self) -> None:
            self._losses = iter(losses)
            self.peft_config = {"default": SimpleNamespace(base_model_name_or_path="")}

        def parameters(self):
            if case == "no-trainable":
                return iter(())
            return super().parameters()

        def train(self) -> TrainingModel:
            return self

        def eval(self) -> TrainingModel:
            return self

        def __call__(self, **_kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(loss=next(self._losses))

        def save_pretrained(self, path: Path, **_kwargs: object) -> None:
            path.mkdir(parents=True)

    class ReloadedModel:
        def merge_and_unload(self) -> BaselineModel:
            return BaselineModel()

    class TransferValue:
        def to(self, _device: object) -> TransferValue:
            return self

    class Optimizer:
        def zero_grad(self, **_kwargs: object) -> None:
            return None

        def step(self) -> None:
            return None

    training_model = TrainingModel()
    monkeypatch.setattr(
        qwen3_profile,
        "load_model_and_tokenizer",
        lambda **_kwargs: (BaselineModel(), object()),
    )
    tokenizer_digests = iter(
        ("tokenizer-a", "tokenizer-b" if case == "tokenizer" else "tokenizer-a")
    )
    monkeypatch.setattr(
        qwen3_profile,
        "save_checkpoint",
        lambda *_args: next(tokenizer_digests),
    )
    checkpoint_digests = iter(
        ("checkpoint-a", "checkpoint-a" if case == "identical" else "checkpoint-b")
    )
    monkeypatch.setattr(
        integration,
        "checkpoint_tree_sha256",
        lambda *_args: next(checkpoint_digests),
    )
    monkeypatch.setattr(
        integration,
        "_continuation_training_batch",
        lambda *_args, **_kwargs: {"input_ids": TransferValue()},
    )
    monkeypatch.setattr(
        peft, "get_peft_model", lambda *_args, **_kwargs: training_model
    )
    monkeypatch.setattr(
        peft.PeftModel,
        "from_pretrained",
        lambda *_args, **_kwargs: ReloadedModel(),
    )
    monkeypatch.setattr(
        transformers.AutoModelForCausalLM,
        "from_pretrained",
        lambda *_args, **_kwargs: BaselineModel(),
    )
    monkeypatch.setattr(torch.optim, "AdamW", lambda *_args, **_kwargs: Optimizer())
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda *_args: None)
    cleared: list[bool] = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: case == "tokenizer")
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: cleared.append(True))

    with pytest.raises(RuntimeError, match=message):
        integration._create_peft_checkpoints(integration._paths(tmp_path / case))
    if case == "tokenizer":
        assert cleared == [True]


@pytest.mark.parametrize(
    ("case", "message"),
    (
        ("no-quantized", "did not create any quantized tensors"),
        ("incomplete", "state is incomplete"),
        ("mismatch", "does not preserve TorchAO materialization"),
    ),
)
def test_torchao_checkpoint_authoring_rejects_invalid_materialization_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    message: str,
) -> None:
    import torchao.quantization
    import transformers

    if case == "no-quantized":
        monkeypatch.setattr(
            torchao.quantization,
            "quantize_",
            lambda *_args, **_kwargs: None,
        )
    else:

        class SubjectModel:
            def eval(self) -> SubjectModel:
                return self

            def load_state_dict(
                self, _state: object, *, strict: bool
            ) -> SimpleNamespace:
                assert strict is True
                missing = ["model.layers.0.mlp.down_proj.weight"]
                return SimpleNamespace(
                    missing_keys=missing if case == "incomplete" else [],
                    unexpected_keys=[],
                )

            def state_dict(self) -> dict[str, object]:
                return {}

        monkeypatch.setattr(
            transformers.AutoModelForCausalLM,
            "from_pretrained",
            lambda *_args, **_kwargs: SubjectModel(),
        )

    with pytest.raises(RuntimeError, match=message):
        integration._create_torchao_checkpoints(integration._paths(tmp_path / case))


@pytest.mark.parametrize(
    ("case", "message"),
    (
        ("tokenizer", "baseline and subject tokenizers do not match"),
        ("identical", "subject is identical to baseline"),
        ("nonfinite-logits", "probe produced non-finite logits"),
        ("nonfinite-delta", "probe produced a non-finite difference"),
        ("persisted", "saved checkpoint does not preserve TorchAO materialization"),
    ),
)
def test_torchao_checkpoint_authoring_rejects_post_materialization_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    message: str,
) -> None:
    import torch
    import transformers

    if case == "tokenizer":
        save_checkpoint = qwen3_profile.save_checkpoint
        save_count = 0

        def save_with_tokenizer_drift(*args: object, **kwargs: object) -> str:
            nonlocal save_count
            digest = save_checkpoint(*args, **kwargs)  # type: ignore[arg-type]
            save_count += 1
            return digest if save_count == 1 else "changed-tokenizer"

        monkeypatch.setattr(
            qwen3_profile,
            "save_checkpoint",
            save_with_tokenizer_drift,
        )
    elif case == "identical":
        monkeypatch.setattr(
            integration,
            "checkpoint_tree_sha256",
            lambda _path: "identical-checkpoint",
        )
    elif case in {"nonfinite-logits", "nonfinite-delta"}:
        isfinite = torch.isfinite

        def reject_probe_tensor(value: object) -> object:
            tensor = value  # preserve the real tensor protocol for this boundary test
            if case == "nonfinite-logits" and tensor.ndim == 3:  # type: ignore[attr-defined]
                return torch.zeros_like(tensor, dtype=torch.bool)  # type: ignore[arg-type]
            if (
                case == "nonfinite-delta"
                and tensor.ndim == 2  # type: ignore[attr-defined]
                and tensor.shape[0] == 50  # type: ignore[attr-defined]
            ):
                return torch.zeros_like(tensor, dtype=torch.bool)  # type: ignore[arg-type]
            return isfinite(tensor)  # type: ignore[arg-type]

        monkeypatch.setattr(torch, "isfinite", reject_probe_tensor)
    else:
        load_model = transformers.AutoModelForCausalLM.from_pretrained
        load_count = 0

        def load_with_persisted_drift(*args: object, **kwargs: object) -> object:
            nonlocal load_count
            load_count += 1
            if load_count == 2:

                class PersistedModel:
                    def eval(self) -> PersistedModel:
                        return self

                    def state_dict(self) -> dict[str, object]:
                        return {}

                return PersistedModel()
            return load_model(*args, **kwargs)

        monkeypatch.setattr(
            transformers.AutoModelForCausalLM,
            "from_pretrained",
            load_with_persisted_drift,
        )

    with pytest.raises(RuntimeError, match=message):
        integration._create_torchao_checkpoints(integration._paths(tmp_path / case))


def test_preparation_records_absence_of_transformation_observation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def checkpoints(
        paths: integration.ExamplePaths, _integration: str, **_kwargs: object
    ):
        created: dict[str, Path] = {}
        for role in ("baseline", "subject"):
            checkpoint = paths.evaluation / "models" / role
            checkpoint.mkdir(parents=True)
            (checkpoint / "config.json").write_text("{}\n", encoding="utf-8")
            created[role] = checkpoint
        return created, "a" * 64

    monkeypatch.setattr(integration, "_create_checkpoints", checkpoints)
    paths, _anchors = integration._prepare_workspace(
        tmp_path / "no-transformation",
        integration="hf-transformers",
        runtime_image_digest=ZERO_DIGEST,
    )

    request = yaml.safe_load(paths.request.read_text(encoding="utf-8"))
    assert "observations" not in request


def test_preparation_rejects_existing_and_unknown_workspace(tmp_path: Path) -> None:
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError, match="workspace already exists"):
        integration._prepare_workspace(
            existing,
            integration="hf-transformers",
            runtime_image_digest=ZERO_DIGEST,
        )
    paths = integration._paths(tmp_path / "unused")
    with pytest.raises(RuntimeError, match="unsupported integration"):
        integration._create_checkpoints(paths, "unknown")
    with pytest.raises(ValueError, match="unsupported comparison metric"):
        integration._prepare_workspace(
            tmp_path / "invalid-metric",
            integration="hf-transformers",
            runtime_image_digest=ZERO_DIGEST,
            metric="accuracy",
        )


def test_preparation_rejects_incoherent_trust_modes(tmp_path: Path) -> None:
    key = tmp_path / "key.pem"
    integration._write_private_key(key)
    common = {
        "integration": "hf-transformers",
        "runtime_image_digest": ZERO_DIGEST,
    }

    with pytest.raises(ValueError, match="must be supplied together"):
        integration._prepare_workspace(
            tmp_path / "partial-trust",
            evidence_signing_key=key,
            **common,
        )
    with pytest.raises(ValueError, match="cannot use ephemeral mode"):
        integration._prepare_workspace(
            tmp_path / "mixed-trust",
            evidence_signing_key=key,
            verifier_signing_key=key,
            trust_root=tmp_path / "mixed-trust-root",
            **common,
        )
    with pytest.raises(ValueError, match="caller-owned"):
        integration._prepare_workspace(
            tmp_path / "missing-trust",
            ephemeral_trust_root=False,
            **common,
        )
    with pytest.raises(ValueError, match="must be distinct"):
        integration._prepare_workspace(
            tmp_path / "shared-key",
            evidence_signing_key=key,
            verifier_signing_key=key,
            trust_root=tmp_path / "shared-key-trust",
            ephemeral_trust_root=False,
            **common,
        )


def test_hf_subject_transformation_rejects_a_nonpositive_margin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch
    import transformers

    model, tokenizer = qwen3_profile.load_model_and_tokenizer(
        torch=torch,
        transformers=transformers,
    )
    with torch.no_grad():
        model.lm_head.weight.zero_()
    monkeypatch.setattr(
        torch.linalg,
        "lstsq",
        lambda rows, _desired: SimpleNamespace(
            solution=torch.zeros((rows.shape[1], 1))
        ),
    )

    with pytest.raises(RuntimeError, match="missed a prompt"):
        integration._target_row_derivative(
            model,
            tokenizer,
            torch,
            expected_output=" target",
        )


@pytest.mark.parametrize(
    ("creator", "missing", "message"),
    (
        (
            integration._create_hf_checkpoints,
            "safetensors",
            "Hugging Face dependencies",
        ),
        (integration._create_peft_checkpoints, "peft", "PEFT example requires"),
        (
            integration._create_torchao_checkpoints,
            "torchao",
            "TorchAO example requires",
        ),
    ),
)
def test_checkpoint_authors_report_missing_optional_dependencies(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    creator: object,
    missing: str,
    message: str,
) -> None:
    real_import = builtins.__import__

    def import_without_optional(
        name: str,
        globals: object = None,
        locals: object = None,
        fromlist: object = (),
        level: int = 0,
    ) -> object:
        if name == missing:
            raise ImportError(f"missing {missing}")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", import_without_optional)
    paths = integration._paths(tmp_path / missing)
    with pytest.raises(RuntimeError, match=message):
        if creator is integration._create_hf_checkpoints:
            creator(paths, expected_output=" target")  # type: ignore[operator]
        else:
            creator(paths)  # type: ignore[operator]


def test_execute_invokes_public_commands_and_checks_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = integration._paths(tmp_path)
    report = paths.evidence / "reports/evaluation.report.json"
    report.parent.mkdir(parents=True)
    report.write_text(
        json.dumps({"comparison": {"value": 0.75}, "verdict": "pass"}),
        encoding="utf-8",
    )
    commands: list[list[str]] = []
    monkeypatch.setattr(integration, "_run", lambda command: commands.append(command))

    integration._execute(
        paths,
        container_engine="docker",
        runtime_image="example:current",
        runtime_image_digest=ZERO_DIGEST,
        runtime_device="cpu",
    )

    assert [command[3] for command in commands] == ["evaluate", "verify", "report"]
    assert "--trust-profile" in commands[1]
    assert "--html" in commands[2]

    report.write_text(
        json.dumps({"comparison": {"value": "bad"}, "verdict": "pass"}),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="did not produce a passing ratio"):
        integration._execute(
            paths,
            container_engine="docker",
            runtime_image="example:current",
            runtime_image_digest=ZERO_DIGEST,
            runtime_device="cpu",
        )


def test_command_runner_surfaces_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        integration,
        "run_bounded_command",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 3, stdout="out\n", stderr="bad\n"
        ),
    )
    with pytest.raises(RuntimeError, match="status 3"):
        integration._run(["false"])


def test_command_runner_accepts_success_and_handles_empty_diagnostic(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        integration,
        "run_bounded_command",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 0, stdout="done\n", stderr=""
        ),
    )
    integration._run(["true"])
    assert "done" in capsys.readouterr().out

    monkeypatch.setattr(
        integration,
        "run_bounded_command",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 3, stdout="", stderr=""
        ),
    )
    with pytest.raises(RuntimeError, match="status 3"):
        integration._run(["false"])


def test_run_main_prepare_only_and_input_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared = integration.main(
        [
            "hf-transformers",
            "--workspace",
            str(tmp_path / "prepared"),
            "--runtime-image-digest",
            ZERO_DIGEST,
            "--prepare-only",
            "--ephemeral-trust-root",
        ]
    )
    assert prepared == 0

    existing = tmp_path / "existing"
    existing.mkdir()
    assert (
        integration.main(
            [
                "hf-transformers",
                "--workspace",
                str(existing),
                "--runtime-image-digest",
                ZERO_DIGEST,
                "--prepare-only",
                "--ephemeral-trust-root",
            ]
        )
        == 2
    )
    missing = tmp_path / "missing-prepared"
    linked = tmp_path / "linked-prepared"
    linked.symlink_to(missing, target_is_directory=True)
    assert (
        integration.main(
            [
                "hf-transformers",
                "--workspace",
                str(linked),
                "--runtime-image-digest",
                ZERO_DIGEST,
                "--prepare-only",
                "--ephemeral-trust-root",
            ]
        )
        == 2
    )
    assert not missing.exists()
    with pytest.raises(SystemExit, match="full execution requires"):
        integration.main(
            [
                "hf-transformers",
                "--workspace",
                str(tmp_path / "missing-image"),
                "--runtime-image-digest",
                ZERO_DIGEST,
                "--ephemeral-trust-root",
            ]
        )


def test_run_main_rejects_incoherent_trust_arguments(tmp_path: Path) -> None:
    common = [
        "hf-transformers",
        "--workspace",
        str(tmp_path / "workspace"),
        "--runtime-image-digest",
        ZERO_DIGEST,
        "--prepare-only",
    ]
    key = tmp_path / "key.pem"
    trust_root = tmp_path / "trust"

    with pytest.raises(SystemExit, match="must be supplied together"):
        integration.main([*common, "--evidence-signing-key", str(key)])
    with pytest.raises(SystemExit, match="caller-owned"):
        integration.main(common)
    with pytest.raises(SystemExit, match="cannot be combined"):
        integration.main(
            [
                *common,
                "--evidence-signing-key",
                str(key),
                "--verifier-signing-key",
                str(key),
                "--trust-root",
                str(trust_root),
                "--ephemeral-trust-root",
            ]
        )


def test_run_main_executes_the_prepared_transaction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed: list[Path] = []
    monkeypatch.setattr(
        integration,
        "_execute",
        lambda paths, **_kwargs: observed.append(paths.root),
    )
    workspace = tmp_path / "executed"
    assert (
        integration.main(
            [
                "hf-transformers",
                "--workspace",
                str(workspace),
                "--runtime-image",
                "runtime@" + ZERO_DIGEST,
                "--runtime-image-digest",
                ZERO_DIGEST,
                "--ephemeral-trust-root",
            ]
        )
        == 0
    )
    assert observed == [workspace]


def test_integration_helpers_fail_closed_on_malformed_model_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class BadTargetTokenizer:
        def __call__(self, *_args: object, **_kwargs: object) -> dict[str, object]:
            return {"input_ids": [1, 2]}

    with pytest.raises(RuntimeError, match="one-token expected continuation"):
        integration._single_target_id(BadTargetTokenizer(), " target")

    class NonTensorTokenizer:
        def __call__(self, *_args: object, **_kwargs: object) -> dict[str, object]:
            return {"attention_mask": [], "input_ids": []}

    with pytest.raises(RuntimeError, match="did not return tensors"):
        integration._prompt_batch(
            NonTensorTokenizer(), object(), expected_output=" target"
        )

    class EmptyPromptTokenizer:
        pad_token_id = 0

        def __call__(self, value: object, **_kwargs: object) -> dict[str, object]:
            return {"input_ids": [7] if value == " target" else []}

    with pytest.raises(RuntimeError, match="empty prompt"):
        integration._continuation_training_batch(
            EmptyPromptTokenizer(), object(), expected_output=" target"
        )

    monkeypatch.setattr(
        integration,
        "_prompt_batch",
        lambda *_args, **_kwargs: {
            "input_ids": object(),
            "attention_mask": object(),
        },
    )
    model = type(
        "Model", (), {"base_model_prefix": "missing", "eval": lambda self: None}
    )()
    with pytest.raises(RuntimeError, match="does not expose its causal backbone"):
        integration._target_row_derivative(
            model, object(), object(), expected_output=" target"
        )

    with pytest.raises(ValueError, match="exact-match preparation"):
        integration._prepare_workspace(
            tmp_path / "invalid-metric-integration",
            integration="peft-lora",
            runtime_image_digest=ZERO_DIGEST,
            metric="exact_match",
        )


def test_launch_helpers_require_committed_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    replies = iter(("a" * 40, ""))
    monkeypatch.setattr(launch, "_git", lambda *args: next(replies))
    assert launch._require_committed_checkout(tmp_path) == "a" * 40

    replies = iter(("b" * 40, " M tracked.py"))
    monkeypatch.setattr(launch, "_git", lambda *args: next(replies))
    with pytest.raises(RuntimeError, match="commit or stash"):
        launch._require_committed_checkout(tmp_path)


def test_runtime_image_builds_from_authenticated_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    build = tmp_path / "build"
    build.mkdir()
    commands: list[list[str]] = []
    monkeypatch.setattr(launch, "_require_committed_checkout", lambda _repo: "c" * 40)
    monkeypatch.setattr(launch, "_git", lambda *args: "1234567890")

    def write_statement(command: list[str]) -> None:
        statement = Path(command[command.index("--statement") + 1])
        statement.write_text(
            json.dumps(
                {
                    "base_image": None,
                    "build_arguments": {},
                    "dockerfile": {"path": "runtime/Dockerfile", "sha256": ZERO_DIGEST},
                    "format_version": "invarlock/runtime-image-build-v1",
                    "image": "custom-example:" + "c" * 12,
                    "ok": True,
                    "platform": None,
                    "runtime_image_id": "sha256:" + "d" * 64,
                    "source_bundle_sha256": ZERO_DIGEST,
                    "source_commit": "c" * 40,
                }
            ),
            encoding="utf-8",
        )

    def fake_run(
        command: list[str], *, cwd: Path, capture_output: bool = False
    ) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        if "qualification_source.py" in " ".join(command):
            output = json.dumps({"source_bundle_sha256": ZERO_DIGEST})
        elif "authenticated_runtime_build.py" in " ".join(command):
            write_statement(command)
            output = ""
        elif command[1:4] == ["image", "inspect", "--format"]:
            output = "sha256:" + ("d" * 64) + "\t" + ("c" * 40) + "\t" + ZERO_DIGEST
        else:
            output = ""
        return subprocess.CompletedProcess(command, 0, stdout=output, stderr="")

    monkeypatch.setattr(launch, "_run", fake_run)
    image, digest = launch._runtime_image(
        repository=repository,
        build_root=build,
        container_engine="docker",
        dockerfile="addins/example/Dockerfile",
        image_prefix="custom-example",
    )
    assert image == "sha256:" + ("d" * 64)
    assert digest == "sha256:" + ("d" * 64)
    assert any("authenticated_runtime_build.py" in " ".join(item) for item in commands)
    build_command = next(
        item for item in commands if "authenticated_runtime_build.py" in " ".join(item)
    )
    assert "addins/example/Dockerfile" in build_command
    assert "custom-example:" + "c" * 12 in build_command
    assert build_command[build_command.index("--statement") + 1] == str(
        build / "runtime-build.json"
    )
    assert not any(
        command[:4] == ["docker", "image", "inspect", "--format"]
        and command[-1] == "custom-example:" + "c" * 12
        for command in commands
    )


def test_runtime_image_rejects_source_label_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    build = tmp_path / "build"
    build.mkdir()
    monkeypatch.setattr(launch, "_require_committed_checkout", lambda _repo: "c" * 40)
    monkeypatch.setattr(launch, "_git", lambda *args: "1234567890")

    def fake_run(
        command: list[str], *, cwd: Path, capture_output: bool = False
    ) -> subprocess.CompletedProcess[str]:
        if "qualification_source.py" in " ".join(command):
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=json.dumps({"source_bundle_sha256": ZERO_DIGEST}),
                stderr="",
            )
        if "authenticated_runtime_build.py" in " ".join(command):
            Path(command[command.index("--statement") + 1]).write_text(
                json.dumps(
                    {
                        "base_image": None,
                        "build_arguments": {},
                        "dockerfile": {
                            "path": "runtime/Dockerfile",
                            "sha256": ZERO_DIGEST,
                        },
                        "format_version": "invarlock/runtime-image-build-v1",
                        "image": "invarlock-example-runtime:" + "c" * 12,
                        "ok": True,
                        "platform": None,
                        "runtime_image_id": "sha256:" + "d" * 64,
                        "source_bundle_sha256": ZERO_DIGEST,
                        "source_commit": "c" * 40,
                    }
                ),
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="sha256:" + "d" * 64 + "\t" + "e" * 40 + "\t" + ZERO_DIGEST,
            stderr="",
        )

    monkeypatch.setattr(launch, "_run", fake_run)
    with pytest.raises(RuntimeError, match="not bound to the source"):
        launch._runtime_image(
            repository=repository,
            build_root=build,
            container_engine="docker",
        )


def test_runtime_image_authenticates_a_layered_base(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    build = tmp_path / "build"
    build.mkdir()
    commands: list[list[str]] = []
    base = "registry.example/runtime@sha256:" + ("a" * 64)
    monkeypatch.setattr(launch, "_require_committed_checkout", lambda _repo: "c" * 40)
    monkeypatch.setattr(launch, "_git", lambda *args: "1234567890")

    def write_statement(command: list[str]) -> None:
        statement = Path(command[command.index("--statement") + 1])
        statement.write_text(
            json.dumps(
                {
                    "base_image": base,
                    "build_arguments": {},
                    "dockerfile": {"path": "runtime/Dockerfile", "sha256": ZERO_DIGEST},
                    "format_version": "invarlock/runtime-image-build-v1",
                    "image": "invarlock-example-runtime:" + "c" * 12,
                    "ok": True,
                    "platform": None,
                    "runtime_image_id": "sha256:" + "d" * 64,
                    "source_bundle_sha256": ZERO_DIGEST,
                    "source_commit": "c" * 40,
                }
            ),
            encoding="utf-8",
        )

    def fake_run(
        command: list[str], *, cwd: Path, capture_output: bool = False
    ) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        if "qualification_source.py" in " ".join(command):
            output = json.dumps({"source_bundle_sha256": ZERO_DIGEST})
        elif "authenticated_runtime_build.py" in " ".join(command):
            write_statement(command)
            output = ""
        elif command[1:4] == ["image", "inspect", "--format"]:
            output = "sha256:" + ("d" * 64) + "\t" + ("c" * 40) + "\t" + ZERO_DIGEST
        else:
            output = ""
        return subprocess.CompletedProcess(command, 0, stdout=output, stderr="")

    monkeypatch.setattr(launch, "_run", fake_run)
    launch._runtime_image(
        repository=repository,
        build_root=build,
        container_engine="docker",
        dockerfile="addins/multimodal/runtime/Dockerfile",
        authenticated_base_image=base,
    )
    command = next(
        item for item in commands if "authenticated_runtime_build.py" in " ".join(item)
    )
    assert command[command.index("--require-base-source-labels") + 1] == base
    assert f"RUNTIME_BASE_IMAGE={base}" in command


def test_local_image_publication_is_digest_bound_and_disposable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commands: list[list[str]] = []
    digest = "sha256:" + ("a" * 64)
    endpoint = "127.0.0.1:49152"
    volume = "invarlock-example-registry-fixed"
    container = volume
    published = f"{endpoint}/example-runtime:{'a' * 12}"
    canonical = f"{endpoint}/example-runtime@{digest}"
    monkeypatch.setattr(local_registry.secrets, "token_hex", lambda _length: "fixed")

    def fake_run(
        command: list[str], *, cwd: Path, capture_output: bool = False
    ) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        if command[1:3] == ["volume", "create"]:
            output = volume
        elif command[1] == "run":
            output = "b" * 64
        elif command[1] == "port":
            output = endpoint
        elif command[1:3] == ["image", "inspect"]:
            output = f'{digest} ["{canonical}"]'
        else:
            output = ""
        return subprocess.CompletedProcess(command, 0, stdout=output, stderr="")

    monkeypatch.setattr(local_registry, "_run", fake_run)
    with local_registry.published_local_image(
        repository=tmp_path,
        container_engine="docker",
        image=digest,
        image_digest=digest,
        repository_name="example-runtime",
    ) as reference:
        assert reference == canonical
        assert commands[-1][1] == "image"
    run = next(command for command in commands if command[1] == "run")
    assert local_registry.REGISTRY_IMAGE in run
    assert "--read-only" in run
    assert "--cap-drop=ALL" in run
    assert "127.0.0.1::5000" in run
    assert ["docker", "push", published] in commands
    assert commands[-3:] == [
        ["docker", "image", "remove", published],
        ["docker", "container", "rm", "--force", container],
        ["docker", "volume", "rm", "--force", volume],
    ]


def test_local_registry_command_runner_reports_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    del monkeypatch
    success = [sys.executable, "-c", "print('ready', end='')"]
    stderr = [
        sys.executable,
        "-c",
        "import sys; print('denied', file=sys.stderr); sys.exit(9)",
    ]
    silent = [sys.executable, "-c", "import sys; sys.exit(7)"]
    assert (
        local_registry._run(success, cwd=tmp_path, capture_output=True).stdout
        == "ready"
    )
    with pytest.raises(RuntimeError, match=r"status 9: .*denied"):
        local_registry._run(stderr, cwd=tmp_path)
    with pytest.raises(RuntimeError, match=r"status 7:"):
        local_registry._run(silent, cwd=tmp_path)
    with pytest.raises(RuntimeError, match="could not start"):
        local_registry._run([str(tmp_path / "missing-command")], cwd=tmp_path)


def test_local_image_publication_cleans_nothing_when_volume_creation_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commands: list[list[str]] = []
    digest = "sha256:" + ("a" * 64)
    monkeypatch.setattr(local_registry.secrets, "token_hex", lambda _length: "fixed")

    def fail_first_command(
        command: list[str], *, cwd: Path, capture_output: bool = False
    ) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        raise OSError("volume creation failed")

    monkeypatch.setattr(local_registry, "_run", fail_first_command)
    with pytest.raises(OSError, match="volume creation failed"):
        with local_registry.published_local_image(
            repository=tmp_path,
            container_engine="docker",
            image=digest,
            image_digest=digest,
            repository_name="example-runtime",
        ):
            raise AssertionError("publication must fail before yielding")
    assert commands == [
        ["docker", "volume", "create", "invarlock-example-registry-fixed"]
    ]


@pytest.mark.parametrize(
    ("failure", "message"),
    (
        ("volume", "unexpected registry volume"),
        ("container", "invalid registry identity"),
        ("endpoint", "invalid registry endpoint"),
        ("large-port", "invalid registry endpoint"),
        ("identity", "changed identity"),
        ("json", "digests are unreadable"),
        ("json-object", "invalid shape"),
        ("digest", "lacks one canonical digest"),
    ),
)
def test_local_image_publication_fails_closed_and_cleans_up(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
    message: str,
) -> None:
    commands: list[list[str]] = []
    digest = "sha256:" + ("a" * 64)
    volume = "invarlock-example-registry-fixed"
    endpoint = "127.0.0.1:49152"
    monkeypatch.setattr(local_registry.secrets, "token_hex", lambda _length: "fixed")

    def fake_run(
        command: list[str], *, cwd: Path, capture_output: bool = False
    ) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        if command[1:3] == ["volume", "create"]:
            output = "wrong" if failure == "volume" else volume
        elif command[1] == "run":
            output = "invalid" if failure == "container" else "b" * 64
        elif command[1] == "port":
            if failure == "endpoint":
                output = "0.0.0.0:49152"
            elif failure == "large-port":
                output = "127.0.0.1:99999"
            else:
                output = endpoint
        elif command[1:3] == ["image", "inspect"]:
            observed = "sha256:" + ("b" * 64) if failure == "identity" else digest
            if failure == "json":
                details = "not-json"
            elif failure == "json-object":
                details = "{}"
            elif failure == "digest":
                details = "[]"
            else:
                details = json.dumps([f"{endpoint}/example-runtime@{digest}"])
            output = f"{observed} {details}"
        else:
            output = ""
        return subprocess.CompletedProcess(command, 0, stdout=output, stderr="")

    monkeypatch.setattr(local_registry, "_run", fake_run)
    with pytest.raises(RuntimeError, match=message):
        with local_registry.published_local_image(
            repository=tmp_path,
            container_engine="docker",
            image=digest,
            image_digest=digest,
            repository_name="example-runtime",
        ):
            raise AssertionError("publication must fail before yielding")
    assert commands[-1] == ["docker", "volume", "rm", "--force", volume]


def test_local_image_publication_supports_podman_and_cleanup_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commands: list[list[str]] = []
    digest = "sha256:" + ("a" * 64)
    endpoint = "127.0.0.1:49152"
    canonical = f"{endpoint}/example-runtime@{digest}"
    monkeypatch.setattr(local_registry.secrets, "token_hex", lambda _length: "fixed")

    def fake_run(
        command: list[str], *, cwd: Path, capture_output: bool = False
    ) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        if "remove" in command or "rm" in command:
            raise RuntimeError("cleanup failed")
        if command[1:3] == ["volume", "create"]:
            output = "invarlock-example-registry-fixed"
        elif command[1] == "run":
            output = "b" * 64
        elif command[1] == "port":
            output = endpoint
        elif command[1:3] == ["image", "inspect"]:
            output = f'{digest} ["{canonical}"]'
        else:
            output = ""
        return subprocess.CompletedProcess(command, 0, stdout=output, stderr="")

    monkeypatch.setattr(local_registry, "_run", fake_run)
    with local_registry.published_local_image(
        repository=tmp_path,
        container_engine="podman",
        image=digest,
        image_digest=digest,
        repository_name="example-runtime",
    ) as reference:
        assert reference == canonical
    assert [
        "podman",
        "push",
        "--tls-verify=false",
        f"{endpoint}/example-runtime:aaaaaaaaaaaa",
    ] in commands
    local_registry._cleanup_command(["podman", "rm", "missing"], cwd=tmp_path)


def test_local_image_publication_rejects_invalid_input_digest(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="sha256 image digest"):
        with local_registry.published_local_image(
            repository=tmp_path,
            container_engine="docker",
            image="not-an-image",
            image_digest="not-a-digest",
            repository_name="example-runtime",
        ):
            raise AssertionError("invalid digest must fail before yielding")


def test_runtime_image_rejects_non_digest_engine_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    build = tmp_path / "build"
    build.mkdir()
    monkeypatch.setattr(launch, "_require_committed_checkout", lambda _repo: "c" * 40)
    monkeypatch.setattr(launch, "_git", lambda *args: "1234567890")

    def invalid_inspect(
        command: list[str], *, cwd: Path, capture_output: bool = False
    ) -> subprocess.CompletedProcess[str]:
        if "qualification_source.py" in " ".join(command):
            output = json.dumps({"source_bundle_sha256": ZERO_DIGEST})
        elif "authenticated_runtime_build.py" in " ".join(command):
            statement = Path(command[command.index("--statement") + 1])
            statement.write_text(
                json.dumps(
                    {
                        "base_image": None,
                        "build_arguments": {},
                        "dockerfile": {
                            "path": "runtime/Dockerfile",
                            "sha256": ZERO_DIGEST,
                        },
                        "format_version": "invarlock/runtime-image-build-v1",
                        "image": "invarlock-example-runtime:" + "c" * 12,
                        "ok": True,
                        "platform": None,
                        "runtime_image_id": "not-a-digest",
                        "source_bundle_sha256": ZERO_DIGEST,
                        "source_commit": "c" * 40,
                    }
                ),
                encoding="utf-8",
            )
            output = ""
        elif command[1:4] == ["image", "inspect", "--format"]:
            output = "not-a-digest"
        else:
            output = ""
        return subprocess.CompletedProcess(command, 0, stdout=output, stderr="")

    monkeypatch.setattr(launch, "_run", invalid_inspect)
    with pytest.raises(RuntimeError, match="runtime build statement does not bind"):
        launch._runtime_image(
            repository=repository,
            build_root=build,
            container_engine="docker",
        )


def test_launch_main_dispatches_prepare_and_full_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commands: list[list[str]] = []

    def fake_run(
        command: list[str], *, cwd: Path, capture_output: bool = False
    ) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(launch, "_run", fake_run)
    assert (
        launch.main(
            [
                "hf-transformers",
                "--prepare-only",
                "--workspace",
                str(tmp_path / "prepare"),
                "--ephemeral-trust-root",
            ]
        )
        == 0
    )
    assert "--prepare-only" in commands[-1]

    disposable = tmp_path / "generated"
    disposable.mkdir()
    monkeypatch.setattr(
        launch.tempfile,
        "mkdtemp",
        lambda *, prefix: str(disposable),
    )
    assert (
        launch.main(["hf-transformers", "--prepare-only", "--ephemeral-trust-root"])
        == 0
    )
    assert commands[-1][commands[-1].index("--workspace") + 1] == str(
        disposable / "transaction"
    )

    runtime_calls: list[dict[str, object]] = []

    def fake_runtime(**kwargs: object) -> tuple[str, str]:
        runtime_calls.append(kwargs)
        return "example:current", "sha256:" + ("e" * 64)

    monkeypatch.setattr(launch, "_runtime_image", fake_runtime)
    assert (
        launch.main(
            [
                "peft-lora",
                "--workspace",
                str(tmp_path / "full"),
                "--runtime-device",
                "cuda:1",
                "--ephemeral-trust-root",
            ]
        )
        == 0
    )
    assert "example:current" in commands[-1]
    assert runtime_calls[0]["dockerfile"] == "runtime/Dockerfile.cuda"
    assert runtime_calls[0]["image_prefix"] == "invarlock-example-runtime-cuda"
    assert commands[-1][commands[-1].index("--runtime-device") + 1] == "cuda:1"

    with pytest.raises(SystemExit):
        launch._parser().parse_args(["hf-transformers", "--runtime-device", "gpu:1"])

    existing = tmp_path / "existing"
    existing.mkdir()
    assert (
        launch.main(
            [
                "hf-transformers",
                "--prepare-only",
                "--workspace",
                str(existing),
                "--ephemeral-trust-root",
            ]
        )
        == 2
    )
    missing = tmp_path / "missing-workspace"
    linked = tmp_path / "linked-workspace"
    linked.symlink_to(missing, target_is_directory=True)
    assert (
        launch.main(
            [
                "hf-transformers",
                "--prepare-only",
                "--workspace",
                str(linked),
                "--ephemeral-trust-root",
            ]
        )
        == 2
    )
    assert not missing.exists()


def test_launch_runner_reports_subprocess_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    del monkeypatch
    failing = [
        sys.executable,
        "-c",
        "import sys; print('failed', file=sys.stderr); sys.exit(7)",
    ]
    with pytest.raises(RuntimeError, match="status 7"):
        launch._run(failing, cwd=tmp_path)

    silent = [sys.executable, "-c", "import sys; sys.exit(9)"]
    with pytest.raises(RuntimeError, match="status 9:"):
        launch._run(silent, cwd=tmp_path)


def test_shared_and_registry_runners_bound_output_and_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(launch, "_COMMAND_STDOUT_LIMIT", 8)
    with pytest.raises(RuntimeError, match="stdout limit exceeded"):
        launch._run(
            [sys.executable, "-c", "print('x' * 100)"],
            cwd=tmp_path,
            capture_output=True,
        )
    monkeypatch.setattr(launch, "_COMMAND_TIMEOUT_SECONDS", 1)
    with pytest.raises(RuntimeError, match="timed out"):
        launch._run([sys.executable, "-c", "import time; time.sleep(2)"], cwd=tmp_path)

    monkeypatch.setattr(local_registry, "_COMMAND_OUTPUT_LIMIT", 8)
    with pytest.raises(RuntimeError, match="stdout limit exceeded"):
        local_registry._run(
            [sys.executable, "-c", "print('x' * 100)"],
            cwd=tmp_path,
            capture_output=True,
        )
    monkeypatch.setattr(local_registry, "_COMMAND_TIMEOUT_SECONDS", 1)
    with pytest.raises(RuntimeError, match="timed out"):
        local_registry._run(
            [sys.executable, "-c", "import time; time.sleep(2)"], cwd=tmp_path
        )


def test_shared_runner_success_capture_modes_and_start_failure(tmp_path: Path) -> None:
    command = [
        sys.executable,
        "-c",
        "import sys; print('out'); print('err', file=sys.stderr)",
    ]
    uncaptured = launch._run(command, cwd=tmp_path)
    assert uncaptured.returncode == 0
    assert uncaptured.stdout is None
    captured = launch._run(command, cwd=tmp_path, capture_output=True)
    assert captured.stdout == "out\n"
    assert captured.stderr == "err\n"
    with pytest.raises(RuntimeError, match="could not start"):
        launch._run([str(tmp_path / "missing-command")], cwd=tmp_path)


def test_runtime_image_requires_source_bundle_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    build = tmp_path / "build"
    build.mkdir()
    monkeypatch.setattr(launch, "_require_committed_checkout", lambda _repo: "c" * 40)
    monkeypatch.setattr(
        launch,
        "_run",
        lambda command, **_kwargs: subprocess.CompletedProcess(
            command, 0, stdout="{}", stderr=""
        ),
    )

    with pytest.raises(RuntimeError, match="did not return its digest"):
        launch._runtime_image(
            repository=repository,
            build_root=build,
            container_engine="docker",
        )


def test_qwen3_profile_uses_an_immutable_official_revision() -> None:
    assert qwen3_profile.MODEL_ID == "Qwen/Qwen3-0.6B"
    assert qwen3_profile.MODEL_REVISION == ("c1899de289a04d12100db370d81485cdf75e47ca")
    assert qwen3_profile.PEFT_TARGET_MODULES == ("q_proj", "v_proj")


def test_qwen3_loader_passes_the_pin_to_model_and_tokenizer() -> None:
    calls: list[tuple[str, str, dict[str, object]]] = []

    class Tokenizer:
        pad_token_id = 0
        eos_token_id = 2

    class Config:
        use_cache = True

    class Model:
        config = Config()

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs: object) -> Tokenizer:
            calls.append(("tokenizer", model_id, kwargs))
            return Tokenizer()

    class AutoModel:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs: object) -> Model:
            calls.append(("model", model_id, kwargs))
            return Model()

    Transformers = type(
        "Transformers",
        (),
        {"AutoTokenizer": AutoTokenizer, "AutoModelForCausalLM": AutoModel},
    )

    model, _tokenizer = REAL_QWEN3_LOADER(torch=object(), transformers=Transformers)
    assert model.config.use_cache is False
    assert [call[0] for call in calls] == ["tokenizer", "model"]
    assert all(call[1] == qwen3_profile.MODEL_ID for call in calls)
    assert all(call[2]["revision"] == qwen3_profile.MODEL_REVISION for call in calls)
    assert all(call[2]["trust_remote_code"] is False for call in calls)
    assert calls[1][2]["use_safetensors"] is True
    assert calls[1][2]["dtype"] == "auto"


def test_qwen3_loader_uses_eos_for_padding_and_rejects_missing_tokens() -> None:
    class Config:
        use_cache = True

    class Model:
        config = Config()

    class AutoModel:
        @staticmethod
        def from_pretrained(*_args: object, **_kwargs: object) -> Model:
            return Model()

    class Tokenizer:
        pad_token_id = None
        eos_token_id: int | None = 2
        pad_token: object = None
        eos_token = "</s>"

    tokenizer = Tokenizer()
    transformers = type(
        "Transformers",
        (),
        {
            "AutoTokenizer": type(
                "AutoTokenizer",
                (),
                {"from_pretrained": staticmethod(lambda *_args, **_kwargs: tokenizer)},
            ),
            "AutoModelForCausalLM": AutoModel,
        },
    )
    _model, observed = REAL_QWEN3_LOADER(torch=object(), transformers=transformers)
    assert observed.pad_token == "</s>"

    tokenizer.eos_token_id = None
    with pytest.raises(RuntimeError, match="no padding token"):
        REAL_QWEN3_LOADER(torch=object(), transformers=transformers)
