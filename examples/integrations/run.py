#!/usr/bin/env python3
"""Run one maintained Qwen3 Hugging Face ecosystem integration journey.

Every journey starts from one official revision-pinned Qwen3 checkpoint. The
Transformers journey creates an explicit behavioral derivative, the PEFT
journey trains, saves, reloads, and merges a LoRA adapter, and the TorchAO
journey applies INT8 weight-only quantization before materializing a portable
checkpoint. All three finish through evaluate, verify, and report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import stat
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

try:
    from examples.integrations.bounded_command import run_bounded_command
except ModuleNotFoundError as exc:  # pragma: no cover - flat-script compatibility
    if exc.name != "examples":
        raise
    from bounded_command import run_bounded_command  # type: ignore[no-redef]

try:
    from examples.integrations import qwen3_profile
except ModuleNotFoundError as exc:
    if exc.name != "examples":
        raise
    # Direct script execution places this directory, rather than the repository
    # root, on sys.path. Keep the maintained one-command entry point usable with
    # the same PYTHONPATH=src boundary as an installed core package.
    import qwen3_profile  # type: ignore[no-redef]
try:
    from examples.integrations.trust_material import (
        create_trust_material,
        load_external_key,
        validate_new_trust_root,
    )
except ModuleNotFoundError as exc:
    if exc.name != "examples":
        raise
    from trust_material import (  # type: ignore[no-redef]
        create_trust_material,
        load_external_key,
        validate_new_trust_root,
    )
from invarlock.core.checkpoint_identity import checkpoint_tree_sha256
from invarlock.core.runtime_provider import ModelRuntimeSpec, artifact_identity_sha256
from invarlock.core.runtime_provider.types import JSONScalar
from invarlock.core.schedule_preparation import (
    LocalDatasetRequest,
    prepare_local_evaluation_schedule_bytes,
)
from invarlock.evidence_pack_contract import canonical_json_bytes, normalize_digest
from invarlock.evidence_pack_integrity import public_key_fingerprint
from invarlock.runtime_providers.hf_transformers import HFTransformersProvider

_SEED = 20_260_716
_INTEGRATIONS = ("hf-transformers", "peft-lora", "torchao-int8")
_METRICS = ("normalized_nll_per_utf8_byte", "exact_match")


@dataclass(frozen=True)
class ExamplePaths:
    """The paths shared by preparation and the three public commands."""

    root: Path
    evaluation: Path
    request: Path
    evidence: Path
    verifier: Path
    trusted_inputs: Path
    independent_policy: Path
    evidence_key: Path
    verifier_key: Path
    receipt: Path
    html_report: Path


def _paths(
    root: Path,
    *,
    evidence_key: Path | None = None,
    trust_root: Path | None = None,
) -> ExamplePaths:
    evaluation = root / "evaluation"
    verifier = trust_root or root / "verifier"
    receipt_root = root / "verifier-output" if trust_root is not None else verifier
    return ExamplePaths(
        root=root,
        evaluation=evaluation,
        request=evaluation / "request.yaml",
        evidence=evaluation / "evidence",
        verifier=verifier,
        trusted_inputs=verifier / "trusted-inputs.json",
        independent_policy=verifier / "policy" / "acceptance.json",
        evidence_key=evidence_key or root / "keys" / "evidence-signer.pem",
        verifier_key=(
            verifier / "verifier.pem"
            if trust_root is not None
            else verifier / "keys" / "verifier.pem"
        ),
        receipt=receipt_root / "verification.receipt.json",
        html_report=root / "comparison-report.html",
    )


def _write_private_key(path: Path) -> str:
    key = ed25519.Ed25519PrivateKey.generate()
    path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    return public_key_fingerprint(key.public_key())


def _example_records(*, expected_output: str = " target") -> tuple[dict[str, str], ...]:
    """Return 50 distinct contexts for one expected continuation."""

    tokens = ("alpha", "beta", "other")
    records: list[dict[str, str]] = []
    for index in range(50):
        value = index
        prompt_tokens: list[str] = []
        for _position in range(4):
            prompt_tokens.append(tokens[value % len(tokens)])
            value //= len(tokens)
        records.append(
            {
                "expected": expected_output,
                "id": f"expected-token-{index:02d}",
                "prompt": " ".join(reversed(prompt_tokens)),
            }
        )
    return tuple(records)


def _single_target_id(tokenizer: Any, expected_output: str) -> int:
    encoded = tokenizer(expected_output, add_special_tokens=False)
    token_ids = encoded["input_ids"]
    if not isinstance(token_ids, list) or len(token_ids) != 1:
        raise RuntimeError(
            "the pinned Qwen3 profile requires a one-token expected continuation"
        )
    return int(token_ids[0])


def _prompt_batch(
    tokenizer: Any, torch: Any, *, expected_output: str
) -> dict[str, Any]:
    records = _example_records(expected_output=expected_output)
    encoded = tokenizer(
        [record["prompt"] for record in records],
        add_special_tokens=True,
        padding=True,
        return_tensors="pt",
    )
    if not hasattr(encoded["input_ids"], "shape"):
        raise RuntimeError("the pinned Qwen3 tokenizer did not return tensors")
    target_id = _single_target_id(tokenizer, expected_output)
    targets = torch.full((len(records), 1), target_id, dtype=encoded["input_ids"].dtype)
    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "target_ids": targets,
    }


def _continuation_training_batch(
    tokenizer: Any,
    torch: Any,
    *,
    expected_output: str,
) -> dict[str, Any]:
    target_id = _single_target_id(tokenizer, expected_output)
    sequences: list[list[int]] = []
    prompt_lengths: list[int] = []
    for record in _example_records(expected_output=expected_output):
        prompt_ids = tokenizer(record["prompt"], add_special_tokens=True)["input_ids"]
        if not isinstance(prompt_ids, list) or not prompt_ids:
            raise RuntimeError("the pinned Qwen3 tokenizer returned an empty prompt")
        prompt_lengths.append(len(prompt_ids))
        sequences.append([int(value) for value in prompt_ids] + [target_id])
    width = max(len(sequence) for sequence in sequences)
    input_ids = torch.full(
        (len(sequences), width),
        int(tokenizer.pad_token_id),
        dtype=torch.long,
    )
    attention_mask = torch.zeros_like(input_ids)
    labels = torch.full_like(input_ids, -100)
    for index, sequence in enumerate(sequences):
        length = len(sequence)
        input_ids[index, :length] = torch.tensor(sequence, dtype=torch.long)
        attention_mask[index, :length] = 1
        labels[index, prompt_lengths[index]] = target_id
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def _target_row_derivative(
    model: Any,
    tokenizer: Any,
    torch: Any,
    *,
    expected_output: str,
) -> tuple[int, float]:
    """Fit one output row so every maintained prompt favors the target token."""

    batch = _prompt_batch(tokenizer, torch, expected_output=expected_output)
    model.eval()
    backbone = getattr(model, str(model.base_model_prefix), None)
    if backbone is None:
        raise RuntimeError("the Qwen3 model does not expose its causal backbone")
    with torch.inference_mode():
        hidden_states = backbone(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            return_dict=True,
            use_cache=False,
        ).last_hidden_state
        positions = torch.stack(
            [
                torch.nonzero(mask, as_tuple=False)[-1, 0]
                for mask in batch["attention_mask"]
            ]
        )
        rows = hidden_states[torch.arange(hidden_states.shape[0]), positions].float()
        logits = model.lm_head(rows.to(model.lm_head.weight.dtype)).float()
        target_id = int(batch["target_ids"][0, 0])
        logits[:, target_id] = -torch.inf
        desired = logits.max(dim=1).values + 12.0
        solution = torch.linalg.lstsq(rows, desired[:, None]).solution[:, 0]
        model.lm_head.weight[target_id].copy_(solution.to(model.lm_head.weight.dtype))
        revised = model.lm_head(rows.to(model.lm_head.weight.dtype)).float()
        revised[:, target_id] = -torch.inf
        margin = float(
            (
                model.lm_head(rows.to(model.lm_head.weight.dtype)).float()[:, target_id]
                - revised.max(dim=1).values
            )
            .min()
            .item()
        )
    if margin <= 0.0:
        raise RuntimeError("the Qwen3 subject transformation missed a prompt")
    return target_id, margin


def _create_hf_checkpoints(
    paths: ExamplePaths, *, expected_output: str
) -> tuple[dict[str, Path], str]:
    try:
        import safetensors  # noqa: F401
        import torch
        import transformers
    except ImportError as exc:
        raise RuntimeError(
            "the example requires the InvarLock Hugging Face dependencies"
        ) from exc

    torch.manual_seed(_SEED)
    model, tokenizer = qwen3_profile.load_model_and_tokenizer(
        torch=torch, transformers=transformers
    )
    baseline = paths.evaluation / "models" / "baseline"
    tokenizer_digest = qwen3_profile.save_checkpoint(model, tokenizer, baseline)
    baseline_digest = checkpoint_tree_sha256(baseline)
    target_id, margin = _target_row_derivative(
        model,
        tokenizer,
        torch,
        expected_output=expected_output,
    )
    subject = paths.evaluation / "models" / "subject"
    observed_digest = qwen3_profile.save_checkpoint(model, tokenizer, subject)
    if observed_digest != tokenizer_digest:
        raise RuntimeError("the Qwen3 baseline and subject tokenizers do not match")
    subject_digest = checkpoint_tree_sha256(subject)
    if baseline_digest == subject_digest:
        raise RuntimeError("the transformed Qwen3 subject is identical to its baseline")
    (paths.evaluation / "inputs" / "subject-transformation.json").write_bytes(
        canonical_json_bytes(
            {
                "format": "invarlock/example-hf-transformers-summary-v1",
                "library": "transformers",
                "library_version": str(transformers.__version__),
                **qwen3_profile.provenance(checkpoint_tree_sha256=baseline_digest),
                "method": "causal-output-row-fit",
                "expected_output": expected_output,
                "target_token_id": target_id,
                "minimum_logit_margin": margin,
                "subject_checkpoint_tree_sha256": subject_digest,
            }
        )
    )
    return {"baseline": baseline, "subject": subject}, tokenizer_digest


def _create_peft_checkpoints(paths: ExamplePaths) -> tuple[dict[str, Path], str]:
    try:
        import peft
        import safetensors  # noqa: F401
        import torch
        import transformers
    except ImportError as exc:
        raise RuntimeError(
            "the PEFT example requires the dependencies installed by "
            "`make example-peft-lora`"
        ) from exc

    torch.manual_seed(_SEED)
    torch.cuda.manual_seed_all(_SEED)
    baseline_model, tokenizer = qwen3_profile.load_model_and_tokenizer(
        torch=torch, transformers=transformers
    )
    baseline = paths.evaluation / "models" / "baseline"
    tokenizer_digest = qwen3_profile.save_checkpoint(
        baseline_model, tokenizer, baseline
    )
    baseline_digest = checkpoint_tree_sha256(baseline)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dtype = next(baseline_model.parameters()).dtype

    training_model = peft.get_peft_model(
        baseline_model.to(device),
        peft.LoraConfig(
            task_type=peft.TaskType.CAUSAL_LM,
            r=4,
            lora_alpha=8,
            lora_dropout=0.0,
            target_modules=list(qwen3_profile.PEFT_TARGET_MODULES),
            bias="none",
        ),
    )
    training_model.train()
    trainable = [
        parameter
        for parameter in training_model.parameters()
        if parameter.requires_grad
    ]
    if not trainable:
        raise RuntimeError("PEFT did not expose trainable LoRA parameters")
    batch = {
        name: value.to(device)
        for name, value in _continuation_training_batch(
            tokenizer, torch, expected_output=" target"
        ).items()
    }
    optimizer = torch.optim.AdamW(trainable, lr=0.002)

    training_model.eval()
    with torch.no_grad():
        initial = training_model(
            **batch,
        ).loss
    if initial is None or not torch.isfinite(initial):
        raise RuntimeError("PEFT training produced a non-finite initial loss")
    initial_loss = float(initial.detach())
    training_model.train()
    for _step in range(12):
        optimizer.zero_grad(set_to_none=True)
        loss = training_model(
            **batch,
        ).loss
        if loss is None or not torch.isfinite(loss):
            raise RuntimeError("PEFT training produced a non-finite loss")
        loss.backward()
        optimizer.step()
    training_model.eval()
    with torch.no_grad():
        final = training_model(
            **batch,
        ).loss
    if final is None or not torch.isfinite(final):
        raise RuntimeError("PEFT training produced a non-finite final loss")
    final_loss = float(final.detach())
    if final_loss >= initial_loss:
        raise RuntimeError("PEFT training did not improve the target-token loss")

    adapter = paths.root / "upstream" / "peft-adapter"
    adapter.parent.mkdir(parents=True)
    training_model.peft_config["default"].base_model_name_or_path = str(baseline)
    training_model.save_pretrained(adapter, safe_serialization=True)
    del training_model, baseline_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    reloaded_base = transformers.AutoModelForCausalLM.from_pretrained(
        baseline,
        local_files_only=True,
        dtype=model_dtype,
        trust_remote_code=False,
    ).to(device)
    reloaded = peft.PeftModel.from_pretrained(
        reloaded_base, adapter, is_trainable=False
    )
    subject_model = reloaded.merge_and_unload().to("cpu")
    subject = paths.evaluation / "models" / "subject"
    observed_digest = qwen3_profile.save_checkpoint(subject_model, tokenizer, subject)
    if observed_digest != tokenizer_digest:
        raise RuntimeError("the PEFT baseline and subject tokenizers do not match")
    subject_digest = checkpoint_tree_sha256(subject)
    if baseline_digest == subject_digest:
        raise RuntimeError("the merged PEFT subject is identical to its baseline")
    (paths.evaluation / "inputs" / "subject-transformation.json").write_bytes(
        canonical_json_bytes(
            {
                "format": "invarlock/example-peft-summary-v1",
                "library": "peft",
                "library_version": str(peft.__version__),
                **qwen3_profile.provenance(checkpoint_tree_sha256=baseline_digest),
                "target_modules": list(qwen3_profile.PEFT_TARGET_MODULES),
                "training_record_count": len(_example_records()),
                "training_steps": 12,
                "training_device": device.type,
                "initial_loss": initial_loss,
                "final_loss": final_loss,
                "merged_subject_sha256": subject_digest,
                "saved_adapter": "peft-adapter",
            }
        )
    )
    return {"baseline": baseline, "subject": subject}, tokenizer_digest


def _create_torchao_checkpoints(paths: ExamplePaths) -> tuple[dict[str, Path], str]:
    try:
        import safetensors  # noqa: F401
        import torch
        import torchao
        import transformers
        from torchao.quantization import Int8WeightOnlyConfig, quantize_
    except ImportError as exc:
        raise RuntimeError(
            "the TorchAO example requires the dependencies installed by "
            "`make example-torchao-int8`"
        ) from exc

    torch.manual_seed(_SEED)
    baseline_model, tokenizer = qwen3_profile.load_model_and_tokenizer(
        torch=torch, transformers=transformers
    )
    baseline = paths.evaluation / "models" / "baseline"
    tokenizer_digest = qwen3_profile.save_checkpoint(
        baseline_model, tokenizer, baseline
    )
    baseline_digest = checkpoint_tree_sha256(baseline)
    model_dtype = next(baseline_model.parameters()).dtype

    quantized_model = baseline_model.eval()

    def quantization_target(module: Any, fully_qualified_name: str) -> bool:
        # Qwen ties lm_head.weight to the token embedding. Quantizing only the
        # output projection would break that alias, and a dense HF checkpoint
        # cannot represent the two different values. Keep the tied output
        # unmodified so every transformed tensor can be materialized exactly.
        return isinstance(module, torch.nn.Linear) and fully_qualified_name != "lm_head"

    quantization_config = Int8WeightOnlyConfig(version=2)
    quantize_(
        quantized_model,
        quantization_config,
        filter_fn=quantization_target,
    )
    quantized_names = sorted(
        name
        for name, value in quantized_model.state_dict().items()
        if type(value).__module__.startswith("torchao.")
    )
    if not quantized_names:
        raise RuntimeError("TorchAO did not create any quantized tensors")
    if "lm_head.weight" in quantized_names:
        raise RuntimeError("TorchAO quantized the tied output projection")

    materialized_state = {
        name: (
            value.dequantize().detach().cpu().clone()
            if hasattr(value, "dequantize")
            else value.detach().cpu().clone()
        )
        for name, value in quantized_model.state_dict().items()
    }
    if any(
        value.is_floating_point() and not bool(torch.isfinite(value).all())
        for value in materialized_state.values()
    ):
        raise RuntimeError("TorchAO materialization produced a non-finite tensor")
    subject_model = transformers.AutoModelForCausalLM.from_pretrained(
        baseline,
        local_files_only=True,
        dtype=model_dtype,
        trust_remote_code=False,
    ).eval()
    loading = subject_model.load_state_dict(materialized_state, strict=True)
    if loading.missing_keys or loading.unexpected_keys:
        raise RuntimeError("the materialized TorchAO state is incomplete")
    loaded_state = subject_model.state_dict()
    loaded_mismatches = [
        name
        for name in quantized_names
        if name not in loaded_state
        or loaded_state[name].shape != materialized_state[name].shape
        or loaded_state[name].dtype != materialized_state[name].dtype
        or not torch.equal(loaded_state[name], materialized_state[name])
    ]
    if loaded_mismatches:
        raise RuntimeError(
            "the dense model does not preserve TorchAO materialization: "
            + ", ".join(loaded_mismatches[:5])
        )

    records = _example_records()
    probe = tokenizer(
        [record["prompt"] for record in records],
        add_special_tokens=True,
        padding=True,
        return_tensors="pt",
    )
    with torch.inference_mode():
        live_logits = quantized_model(**probe, use_cache=False).logits
        materialized_logits = subject_model(**probe, use_cache=False).logits
    if not bool(torch.isfinite(live_logits).all()) or not bool(
        torch.isfinite(materialized_logits).all()
    ):
        raise RuntimeError("TorchAO materialization probe produced non-finite logits")
    record_indices = torch.arange(len(records))
    final_positions = probe["attention_mask"].sum(dim=1) - 1
    live_next_token_logits = live_logits[record_indices, final_positions].float()
    materialized_next_token_logits = materialized_logits[
        record_indices, final_positions
    ].float()
    logit_delta = (live_next_token_logits - materialized_next_token_logits).abs()
    if not bool(torch.isfinite(logit_delta).all()):
        raise RuntimeError(
            "TorchAO materialization probe produced a non-finite difference"
        )
    max_abs_logit_delta = float(logit_delta.max())
    mean_abs_logit_delta = float(logit_delta.mean())
    top1_agreement_count = int(
        (
            live_next_token_logits.argmax(dim=-1)
            == materialized_next_token_logits.argmax(dim=-1)
        )
        .sum()
        .item()
    )

    materialized_digest = hashlib.sha256()
    for name in quantized_names:
        tensor = materialized_state[name].contiguous()
        descriptor = canonical_json_bytes(
            {
                "dtype": str(tensor.dtype),
                "name": name,
                "shape": list(tensor.shape),
            }
        )
        payload = tensor.view(torch.uint8).numpy().tobytes(order="C")
        materialized_digest.update(len(descriptor).to_bytes(8, "big"))
        materialized_digest.update(descriptor)
        materialized_digest.update(len(payload).to_bytes(8, "big"))
        materialized_digest.update(payload)

    subject = paths.evaluation / "models" / "subject"
    observed_digest = qwen3_profile.save_checkpoint(subject_model, tokenizer, subject)
    if observed_digest != tokenizer_digest:
        raise RuntimeError("the TorchAO baseline and subject tokenizers do not match")
    subject_digest = checkpoint_tree_sha256(subject)
    if baseline_digest == subject_digest:
        raise RuntimeError("the materialized TorchAO subject is identical to baseline")
    persisted_model = transformers.AutoModelForCausalLM.from_pretrained(
        subject,
        local_files_only=True,
        dtype=model_dtype,
        trust_remote_code=False,
    ).eval()
    persisted_state = persisted_model.state_dict()
    persisted_mismatches = [
        name
        for name in quantized_names
        if name not in persisted_state
        or persisted_state[name].shape != materialized_state[name].shape
        or persisted_state[name].dtype != materialized_state[name].dtype
        or not torch.equal(persisted_state[name], materialized_state[name])
    ]
    if persisted_mismatches:
        raise RuntimeError(
            "the saved checkpoint does not preserve TorchAO materialization: "
            + ", ".join(persisted_mismatches[:5])
        )
    (paths.evaluation / "inputs" / "subject-transformation.json").write_bytes(
        canonical_json_bytes(
            {
                "format": "invarlock/example-torchao-summary-v1",
                "library": "torchao",
                "library_version": str(torchao.__version__),
                "torch_version": str(torch.__version__),
                "transformers_version": str(transformers.__version__),
                **qwen3_profile.provenance(checkpoint_tree_sha256=baseline_digest),
                "quantization": {
                    "configuration": "Int8WeightOnlyConfig(version=2)",
                    "excluded_modules": ["lm_head"],
                    "materialization": "dequantize-dense-state-v1",
                    "selected_module_type": "torch.nn.Linear",
                },
                "quantized_tensors": quantized_names,
                "quantized_tensor_count": len(quantized_names),
                "dequantized_tensor_state_sha256": (
                    "sha256:" + materialized_digest.hexdigest()
                ),
                "dequantized_tensor_state_loaded_exact": True,
                "dequantized_tensor_state_save_reload_exact": True,
                "live_kernel_observation": {
                    "authority": "observation",
                    "device": "cpu",
                    "input_records_sha256": "sha256:"
                    + hashlib.sha256(
                        canonical_json_bytes({"records": list(records)})
                    ).hexdigest(),
                    "max_abs_next_token_logit_delta": max_abs_logit_delta,
                    "mean_abs_next_token_logit_delta": mean_abs_logit_delta,
                    "record_count": len(records),
                    "top1_agreement_count": top1_agreement_count,
                },
                "materialized_subject_sha256": subject_digest,
            }
        )
    )
    return {"baseline": baseline, "subject": subject}, tokenizer_digest


def _create_checkpoints(
    paths: ExamplePaths, integration: str, *, expected_output: str = " target"
) -> tuple[dict[str, Path], str]:
    if integration == "hf-transformers":
        return _create_hf_checkpoints(paths, expected_output=expected_output)
    if integration == "peft-lora":
        return _create_peft_checkpoints(paths)
    if integration == "torchao-int8":
        return _create_torchao_checkpoints(paths)
    raise RuntimeError(f"unsupported integration: {integration}")


def _settings(checkpoint: Path, tokenizer_digest: str) -> dict[str, JSONScalar]:
    return {
        "batch_size": 1,
        "checkpoint_tree_sha256": checkpoint_tree_sha256(checkpoint),
        "context_length": 32,
        "max_output_tokens": 1,
        "offline": True,
        "seed": _SEED,
        "timeout_seconds": 300,
        "tokenizer_metadata_sha256": tokenizer_digest,
    }


def _prepare_workspace(
    root: Path,
    *,
    integration: str,
    runtime_image_digest: str,
    metric: str = "normalized_nll_per_utf8_byte",
    evidence_signing_key: Path | None = None,
    verifier_signing_key: Path | None = None,
    trust_root: Path | None = None,
    ephemeral_trust_root: bool = True,
) -> tuple[ExamplePaths, dict[str, str]]:
    if metric not in _METRICS:
        raise ValueError(f"unsupported comparison metric: {metric}")
    if metric == "exact_match" and integration != "hf-transformers":
        raise ValueError("exact-match preparation requires hf-transformers")
    external_trust = (
        evidence_signing_key is not None
        or verifier_signing_key is not None
        or trust_root is not None
    )
    if external_trust and not all(
        value is not None
        for value in (evidence_signing_key, verifier_signing_key, trust_root)
    ):
        raise ValueError(
            "evidence key, verifier key, and trust root must be supplied together"
        )
    if external_trust and ephemeral_trust_root:
        raise ValueError("external trust material cannot use ephemeral mode")
    if not external_trust and not ephemeral_trust_root:
        raise ValueError(
            "caller-owned evidence/verifier keys and trust root are required"
        )
    root = root.expanduser().resolve()
    if root.exists():
        raise FileExistsError(
            f"workspace already exists: {root}; choose a new disposable path"
        )
    root.parent.mkdir(parents=True, exist_ok=True)
    root.mkdir()
    paths = _paths(
        root,
        evidence_key=(
            evidence_signing_key.expanduser().absolute() if external_trust else None
        ),
        trust_root=(trust_root.expanduser().absolute() if external_trust else None),
    )
    try:
        (paths.evaluation / "inputs").mkdir(parents=True)
        if external_trust:
            assert evidence_signing_key is not None
            assert verifier_signing_key is not None
            assert trust_root is not None
            trust_root = validate_new_trust_root(trust_root, transaction_root=root)
            evidence_key_path, evidence_key_bytes, evidence_signer = load_external_key(
                evidence_signing_key,
                transaction_root=root,
                label="evidence signing key",
            )
            _verifier_key_path, verifier_key_bytes, verifier = load_external_key(
                verifier_signing_key,
                transaction_root=root,
                label="verifier signing key",
            )
            if evidence_signer == verifier:
                raise ValueError("evidence and verifier signing keys must be distinct")
            paths = _paths(
                root,
                evidence_key=evidence_key_path,
                trust_root=trust_root.expanduser().absolute(),
            )
        else:
            paths.independent_policy.parent.mkdir(parents=True)
            paths.evidence_key.parent.mkdir(parents=True)
            paths.verifier_key.parent.mkdir(parents=True)
        paths.receipt.parent.mkdir(parents=True, exist_ok=True)

        expected_output = "target" if metric == "exact_match" else " target"
        checkpoints, tokenizer_digest = _create_checkpoints(
            paths,
            integration,
            expected_output=expected_output,
        )
        settings = {
            role: _settings(checkpoints[role], tokenizer_digest)
            for role in ("baseline", "subject")
        }

        dataset_bytes = b"".join(
            canonical_json_bytes(record)
            for record in _example_records(expected_output=expected_output)
        )
        dataset = paths.evaluation / "inputs" / "records.jsonl"
        dataset.write_bytes(dataset_bytes)
        dataset_sha256 = hashlib.sha256(dataset_bytes).hexdigest()
        dataset_name = (
            f"{integration}-exact-match-smoke"
            if metric == "exact_match"
            else f"{integration}-smoke"
        )

        schedule = prepare_local_evaluation_schedule_bytes(
            LocalDatasetRequest(
                path=dataset,
                sha256=dataset_sha256,
                name=dataset_name,
                split="validation",
                input_field="prompt",
                expected_output_field="expected",
                id_field="id",
            ),
            dataset_bytes,
        )

        if metric == "exact_match":
            policy = {
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
        else:
            ratio_max = 1.01 if integration == "torchao-int8" else 1.0
            policy = {
                "resolved_policy": {
                    "metrics": {
                        "normalized_nll_per_utf8_byte": {"ratio_max": ratio_max}
                    }
                }
            }
        policy_bytes = canonical_json_bytes(policy)
        request_policy = paths.evaluation / "inputs" / "acceptance.json"
        request_policy.write_bytes(policy_bytes)
        if not external_trust:
            paths.independent_policy.write_bytes(policy_bytes)

        def side(role: str) -> dict[str, object]:
            model_id = f"invarlock-example/{integration}-{role}"
            artifact_digest = settings[role]["checkpoint_tree_sha256"]
            locator = (
                f"hf://{qwen3_profile.MODEL_ID}@{qwen3_profile.MODEL_REVISION}"
                f"#checkpoint-tree-sha256:{artifact_digest}"
                if role == "baseline"
                else f"generated://{model_id}@sha256:{artifact_digest}"
            )
            return {
                "artifact": {
                    "path": f"models/{role}",
                    "model_id": model_id,
                    "locator": locator,
                },
                "runtime": {
                    "provider": "hf_transformers",
                    "settings": settings[role],
                },
            }

        request = {
            "format_version": "invarlock/evaluation-request-v1",
            "comparison": {
                "baseline": side("baseline"),
                "subject": side("subject"),
                "dataset": {
                    "path": "inputs/records.jsonl",
                    "sha256": dataset_sha256,
                    "format": "jsonl",
                    "name": dataset_name,
                    "split": "validation",
                    "input_field": "prompt",
                    "expected_output_field": "expected",
                    "id_field": "id",
                },
                "policy": "inputs/acceptance.json",
                "task": "text_causal",
                "metric": metric,
            },
            "execution": {"mode": "run"},
            "output": {"evidence": "evidence"},
        }
        transformation = paths.evaluation / "inputs" / "subject-transformation.json"
        if transformation.is_file():
            request["observations"] = [
                {
                    "id": f"{integration}-subject-transformation",
                    "kind": "artifact_transformation",
                    "scope": "subject",
                    "path": "inputs/subject-transformation.json",
                }
            ]
        paths.request.write_text(
            yaml.safe_dump(request, sort_keys=False), encoding="utf-8"
        )

        provider = HFTransformersProvider()
        artifact_anchors = {
            role: "sha256:"
            + artifact_identity_sha256(
                provider.identify_artifact(
                    ModelRuntimeSpec(
                        provider_name="hf_transformers",
                        model_id=f"invarlock-example/{integration}-{role}",
                        settings=settings[role],
                    )
                )
            )
            for role in ("baseline", "subject")
        }
        if external_trust:
            assert evidence_key_bytes is not None
            assert verifier_key_bytes is not None
            assert trust_root is not None
        else:
            evidence_signer = _write_private_key(paths.evidence_key)
            verifier = _write_private_key(paths.verifier_key)
            paths.evidence_key.with_suffix(".fingerprint").write_text(
                evidence_signer + "\n", encoding="ascii"
            )
            paths.verifier_key.with_suffix(".fingerprint").write_text(
                verifier + "\n", encoding="ascii"
            )
        anchors = {
            "baseline_artifact_digest": artifact_anchors["baseline"],
            "subject_artifact_digest": artifact_anchors["subject"],
            "schedule_digest": f"sha256:{schedule.schedule_sha256}",
            "baseline_runtime_digest": normalize_digest(
                runtime_image_digest, label="runtime image digest"
            ),
            "subject_runtime_digest": normalize_digest(
                runtime_image_digest, label="runtime image digest"
            ),
            "evidence_signer_fingerprint": evidence_signer,
        }
        trust_profile = {
            "format": "invarlock/trust-inputs-v1",
            "policy": {"path": "policy/acceptance.json"},
            "anchors": anchors,
            "verifier": {
                "identity": f"invarlock-example/{integration}-verifier",
                "signing_key_path": "keys/verifier.pem",
            },
            "allow_installed_scorers": False,
        }
        if external_trust:
            material = create_trust_material(
                transaction_root=root,
                evidence_key=paths.evidence_key,
                verifier_key_bytes=verifier_key_bytes,
                evidence_fingerprint=evidence_signer,
                verifier_fingerprint=verifier,
                trust_root=trust_root,
                policy_bytes=policy_bytes,
                verifier_identity=f"invarlock-example/{integration}-verifier",
                anchors=anchors,
            )
            if material.trusted_inputs != paths.trusted_inputs:
                raise ValueError(
                    "external trust material resolved to an unexpected root"
                )
        else:
            paths.trusted_inputs.write_bytes(canonical_json_bytes(trust_profile))
        return paths, anchors
    except Exception:
        shutil.rmtree(root)
        raise


def _run(command: list[str]) -> None:
    rendered = " ".join(command)
    print(f"+ {rendered}")
    completed = run_bounded_command(
        command,
        capture_output=True,
        label="Qwen3 integration command",
    )
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.returncode != 0:
        if completed.stderr:
            print(completed.stderr, file=sys.stderr, end="")
        raise RuntimeError(
            f"command exited with status {completed.returncode}: {rendered}"
        )


def _execute(
    paths: ExamplePaths,
    *,
    container_engine: str,
    runtime_image: str,
    runtime_image_digest: str,
    runtime_device: str,
) -> None:
    base = [sys.executable, "-m", "invarlock"]
    _run(
        [
            *base,
            "evaluate",
            str(paths.request),
            "--signing-key",
            str(paths.evidence_key),
            "--container-engine",
            container_engine,
            "--runtime-image",
            runtime_image,
            "--runtime-image-digest",
            runtime_image_digest,
            "--runtime-device",
            runtime_device,
            "--json",
        ]
    )
    _run(
        [
            *base,
            "verify",
            str(paths.evidence),
            "--trust-profile",
            str(paths.trusted_inputs),
            "--receipt",
            str(paths.receipt),
            "--json",
        ]
    )
    _run(
        [
            *base,
            "report",
            str(paths.evidence),
            "--html",
            str(paths.html_report),
        ]
    )
    report = json.loads(
        (paths.evidence / "reports" / "evaluation.report.json").read_text(
            encoding="utf-8"
        )
    )
    ratio = report.get("comparison", {}).get("value")
    if report.get("verdict") != "pass" or not isinstance(ratio, (int, float)):
        raise RuntimeError("the evaluated comparison did not produce a passing ratio")
    print(f"PASS subject normalized-NLL ratio: {ratio:.6f}")
    print(f"Evidence: {paths.evidence}")
    print(f"Receipt: {paths.receipt}")
    print(f"Report: {paths.html_report}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("integration", choices=_INTEGRATIONS)
    parser.add_argument(
        "--workspace",
        type=Path,
        required=True,
        help="New disposable workspace; an existing path is never overwritten.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Prepare inputs and print the canonical commands without running OCI.",
    )
    parser.add_argument(
        "--container-engine", choices=("docker", "podman"), default="docker"
    )
    parser.add_argument(
        "--runtime-image",
        help="Locally available CPU runtime image reference used by evaluate.",
    )
    parser.add_argument(
        "--runtime-image-digest",
        required=True,
        help=(
            "Pinned sha256 digest of the local CPU runtime image; required to "
            "close the generated verifier trust profile."
        ),
    )
    parser.add_argument("--evidence-signing-key", type=Path)
    parser.add_argument("--verifier-signing-key", type=Path)
    parser.add_argument("--trust-root", type=Path)
    parser.add_argument(
        "--ephemeral-trust-root",
        action="store_true",
        help="Use disposable generated keys; never use this mode for acceptance.",
    )
    parser.add_argument("--runtime-device", default="cpu")
    parser.add_argument(
        "--metric",
        choices=_METRICS,
        default="normalized_nll_per_utf8_byte",
        help="Metric used to author the fixed example dataset and policy.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    trust_values = (
        arguments.evidence_signing_key,
        arguments.verifier_signing_key,
        arguments.trust_root,
    )
    provided_trust = any(value is not None for value in trust_values)
    external_trust = all(value is not None for value in trust_values)
    if provided_trust and not external_trust:
        raise SystemExit(
            "--evidence-signing-key, --verifier-signing-key, and --trust-root "
            "must be supplied together"
        )
    if not external_trust and not arguments.ephemeral_trust_root:
        raise SystemExit(
            "caller-owned --evidence-signing-key, --verifier-signing-key, and "
            "--trust-root are required; use --ephemeral-trust-root only for a "
            "disposable non-acceptance demo"
        )
    if external_trust and arguments.ephemeral_trust_root:
        raise SystemExit(
            "--ephemeral-trust-root cannot be combined with caller-owned trust"
        )
    if not arguments.prepare_only and not arguments.runtime_image:
        raise SystemExit("full execution requires --runtime-image")
    try:
        paths, _anchors = _prepare_workspace(
            arguments.workspace,
            integration=arguments.integration,
            runtime_image_digest=arguments.runtime_image_digest,
            metric=arguments.metric,
            evidence_signing_key=arguments.evidence_signing_key,
            verifier_signing_key=arguments.verifier_signing_key,
            trust_root=arguments.trust_root,
            ephemeral_trust_root=arguments.ephemeral_trust_root,
        )
        print(f"Prepared: {paths.root}")
        print(f"Request: {paths.request}")
        print(f"Independent trust inputs: {paths.trusted_inputs}")
        print(f"Keys outside request tree: {paths.evidence_key.parent}")
        if arguments.prepare_only:
            print(
                "Preparation stops before execution. Run the maintained example "
                "command without --prepare-only to complete evaluate, verify, and "
                "report."
            )
            return 0
        _execute(
            paths,
            container_engine=arguments.container_engine,
            runtime_image=arguments.runtime_image,
            runtime_image_digest=arguments.runtime_image_digest,
            runtime_device=arguments.runtime_device,
        )
    except (FileExistsError, RuntimeError, OSError, ValueError) as exc:
        print(f"FAIL {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
