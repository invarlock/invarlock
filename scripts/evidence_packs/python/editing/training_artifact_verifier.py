"""Independent artifact recomputation for tiny training-profile subjects."""

from __future__ import annotations

import hashlib
import math
import os
import stat
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from invarlock.evidence_pack_json import (
    StrictJsonError,
    parse_json_bytes,
    read_regular_file_bytes,
)
from invarlock.peft_runtime import PeftRuntimeError, load_dense_peft_model
from invarlock.training_model_load import load_diagnostics_sha256

from . import training_runtime as runtime
from .training_contract import (
    FineTuneTrainingProfile,
    LoraTrainingProfile,
    TrainingProfile,
    canonical_json_bytes,
)
from .training_receipt import require_valid_training_receipt
from .training_runtime_provider import expected_dataset_provider_binding

_REPO_ROOT = Path(__file__).resolve().parents[4]
_RECEIPT_NAME = "training_receipt.json"
_OPTIMIZER_EXECUTION_PROOF_SCHEMA = "invarlock/independent-optimizer-execution-proof-v1"


def _directory_identity(value: os.stat_result) -> tuple[int, int]:
    return (value.st_dev, value.st_ino)


def _subject_directory(path: Path) -> tuple[Path, tuple[int, int]]:
    try:
        before = path.lstat()
    except OSError as exc:
        raise runtime.TrainingRuntimeError(
            f"training subject is not a directory: {path}"
        ) from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
        raise runtime.TrainingRuntimeError(
            f"training subject is not a non-symlink directory: {path}"
        )
    try:
        resolved = path.resolve(strict=True)
        after = path.lstat()
        resolved_stat = resolved.stat()
    except OSError as exc:
        raise runtime.TrainingRuntimeError(
            f"training subject changed while being resolved: {path}"
        ) from exc
    identity = _directory_identity(before)
    if identity != _directory_identity(after) or identity != _directory_identity(
        resolved_stat
    ):
        raise runtime.TrainingRuntimeError(
            f"training subject changed while being resolved: {path}"
        )
    return resolved, identity


def _require_subject_identity(
    path: Path, expected: tuple[int, int], *, phase: str
) -> None:
    try:
        observed = path.lstat()
    except OSError as exc:
        raise runtime.TrainingRuntimeError(f"training subject changed {phase}") from exc
    if (
        stat.S_ISLNK(observed.st_mode)
        or not stat.S_ISDIR(observed.st_mode)
        or _directory_identity(observed) != expected
    ):
        raise runtime.TrainingRuntimeError(f"training subject changed {phase}")


def _require_digest_match(*, label: str, observed: str, expected: Any) -> None:
    if observed != expected:
        raise runtime.TrainingRuntimeError(
            f"{label} digest mismatch: expected {expected!r}, observed {observed!r}"
        )


def _execution_receipt_fields(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Return the execution facts that must match an independent rerun.

    The receipt digest alone cannot prove that an optimizer was invoked: a
    producer can recompute it after editing the receipt.  These values bind a
    bounded rerun to the selected profile, the published checkpoint, and the
    runtime/toolchain facts which determine the training result.
    """

    hashes = receipt["hashes"]
    runtime_facts = receipt["runtime"]
    training = receipt["training"]
    return {
        "profile_id": receipt["profile_id"],
        "profile_sha256": receipt["profile_sha256"],
        "edit_type": receipt["edit_type"],
        "receipt_sha256": receipt["receipt_sha256"],
        "dataset_provider": receipt["dataset_provider"],
        "runtime": runtime_facts,
        "optimizer": receipt["optimizer"],
        "training": training,
        "baseline_state_sha256": hashes["baseline_state_sha256"],
        "post_training_state_sha256": hashes["post_training_state_sha256"],
        "reloaded_subject_state_sha256": hashes["reloaded_subject_state_sha256"],
        "delta_sha256": hashes["delta_sha256"],
        "subject_tree_sha256": hashes["subject_tree_sha256"],
        "changes": receipt["changes"],
        "reload_smoke": receipt["reload_smoke"],
        "lora": receipt.get("lora"),
    }


def _independent_optimizer_execution_proof(
    profile: TrainingProfile,
    receipt: Mapping[str, Any],
    *,
    repo_root: Path,
    local_files_only: bool,
    dataset_provider_policy: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    """Rerun the bounded profile and reject receipt-only optimizer claims.

    The rerun deliberately invokes the real optimizer path with a fresh model,
    fresh optimizer, and fresh output directory.  Its publication skips the
    artifact verifier solely to avoid recursively requesting this same proof;
    the outer verifier validates both artifacts before accepting the result.
    """

    expected_dataset_provider = expected_dataset_provider_binding(
        profile, dataset_provider_policy=dataset_provider_policy
    )
    if receipt.get("dataset_provider") != expected_dataset_provider:
        raise runtime.TrainingRuntimeError(
            "training receipt dataset provider does not match the immutable "
            "provider policy"
        )

    original = _execution_receipt_fields(receipt)
    runtime_facts = original["runtime"]
    if not isinstance(runtime_facts, Mapping):
        raise runtime.TrainingRuntimeError(
            "training receipt runtime facts are unavailable for independent replay"
        )
    sealed_runtime_image_digest = runtime_facts.get("container_image_digest")
    training = original["training"]
    if (
        not isinstance(training, Mapping)
        or training.get("requested_steps") != profile.steps
        or training.get("completed_steps") != profile.steps
        or profile.steps <= 0
    ):
        raise runtime.TrainingRuntimeError(
            "training receipt lacks the required completed optimizer steps"
        )

    with tempfile.TemporaryDirectory(prefix="invarlock-optimizer-proof-") as root:
        try:
            rerun = runtime._run_training_profile(
                profile,
                Path(root) / "subject",
                repo_root=repo_root,
                local_files_only=local_files_only,
                verify_artifact=False,
                dataset_provider_policy=expected_dataset_provider,
                runtime_image_digest=sealed_runtime_image_digest,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            raise runtime.TrainingRuntimeError(
                "independent optimizer execution proof could not rerun the profile: "
                + str(exc)
            ) from exc

        if not isinstance(rerun, runtime.TrainingRunResult):
            raise runtime.TrainingRuntimeError(
                "independent optimizer execution proof is unavailable"
            )
        independently_replayed = _execution_receipt_fields(rerun.receipt)

    independent_training = independently_replayed["training"]
    if (
        not isinstance(independent_training, Mapping)
        or independent_training.get("requested_steps") != profile.steps
        or independent_training.get("completed_steps") != profile.steps
    ):
        raise runtime.TrainingRuntimeError(
            "independent optimizer execution proof contains no completed steps"
        )
    if independently_replayed != original:
        raise runtime.TrainingRuntimeError(
            "independent optimizer execution proof does not match the published "
            "artifact, receipt, profile, and runtime identities"
        )
    return {
        "schema": _OPTIMIZER_EXECUTION_PROOF_SCHEMA,
        "profile_id": profile.profile_id,
        "profile_sha256": profile.profile_sha256,
        "edit_type": profile.edit_type,
        "receipt_sha256": receipt["receipt_sha256"],
        "subject_tree_sha256": original["subject_tree_sha256"],
        "post_training_state_sha256": original["post_training_state_sha256"],
        "runtime": original["runtime"],
        "completed_steps": profile.steps,
    }


def _lora_config_value(value: Any) -> Any:
    return getattr(value, "value", value)


def _normalize_lora_config(value: Any) -> Any:
    value = _lora_config_value(value)
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise runtime.TrainingRuntimeError(
                "serialized LoRA configuration contains a non-string field name"
            )
        return {
            key: _normalize_lora_config(item) for key, item in sorted(value.items())
        }
    if isinstance(value, (list, tuple)):
        return [_normalize_lora_config(item) for item in value]
    if isinstance(value, (set, frozenset)):
        normalized = [_normalize_lora_config(item) for item in value]
        return sorted(normalized, key=canonical_json_bytes)
    raise runtime.TrainingRuntimeError(
        "serialized LoRA configuration contains an unsupported value type: "
        f"{type(value).__name__}"
    )


def _config_mapping(config: Any, *, label: str) -> dict[str, Any]:
    to_dict = getattr(config, "to_dict", None)
    if not callable(to_dict):
        raise runtime.TrainingRuntimeError(f"{label} does not expose to_dict()")
    value = to_dict()
    if not isinstance(value, Mapping):
        raise runtime.TrainingRuntimeError(
            f"{label} to_dict() did not return a mapping"
        )
    normalized = _normalize_lora_config(value)
    if not isinstance(normalized, dict):  # pragma: no cover - Mapping above
        raise runtime.TrainingRuntimeError(f"{label} is not a configuration mapping")
    return normalized


def _expected_serialized_lora_config(
    profile: LoraTrainingProfile, peft_deps: runtime.PeftDependencies
) -> dict[str, Any]:
    expected_config = peft_deps.lora_config_cls(
        r=profile.lora.rank,
        lora_alpha=profile.lora.alpha,
        lora_dropout=profile.lora.dropout,
        target_modules=list(profile.lora.target_modules),
        bias=profile.lora.bias,
        task_type=profile.lora.task_type,
        fan_in_fan_out=profile.lora.fan_in_fan_out,
    )
    expected = _config_mapping(
        expected_config, label="expected pinned LoRA configuration"
    )
    if "inference_mode" in expected:
        expected["inference_mode"] = True
    if "base_model_name_or_path" in expected:
        expected["base_model_name_or_path"] = profile.model_id
    return expected


def _require_serialized_lora_config_file(
    adapter_dir: Path,
    profile: LoraTrainingProfile,
    peft_deps: runtime.PeftDependencies,
    *,
    expected_sha256: object,
) -> dict[str, Any]:
    config_path = adapter_dir / "adapter_config.json"
    try:
        raw = read_regular_file_bytes(
            config_path, label="serialized LoRA configuration"
        )
        serialized_mapping = parse_json_bytes(
            raw,
            label="serialized LoRA configuration",
        )
    except StrictJsonError as exc:
        raise runtime.TrainingRuntimeError(
            f"unable to read serialized LoRA configuration: {exc}"
        ) from exc
    if not isinstance(serialized_mapping, dict):
        raise runtime.TrainingRuntimeError(
            "serialized LoRA configuration must be a JSON object"
        )
    _require_digest_match(
        label="serialized LoRA adapter configuration",
        observed="sha256:" + hashlib.sha256(raw).hexdigest(),
        expected=expected_sha256,
    )
    expected = _expected_serialized_lora_config(profile, peft_deps)
    observed_file = _normalize_lora_config(serialized_mapping)
    if observed_file != expected:
        mismatches = sorted(
            key
            for key in set(observed_file) | set(expected)
            if observed_file.get(key) != expected.get(key)
        )
        raise runtime.TrainingRuntimeError(
            "serialized LoRA configuration does not match the immutable profile: "
            + ", ".join(mismatches)
        )
    return expected


def _require_loaded_lora_config(
    serialized_adapter: Any, expected: dict[str, Any]
) -> None:
    configs = getattr(serialized_adapter, "peft_config", None)
    if not isinstance(configs, Mapping) or len(configs) != 1:
        raise runtime.TrainingRuntimeError(
            "serialized LoRA artifact must contain exactly one PEFT configuration"
        )
    config = next(iter(configs.values()))
    observed_loaded = _config_mapping(
        config, label="loaded serialized LoRA configuration"
    )
    if observed_loaded != expected:
        raise runtime.TrainingRuntimeError(
            "loaded serialized LoRA configuration does not match its pinned form"
        )


def _verify_training_artifact(
    profile: TrainingProfile,
    subject_dir: Path,
    *,
    repo_root: Path = _REPO_ROOT,
    local_files_only: bool = True,
    dataset_provider_policy: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    """Recompute artifact evidence for a completed tiny training profile.

    This independently checks the immutable inputs and published checkpoint. It
    does not attest that a producer executed the optimizer history recorded in
    the receipt; trusted execution or an independent rerun is required for that.
    """

    subject_dir, subject_identity = _subject_directory(subject_dir)
    receipt_path = subject_dir / _RECEIPT_NAME
    receipt_snapshot = runtime._receipt_file_snapshot(
        receipt_path, label="training receipt"
    )
    receipt = require_valid_training_receipt(receipt_snapshot.payload, profile=profile)
    _require_subject_identity(
        subject_dir, subject_identity, phase="while opening its receipt"
    )
    runtime._validate_profile(profile, repo_root=repo_root)

    _require_digest_match(
        label="subject artifact tree",
        observed=runtime.directory_sha256(
            subject_dir, exclude=frozenset({_RECEIPT_NAME})
        ),
        expected=receipt["hashes"]["subject_tree_sha256"],
    )

    deps = runtime._load_runtime_dependencies()
    runtime._configure_determinism(deps.torch, profile)
    device, dtype = runtime._device_and_dtype(deps.torch, profile)
    rows = runtime._load_rows(profile, repo_root=repo_root)
    load_options = {
        "revision": profile.model_revision,
        "local_files_only": local_files_only,
        "trust_remote_code": False,
    }
    tokenizer = deps.auto_tokenizer.from_pretrained(profile.model_id, **load_options)
    if getattr(tokenizer, "pad_token_id", None) is None:
        if getattr(tokenizer, "eos_token", None) is None:
            raise runtime.TrainingRuntimeError(
                "tokenizer has neither a pad token nor an EOS token"
            )
        tokenizer.pad_token = tokenizer.eos_token
    batches, token_count, preprocessing_hash = runtime._prepare_batches(
        tokenizer, rows, profile, torch=deps.torch
    )
    if token_count != receipt["training_data"]["token_count"]:
        raise runtime.TrainingRuntimeError(
            "training token count does not match the receipt"
        )
    _require_digest_match(
        label="training preprocessing",
        observed=preprocessing_hash,
        expected=receipt["training_data"]["preprocessing_sha256"],
    )

    baseline_model, baseline_load_diagnostics = runtime._load_profile_baseline(
        deps, profile, load_options=load_options
    )
    baseline_load_receipt = receipt["model"]["baseline_load"]
    if baseline_load_diagnostics != baseline_load_receipt["diagnostics"]:
        raise runtime.TrainingRuntimeError(
            "upstream baseline loading diagnostics do not match the receipt"
        )
    _require_digest_match(
        label="upstream baseline loading diagnostics",
        observed=load_diagnostics_sha256(baseline_load_diagnostics),
        expected=baseline_load_receipt["diagnostics_sha256"],
    )
    if hasattr(baseline_model, "config"):
        baseline_model.config.pad_token_id = tokenizer.pad_token_id
    baseline_model.to(device=device, dtype=dtype)
    if isinstance(profile, FineTuneTrainingProfile):
        runtime._require_fixture_sized_model(baseline_model)
    baseline_live_state = baseline_model.state_dict()
    baseline_manifest = runtime._state_manifest(baseline_live_state, torch=deps.torch)
    if isinstance(profile, LoraTrainingProfile):
        _require_digest_match(
            label="baseline state manifest",
            observed=runtime._state_manifest_sha256(baseline_manifest),
            expected=receipt["lora"]["base_state_manifest_sha256"],
        )
    baseline_state = (
        runtime._snapshot(baseline_model)
        if isinstance(profile, FineTuneTrainingProfile)
        else None
    )
    baseline_targets: dict[str, Any] = {}
    if isinstance(profile, LoraTrainingProfile):
        lora_receipt = receipt.get("lora")
        target_names = (
            lora_receipt.get("merge_target_names")
            if isinstance(lora_receipt, Mapping)
            else None
        )
        if not isinstance(target_names, list):
            raise runtime.TrainingRuntimeError(
                "LoRA receipt lacks exact merge target names"
            )
        baseline_targets = {
            name: baseline_live_state[name].detach().cpu().clone()
            for name in target_names
            if name in baseline_live_state
        }
        if set(baseline_targets) != set(target_names):
            raise runtime.TrainingRuntimeError(
                "LoRA verifier target names do not match baseline state"
            )
    baseline_hash = runtime.tensor_state_sha256(baseline_live_state, torch=deps.torch)
    _require_digest_match(
        label="baseline model state",
        observed=baseline_hash,
        expected=receipt["hashes"]["baseline_state_sha256"],
    )

    with tempfile.TemporaryDirectory(
        prefix=".training-verifier.", dir=subject_dir.parent
    ) as temporary:
        verifier_temp = Path(temporary)
        runtime._save_model_and_tokenizer(
            baseline_model, tokenizer, verifier_temp / "baseline-artifact"
        )
        tokenizer_only = verifier_temp / "tokenizer-only"
        tokenizer_only.mkdir()
        tokenizer.save_pretrained(tokenizer_only)
        _require_digest_match(
            label="baseline artifact tree",
            observed=runtime.directory_sha256(verifier_temp / "baseline-artifact"),
            expected=receipt["hashes"]["baseline_tree_sha256"],
        )
        _require_digest_match(
            label="tokenizer artifact",
            observed=runtime.directory_sha256(tokenizer_only),
            expected=receipt["model"]["tokenizer_sha256"],
        )

    if isinstance(profile, LoraTrainingProfile):
        del baseline_live_state
        del baseline_model
        runtime.gc.collect()
        if deps.torch.cuda.is_available():
            deps.torch.cuda.empty_cache()
    subject_model, _ = runtime._load_saved_subject(
        deps,
        subject_dir,
        load_options={"local_files_only": True, "trust_remote_code": False},
    )
    subject_model.to(device=device, dtype=dtype)
    subject_live_state = subject_model.state_dict()
    subject_state = (
        runtime._snapshot(subject_model)
        if isinstance(profile, FineTuneTrainingProfile)
        else None
    )
    subject_hash = runtime.tensor_state_sha256(subject_live_state, torch=deps.torch)
    _require_digest_match(
        label="subject model state",
        observed=subject_hash,
        expected=receipt["hashes"]["post_training_state_sha256"],
    )
    _require_digest_match(
        label="reloaded subject model state",
        observed=subject_hash,
        expected=receipt["hashes"]["reloaded_subject_state_sha256"],
    )
    observed_reload_smoke = {
        "passed": True,
        "state_hash_matches": True,
        **runtime._reload_forward_smoke(
            subject_model,
            batches[0],
            deps=deps,
            device=device,
        ),
    }
    if observed_reload_smoke != receipt["reload_smoke"]:
        mismatches = sorted(
            field
            for field in set(observed_reload_smoke) | set(receipt["reload_smoke"])
            if observed_reload_smoke.get(field) != receipt["reload_smoke"].get(field)
        )
        raise runtime.TrainingRuntimeError(
            "reloaded subject inference evidence mismatch: " + ", ".join(mismatches)
        )
    if isinstance(profile, LoraTrainingProfile):
        delta_hash, changed_tensors, max_abs_delta, changed_names = (
            runtime._streaming_lora_delta_evidence(
                baseline_manifest=baseline_manifest,
                baseline_targets=baseline_targets,
                after=subject_live_state,
                torch=deps.torch,
            )
        )
        expected_targets = frozenset(receipt["lora"]["merge_target_names"])
        if changed_names != expected_targets:
            raise runtime.TrainingRuntimeError(
                "observed LoRA merge scope does not exactly match the receipt"
            )
        observed_names_hash = (
            "sha256:"
            + runtime.sha256(
                runtime.canonical_json_bytes(sorted(changed_names))
            ).hexdigest()
        )
        _require_digest_match(
            label="observed LoRA merge target names",
            observed=observed_names_hash,
            expected=receipt["lora"]["observed_merged_changed_names_sha256"],
        )
        if changed_tensors != receipt["lora"]["merged_changed_tensor_count"]:
            raise runtime.TrainingRuntimeError(
                "LoRA changed tensor count does not match the receipt"
            )
    else:
        assert baseline_state is not None and subject_state is not None
        delta_hash, changed_tensors, max_abs_delta, changed_names = (
            runtime._delta_evidence(baseline_state, subject_state, torch=deps.torch)
        )
    _require_digest_match(
        label="training delta",
        observed=delta_hash,
        expected=receipt["hashes"]["delta_sha256"],
    )
    if changed_tensors != receipt["changes"]["changed_tensors"]:
        raise runtime.TrainingRuntimeError(
            "changed tensor count does not match the receipt"
        )
    independently_changed_params = sum(
        int(subject_live_state[name].numel()) for name in changed_names
    )
    independently_total_params = sum(
        int(tensor.numel()) for tensor in subject_live_state.values()
    )
    if independently_changed_params != receipt["changes"]["changed_params"]:
        raise runtime.TrainingRuntimeError(
            "changed parameter count does not match the independently replayed artifact"
        )
    if independently_total_params != receipt["changes"]["total_params"]:
        raise runtime.TrainingRuntimeError(
            "total parameter count does not match the independently replayed artifact"
        )
    if not math.isclose(
        max_abs_delta,
        float(receipt["changes"]["max_abs_delta"]),
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise runtime.TrainingRuntimeError(
            "maximum tensor delta does not match the receipt"
        )

    peft_deps: runtime.PeftDependencies | None = None
    if isinstance(profile, LoraTrainingProfile):
        del subject_live_state
        del subject_model
        runtime.gc.collect()
        if deps.torch.cuda.is_available():
            deps.torch.cuda.empty_cache()
        baseline_model, repeated_baseline_diagnostics = runtime._load_profile_baseline(
            deps, profile, load_options=load_options
        )
        if repeated_baseline_diagnostics != baseline_load_diagnostics:
            raise runtime.TrainingRuntimeError(
                "upstream baseline loading diagnostics changed during verification"
            )
        baseline_model.to(device=device, dtype=dtype)
        peft_deps = runtime._load_peft_dependencies()
        adapter_dir = subject_dir / "adapter"
        _require_digest_match(
            label="serialized LoRA adapter tree",
            observed=runtime.directory_sha256(adapter_dir),
            expected=receipt["lora"]["adapter_tree_sha256"],
        )
        expected_lora_config = _require_serialized_lora_config_file(
            adapter_dir,
            profile,
            peft_deps,
            expected_sha256=receipt["lora"]["serialized_adapter_config_sha256"],
        )
        with runtime._hf_offline_if(local_files_only):
            serialized_config = peft_deps.lora_config_cls.from_pretrained(
                adapter_dir,
                local_files_only=local_files_only,
            )
            try:
                serialized_adapter = load_dense_peft_model(
                    baseline_model,
                    serialized_config,
                    adapter_dir,
                    from_pretrained=peft_deps.peft_model_cls.from_pretrained,
                    is_trainable=False,
                    local_files_only=local_files_only,
                )
            except PeftRuntimeError as exc:
                raise runtime.TrainingRuntimeError(str(exc)) from exc
        _require_loaded_lora_config(serialized_adapter, expected_lora_config)
        serialized_adapter_state = {
            name: tensor.detach().cpu().clone()
            for name, tensor in peft_deps.get_peft_model_state_dict(
                serialized_adapter, save_embedding_layers=False
            ).items()
        }
        serialized_adapter_hash = runtime.tensor_state_sha256(
            serialized_adapter_state, torch=deps.torch
        )
        _require_digest_match(
            label="serialized LoRA adapter state",
            observed=serialized_adapter_hash,
            expected=receipt["lora"]["serialized_adapter_state_sha256"],
        )
        merged_from_serialized = serialized_adapter.merge_and_unload()
        merged_hash = runtime.tensor_state_sha256(
            merged_from_serialized.state_dict(), torch=deps.torch
        )
        _require_digest_match(
            label="serialized LoRA merge",
            observed=merged_hash,
            expected=subject_hash,
        )
        del serialized_adapter_state
        del serialized_adapter
        del baseline_model
        del merged_from_serialized
        runtime.gc.collect()
        if deps.torch.cuda.is_available():
            deps.torch.cuda.empty_cache()

    observed_toolchain = runtime._toolchain(deps, peft_deps)
    runtime._require_expected_toolchain(profile, deps, peft_deps)
    if observed_toolchain != receipt["runtime"]["toolchain"]:
        raise runtime.TrainingRuntimeError(
            "training verifier toolchain does not match the receipt"
        )
    _independent_optimizer_execution_proof(
        profile,
        receipt,
        repo_root=repo_root,
        local_files_only=local_files_only,
        dataset_provider_policy=dataset_provider_policy,
    )
    _require_digest_match(
        label="final subject artifact tree",
        observed=runtime.directory_sha256(
            subject_dir, exclude=frozenset({runtime._RECEIPT_NAME})
        ),
        expected=receipt["hashes"]["subject_tree_sha256"],
    )
    runtime._require_unchanged_receipt(
        receipt_path,
        receipt_snapshot,
        phase="during artifact verification",
    )
    _require_subject_identity(
        subject_dir, subject_identity, phase="during artifact verification"
    )
    return receipt


def verify_training_artifact(
    profile: TrainingProfile,
    subject_dir: Path,
    *,
    repo_root: Path = _REPO_ROOT,
    local_files_only: bool = True,
    dataset_provider_policy: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    """Verify an artifact with local mode enforced for the whole replay."""

    with runtime._hf_offline_if(local_files_only):
        return _verify_training_artifact(
            profile,
            subject_dir,
            repo_root=repo_root,
            local_files_only=local_files_only,
            dataset_provider_policy=dataset_provider_policy,
        )


__all__ = ["verify_training_artifact"]
