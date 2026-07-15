"""Deterministic real-training runtime for evidence-pack model edits.

The runtime publishes a subject only after real optimizer steps, state-change
checks, a save/reload smoke test, and fail-closed receipt validation succeed.
LoRA evidence uses bounded streaming state identities and selected target
snapshots. Full-parameter training remains intentionally fixture-sized because
its optimizer and exact delta proof retain full model state.
"""

from __future__ import annotations

import ctypes
import gc
import importlib
import os
import platform
import random
import shutil
import stat
import sys
import tempfile
import weakref
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from invarlock.evidence_pack_json import StrictJsonError, read_json_object_snapshot
from invarlock.peft_runtime import adapter_module_count as _package_adapter_module_count
from invarlock.training_batch_execution import TrainingBatchError
from invarlock.training_batch_execution import (
    prepare_batches as _package_prepare_batches,
)
from invarlock.training_batch_execution import (
    reload_forward_smoke as _package_reload_forward_smoke,
)
from invarlock.training_batch_execution import (
    toolchain as _package_toolchain,
)
from invarlock.training_batch_execution import (
    train as _package_train,
)
from invarlock.training_model_load import TrainingModelLoadError
from invarlock.training_model_load import (
    configure_causal_lm_loss as _package_configure_causal_lm_loss,
)
from invarlock.training_model_load import (
    load_diagnostics_sha256 as _package_load_diagnostics_sha256,
)
from invarlock.training_model_load import (
    load_model_with_diagnostics as _package_load_model_with_diagnostics,
)
from invarlock.training_protocol import read_jsonl_snapshot
from invarlock.training_receipt_builder import build_common_receipt
from invarlock.training_state_evidence import (
    state_manifest_sha256 as _state_manifest_sha256,
)

from .training_contract import (
    FineTuneTrainingProfile,
    LoraTrainingProfile,
    TrainingProfile,
    canonical_json_bytes,
)
from .training_receipt import (
    TRAINING_RECEIPT_SCHEMA,
    canonical_receipt_digest,
    with_receipt_digest,
)
from .training_receipt import (
    require_valid_training_receipt as validate_training_receipt,
)
from .training_runtime_errors import TrainingRuntimeError
from .training_runtime_evidence import (
    _delta_evidence,
    _peft_base_state,
    _peft_merge_target_names,
    _require_fixture_sized_model,
    _require_state_manifest,
    _snapshot,
    _state_manifest,
    _streaming_lora_delta_evidence,
    _tensor_bytes,
    directory_sha256,
    tensor_state_sha256,
)
from .training_runtime_lora import execute_lora_training
from .training_runtime_provider import dataset_provider_binding
from .training_runtime_publication import (
    discard_failed_publication,
    fsync_directory,
    publish_directory_no_replace,
)
from .training_runtime_validation import (
    profile_mapping,
    validate_profile,
)

_REPO_ROOT = Path(__file__).resolve().parents[4]
_RECEIPT_NAME = "training_receipt.json"


def _adapter_module_count(state: Mapping[str, Any]) -> int:
    return _package_adapter_module_count(state)


@dataclass(frozen=True)
class RuntimeDependencies:
    torch: Any
    auto_model: Any
    auto_tokenizer: Any
    optimizer_cls: Any
    transformers_version: str


@dataclass(frozen=True)
class PeftDependencies:
    lora_config_cls: Any
    get_peft_model: Any
    get_peft_model_state_dict: Any
    peft_model_cls: Any
    version: str


@dataclass(frozen=True)
class TrainingRunResult:
    subject_dir: Path
    receipt_path: Path
    receipt: dict[str, Any]


@dataclass(frozen=True)
class _ReceiptFileSnapshot:
    raw: bytes
    identity: tuple[int, int, int, int, int]
    canonical_digest: str
    payload: dict[str, Any]


def _path_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _directory_identity(path: Path, *, label: str) -> tuple[int, int]:
    try:
        value = path.lstat()
    except OSError as exc:
        raise TrainingRuntimeError(f"{label} is unavailable: {path}") from exc
    if stat.S_ISLNK(value.st_mode) or not stat.S_ISDIR(value.st_mode):
        raise TrainingRuntimeError(f"{label} must be a non-symlink directory: {path}")
    return (value.st_dev, value.st_ino)


def _receipt_file_snapshot(path: Path, *, label: str) -> _ReceiptFileSnapshot:
    try:
        before = path.lstat()
    except OSError as exc:
        raise TrainingRuntimeError(
            f"unable to read {label}: unavailable: {path}"
        ) from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise TrainingRuntimeError(f"{label} must be a regular file: {path}")
    try:
        raw, payload = read_json_object_snapshot(path, label=label)
    except StrictJsonError as exc:
        raise TrainingRuntimeError(f"unable to read {label}: {exc}") from exc
    try:
        after = path.lstat()
    except OSError as exc:
        raise TrainingRuntimeError(f"{label} changed while being read: {path}") from exc
    identity = _path_identity(before)
    if identity != _path_identity(after):
        raise TrainingRuntimeError(f"{label} changed while being read: {path}")
    return _ReceiptFileSnapshot(
        raw=raw,
        identity=identity,
        canonical_digest=canonical_receipt_digest(payload),
        payload=payload,
    )


def _require_unchanged_receipt(
    path: Path, expected: _ReceiptFileSnapshot, *, phase: str
) -> _ReceiptFileSnapshot:
    observed = _receipt_file_snapshot(path, label="persisted training receipt")
    if (
        observed.identity != expected.identity
        or observed.raw != expected.raw
        or observed.canonical_digest != expected.canonical_digest
    ):
        raise TrainingRuntimeError(f"training receipt changed {phase}")
    return observed


def _load_runtime_dependencies() -> RuntimeDependencies:
    try:
        torch = importlib.import_module("torch")
        transformers = importlib.import_module("transformers")
    except ImportError as exc:  # pragma: no cover - depends on optional extras
        raise TrainingRuntimeError(
            "real training requires the torch and transformers dependencies"
        ) from exc
    return RuntimeDependencies(
        torch=torch,
        auto_model=transformers.AutoModelForCausalLM,
        auto_tokenizer=transformers.AutoTokenizer,
        optimizer_cls=torch.optim.AdamW,
        transformers_version=str(transformers.__version__),
    )


def _load_peft_dependencies() -> PeftDependencies:
    try:
        peft = importlib.import_module("peft")
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise TrainingRuntimeError(
            "lora_merge training requires the optional `peft` package; "
            "install a PEFT version compatible with the pinned torch/transformers stack"
        ) from exc
    get_state = getattr(peft, "get_peft_model_state_dict", None)
    if not callable(get_state):
        raise TrainingRuntimeError(
            "the pinned PEFT runtime lacks get_peft_model_state_dict"
        )
    return PeftDependencies(
        lora_config_cls=peft.LoraConfig,
        get_peft_model=peft.get_peft_model,
        get_peft_model_state_dict=get_state,
        peft_model_cls=peft.PeftModel,
        version=str(peft.__version__),
    )


def _profile_mapping(profile: TrainingProfile) -> dict[str, Any]:
    return profile_mapping(profile)


def _validate_profile(profile: TrainingProfile, *, repo_root: Path) -> None:
    validate_profile(
        profile.profile_id,
        _profile_mapping(profile),
        edit_type=profile.edit_type,
        repo_root=repo_root,
        error_type=TrainingRuntimeError,
    )


def _configure_determinism(torch: Any, profile: TrainingProfile) -> None:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(profile.seed)
    torch.manual_seed(profile.seed)
    if hasattr(torch, "cuda"):
        torch.cuda.manual_seed_all(profile.seed)
    torch.use_deterministic_algorithms(profile.deterministic_algorithms)
    cudnn = getattr(getattr(torch, "backends", None), "cudnn", None)
    if cudnn is not None:
        cudnn.deterministic = profile.deterministic_algorithms
        cudnn.benchmark = False


def _device_and_dtype(torch: Any, profile: TrainingProfile) -> tuple[Any, Any]:
    if profile.device == "cuda" and not torch.cuda.is_available():
        raise TrainingRuntimeError("profile requires CUDA but CUDA is unavailable")
    mps = getattr(getattr(torch, "backends", None), "mps", None)
    if profile.device == "mps" and (mps is None or not mps.is_available()):
        raise TrainingRuntimeError("profile requires MPS but MPS is unavailable")
    dtype = getattr(torch, profile.dtype, None)
    if dtype is None:
        raise TrainingRuntimeError(f"unsupported torch dtype: {profile.dtype}")
    return torch.device(profile.device), dtype


def _load_rows(profile: TrainingProfile, *, repo_root: Path) -> list[str]:
    path = profile.training_data.resolve(repo_root)
    try:
        raw, records = read_jsonl_snapshot(path, label="training data")
    except StrictJsonError as exc:
        raise TrainingRuntimeError(f"training data is not valid JSONL: {exc}") from exc
    if "sha256:" + sha256(raw).hexdigest() != profile.training_data.sha256:
        raise TrainingRuntimeError("vendored training data digest changed before use")
    rows: list[str] = []
    for line_number, row in enumerate(records, start=1):
        text = (
            row.get(profile.training_data.text_field) if isinstance(row, dict) else None
        )
        if not isinstance(text, str) or not text.strip():
            raise TrainingRuntimeError(
                f"training data line {line_number} lacks the configured text field"
            )
        rows.append(text)
    if len(rows) != profile.training_data.rows:
        raise TrainingRuntimeError(
            "vendored training data row count changed before use"
        )
    return rows


def _save_model_and_tokenizer(model: Any, tokenizer: Any, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(path, safe_serialization=True)
    tokenizer.save_pretrained(path)


@contextmanager
def _hf_offline_if(enabled: bool) -> Iterator[None]:
    names = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    previous = {name: os.environ.get(name) for name in names}
    if enabled:
        for name in names:
            os.environ[name] = "1"
    try:
        yield
    finally:
        if enabled:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value


def _publish_directory_no_replace(staging: Path, output_dir: Path) -> None:
    publish_directory_no_replace(
        staging,
        output_dir,
        ctypes_module=ctypes,
        platform_system=platform.system,
        error_type=TrainingRuntimeError,
    )


def _fsync_directory(path: Path) -> None:
    fsync_directory(path, error_type=TrainingRuntimeError)


def _discard_failed_publication(
    output_dir: Path, *, expected_identity: tuple[int, int]
) -> None:
    discard_failed_publication(
        output_dir,
        expected_identity=expected_identity,
        publish=_publish_directory_no_replace,
        fsync=_fsync_directory,
        directory_identity=_directory_identity,
        error_type=TrainingRuntimeError,
    )


def _prepare_batches(
    tokenizer: Any,
    rows: Sequence[str],
    profile: TrainingProfile,
    *,
    torch: Any,
) -> tuple[list[dict[str, Any]], int, str]:
    try:
        return _package_prepare_batches(
            tokenizer,
            rows,
            profile,
            torch=torch,
            state_sha256=tensor_state_sha256,
        )
    except TrainingBatchError as exc:
        raise TrainingRuntimeError(str(exc)) from exc


def _train(
    model: Any,
    parameters: Sequence[Any],
    batches: Sequence[Mapping[str, Any]],
    profile: TrainingProfile,
    *,
    deps: RuntimeDependencies,
    device: Any,
) -> list[float]:
    try:
        return _package_train(
            model,
            parameters,
            batches,
            profile,
            optimizer_cls=deps.optimizer_cls,
            device=device,
        )
    except TrainingBatchError as exc:
        raise TrainingRuntimeError(str(exc)) from exc


def _toolchain(
    deps: RuntimeDependencies, peft: PeftDependencies | None
) -> dict[str, str]:
    return _package_toolchain(
        deps.torch, deps.transformers_version, peft.version if peft else None
    )


def _require_expected_toolchain(
    profile: TrainingProfile,
    deps: RuntimeDependencies,
    peft: PeftDependencies | None,
) -> None:
    observed = _toolchain(deps, peft)
    expected = {
        "python": profile.toolchain.python,
        "torch": profile.toolchain.torch,
        "transformers": profile.toolchain.transformers,
    }
    if profile.toolchain.peft is not None:
        expected["peft"] = profile.toolchain.peft
    mismatches = [
        f"{package}={observed.get(package)!r} (expected {version!r})"
        for package, version in expected.items()
        if str(observed.get(package, "")) != version
    ]
    if mismatches:
        raise TrainingRuntimeError(
            "training toolchain does not match the immutable profile: "
            + ", ".join(mismatches)
        )


def _reload_forward_smoke(
    model: Any,
    batch: Mapping[str, Any],
    *,
    deps: RuntimeDependencies,
    device: Any,
) -> dict[str, Any]:
    """Run repeatable finite-logit inference against the reloaded checkpoint."""

    try:
        return _package_reload_forward_smoke(
            model,
            batch,
            torch=deps.torch,
            device=device,
            state_sha256=tensor_state_sha256,
            tensor_bytes=_tensor_bytes,
        )
    except TrainingBatchError as exc:
        raise TrainingRuntimeError(str(exc)) from exc


def _load_model_with_diagnostics(
    deps: RuntimeDependencies,
    source: object,
    *,
    load_options: Mapping[str, object],
    expected_unexpected_keys: Sequence[str],
    label: str,
) -> tuple[Any, dict[str, object]]:
    try:
        return _package_load_model_with_diagnostics(
            deps.auto_model,
            source,
            load_options=load_options,
            expected_unexpected_keys=expected_unexpected_keys,
            label=label,
        )
    except TrainingModelLoadError as exc:
        raise TrainingRuntimeError(str(exc)) from exc


def _load_profile_baseline(
    deps: RuntimeDependencies,
    profile: TrainingProfile,
    *,
    load_options: Mapping[str, object],
) -> tuple[Any, dict[str, object]]:
    model, diagnostics = _load_model_with_diagnostics(
        deps,
        profile.model_id,
        load_options=load_options,
        expected_unexpected_keys=profile.model_load.expected_unexpected_keys,
        label="upstream baseline model",
    )
    try:
        _package_configure_causal_lm_loss(
            model,
            loss_function=profile.model_load.loss_function,
        )
    except TrainingModelLoadError as exc:
        raise TrainingRuntimeError(str(exc)) from exc
    return model, diagnostics


def _load_saved_subject(
    deps: RuntimeDependencies,
    source: object,
    *,
    load_options: Mapping[str, object],
) -> tuple[Any, dict[str, object]]:
    return _load_model_with_diagnostics(
        deps,
        source,
        load_options=load_options,
        expected_unexpected_keys=(),
        label="saved training subject",
    )


def _run_training_profile(
    profile: TrainingProfile,
    output_dir: Path,
    *,
    repo_root: Path = _REPO_ROOT,
    local_files_only: bool = True,
    verify_artifact: bool = True,
    dataset_provider_policy: Mapping[str, object] | None = None,
    runtime_image_digest: str | None = None,
) -> TrainingRunResult:
    """Execute, artifact-check, and atomically publish a real training profile."""

    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise TrainingRuntimeError(f"refusing to replace existing output: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    _validate_profile(profile, repo_root=repo_root)
    deps = _load_runtime_dependencies()
    if isinstance(profile, FineTuneTrainingProfile):
        _require_expected_toolchain(profile, deps, None)
    _configure_determinism(deps.torch, profile)
    device, dtype = _device_and_dtype(deps.torch, profile)
    rows = _load_rows(profile, repo_root=repo_root)
    dataset_provider = dataset_provider_binding(
        profile, dataset_provider_policy=dataset_provider_policy
    )

    load_options = {
        "revision": profile.model_revision,
        "local_files_only": local_files_only,
        "trust_remote_code": False,
    }
    tokenizer = deps.auto_tokenizer.from_pretrained(profile.model_id, **load_options)
    if getattr(tokenizer, "pad_token_id", None) is None:
        if getattr(tokenizer, "eos_token", None) is None:
            raise TrainingRuntimeError(
                "tokenizer has neither a pad token nor an EOS token"
            )
        tokenizer.pad_token = tokenizer.eos_token
    model, baseline_load_diagnostics = _load_profile_baseline(
        deps, profile, load_options=load_options
    )
    if hasattr(model, "config"):
        model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device=device, dtype=dtype)
    baseline_state = model.state_dict()
    baseline_hash = tensor_state_sha256(baseline_state, torch=deps.torch)
    baseline_manifest = _state_manifest(baseline_state, torch=deps.torch)
    baseline_manifest_hash = _state_manifest_sha256(baseline_manifest)
    baseline: dict[str, Any] | None = None
    baseline_targets: dict[str, Any] = {}
    if isinstance(profile, FineTuneTrainingProfile):
        _require_fixture_sized_model(model)
        baseline = _snapshot(model)
    batches, token_count, preprocessing_hash = _prepare_batches(
        tokenizer, rows, profile, torch=deps.torch
    )

    staging: Path | None = None
    baseline_temp: Path | None = None
    published = False
    peft_deps: PeftDependencies | None = None
    lora_receipt: dict[str, Any] | None = None
    try:
        staging = Path(
            tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
        )
        baseline_temp = Path(
            tempfile.mkdtemp(prefix=".training-baseline.", dir=output_dir.parent)
        )
        _save_model_and_tokenizer(model, tokenizer, baseline_temp / "artifact")
        tokenizer_only = baseline_temp / "tokenizer-only"
        tokenizer_only.mkdir()
        tokenizer.save_pretrained(tokenizer_only)
        tokenizer_hash = directory_sha256(tokenizer_only)
        baseline_tree_hash = directory_sha256(baseline_temp / "artifact")

        if isinstance(profile, FineTuneTrainingProfile):
            named_parameters = list(model.named_parameters())
            if not named_parameters or any(
                not parameter.requires_grad for _, parameter in named_parameters
            ):
                raise TrainingRuntimeError(
                    "fine_tune requires every model parameter to be trainable"
                )
            trainable_names = {name for name, _ in named_parameters}
            losses = _train(
                model,
                [parameter for _, parameter in named_parameters],
                batches,
                profile,
                deps=deps,
                device=device,
            )
            subject_model = model
        elif isinstance(profile, LoraTrainingProfile):
            (
                subject_model,
                losses,
                peft_deps,
                lora_receipt,
                baseline_targets,
            ) = execute_lora_training(
                sys.modules[__name__],
                profile=profile,
                model=model,
                tokenizer=tokenizer,
                deps=deps,
                baseline_hash=baseline_hash,
                baseline_manifest=baseline_manifest,
                baseline_manifest_hash=baseline_manifest_hash,
                baseline_state=baseline_state,
                baseline_load_diagnostics=baseline_load_diagnostics,
                batches=batches,
                staging=staging,
                load_options=load_options,
                local_files_only=local_files_only,
                device=device,
                dtype=dtype,
            )
            del model
            del baseline_state
            trainable_names = set()
        else:  # pragma: no cover - closed typed union
            raise TrainingRuntimeError(
                f"unsupported training profile: {type(profile).__name__}"
            )

        post_state = subject_model.state_dict()
        post_hash = tensor_state_sha256(post_state, torch=deps.torch)
        if isinstance(profile, LoraTrainingProfile):
            delta_hash, changed_tensors, max_abs_delta, changed_names = (
                _streaming_lora_delta_evidence(
                    baseline_manifest=baseline_manifest,
                    baseline_targets=baseline_targets,
                    after=post_state,
                    torch=deps.torch,
                )
            )
        else:
            assert baseline is not None
            delta_hash, changed_tensors, max_abs_delta, changed_names = _delta_evidence(
                baseline, post_state, torch=deps.torch
            )
        if isinstance(profile, LoraTrainingProfile):
            assert lora_receipt is not None
            expected_targets = frozenset(baseline_targets)
            if changed_names != expected_targets:
                raise TrainingRuntimeError("LoRA merge changed an invalid tensor scope")
            lora_receipt["state_evidence_policy"] = "streaming-per-tensor-digests-v1"
            lora_receipt["expected_merge_target_names_sha256"] = (
                "sha256:"
                + sha256(canonical_json_bytes(sorted(expected_targets))).hexdigest()
            )
            lora_receipt["merge_target_names"] = sorted(expected_targets)
            lora_receipt["observed_merged_changed_names_sha256"] = (
                "sha256:"
                + sha256(canonical_json_bytes(sorted(changed_names))).hexdigest()
            )
            lora_receipt["merged_changed_tensor_count"] = changed_tensors
            lora_receipt["merge_scope_exact"] = True
        if changed_tensors < 1 or max_abs_delta <= 0.0:
            raise TrainingRuntimeError("optimizer steps did not change model tensors")
        changed_params = sum(int(post_state[name].numel()) for name in changed_names)
        total_params = sum(int(tensor.numel()) for tensor in post_state.values())
        if changed_params < 1 or total_params < changed_params:
            raise TrainingRuntimeError("optimizer parameter coverage is invalid")
        if isinstance(
            profile, FineTuneTrainingProfile
        ) and not trainable_names.issubset(changed_names):
            unchanged = sorted(trainable_names - changed_names)[:3]
            raise TrainingRuntimeError(
                f"full fine-tune left trainable tensors unchanged: {unchanged}"
            )
        if post_hash == baseline_hash:
            raise TrainingRuntimeError("trained subject state equals the baseline")
        if lora_receipt is not None:
            lora_receipt["merged_state_sha256"] = post_hash

        subject_model.save_pretrained(staging, safe_serialization=True)
        tokenizer.save_pretrained(staging)
        subject_tree_hash = directory_sha256(staging)
        if subject_tree_hash == baseline_tree_hash:
            raise TrainingRuntimeError(
                "saved subject artifact tree equals the baseline tree"
            )

        if isinstance(profile, LoraTrainingProfile):
            del post_state
            del subject_model
            gc.collect()
            if deps.torch.cuda.is_available():
                deps.torch.cuda.empty_cache()

        reloaded, _ = _load_saved_subject(
            deps,
            staging,
            load_options={"local_files_only": True, "trust_remote_code": False},
        )
        reloaded.to(device=device, dtype=dtype)
        reloaded_hash = tensor_state_sha256(reloaded.state_dict(), torch=deps.torch)
        if reloaded_hash != post_hash:
            raise TrainingRuntimeError(
                "saved subject failed the reload state-hash check"
            )
        reload_forward = _reload_forward_smoke(
            reloaded,
            batches[0],
            deps=deps,
            device=device,
        )

        receipt = build_common_receipt(
            profile,
            schema=TRAINING_RECEIPT_SCHEMA,
            toolchain=_toolchain(deps, None),
            tokenizer_hash=tokenizer_hash,
            token_count=token_count,
            preprocessing_hash=preprocessing_hash,
            losses=losses,
            baseline_hash=baseline_hash,
            baseline_tree_hash=baseline_tree_hash,
            post_hash=post_hash,
            delta_hash=delta_hash,
            subject_tree_hash=subject_tree_hash,
            reloaded_hash=reloaded_hash,
            changed_tensors=changed_tensors,
            changed_params=changed_params,
            total_params=total_params,
            max_abs_delta=max_abs_delta,
            reload_forward=reload_forward,
            loss_function=profile.model_load.loss_function,
            baseline_load_diagnostics=baseline_load_diagnostics,
            baseline_load_diagnostics_sha256=_package_load_diagnostics_sha256(
                baseline_load_diagnostics
            ),
            dataset_provider=dataset_provider,
            container_image_digest=runtime_image_digest,
        )
        if peft_deps is not None and lora_receipt is not None:
            receipt["runtime"]["toolchain"] = _toolchain(deps, peft_deps)
            receipt["lora"] = lora_receipt
        receipt = validate_training_receipt(
            with_receipt_digest(receipt), profile=profile
        )
        receipt_path = staging / _RECEIPT_NAME
        receipt_path.write_bytes(canonical_json_bytes(receipt) + b"\n")
        persisted_snapshot = _receipt_file_snapshot(
            receipt_path, label="persisted training receipt"
        )
        validate_training_receipt(persisted_snapshot.payload, profile=profile)
        if (
            directory_sha256(staging, exclude=frozenset({_RECEIPT_NAME}))
            != subject_tree_hash
        ):
            raise TrainingRuntimeError(
                "subject artifact changed while writing its receipt"
            )
        reloaded_reference = weakref.ref(reloaded)
        del reloaded
        if isinstance(profile, FineTuneTrainingProfile):
            del post_state
            del baseline_state
            del subject_model
            del model
            del named_parameters
        gc.collect()
        if deps.torch.cuda.is_available():
            deps.torch.cuda.empty_cache()
        if reloaded_reference() is not None:
            raise TrainingRuntimeError(
                "reloaded subject remained live before independent verification"
            )

        if verify_artifact:
            verify_training_artifact(
                profile,
                staging,
                repo_root=repo_root,
                local_files_only=local_files_only,
                dataset_provider_policy=dataset_provider,
            )
            _require_unchanged_receipt(
                receipt_path,
                persisted_snapshot,
                phase="during final verification",
            )
            if (
                directory_sha256(staging, exclude=frozenset({_RECEIPT_NAME}))
                != subject_tree_hash
            ):
                raise TrainingRuntimeError(
                    "subject artifact changed after final verification"
                )
            _require_unchanged_receipt(
                receipt_path,
                persisted_snapshot,
                phase="immediately before publication",
            )
        staging_identity = _directory_identity(
            staging, label="verified training subject staging directory"
        )
        _publish_directory_no_replace(staging, output_dir)
        published = True
        try:
            if (
                _directory_identity(output_dir, label="published training subject")
                != staging_identity
            ):
                raise TrainingRuntimeError(
                    "published training subject identity does not match verified staging"
                )
            if (
                directory_sha256(output_dir, exclude=frozenset({_RECEIPT_NAME}))
                != subject_tree_hash
            ):
                raise TrainingRuntimeError(
                    "published training subject does not match the verified artifact tree"
                )
            _require_unchanged_receipt(
                output_dir / _RECEIPT_NAME,
                persisted_snapshot,
                phase="during publication",
            )
            if (
                _directory_identity(output_dir, label="published training subject")
                != staging_identity
            ):
                raise TrainingRuntimeError(
                    "published training subject changed during rebind"
                )
        except Exception:
            _discard_failed_publication(output_dir, expected_identity=staging_identity)
            raise
        _fsync_directory(output_dir.parent)
        return TrainingRunResult(
            subject_dir=output_dir,
            receipt_path=output_dir / _RECEIPT_NAME,
            receipt=receipt,
        )
    finally:
        if baseline_temp is not None:
            shutil.rmtree(baseline_temp, ignore_errors=True)
        if not published and staging is not None:
            shutil.rmtree(staging, ignore_errors=True)


def run_training_profile(
    profile: TrainingProfile,
    output_dir: Path,
    *,
    repo_root: Path = _REPO_ROOT,
    local_files_only: bool = True,
    dataset_provider_policy: Mapping[str, object] | None = None,
) -> TrainingRunResult:
    """Execute and publish a training profile with local mode enforced globally."""

    with _hf_offline_if(local_files_only):
        return _run_training_profile(
            profile,
            output_dir,
            repo_root=repo_root,
            local_files_only=local_files_only,
            dataset_provider_policy=dataset_provider_policy,
            runtime_image_digest=(
                os.environ.get("INVARLOCK_EXPECTED_RUNTIME_IMAGE_DIGEST") or None
            ),
        )


def verify_training_artifact(
    profile: TrainingProfile,
    subject_dir: Path,
    *,
    repo_root: Path = _REPO_ROOT,
    local_files_only: bool = True,
    dataset_provider_policy: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    """Delegate to the independently owned artifact verifier."""

    from .training_artifact_verifier import verify_training_artifact as verify

    return verify(
        profile,
        subject_dir,
        repo_root=repo_root,
        local_files_only=local_files_only,
        dataset_provider_policy=dataset_provider_policy,
    )


__all__ = [
    "TrainingRunResult",
    "TrainingRuntimeError",
    "_peft_base_state",
    "_peft_merge_target_names",
    "_require_state_manifest",
    "directory_sha256",
    "run_training_profile",
    "tensor_state_sha256",
    "verify_training_artifact",
]
