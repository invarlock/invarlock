"""Independent runtime reload proof for replay-validated edit artifacts.

This helper is deliberately limited to the edit families with an exact,
verifier-grade transformation replay.  It is not a generic model-serving
smoke test: callers must first provide a successful replay sidecar bound to
the exact local checkpoint tree.  The helper then reloads the local tokenizer
and causal-LM checkpoint twice, observes deterministic finite logits for one
fixed prompt, and records only portable identities and measurements.

The proof is written outside the checkpoint tree.  That keeps the checkpoint
identity stable and prevents an operational runtime sidecar from becoming an
unbound model input.
"""

from __future__ import annotations

import argparse
import gc
import importlib
import json
import os
import stat
import tempfile
import weakref
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from invarlock import transformation_runtime_proof as runtime_proof_contract
from invarlock.core.checkpoint_identity import checkpoint_tree_sha256

RUNTIME_LOAD_DIAGNOSTICS_SCHEMA = runtime_proof_contract.RUNTIME_LOAD_DIAGNOSTICS_SCHEMA
RUNTIME_RELOAD_PROOF_SCHEMA = runtime_proof_contract.RUNTIME_RELOAD_PROOF_SCHEMA
RUNTIME_STORAGE_KEY_AUDIT_SCHEMA = (
    runtime_proof_contract.RUNTIME_STORAGE_KEY_AUDIT_SCHEMA
)
RuntimeReloadProofError = runtime_proof_contract.RuntimeReloadProofError

RUNTIME_RELOAD_PROMPT = "InvarLock verifier-grade transformation runtime proof."
_REPLAY_SCHEMAS = runtime_proof_contract.REPLAY_SCHEMAS
_TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "tokenizer.model",
    "spiece.model",
    "vocab.json",
)
_WEIGHT_FILES = ("model.safetensors", "model.safetensors.index.json")
_PROOF_KEYS = runtime_proof_contract.PROOF_KEYS
_artifact_storage_keys = runtime_proof_contract.artifact_storage_keys
_local_checkpoint_identity = runtime_proof_contract.local_checkpoint_identity
_require_regular_file = runtime_proof_contract.require_regular_file
_sha256_bytes = runtime_proof_contract.sha256_bytes
_storage_key_audit = runtime_proof_contract.storage_key_audit
_strict_json_object = runtime_proof_contract.strict_json_object
_validate_proof_payload = runtime_proof_contract.validate_proof_payload
_LOAD_DIAGNOSTIC_FIELDS = frozenset(
    {"unexpected_keys", "missing_keys", "mismatched_keys", "error_msgs"}
)


@dataclass(frozen=True)
class RuntimeReloadDependencies:
    """The optional runtime dependencies required only when a proof runs."""

    torch: Any
    auto_model: Any
    auto_tokenizer: Any


@dataclass(frozen=True)
class ReplayBinding:
    """The narrow replay fields that the runtime proof must bind exactly."""

    schema: str
    edit_type: str
    artifact_identity: dict[str, str]


@dataclass(frozen=True)
class _RuntimeObservation:
    token_ids_sha256: str
    token_ids_shape: list[int]
    logits_sha256: str
    logits_shape: list[int]
    input_device: str
    load_diagnostics: dict[str, list[object]]
    storage_key_audit: dict[str, object]
    model_reference: weakref.ReferenceType[Any] | None
    tokenizer_reference: weakref.ReferenceType[Any] | None


def _load_runtime_dependencies() -> RuntimeReloadDependencies:
    try:
        torch = importlib.import_module("torch")
        transformers = importlib.import_module("transformers")
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeReloadProofError(
            "runtime reload proof requires torch and transformers"
        ) from exc
    return RuntimeReloadDependencies(
        torch=torch,
        auto_model=transformers.AutoModelForCausalLM,
        auto_tokenizer=transformers.AutoTokenizer,
    )


def read_replay_binding(replay_path: Path) -> ReplayBinding:
    """Require a successful verifier-grade replay with a local artifact identity."""

    replay = _strict_json_object(replay_path, label="replay sidecar")
    schema = replay.get("schema")
    edit_type = replay.get("edit_type")
    if not isinstance(schema, str) or not isinstance(edit_type, str):
        raise RuntimeReloadProofError("replay sidecar lacks a verifier-grade type")
    permitted_edit_types = _REPLAY_SCHEMAS.get(schema)
    if permitted_edit_types is None or edit_type not in permitted_edit_types:
        raise RuntimeReloadProofError(
            "replay sidecar is not verifier-grade transformation evidence"
        )
    if replay.get("ok") is not True:
        raise RuntimeReloadProofError("replay sidecar is not successful")
    issues = replay.get("issues")
    if not isinstance(issues, list) or issues:
        raise RuntimeReloadProofError("replay sidecar has unresolved issues")
    _local_checkpoint_identity(
        replay.get("baseline_identity"), label="replay baseline identity"
    )
    artifact_identity = _local_checkpoint_identity(
        replay.get("artifact_identity"), label="replay artifact identity"
    )
    return ReplayBinding(
        schema=schema,
        edit_type=edit_type,
        artifact_identity=artifact_identity,
    )


def _require_runtime_layout(artifact_dir: Path) -> None:
    try:
        mode = artifact_dir.lstat().st_mode
    except OSError as exc:
        raise RuntimeReloadProofError("artifact directory is unavailable") from exc
    if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
        raise RuntimeReloadProofError("artifact directory must be a regular directory")
    _require_regular_file(artifact_dir / "config.json", label="artifact config")
    if not any((artifact_dir / name).is_file() for name in _TOKENIZER_FILES):
        raise RuntimeReloadProofError("artifact tokenizer files are missing")
    if not any((artifact_dir / name).is_file() for name in _WEIGHT_FILES):
        raise RuntimeReloadProofError("artifact model weights are missing")


def _artifact_identity(artifact_dir: Path) -> dict[str, str]:
    try:
        digest = checkpoint_tree_sha256(artifact_dir)
    except (OSError, ValueError) as exc:
        raise RuntimeReloadProofError("artifact tree identity is unavailable") from exc
    return {"kind": "local_checkpoint_tree", "sha256": digest}


def _tensor_bytes(tensor: Any, *, torch: Any) -> bytes:
    value = tensor.detach().to(device="cpu").contiguous().reshape(-1)
    try:
        return bytes(value.view(torch.uint8).numpy().tobytes())
    except (AttributeError, RuntimeError, TypeError) as exc:
        raise RuntimeReloadProofError("runtime tensor cannot be canonicalized") from exc


def _tensor_sha256(tensor: Any, *, torch: Any) -> str:
    return _sha256_bytes(_tensor_bytes(tensor, torch=torch))


def _shape(value: Any, *, label: str) -> list[int]:
    try:
        shape = list(value.shape)
    except (AttributeError, TypeError) as exc:
        raise RuntimeReloadProofError(f"{label} has no tensor shape") from exc
    if not shape or any(
        isinstance(item, bool) or not isinstance(item, int) or item <= 0
        for item in shape
    ):
        raise RuntimeReloadProofError(f"{label} has an invalid tensor shape")
    return shape


def _resolve_device(torch: Any, requested: str) -> Any:
    if requested not in {"auto", "cpu", "cuda"}:
        raise RuntimeReloadProofError("runtime proof device must be auto, cpu, or cuda")
    cuda_available = bool(
        getattr(getattr(torch, "cuda", None), "is_available", lambda: False)()
    )
    if requested == "cuda" and not cuda_available:
        raise RuntimeReloadProofError(
            "runtime proof requires CUDA but CUDA is unavailable"
        )
    resolved = (
        "cuda"
        if requested == "cuda" or (requested == "auto" and cuda_available)
        else "cpu"
    )
    try:
        return torch.device(resolved)
    except (AttributeError, TypeError, RuntimeError) as exc:
        raise RuntimeReloadProofError(
            "runtime proof could not configure its device"
        ) from exc


def _safe_weak_reference(value: Any) -> weakref.ReferenceType[Any] | None:
    try:
        return weakref.ref(value)
    except TypeError:
        return None


def _release_runtime_memory(torch: Any, *, device: Any) -> None:
    gc.collect()
    if str(device) != "cuda":
        return
    empty_cache = getattr(getattr(torch, "cuda", None), "empty_cache", None)
    if callable(empty_cache):
        empty_cache()


def _model_load_options(
    dependencies: RuntimeReloadDependencies,
    *,
    dispatch_managed: bool,
) -> dict[str, object]:
    """Return the pinned local-only load policy for one runtime proof reload."""

    options: dict[str, object] = {
        "local_files_only": True,
        "trust_remote_code": False,
        # A finite forward is not sufficient evidence: transformers can load
        # a model while silently dropping injected or incompatible weights.
        "output_loading_info": True,
    }
    if not dispatch_managed:
        return options
    dtype = getattr(dependencies.torch, "bfloat16", None)
    if dtype is None:
        raise RuntimeReloadProofError("runtime proof requires torch bfloat16 support")
    # Transformers 5.x accepts ``dtype``.  ``torch_dtype`` is the older alias
    # and would make this evidence path depend on a compatibility fallback.
    options.update(
        {
            "dtype": dtype,
            "device_map": "auto",
            "low_cpu_mem_usage": True,
        }
    )
    return options


def _clean_load_diagnostics(value: object) -> dict[str, list[object]]:
    """Require one complete, empty Transformers loading diagnostic record."""

    if not isinstance(value, Mapping) or set(value) != _LOAD_DIAGNOSTIC_FIELDS:
        raise RuntimeReloadProofError(
            "artifact model loader did not return loading diagnostics"
        )
    diagnostics: dict[str, list[object]] = {}
    for field in sorted(_LOAD_DIAGNOSTIC_FIELDS):
        entries = value.get(field)
        if not isinstance(entries, (list, tuple, set, frozenset)):
            raise RuntimeReloadProofError(
                f"artifact model loading diagnostics lack {field}"
            )
        if entries:
            raise RuntimeReloadProofError(
                f"artifact model loading diagnostics report {field}"
            )
        diagnostics[field] = []
    return diagnostics


def _load_model_with_diagnostics(
    artifact_dir: Path,
    *,
    dependencies: RuntimeReloadDependencies,
    dispatch_managed: bool,
) -> tuple[Any, dict[str, list[object]]]:
    """Load exactly once and reject every non-clean checkpoint-key outcome."""

    loaded = dependencies.auto_model.from_pretrained(
        artifact_dir,
        **_model_load_options(dependencies, dispatch_managed=dispatch_managed),
    )
    if not isinstance(loaded, tuple) or len(loaded) != 2:
        raise RuntimeReloadProofError(
            "artifact model loader did not return model and loading diagnostics"
        )
    model, loading_info = loaded
    if model is None:
        raise RuntimeReloadProofError("artifact model could not be loaded")
    return model, _clean_load_diagnostics(loading_info)


def _concrete_device(value: object, *, torch: Any) -> Any | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"", "disk", "meta"}:
        return None
    normalized: object = f"cuda:{value}" if isinstance(value, int) else value
    try:
        device = torch.device(normalized)
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return None
    if str(device) in {"disk", "meta"}:
        return None
    return device


def _first_parameter_device(model: Any, *, torch: Any) -> Any | None:
    parameters = getattr(model, "parameters", None)
    if not callable(parameters):
        return None
    try:
        first_parameter = next(iter(parameters()), None)
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return None
    return _concrete_device(
        getattr(first_parameter, "device", None),
        torch=torch,
    )


def _model_input_device(
    model: Any,
    *,
    torch: Any,
    fallback: Any,
    dispatch_managed: bool,
) -> Any:
    """Find the device that receives input IDs without relocating a model."""

    get_embeddings = getattr(model, "get_input_embeddings", None)
    if callable(get_embeddings):
        try:
            embeddings = get_embeddings()
        except (AttributeError, RuntimeError, TypeError, ValueError):
            embeddings = None
        if embeddings is not None:
            for value in (
                getattr(getattr(embeddings, "weight", None), "device", None),
                getattr(embeddings, "device", None),
            ):
                device = _concrete_device(value, torch=torch)
                if device is not None:
                    return device

    device_map = getattr(model, "hf_device_map", None)
    if isinstance(device_map, Mapping):
        prioritized_keys = [
            "",
            *sorted(
                key
                for key in device_map
                if isinstance(key, str)
                and key.endswith(("embed_tokens", "wte", "word_embeddings", "embed_in"))
            ),
        ]
        seen_keys: set[object] = set()
        for key in [*prioritized_keys, *device_map]:
            if key in seen_keys:
                continue
            seen_keys.add(key)
            device = _concrete_device(device_map.get(key), torch=torch)
            if device is not None:
                return device

    for value in (getattr(model, "device", None),):
        device = _concrete_device(value, torch=torch)
        if device is not None:
            return device
    parameter_device = _first_parameter_device(model, torch=torch)
    if parameter_device is not None:
        return parameter_device
    if dispatch_managed:
        raise RuntimeReloadProofError(
            "dispatch-managed model does not expose an input device"
        )
    return fallback


def _one_runtime_observation(
    artifact_dir: Path,
    *,
    dependencies: RuntimeReloadDependencies,
    device: Any,
    previous_model: weakref.ReferenceType[Any] | None,
    previous_tokenizer: weakref.ReferenceType[Any] | None,
) -> _RuntimeObservation:
    tokenizer: Any = None
    model: Any = None
    load_diagnostics: dict[str, list[object]] | None = None
    storage_key_audit: dict[str, object] | None = None
    try:
        load_options = {"local_files_only": True, "trust_remote_code": False}
        tokenizer = dependencies.auto_tokenizer.from_pretrained(
            artifact_dir, **load_options
        )
        if tokenizer is None:
            raise RuntimeReloadProofError("artifact tokenizer could not be loaded")
        if previous_tokenizer is not None and previous_tokenizer() is tokenizer:
            raise RuntimeReloadProofError("tokenizer loader reused a prior instance")
        encoded = tokenizer(RUNTIME_RELOAD_PROMPT, return_tensors="pt")
        if not isinstance(encoded, Mapping):
            raise RuntimeReloadProofError(
                "artifact tokenizer did not return tensor inputs"
            )
        token_ids = encoded.get("input_ids")
        if (
            token_ids is None
            or not dependencies.torch.is_tensor(token_ids)
            or int(token_ids.numel()) <= 0
        ):
            raise RuntimeReloadProofError("artifact tokenizer did not return input_ids")
        token_ids_shape = _shape(token_ids, label="token ids")
        token_ids_sha256 = _tensor_sha256(token_ids, torch=dependencies.torch)
        dispatch_managed = str(device) == "cuda"
        model, load_diagnostics = _load_model_with_diagnostics(
            artifact_dir,
            dependencies=dependencies,
            dispatch_managed=dispatch_managed,
        )
        storage_key_audit = _storage_key_audit(artifact_dir, model=model)
        if previous_model is not None and previous_model() is model:
            raise RuntimeReloadProofError("model loader reused a prior instance")
        model.eval()
        input_device = _model_input_device(
            model,
            torch=dependencies.torch,
            fallback=device,
            dispatch_managed=dispatch_managed,
        )
        inputs: dict[str, Any] = {}
        for name, value in encoded.items():
            if not isinstance(name, str) or not dependencies.torch.is_tensor(value):
                raise RuntimeReloadProofError(
                    "artifact tokenizer returned a non-tensor model input"
                )
            inputs[name] = value.to(input_device)
        with dependencies.torch.inference_mode():
            output = model(**inputs)
        logits = (
            output.get("logits")
            if isinstance(output, Mapping)
            else getattr(output, "logits", None)
        )
        if (
            logits is None
            or not dependencies.torch.is_tensor(logits)
            or int(logits.numel()) <= 0
        ):
            raise RuntimeReloadProofError("artifact model did not return logits")
        observed_logits = logits.detach().float().cpu().contiguous()
        if not bool(dependencies.torch.isfinite(observed_logits).all().item()):
            raise RuntimeReloadProofError("artifact model returned non-finite logits")
        assert load_diagnostics is not None
        assert storage_key_audit is not None
        return _RuntimeObservation(
            token_ids_sha256=token_ids_sha256,
            token_ids_shape=token_ids_shape,
            logits_sha256=_tensor_sha256(observed_logits, torch=dependencies.torch),
            logits_shape=_shape(observed_logits, label="logits"),
            input_device=str(input_device),
            load_diagnostics=load_diagnostics,
            storage_key_audit=storage_key_audit,
            model_reference=_safe_weak_reference(model),
            tokenizer_reference=_safe_weak_reference(tokenizer),
        )
    except RuntimeReloadProofError:
        raise
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise RuntimeReloadProofError("artifact runtime reload failed") from exc
    finally:
        del model
        del tokenizer
        _release_runtime_memory(dependencies.torch, device=device)


def run_runtime_reload_proof(
    artifact_dir: Path,
    *,
    replay_path: Path,
    expected_identity: Mapping[str, object] | None = None,
    device: str = "auto",
) -> dict[str, object]:
    """Independently prove two fresh reloads are finite and deterministic.

    ``replay_path`` must contain a successful replay sidecar from one of the
    exact generated-transformation or pruning validators.  The proof never
    writes inside ``artifact_dir`` and is valid only while the observed tree
    identity remains identical before and after both reloads.
    """

    artifact_dir = artifact_dir.expanduser().absolute()
    binding = read_replay_binding(replay_path)
    if expected_identity is not None:
        expected = _local_checkpoint_identity(
            expected_identity, label="expected artifact identity"
        )
        if expected != binding.artifact_identity:
            raise RuntimeReloadProofError(
                "expected artifact identity does not match replay evidence"
            )
    _require_runtime_layout(artifact_dir)
    before = _artifact_identity(artifact_dir)
    if before != binding.artifact_identity:
        raise RuntimeReloadProofError(
            "artifact identity does not match replay evidence"
        )

    dependencies = _load_runtime_dependencies()
    resolved_device = _resolve_device(dependencies.torch, device)
    first = _one_runtime_observation(
        artifact_dir,
        dependencies=dependencies,
        device=resolved_device,
        previous_model=None,
        previous_tokenizer=None,
    )
    second = _one_runtime_observation(
        artifact_dir,
        dependencies=dependencies,
        device=resolved_device,
        previous_model=first.model_reference,
        previous_tokenizer=first.tokenizer_reference,
    )
    observations = (
        first.token_ids_sha256,
        first.token_ids_shape,
        first.logits_sha256,
        first.logits_shape,
        first.input_device,
    )
    repeated_observations = (
        second.token_ids_sha256,
        second.token_ids_shape,
        second.logits_sha256,
        second.logits_shape,
        second.input_device,
    )
    if observations != repeated_observations:
        raise RuntimeReloadProofError("artifact runtime reload was not deterministic")
    if first.load_diagnostics != second.load_diagnostics:
        raise RuntimeReloadProofError(
            "artifact runtime reload loading diagnostics were not deterministic"
        )
    if first.storage_key_audit != second.storage_key_audit:
        raise RuntimeReloadProofError(
            "artifact runtime reload storage-key audits were not deterministic"
        )
    after = _artifact_identity(artifact_dir)
    if after != before:
        raise RuntimeReloadProofError("artifact tree changed during runtime reload")

    proof: dict[str, object] = {
        "schema": RUNTIME_RELOAD_PROOF_SCHEMA,
        "ok": True,
        "replay_schema": binding.schema,
        "edit_type": binding.edit_type,
        "artifact_identity": after,
        "replay_artifact_identity": binding.artifact_identity,
        "prompt_sha256": _sha256_bytes(RUNTIME_RELOAD_PROMPT.encode("utf-8")),
        "device": str(resolved_device),
        "input_device": first.input_device,
        "reload_runs": 2,
        "token_ids_sha256": first.token_ids_sha256,
        "token_ids_shape": first.token_ids_shape,
        "logits_sha256": first.logits_sha256,
        "logits_shape": first.logits_shape,
        "all_logits_finite": True,
        "repeat_deterministic": True,
        "load_diagnostics": {
            "schema": RUNTIME_LOAD_DIAGNOSTICS_SCHEMA,
            "reloads": [first.load_diagnostics, second.load_diagnostics],
        },
        "storage_key_audit": {
            "schema": RUNTIME_STORAGE_KEY_AUDIT_SCHEMA,
            "reloads": [first.storage_key_audit, second.storage_key_audit],
        },
    }
    _validate_proof_payload(proof)
    return proof


def write_runtime_reload_proof(
    output_path: Path,
    proof: Mapping[str, object],
    *,
    artifact_dir: Path,
    replay_path: Path | None = None,
) -> None:
    """Atomically write a validated proof outside the artifact tree.

    The limited proof schema deliberately contains no path-bearing fields.  A
    caller cannot use this writer to place a free-form operational report in a
    public evidence directory.
    """

    _validate_proof_payload(proof)
    artifact_dir = artifact_dir.expanduser().absolute()
    try:
        artifact_root = artifact_dir.resolve(strict=True)
        candidate = output_path.expanduser().absolute().resolve(strict=False)
    except OSError as exc:
        raise RuntimeReloadProofError("runtime proof output path is invalid") from exc
    try:
        candidate.relative_to(artifact_root)
    except ValueError:
        pass
    else:
        raise RuntimeReloadProofError(
            "runtime proof output must be outside the artifact tree"
        )
    if replay_path is not None:
        try:
            replay_resolved = replay_path.expanduser().absolute().resolve(strict=True)
        except OSError as exc:
            raise RuntimeReloadProofError("replay sidecar is unavailable") from exc
        if candidate == replay_resolved:
            raise RuntimeReloadProofError(
                "runtime proof output must not replace replay evidence"
            )
    if output_path.is_symlink():
        raise RuntimeReloadProofError("runtime proof output must not be a symlink")
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        parent = output_path.parent.resolve(strict=True)
        resolved_output = (parent / output_path.name).resolve(strict=False)
    except OSError as exc:
        raise RuntimeReloadProofError(
            "runtime proof output path is unavailable"
        ) from exc
    try:
        resolved_output.relative_to(artifact_root)
    except ValueError:
        pass
    else:
        raise RuntimeReloadProofError(
            "runtime proof output must be outside the artifact tree"
        )
    if output_path.exists() and not output_path.is_file():
        raise RuntimeReloadProofError("runtime proof output must be a regular file")

    encoded = (
        json.dumps(dict(proof), allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, output_path)
        temporary_path = None
    except OSError as exc:
        raise RuntimeReloadProofError("could not write runtime proof") from exc
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _parse_identity_json(raw: str) -> dict[str, object]:
    def no_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    def reject_nonstandard_constant(value: str) -> object:
        raise ValueError(f"non-standard JSON constant {value!r}")

    try:
        payload = json.loads(
            raw,
            object_pairs_hook=no_duplicate_keys,
            parse_constant=reject_nonstandard_constant,
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeReloadProofError("expected identity must be strict JSON") from exc
    if not isinstance(payload, dict):
        raise RuntimeReloadProofError("expected identity must be a JSON object")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Prove deterministic fresh runtime reloads for a replay-validated artifact."
    )
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--replay", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--expected-identity-json",
        help="optional typed local checkpoint identity that must match replay evidence",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="auto uses CUDA when available; cuda requires it",
    )
    args = parser.parse_args(argv)
    try:
        expected_identity = (
            _parse_identity_json(args.expected_identity_json)
            if args.expected_identity_json is not None
            else None
        )
        proof = run_runtime_reload_proof(
            args.artifact,
            replay_path=args.replay,
            expected_identity=expected_identity,
            device=args.device,
        )
        write_runtime_reload_proof(
            args.out,
            proof,
            artifact_dir=args.artifact,
            replay_path=args.replay,
        )
    except RuntimeReloadProofError as exc:
        parser.error(str(exc))
    print(json.dumps(proof, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
