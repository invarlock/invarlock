"""Built-in Hugging Face provider for authenticated local checkpoints.

The provider loads a resolved local checkpoint, records its model and tokenizer
identity, and emits per-record scoring facts for independent verification.
Optional backend imports remain lazy so the core package stays Torch-free.
"""

from __future__ import annotations

import hashlib
import importlib
import math
import re
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, cast

from invarlock import __version__ as INVARLOCK_VERSION
from invarlock.core.checkpoint_identity import (
    CheckpointIdentityError,
    checkpoint_tree_sha256,
)
from invarlock.core.runtime_provider import (
    INVARLOCK_RUNTIME_PROVIDER_ABI,
    EvaluationBatch,
    HFSnapshotArtifactIdentity,
    ModelRuntimeSpec,
    RuntimeArtifactResources,
    RuntimeBackendIdentity,
    RuntimeDeviceFacts,
    RuntimeExecutionContext,
    RuntimeExecutionSettings,
    RuntimeProviderCapabilities,
    RuntimeProviderPluginIdentity,
    RuntimeProviderReceipt,
    RuntimeScoringRecord,
    ScoringObservation,
    artifact_identity_sha256,
    exact_match_output_text,
)
from invarlock.core.runtime_provider.behavioral_observation import (
    runtime_scoring_records_sha256,
)
from invarlock.core.runtime_provider.types import (
    JSONScalar,
    RuntimeScorer,
    evaluation_input_parts_sha256,
)
from invarlock.runtime_providers._hf_safetensors_identity import (
    HFSafetensorsIdentityError,
    safetensors_storage_keys,
)
from invarlock.runtime_providers._hf_transformers_identity import (
    _installed_backend_identity,
    _json_safe_tokenizer_value,
    _observed_device_facts,
    _sha256,
    hf_tokenizer_contract_sha256,
)
from invarlock.runtime_security_helpers import (
    network_allowed,
    remote_code_allowed,
    resolve_runtime_image,
    resolve_runtime_image_digest,
    strict_container_boundary_present,
    third_party_plugins_allowed,
)

_ALLOWED_SETTINGS = frozenset(
    {
        "batch_size",
        "checkpoint_tree_sha256",
        "context_length",
        "immutable_revision",
        "max_output_tokens",
        "offline",
        "seed",
        "timeout_seconds",
        "tokenizer_metadata_sha256",
    }
)
_POSITIVE_INTEGER_SETTINGS = frozenset(
    {"batch_size", "context_length", "max_output_tokens", "timeout_seconds"}
)
_REQUIRED_RECEIPT_SETTINGS = frozenset(
    {
        "batch_size",
        "context_length",
        "max_output_tokens",
        "offline",
        "seed",
        "timeout_seconds",
    }
)
_SHA256 = re.compile(r"^[a-f0-9]{64}$")
_WINDOWS_ABSOLUTE_PATH = re.compile(r"^[A-Za-z]:[\\/]")


def _optional_text(settings: Mapping[str, JSONScalar], name: str) -> str | None:
    value = settings.get(name)
    if value is None:
        return None
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty trimmed string")
    return value


def _optional_sha256(settings: Mapping[str, JSONScalar], name: str) -> str | None:
    value = _optional_text(settings, name)
    if value is None:
        return None
    canonical = value.removeprefix("sha256:")
    if _SHA256.fullmatch(canonical) is None:
        raise ValueError(f"{name} must be a sha256 digest")
    return canonical


def _is_local_path_like(model_id: str) -> bool:
    try:
        exists_locally = Path(model_id).exists()
    except OSError:
        # Path APIs reject some malformed/oversized host inputs. Treat those as
        # path-like so they can never flow into a public artifact identity.
        exists_locally = True
    return bool(
        Path(model_id).is_absolute()
        or exists_locally
        or model_id.startswith(("./", "../", "~/"))
        or "\\" in model_id
        or _WINDOWS_ABSOLUTE_PATH.match(model_id)
    )


def _validate_setting_values(spec: ModelRuntimeSpec) -> None:
    for name in _POSITIVE_INTEGER_SETTINGS:
        value = spec.settings.get(name)
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
        ):
            raise ValueError(f"{name} must be a positive integer")

    seed = spec.settings.get("seed")
    if seed is not None and (
        isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
    ):
        raise ValueError("seed must be a non-negative integer")

    offline = spec.settings.get("offline")
    if offline is not None and not isinstance(offline, bool):
        raise ValueError("offline must be boolean")

    _optional_text(spec.settings, "immutable_revision")
    _optional_sha256(spec.settings, "checkpoint_tree_sha256")
    _optional_sha256(spec.settings, "tokenizer_metadata_sha256")


def _required_integer(
    settings: Mapping[str, JSONScalar], name: str, *, positive: bool
) -> int:
    value = settings.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        label = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must be a {label} integer")
    if (positive and value <= 0) or (not positive and value < 0):
        label = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must be a {label} integer")
    return value


def _strict_execution_settings(spec: ModelRuntimeSpec) -> RuntimeExecutionSettings:
    missing = _REQUIRED_RECEIPT_SETTINGS - set(spec.settings)
    if missing:
        rendered = ", ".join(sorted(missing))
        raise ValueError(
            "strict hf_transformers receipt requires setting(s): " + rendered
        )
    if spec.settings.get("offline") is not True:
        raise ValueError("strict hf_transformers receipt requires offline=true")
    return RuntimeExecutionSettings(
        seed=_required_integer(spec.settings, "seed", positive=False),
        context_length=_required_integer(
            spec.settings, "context_length", positive=True
        ),
        batch_size=_required_integer(spec.settings, "batch_size", positive=True),
        max_output_tokens=_required_integer(
            spec.settings, "max_output_tokens", positive=True
        ),
        timeout_seconds=_required_integer(
            spec.settings, "timeout_seconds", positive=True
        ),
        allow_network=False,
    )


def _image_ref_matches_digest(image_ref: str, image_digest: str) -> bool:
    if image_ref == image_digest:
        return True
    repository, separator, digest = image_ref.rpartition("@")
    return bool(repository and separator and digest == image_digest)


def _require_strict_runtime_boundary(context: RuntimeExecutionContext) -> None:
    if context.allow_network:
        raise ValueError("strict hf_transformers execution must disable network")
    if network_allowed():
        raise ValueError("strict hf_transformers execution must be offline")
    if remote_code_allowed():
        raise ValueError("strict hf_transformers execution must disable remote code")
    if third_party_plugins_allowed():
        raise ValueError(
            "strict hf_transformers execution must disable third-party plugins"
        )
    if not strict_container_boundary_present():
        raise ValueError(
            "strict hf_transformers execution requires the authenticated "
            "container boundary"
        )
    image_digest = context.container_image_digest
    if image_digest is None:
        raise ValueError(
            "strict hf_transformers execution requires a pinned outer container image"
        )
    if resolve_runtime_image_digest() != image_digest:
        raise ValueError(
            "strict hf_transformers runtime image digest does not match the "
            "container context"
        )
    if not _image_ref_matches_digest(resolve_runtime_image(), image_digest):
        raise ValueError(
            "strict hf_transformers runtime image reference must embed the exact digest"
        )


def _require_safetensors_match(checkpoint: Path, *, model: object) -> None:
    """Require every authenticated safetensors value in the live model state."""

    try:
        from safetensors import safe_open

    except ImportError as exc:  # pragma: no cover - optional dependency boundary
        raise RuntimeError(
            "strict HF artifact binding requires the safetensors runtime"
        ) from exc

    state_dict = getattr(model, "state_dict", None)
    if not callable(state_dict):
        raise RuntimeError("strict HF native model does not expose state_dict")
    try:
        live_state = state_dict()
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        raise RuntimeError("strict HF native model state is unavailable") from exc
    if not isinstance(live_state, Mapping):
        raise RuntimeError("strict HF native model state is unavailable")

    try:
        authenticated_keys = safetensors_storage_keys(checkpoint)
    except HFSafetensorsIdentityError as exc:
        raise RuntimeError(
            "strict HF checkpoint must use a canonical safetensors layout"
        ) from exc
    if not authenticated_keys.issubset(live_state):
        raise ValueError(
            "strict HF loaded model is missing authenticated checkpoint tensors"
        )

    observed_keys: set[str] = set()
    try:
        shards = sorted(checkpoint.glob("*.safetensors"), key=lambda path: path.name)
        for shard in shards:
            with safe_open(str(shard), framework="pt", device="cpu") as handle:
                for key in handle.keys():
                    observed_keys.add(key)
                    stored = handle.get_tensor(key)
                    live = live_state[key]
                    detach = getattr(live, "detach", None)
                    if not callable(detach):
                        raise RuntimeError(
                            "strict HF native model state contains a non-tensor value"
                        )
                    live_cpu = detach().to(device="cpu").contiguous()
                    if (
                        stored.dtype != live_cpu.dtype
                        or tuple(stored.shape) != tuple(live_cpu.shape)
                        or not bool(stored.equal(live_cpu))
                    ):
                        raise ValueError(
                            "strict HF loaded model tensors do not match the "
                            "authenticated checkpoint"
                        )
    except (OSError, RuntimeError, ValueError):
        raise
    if observed_keys != authenticated_keys:
        raise RuntimeError(
            "strict HF checkpoint tensor inventory changed during binding"
        )


def _require_model_config_match(checkpoint: Path, *, model: object) -> None:
    """Compare the live model config with a fresh offline checkpoint load."""

    transformers = importlib.import_module("transformers")
    auto_config = getattr(transformers, "AutoConfig", None)
    from_pretrained = getattr(auto_config, "from_pretrained", None)
    live_config = getattr(model, "config", None)
    live_to_dict = getattr(live_config, "to_dict", None)
    if not callable(from_pretrained) or not callable(live_to_dict):
        raise RuntimeError("strict HF model configuration is unavailable")
    try:
        authenticated_config = from_pretrained(
            str(checkpoint),
            local_files_only=True,
            trust_remote_code=False,
        )
        authenticated_to_dict = getattr(authenticated_config, "to_dict", None)
        if not callable(authenticated_to_dict):
            raise TypeError
        live_payload = live_to_dict()
        authenticated_payload = authenticated_to_dict()
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "strict HF checkpoint configuration could not be authenticated"
        ) from exc
    if not isinstance(live_payload, Mapping) or not isinstance(
        authenticated_payload, Mapping
    ):
        raise RuntimeError("strict HF model configuration is unavailable")
    if (
        live_config.__class__.__module__,
        live_config.__class__.__qualname__,
    ) != (
        authenticated_config.__class__.__module__,
        authenticated_config.__class__.__qualname__,
    ):
        raise ValueError(
            "strict HF live model config class does not match the checkpoint"
        )
    normalized_live = cast(dict[str, object], _json_safe_tokenizer_value(live_payload))
    normalized_authenticated = cast(
        dict[str, object], _json_safe_tokenizer_value(authenticated_payload)
    )
    # This locator is not behavioral and differs when an already-created model is
    # saved before being loaded through ``from_pretrained``.
    normalized_live.pop("_name_or_path", None)
    normalized_authenticated.pop("_name_or_path", None)
    if normalized_live != normalized_authenticated:
        raise ValueError(
            "strict HF live model config does not match the authenticated checkpoint"
        )


def _require_model_eval_mode(model: object) -> None:
    modules = getattr(model, "modules", None)
    if not callable(modules):
        raise RuntimeError("strict HF native model does not expose module state")
    try:
        bound_modules = tuple(modules())
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "strict HF native model module state is unavailable"
        ) from exc
    if not bound_modules or any(
        getattr(module, "training", None) is not False for module in bound_modules
    ):
        raise RuntimeError(
            "strict HF causal scoring requires model.eval() for every submodule"
        )


def require_loaded_hf_checkpoint_binding(
    *,
    spec: ModelRuntimeSpec,
    identity: HFSnapshotArtifactIdentity,
    model: object,
    tokenizer: object,
    checkpoint: Path | None = None,
) -> None:
    """Bind the live model/tokenizer to one locally authenticated checkpoint."""

    checkpoint = (
        Path(spec.model_id).expanduser() if checkpoint is None else Path(checkpoint)
    )
    if not checkpoint.is_dir():
        raise RuntimeError(
            "strict HF runtime-behavior evidence requires a materialized local "
            "checkpoint; a remote revision alone cannot bind the loaded model"
        )
    declared_tree = identity.checkpoint_tree_sha256
    if declared_tree is None:
        raise RuntimeError(
            "strict HF runtime-behavior evidence requires checkpoint_tree_sha256"
        )
    try:
        before = checkpoint_tree_sha256(checkpoint).removeprefix("sha256:")
    except (CheckpointIdentityError, OSError) as exc:
        raise RuntimeError(
            "strict HF checkpoint tree could not be authenticated"
        ) from exc
    if before != declared_tree:
        raise ValueError(
            "strict HF checkpoint tree does not match the authenticated identity"
        )
    tokenizer_sha256 = hf_tokenizer_contract_sha256(tokenizer)
    if tokenizer_sha256 != identity.tokenizer_metadata_sha256:
        raise ValueError(
            "strict HF live tokenizer does not match tokenizer_metadata_sha256"
        )
    _require_model_eval_mode(model)
    _require_model_config_match(checkpoint, model=model)
    _require_safetensors_match(checkpoint, model=model)
    try:
        after = checkpoint_tree_sha256(checkpoint).removeprefix("sha256:")
    except (CheckpointIdentityError, OSError) as exc:
        raise RuntimeError(
            "strict HF checkpoint tree could not be reauthenticated"
        ) from exc
    if after != before:
        raise RuntimeError("strict HF checkpoint tree changed during model binding")


@dataclass(frozen=True)
class HFTransformersCausalScorer:
    """Provider-owned deterministic scorer bound to one model and tokenizer."""

    model: object = field(repr=False, compare=False)
    tokenizer: object = field(repr=False, compare=False)
    artifact_identity_sha256: str
    checkpoint_path: Path | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if _SHA256.fullmatch(self.artifact_identity_sha256) is None:
            raise ValueError("artifact_identity_sha256 must be a sha256 digest")
        if self.checkpoint_path is not None:
            checkpoint = Path(self.checkpoint_path)
            if not checkpoint.is_absolute():
                raise ValueError("checkpoint_path must be absolute")
            object.__setattr__(self, "checkpoint_path", checkpoint)

    def require_binding(self, *, model: object, artifact_identity_sha256: str) -> None:
        if self.model is not model:
            raise ValueError("strict HF scorer is not bound to the exact native model")
        if self.artifact_identity_sha256 != artifact_identity_sha256:
            raise ValueError(
                "strict HF scorer artifact identity does not match the model spec"
            )

    def __call__(
        self, batch: EvaluationBatch, settings: RuntimeExecutionSettings
    ) -> ScoringObservation:
        if settings.batch_size != 1:
            raise ValueError("strict HF causal scoring currently requires batch_size=1")
        torch = importlib.import_module("torch")
        tokenizer_call = self.tokenizer if callable(self.tokenizer) else None
        decode = getattr(self.tokenizer, "decode", None)
        if (
            not callable(self.model)
            or not callable(tokenizer_call)
            or not callable(decode)
        ):
            raise RuntimeError(
                "strict HF causal scorer requires model and tokenizer APIs"
            )
        _require_model_eval_mode(self.model)

        parameters = tuple(cast(Any, self.model).parameters())
        tensors = parameters or tuple(cast(Any, self.model).buffers())
        if not tensors:
            raise RuntimeError("strict HF native model has no execution tensors")
        device = tensors[0].device
        records = []
        deterministic_enabled = bool(torch.are_deterministic_algorithms_enabled())
        deterministic_warn_only = bool(
            torch.is_deterministic_algorithms_warn_only_enabled()
        )
        torch.use_deterministic_algorithms(True, warn_only=False)
        try:
            with torch.random.fork_rng(), torch.inference_mode():
                torch.manual_seed(settings.seed)
                if getattr(device, "type", None) == "cuda":
                    torch.cuda.manual_seed_all(settings.seed)
                for record in batch.records:
                    deadline = time.monotonic() + settings.timeout_seconds
                    if batch.task != "text_causal":
                        raise ValueError(
                            "built-in HF execution supports only text_causal"
                        )
                    if record.input_parts:
                        if len(record.input_parts) != 1 or (
                            record.input_parts[0].kind != "text"
                            or record.input_parts[0].role != "prompt"
                        ):
                            raise ValueError(
                                "built-in HF causal execution requires one prompt "
                                "text input part"
                            )
                        expected_input_sha256 = evaluation_input_parts_sha256(
                            record.input_parts
                        )
                    else:
                        expected_input_sha256 = hashlib.sha256(
                            record.input_text.encode("utf-8")
                        ).hexdigest()
                    if expected_input_sha256 != record.input_sha256:
                        raise ValueError(
                            "runtime evaluation input does not match input_sha256"
                        )
                    encoded = tokenizer_call(
                        record.input_text,
                        add_special_tokens=True,
                        max_length=settings.context_length,
                        return_tensors="pt",
                        truncation=True,
                    )
                    if not isinstance(encoded, Mapping) or "input_ids" not in encoded:
                        raise RuntimeError(
                            "strict HF tokenizer did not return input_ids"
                        )
                    model_inputs = {
                        key: value.to(device)
                        for key, value in encoded.items()
                        if hasattr(value, "to")
                    }
                    input_ids = model_inputs.get("input_ids")
                    if (
                        input_ids is None
                        or input_ids.ndim != 2
                        or input_ids.shape[0] != 1
                    ):
                        raise RuntimeError(
                            "strict HF tokenizer returned invalid input_ids"
                        )
                    if input_ids.shape[1] < 1:
                        raise RuntimeError(
                            "strict HF tokenizer returned empty input_ids"
                        )
                    nll_facts: tuple[float, int, int] | None = None
                    if batch.metric == "normalized_nll_per_utf8_byte":
                        if record.expected_output is None:
                            raise ValueError(
                                "strict HF normalized NLL requires expected_output"
                            )
                        target = tokenizer_call(
                            record.expected_output,
                            add_special_tokens=False,
                            return_tensors="pt",
                            truncation=False,
                        )
                        if not isinstance(target, Mapping) or "input_ids" not in target:
                            raise RuntimeError(
                                "strict HF tokenizer did not return target input_ids"
                            )
                        target_ids = target["input_ids"]
                        if hasattr(target_ids, "to"):
                            target_ids = target_ids.to(device)
                        if (
                            not hasattr(target_ids, "ndim")
                            or target_ids.ndim != 2
                            or target_ids.shape[0] != 1
                            or target_ids.shape[1] < 1
                        ):
                            raise RuntimeError(
                                "strict HF tokenizer returned invalid target input_ids"
                            )
                        target_count = int(target_ids.shape[1])
                        if target_count > settings.max_output_tokens:
                            raise ValueError(
                                "strict HF normalized NLL target exceeds "
                                "max_output_tokens"
                            )
                        prompt_token_ids = [
                            int(input_ids[0, index].item())
                            for index in range(int(input_ids.shape[1]))
                        ]
                        target_token_ids = [
                            int(target_ids[0, index].item())
                            for index in range(target_count)
                        ]
                        decoded_prompt = decode(
                            prompt_token_ids,
                            clean_up_tokenization_spaces=False,
                            skip_special_tokens=False,
                        )
                        decoded_continuation = decode(
                            prompt_token_ids + target_token_ids,
                            clean_up_tokenization_spaces=False,
                            skip_special_tokens=False,
                        )
                        if (
                            not isinstance(decoded_prompt, str)
                            or not isinstance(decoded_continuation, str)
                            or decoded_continuation
                            != decoded_prompt + record.expected_output
                        ):
                            raise ValueError(
                                "strict HF normalized NLL target is not an exact "
                                "tokenizer continuation of the prompt"
                            )
                        history = input_ids
                        target_logprobs: list[float] = []
                        for target_index in range(target_count):
                            if time.monotonic() >= deadline:
                                raise TimeoutError(
                                    "strict HF normalized NLL scoring timed out"
                                )
                            window = history[:, -settings.context_length :]
                            attention_mask = torch.ones_like(window)
                            result = self.model(
                                input_ids=window,
                                attention_mask=attention_mask,
                                return_dict=True,
                                use_cache=False,
                            )
                            logits = getattr(result, "logits", None)
                            if (
                                logits is None
                                or logits.ndim != 3
                                or logits.shape[0] != 1
                                or logits.shape[1] != window.shape[1]
                                or not bool(torch.isfinite(logits[:, -1, :]).all())
                            ):
                                raise RuntimeError(
                                    "strict HF model returned invalid causal logits"
                                )
                            target_token = target_ids[
                                :, target_index : target_index + 1
                            ]
                            if (
                                int(target_token.item()) < 0
                                or int(target_token.item()) >= logits.shape[-1]
                            ):
                                raise RuntimeError(
                                    "strict HF target token is outside the model vocabulary"
                                )
                            token_logprob = torch.log_softmax(
                                logits[:, -1, :], dim=-1
                            ).gather(1, target_token)
                            value = float(token_logprob.item())
                            if not math.isfinite(value) or value > 0:
                                raise RuntimeError(
                                    "strict HF model returned an invalid target logprob"
                                )
                            target_logprobs.append(value)
                            history = torch.cat((history, target_token), dim=1)
                        nll_facts = (
                            math.fsum(target_logprobs),
                            target_count,
                            len(record.expected_output.encode("utf-8")),
                        )
                    output_text: str | None = None
                    if batch.metric == "exact_match":
                        generated = input_ids
                        new_tokens: list[int] = []
                        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
                        for _ in range(settings.max_output_tokens):
                            if time.monotonic() >= deadline:
                                raise TimeoutError("strict HF causal scoring timed out")
                            window = generated[:, -settings.context_length :]
                            attention_mask = torch.ones_like(window)
                            result = self.model(
                                input_ids=window,
                                attention_mask=attention_mask,
                                return_dict=True,
                                use_cache=False,
                            )
                            logits = getattr(result, "logits", None)
                            if (
                                logits is None
                                or logits.ndim != 3
                                or logits.shape[0] != 1
                                or logits.shape[1] != window.shape[1]
                                or not bool(torch.isfinite(logits[:, -1, :]).all())
                            ):
                                raise RuntimeError(
                                    "strict HF model returned invalid causal logits"
                                )
                            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                            token_id = int(next_token.item())
                            new_tokens.append(token_id)
                            generated = torch.cat((generated, next_token), dim=1)
                            if (
                                isinstance(eos_token_id, int)
                                and not isinstance(eos_token_id, bool)
                                and token_id == eos_token_id
                            ):
                                break
                        assert callable(decode)
                        output_text = decode(
                            new_tokens,
                            clean_up_tokenization_spaces=False,
                            skip_special_tokens=True,
                        )
                        try:
                            output_text = exact_match_output_text(output_text)
                        except ValueError as exc:
                            raise RuntimeError(
                                "strict HF tokenizer returned invalid user-visible text"
                            ) from exc
                    records.append(
                        RuntimeScoringRecord(
                            record_id=record.record_id,
                            input_sha256=record.input_sha256,
                            status="ok",
                            output_text=output_text,
                            output_sha256=(
                                _sha256(output_text.encode("utf-8"))
                                if output_text is not None
                                else None
                            ),
                            logprob_sum=(nll_facts[0] if nll_facts else None),
                            token_count=(nll_facts[1] if nll_facts else None),
                            utf8_byte_count=(nll_facts[2] if nll_facts else None),
                        )
                    )
        finally:
            torch.use_deterministic_algorithms(
                deterministic_enabled,
                warn_only=deterministic_warn_only,
            )
        aggregate_source_sha256 = runtime_scoring_records_sha256(
            [cast(dict[str, object], asdict(record)) for record in records]
        )
        return ScoringObservation(
            provider_name=HFTransformersProvider.name,
            artifact_identity_sha256=self.artifact_identity_sha256,
            schedule_sha256=batch.schedule_sha256,
            records=tuple(records),
            aggregate_source_sha256=aggregate_source_sha256,
        )


def _require_strict_execution_binding(
    *,
    spec: ModelRuntimeSpec,
    identity: HFSnapshotArtifactIdentity,
    context: RuntimeExecutionContext,
) -> RuntimeScorer:
    scorer = context.scorer
    if not isinstance(scorer, HFTransformersCausalScorer):
        raise RuntimeError(
            "strict HF runtime-behavior evidence requires the provider-owned "
            "HFTransformersCausalScorer; arbitrary scorer callbacks are not "
            "authenticated"
        )
    assert context.provider_state is not None
    expected_identity_sha256 = artifact_identity_sha256(identity)
    scorer.require_binding(
        model=context.provider_state,
        artifact_identity_sha256=expected_identity_sha256,
    )
    require_loaded_hf_checkpoint_binding(
        spec=spec,
        identity=identity,
        model=context.provider_state,
        tokenizer=scorer.tokenizer,
        checkpoint=scorer.checkpoint_path,
    )
    return scorer


@dataclass(frozen=True)
class _HFReceiptProvenance:
    plugin: RuntimeProviderPluginIdentity
    backend: RuntimeBackendIdentity
    capabilities: RuntimeProviderCapabilities
    artifact_identity: HFSnapshotArtifactIdentity
    execution_settings: RuntimeExecutionSettings
    device: RuntimeDeviceFacts
    outer_image_digest: str


@dataclass
class _HFTransformersSession:
    _scorer: RuntimeScorer
    _close_callback: Callable[[], None] | None = None
    _artifact_identity_sha256: str | None = None
    _receipt_provenance: _HFReceiptProvenance | None = None
    _revalidate_binding: Callable[[], None] | None = field(
        default=None, repr=False, compare=False
    )
    _latest_observation_sha256: str | None = None
    _closed: bool = False

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("runtime provider session is closed")

    def score(self, batch: EvaluationBatch) -> ScoringObservation:
        """Delegate scoring and fail closed on schedule/pairing drift."""

        self._require_open()
        self._latest_observation_sha256 = None
        provenance = self._receipt_provenance
        if provenance is None:
            direct_scorer = cast(
                Callable[[EvaluationBatch], ScoringObservation], self._scorer
            )
            observation = direct_scorer(batch)
        else:
            if self._revalidate_binding is not None:
                self._revalidate_binding()
            try:
                observation = self._scorer(batch, provenance.execution_settings)
            except TypeError as exc:
                raise RuntimeError(
                    "strict hf_transformers scorer must accept the exact runtime "
                    "execution settings contract"
                ) from exc
            finally:
                if self._revalidate_binding is not None:
                    self._revalidate_binding()
        if not isinstance(observation, ScoringObservation):
            raise TypeError("runtime scorer must return ScoringObservation")
        if observation.provider_name != HFTransformersProvider.name:
            raise ValueError("scoring observation provider does not match session")
        if (
            self._artifact_identity_sha256 is not None
            and observation.artifact_identity_sha256 != self._artifact_identity_sha256
        ):
            raise ValueError(
                "scoring observation artifact identity does not match session"
            )
        if observation.schedule_sha256 != batch.schedule_sha256:
            raise ValueError("scoring observation schedule does not match batch")
        expected_pairing = tuple(
            (record.record_id, record.input_sha256) for record in batch.records
        )
        observed_pairing = tuple(
            (record.record_id, record.input_sha256) for record in observation.records
        )
        if observed_pairing != expected_pairing:
            raise ValueError("scoring observation pairing does not match batch")
        from invarlock.runtime_provider_evidence import encode_scoring_observation

        self._latest_observation_sha256 = _sha256(
            encode_scoring_observation(observation)
        )
        return observation

    def runtime_receipt(self) -> RuntimeProviderReceipt:
        """Return strict provenance bound to the latest complete observation."""

        self._require_open()
        if self._receipt_provenance is None:
            raise RuntimeError(
                "runtime provider receipt requires strict authenticated execution"
            )
        if self._latest_observation_sha256 is None:
            raise RuntimeError("runtime provider receipt is unavailable before scoring")
        provenance = self._receipt_provenance
        return RuntimeProviderReceipt(
            plugin=provenance.plugin,
            backend=provenance.backend,
            capabilities=provenance.capabilities,
            artifact_identity=provenance.artifact_identity,
            execution_settings=provenance.execution_settings,
            device=provenance.device,
            outer_image_digest=provenance.outer_image_digest,
            scoring_observation_sha256=self._latest_observation_sha256,
        )

    def close(self) -> None:
        """Run the existing lifecycle callback at most once."""

        if self._closed:
            return
        self._closed = True
        if self._close_callback is not None:
            self._close_callback()


class HFTransformersProvider:
    """Reference provider for the existing in-process HF adapter pipeline."""

    name = "hf_transformers"
    abi_version = INVARLOCK_RUNTIME_PROVIDER_ABI

    def validate_config(self, spec: ModelRuntimeSpec) -> None:
        if spec.provider_name != self.name:
            raise ValueError(
                f"provider_name must be {self.name!r}, got {spec.provider_name!r}"
            )
        unknown = set(spec.settings) - _ALLOWED_SETTINGS
        if unknown:
            rendered = ", ".join(sorted(unknown))
            raise ValueError(f"unsupported hf_transformers setting(s): {rendered}")
        _validate_setting_values(spec)

    def capabilities(self) -> RuntimeProviderCapabilities:
        return RuntimeProviderCapabilities(
            provider_name=self.name,
            artifact_formats=("hf_snapshot",),
            tasks=("text_causal",),
            metrics=(
                "exact_match",
                "normalized_nll_per_utf8_byte",
            ),
            execution_modes=("in_process",),
            required_extra="hf",
            required_image=None,
        )

    def identify_artifact(self, spec: ModelRuntimeSpec) -> HFSnapshotArtifactIdentity:
        self.validate_config(spec)
        immutable_revision = _optional_text(spec.settings, "immutable_revision")
        checkpoint_tree_sha256 = _optional_sha256(
            spec.settings, "checkpoint_tree_sha256"
        )
        tokenizer_metadata_sha256 = _optional_sha256(
            spec.settings, "tokenizer_metadata_sha256"
        )
        if immutable_revision is None and checkpoint_tree_sha256 is None:
            raise ValueError(
                "hf_transformers requires an immutable identity revision or tree digest"
            )
        if tokenizer_metadata_sha256 is None:
            raise ValueError(
                "hf_transformers requires tokenizer_metadata_sha256 for artifact identity"
            )
        logical_model_id = spec.model_id
        if _is_local_path_like(spec.model_id):
            if checkpoint_tree_sha256 is None:
                raise ValueError(
                    "local hf_transformers paths require checkpoint_tree_sha256"
                )
            logical_model_id = f"local-checkpoint-{checkpoint_tree_sha256[:12]}"
        return HFSnapshotArtifactIdentity(
            model_id=logical_model_id,
            immutable_revision=immutable_revision,
            checkpoint_tree_sha256=checkpoint_tree_sha256,
            tokenizer_metadata_sha256=tokenizer_metadata_sha256,
        )

    def prepare_execution(
        self,
        spec: ModelRuntimeSpec,
        resources: RuntimeArtifactResources,
    ) -> RuntimeExecutionContext:
        """Load and authenticate one local checkpoint without network or remote code."""

        self.validate_config(spec)
        resources.require_support_names(frozenset())
        checkpoint = resources.primary_path()
        if not checkpoint.is_dir():
            raise ValueError("hf_transformers primary artifact must be a directory")
        identity = self.identify_artifact(spec)
        expected_tree = identity.checkpoint_tree_sha256
        if expected_tree is None:
            raise ValueError(
                "hf_transformers preparation requires checkpoint_tree_sha256"
            )
        try:
            observed_tree = checkpoint_tree_sha256(checkpoint).removeprefix("sha256:")
        except (CheckpointIdentityError, OSError) as exc:
            raise ValueError(
                "hf_transformers checkpoint could not be authenticated"
            ) from exc
        if observed_tree != expected_tree:
            raise ValueError(
                "hf_transformers primary artifact does not match checkpoint_tree_sha256"
            )

        transformers = importlib.import_module("transformers")
        auto_tokenizer = getattr(transformers, "AutoTokenizer", None)
        tokenizer_from_pretrained = getattr(auto_tokenizer, "from_pretrained", None)
        auto_model = getattr(transformers, "AutoModelForCausalLM", None)
        model_from_pretrained = getattr(auto_model, "from_pretrained", None)
        if not callable(tokenizer_from_pretrained):
            raise RuntimeError("transformers AutoTokenizer is unavailable")
        if not callable(model_from_pretrained):
            raise RuntimeError("transformers AutoModelForCausalLM is unavailable")
        tokenizer = tokenizer_from_pretrained(
            str(checkpoint),
            local_files_only=True,
            trust_remote_code=False,
        )
        model = model_from_pretrained(
            str(checkpoint),
            local_files_only=True,
            trust_remote_code=False,
            use_safetensors=True,
        )
        move_to_device = getattr(model, "to", None)
        if not callable(move_to_device):
            raise RuntimeError("loaded HF model does not expose to()")
        model = move_to_device(resources.device_kind)
        evaluate = getattr(model, "eval", None)
        if not callable(evaluate):
            raise RuntimeError("loaded HF model does not expose eval()")
        evaluate()
        identity_sha256 = artifact_identity_sha256(identity)
        scorer = HFTransformersCausalScorer(
            model=model,
            tokenizer=tokenizer,
            artifact_identity_sha256=identity_sha256,
            checkpoint_path=checkpoint,
        )
        require_loaded_hf_checkpoint_binding(
            spec=spec,
            identity=identity,
            model=model,
            tokenizer=tokenizer,
            checkpoint=checkpoint,
        )
        return RuntimeExecutionContext(
            strict=True,
            allow_network=False,
            container_image_digest=resources.container_image_digest,
            device_kind=resources.device_kind,
            artifact_identity_sha256=identity_sha256,
            provider_state=model,
            scorer=scorer,
        )

    def open(
        self,
        spec: ModelRuntimeSpec,
        context: RuntimeExecutionContext,
    ) -> _HFTransformersSession:
        self.validate_config(spec)
        if context.strict:
            _require_strict_runtime_boundary(context)
        if context.strict and context.artifact_identity_sha256 is None:
            raise ValueError(
                "strict hf_transformers execution requires artifact_identity_sha256"
            )
        for field_name in ("provider_state", "scorer"):
            if getattr(context, field_name) is None:
                raise ValueError(
                    f"hf_transformers requires prebound {field_name} in context"
                )
        identity = self.identify_artifact(spec)
        if context.artifact_identity_sha256 is not None:
            expected_identity_sha256 = artifact_identity_sha256(identity)
            if context.artifact_identity_sha256 != expected_identity_sha256:
                raise ValueError(
                    "runtime context artifact identity does not match model spec"
                )
        provenance: _HFReceiptProvenance | None = None
        revalidate_binding: Callable[[], None] | None = None
        runtime_scorer = cast(RuntimeScorer, context.scorer)
        if context.strict:
            runtime_scorer = _require_strict_execution_binding(
                spec=spec,
                identity=identity,
                context=context,
            )
            image_digest = context.container_image_digest
            assert image_digest is not None
            execution_settings = _strict_execution_settings(spec)
            backend = _installed_backend_identity(context.provider_state)
            device = _observed_device_facts(
                context.provider_state,
                expected_device_kind=context.device_kind,
            )
            provenance = _HFReceiptProvenance(
                plugin=RuntimeProviderPluginIdentity(
                    name=self.name,
                    distribution="invarlock",
                    distribution_version=INVARLOCK_VERSION,
                ),
                backend=backend,
                capabilities=self.capabilities(),
                artifact_identity=identity,
                execution_settings=execution_settings,
                device=device,
                outer_image_digest=image_digest,
            )
            if isinstance(runtime_scorer, HFTransformersCausalScorer):
                model = context.provider_state
                tokenizer = runtime_scorer.tokenizer

                def revalidate_binding() -> None:
                    require_loaded_hf_checkpoint_binding(
                        spec=spec,
                        identity=identity,
                        model=model,
                        tokenizer=tokenizer,
                        checkpoint=runtime_scorer.checkpoint_path,
                    )

        return _HFTransformersSession(
            _scorer=runtime_scorer,
            _close_callback=context.close_callback,
            _artifact_identity_sha256=context.artifact_identity_sha256,
            _receipt_provenance=provenance,
            _revalidate_binding=revalidate_binding,
        )


__all__ = [
    "HFTransformersCausalScorer",
    "HFTransformersProvider",
    "INVARLOCK_RUNTIME_PROVIDER_ABI",
    "hf_tokenizer_contract_sha256",
    "require_loaded_hf_checkpoint_binding",
]
