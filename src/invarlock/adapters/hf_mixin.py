"""
Shared HuggingFace adapter mixin.
=================================

Provides reusable functionality for InvarLock's HuggingFace adapters:
- Device resolution helpers
- Safe device movement for quantized models
- Snapshot/restore with device awareness
- Chunked snapshot helpers to reduce peak memory usage
- Lightweight config serialization
- Weight-tying detection plumbing
- Quantization detection and capabilities
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from .hf_mixin_loading import HFPretrainedLoadDiagnostic, _is_local_loader_cache_miss
from .hf_mixin_snapshot import (
    _deserialize_snapshot_blob as _deserialize_snapshot_blob,
)
from .hf_mixin_snapshot import (
    _ensure_secure_dir as _ensure_secure_dir,
)
from .hf_mixin_snapshot import (
    _load_chunked_tensor as _load_chunked_tensor,
)
from .hf_mixin_snapshot import (
    _require_safetensors_runtime as _require_safetensors_runtime,
)
from .hf_mixin_snapshot import (
    _resolve_named_parameter as _resolve_named_parameter,
)
from .hf_mixin_snapshot import (
    _sanitize_param_name as _sanitize_param_name,
)
from .hf_mixin_snapshot import (
    _serialize_snapshot_blob as _serialize_snapshot_blob,
)
from .hf_mixin_snapshot import (
    _set_named_parameter_alias as _set_named_parameter_alias,
)
from .hf_mixin_snapshot import (
    restore_model,
    restore_model_chunked,
    snapshot_model,
    snapshot_model_chunked,
)

if TYPE_CHECKING:
    from .capabilities import ModelCapabilities, QuantizationConfig

SCALAR_TYPES = (int, float, str, bool)
_BENIGN_HF_UNEXPECTED_KEY_RE = re.compile(r"(?:^|.*\.)attn\.(?:masked_)?bias$")


def _iter_named_parameters_preserving_ties(
    model: torch.nn.Module,
) -> list[tuple[str, torch.nn.Parameter]]:
    """Return named parameters while preserving duplicate aliases when supported."""

    named_parameters = getattr(model, "named_parameters", None)
    if not callable(named_parameters):
        raise TypeError("model.named_parameters must be callable")
    try:
        return list(named_parameters(remove_duplicate=False))
    except TypeError:  # pragma: no cover - torch version dependent
        aliased_params: list[tuple[str, torch.nn.Parameter]] = []
        for module_name, module in model.named_modules():
            raw_parameters = getattr(module, "_parameters", {})
            if not isinstance(raw_parameters, dict):
                continue
            for attr_name, value in raw_parameters.items():
                if not isinstance(value, torch.nn.Parameter):
                    continue
                full_name = f"{module_name}.{attr_name}" if module_name else attr_name
                aliased_params.append((full_name, value))
        if aliased_params:
            return aliased_params
        return list(named_parameters())


class HFAdapterMixin:
    """Reusable utilities for HuggingFace-backed adapters."""

    def _is_benign_hf_unexpected_key(self, key: object) -> bool:
        """Return True for known benign HF checkpoint keys.

        GPT-2 style checkpoints commonly surface `attn.bias` / `attn.masked_bias`
        as unexpected keys even for identical checkpoints. They do not indicate a
        broken load and should not leak as noisy console output.
        """

        return isinstance(key, str) and bool(_BENIGN_HF_UNEXPECTED_KEY_RE.search(key))

    @property
    def pretrained_load_diagnostics(self) -> tuple[HFPretrainedLoadDiagnostic, ...]:
        diagnostics = getattr(self, "_pretrained_load_diagnostics", ())
        return diagnostics if isinstance(diagnostics, tuple) else ()

    def _capture_filtered_loading_info(
        self,
        loading_info: Mapping[str, Any] | None,
    ) -> tuple[HFPretrainedLoadDiagnostic, ...]:
        """Capture actionable HF loading info as typed diagnostics."""
        if not isinstance(loading_info, Mapping):
            return ()

        unexpected_keys = [
            key
            for key in list(loading_info.get("unexpected_keys") or [])
            if not self._is_benign_hf_unexpected_key(key)
        ]
        missing_keys = list(loading_info.get("missing_keys") or [])
        mismatched_keys = list(loading_info.get("mismatched_keys") or [])
        error_msgs = [msg for msg in list(loading_info.get("error_msgs") or []) if msg]

        mismatch_names: list[str] = []
        for item in mismatched_keys:
            if isinstance(item, Mapping):
                name = item.get("key") or item.get("name")
                if isinstance(name, str) and name:
                    mismatch_names.append(name)
                    continue
            mismatch_names.append(str(item))

        diagnostics: list[HFPretrainedLoadDiagnostic] = []
        if unexpected_keys:
            diagnostics.append(
                HFPretrainedLoadDiagnostic(
                    kind="unexpected_keys",
                    entries=tuple(str(key) for key in unexpected_keys),
                )
            )
        if missing_keys:
            diagnostics.append(
                HFPretrainedLoadDiagnostic(
                    kind="missing_keys",
                    entries=tuple(str(key) for key in missing_keys),
                )
            )
        if mismatch_names:
            diagnostics.append(
                HFPretrainedLoadDiagnostic(
                    kind="mismatched_keys",
                    entries=tuple(mismatch_names),
                )
            )
        if error_msgs:
            diagnostics.append(
                HFPretrainedLoadDiagnostic(
                    kind="error_messages",
                    entries=tuple(str(msg) for msg in error_msgs),
                )
            )
        return tuple(diagnostics)

    def _load_pretrained_model(self, loader: Any, model_id: str, **kwargs: Any) -> Any:
        """Load a HF model while filtering known benign loading-info noise."""

        self._pretrained_load_diagnostics = ()
        prefer_local_files_only = bool(kwargs.pop("prefer_local_files_only", False))
        collect_loading_info = bool(kwargs.pop("collect_loading_info", True))
        load_device = kwargs.pop("load_device", None)
        from .hf_loading import apply_memory_efficient_load_defaults

        kwargs = apply_memory_efficient_load_defaults(
            model_id,
            kwargs,
            load_device=load_device,
        )
        try:
            if not collect_loading_info:
                if prefer_local_files_only:
                    try:
                        loaded = loader.from_pretrained(
                            model_id,
                            local_files_only=True,
                            **kwargs,
                        )
                    except OSError as local_error:
                        if not _is_local_loader_cache_miss(local_error):
                            raise
                        loaded = loader.from_pretrained(model_id, **kwargs)
                else:
                    loaded = loader.from_pretrained(model_id, **kwargs)
            elif prefer_local_files_only:
                loaded = loader.from_pretrained(
                    model_id,
                    output_loading_info=True,
                    local_files_only=True,
                    **kwargs,
                )
            else:
                loaded = loader.from_pretrained(
                    model_id, output_loading_info=True, **kwargs
                )
        except TypeError as exc:
            if "output_loading_info" not in str(exc):
                raise
            if prefer_local_files_only:
                try:
                    loaded = loader.from_pretrained(
                        model_id, local_files_only=True, **kwargs
                    )
                except OSError as local_error:
                    if not _is_local_loader_cache_miss(local_error):
                        raise
                    try:
                        loaded = loader.from_pretrained(
                            model_id, output_loading_info=True, **kwargs
                        )
                    except TypeError as retry_exc:
                        if "output_loading_info" not in str(retry_exc):
                            raise
                        loaded = loader.from_pretrained(model_id, **kwargs)
            else:
                loaded = loader.from_pretrained(model_id, **kwargs)
        except OSError as exc:
            if not prefer_local_files_only or not _is_local_loader_cache_miss(exc):
                raise
            try:
                loaded = loader.from_pretrained(
                    model_id, output_loading_info=True, **kwargs
                )
            except TypeError as retry_exc:
                if "output_loading_info" not in str(retry_exc):
                    raise
                loaded = loader.from_pretrained(model_id, **kwargs)

        if (
            isinstance(loaded, tuple)
            and len(loaded) == 2
            and isinstance(loaded[1], Mapping)
        ):
            model, loading_info = loaded
            self._pretrained_load_diagnostics = self._capture_filtered_loading_info(
                loading_info
            )
            return model
        return loaded

    # ------------------------------------------------------------------
    # Device helpers
    # ------------------------------------------------------------------
    def _resolve_device(
        self, device: str | torch.device | None = "auto"
    ) -> torch.device:
        """
        Resolve a target torch.device for model placement.

        Args:
            device: Requested device ("auto" selects CUDA→MPS→CPU).

        Returns:
            torch.device for placement.
        """

        if isinstance(device, torch.device):
            return device

        device_str = "auto" if device is None else str(device)
        device_str = device_str.lower()

        if device_str == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")

        return torch.device(device_str)

    def _safe_to_device(
        self,
        model: torch.nn.Module,
        device: str | torch.device | None = "auto",
        capabilities: ModelCapabilities | None = None,
    ) -> torch.nn.Module:
        """
        Safely move model to device, respecting quantization constraints.

        For quantized models (BNB, AWQ, GPTQ, torchao, HQQ), device movement may be
        impossible or already handled by the loading mechanism. This
        method checks the model's capabilities before attempting .to().

        Args:
            model: The model to move.
            device: Target device ("auto", "cuda", "mps", "cpu").
            capabilities: Pre-computed capabilities, or None to auto-detect.

        Returns:
            The model (possibly on the new device, or unchanged if not movable).
        """
        target_device = self._resolve_device(device)

        # If transformers already sharded/placed the model, skip explicit .to().
        if getattr(model, "hf_device_map", None):
            return model

        # Auto-detect capabilities if not provided
        if capabilities is None:
            capabilities = self._detect_capabilities(model)

        # Check if model can be moved
        if capabilities is not None and not capabilities.device_movable:
            # Model handles its own device placement (e.g., BNB, AWQ, GPTQ, torchao, HQQ)
            # Log this decision for debugging but don't attempt .to()
            return model

        # Safe to move
        return model.to(target_device)

    def _detect_capabilities(self, model: torch.nn.Module) -> ModelCapabilities | None:
        """
        Detect model capabilities from a loaded model instance.

        Args:
            model: Loaded model instance.

        Returns:
            ModelCapabilities if detection succeeds, None otherwise.
        """
        try:
            from .capabilities import detect_capabilities_from_model

            return detect_capabilities_from_model(model)
        except ImportError:
            return None

    def _is_quantized_model(self, model: torch.nn.Module) -> bool:
        """
        Check if a model is quantized (BNB, AWQ, GPTQ, torchao, HQQ).

        This is a quick heuristic check that doesn't require full
        capability detection.

        Args:
            model: Model to check.

        Returns:
            True if the model appears to be quantized.
        """
        config = getattr(model, "config", None)
        if config is None:
            return False

        # Check for quantization_config attribute
        quant_cfg = getattr(config, "quantization_config", None)
        if quant_cfg is not None:
            return True

        # Check for BNB-specific attributes on the model
        if hasattr(model, "is_loaded_in_8bit") and model.is_loaded_in_8bit:
            return True
        if hasattr(model, "is_loaded_in_4bit") and model.is_loaded_in_4bit:
            return True

        # Check for quantized module types in the model
        for module in model.modules():
            module_name = module.__class__.__name__.lower()
            if any(
                q in module_name
                for q in [
                    "linear8bit",
                    "linear4bit",
                    "quantlinear",
                    "awqlinear",
                    "torchao",
                    "hqq",
                ]
            ):
                return True

        return False

    def _detect_quantization_config(
        self, model: torch.nn.Module
    ) -> QuantizationConfig | None:
        """
        Detect quantization configuration from a model.

        Args:
            model: Model to inspect.

        Returns:
            QuantizationConfig if quantization detected, None otherwise.
        """
        try:
            from .capabilities import detect_quantization_from_config

            config = getattr(model, "config", None)
            if config is not None:
                quant_cfg = detect_quantization_from_config(config)
                if quant_cfg.is_quantized():
                    return quant_cfg
        except ImportError:
            pass
        return None

    # ------------------------------------------------------------------
    # HF save/export helpers
    # ------------------------------------------------------------------
    def save_pretrained(self, model: torch.nn.Module, path: str | Path) -> bool:
        """
        Save a HuggingFace model in a HF-loadable directory.

        Args:
            model: HF Transformers model implementing save_pretrained
            path: Target directory path

        Returns:
            True on success, False otherwise
        """
        try:
            p = Path(path)
            p.mkdir(parents=True, exist_ok=True)
            # Most HF models implement save_pretrained
            save = getattr(model, "save_pretrained", None)
            if callable(save):
                save(str(p))
                return True
        except (OSError, RuntimeError, TypeError, ValueError):
            return False
        return False

    # ------------------------------------------------------------------
    # Snapshot / restore
    # ------------------------------------------------------------------
    def snapshot(self, model: torch.nn.Module) -> bytes:
        """
        Serialize model state with device awareness and weight-tying metadata.

        Args:
            model: HuggingFace model instance.

        Returns:
            Bytes payload backed by safetensors plus JSON metadata.
        """

        return snapshot_model(self, model)

    def restore(self, model: torch.nn.Module, blob: bytes) -> None:
        """
        Restore model state produced by `snapshot`.

        Args:
            model: Model to restore in-place.
            blob: Bytes payload from snapshot.
        """

        restore_model(self, model, blob)

    # ------------------------------------------------------------------
    # Chunked snapshot helpers
    # ------------------------------------------------------------------
    def snapshot_chunked(
        self, model: torch.nn.Module, *, prefix: str = "invarlock-snap-"
    ) -> str:
        """
        Create a chunked snapshot on disk to minimise in-memory footprint.

        Each parameter and buffer is serialized individually so only a single
        tensor resides in memory at a time. Metadata is recorded in manifest.json.
        """

        return snapshot_model_chunked(self, model, prefix=prefix)

    def restore_chunked(self, model: torch.nn.Module, snapshot_path: str) -> None:
        """
        Restore a chunked snapshot produced by `snapshot_chunked`.

        Args:
            model: Model to restore in-place.
            snapshot_path: Directory path created by `snapshot_chunked`.
        """

        restore_model_chunked(self, model, snapshot_path)

    # ------------------------------------------------------------------
    # Weight-tying hooks (overridden by concrete adapters)
    # ------------------------------------------------------------------
    def _extract_weight_tying_info(self, model: torch.nn.Module) -> dict[str, str]:
        """Return mapping of tied parameter names to source parameter names."""

        tying: dict[str, str] = {}
        params = dict(_iter_named_parameters_preserving_ties(model))

        def _is_tied(name_a: str, name_b: str) -> bool:
            a = params.get(name_a)
            b = params.get(name_b)
            if a is None or b is None:
                return False
            try:
                if a is b:
                    return True
                if hasattr(a, "data_ptr") and hasattr(b, "data_ptr"):
                    return int(a.data_ptr()) == int(b.data_ptr())
            except (RuntimeError, TypeError, ValueError):
                return False
            return False

        if _is_tied("lm_head.weight", "transformer.wte.weight"):
            tying["lm_head.weight"] = "transformer.wte.weight"

        if _is_tied("lm_head.weight", "model.embed_tokens.weight"):
            tying["lm_head.weight"] = "model.embed_tokens.weight"

        decoder_name = "cls.predictions.decoder.weight"
        if decoder_name in params:
            for candidate in (
                "bert.embeddings.word_embeddings.weight",
                "embeddings.word_embeddings.weight",
            ):
                if _is_tied(decoder_name, candidate):
                    tying[decoder_name] = candidate
                    break

        return tying

    def _restore_weight_tying(
        self, model: torch.nn.Module, tied_param: str, source_param: str
    ) -> None:
        """Restore a weight-tying relationship by rebinding the parameter."""
        source = _resolve_named_parameter(model, source_param)
        if source is None:
            raise KeyError(f"Missing source parameter for weight tying: {source_param}")
        _set_named_parameter_alias(model, tied_param, source)

    def validate_weight_tying(
        self,
        model: torch.nn.Module,
        *,
        expected_tying: Mapping[str, str] | None = None,
    ) -> None:
        """Raise if a known weight-tying relationship has been broken."""
        tying = dict(expected_tying or self._extract_weight_tying_info(model))
        if not tying:
            return

        model_params = dict(model.named_parameters())
        for tied_param, source_param in tying.items():
            tied = model_params.get(tied_param)
            source = model_params.get(source_param)
            if tied is None:
                tied = _resolve_named_parameter(model, tied_param)
            if source is None:
                source = _resolve_named_parameter(model, source_param)
            if tied is None or source is None:
                from invarlock.core.exceptions import AdapterError

                raise AdapterError(
                    code="E202",
                    message="ADAPTER-STRUCTURE-INVALID: missing tied/source parameter",
                    details={
                        "tied_param": tied_param,
                        "source_param": source_param,
                    },
                )
            same_storage = tied is source
            if not same_storage:
                try:
                    same_storage = int(tied.data_ptr()) == int(source.data_ptr())
                except (RuntimeError, TypeError, ValueError):
                    same_storage = False
            if not same_storage:
                from invarlock.core.exceptions import AdapterError

                raise AdapterError(
                    code="E202",
                    message=(
                        "ADAPTER-STRUCTURE-INVALID: weight-tying storage alias "
                        "was not restored"
                    ),
                    details={
                        "tied_param": tied_param,
                        "source_param": source_param,
                    },
                )

    # ------------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------------
    def _serialize_config(self, config: Any) -> dict[str, Any]:
        """Serialize HuggingFace config fields into simple Python types."""

        def _collect(data: dict[str, Any]) -> dict[str, Any]:
            out: dict[str, Any] = {}
            for key, value in data.items():
                if key.startswith("_") or key in {"method_calls"}:
                    continue
                if value is None or isinstance(value, SCALAR_TYPES):
                    out[key] = value
                elif isinstance(value, list | dict):
                    out[key] = value
            return out

        to_dict = getattr(config, "to_dict", None)
        if callable(to_dict):
            try:
                data = to_dict()
            except (RuntimeError, TypeError, ValueError):
                data = None
            if isinstance(data, dict):
                return _collect(data)

        try:
            data = vars(config)
        except TypeError:
            data = None
        if isinstance(data, dict) and data:
            return _collect(data)

        result: dict[str, Any] = {}
        for key in dir(config):
            if key.startswith("_") or key in {"torch_dtype"}:
                continue
            try:
                value = getattr(config, key)
            except AttributeError:
                continue
            if callable(value):
                continue
            if value is None or isinstance(value, SCALAR_TYPES):
                result[key] = value
            elif isinstance(value, list | dict):
                result[key] = value
        return result
