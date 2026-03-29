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

import base64
import json
import re
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from invarlock.security import is_secure_path

if TYPE_CHECKING:
    from .capabilities import ModelCapabilities, QuantizationConfig

SCALAR_TYPES = (int, float, str, bool)
_BENIGN_HF_UNEXPECTED_KEY_RE = re.compile(r"(?:^|.*\.)attn\.(?:masked_)?bias$")


@dataclass(frozen=True)
class HFPretrainedLoadDiagnostic:
    kind: str
    entries: tuple[str, ...]


def _sanitize_param_name(name: str) -> str:
    """Return a filesystem-safe parameter name."""
    return name.replace(".", "__").replace("/", "_")


def _ensure_secure_dir(path: Path) -> None:
    """Ensure snapshot directory uses 0o700 permissions."""
    path.mkdir(parents=True, exist_ok=True)
    path.chmod(0o700)
    if not is_secure_path(path):
        raise RuntimeError(
            f"Snapshot directory {path} must have permissions 0o700 for security."
        )


def _resolve_named_parameter(
    module: torch.nn.Module, path: str
) -> torch.nn.Parameter | None:
    """Resolve a parameter by dotted path, returning None if missing."""
    current: Any = module
    parts = path.split(".")
    for name in parts[:-1]:
        current = getattr(current, name, None)
        if current is None:
            return None
    leaf = getattr(current, parts[-1], None)
    if isinstance(leaf, torch.nn.Parameter):
        return leaf
    return None


def _require_safetensors_runtime() -> tuple[Any, Any, Any, Any]:
    try:
        from safetensors.torch import load as load_tensors
        from safetensors.torch import load_file as load_tensor_file
        from safetensors.torch import save as save_tensors
        from safetensors.torch import save_file as save_tensor_file
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "safetensors is required for secure HF snapshot restore. "
            "Install the adapters extra or add safetensors to the runtime image."
        ) from exc
    return save_tensors, load_tensors, save_tensor_file, load_tensor_file


def _serialize_snapshot_blob(
    *,
    tensors: Mapping[str, torch.Tensor],
    metadata: dict[str, Any],
) -> bytes:
    save_tensors, _, _, _ = _require_safetensors_runtime()
    tensor_blob = save_tensors(
        {name: tensor.detach().cpu().contiguous() for name, tensor in tensors.items()}
    )
    envelope = {
        "format": "invarlock-safetensors-v1",
        "metadata": metadata,
        "tensors_base64": base64.b64encode(tensor_blob).decode("ascii"),
    }
    return json.dumps(envelope, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _deserialize_snapshot_blob(
    blob: bytes,
) -> tuple[dict[str, Any], Mapping[str, torch.Tensor]]:
    _, load_tensors, _, _ = _require_safetensors_runtime()
    envelope = json.loads(blob.decode("utf-8"))
    if not isinstance(envelope, dict):
        raise TypeError("Invalid snapshot payload")
    if envelope.get("format") != "invarlock-safetensors-v1":
        raise ValueError("Unsupported snapshot payload format")
    metadata = envelope.get("metadata")
    if not isinstance(metadata, dict):
        raise TypeError("Invalid snapshot payload metadata")
    encoded = envelope.get("tensors_base64")
    if not isinstance(encoded, str) or not encoded:
        raise TypeError("Invalid snapshot payload tensors")
    tensor_blob = base64.b64decode(encoded.encode("ascii"))
    tensors = load_tensors(tensor_blob)
    if not isinstance(tensors, Mapping):
        raise TypeError("Invalid snapshot tensor mapping")
    return metadata, tensors


def _load_chunked_tensor(path: Path) -> torch.Tensor:
    _, _, _, load_tensor_file = _require_safetensors_runtime()
    payload = load_tensor_file(str(path))
    tensor = payload.get("tensor") if isinstance(payload, Mapping) else None
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Invalid snapshot tensor payload for {path.name}")
    return tensor


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
        try:
            if prefer_local_files_only:
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
                loaded = loader.from_pretrained(
                    model_id, local_files_only=True, **kwargs
                )
            else:
                loaded = loader.from_pretrained(model_id, **kwargs)
        except Exception:
            if not prefer_local_files_only:
                raise
            try:
                loaded = loader.from_pretrained(
                    model_id, output_loading_info=True, **kwargs
                )
            except TypeError as exc:
                if "output_loading_info" not in str(exc):
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

        For quantized models (BNB, AWQ, GPTQ), device movement may be
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
            # Model handles its own device placement (e.g., BNB, AWQ, GPTQ)
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
        Check if a model is quantized (BNB, AWQ, GPTQ).

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
                for q in ["linear8bit", "linear4bit", "quantlinear", "awqlinear"]
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
        except Exception:
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

        state_dict: dict[str, torch.Tensor] = {}
        device_map: dict[str, str] = {}

        for name, param in model.named_parameters():
            state_key = f"params.{name}"
            state_dict[state_key] = param.detach().cpu().clone()
            device_map[state_key] = str(param.device)

        for name, buffer in model.named_buffers():
            state_key = f"buffers.{name}"
            state_dict[state_key] = buffer.detach().cpu().clone()
            device_map[state_key] = str(buffer.device)

        metadata: dict[str, Any] = {
            "config": self._serialize_config(model.config)
            if hasattr(model, "config")
            else {},
            "device_map": device_map,
            "model_class": model.__class__.__name__,
            "weight_tying": self._extract_weight_tying_info(model),
        }
        return _serialize_snapshot_blob(tensors=state_dict, metadata=metadata)

    def restore(self, model: torch.nn.Module, blob: bytes) -> None:
        """
        Restore model state produced by `snapshot`.

        Args:
            model: Model to restore in-place.
            blob: Bytes payload from snapshot.
        """

        metadata, state_dict = _deserialize_snapshot_blob(blob)
        device_map = metadata.get("device_map", {})
        if not isinstance(device_map, dict):
            device_map = {}

        for name, param in model.named_parameters():
            state_key = f"params.{name}"
            if state_key not in state_dict:
                continue
            target_device = torch.device(device_map.get(state_key, "cpu"))
            with torch.no_grad():
                param.copy_(state_dict[state_key].to(target_device))

        for name, buffer_param in model.named_buffers():
            state_key = f"buffers.{name}"
            if state_key not in state_dict:
                continue
            target_device = torch.device(device_map.get(state_key, "cpu"))
            buffer_param.copy_(state_dict[state_key].to(target_device))

        original_tying = metadata.get("weight_tying", {})
        if isinstance(original_tying, dict) and original_tying:
            current_tying = self._extract_weight_tying_info(model)
            for tied_param, source_param in original_tying.items():
                if current_tying.get(tied_param) != source_param:
                    self._restore_weight_tying(model, tied_param, source_param)

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

        snapshot_dir = Path(tempfile.mkdtemp(prefix=prefix))
        _ensure_secure_dir(snapshot_dir)
        _, _, save_tensor_file, _ = _require_safetensors_runtime()

        manifest: dict[str, Any] = {
            "model_class": model.__class__.__name__,
            "config": self._serialize_config(model.config)
            if hasattr(model, "config")
            else {},
            "tensor_format": "safetensors",
            "params": {},
            "params_meta": {},
            "buffers": {},
            "buffers_meta": {},
            "device_map": {},
            "weight_tying": self._extract_weight_tying_info(model),
        }

        for name, param in model.named_parameters():
            filename = f"param__{_sanitize_param_name(name)}.safetensors"
            file_path = snapshot_dir / filename
            save_tensor_file(
                {"tensor": param.detach().cpu().contiguous()},
                str(file_path),
            )
            manifest["params"][name] = filename
            manifest["params_meta"][name] = {
                "shape": [int(x) for x in param.shape],
                "dtype": str(param.dtype),
            }
            manifest["device_map"][name] = str(param.device)

        for name, buffer in model.named_buffers():
            filename = f"buffer__{_sanitize_param_name(name)}.safetensors"
            file_path = snapshot_dir / filename
            save_tensor_file(
                {"tensor": buffer.detach().cpu().contiguous()},
                str(file_path),
            )
            manifest["buffers"][name] = filename
            manifest["buffers_meta"][name] = {
                "shape": [int(x) for x in buffer.shape],
                "dtype": str(buffer.dtype),
            }
            manifest["device_map"][f"buffer::{name}"] = str(buffer.device)

        manifest_path = snapshot_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return str(snapshot_dir)

    def restore_chunked(self, model: torch.nn.Module, snapshot_path: str) -> None:
        """
        Restore a chunked snapshot produced by `snapshot_chunked`.

        Args:
            model: Model to restore in-place.
            snapshot_path: Directory path created by `snapshot_chunked`.
        """

        snapshot_dir = Path(snapshot_path)
        manifest_path = snapshot_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing manifest for snapshot at {snapshot_path}")

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        param_map = dict(model.named_parameters())
        buffer_map = dict(model.named_buffers())

        device_map = manifest.get("device_map", {})

        params_manifest = manifest.get("params", {})
        if not isinstance(params_manifest, dict):
            raise TypeError("Invalid snapshot manifest: params must be a mapping")
        buffers_manifest = manifest.get("buffers", {})
        if not isinstance(buffers_manifest, dict):
            raise TypeError("Invalid snapshot manifest: buffers must be a mapping")
        params_meta = manifest.get("params_meta", {})
        buffers_meta = manifest.get("buffers_meta", {})

        # Preflight: ensure manifest/model agreement and tensor readability before copying.
        for name, filename in params_manifest.items():
            if name not in param_map:
                raise KeyError(f"Snapshot parameter missing in target model: {name}")
            if not isinstance(filename, str) or not filename:
                raise TypeError(f"Invalid snapshot manifest filename for param: {name}")
            file_path = snapshot_dir / filename
            if not file_path.exists():
                raise FileNotFoundError(
                    f"Missing snapshot tensor for param: {file_path}"
                )
            tensor = _load_chunked_tensor(file_path)
            meta = params_meta.get(name) if isinstance(params_meta, dict) else None
            if isinstance(meta, dict):
                expected_shape = meta.get("shape")
                expected_dtype = meta.get("dtype")
                if isinstance(expected_shape, list) and list(tensor.shape) != list(
                    expected_shape
                ):
                    raise ValueError(
                        f"Snapshot tensor shape mismatch for param: {name}"
                    )
                if isinstance(expected_dtype, str) and expected_dtype:
                    if str(tensor.dtype) != expected_dtype:
                        raise ValueError(
                            f"Snapshot tensor dtype mismatch for param: {name}"
                        )

        for name, filename in buffers_manifest.items():
            if name not in buffer_map:
                raise KeyError(f"Snapshot buffer missing in target model: {name}")
            if not isinstance(filename, str) or not filename:
                raise TypeError(
                    f"Invalid snapshot manifest filename for buffer: {name}"
                )
            file_path = snapshot_dir / filename
            if not file_path.exists():
                raise FileNotFoundError(
                    f"Missing snapshot tensor for buffer: {file_path}"
                )
            tensor = _load_chunked_tensor(file_path)
            meta = buffers_meta.get(name) if isinstance(buffers_meta, dict) else None
            if isinstance(meta, dict):
                expected_shape = meta.get("shape")
                expected_dtype = meta.get("dtype")
                if isinstance(expected_shape, list) and list(tensor.shape) != list(
                    expected_shape
                ):
                    raise ValueError(
                        f"Snapshot tensor shape mismatch for buffer: {name}"
                    )
                if isinstance(expected_dtype, str) and expected_dtype:
                    if str(tensor.dtype) != expected_dtype:
                        raise ValueError(
                            f"Snapshot tensor dtype mismatch for buffer: {name}"
                        )

        # Restore parameters/buffers (second pass) after successful preflight.
        for name, filename in params_manifest.items():
            target = param_map[name]
            target_device = torch.device(device_map.get(name, str(target.device)))
            tensor = _load_chunked_tensor(snapshot_dir / filename)
            with torch.no_grad():
                target.copy_(tensor.to(target_device))

        for name, filename in buffers_manifest.items():
            target = buffer_map[name]
            key = f"buffer::{name}"
            target_device = torch.device(device_map.get(key, str(target.device)))
            tensor = _load_chunked_tensor(snapshot_dir / filename)
            target.copy_(tensor.to(target_device))

        original_tying = manifest.get("weight_tying", {})
        if isinstance(original_tying, dict) and original_tying:
            current_tying = self._extract_weight_tying_info(model)
            for tied_param, source_param in original_tying.items():
                if current_tying.get(tied_param) != source_param:
                    self._restore_weight_tying(model, tied_param, source_param)

    # ------------------------------------------------------------------
    # Weight-tying hooks (overridden by concrete adapters)
    # ------------------------------------------------------------------
    def _extract_weight_tying_info(self, model: torch.nn.Module) -> dict[str, str]:
        """Return mapping of tied parameter names to source parameter names."""

        tying: dict[str, str] = {}
        try:
            named = model.named_parameters(remove_duplicate=False)  # type: ignore[call-arg]
        except TypeError:  # pragma: no cover - torch version dependent
            named = model.named_parameters()
        params = dict(named)

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
            except Exception:
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
        """Restore a weight-tying relationship (no-op by default)."""
        model_params = dict(model.named_parameters())
        tied = model_params.get(tied_param)
        source = model_params.get(source_param)
        if tied is None or source is None:
            return
        with torch.no_grad():
            tied.copy_(source)

    def validate_weight_tying(self, model: torch.nn.Module) -> None:
        """Raise if a known weight-tying relationship has been broken."""
        tying = self._extract_weight_tying_info(model)
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
            if not torch.allclose(tied, source):
                from invarlock.core.exceptions import AdapterError

                raise AdapterError(
                    code="E202",
                    message="ADAPTER-STRUCTURE-INVALID: weight-tying invariant violated",
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
            except Exception:
                data = None
            if isinstance(data, dict):
                return _collect(data)

        try:
            data = vars(config)
        except TypeError:
            data = None
        if isinstance(data, dict):
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
