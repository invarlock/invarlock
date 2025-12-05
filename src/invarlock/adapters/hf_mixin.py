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

import io
import json
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from invarlock.security import is_secure_path

if TYPE_CHECKING:
    from .capabilities import ModelCapabilities, QuantizationConfig

SCALAR_TYPES = (int, float, str, bool)


def _sanitize_param_name(name: str) -> str:
    """Return a filesystem-safe parameter name."""
    return name.replace(".", "__").replace("/", "_")


def _ensure_secure_dir(path: Path) -> None:
    """Ensure snapshot directory uses 0o700 permissions."""
    path.mkdir(parents=True, exist_ok=True)
    os.chmod(path, 0o700)
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


class HFAdapterMixin:
    """Reusable utilities for HuggingFace-backed adapters."""

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
            Bytes payload produced by torch.save.
        """

        state_dict: dict[str, Any] = {}
        device_map: dict[str, str] = {}

        for name, param in model.named_parameters():
            state_key = f"params.{name}"
            state_dict[state_key] = param.detach().cpu().clone()
            device_map[state_key] = str(param.device)

        for name, buffer in model.named_buffers():
            state_key = f"buffers.{name}"
            state_dict[state_key] = buffer.detach().cpu().clone()
            device_map[state_key] = str(buffer.device)

        if hasattr(model, "config"):
            state_dict["config"] = self._serialize_config(model.config)

        state_dict["device_map"] = device_map
        state_dict["model_class"] = model.__class__.__name__
        state_dict["weight_tying"] = self._extract_weight_tying_info(model)

        buffer = io.BytesIO()
        torch.save(state_dict, buffer)
        return buffer.getvalue()

    def restore(self, model: torch.nn.Module, blob: bytes) -> None:
        """
        Restore model state produced by `snapshot`.

        Args:
            model: Model to restore in-place.
            blob: Bytes payload from snapshot.
        """

        buffer = io.BytesIO(blob)
        state_dict = torch.load(buffer, map_location="cpu", weights_only=False)

        device_map: dict[str, str] = state_dict.get("device_map", {})

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

        original_tying = state_dict.get("weight_tying", {})
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

        manifest: dict[str, Any] = {
            "model_class": model.__class__.__name__,
            "config": self._serialize_config(model.config)
            if hasattr(model, "config")
            else {},
            "params": {},
            "buffers": {},
            "device_map": {},
            "weight_tying": self._extract_weight_tying_info(model),
        }

        for name, param in model.named_parameters():
            filename = f"param__{_sanitize_param_name(name)}.pt"
            file_path = snapshot_dir / filename
            torch.save(param.detach().cpu(), file_path)
            manifest["params"][name] = filename
            manifest["device_map"][name] = str(param.device)

        for name, buffer in model.named_buffers():
            filename = f"buffer__{_sanitize_param_name(name)}.pt"
            file_path = snapshot_dir / filename
            torch.save(buffer.detach().cpu(), file_path)
            manifest["buffers"][name] = filename
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

        for name, filename in manifest.get("params", {}).items():
            file_path = snapshot_dir / filename
            if name not in param_map or not file_path.exists():
                continue
            target = param_map[name]
            target_device = torch.device(device_map.get(name, str(target.device)))
            tensor = torch.load(file_path, map_location="cpu")
            with torch.no_grad():
                target.copy_(tensor.to(target_device))

        for name, filename in manifest.get("buffers", {}).items():
            file_path = snapshot_dir / filename
            if name not in buffer_map or not file_path.exists():
                continue
            target = buffer_map[name]
            key = f"buffer::{name}"
            target_device = torch.device(device_map.get(key, str(target.device)))
            tensor = torch.load(file_path, map_location="cpu")
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
        param_names = set(dict(model.named_parameters()).keys())

        if "lm_head.weight" in param_names and "transformer.wte.weight" in param_names:
            tying["lm_head.weight"] = "transformer.wte.weight"

        decoder_name = "cls.predictions.decoder.weight"
        if decoder_name in param_names:
            for candidate in (
                "bert.embeddings.word_embeddings.weight",
                "embeddings.word_embeddings.weight",
            ):
                if candidate in param_names:
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

        result: dict[str, Any] = {}
        for key in dir(config):
            if key.startswith("_"):
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
