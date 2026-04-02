"""Support types and resource/dependency helpers for eval metrics."""

from __future__ import annotations

import gc
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import psutil
import torch
import torch.nn as nn

from invarlock.core.exceptions import MetricsError, ValidationError

logger = logging.getLogger(__name__)


class DependencyError(MetricsError):
    """Raised when required dependencies are missing."""


class ResourceError(MetricsError):
    """Raised when insufficient resources are available."""


MetricsProgressPhase = Literal["activation_collection", "mi_gini_cpu"]


@dataclass(frozen=True)
class MetricsProgressUpdate:
    """Neutral progress update emitted by reusable metrics code."""

    phase: MetricsProgressPhase
    completed: int
    total: int | None


MetricsProgressObserver = Callable[[MetricsProgressUpdate], None]


@dataclass
class MetricsConfig:
    """Configuration for metrics calculation with sensible defaults."""

    oracle_windows: int = 16
    max_tokens: int = 256
    max_samples_per_layer: int = 25_000

    auto_batch_size: bool = True
    memory_limit_gb: float | None = None
    cpu_fallback_threshold_gb: float = 0.5

    use_cache: bool = True
    cache_dir: Path | None = None
    progress_observer: MetricsProgressObserver | None = None

    clip_value: float = 1e3
    nan_replacement: float = 0.0
    inf_replacement: float = 1e4

    device: torch.device | None = None
    force_cpu: bool = False
    cleanup_after: bool = True

    strict_validation: bool = True
    allow_empty_data: bool = False

    sigma_max_margin: float = 0.98
    mi_gini_subsample_ratio: float = 0.05
    head_energy_layers_filter: bool = True

    def __post_init__(self) -> None:
        if self.oracle_windows < 0:
            raise ValidationError(
                code="E402",
                message="METRICS-VALIDATION-FAILED",
                details={"reason": "oracle_windows must be non-negative"},
            )
        if self.max_tokens <= 0:
            raise ValidationError(
                code="E402",
                message="METRICS-VALIDATION-FAILED",
                details={"reason": "max_tokens must be positive"},
            )
        if self.memory_limit_gb is not None and self.memory_limit_gb <= 0:
            raise ValidationError(
                code="E402",
                message="METRICS-VALIDATION-FAILED",
                details={"reason": "memory_limit_gb must be positive"},
            )

        if self.use_cache and self.cache_dir is None:
            self.cache_dir = Path.home() / ".cache" / "invarlock_metrics"
            self.cache_dir.mkdir(parents=True, exist_ok=True)


class ResourceManager:
    """Manages computational resources and memory usage."""

    def __init__(self, config: MetricsConfig):
        self.config = config
        self.device = self._determine_device()
        self.memory_info = self._get_memory_info()

    def _determine_device(self) -> torch.device:
        if self.config.force_cpu:
            return torch.device("cpu")
        if self.config.device is not None:
            return self.config.device
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _get_memory_info(self) -> dict[str, float]:
        info: dict[str, float] = {}
        vm = psutil.virtual_memory()
        info["system_total_gb"] = vm.total / (1024**3)
        info["system_available_gb"] = vm.available / (1024**3)

        if self.device.type == "cuda":
            total_memory = torch.cuda.get_device_properties(0).total_memory
            info["gpu_total_gb"] = total_memory / (1024**3)
            info["gpu_free_gb"] = (total_memory - torch.cuda.memory_allocated()) / (
                1024**3
            )

        return info

    def estimate_memory_usage(
        self, model: nn.Module, batch_size: int, seq_length: int
    ) -> float:
        param_memory = sum(p.numel() * p.element_size() for p in model.parameters()) / (
            1024**3
        )

        if hasattr(model, "config"):
            hidden_size = getattr(
                model.config, "n_embd", getattr(model.config, "hidden_size", 768)
            )
            num_layers = getattr(
                model.config, "n_layer", getattr(model.config, "num_hidden_layers", 12)
            )
            activation_memory = (
                batch_size * seq_length * hidden_size * num_layers * 4
            ) / (1024**3)
        else:
            activation_memory = param_memory * 2

        return param_memory + activation_memory

    def should_use_cpu_fallback(self, estimated_memory_gb: float) -> bool:
        if self.device.type == "cpu":
            return False

        available_memory = self.memory_info.get(
            "gpu_free_gb", self.memory_info.get("system_available_gb", 8.0)
        )
        return estimated_memory_gb > (
            available_memory - self.config.cpu_fallback_threshold_gb
        )

    def cleanup(self) -> None:
        if self.config.cleanup_after:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()


class DependencyManager:
    """Manages optional dependencies with graceful degradation."""

    def __init__(self):
        self.available_modules: dict[str, Any] = {}
        self.missing_modules: list[tuple[str, str]] = []
        self._check_dependencies()

    def _check_dependencies(self) -> None:
        try:
            from .lens2_mi import mi_scores

            self.available_modules["mi_scores"] = mi_scores
            logger.info("✓ lens2_mi module available")
        except ImportError as e:
            self.missing_modules.append(("lens2_mi", str(e)))
            logger.warning("✗ lens2_mi module not available - MI-Gini will be NaN")

        try:
            from .lens3 import scan_model_gains

            self.available_modules["scan_model_gains"] = scan_model_gains
            logger.info("✓ lens3 module available")
        except ImportError as e:
            self.missing_modules.append(("lens3", str(e)))
            logger.warning("✗ lens3 module not available - σ_max will be NaN")

    def get_module(self, name: str) -> Any:
        if name in self.available_modules:
            return self.available_modules[name]
        raise DependencyError(
            code="E203",
            message=f"DEPENDENCY-MISSING: module {name} is not available",
            details={"module": name},
        )

    def is_available(self, name: str) -> bool:
        return name in self.available_modules

    def get_missing_dependencies(self) -> list[tuple[str, str]]:
        return self.missing_modules.copy()


class InputValidator:
    """Validates inputs for metrics calculation."""

    @staticmethod
    def validate_model(model: nn.Module, config: MetricsConfig) -> None:
        if not isinstance(model, nn.Module):
            raise ValidationError(
                code="E402",
                message="METRICS-VALIDATION-FAILED",
                details={"reason": f"Expected nn.Module, got {type(model)}"},
            )

        try:
            param_count = sum(1 for _ in model.parameters())
        except Exception as exc:
            raise ValidationError(
                code="E402",
                message="METRICS-VALIDATION-FAILED",
                details={
                    "reason": "Model parameter iteration failed",
                    "error": str(exc),
                },
            ) from exc

        if param_count == 0:
            if config.strict_validation:
                raise ValidationError(
                    code="E402",
                    message="METRICS-VALIDATION-FAILED",
                    details={"reason": "Model has no parameters"},
                )
            logger.warning("Model has no parameters")

    @staticmethod
    def validate_dataloader(dataloader, config: MetricsConfig) -> None:
        if dataloader is None:
            raise ValidationError(
                code="E402",
                message="METRICS-VALIDATION-FAILED",
                details={"reason": "Dataloader cannot be None"},
            )

        try:
            first_batch = next(iter(dataloader))
            if not first_batch:
                if not config.allow_empty_data:
                    raise ValidationError(
                        code="E402",
                        message="METRICS-VALIDATION-FAILED",
                        details={"reason": "Dataloader is empty"},
                    )
                logger.warning("Dataloader is empty")
        except StopIteration as e:
            if not config.allow_empty_data:
                raise ValidationError(
                    code="E402",
                    message="METRICS-VALIDATION-FAILED",
                    details={"reason": "Dataloader is empty"},
                ) from e
            logger.warning("Dataloader is empty")

    @staticmethod
    def validate_tensor(
        tensor: torch.Tensor, name: str, config: MetricsConfig
    ) -> torch.Tensor:
        if not isinstance(tensor, torch.Tensor):
            raise ValidationError(
                code="E402",
                message="METRICS-VALIDATION-FAILED",
                details={"reason": f"{name} must be a tensor, got {type(tensor)}"},
            )

        if torch.isnan(tensor).any():
            if config.strict_validation:
                raise ValidationError(
                    code="E402",
                    message="METRICS-VALIDATION-FAILED",
                    details={"reason": f"{name} contains NaN values"},
                )
            logger.warning(
                f"{name} contains NaN values, replacing with {config.nan_replacement}"
            )
            tensor = torch.nan_to_num(tensor, nan=config.nan_replacement)

        if torch.isinf(tensor).any():
            if config.strict_validation:
                raise ValidationError(
                    code="E402",
                    message="METRICS-VALIDATION-FAILED",
                    details={"reason": f"{name} contains Inf values"},
                )
            logger.warning(
                f"{name} contains Inf values, replacing with ±{config.inf_replacement}"
            )
            tensor = torch.nan_to_num(
                tensor,
                posinf=config.inf_replacement,
                neginf=-config.inf_replacement,
            )

        return tensor


__all__ = [
    "DependencyError",
    "DependencyManager",
    "InputValidator",
    "MetricsConfig",
    "MetricsProgressObserver",
    "MetricsProgressPhase",
    "MetricsProgressUpdate",
    "ResourceError",
    "ResourceManager",
]
