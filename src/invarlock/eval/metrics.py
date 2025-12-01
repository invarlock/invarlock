"""
invarlock.metrics
=============

Enhanced diagnostic helpers used by the Phase-2 notebooks with improved
robustness, performance, and configurability.

Public entry point
------------------
    >>> from invarlock.metrics import calculate_lens_metrics_for_model, MetricsConfig
    >>> config = MetricsConfig(oracle_windows=32, max_tokens=512)
    >>> metrics = calculate_lens_metrics_for_model(model, dataloader, config=config)
"""

from __future__ import annotations

import gc
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import torch
import torch.nn as nn

from invarlock.core.error_utils import wrap_errors
from invarlock.core.exceptions import MetricsError, ValidationError

# ── Enhanced logging setup ─────────────────────────────────────────────────
logger = logging.getLogger(__name__)


try:  # Optional dependency: tqdm (progress bars)
    from tqdm.auto import tqdm as _tqdm
except Exception:  # pragma: no cover - exercised only when tqdm is absent

    class _TqdmShim:
        def __init__(self, iterable=None, total=None, **kwargs):
            self._iterable = iterable
            self.total = total

        def __iter__(self):
            if self._iterable is None:
                return iter(())
            return iter(self._iterable)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def update(self, n: int = 1) -> None:
            return None

    def _tqdm(iterable=None, *args, **kwargs):
        return _TqdmShim(iterable=iterable, **kwargs)


tqdm = _tqdm


class DependencyError(MetricsError):
    """Raised when required dependencies are missing."""

    pass


class ResourceError(MetricsError):
    """Raised when insufficient resources are available."""

    pass


## Note: Use ValidationError from invarlock.core.exceptions


def bootstrap_confidence_interval(
    samples: list[float] | np.ndarray,
    n_bootstrap: int = 500,
    alpha: float = 0.05,
    statistic: callable = np.mean,
    random_state: np.random.Generator | None = None,
) -> tuple[float, float]:
    """
    Compute a bootstrap confidence interval for a 1D sample.

    Args:
        samples: 1D iterable of numeric samples.
        n_bootstrap: Number of bootstrap resamples.
        alpha: Significance level (0 < alpha < 1).
        statistic: Statistic function to apply to each resample.
        random_state: Optional numpy random generator for reproducibility.

    Returns:
        (lower, upper) confidence bounds.

    Raises:
        ValidationError(E402): For invalid inputs (shape/empty/range).
        MetricsError(E401): For compute/statistic failures during bootstrap.
    """
    data = np.asarray(samples, dtype=float)
    if data.ndim != 1:
        raise ValidationError(
            code="E402",
            message="METRICS-VALIDATION-FAILED",
            details={"reason": "samples must be 1-dimensional"},
        )
    if data.size == 0:
        raise ValidationError(
            code="E402",
            message="METRICS-VALIDATION-FAILED",
            details={"reason": "samples cannot be empty"},
        )
    if not 0.0 < alpha < 1.0:
        raise ValidationError(
            code="E402",
            message="METRICS-VALIDATION-FAILED",
            details={"reason": "alpha must be between 0 and 1", "alpha": alpha},
        )
    if n_bootstrap <= 0:
        raise ValidationError(
            code="E402",
            message="METRICS-VALIDATION-FAILED",
            details={
                "reason": "n_bootstrap must be positive",
                "n_bootstrap": n_bootstrap,
            },
        )

    with wrap_errors(MetricsError, "E401", "METRICS-COMPUTE-FAILED"):
        rng = random_state or np.random.default_rng()
        stats = np.empty(n_bootstrap, dtype=float)
        for i in range(n_bootstrap):
            indices = rng.integers(0, data.size, size=data.size)
            stats[i] = statistic(data[indices])

        lower = float(np.percentile(stats, 100 * (alpha / 2)))
        upper = float(np.percentile(stats, 100 * (1 - alpha / 2)))
        return lower, upper


@dataclass
class MetricsConfig:
    """Configuration for metrics calculation with sensible defaults."""

    # Core parameters
    oracle_windows: int = 16
    max_tokens: int = 256
    max_samples_per_layer: int = 25_000

    # Memory management
    auto_batch_size: bool = True
    memory_limit_gb: float | None = None
    cpu_fallback_threshold_gb: float = 0.5

    # Performance options
    use_cache: bool = True
    cache_dir: Path | None = None
    progress_bars: bool = True

    # Numerical stability
    clip_value: float = 1e3
    nan_replacement: float = 0.0
    inf_replacement: float = 1e4

    # Device management
    device: torch.device | None = None
    force_cpu: bool = False
    cleanup_after: bool = True

    # Validation options
    strict_validation: bool = True
    allow_empty_data: bool = False

    # Lens-specific parameters
    sigma_max_margin: float = 0.98
    mi_gini_subsample_ratio: float = 0.05
    head_energy_layers_filter: bool = True

    def __post_init__(self):
        """Validate configuration after initialization."""
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

        # Set default cache directory
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
        """Determine the best device to use."""
        if self.config.force_cpu:
            return torch.device("cpu")

        if self.config.device is not None:
            return self.config.device

        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    def _get_memory_info(self) -> dict[str, float]:
        """Get current memory information."""
        info = {}

        # System memory
        vm = psutil.virtual_memory()
        info["system_total_gb"] = vm.total / (1024**3)
        info["system_available_gb"] = vm.available / (1024**3)

        # GPU memory
        if self.device.type == "cuda":
            info["gpu_total_gb"] = torch.cuda.get_device_properties(0).total_memory / (
                1024**3
            )
            info["gpu_free_gb"] = (
                torch.cuda.get_device_properties(0).total_memory
                - torch.cuda.memory_allocated()
            ) / (1024**3)

        return info

    def estimate_memory_usage(
        self, model: nn.Module, batch_size: int, seq_length: int
    ) -> float:
        """Estimate memory usage in GB for given parameters."""
        # Model parameters
        param_memory = sum(p.numel() * p.element_size() for p in model.parameters()) / (
            1024**3
        )

        # Activation memory (rough estimate)
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
            activation_memory = param_memory * 2  # Conservative estimate

        return param_memory + activation_memory

    def should_use_cpu_fallback(self, estimated_memory_gb: float) -> bool:
        """Determine if CPU fallback should be used."""
        if self.device.type == "cpu":
            return False

        available_memory = self.memory_info.get(
            "gpu_free_gb", self.memory_info.get("system_available_gb", 8.0)
        )

        return estimated_memory_gb > (
            available_memory - self.config.cpu_fallback_threshold_gb
        )

    def cleanup(self):
        """Clean up GPU memory."""
        if self.config.cleanup_after:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()


# ── Enhanced dependency management ─────────────────────────────────────────
class DependencyManager:
    """Manages optional dependencies with graceful degradation."""

    def __init__(self):
        self.available_modules: dict[str, Any] = {}
        self.missing_modules: list[tuple[str, str]] = []
        self._check_dependencies()

    def _check_dependencies(self):
        """Check availability of optional dependencies."""
        # Check lens2_mi
        try:
            from .lens2_mi import mi_scores

            self.available_modules["mi_scores"] = mi_scores
            logger.info("✓ lens2_mi module available")
        except ImportError as e:
            self.missing_modules.append(("lens2_mi", str(e)))
            logger.warning("✗ lens2_mi module not available - MI-Gini will be NaN")

        # Check lens3
        try:
            from .lens3 import scan_model_gains

            self.available_modules["scan_model_gains"] = scan_model_gains
            logger.info("✓ lens3 module available")
        except ImportError as e:
            self.missing_modules.append(("lens3", str(e)))
            logger.warning("✗ lens3 module not available - σ_max will be NaN")

    def get_module(self, name: str):
        """Get a module if available, otherwise raise DependencyError."""
        if name in self.available_modules:
            return self.available_modules[name]
        raise DependencyError(
            code="E203",
            message=f"DEPENDENCY-MISSING: module {name} is not available",
            details={"module": name},
        )

    def is_available(self, name: str) -> bool:
        """Check if a module is available."""
        return name in self.available_modules

    def get_missing_dependencies(self) -> list[tuple[str, str]]:
        """Get list of missing dependencies with error messages."""
        return self.missing_modules.copy()


# ── Input validation ───────────────────────────────────────────────────────
class InputValidator:
    """Validates inputs for metrics calculation."""

    @staticmethod
    def validate_model(model: nn.Module, config: MetricsConfig) -> None:
        """Validate model input."""
        if not isinstance(model, nn.Module):
            raise ValidationError(
                code="E402",
                message="METRICS-VALIDATION-FAILED",
                details={"reason": f"Expected nn.Module, got {type(model)}"},
            )

        # Check if model has parameters
        try:
            param_count = sum(1 for _ in model.parameters())
            if param_count == 0:
                if config.strict_validation:
                    raise ValidationError(
                        code="E402",
                        message="METRICS-VALIDATION-FAILED",
                        details={"reason": "Model has no parameters"},
                    )
                else:
                    logger.warning("Model has no parameters")
        except Exception as e:
            logger.debug(f"Could not count model parameters: {e}")

    @staticmethod
    def validate_dataloader(dataloader, config: MetricsConfig) -> None:
        """Validate dataloader input."""
        if dataloader is None:
            raise ValidationError(
                code="E402",
                message="METRICS-VALIDATION-FAILED",
                details={"reason": "Dataloader cannot be None"},
            )

        # Check if dataloader has data
        try:
            first_batch = next(iter(dataloader))
            if not first_batch:
                if not config.allow_empty_data:
                    raise ValidationError(
                        code="E402",
                        message="METRICS-VALIDATION-FAILED",
                        details={"reason": "Dataloader is empty"},
                    )
                else:
                    logger.warning("Dataloader is empty")
        except StopIteration as e:
            if not config.allow_empty_data:
                raise ValidationError(
                    code="E402",
                    message="METRICS-VALIDATION-FAILED",
                    details={"reason": "Dataloader is empty"},
                ) from e
            else:
                logger.warning("Dataloader is empty")

    @staticmethod
    def validate_tensor(
        tensor: torch.Tensor, name: str, config: MetricsConfig
    ) -> torch.Tensor:
        """Validate and sanitize tensor."""
        if not isinstance(tensor, torch.Tensor):
            raise ValidationError(
                code="E402",
                message="METRICS-VALIDATION-FAILED",
                details={"reason": f"{name} must be a tensor, got {type(tensor)}"},
            )

        # Check for NaN/Inf
        if torch.isnan(tensor).any():
            if config.strict_validation:
                raise ValidationError(
                    code="E402",
                    message="METRICS-VALIDATION-FAILED",
                    details={"reason": f"{name} contains NaN values"},
                )
            else:
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
            else:
                logger.warning(
                    f"{name} contains Inf values, replacing with ±{config.inf_replacement}"
                )
                tensor = torch.nan_to_num(
                    tensor,
                    posinf=config.inf_replacement,
                    neginf=-config.inf_replacement,
                )

        return tensor


# ── Enhanced helper functions ──────────────────────────────────────────────
def _gini_vectorized(vec: torch.Tensor) -> float:
    """Optimized Gini coefficient calculation."""
    flat = vec.flatten().abs().float()
    if flat.numel() == 0 or torch.sum(flat) == 0:
        return float("nan")

    # Use more efficient sorting and cumsum
    sorted_vals = torch.sort(flat)[0]
    n = sorted_vals.numel()

    # Vectorized Gini calculation
    indices = torch.arange(1, n + 1, dtype=torch.float32, device=flat.device)
    gini = (2 * torch.sum(indices * sorted_vals) / torch.sum(sorted_vals) - (n + 1)) / n

    return gini.item()


def _mi_gini_optimized_cpu_path(
    feats_cpu: torch.Tensor,
    targ_cpu: torch.Tensor,
    max_per_layer: int,
    config: MetricsConfig,
) -> float:
    """Optimized MI Gini calculation on CPU with better memory management."""
    L, N, _ = feats_cpu.shape

    # Subsample if dataset is too large
    if N > max_per_layer:
        sel = torch.randperm(N)[:max_per_layer]
        feats_cpu = feats_cpu[:, sel, :]
        targ_cpu = targ_cpu[sel]

    # Get MI function
    dep_manager = DependencyManager()
    if not dep_manager.is_available("mi_scores"):
        return float("nan")

    mi_scores_fn = dep_manager.get_module("mi_scores")

    # Process in chunks to manage memory
    chunk_size = min(8, L)  # Process 8 layers at a time
    mi_scores_all = []

    progress_desc = "MI-Gini (CPU optimized)"
    with tqdm(
        total=L, desc=progress_desc, disable=not config.progress_bars, leave=False
    ) as pbar:
        for i in range(0, L, chunk_size):
            end_idx = min(i + chunk_size, L)
            chunk_feats = feats_cpu[i:end_idx]

            # Vectorized processing for the chunk
            chunk_scores = []
            for j in range(chunk_feats.shape[0]):
                try:
                    score = mi_scores_fn(chunk_feats[j], targ_cpu)
                    chunk_scores.append(score)
                except Exception as e:
                    logger.warning(f"MI calculation failed for layer {i + j}: {e}")
                    chunk_scores.append(torch.zeros_like(chunk_feats[j, 0, :]))

            mi_scores_all.extend(chunk_scores)
            pbar.update(end_idx - i)

    if not mi_scores_all:
        return float("nan")

    try:
        mi_mat = torch.stack(mi_scores_all)
        return _gini_vectorized(mi_mat)
    except Exception as e:
        logger.warning(f"Failed to stack MI scores: {e}")
        return float("nan")


def _locate_transformer_blocks_enhanced(model: nn.Module) -> list[nn.Module] | None:
    """Enhanced transformer block detection with better model support."""

    # Standard GPT2 patterns - safer approach
    def safe_getattr_chain(obj, *attrs):
        """Safely get nested attributes."""
        for attr in attrs:
            if obj is None:
                return None
            obj = getattr(obj, attr, None)
        return obj

    patterns = [
        lambda m: safe_getattr_chain(m, "transformer", "h"),
        lambda m: safe_getattr_chain(m, "h"),  # Bare GPT2Model
        lambda m: safe_getattr_chain(m, "base_model", "h"),  # Common wrappers
        lambda m: safe_getattr_chain(m, "model", "h"),  # Some wrappers
        lambda m: safe_getattr_chain(m, "transformer", "layers"),  # Alternative naming
    ]

    for pattern in patterns:
        try:
            blocks = pattern(model)
            if blocks is not None and hasattr(blocks, "__len__") and len(blocks) > 0:
                logger.debug(f"Found {len(blocks)} transformer blocks using pattern")
                return list(blocks)
        except (AttributeError, TypeError):
            continue

    # Fallback: search for transformer-like modules
    transformer_modules = []
    for name, module in model.named_modules():
        if any(attr in name.lower() for attr in ["block", "layer", "transformer"]):
            if hasattr(module, "attn") and hasattr(module, "mlp"):
                transformer_modules.append(module)

    if transformer_modules:
        logger.debug(
            f"Found {len(transformer_modules)} transformer blocks via fallback search"
        )
        return transformer_modules

    logger.warning("Could not locate transformer blocks in model")
    return None


# ── Result caching ─────────────────────────────────────────────────────────
class ResultCache:
    """Simple result caching for expensive operations."""

    def __init__(self, config: MetricsConfig):
        self.config = config
        self.cache: dict[str, dict[str, float]] = {}
        self.enabled = config.use_cache

    def _get_cache_key(
        self, model: nn.Module, dataloader, config: MetricsConfig
    ) -> str:
        """Generate cache key for model and data."""
        # Simple hash based on model parameters and config
        model_hash = hash(tuple(p.data_ptr() for p in model.parameters()))
        config_hash = hash(
            (config.oracle_windows, config.max_tokens, config.max_samples_per_layer)
        )
        return f"{model_hash}_{config_hash}"

    def get(self, key: str) -> dict[str, float] | None:
        """Get cached result."""
        if not self.enabled:
            return None
        return self.cache.get(key)

    def set(self, key: str, result: dict[str, float]) -> None:
        """Cache result."""
        if self.enabled:
            self.cache[key] = result.copy()

    def clear(self) -> None:
        """Clear cache."""
        self.cache.clear()


# ── Main metrics calculation function ──────────────────────────────────────
@torch.no_grad()
def calculate_lens_metrics_for_model(
    model: nn.Module,
    dataloader,
    *,
    config: MetricsConfig | None = None,
    oracle_windows: int | None = None,  # Backward compatibility
    device: torch.device | None = None,  # Backward compatibility
) -> dict[str, float]:
    """
    Calculate comprehensive lens metrics for a model with enhanced robustness.

    Args:
        model: The neural network model to analyze
        dataloader: DataLoader providing input data
        config: MetricsConfig object with all parameters
        oracle_windows: (deprecated) Number of windows to process
        device: (deprecated) Device to use for computation

    Returns:
        Dictionary containing calculated metrics

    Raises:
        MetricsError: If calculation fails due to various reasons
    """
    # Handle backward compatibility
    if config is None:
        config = MetricsConfig()
        if oracle_windows is not None:
            config.oracle_windows = oracle_windows
        if device is not None:
            config.device = device

    # Initialize managers
    dep_manager = DependencyManager()
    resource_manager = ResourceManager(config)
    validator = InputValidator()
    cache = ResultCache(config)

    # Validate inputs
    validator.validate_model(model, config)
    validator.validate_dataloader(dataloader, config)

    # Check cache
    cache_key = cache._get_cache_key(model, dataloader, config)
    cached_result = cache.get(cache_key)
    if cached_result is not None:
        logger.info("Using cached metrics result")
        return cached_result

    start_time = time.time()
    logger.info(
        f"Starting metrics calculation with config: oracle_windows={config.oracle_windows}, "
        f"max_tokens={config.max_tokens}, device={resource_manager.device}"
    )

    # Pre-evaluation checks
    try:
        _perform_pre_eval_checks(model, dataloader, resource_manager.device, config)
    except Exception as e:
        logger.warning(f"Pre-evaluation checks failed: {e}")

    # Unwrap common wrappers if present
    if hasattr(model, "base_model"):
        try:
            model = model.base_model
        except Exception:
            pass

    model.eval()
    device = resource_manager.device

    # Initialize results
    results = {
        "sigma_max": float("nan"),
        "head_energy": float("nan"),
        "mi_gini": float("nan"),
    }

    skipped_metrics: list[str] = []

    try:
        # Collect activations with progress tracking
        logger.info("Collecting model activations...")
        activation_data = _collect_activations(model, dataloader, config, device)

        if not activation_data["hidden_states"]:
            logger.warning("No activations collected - returning default values")
            return _finalize_results(
                results, skipped_metrics, cache, cache_key, start_time
            )

        # Calculate each metric
        results["sigma_max"] = _calculate_sigma_max(
            model, activation_data["first_batch"], dep_manager, config, device
        )

        results["head_energy"] = _calculate_head_energy(
            activation_data["hidden_states"], config
        )

        results["mi_gini"] = _calculate_mi_gini(
            model, activation_data, dep_manager, config, device
        )

    except Exception as e:
        logger.error(f"Metrics calculation failed: {e}")
        if config.strict_validation:
            raise MetricsError(f"Metrics calculation failed: {e}") from e

    finally:
        resource_manager.cleanup()

    return _finalize_results(results, skipped_metrics, cache, cache_key, start_time)


def _perform_pre_eval_checks(
    model: nn.Module, dataloader, device: torch.device, config: MetricsConfig
) -> None:
    """Perform pre-evaluation sanity checks."""
    # Check model context length vs data
    try:
        tok_len_attr = getattr(model.config, "n_positions", None) or getattr(
            model.config, "max_position_embeddings", None
        )
        if tok_len_attr:
            sample_batch = next(iter(dataloader))
            sample_ids = sample_batch["input_ids"]
            if sample_ids.shape[1] > tok_len_attr:
                logger.warning(
                    f"Input sequence length {sample_ids.shape[1]} exceeds "
                    f"model limit {tok_len_attr}"
                )
    except Exception as e:
        logger.debug(f"Context length check failed: {e}")

    # Dry run forward pass
    try:
        dry_batch = next(iter(dataloader))
        model_input = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in dry_batch.items()
        }
        _ = model(**model_input)
        logger.debug("Pre-evaluation dry run successful")
    except Exception as e:
        logger.warning(f"Pre-evaluation dry run failed: {e}")


def _collect_activations(
    model: nn.Module, dataloader, config: MetricsConfig, device: torch.device
) -> dict[str, Any]:
    """Collect model activations with enhanced error handling."""
    hidden_states_list = []
    fc1_activations_list = []
    targets_list = []
    first_batch = None

    # Progress tracking
    total_batches = (
        min(config.oracle_windows, len(dataloader))
        if hasattr(dataloader, "__len__")
        else config.oracle_windows
    )

    with tqdm(
        total=total_batches,
        desc="Collecting activations",
        disable=not config.progress_bars,
    ) as pbar:
        for i, batch in enumerate(dataloader):
            if i >= config.oracle_windows:
                break

            try:
                # Store first batch for later use
                if first_batch is None:
                    first_batch = {
                        k: v.to(device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()
                    }

                # Move batch to device
                input_ids = batch["input_ids"].to(device)

                # Limit sequence length
                if input_ids.shape[1] > config.max_tokens:
                    input_ids = input_ids[:, : config.max_tokens]

                # Forward pass with hidden states
                output = model(input_ids, output_hidden_states=True)

                # Collect hidden states (exclude first and last)
                if hasattr(output, "hidden_states") and len(output.hidden_states) > 2:
                    hidden_states = torch.stack(output.hidden_states[1:-1])
                    hidden_states = validator.validate_tensor(
                        hidden_states, f"hidden_states_batch_{i}", config
                    )
                    hidden_states_list.append(hidden_states)

                # Collect FC1 activations for MI-Gini
                fc1_acts = _extract_fc1_activations(model, output, config)
                if fc1_acts is not None:
                    fc1_activations_list.append(fc1_acts)
                    targets_list.append(
                        input_ids[:, 1:]
                    )  # Shifted for next-token prediction

                pbar.update(1)

            except Exception as e:
                logger.warning(f"Failed to process batch {i}: {e}")
                continue

    return {
        "hidden_states": hidden_states_list,
        "fc1_activations": fc1_activations_list,
        "targets": targets_list,
        "first_batch": first_batch,
    }


def _extract_fc1_activations(
    model: nn.Module, output, config: MetricsConfig
) -> torch.Tensor | None:
    """Extract FC1 activations for MI-Gini calculation."""
    blocks = _locate_transformer_blocks_enhanced(model)
    if blocks is None:
        return None

    try:
        valid_activations = []
        for idx, block in enumerate(blocks):
            if hasattr(block, "mlp") and hasattr(block.mlp, "c_fc"):
                try:
                    # Get hidden state for this layer
                    if (
                        hasattr(output, "hidden_states")
                        and len(output.hidden_states) > idx + 1
                    ):
                        hidden_state = output.hidden_states[idx + 1]
                        activation = block.mlp.c_fc(hidden_state)
                        activation = validator.validate_tensor(
                            activation, f"fc1_activation_{idx}", config
                        )
                        valid_activations.append(activation)
                except Exception as e:
                    logger.debug(
                        f"Failed to extract FC1 activation for block {idx}: {e}"
                    )
                    continue

        if valid_activations:
            # Check for consistent shapes
            shapes = [act.shape for act in valid_activations]
            if len(set(shapes)) > 1:
                logger.warning(f"Inconsistent FC1 activation shapes: {set(shapes)}")
                # Use most common shape
                from collections import Counter

                most_common_shape = Counter(shapes).most_common(1)[0][0]
                valid_activations = [
                    act for act in valid_activations if act.shape == most_common_shape
                ]

            return torch.stack(valid_activations)

    except Exception as e:
        logger.warning(f"FC1 activation extraction failed: {e}")

    return None


def _calculate_sigma_max(
    model: nn.Module,
    first_batch: dict | None,
    dep_manager: DependencyManager,
    config: MetricsConfig,
    device: torch.device,
) -> float:
    """Calculate sigma_max metric via Lens-3."""
    if not dep_manager.is_available("scan_model_gains"):
        logger.info("Skipping σ_max: scan_model_gains not available")
        return float("nan")

    if first_batch is None:
        logger.info("Skipping σ_max: no data batch available")
        return float("nan")

    try:
        scan_model_gains = dep_manager.get_module("scan_model_gains")
        gains_df = scan_model_gains(model, first_batch)

        if gains_df is None:
            logger.warning("scan_model_gains returned None")
            return float("nan")

        # Filter out embedding and head layers if possible
        if hasattr(gains_df, "columns") and "name" in gains_df.columns:
            mask = ~gains_df["name"].str.contains(
                "embed|lm_head", case=False, regex=True
            )
            filtered_gains = gains_df[mask]
        else:
            logger.info("Could not filter layers by name for σ_max")
            filtered_gains = gains_df

        if len(filtered_gains) == 0:
            logger.warning("No valid layers found for σ_max computation")
            return float("nan")

        # Extract gains
        gains_values = getattr(
            filtered_gains, "gain", getattr(filtered_gains, "values", [])
        )
        gains_tensor = torch.as_tensor(gains_values, dtype=torch.float32, device=device)

        if gains_tensor.numel() == 0:
            logger.warning("No gain values found")
            return float("nan")

        # Validate and get max
        gains_tensor = validator.validate_tensor(
            gains_tensor, "sigma_max_gains", config
        )
        finite_mask = torch.isfinite(gains_tensor)

        if not finite_mask.any():
            logger.warning("All σ_max gains are NaN/Inf")
            return float("nan")

        sigma_max = torch.max(gains_tensor[finite_mask]).item()
        logger.debug(f"Calculated σ_max: {sigma_max:.4f}")
        return sigma_max

    except Exception as e:
        logger.warning(f"σ_max calculation failed: {e}")
        return float("nan")


def _calculate_head_energy(
    hidden_states_list: list[torch.Tensor], config: MetricsConfig
) -> float:
    """Calculate head energy metric (mean squared activation per layer)."""
    if not hidden_states_list:
        logger.info("Skipping head energy: no hidden states available")
        return float("nan")

    try:
        # Concatenate all hidden states: [L, N, T, D]
        hidden_stack = torch.cat(hidden_states_list, dim=1)

        # Crop to max_tokens
        hidden_crop = hidden_stack[:, :, : config.max_tokens, :]

        # Sanitize
        hidden_crop = validator.validate_tensor(
            hidden_crop, "head_energy_hidden_states", config
        )

        # Calculate mean squared activation per layer
        squared_activations = hidden_crop.float().pow(2).mean(dim=-1)  # [L, N, T]
        per_layer_energy = squared_activations.mean(dim=(1, 2))  # [L]

        # Filter finite values
        finite_mask = torch.isfinite(per_layer_energy)
        if not finite_mask.any():
            logger.warning("All head energies are NaN/Inf")
            return float("nan")

        head_energy = per_layer_energy[finite_mask].mean().item()
        logger.debug(f"Calculated head energy: {head_energy:.6f}")
        return head_energy

    except Exception as e:
        logger.warning(f"Head energy calculation failed: {e}")
        return float("nan")


def _calculate_mi_gini(
    model: nn.Module,
    activation_data: dict[str, Any],
    dep_manager: DependencyManager,
    config: MetricsConfig,
    device: torch.device,
) -> float:
    """Calculate MI-based Gini coefficient."""
    if not dep_manager.is_available("mi_scores"):
        logger.info("Skipping MI-Gini: mi_scores not available")
        return float("nan")

    if not activation_data["fc1_activations"] or not activation_data["targets"]:
        logger.info("Skipping MI-Gini: no FC1 activations available")
        return float("nan")

    try:
        # Concatenate activations and targets
        fc1_all = torch.cat(activation_data["fc1_activations"], dim=1)  # [L, N, T, D]
        targ_all = torch.cat(activation_data["targets"], dim=0)  # [N, T]

        # Trim to align dimensions (remove last token from activations)
        fc1_trim = fc1_all[:, :, :-1, :]  # [L, N, T-1, D]

        # Crop to max_tokens
        fc1_trim = fc1_trim[:, :, : config.max_tokens, :]
        targ_trim = targ_all[:, : config.max_tokens]

        # Reshape for MI calculation
        L, N, T, D = fc1_trim.shape
        fc1_flat = fc1_trim.permute(0, 2, 1, 3).reshape(L, -1, D)  # [L, N*T, D]
        targ_flat = targ_trim.flatten()  # [N*T]

        # Validate tensors
        fc1_flat = InputValidator.validate_tensor(fc1_flat, "mi_gini_features", config)
        targ_flat = InputValidator.validate_tensor(targ_flat, "mi_gini_targets", config)

        # Get MI scores function
        mi_scores_fn = dep_manager.get_module("mi_scores")

        # Try GPU calculation first
        try:
            logger.debug("Attempting MI-Gini calculation on GPU")
            mi_scores_result = mi_scores_fn(fc1_flat, targ_flat)
            mi_gini = _gini_vectorized(mi_scores_result)
            logger.debug(f"Calculated MI-Gini (GPU): {mi_gini:.6f}")
            return mi_gini

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning("GPU OOM for MI-Gini, falling back to CPU")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # CPU fallback with subsampling
                mi_gini = _mi_gini_optimized_cpu_path(
                    fc1_flat.cpu().float(),
                    targ_flat.cpu(),
                    config.max_samples_per_layer,
                    config,
                )
                logger.debug(f"Calculated MI-Gini (CPU): {mi_gini:.6f}")
                return mi_gini
            else:
                raise

    except Exception as e:
        logger.warning(f"MI-Gini calculation failed: {e}")
        return float("nan")


def _finalize_results(
    results: dict[str, Any],
    skipped_metrics: list[str],
    cache: ResultCache,
    cache_key: str,
    start_time: float,
) -> dict[str, float]:
    """Finalize and validate results."""
    # Ensure all values are finite or NaN
    for key, value in results.items():
        if not isinstance(value, int | float):
            logger.warning(
                f"Metric {key} has invalid type {type(value)}, setting to NaN"
            )
            results[key] = float("nan")
        elif not (math.isnan(value) or math.isfinite(value)):
            logger.warning(f"Metric {key} is infinite, setting to NaN")
            results[key] = float("nan")

    # Log skipped metrics
    if skipped_metrics:
        logger.info(f"Skipped metrics: {', '.join(skipped_metrics)}")

    # Cache results
    cache.set(cache_key, results)

    # Log completion
    elapsed = time.time() - start_time
    logger.info(f"Metrics calculation completed in {elapsed:.2f}s: {results}")

    return results


# ── Backward compatibility functions ──────────────────────────────────────
def _gini(vec: torch.Tensor) -> float:
    """Legacy Gini function for backward compatibility."""
    return _gini_vectorized(vec)


def _mi_gini_cpu_safe_path(
    feats_cpu: torch.Tensor, targ_cpu: torch.Tensor, max_per_layer: int
) -> float:
    """Legacy CPU MI-Gini function for backward compatibility."""
    config = MetricsConfig(max_samples_per_layer=max_per_layer, progress_bars=True)
    return _mi_gini_optimized_cpu_path(feats_cpu, targ_cpu, max_per_layer, config)


def _locate_transformer_blocks(model: nn.Module) -> list[nn.Module] | None:
    """Legacy transformer block locator for backward compatibility."""
    return _locate_transformer_blocks_enhanced(model)


# ── Additional utility functions ───────────────────────────────────────────
def get_metrics_info() -> dict[str, Any]:
    """Get information about available metrics and dependencies."""
    dep_manager = DependencyManager()

    return {
        "available_metrics": ["sigma_max", "head_energy", "mi_gini"],
        "available_dependencies": list(dep_manager.available_modules.keys()),
        "missing_dependencies": dep_manager.get_missing_dependencies(),
        "default_config": MetricsConfig().__dict__,
    }


def validate_metrics_environment() -> bool:
    """Validate that the metrics environment is properly set up."""
    try:
        dep_manager = DependencyManager()
        MetricsConfig()

        # Check basic dependencies

        logger.info("✓ Basic dependencies available")

        # Check optional dependencies
        available_count = len(dep_manager.available_modules)
        total_count = available_count + len(dep_manager.missing_modules)

        logger.info(
            f"✓ {available_count}/{total_count} optional dependencies available"
        )

        if dep_manager.missing_modules:
            logger.warning("Some optional dependencies are missing:")
            for name, error in dep_manager.missing_modules:
                logger.warning(f"  - {name}: {error}")

        return True

    except Exception as e:
        logger.error(f"Environment validation failed: {e}")
        return False


# ── Import necessary modules for validation ────────────────────────────────
# Note: math is already imported at top of file

# Global validator instance for use in helper functions
validator = InputValidator()


# ── Perplexity validation ──────────────────────────────────────────────────
class PerplexityStatus:
    """Quality status levels for ppl-like primary metrics (perplexity)."""

    EXCELLENT = "excellent"  # < 50
    GOOD = "good"  # 50-100
    ACCEPTABLE = "acceptable"  # 100-200
    POOR = "poor"  # 200-500
    UNUSABLE = "unusable"  # > 500

    @classmethod
    def from_value(cls, ppl: float, vocab_size: int | None = None) -> str:
        """Get status from perplexity value."""
        if ppl < 50:
            return cls.EXCELLENT
        elif ppl < 100:
            return cls.GOOD
        elif ppl < 200:
            return cls.ACCEPTABLE
        elif ppl < 500:
            return cls.POOR
        else:
            return cls.UNUSABLE


def validate_perplexity(
    ppl: float,
    vocab_size: int | None = None,
    context: str = "evaluation",
    warn_threshold: float = 200.0,
    error_threshold: float = 2000.0,
    allow_high: bool = False,
) -> tuple[bool, str, str]:
    """
    Validate perplexity value and provide feedback.

    Args:
        ppl: Perplexity value to validate
        vocab_size: Vocabulary size for context-aware validation
        context: Context string for error messages
        warn_threshold: Threshold for warning (default 200)
        error_threshold: Threshold for error (default 2000)
        allow_high: Allow high perplexity values (for testing)

    Returns:
        Tuple of (is_valid, status, message)
    """
    # Check for invalid values
    if math.isnan(ppl) or math.isinf(ppl):
        return False, "invalid", f"Perplexity is {ppl}"

    if ppl < 1.0:
        return False, "invalid", f"Perplexity {ppl:.2f} is less than 1.0"

    # Get status
    status = PerplexityStatus.from_value(ppl, vocab_size)

    # Adjust thresholds based on vocab size if provided
    if vocab_size is not None:
        # For untrained models, ppl-like PM ≈ vocab_size is expected
        # Adjust thresholds accordingly
        warn_threshold = max(warn_threshold, vocab_size * 0.5)
        error_threshold = max(error_threshold, vocab_size * 2.0)

    # Generate message based on status
    if ppl > error_threshold and not allow_high:
        message = (
            f"Perplexity {ppl:.1f} exceeds error threshold {error_threshold:.0f} "
            f"in {context}. Model appears to be untrained or corrupted."
        )
        return False, status, message

    elif ppl > warn_threshold:
        message = (
            f"Perplexity {ppl:.1f} exceeds warning threshold {warn_threshold:.0f} "
            f"in {context}. Model may be severely degraded."
        )
        if not allow_high:
            logger.warning(message)
        return True, status, message

    elif status == PerplexityStatus.POOR:
        message = f"Perplexity {ppl:.1f} indicates poor model quality in {context}."
        logger.info(message)
        return True, status, message

    elif status == PerplexityStatus.ACCEPTABLE:
        message = f"Perplexity {ppl:.1f} is acceptable for {context}."
        return True, status, message

    else:
        message = f"Perplexity {ppl:.1f} is {status} for {context}."
        return True, status, message


# ── Helper function for robust forward pass ────────────────────────────────
def _forward_loss_causal(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    labels: torch.Tensor | None = None,
) -> tuple[float, torch.Tensor | None]:
    """
    Robust forward that handles HF ModelOutput or tuple, computes loss if needed.
    Returns (loss_value: float, logits: torch.Tensor or None).
    """
    import torch.nn.functional as F

    # 1) Prefer dict-style outputs
    try:
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )
        # If we got a ModelOutput, use it
        if hasattr(outputs, "loss") and outputs.loss is not None:
            return float(outputs.loss.detach().cpu()), getattr(outputs, "logits", None)
        logits = getattr(outputs, "logits", None)
    except (TypeError, AttributeError):
        # Some stub models/tests may not accept return_dict
        outputs = model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        if isinstance(outputs, tuple | list):
            # If labels were provided, many HF models put loss first, logits second
            if (
                labels is not None
                and len(outputs) >= 2
                and torch.is_tensor(outputs[0])
                and outputs[0].ndim == 0
            ):
                return float(outputs[0].detach().cpu()), outputs[1] if len(
                    outputs
                ) > 1 else None
            # Otherwise first is logits
            logits = outputs[0] if len(outputs) > 0 else None
        else:
            # Custom object: try attributes
            maybe_loss = getattr(outputs, "loss", None)
            maybe_logits = getattr(outputs, "logits", None)
            if maybe_loss is not None:
                return float(maybe_loss.detach().cpu()), maybe_logits
            logits = maybe_logits

    # 2) If we're here, we have logits but no loss → compute it manually
    if logits is None:
        raise MetricsError(
            code="E401",
            message="METRICS-COMPUTE-FAILED: model returned neither loss nor logits",
        )

    if labels is None:
        raise ValidationError(
            code="E402",
            message="METRICS-VALIDATION-FAILED",
            details={"reason": "labels are required to compute perplexity loss"},
        )

    # Causal LM shift
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="mean",
    )
    return float(loss.detach().cpu()), logits


def _resolve_eval_device(
    model: nn.Module, device: str | torch.device | None
) -> torch.device:
    """
    Resolve evaluation device with graceful MPS fallback.

    If MPS is requested but unavailable (common in CI or non‑MacOS builds),
    fall back to CPU instead of raising at tensor .to(device) calls.
    """
    if device is None:
        try:
            resolved = next(model.parameters()).device
        except StopIteration:
            resolved = torch.device("cpu")
    else:
        resolved = torch.device(device) if isinstance(device, str) else device

    # Handle MPS when backend is not actually usable
    try:
        if isinstance(resolved, torch.device) and resolved.type == "mps":
            mps_backend = getattr(torch.backends, "mps", None)
            is_available = bool(
                mps_backend is not None
                and hasattr(mps_backend, "is_available")
                and mps_backend.is_available()
            )
            if not is_available:
                logger.warning(
                    "Requested device 'mps' for metrics evaluation but MPS backend "
                    "is not available; falling back to CPU."
                )
                resolved = torch.device("cpu")
    except Exception:
        # On any introspection failure, be conservative and fall back to CPU
        resolved = torch.device("cpu")

    return resolved


# ── Perplexity calculation ─────────────────────────────────────────────────
@torch.no_grad()
def calculate_perplexity(
    model: nn.Module,
    dataloader,
    max_batches: int = 100,
    device: str | torch.device | None = None,
) -> float:
    """
    DEPRECATED: Use compute_perplexity for new code.
    This is an alias for backward compatibility with tests.
    """
    return compute_perplexity(model, dataloader, max_samples=max_batches, device=device)


@torch.no_grad()
def compute_perplexity_strict(
    model: nn.Module, dataloader, device: str | torch.device | None = None
) -> float:
    """
    Compute perplexity with strict token-level accounting.

    Args:
        model: Language model to evaluate
        dataloader: DataLoader providing input sequences
        device: Device to use for computation

    Returns:
        Perplexity value

    Raises:
        ValueError: If no valid tokens found for perplexity computation
    """
    device = _resolve_eval_device(model, device)

    model.eval()
    nll_sum = 0.0
    tok_count = 0

    for batch in dataloader:
        # Handle different batch formats
        if isinstance(batch, dict):
            input_ids = batch.get("input_ids", batch.get("inputs", None))
            labels = batch.get("labels", None)
            attention_mask = batch.get("attention_mask", None)
            token_type_ids = batch.get("token_type_ids", None)
        elif isinstance(batch, tuple | list):
            input_ids = batch[0] if len(batch) > 0 else None
            labels = batch[1] if len(batch) > 1 else None
            attention_mask = batch[2] if len(batch) > 2 else None
            token_type_ids = batch[3] if len(batch) > 3 else None
        else:
            input_ids = batch
            labels = None
            attention_mask = None
            token_type_ids = None

        if input_ids is None or not isinstance(input_ids, torch.Tensor):
            continue

        input_ids = input_ids.to(device)
        attn = attention_mask.to(device) if attention_mask is not None else None
        token_type_ids_t = (
            token_type_ids.to(device) if token_type_ids is not None else None
        )

        # Default causal labels
        if labels is None:
            labels = input_ids.clone()
            if attn is not None:
                labels[attn == 0] = -100
        else:
            labels = labels.to(device)

        # Skip if sequence too short
        if input_ids.size(1) < 2:
            continue

        is_masked_lm = hasattr(model, "config") and getattr(
            model.config, "model_type", ""
        ) in {"bert", "roberta", "distilbert", "albert"}

        if is_masked_lm:
            masked_labels = labels.clone()
            if attn is not None:
                masked_labels = masked_labels.masked_fill(attn == 0, -100)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attn,
                token_type_ids=token_type_ids_t,
                labels=masked_labels,
                return_dict=True,
            )
            loss = outputs.loss
            if loss is None:
                continue
            valid_tokens = int((masked_labels != -100).sum().item())
            if valid_tokens == 0:
                continue
            nll_sum += float(loss.item()) * valid_tokens
            tok_count += valid_tokens
            continue

        # Forward (don't trust .loss, compute ourselves)
        try:
            outputs = model(input_ids=input_ids, attention_mask=attn, return_dict=True)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        except Exception:
            # Fallback for non-standard models
            outputs = model(input_ids=input_ids, attention_mask=attn)
            if isinstance(outputs, tuple | list):
                logits = outputs[0]
            else:
                logits = outputs.logits if hasattr(outputs, "logits") else outputs

        # Causal shift
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        shift_mask = attn[:, 1:] if attn is not None else None

        valid = shift_labels != -100
        if shift_mask is not None:
            valid = valid & shift_mask.bool()

        if not valid.any():
            continue

        log_probs = shift_logits.log_softmax(dim=-1)  # [B,T-1,V]
        tgt = shift_labels.clamp_min(0).unsqueeze(-1)  # [B,T-1,1]
        nll = -log_probs.gather(-1, tgt).squeeze(-1)  # [B,T-1]

        nll_sum += nll[valid].sum().item()
        tok_count += int(valid.sum().item())

    if tok_count == 0:
        raise ValidationError(
            code="E402",
            message="METRICS-VALIDATION-FAILED",
            details={
                "reason": "No valid tokens for perplexity (all masked or seq_len<=1)."
            },
        )

    return float(torch.exp(torch.tensor(nll_sum / tok_count)))


@torch.no_grad()
def compute_perplexity(
    model: nn.Module,
    dataloader,
    max_samples: int = 100,
    device: str | torch.device | None = None,
) -> float:
    """
    Compute perplexity of a language model on a dataset.

    ALWAYS uses strict token-level accounting to avoid padding issues.

    Args:
        model: Language model to evaluate
        dataloader: DataLoader providing input sequences
        max_samples: Maximum number of batches to evaluate
        device: Device to use for computation

    Returns:
        Perplexity value

    Raises:
        ValueError: If no valid tokens found
    """
    device = _resolve_eval_device(model, device)

    model.eval()
    nll_sum = 0.0
    tok_count = 0
    batch_count = 0

    for i, batch in enumerate(dataloader):
        # Check max_samples limit
        if max_samples is not None and i >= max_samples:
            break

        # Handle different batch formats
        if isinstance(batch, dict):
            input_ids = batch.get("input_ids", batch.get("inputs", None))
            labels = batch.get("labels", None)
            attention_mask = batch.get("attention_mask", None)
        elif isinstance(batch, tuple | list):
            input_ids = batch[0] if len(batch) > 0 else None
            labels = batch[1] if len(batch) > 1 else None
            attention_mask = batch[2] if len(batch) > 2 else None
        else:
            input_ids = batch
            labels = None
            attention_mask = None

        if input_ids is None or not isinstance(input_ids, torch.Tensor):
            continue

        input_ids = input_ids.to(device)
        attn = attention_mask.to(device) if attention_mask is not None else None

        # Default causal labels
        if labels is None:
            labels = input_ids.clone()
            if attn is not None:
                labels[attn == 0] = -100
        else:
            labels = labels.to(device)

        # Skip if sequence too short
        if input_ids.size(1) < 2:
            continue

        # Forward pass - get logits
        try:
            outputs = model(input_ids=input_ids, attention_mask=attn, return_dict=True)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        except Exception:
            # Fallback for non-standard models
            outputs = model(input_ids=input_ids, attention_mask=attn)
            if isinstance(outputs, tuple | list):
                logits = outputs[0]
            else:
                logits = outputs.logits if hasattr(outputs, "logits") else outputs

        # Causal shift for next-token prediction
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        shift_mask = attn[:, 1:] if attn is not None else None

        # Identify valid (non-padding) tokens
        valid = shift_labels != -100
        if shift_mask is not None:
            valid = valid & shift_mask.bool()

        if not valid.any():
            continue

        # Compute negative log-likelihood
        log_probs = shift_logits.log_softmax(dim=-1)  # [B,T-1,V]
        tgt = shift_labels.clamp_min(0).unsqueeze(-1)  # [B,T-1,1]

        # MPS workaround: gather operation can fail on MPS, use CPU fallback
        if str(device).startswith("mps"):
            log_probs_cpu = log_probs.cpu()
            tgt_cpu = tgt.cpu()
            nll_cpu = -log_probs_cpu.gather(-1, tgt_cpu).squeeze(-1)
            nll = nll_cpu.to(device)
        else:
            nll = -log_probs.gather(-1, tgt).squeeze(-1)  # [B,T-1]

        # Accumulate only for valid tokens
        nll_sum += nll[valid].sum().item()
        tok_count += int(valid.sum().item())
        batch_count += 1

    if tok_count == 0:
        raise ValidationError(
            code="E402",
            message="METRICS-VALIDATION-FAILED",
            details={
                "reason": (
                    f"No valid tokens for perplexity computation after {batch_count} batches. "
                    "All tokens were either padding or sequences were too short (<=1 token). "
                    "Ensure your data contains sequences of at least 2 tokens."
                )
            },
        )

    # Compute perplexity from average NLL
    avg_nll = nll_sum / tok_count
    ppl = float(math.exp(avg_nll))

    # Sanity check
    if ppl < 1.0:
        logger.warning(
            f"Computed perplexity {ppl:.2f} is less than 1.0, setting to 1.0"
        )
        ppl = 1.0
    elif not math.isfinite(ppl):
        logger.warning(f"Computed perplexity is not finite: {ppl}")
        ppl = float("inf")

    return ppl


# ── New Unified Evaluation Functions ──────────────────────────────────────


@torch.no_grad()
def compute_ppl(
    model: nn.Module,
    adapter: Any | None,
    window: Any,  # EvaluationWindow
    device: str | torch.device | None = None,
) -> float:
    """
    Compute perplexity for a specific evaluation window.

    This is the new unified evaluation function that works with EvaluationWindow objects
    from the data loading system.

    Args:
        model: Language model to evaluate
        adapter: Model adapter (unused currently, for future extensibility)
        window: EvaluationWindow with tokenized samples
        device: Device to use for computation

    Returns:
        Perplexity value for the window
    """
    device = _resolve_eval_device(model, device)

    model.eval()
    nll_sum = 0.0
    tok_count = 0

    # Process each sample in the window
    for input_ids, attention_mask in zip(
        window.input_ids, window.attention_masks, strict=False
    ):
        if not input_ids:
            continue

        # Convert to tensors
        input_ids_tensor = (
            torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)
        )
        attention_mask_tensor = (
            torch.tensor(attention_mask, dtype=torch.long).unsqueeze(0).to(device)
        )

        # Skip sequences that are too short
        if input_ids_tensor.size(1) < 2:
            continue

        # Forward pass
        try:
            outputs = model(
                input_ids=input_ids_tensor,
                attention_mask=attention_mask_tensor,
                return_dict=True,
            )
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        except Exception:
            # Fallback for non-standard models
            outputs = model(
                input_ids=input_ids_tensor, attention_mask=attention_mask_tensor
            )
            if isinstance(outputs, tuple | list):
                logits = outputs[0]
            else:
                logits = outputs.logits if hasattr(outputs, "logits") else outputs

        # Causal shift for next-token prediction
        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids_tensor[:, 1:]
        shift_mask = attention_mask_tensor[:, 1:]

        # Identify valid (non-padding) tokens
        valid = (shift_labels != -100) & shift_mask.bool()

        if not valid.any():
            continue

        # Compute negative log-likelihood
        log_probs = shift_logits.log_softmax(dim=-1)  # [B,T-1,V]
        tgt = shift_labels.clamp_min(0).unsqueeze(-1)  # [B,T-1,1]

        # Handle MPS device issues with gather
        if str(device).startswith("mps"):
            log_probs_cpu = log_probs.cpu()
            tgt_cpu = tgt.cpu()
            nll_cpu = -log_probs_cpu.gather(-1, tgt_cpu).squeeze(-1)
            nll = nll_cpu.to(device)
        else:
            nll = -log_probs.gather(-1, tgt).squeeze(-1)  # [B,T-1]

        # Accumulate only for valid tokens
        nll_sum += nll[valid].sum().item()
        tok_count += int(valid.sum().item())

    if tok_count == 0:
        raise ValidationError(
            code="E402",
            message="METRICS-VALIDATION-FAILED",
            details={
                "reason": "No valid tokens for perplexity computation in evaluation window",
            },
        )

    # Compute perplexity from average NLL
    avg_nll = nll_sum / tok_count
    ppl = float(math.exp(avg_nll))

    # Sanity check
    if ppl < 1.0:
        logger.warning(
            f"Computed perplexity {ppl:.2f} is less than 1.0, setting to 1.0"
        )
        ppl = 1.0
    elif not math.isfinite(ppl):
        logger.warning(f"Computed perplexity is not finite: {ppl}")
        ppl = float("inf")

    return ppl


def measure_latency(
    model: nn.Module,
    window: Any,  # EvaluationWindow
    device: str | torch.device | None = None,
    warmup_steps: int = 3,
    measurement_steps: int = 10,
) -> float:
    """
    Measure inference latency per token.

    Args:
        model: Model to measure
        window: EvaluationWindow with samples to use for measurement
        device: Device to use for measurement
        warmup_steps: Number of warmup iterations
        measurement_steps: Number of measurement iterations

    Returns:
        Average latency in milliseconds per token
    """
    if device is None:
        device = next(model.parameters()).device
    else:
        device = torch.device(device) if isinstance(device, str) else device

    model.eval()

    # Select a representative sample for timing
    if not window.input_ids:
        return 0.0

    # Use the first valid sample
    sample_input_ids = None
    sample_attention_mask = None

    for input_ids, attention_mask in zip(
        window.input_ids, window.attention_masks, strict=False
    ):
        if len(input_ids) > 10:  # Ensure reasonable length
            sample_input_ids = (
                torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)
            )
            sample_attention_mask = (
                torch.tensor(attention_mask, dtype=torch.long).unsqueeze(0).to(device)
            )
            break

    if sample_input_ids is None:
        return 0.0

    # Warmup
    with torch.no_grad():
        for _ in range(warmup_steps):
            try:
                _ = model(
                    input_ids=sample_input_ids, attention_mask=sample_attention_mask
                )
            except Exception:
                # If there are issues with the model, return 0
                return 0.0

    # Synchronize for accurate timing
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Measure latency
    start_time = time.time()

    with torch.no_grad():
        for _ in range(measurement_steps):
            _ = model(input_ids=sample_input_ids, attention_mask=sample_attention_mask)

    if device.type == "cuda":
        torch.cuda.synchronize()

    end_time = time.time()

    # Calculate per-token latency
    total_time_ms = (end_time - start_time) * 1000  # Convert to milliseconds
    total_tokens = int(sample_attention_mask.sum().item()) * measurement_steps

    if total_tokens == 0:
        return 0.0

    latency_ms_per_token = total_time_ms / total_tokens

    logger.debug(
        f"Measured latency: {latency_ms_per_token:.3f} ms/token over {measurement_steps} steps"
    )
    return latency_ms_per_token


def measure_memory(
    model: nn.Module,
    window: Any,  # EvaluationWindow
    device: str | torch.device | None = None,
) -> float:
    """
    Measure peak memory usage during inference.

    Args:
        model: Model to measure
        window: EvaluationWindow with samples to use for measurement
        device: Device to measure memory on

    Returns:
        Peak memory usage in MB
    """
    if device is None:
        device = next(model.parameters()).device
    else:
        device = torch.device(device) if isinstance(device, str) else device

    model.eval()

    # Get baseline memory
    if device.type == "cuda":
        torch.cuda.empty_cache()
        baseline_memory = torch.cuda.memory_allocated() / (1024 * 1024)
        torch.cuda.reset_peak_memory_stats()
    else:
        # For CPU/MPS, use psutil for system memory
        import psutil

        process = psutil.Process()
        baseline_memory = process.memory_info().rss / (1024 * 1024)

    # Run inference on a few samples to measure memory
    max_memory = baseline_memory

    with torch.no_grad():
        for i, (input_ids, attention_mask) in enumerate(
            zip(window.input_ids, window.attention_masks, strict=False)
        ):
            if i >= 5:  # Only measure on first 5 samples
                break

            if not input_ids:
                continue

            try:
                input_ids_tensor = (
                    torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)
                )
                attention_mask_tensor = (
                    torch.tensor(attention_mask, dtype=torch.long)
                    .unsqueeze(0)
                    .to(device)
                )

                _ = model(
                    input_ids=input_ids_tensor, attention_mask=attention_mask_tensor
                )

                # Measure memory after forward pass
                if device.type == "cuda":
                    current_memory = torch.cuda.memory_allocated() / (1024 * 1024)
                else:
                    current_memory = process.memory_info().rss / (1024 * 1024)

                max_memory = max(max_memory, current_memory)

            except Exception as e:
                logger.debug(f"Memory measurement failed for sample {i}: {e}")
                continue

    peak_memory_mb = max_memory
    logger.debug(f"Peak memory usage: {peak_memory_mb:.1f} MB")

    return peak_memory_mb


def compute_parameter_deltas(
    model_before: nn.Module, model_after: nn.Module, adapter: Any | None = None
) -> dict[str, Any]:
    """
    Compute precise parameter deltas between before and after models.

    Args:
        model_before: Model state before edit
        model_after: Model state after edit
        adapter: Model adapter for architecture-specific analysis

    Returns:
        Dictionary with parameter delta information:
        - params_changed: Number of parameters that were modified
        - layers_modified: Number of layers that were changed
        - sparsity: Overall sparsity ratio (if applicable)
    """
    deltas = {
        "params_changed": 0,
        "layers_modified": 0,
        "sparsity": None,
    }

    try:
        # Compare parameters
        before_params = dict(model_before.named_parameters())
        after_params = dict(model_after.named_parameters())

        modified_layers = set()
        total_changed = 0

        for name, before_param in before_params.items():
            if name not in after_params:
                continue

            after_param = after_params[name]

            # Check if parameter changed
            if not torch.equal(before_param.data, after_param.data):
                total_changed += before_param.numel()

                # Extract layer information from parameter name
                layer_match = None
                if ".h." in name or ".layers." in name:
                    # Extract layer number for transformer models
                    import re

                    match = re.search(r"\.(?:h|layers)\.(\d+)\.", name)
                    if match:
                        layer_match = int(match.group(1))
                        modified_layers.add(layer_match)

        deltas["params_changed"] = total_changed
        deltas["layers_modified"] = len(modified_layers)

        # Structural deltas (like head/neuron counts) are not tracked in this profile

        # Compute overall sparsity if applicable
        total_params_before = sum(p.numel() for p in model_before.parameters())
        total_params_after = sum(p.numel() for p in model_after.parameters())

        if total_params_after < total_params_before:
            deltas["sparsity"] = 1.0 - (total_params_after / total_params_before)

    except Exception as e:
        logger.warning(f"Parameter delta computation failed: {e}")

    return deltas


def analyze_spectral_changes(
    model_before: nn.Module, model_after: nn.Module, scope: str = "ffn"
) -> dict[str, Any]:
    """
    Analyze spectral norm changes between model states.

    Args:
        model_before: Model before edit
        model_after: Model after edit
        scope: Scope for spectral analysis ("ffn", "all")

    Returns:
        Dictionary with spectral analysis results
    """
    try:
        # Import spectral analysis if available
        from invarlock.guards.spectral import compute_spectral_norms

        before_norms = compute_spectral_norms(model_before, scope=scope)
        after_norms = compute_spectral_norms(model_after, scope=scope)

        # Compute changes
        changes = {}
        for layer_name in before_norms:
            if layer_name in after_norms:
                before_norm = before_norms[layer_name]
                after_norm = after_norms[layer_name]
                change_ratio = after_norm / before_norm if before_norm > 0 else 1.0
                changes[layer_name] = {
                    "before": before_norm,
                    "after": after_norm,
                    "ratio": change_ratio,
                }

        # Summary statistics
        ratios = [change["ratio"] for change in changes.values()]
        summary = {
            "layer_changes": changes,
            "mean_ratio": float(np.mean(ratios)) if ratios else 1.0,
            "max_ratio": float(np.max(ratios)) if ratios else 1.0,
            "min_ratio": float(np.min(ratios)) if ratios else 1.0,
            "layers_analyzed": len(changes),
        }

        return summary

    except ImportError:
        logger.debug("Spectral analysis not available")
        return {"error": "spectral_analysis_unavailable"}
    except Exception as e:
        logger.warning(f"Spectral analysis failed: {e}")
        return {"error": str(e)}


def analyze_rmt_changes(
    model_before: nn.Module, model_after: nn.Module
) -> dict[str, Any]:
    """
    Analyze RMT (Random Matrix Theory) changes between model states.

    Args:
        model_before: Model before edit
        model_after: Model after edit

    Returns:
        Dictionary with RMT analysis results
    """
    try:
        # Import RMT analysis if available
        from invarlock.guards.rmt import compute_mp_stats

        before_stats = compute_mp_stats(model_before)
        after_stats = compute_mp_stats(model_after)

        # Analyze changes in MP statistics
        changes = {}
        for layer_name in before_stats:
            if layer_name in after_stats:
                before_mp = before_stats[layer_name]
                after_mp = after_stats[layer_name]
                changes[layer_name] = {
                    "before": before_mp,
                    "after": after_mp,
                    "stable": abs(before_mp - after_mp) < 0.1,  # Stability threshold
                }

        # Count stable vs unstable layers
        stable_count = sum(
            1 for change in changes.values() if change.get("stable", False)
        )
        total_count = len(changes)

        summary = {
            "layer_changes": changes,
            "stable_layers": stable_count,
            "total_layers": total_count,
            "stability_ratio": stable_count / total_count if total_count > 0 else 0.0,
        }

        return summary

    except ImportError:
        logger.debug("RMT analysis not available")
        return {"error": "rmt_analysis_unavailable"}
    except Exception as e:
        logger.warning(f"RMT analysis failed: {e}")
        return {"error": str(e)}


# ── Integration with existing system ───────────────────────────────────────

# Update exports to include new functions (add to existing __all__ if it exists)
try:
    __all__.extend(
        [
            "bootstrap_confidence_interval",
            "compute_ppl",
            "measure_latency",
            "measure_memory",
            "compute_parameter_deltas",
            "analyze_spectral_changes",
            "analyze_rmt_changes",
        ]
    )
except NameError:
    # If __all__ doesn't exist, create it with the new functions
    __all__ = [
        "bootstrap_confidence_interval",
        "compute_ppl",
        "measure_latency",
        "measure_memory",
        "compute_parameter_deltas",
        "analyze_spectral_changes",
        "analyze_rmt_changes",
    ]
