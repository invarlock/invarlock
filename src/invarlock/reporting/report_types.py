"""
InvarLock Evaluation Report Types
============================

Canonical data structures for the unified evaluation harness.
This is the single source of truth for all evaluation results.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

try:  # Python 3.12+
    from typing import NotRequired, TypedDict
except ImportError:  # Legacy fallback
    from typing import NotRequired

    from typing_extensions import TypedDict


class AutoConfig(TypedDict):
    """Auto-tuning configuration metadata."""

    enabled: bool  # Whether auto-tuning was enabled
    tier: str  # Policy tier used ("conservative", "balanced", "aggressive")
    probes_used: int  # Number of micro-probes actually executed
    target_pm_ratio: (
        float | None
    )  # Target primary-metric ratio override (ppl-like kinds)


class MetaData(TypedDict):
    """Metadata about the model and execution environment."""

    model_id: str  # Model identifier (e.g., "gpt2", "path/to/model")
    adapter: str  # Adapter name (e.g., "hf_gpt2")
    commit: str  # Git commit SHA
    seed: int  # Random seed used for evaluation
    device: str  # Device used ("cpu", "cuda", "mps")
    ts: str  # ISO timestamp of evaluation
    auto: AutoConfig | None  # Auto-tuning configuration (if used)


class DataConfig(TypedDict):
    """Configuration of evaluation dataset and windowing."""

    dataset: str  # Dataset name (e.g., "wikitext2")
    split: str  # Dataset split ("validation", "test")
    seq_len: int  # Sequence length for tokenization
    stride: int  # Stride for window generation
    preview_n: int  # Number of preview samples (default: 100)
    final_n: int  # Number of final samples (default: 100)
    tokenizer_name: NotRequired[str]
    tokenizer_hash: NotRequired[str]
    vocab_size: NotRequired[int]
    bos_token: NotRequired[str | None]
    eos_token: NotRequired[str | None]
    pad_token: NotRequired[str | None]
    add_prefix_space: NotRequired[bool | None]
    dataset_hash: NotRequired[str]
    preview_hash: NotRequired[str]
    final_hash: NotRequired[str]
    preview_total_tokens: NotRequired[int]
    final_total_tokens: NotRequired[int]
    masked_tokens_total: NotRequired[int]
    masked_tokens_preview: NotRequired[int]
    masked_tokens_final: NotRequired[int]
    loss_type: NotRequired[str]


class EditInfo(TypedDict):
    """Information about the edit applied to the model."""

    name: str  # Edit type (e.g., "quant_rtn")
    plan_digest: str  # Hash of the edit plan for reproducibility
    deltas: EditDeltas  # Computed parameter changes


class EditDeltas(TypedDict):
    """Precise parameter deltas from the edit."""

    params_changed: int  # Number of parameters modified
    sparsity: float | None  # Overall sparsity ratio (if applicable)
    bitwidth_map: dict[str, Any] | None  # Bitwidth changes (if applicable)
    layers_modified: int  # Number of layers that were changed


class GuardReport(TypedDict):
    """Report from a single guard."""

    name: str  # Guard identifier
    policy: dict[str, Any]  # Policy parameters used
    metrics: dict[str, float]  # Computed metrics
    actions: list[str]  # Actions taken by the guard
    violations: list[str]  # Policy violations detected


class EvalMetrics(TypedDict, total=False):
    """Core evaluation metrics (primary metric canonical)."""

    # Canonical primary metric snapshot
    primary_metric: dict[str, Any]

    # Optional legacy/aux fields retained for guard telemetry and debug
    latency_ms_per_tok: float  # Average latency per token in milliseconds
    memory_mb_peak: float  # Peak memory usage in MB
    spectral: dict[str, Any]  # Spectral norm summaries
    rmt: dict[str, Any]  # RMT statistics
    invariants: dict[str, Any]  # Model invariant check results
    logloss_preview: float
    logloss_final: float
    logloss_delta: float
    logloss_preview_ci: tuple[float, float]
    logloss_final_ci: tuple[float, float]
    logloss_delta_ci: tuple[float, float]
    eval_samples: int
    total_tokens: int
    preview_total_tokens: int
    final_total_tokens: int
    window_overlap_fraction: float
    window_match_fraction: float
    window_pairing_reason: str | None
    window_pairing_preview: dict[str, Any]
    window_pairing_final: dict[str, Any]
    paired_windows: int
    paired_delta_summary: dict[str, Any]
    bootstrap: dict[str, Any]
    reduction: NotRequired[dict[str, Any]]
    moe: NotRequired[dict[str, Any]]


class Artifacts(TypedDict):
    """Paths to generated artifacts."""

    events_path: str  # Path to event log file
    logs_path: str  # Path to detailed logs
    checkpoint_path: str | None  # Path to model checkpoint (if saved)


class Flags(TypedDict):
    """Boolean flags about the evaluation."""

    guard_recovered: bool  # Whether any guard triggered recovery
    rollback_reason: str | None  # Reason for rollback (if any)


class RunReport(TypedDict):
    """
    Canonical report structure for InvarLock evaluation results.

    This is the single source of truth for all evaluation outputs.
    No other report formats should exist in the codebase.
    """

    meta: MetaData  # Model and environment metadata
    data: DataConfig  # Dataset configuration used
    edit: EditInfo  # Edit information and deltas
    guards: list[GuardReport]  # Reports from all guards
    metrics: EvalMetrics  # Core evaluation metrics
    artifacts: Artifacts  # Generated file paths
    flags: Flags  # Status flags
    evaluation_windows: NotRequired[dict[str, Any]]
    # Optional extras kept for richer downstream processing
    guard_overhead: NotRequired[dict[str, Any]]
    provenance: NotRequired[dict[str, Any]]


# Utility functions for creating reports
def create_empty_report() -> RunReport:
    """Create an empty RunReport with default values."""
    return RunReport(
        meta=MetaData(
            model_id="",
            adapter="",
            commit="",
            seed=42,
            device="cpu",
            ts=datetime.now().isoformat(),
            auto=None,
        ),
        data=DataConfig(
            dataset="", split="", seq_len=0, stride=0, preview_n=100, final_n=100
        ),
        edit=EditInfo(
            name="",
            plan_digest="",
            deltas=EditDeltas(
                params_changed=0,
                sparsity=None,
                bitwidth_map=None,
                layers_modified=0,
            ),
        ),
        guards=[],
        metrics=EvalMetrics(
            primary_metric={},
            latency_ms_per_tok=0.0,
            memory_mb_peak=0.0,
            spectral={},
            rmt={},
            invariants={},
        ),
        artifacts=Artifacts(events_path="", logs_path="", checkpoint_path=None),
        flags=Flags(guard_recovered=False, rollback_reason=None),
    )


def validate_report(report: RunReport) -> bool:
    """
    Validate that a RunReport has all required fields.

    Args:
        report: RunReport to validate

    Returns:
        True if valid, False otherwise
    """
    try:
        # Check that all top-level keys exist
        required_keys = {
            "meta",
            "data",
            "edit",
            "guards",
            "metrics",
            "artifacts",
            "flags",
        }
        if not all(key in report for key in required_keys):
            return False

        # Basic type checks
        guards = report.get("guards")
        if not isinstance(guards, list):
            return False

        metrics = report.get("metrics", {})
        # Canonical: require a primary_metric dict (kind/finals are checked downstream)
        pm = metrics.get("primary_metric") if isinstance(metrics, dict) else None
        if isinstance(pm, dict) and pm:
            pm_kind = pm.get("kind")
            pm_final = pm.get("final")
            # kind must be a non-empty string; if 'final' is present it must be numeric
            if not (isinstance(pm_kind, str) and pm_kind):
                return False
            if pm_final is not None and not isinstance(pm_final, int | float):
                return False
        else:
            # PM-only: legacy ppl_* acceptance removed
            return False

        meta = report.get("meta", {})
        seed = meta.get("seed")
        if not isinstance(seed, int):
            return False

        return True

    except (KeyError, TypeError):
        return False


# Export all types for type hints
__all__ = [
    "RunReport",
    "MetaData",
    "AutoConfig",
    "DataConfig",
    "EditInfo",
    "EditDeltas",
    "GuardReport",
    "EvalMetrics",
    "Artifacts",
    "Flags",
    "create_empty_report",
    "validate_report",
]
