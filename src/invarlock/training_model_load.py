"""Fail-closed model-loading diagnostics for training evidence."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

TRAINING_MODEL_LOAD_DIAGNOSTICS_SCHEMA = "invarlock/training-model-load-diagnostics-v1"
_DIAGNOSTIC_FIELDS = frozenset(
    {"missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"}
)


class TrainingModelLoadError(RuntimeError):
    """Raised when a model load or its declared semantics cannot be trusted."""


def _string_entries(value: object, *, field: str) -> list[str]:
    if not isinstance(value, (list, tuple, set, frozenset)):
        raise TrainingModelLoadError(f"model loading diagnostics lack {field}")
    entries = list(value)
    if any(not isinstance(item, str) or not item for item in entries):
        raise TrainingModelLoadError(
            f"model loading diagnostics contain invalid {field}"
        )
    return sorted(set(entries))


def normalize_load_diagnostics(
    value: object,
    *,
    expected_unexpected_keys: Sequence[str],
    label: str,
) -> dict[str, object]:
    """Normalize one complete record and enforce its exact accepted migration."""

    if not isinstance(value, Mapping) or set(value) != _DIAGNOSTIC_FIELDS:
        raise TrainingModelLoadError(
            f"{label} loader did not return complete loading diagnostics"
        )
    normalized = {
        field: _string_entries(value.get(field), field=field)
        for field in sorted(_DIAGNOSTIC_FIELDS)
    }
    for field in ("missing_keys", "mismatched_keys", "error_msgs"):
        if normalized[field]:
            raise TrainingModelLoadError(f"{label} loading diagnostics report {field}")
    expected = sorted(set(expected_unexpected_keys))
    if list(expected_unexpected_keys) != expected:
        raise TrainingModelLoadError(
            f"{label} expected unexpected-key policy is not sorted and unique"
        )
    if normalized["unexpected_keys"] != expected:
        raise TrainingModelLoadError(
            f"{label} unexpected keys do not exactly match the immutable profile"
        )
    return {
        "schema": TRAINING_MODEL_LOAD_DIAGNOSTICS_SCHEMA,
        "policy": "exact_source_key_migration",
        **normalized,
    }


def load_model_with_diagnostics(
    auto_model: Any,
    source: object,
    *,
    load_options: Mapping[str, object],
    expected_unexpected_keys: Sequence[str],
    label: str,
) -> tuple[Any, dict[str, object]]:
    """Load exactly once and reject every undeclared checkpoint-key outcome."""

    if "output_loading_info" in load_options:
        raise TrainingModelLoadError(
            f"{label} load options must not override output_loading_info"
        )
    loaded = auto_model.from_pretrained(
        source,
        **dict(load_options),
        output_loading_info=True,
    )
    if not isinstance(loaded, tuple) or len(loaded) != 2 or loaded[0] is None:
        raise TrainingModelLoadError(
            f"{label} loader did not return model and loading diagnostics"
        )
    model, diagnostics = loaded
    return model, normalize_load_diagnostics(
        diagnostics,
        expected_unexpected_keys=expected_unexpected_keys,
        label=label,
    )


def load_diagnostics_sha256(value: Mapping[str, object]) -> str:
    """Hash the closed normalized diagnostics representation."""

    encoded = json.dumps(
        dict(value),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def configure_causal_lm_loss(model: Any, *, loss_function: str) -> None:
    """Select the exact labeled-forward loss named by an immutable profile."""

    if loss_function != "ForCausalLM":
        raise TrainingModelLoadError("training loss function must be ForCausalLM")
    if not hasattr(model, "loss_type"):
        raise TrainingModelLoadError(
            "training model does not expose an explicit loss_type selector"
        )
    model.loss_type = loss_function
    try:
        resolved = model.loss_function
    except (AttributeError, KeyError, RuntimeError, TypeError, ValueError) as exc:
        raise TrainingModelLoadError(
            "training model could not resolve the pinned loss function"
        ) from exc
    if not callable(resolved) or getattr(model, "loss_type", None) != loss_function:
        raise TrainingModelLoadError(
            "training model did not retain the pinned loss function"
        )


__all__ = [
    "TRAINING_MODEL_LOAD_DIAGNOSTICS_SCHEMA",
    "TrainingModelLoadError",
    "configure_causal_lm_loss",
    "load_diagnostics_sha256",
    "load_model_with_diagnostics",
    "normalize_load_diagnostics",
]
