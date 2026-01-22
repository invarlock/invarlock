from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from typing import Any


def fix_layer_drop_config(
    config: Any,
    *,
    total_layers: int,
    kept_layers: int,
    baseline_config: Mapping[str, Any] | None = None,
) -> None:
    if config is None:
        return

    if not isinstance(total_layers, int) or not isinstance(kept_layers, int):
        return
    if total_layers < 1 or kept_layers < 1 or kept_layers > total_layers:
        return

    for key in ("num_hidden_layers", "n_layer", "num_layers"):
        if hasattr(config, key):
            try:
                setattr(config, key, int(kept_layers))
            except Exception:
                continue

    # Some architectures (e.g., Qwen2) store per-layer config lists such as
    # `layer_types`. If we shrink the transformer stack, these lists must be
    # truncated to match the new `num_hidden_layers` or model loading fails.
    try:
        items = list(vars(config).items())
    except Exception:
        items = []

    for name, value in items:
        if "layer" not in name:
            continue
        if not isinstance(value, list):
            continue
        if len(value) != total_layers:
            continue
        try:
            setattr(config, name, value[:kept_layers])
        except Exception:
            continue

    if baseline_config is None:
        return

    # Some configs can lose optional attributes during save/load (custom
    # transformers configs + trust_remote_code). Preserve baseline settings
    # when they are present but become null on the mutated config.
    if (
        hasattr(config, "sliding_window")
        and getattr(config, "sliding_window", None) is None
    ):
        sliding_window = baseline_config.get("sliding_window")
        if isinstance(sliding_window, int) and sliding_window > 0:
            try:
                config.sliding_window = sliding_window
            except Exception:
                pass


def fix_layer_drop_config_json(
    config: MutableMapping[str, Any],
    *,
    total_layers: int,
    kept_layers: int,
    baseline_config: Mapping[str, Any] | None = None,
) -> None:
    if not isinstance(total_layers, int) or not isinstance(kept_layers, int):
        return
    if total_layers < 1 or kept_layers < 1 or kept_layers > total_layers:
        return

    for key in ("num_hidden_layers", "n_layer", "num_layers"):
        if key in config:
            try:
                config[key] = int(kept_layers)
            except Exception:
                pass

    for name, value in list(config.items()):
        if "layer" not in name:
            continue
        if not isinstance(value, list):
            continue
        if len(value) != total_layers:
            continue
        config[name] = value[:kept_layers]

    if baseline_config is None:
        return

    if config.get("sliding_window") is None:
        sliding_window = baseline_config.get("sliding_window")
        if isinstance(sliding_window, int) and sliding_window > 0:
            config["sliding_window"] = sliding_window
