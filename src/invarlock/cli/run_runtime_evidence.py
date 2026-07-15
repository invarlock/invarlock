"""Best-effort runtime evidence capture for config-driven runs."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any


def capture_backend_inventory(
    *,
    adapter: Any,
    cfg: Any,
    model: Any,
    run_config: Any,
    extract_load_kwargs: Callable[..., dict[str, Any]],
    error_type: type[Exception],
    build_inventory: Callable[..., dict[str, Any] | None],
    filename: str,
) -> None:
    """Record the observed backend inventory in context and beside events."""

    try:
        load_kwargs = extract_load_kwargs(cfg, invarlock_error_cls=error_type)
    except (AttributeError, KeyError, TypeError, ValueError, error_type):
        load_kwargs = {}
    quantization_config = load_kwargs.get("quantization_config")
    inventory = build_inventory(
        adapter=str(getattr(adapter, "name", "") or ""),
        quantization_config=(
            quantization_config if isinstance(quantization_config, dict) else {}
        ),
        model=model,
        load_smoke=True,
        inference_smoke=False,
    )
    if inventory is None:
        return
    context = getattr(run_config, "context", None)
    if isinstance(context, dict):
        context["_backend_inventory"] = inventory
    event_path = getattr(run_config, "event_path", None)
    if event_path is None:
        return
    try:
        output_dir = Path(event_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / filename).write_text(
            json.dumps(inventory, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    except (OSError, TypeError, ValueError):
        return


def capture_runtime_quantization_proof(
    *,
    adapter: Any,
    model: Any,
    run_config: Any,
    build_proof: Callable[..., dict[str, Any] | None],
    write_sidecar: Callable[[Path, dict[str, Any]], Any],
) -> None:
    """Record live quantized runtime types without making normal runs fatal."""

    try:
        proof = build_proof(
            adapter=str(getattr(adapter, "name", "") or ""),
            model=model,
        )
    except Exception:  # noqa: BLE001 - optional proof capture must not block a run
        return
    if proof is None:
        return
    context = getattr(run_config, "context", None)
    if isinstance(context, dict):
        context["_runtime_quantization_proof"] = proof
    event_path = getattr(run_config, "event_path", None)
    if event_path is None:
        return
    try:
        write_sidecar(Path(event_path).parent, proof)
    except (OSError, TypeError, ValueError):
        return


__all__ = ["capture_backend_inventory", "capture_runtime_quantization_proof"]
