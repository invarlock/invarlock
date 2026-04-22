from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_LOAD_ERRORS = (
    AttributeError,
    ImportError,
    ModuleNotFoundError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)


def _resolve_core_loader_strategy_fn():
    try:
        from invarlock.adapters.hf_loading import resolve_core_loader_strategy
    except ImportError:  # pragma: no cover - direct module load under pytest
        src_root = Path(__file__).resolve().parents[3] / "src"
        sys.path.insert(0, str(src_root))
        from invarlock.adapters.hf_loading import resolve_core_loader_strategy
    return resolve_core_loader_strategy


def load_causal_model(
    model_path: Path | str,
    *,
    trust_remote_code: bool,
    **load_kwargs: Any,
) -> tuple[Any, str]:
    resolve_core_loader_strategy = _resolve_core_loader_strategy_fn()
    loader_kwargs = dict(load_kwargs)
    loader_kwargs["trust_remote_code"] = trust_remote_code

    primary = resolve_core_loader_strategy(
        task="causal",
        model_id=str(model_path),
        kwargs={"trust_remote_code": trust_remote_code},
        allow_direct_submodule=True,
    )
    strategies = [primary]

    auto_strategy = (
        primary
        if primary.strategy == "auto"
        else resolve_core_loader_strategy(
            task="causal",
            model_id=str(model_path),
            kwargs={"trust_remote_code": trust_remote_code},
            allow_direct_submodule=False,
        )
    )
    direct_fallback = resolve_core_loader_strategy(
        task="causal",
        model_id=str(model_path),
        kwargs={},
        allow_direct_submodule=True,
    )

    if primary.strategy == "auto":
        if direct_fallback.strategy == "direct_submodule":
            strategies.append(direct_fallback)
    else:
        strategies.append(auto_strategy)

    last_error: Exception | None = None
    last_label = "unknown"
    for strategy in strategies:
        last_label = strategy.loader_label
        try:
            model = strategy.loader.from_pretrained(model_path, **loader_kwargs)
            return model, strategy.loader_label
        except _LOAD_ERRORS as exc:
            last_error = exc
            continue

    if last_error is not None:
        raise RuntimeError(
            f"Failed to load causal model via {last_label}: {last_error}"
        ) from last_error
    raise RuntimeError("Failed to resolve causal model loader strategy")
