from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any

LogEvent = Callable[..., None]


@dataclass(frozen=True)
class AdapterLayerModule:
    layer_index: int
    key: str
    module: Any


def _unwrap_model(model: Any) -> Any:
    unwrapped = model
    while hasattr(unwrapped, "module"):
        unwrapped = unwrapped.module
    return unwrapped


def _log(log_event: LogEvent | None, event: str, **details: Any) -> None:
    if log_event is not None:
        log_event(event, **details)


def adapter_layer_count(
    model: Any,
    adapter: Any | None,
    *,
    direct_layer_count: Callable[[], int] | None = None,
    log_event: LogEvent | None = None,
) -> int:
    if adapter is not None:
        describe = getattr(adapter, "describe", None)
        if callable(describe):
            try:
                description = describe(model)
                if isinstance(description, Mapping):
                    n_layer = int(description.get("n_layer", 0) or 0)
                    if n_layer > 0:
                        return n_layer
            except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
                _log(
                    log_event,
                    "adapter_describe_error",
                    level="DEBUG",
                    message=f"adapter.describe() failed: {exc}",
                )

    if direct_layer_count is not None:
        try:
            n_layer = int(direct_layer_count())
            if n_layer > 0:
                return n_layer
        except (AttributeError, RuntimeError, TypeError, ValueError):
            pass

    config = getattr(_unwrap_model(model), "config", None)
    if config is not None:
        try:
            return int(
                getattr(config, "n_layer", 0)
                or getattr(config, "num_hidden_layers", 0)
                or getattr(config, "num_layers", 0)
                or 0
            )
        except (AttributeError, RuntimeError, TypeError, ValueError):
            # guard-fallback-ok: malformed layer metadata disables adapter fallback.
            return 0
    return 0


def iter_adapter_layer_modules(
    model: Any,
    adapter: Any | None,
    *,
    direct_layer_count: Callable[[], int] | None = None,
    log_event: LogEvent | None = None,
    on_layer_error: Callable[[int, Exception], None] | None = None,
) -> Iterator[AdapterLayerModule]:
    if adapter is None:
        return
    get_layer_modules = getattr(adapter, "get_layer_modules", None)
    if not callable(get_layer_modules):
        return

    n_layer = adapter_layer_count(
        model,
        adapter,
        direct_layer_count=direct_layer_count,
        log_event=log_event,
    )
    if n_layer <= 0:
        _log(
            log_event,
            "adapter_fallback_no_layers",
            level="WARN",
            message="Adapter fallback: could not determine layer count",
        )
        return

    for index in range(n_layer):
        try:
            modules = get_layer_modules(model, index) or {}
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            if on_layer_error is not None:
                on_layer_error(index, exc)
            _log(
                log_event,
                "adapter_layer_modules_error",
                level="DEBUG",
                message=f"adapter.get_layer_modules() failed: {exc}",
                layer=index,
            )
            continue
        if isinstance(modules, Mapping):
            module_items = modules.items()
        elif hasattr(modules, "items"):
            module_items = modules.items()
        else:
            continue
        for key, module in module_items:
            if isinstance(key, str):
                yield AdapterLayerModule(index, key, module)


def iter_named_adapter_scoped_modules(
    model: Any,
    adapter: Any | None,
    *,
    should_include: Callable[[str, Any], bool],
    log_event: LogEvent | None = None,
) -> Iterator[tuple[str, Any]]:
    for item in iter_adapter_layer_modules(model, adapter, log_event=log_event):
        name = f"adapter.layers.{item.layer_index}.{item.key}"
        if should_include(name, item.module):
            yield name, item.module


__all__ = [
    "AdapterLayerModule",
    "adapter_layer_count",
    "iter_adapter_layer_modules",
    "iter_named_adapter_scoped_modules",
]
