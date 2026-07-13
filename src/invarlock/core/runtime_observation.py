"""Narrow, deterministic observations of a loaded model's runtime objects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RuntimeObservation:
    value: Any
    kind: str
    path: str
    fqcn: str


def runtime_type_fqcn(value: Any) -> str:
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def observe_model_runtime(model: Any) -> tuple[bool, tuple[RuntimeObservation, ...]]:
    """Observe each module and only its direct ``weight`` attribute.

    ``named_modules`` supplies stable model paths when available.  The fallback
    index is deterministic for models exposing only ``modules``.
    """

    try:
        named_modules = getattr(model, "named_modules", None)
        modules = getattr(model, "modules", None)
    except Exception:  # noqa: BLE001 - optional observation fails closed
        return False, ()
    try:
        if callable(named_modules):
            entries = tuple((str(path), module) for path, module in named_modules())
        elif callable(modules):
            entries = tuple(
                (str(index), module) for index, module in enumerate(modules())
            )
        else:
            return False, ()
        observations: list[RuntimeObservation] = []
        for path, module in entries:
            observations.append(
                RuntimeObservation(
                    value=module,
                    kind="module",
                    path=path,
                    fqcn=runtime_type_fqcn(module),
                )
            )
            try:
                weight = getattr(module, "weight", None)
            except Exception:  # noqa: BLE001 - optional observation fails closed
                continue
            if weight is not None:
                observations.append(
                    RuntimeObservation(
                        value=weight,
                        kind="direct_weight",
                        path=f"{path}.weight" if path else "weight",
                        fqcn=runtime_type_fqcn(weight),
                    )
                )
        return True, tuple(observations)
    except Exception:  # noqa: BLE001 - optional observation fails closed
        return False, ()


__all__ = ["RuntimeObservation", "observe_model_runtime", "runtime_type_fqcn"]
