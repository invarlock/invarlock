"""SDK projection adapter for native sequential collection transactions."""

from __future__ import annotations

import importlib
import json
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

from invarlock.judge_measurement_types import JudgeMeasurementPlan, JudgeMeasurements

from .contracts import canonical_payload


class _NativeCollection:
    def __init__(self, coordinator: Any) -> None:
        self.coordinator = coordinator

    def sources(self) -> list[dict[str, Any]]:
        return cast(list[dict[str, Any]], json.loads(self.coordinator.sources_json()))

    def admit(self, index: int, request: bytes) -> dict[str, Any]:
        return cast(dict[str, Any], json.loads(self.coordinator.admit(index, request)))

    def complete(self, index: int, source: dict[str, Any]) -> None:
        self.coordinator.complete(index, canonical_payload(source))

    def finish(self) -> JudgeMeasurements:
        # Native finish validates the complete retained measurements contract.
        return cast(JudgeMeasurements, json.loads(self.coordinator.finish_json()))


@contextmanager
def sequential_collection(
    backend: str,
    *,
    directory: Path | None,
    identity: bytes,
    plan: JudgeMeasurementPlan,
    baseline_run: dict[str, Any],
    subject_run: dict[str, Any],
    error_type: type[Exception],
) -> Iterator[_NativeCollection | None]:
    if backend == "python":
        yield None
        return
    if backend != "rust":
        raise error_type("execution backend must be python or rust")
    temporary = None
    coordinator = None
    try:
        try:
            invarlock_native = importlib.import_module("invarlock_native")
        except ImportError as exc:
            raise error_type(
                "the explicit Rust execution backend requires invarlock_native"
            ) from exc
        if directory is None:
            temporary = tempfile.TemporaryDirectory(
                prefix="invarlock-native-collection-"
            )
            directory = Path(temporary.name).resolve(strict=True)
        coordinator = invarlock_native.SequentialCoordinator(
            str(directory.absolute()),
            identity,
            canonical_payload(plan),
            canonical_payload(baseline_run),
            canonical_payload(subject_run),
        )
        yield _NativeCollection(coordinator)
    except error_type:
        raise
    except ValueError as exc:
        raise error_type(str(exc)) from exc
    finally:
        if coordinator is not None:
            coordinator.close()
        if temporary is not None:
            temporary.cleanup()
