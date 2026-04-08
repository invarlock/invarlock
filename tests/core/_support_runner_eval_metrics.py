from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any


class _FakeRunner:
    def __init__(self) -> None:
        self.events: list[tuple[str, str, str, dict[str, Any]]] = []

    def _resolve_policy_flags(self, _config: Any) -> dict[str, bool]:
        return {"allow_calibration_materialize": True}

    def _log_event(
        self,
        component: str,
        operation: str,
        level: str,
        data: dict[str, Any] | None = None,
    ) -> None:
        self.events.append((component, operation, level, data or {}))


class _FakeModel:
    def __init__(self) -> None:
        self._device = "cpu"
        self._param = SimpleNamespace(device="cpu")

    def eval(self) -> None:
        return None

    def parameters(self):
        yield self._param

    def to(self, device) -> _FakeModel:
        self._device = str(device)
        self._param = SimpleNamespace(device=device)
        return self


def _write_ppm(path: Path) -> None:
    path.write_text("P3\n1 1\n255\n255 0 0\n", encoding="utf-8")


def _summary(
    *,
    ppl: float,
    total_tokens: int,
    weighted_log_loss: float,
    num_batches: int,
    log_losses: list[float],
    window_ids: list[int],
    token_counts: list[int] | None = None,
    actual_token_counts: list[int] | None = None,
) -> dict[str, Any]:
    token_counts = token_counts or []
    actual_token_counts = actual_token_counts or token_counts or [1] * num_batches
    return {
        "ppl": ppl,
        "total_tokens": total_tokens,
        "actual_total_tokens": sum(actual_token_counts),
        "num_batches": num_batches,
        "log_losses": log_losses,
        "window_ids": window_ids,
        "tokens": [[wid + 1] for wid in window_ids],
        "attention_masks": [[1] for _ in window_ids],
        "weighted_log_loss": weighted_log_loss,
        "window_token_counts": token_counts,
        "masked_token_counts": token_counts or [1] * num_batches,
        "actual_token_counts": actual_token_counts,
        "labels": [[wid + 1] for wid in window_ids],
    }
