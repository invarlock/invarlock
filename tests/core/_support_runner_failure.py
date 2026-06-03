from __future__ import annotations

from typing import Any

from invarlock.core.api import ModelAdapter


class DummyAdapter(ModelAdapter):
    name = "dummy"

    def can_handle(self, model: Any) -> bool:  # pragma: no cover - not used here
        return True

    def describe(self, model: Any) -> dict[str, Any]:  # pragma: no cover - minimal
        return {"n_layer": 1, "heads_per_layer": [1], "mlp_dims": [3], "tying": {}}

    def snapshot(self, model: Any) -> bytes:  # pragma: no cover - minimal stub
        return b"s"

    def restore(self, model: Any, blob: bytes) -> None:  # pragma: no cover - stub
        return None


class EditStub:
    def __init__(self, name: str = "e", result: dict[str, Any] | None = None):
        self.name = name
        self._result = result or {"name": name, "deltas": {}}

    def can_edit(self, model_desc: dict[str, Any]) -> bool:
        return True

    def apply(
        self,
        model: Any,
        adapter: ModelAdapter,
        plan=None,
        runtime=None,
    ) -> dict[str, Any]:
        _ = model, adapter, plan, runtime
        return dict(self._result)


def _toy_model_with_losses(losses):
    import torch

    class Toy(torch.nn.Module):
        def __init__(self, seq):
            super().__init__()
            self.seq = list(seq)
            self.idx = 0
            self.lin = torch.nn.Linear(3, 3, bias=False)

        def forward(self, *args, **kwargs):
            class Out:
                def __init__(self, val: float):
                    self.loss = type("L", (), {"item": lambda self: float(val)})()

            val = self.seq[self.idx % len(self.seq)]
            self.idx += 1
            return Out(val)

    return Toy(losses)


def _minimal_calibration(n: int) -> list[dict[str, Any]]:
    return [
        {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]} for _ in range(max(1, n))
    ]
