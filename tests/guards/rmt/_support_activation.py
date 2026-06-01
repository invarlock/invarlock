from __future__ import annotations

import torch
import torch.nn as nn

from invarlock.guards import rmt_activation_runtime as runtime


def activation_batch(width: int = 2) -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.ones((1, width)),
        "attention_mask": torch.ones((1, width)),
    }


def activation_kwargs(
    *,
    estimator: dict[str, object] | None = None,
    deadband: float = 0.0,
    margin: float = 0.0,
) -> dict[str, object]:
    return {
        "allowed_suffixes": ("attn",),
        "activation_sampling": None,
        "estimator": estimator,
        "deadband": deadband,
        "margin": margin,
        "classify_family_fn": lambda name: "attn",
    }


class TinyActivationModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = nn.Linear(2, 2, bias=False)

    def forward(self, input_ids, attention_mask=None):  # noqa: ANN001
        _ = attention_mask
        return self.attn(input_ids.float())


class AdapterGenerationInputs:
    def prepare_generation_inputs(self, batch, device):  # noqa: ANN001
        _ = batch
        return {
            "input_ids": torch.ones((1, 2), device=device),
            "attention_mask": torch.ones((1, 2), device=device),
        }


class ActivationGuardStub:
    adapter = None
    margin = 1.0
    deadband = 0.0

    def __init__(
        self,
        *,
        adapter=None,  # noqa: ANN001
        outliers: tuple[int, float, float] | None = (1, 2.0, 3.0),
        raises: bool = False,
    ) -> None:
        self.adapter = adapter
        self._outliers = outliers
        self._raises = raises

    def _get_activation_modules(self, model):  # noqa: ANN001
        return runtime.get_activation_modules(model, allowed_suffixes=("attn",))

    def _activation_svd_outliers(self, output, *, margin, deadband):  # noqa: ANN001
        _ = output, margin, deadband
        if self._raises:
            raise RuntimeError("boom")
        assert self._outliers is not None
        return self._outliers

    def _prepare_activation_inputs(self, batch, device):  # noqa: ANN001
        return runtime.prepare_activation_inputs(batch, device)

    def _batch_token_weight(self, input_ids, attention_mask):  # noqa: ANN001
        return runtime.batch_token_weight(input_ids, attention_mask)
