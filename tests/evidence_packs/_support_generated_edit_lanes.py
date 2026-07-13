from __future__ import annotations

import torch


class TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = torch.nn.Linear(4, 4, bias=False)
        self.mlp = torch.nn.Linear(4, 4, bias=False)


class LayeredTinyModel(torch.nn.Module):
    """Small model whose parameter names exercise layer-qualified edit scopes."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [
                torch.nn.ModuleDict(
                    {
                        "attn": torch.nn.Linear(5, 4, bias=False),
                        "mlp": torch.nn.Linear(5, 4, bias=False),
                    }
                )
                for _ in range(2)
            ]
        )
        self.output = torch.nn.Linear(5, 4, bias=False)
