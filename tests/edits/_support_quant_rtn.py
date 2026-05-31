from __future__ import annotations

import torch

from invarlock.edits.quant_rtn import TargetModule


def target(name: str, module: torch.nn.Module) -> TargetModule:
    return TargetModule(
        name=name,
        module=module,
        selection_reason="test",
        matched_pattern="test",
        parameter_id=id(module.weight),
        module_type=f"{module.__class__.__module__}.{module.__class__.__name__}",
    )
