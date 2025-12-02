"""Built-in no-op edit used for baseline and Compare & Certify (BYOE).

This edit does not modify the model and reports zero deltas. It exists to
support baseline runs and Compare & Certify certification where the subject
checkpoint is produced outside of InvarLock.
"""

from __future__ import annotations

from typing import Any

import torch.nn as nn

from invarlock.core.api import ModelAdapter, ModelEdit


class NoopEdit(ModelEdit):
    """A do-nothing edit that returns empty deltas."""

    name = "noop"

    def can_edit(self, model_desc: dict[str, Any]) -> bool:
        return True

    def preview(
        self, model: nn.Module, adapter: ModelAdapter, calib: Any
    ) -> dict[str, Any]:
        return {"name": self.name, "plan": {}}

    def apply(
        self, model: nn.Module, adapter: ModelAdapter, **kwargs: Any
    ) -> dict[str, Any]:
        return {
            "name": self.name,
            "plan_digest": "noop",
            "plan": {},
            "deltas": {
                "params_changed": 0,
                "sparsity": None,
                "bitwidth_map": None,
                "layers_modified": 0,
            },
            "config": {},
            "model_desc": adapter.describe(model)
            if hasattr(adapter, "describe")
            else {},
        }
