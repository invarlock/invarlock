from __future__ import annotations

import torch.nn as nn

from invarlock.edits.noop import NoopEdit


def test_noop_edit_reports_zero_deltas() -> None:
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2))

    class _Adapter:
        def describe(self, m):
            return {"layers": len(list(m.modules()))}

    edit = NoopEdit()
    plan = edit.preview(model, _Adapter(), None)
    assert plan["name"] == "noop" and plan["plan"] == {}

    result = edit.apply(model, _Adapter())
    assert result["name"] == "noop"
    assert result["deltas"]["params_changed"] == 0
    assert result["deltas"]["layers_modified"] == 0
