import torch.nn as nn

from invarlock.guards.variance import VarianceGuard


class Dummy(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Linear(2, 2, bias=False)


def test_refresh_after_edit_metrics_early_returns_and_logging():
    g = VarianceGuard()
    m = Dummy()

    # Not prepared → immediate return
    g._prepared = False
    g._refresh_after_edit_metrics(m)

    # Prepared and already evaluated post-edit → early return
    g._prepared = True
    g._post_edit_evaluated = True
    g._refresh_after_edit_metrics(m)

    # Prepared, not evaluated, but no calibration batches → logs warning
    g._post_edit_evaluated = False
    g._calibration_batches = []
    g._refresh_after_edit_metrics(m)
