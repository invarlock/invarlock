import torch

from invarlock.guards import rmt as R


class TinyOutlier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(4, 4)
        with torch.no_grad():
            self.layer.weight.mul_(50.0)  # Inflate sigma to trigger outlier

    def named_modules(self, memo=None, prefix=""):
        yield "block.attn.c_proj", self.layer


def test_rmt_detect_verbose_outlier_prints():
    model = TinyOutlier()
    # Construct baseline-aware stats to exercise primary detection branch
    m, n = model.layer.weight.shape
    baseline_sigmas = {"block.attn.c_proj": 1.0}
    baseline_mp_stats = {
        "block.attn.c_proj": {
            "mp_bulk_edge_base": float(R.mp_bulk_edge(m, n, whitened=False))
        }
    }
    out = R.rmt_detect(
        model,
        threshold=1.5,
        verbose=True,
        baseline_sigmas=baseline_sigmas,
        baseline_mp_stats=baseline_mp_stats,
        deadband=0.0,
    )
    assert out["has_outliers"] is True
    assert out["n_layers_flagged"] >= 1
