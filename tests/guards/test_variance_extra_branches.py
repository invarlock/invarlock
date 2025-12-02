import torch

from invarlock.guards.variance import VarianceGuard


class TensorOutModel(torch.nn.Module):
    def __init__(self, out_shape_matches_labels: bool = True):
        super().__init__()
        self.p = torch.nn.Parameter(torch.zeros(1))
        self._matches = out_shape_matches_labels

    def forward(self, inputs, labels=None):
        if self._matches:
            # Return a tensor with the same shape as labels to hit MSE path
            return labels.float() if labels is not None else inputs.float()
        # Return 3D logits-like tensor to hit mean-of-squares fallback
        B, T = inputs.shape
        return torch.zeros(B, T, 2, device=inputs.device)


def test_variance_ab_gate_invalid_ratio_ci():
    g = VarianceGuard()
    # Positive, valid PPLs, but invalid ratio_ci (contains 0)
    g.set_ab_results(ppl_no_ve=100.0, ppl_with_ve=90.0, ratio_ci=(0.0, 0.9))
    should_enable, reason = g._evaluate_ab_gate()
    assert should_enable is False and reason == "invalid_ratio_ci"


def test_variance_compute_ppl_for_batches_mse_and_fallback():
    device = torch.device("cpu")

    # Case 1: outputs is a Tensor with same shape as labels → MSE path
    model_mse = TensorOutModel(True).eval()
    g1 = VarianceGuard()
    batch = {
        "input_ids": torch.ones(1, 4, dtype=torch.long),
        "labels": torch.zeros(1, 4, dtype=torch.long),
    }
    ppl, loss = g1._compute_ppl_for_batches(model_mse, [batch], device)
    assert len(ppl) == len(loss) == 1

    # Case 2: outputs is a Tensor with different shape → mean of squares fallback
    model_mean = TensorOutModel(False).eval()
    g2 = VarianceGuard()
    ppl2, loss2 = g2._compute_ppl_for_batches(model_mean, [batch], device)
    assert len(ppl2) == len(loss2) == 1
