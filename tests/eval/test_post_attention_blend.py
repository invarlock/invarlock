from __future__ import annotations

import pytest

from invarlock.eval.probes.post_attention import blend_neuron_scores

torch = pytest.importorskip("torch")


def test_blend_neuron_scores_with_padding_and_weights():
    a = torch.ones((2, 3))
    b = torch.zeros((1, 2))
    out = blend_neuron_scores([a, b], weights=[0.5, 0.5])
    # Output shape matches the first tensor
    assert tuple(out.shape) == (2, 3)
    # Blending: first contributes 0.5, second contributes zeros → all 0.5
    assert torch.allclose(out, torch.full((2, 3), 0.5))

    # Mismatched weights length raises
    with pytest.raises(ValueError):
        _ = blend_neuron_scores([a, b], weights=[1.0])
