import torch.nn as nn

from invarlock.guards.rmt import analyze_weight_distribution


def test_analyze_weight_distribution_smoke():
    model = nn.Sequential(nn.Linear(8, 8))
    stats = analyze_weight_distribution(model)
    assert isinstance(stats, dict)
    assert "histogram" in stats and "singular_values" in stats
