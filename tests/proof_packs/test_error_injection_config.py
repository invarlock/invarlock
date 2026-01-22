from __future__ import annotations

from scripts.proof_packs.python.error_injection_config import (
    fix_layer_drop_config,
    fix_layer_drop_config_json,
)


class DummyConfig:
    def __init__(self) -> None:
        self.num_hidden_layers = 48
        self.n_layer = 48
        self.layer_types = ["full_attention"] * 48
        self.not_layer_list = [1, 2, 3, 4]
        self.sliding_window = None


def test_fix_layer_drop_truncates_layer_lists_and_updates_counts() -> None:
    cfg = DummyConfig()
    fix_layer_drop_config(cfg, total_layers=48, kept_layers=47, baseline_config={})
    assert cfg.num_hidden_layers == 47
    assert cfg.n_layer == 47
    assert len(cfg.layer_types) == 47
    assert cfg.not_layer_list == [1, 2, 3, 4]


def test_fix_layer_drop_preserves_baseline_sliding_window() -> None:
    cfg = DummyConfig()
    fix_layer_drop_config(
        cfg, total_layers=48, kept_layers=47, baseline_config={"sliding_window": 131072}
    )
    assert cfg.sliding_window == 131072


def test_fix_layer_drop_json_truncates_layer_lists_and_preserves_sliding_window() -> (
    None
):
    cfg = {
        "num_hidden_layers": 47,
        "layer_types": ["full_attention"] * 48,
        "sliding_window": None,
    }
    fix_layer_drop_config_json(
        cfg,
        total_layers=48,
        kept_layers=47,
        baseline_config={"sliding_window": 131072},
    )
    assert cfg["num_hidden_layers"] == 47
    assert len(cfg["layer_types"]) == 47
    assert cfg["sliding_window"] == 131072
