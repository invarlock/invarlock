from __future__ import annotations

import pytest

from invarlock.cli.config import (
    AutoConfig,
    DatasetConfig,
    EvalBootstrapConfig,
)
from invarlock.cli.config import (
    _deep_merge as deep_merge,
)


def test_dataset_config_stride_must_be_leq_seq_len():
    with pytest.raises(ValueError):
        DatasetConfig(seq_len=8, stride=16)


def test_eval_bootstrap_config_validation():
    with pytest.raises(ValueError):
        EvalBootstrapConfig(replicates=0)
    with pytest.raises(ValueError):
        EvalBootstrapConfig(replicates=10, alpha=0.0)


def test_auto_config_validation():
    with pytest.raises(ValueError):
        AutoConfig(probes=11)
    with pytest.raises(ValueError):
        AutoConfig(probes=0, target_pm_ratio=0.9)


def test_deep_merge_nested_overrides():
    a = {"x": {"y": 1, "z": 2}, "k": 3}
    b = {"x": {"y": 9}, "new": 4}
    out = deep_merge(a, b)
    assert out == {"x": {"y": 9, "z": 2}, "k": 3, "new": 4}
