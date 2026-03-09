from __future__ import annotations

import random

import numpy as np

from invarlock import model_utils


def test_set_seed_reseeds_python_and_numpy() -> None:
    model_utils.set_seed(123)
    first = (random.random(), float(np.random.rand()))

    model_utils.set_seed(123)
    second = (random.random(), float(np.random.rand()))

    assert first == second


def test_model_utils_exports_only_live_helper() -> None:
    assert model_utils.__all__ == ["set_seed"]
    for name in [
        "get_device",
        "time_block",
        "json_save",
        "json_load",
        "dump_df",
        "deterministic",
        "extract_input_ids",
    ]:
        assert not hasattr(model_utils, name)
