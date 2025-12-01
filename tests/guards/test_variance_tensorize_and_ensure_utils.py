import numpy as np

from invarlock.guards.variance import VarianceGuard


def test_tensorize_and_ensure_tensor_value_paths():
    g = VarianceGuard()

    # _ensure_tensor_value: ndarray, list, tuple, scalar, and unknown type
    v_np = np.array([1, 2, 3], dtype=np.int64)
    out_np = g._ensure_tensor_value(v_np)
    assert hasattr(out_np, "shape")

    v_list = [1, 2, 3]
    out_list = g._ensure_tensor_value(v_list)
    assert hasattr(out_list, "shape")

    v_tuple = (1, 2)
    out_tuple = g._ensure_tensor_value(v_tuple)
    assert hasattr(out_tuple, "shape")

    v_scalar = 3.14
    out_scalar = g._ensure_tensor_value(v_scalar)
    assert hasattr(out_scalar, "shape")

    class Weird:
        pass

    v_weird = Weird()
    out_weird = g._ensure_tensor_value(v_weird)
    assert out_weird is v_weird

    # _tensorize_calibration_batches: dict/list/tuple fallbacks
    batches = [
        {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "meta": "x"},
        ([4, 5], [6, 7]),
        [8, 9, 10],
    ]
    tb = g._tensorize_calibration_batches(batches)
    assert isinstance(tb, list) and len(tb) == 3
    assert hasattr(tb[0]["input_ids"], "shape")
    assert hasattr(tb[1][0], "shape")
    assert hasattr(tb[2][0], "shape")
