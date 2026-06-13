import numpy as np
import torch

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
    string_list = ["a", "b", "c"]
    assert g._ensure_tensor_value(string_list) is string_list

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


def test_tensorize_calibration_batches_honors_max_seq_len() -> None:
    guard = VarianceGuard(policy={"calibration": {"max_seq_len": 4}})

    batches = [
        {
            "input_ids": torch.arange(10),
            "attention_mask": [1] * 10,
            "labels": torch.arange(10),
            "window_id": "w0",
            "meta": "kept",
        },
        (torch.arange(12).reshape(1, 12), torch.ones(1, 12, dtype=torch.long)),
    ]

    tensorized = guard._tensorize_calibration_batches(batches)

    assert tensorized[0]["input_ids"].tolist() == [0, 1, 2, 3]
    assert tensorized[0]["attention_mask"].tolist() == [1, 1, 1, 1]
    assert tensorized[0]["labels"].tolist() == [0, 1, 2, 3]
    assert tensorized[0]["window_id"] == "w0"
    assert tensorized[0]["meta"] == "kept"
    assert tuple(tensorized[1][0].shape) == (1, 4)
    assert tuple(tensorized[1][1].shape) == (1, 4)
    assert guard._stats["calibration"]["max_seq_len"] == 4
    assert guard._stats["calibration"]["max_observed_seq_len"] == 12
    assert guard._stats["calibration"]["truncation_applied"] is True
