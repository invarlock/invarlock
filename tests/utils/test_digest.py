from __future__ import annotations

import numpy as np

from invarlock.utils.digest import hash_int_array, hash_json


def test_hash_json_stability_and_sensitivity():
    obj1 = {"a": 1, "b": [3, 2, 1], "c": {"x": True}}
    obj2 = {"b": [3, 2, 1], "c": {"x": True}, "a": 1}  # different order
    d1 = hash_json(obj1)
    d2 = hash_json(obj2)
    assert d1 == d2  # order-insensitive

    obj3 = {"a": 1, "b": [3, 2, 9], "c": {"x": True}}
    d3 = hash_json(obj3)
    assert d3 != d1


def test_hash_int_array_stability_and_sensitivity():
    arr = np.asarray([0, 1, 2, 3, 4], dtype=np.int32)
    d1 = hash_int_array(arr)
    d2 = hash_int_array(arr.copy())
    assert d1 == d2

    arr2 = np.asarray([0, 1, 2, 3, 5], dtype=np.int32)
    d3 = hash_int_array(arr2)
    assert d3 != d1
