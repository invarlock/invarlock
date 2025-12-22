from __future__ import annotations

import numpy as np

from invarlock.cli.commands.run import (
    _choose_dataset_split,
    _compute_mask_positions_digest,
    _hash_sequences,
)


def test_choose_dataset_split_behaviors():
    # Requested exact split
    s, fb = _choose_dataset_split(requested="test", available=["train", "test"])
    assert s == "test" and fb is False
    # Alias in available triggers fallback True
    s, fb = _choose_dataset_split(
        requested=None, available=["val", "train"]
    )  # includes alias
    assert s in {"validation", "val"} and fb is True
    # No available list → default validation with fallback
    s, fb = _choose_dataset_split(requested=None, available=None)
    assert s == "validation" and fb is True


def test_hash_sequences_stability():
    seqs = [[1, 2, 3], [4, 5]]
    h = _hash_sequences(seqs)
    assert isinstance(h, str) and len(h) == 32  # blake2s 16-byte digest


def test_hash_sequences_respects_boundaries():
    a = [[1, 2], [3]]
    b = [[1], [2, 3]]
    assert _hash_sequences(a) != _hash_sequences(b)


def test_compute_mask_positions_digest_roundtrip():
    win = {
        "preview": {"labels": [np.array([-100, 2, -100], dtype=np.int32)]},
        "final": {"labels": [np.array([-100, -100, 7], dtype=np.int32)]},
    }
    d = _compute_mask_positions_digest(win)
    assert isinstance(d, str) and len(d) > 0

    # Empty case returns None
    assert _compute_mask_positions_digest({"preview": {"labels": []}}) is None
