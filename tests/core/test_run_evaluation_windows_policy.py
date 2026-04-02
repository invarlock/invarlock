from __future__ import annotations

import pytest

from invarlock.core.run_evaluation_windows_policy import (
    build_fallback_evaluation_windows,
    serialize_evaluation_windows,
)


def test_serialize_evaluation_windows_returns_none_for_empty_input() -> None:
    assert serialize_evaluation_windows(None) is None
    assert serialize_evaluation_windows({}) is None


def test_serialize_evaluation_windows_copies_nested_sequences() -> None:
    payload = serialize_evaluation_windows(
        {
            "preview": {
                "window_ids": [1],
                "logloss": [0.1],
                "input_ids": ((1, 2),),
                "attention_masks": ((1, 1),),
                "token_counts": [2],
                "masked_token_counts": [1],
                "actual_token_counts": [2],
                "labels": ((-100, 2),),
            },
            "final": {"window_ids": [2], "input_ids": ((3, 4),)},
        }
    )
    assert payload == {
        "preview": {
            "window_ids": [1],
            "logloss": [0.1],
            "input_ids": [[1, 2]],
            "attention_masks": [[1, 1]],
            "token_counts": [2],
            "masked_token_counts": [1],
            "actual_token_counts": [2],
            "labels": [[-100, 2]],
        },
        "final": {
            "window_ids": [2],
            "logloss": [],
            "input_ids": [[3, 4]],
            "attention_masks": [],
            "token_counts": [],
            "masked_token_counts": [],
            "actual_token_counts": [],
            "labels": [],
        },
    }


def test_build_fallback_evaluation_windows_for_causal_records() -> None:
    payload = build_fallback_evaluation_windows(
        [{"input_ids": [1, 2], "attention_mask": [1, 1]}],
        [{"input_ids": [3], "attention_mask": [1]}],
        use_mlm=False,
    )
    assert payload == {
        "preview": {
            "window_ids": [0],
            "input_ids": [[1, 2]],
            "attention_masks": [[1, 1]],
            "token_counts": [2],
        },
        "final": {
            "window_ids": [1],
            "input_ids": [[3]],
            "attention_masks": [[1]],
            "token_counts": [1],
        },
    }


def test_build_fallback_evaluation_windows_for_mlm_records() -> None:
    payload = build_fallback_evaluation_windows(
        [{"input_ids": [1, 2], "attention_mask": [1, 1], "labels": [-100, 2]}],
        [{"input_ids": [3], "attention_mask": [1]}],
        use_mlm=True,
        preview_mask_counts=[1],
        final_mask_counts=[0],
    )
    assert payload["preview"]["masked_token_counts"] == [1]
    assert payload["preview"]["labels"] == [[-100, 2]]
    assert payload["final"]["masked_token_counts"] == [0]
    assert payload["final"]["labels"] == [[-100]]


def test_build_fallback_evaluation_windows_raises_for_missing_required_keys() -> None:
    with pytest.raises(KeyError):
        build_fallback_evaluation_windows(
            [{"input_ids": [1, 2]}],
            [],
            use_mlm=False,
        )
