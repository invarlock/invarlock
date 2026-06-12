from __future__ import annotations

import pytest

from invarlock.core.run_policy import (
    _nested_list_payload,
    _token_count,
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
            "example_ids": [],
            "logloss": [0.1],
            "input_ids": [[1, 2]],
            "attention_masks": [[1, 1]],
            "token_counts": [2],
            "masked_token_counts": [1],
            "actual_token_counts": [2],
            "labels": [[-100, 2]],
            "records": [],
        },
        "final": {
            "window_ids": [2],
            "example_ids": [],
            "logloss": [],
            "input_ids": [[3, 4]],
            "attention_masks": [],
            "token_counts": [],
            "masked_token_counts": [],
            "actual_token_counts": [],
            "labels": [],
            "records": [],
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


def test_serialize_evaluation_windows_keeps_records_and_processor_digest() -> None:
    payload = serialize_evaluation_windows(
        {
            "preview": {
                "records": [{"id": "ex-1"}, "ignore-me"],
                "input_records": [{"id": "ex-1", "image_path": "/tmp/a.ppm"}],
                "processor_sha256": "proc-123",
            },
            "final": {"records": "bad"},
        }
    )

    assert payload == {
        "preview": {
            "window_ids": [],
            "example_ids": [],
            "logloss": [],
            "input_ids": [],
            "attention_masks": [],
            "token_counts": [],
            "masked_token_counts": [],
            "actual_token_counts": [],
            "labels": [],
            "records": [{"id": "ex-1"}],
            "input_records": [{"id": "ex-1", "image_path": "/tmp/a.ppm"}],
            "processor_sha256": "proc-123",
        },
        "final": {
            "window_ids": [],
            "example_ids": [],
            "logloss": [],
            "input_ids": [],
            "attention_masks": [],
            "token_counts": [],
            "masked_token_counts": [],
            "actual_token_counts": [],
            "labels": [],
        },
    }


def test_build_fallback_evaluation_windows_for_multimodal_records() -> None:
    payload = build_fallback_evaluation_windows(
        [
            {
                "id": "ex-1",
                "example_id": "ex-1",
                "image_path": "/tmp/a.png",
                "answers": ["cat"],
                "processor_sha256": "proc-123",
            }
        ],
        [
            {
                "id": "ex-2",
                "image_path": "/tmp/b.png",
                "answers": ["dog"],
            }
        ],
        use_mlm=False,
    )

    assert payload == {
        "preview": {
            "example_ids": ["ex-1"],
            "records": [
                {
                    "id": "ex-1",
                    "example_id": "ex-1",
                    "image_path": "/tmp/a.png",
                    "answers": ["cat"],
                    "processor_sha256": "proc-123",
                }
            ],
            "processor_sha256": "proc-123",
        },
        "final": {
            "example_ids": ["ex-2"],
            "records": [
                {
                    "id": "ex-2",
                    "image_path": "/tmp/b.png",
                    "answers": ["dog"],
                }
            ],
        },
    }


def test_token_count_returns_zero_when_len_fails() -> None:
    class _BadRecord(dict):
        def get(self, key, default=None):  # noqa: ANN001
            if key == "input_ids":

                class _BadLen:
                    def __len__(self) -> int:
                        raise TypeError("bad len")

                return _BadLen()
            return super().get(key, default)

    assert _token_count(_BadRecord()) == 0


def test_serialize_evaluation_windows_ignores_non_sequence_scalar_fields() -> None:
    payload = serialize_evaluation_windows(
        {
            "preview": {
                "window_ids": True,
                "example_ids": True,
                "logloss": True,
                "input_ids": True,
                "attention_masks": {"bad": "mask"},
                "token_counts": True,
                "masked_token_counts": True,
                "actual_token_counts": True,
                "labels": True,
            },
            "final": {"window_ids": [1]},
        }
    )

    assert payload == {
        "preview": {
            "window_ids": [],
            "example_ids": [],
            "logloss": [],
            "input_ids": [],
            "attention_masks": [],
            "token_counts": [],
            "masked_token_counts": [],
            "actual_token_counts": [],
            "labels": [],
            "records": [],
        },
        "final": {
            "window_ids": [1],
            "example_ids": [],
            "logloss": [],
            "input_ids": [],
            "attention_masks": [],
            "token_counts": [],
            "masked_token_counts": [],
            "actual_token_counts": [],
            "labels": [],
            "records": [],
        },
    }


def test_serialize_evaluation_windows_filters_non_sequence_nested_items() -> None:
    payload = serialize_evaluation_windows(
        {
            "preview": {
                "input_ids": [(1, 2), "skip-me", [3, 4]],
                "attention_masks": [[1, 1], None],
                "labels": [(-100, 2), 7],
            },
            "final": {},
        }
    )

    assert payload == {
        "preview": {
            "window_ids": [],
            "example_ids": [],
            "logloss": [],
            "input_ids": [[1, 2], [3, 4]],
            "attention_masks": [[1, 1]],
            "token_counts": [],
            "masked_token_counts": [],
            "actual_token_counts": [],
            "labels": [[-100, 2]],
            "records": [],
        },
        "final": {
            "window_ids": [],
            "example_ids": [],
            "logloss": [],
            "input_ids": [],
            "attention_masks": [],
            "token_counts": [],
            "masked_token_counts": [],
            "actual_token_counts": [],
            "labels": [],
            "records": [],
        },
    }


def test_nested_list_payload_skips_non_sequence_item_and_continues_iteration() -> None:
    assert _nested_list_payload([[1], None, [2, 3]]) == [[1], [2, 3]]
