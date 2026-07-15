from __future__ import annotations

import pytest

from invarlock.core.exceptions import DataError
from invarlock.eval.data_tokenization import (
    to_python_token_rows,
    tokenize_texts_padded,
)


class _CoercibleTensor:
    def squeeze(self, _axis: int):
        raise RuntimeError("squeeze unavailable")

    def detach(self):
        raise RuntimeError("detach unavailable")

    def cpu(self):
        raise RuntimeError("cpu unavailable")

    def tolist(self):
        return [1, 2, 3]


class _SelectiveTokenizer:
    def __call__(
        self,
        text: str,
        *,
        truncation: bool,
        padding: str,
        max_length: int,
        return_attention_mask: bool,
    ) -> dict[str, list[int]]:
        del truncation, padding, max_length, return_attention_mask
        if text == "bad":
            raise RuntimeError("tokenizer boom")
        return {"input_ids": [1, 2], "attention_mask": [1, 1]}


class _BatchTokenizer:
    pad_token_id = 9

    def __init__(self) -> None:
        self.calls: list[object] = []

    def __call__(
        self,
        texts,
        *,
        truncation: bool,
        padding: str,
        max_length: int,
        return_attention_mask: bool,
    ):
        del truncation, padding, max_length, return_attention_mask
        self.calls.append(texts)
        rows = [[index + 1, index + 2] for index, _text in enumerate(texts)]
        return {
            "input_ids": rows,
            "attention_mask": [[1, 1] for _text in texts],
        }


def test_to_python_token_rows_recovers_after_runtime_coercion_failures() -> None:
    rows = to_python_token_rows(_CoercibleTensor(), batch_size=1)

    assert rows == [[1, 2, 3]]


def test_to_python_token_rows_rejects_non_batched_rows() -> None:
    with pytest.raises(TypeError, match="Tokenizer did not return batched rows"):
        to_python_token_rows([1, 2, 3], batch_size=2)


def test_tokenize_texts_padded_warns_and_keeps_successes_on_runtime_failure() -> None:
    with pytest.warns(UserWarning, match="Failed to tokenize sample 11"):
        input_ids, attention_masks, kept_positions = tokenize_texts_padded(
            ["ok", "bad"],
            _SelectiveTokenizer(),
            4,
            positions=[10, 11],
            warn_on_failure=True,
        )

    assert input_ids == [[1, 2, 0, 0]]
    assert attention_masks == [[1, 1, 0, 0]]
    assert kept_positions == [10]


def test_tokenize_texts_padded_raises_data_error_without_warning_mode() -> None:
    with pytest.raises(DataError) as excinfo:
        tokenize_texts_padded(
            ["ok", "bad"],
            _SelectiveTokenizer(),
            4,
            positions=[10, 11],
        )

    assert excinfo.value.code == "E304"
    assert excinfo.value.details["failed_positions"] == [11]


def test_tokenize_texts_padded_uses_one_callable_batch_without_changing_rows() -> None:
    tokenizer = _BatchTokenizer()

    input_ids, attention_masks, kept_positions = tokenize_texts_padded(
        ["first", "second", "third"],
        tokenizer,
        4,
        positions=[7, 8, 9],
    )

    assert tokenizer.calls == [["first", "second", "third"]]
    assert input_ids == [[1, 2, 9, 9], [2, 3, 9, 9], [3, 4, 9, 9]]
    assert attention_masks == [[1, 1, 0, 0]] * 3
    assert kept_positions == [7, 8, 9]


def test_tokenize_texts_padded_bounds_callable_batch_size() -> None:
    tokenizer = _BatchTokenizer()
    texts = [f"sample-{index}" for index in range(513)]

    input_ids, attention_masks, kept_positions = tokenize_texts_padded(
        texts,
        tokenizer,
        4,
    )

    assert [len(call) for call in tokenizer.calls] == [256, 256, 1]
    assert len(input_ids) == len(attention_masks) == len(kept_positions) == 513
