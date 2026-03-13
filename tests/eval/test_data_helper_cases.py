from __future__ import annotations

import pytest

from invarlock.eval.data import (
    EvaluationWindow,
    compute_window_hash,
    get_provider,
    list_providers,
)


def test_compute_window_hash_includes_data_when_flagged():
    win = EvaluationWindow(
        input_ids=[[1, 2, 3], [4, 5, 6]],
        attention_masks=[[1, 1, 1], [1, 1, 0]],
        indices=[10, 20],
    )
    h1 = compute_window_hash(win, include_data=False)
    h2 = compute_window_hash(win, include_data=True)
    assert isinstance(h1, str) and isinstance(h2, str)
    assert h1 != h2  # data included changes hash


def test_get_provider_unknown_raises():
    from invarlock.core.exceptions import ValidationError

    with pytest.raises(ValidationError):
        get_provider("not-a-provider")


def test_list_providers_returns_known_names():
    names = list_providers()
    assert isinstance(names, list) and names
    assert all(isinstance(n, str) for n in names)
