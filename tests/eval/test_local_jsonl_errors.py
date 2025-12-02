from __future__ import annotations

import pytest

from invarlock.eval.data import get_provider


def test_local_jsonl_windows_raises_on_empty(tmp_path):
    from invarlock.core.exceptions import DataError

    p = get_provider("local_jsonl", path=str(tmp_path))
    with pytest.raises(DataError):
        _ = p.windows(tokenizer=None, preview_n=1, final_n=1)
