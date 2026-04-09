from __future__ import annotations

import pytest

import invarlock.eval.data as data_mod
import invarlock.eval.data_support as data_support_mod


def test_hf_text_provider_windows_raises_when_empty(monkeypatch):
    monkeypatch.setattr(data_support_mod, "HAS_DATASETS", True)

    class _DS:
        def __iter__(self):
            # All rows missing the 'text' field
            for _ in range(3):
                yield {"not_text": ""}

    monkeypatch.setattr(data_support_mod, "load_dataset", lambda *a, **k: _DS())
    p = data_mod.HFTextProvider(
        dataset_name="stub", config_name=None, text_field="text", max_samples=10
    )
    from invarlock.core.exceptions import DataError

    with pytest.raises(DataError):
        _ = p.windows(tokenizer=None, seq_len=8, stride=4, preview_n=1, final_n=1)
