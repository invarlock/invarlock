from __future__ import annotations

import invarlock.eval.data as data_mod


def test_hf_text_provider_windows_with_stub_dataset(monkeypatch):
    # Ensure provider allows init
    monkeypatch.setattr(data_mod, "HAS_DATASETS", True)

    class _DS:
        def __iter__(self):
            for x in ("alpha", "", "beta"):
                yield {"text": x}

    monkeypatch.setattr(data_mod, "load_dataset", lambda *a, **k: _DS())  # type: ignore[no-untyped-def]
    p = data_mod.HFTextProvider(
        dataset_name="stub", config_name=None, text_field="text", max_samples=10
    )

    class _Tok:
        pad_token_id = 0

        def encode(self, text, truncation=True, max_length=8, padding=None):
            return [1] * min(len(text), max_length)

    prev, fin = p.windows(_Tok(), seq_len=8, stride=4, preview_n=1, final_n=1)
    assert len(prev.indices) == 1 and len(fin.indices) == 1
