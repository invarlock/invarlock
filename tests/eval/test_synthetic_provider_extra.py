from __future__ import annotations

from invarlock.eval.data import get_provider


class _Tok:
    pad_token_id = 0

    def encode(self, text, max_length=16, truncation=True, padding="max_length"):
        # Turn text length into a simple deterministic pattern
        n = min(max_length, max(1, len(text) % max_length))
        return [1] * n + [self.pad_token_id] * (max_length - n)


def test_synthetic_provider_windows_and_capacity():
    p = get_provider("synthetic")
    cap = p.estimate_capacity(_Tok(), seq_len=16, stride=8)
    assert cap["available_nonoverlap"] > 0 and cap["available_unique"] > 0
    prev, fin = p.windows(_Tok(), seq_len=16, stride=8, preview_n=8, final_n=8)
    assert len(prev.indices) == 8 and len(fin.indices) == 8
