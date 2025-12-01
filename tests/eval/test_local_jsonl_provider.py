from __future__ import annotations

from pathlib import Path

from invarlock.eval.data import get_provider


class _Tok:
    pad_token_id = 0

    def encode(self, text, max_length=16, truncation=True, padding="max_length"):
        n = min(max_length, max(1, len(text) % max_length))
        return [1] * n + [self.pad_token_id] * (max_length - n)


def test_local_jsonl_provider_load_and_windows(tmp_path: Path):
    d = tmp_path / "data"
    d.mkdir()
    f = d / "sample.jsonl"
    f.write_text(
        """
{"text": "hello"}
{"bad": 1}
not json
{"text": "world"}
""".strip(),
        encoding="utf-8",
    )
    p = get_provider("local_jsonl", path=str(d))
    texts = p.load()
    assert texts == ["hello", "world"]
    cap = p.estimate_capacity(_Tok(), seq_len=8, stride=4)
    assert cap["available_nonoverlap"] == 2
    prev, fin = p.windows(_Tok(), seq_len=8, stride=4, preview_n=1, final_n=1)
    assert len(prev.indices) == 1 and len(fin.indices) == 1
