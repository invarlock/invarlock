from __future__ import annotations

from pathlib import Path

from invarlock.eval.data import get_provider


def test_local_jsonl_data_files_glob(tmp_path: Path):
    (tmp_path / "a.jsonl").write_text('{"text": "a"}\n', encoding="utf-8")
    (tmp_path / "b.jsonl").write_text('{"text": "b"}\n', encoding="utf-8")
    p = get_provider("local_jsonl", data_files=str(tmp_path / "*.jsonl"))
    texts = p.load()
    assert sorted(texts) == ["a", "b"]
    cap = p.estimate_capacity(tokenizer=None, seq_len=8, stride=4)
    assert cap["available_nonoverlap"] == 2 and cap["candidate_limit"] == 2
