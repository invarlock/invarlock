from __future__ import annotations

from pathlib import Path

import pytest

from invarlock.eval.data import get_provider


def test_local_jsonl_windows_raises_on_empty(tmp_path):
    from invarlock.core.exceptions import DataError

    p = get_provider("local_jsonl", path=str(tmp_path))
    with pytest.raises(DataError):
        _ = p.windows(tokenizer=None, preview_n=1, final_n=1)


def test_local_jsonl_skips_malformed_json_lines(tmp_path: Path):
    jsonl = tmp_path / "rows.jsonl"
    jsonl.write_text(
        "\n".join(
            [
                '{"text": "keep-1"}',
                "not json",
                '{"text": "keep-2"}',
                '{"text": 3}',
            ]
        ),
        encoding="utf-8",
    )

    provider = get_provider("local_jsonl", file=str(jsonl))

    assert provider.load() == ["keep-1", "keep-2"]


def test_local_jsonl_skips_invalid_utf8_file(tmp_path: Path):
    bad = tmp_path / "bad.jsonl"
    good = tmp_path / "good.jsonl"
    bad.write_bytes(b"\xff\xfe\x00")
    good.write_text('{"text": "keep"}\n', encoding="utf-8")

    provider = get_provider("local_jsonl", data_files=[str(bad), str(good)])

    assert provider.load() == ["keep"]


def test_local_jsonl_pairs_skip_invalid_utf8_file(tmp_path: Path):
    bad = tmp_path / "bad.jsonl"
    good = tmp_path / "good.jsonl"
    bad.write_bytes(b"\xff\xfe\x00")
    good.write_text('{"src": "left", "tgt": "right"}\n', encoding="utf-8")

    provider = get_provider(
        "local_jsonl_pairs",
        data_files=[str(bad), str(good)],
        src_field="src",
        tgt_field="tgt",
    )

    assert provider._load_pairs() == [("left", "right")]
