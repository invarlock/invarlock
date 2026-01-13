from __future__ import annotations

import hashlib

import invarlock.reporting.dataset_hashing as hashing_mod


def test_compute_actual_window_hashes_uses_report_hashes():
    report = {
        "data": {
            "preview_hash": "abc",
            "final_hash": "def",
            "dataset_hash": "ghi",
            "preview_total_tokens": 100,
            "final_total_tokens": 120,
        }
    }
    result = hashing_mod._compute_actual_window_hashes(report)
    assert result["preview"] == "blake2s:abc"
    assert result["final"] == "blake2s:def"
    assert result["total_tokens"] == 220


def test_compute_actual_window_hashes_report_hashes_non_int_token_counts():
    report = {
        "data": {
            "preview_hash": "abc",
            "final_hash": "def",
            "dataset_hash": "ghi",
            "preview_total_tokens": "100",
            "final_total_tokens": None,
        }
    }
    result = hashing_mod._compute_actual_window_hashes(report)
    assert result["preview"] == "blake2s:abc"
    assert result["final"] == "blake2s:def"
    assert result["dataset"] == "ghi"
    assert result["total_tokens"] == 0
    assert result["preview_tokens"] == "100"
    assert result["final_tokens"] is None


def test_compute_actual_window_hashes_config_fallback(monkeypatch):
    report = {
        "data": {
            "dataset": "wikitext2",
            "split": "validation",
            "seq_len": 16,
            "preview_n": 2,
            "final_n": 3,
        },
        "meta": {"seed": 42},
    }
    digest = hashlib.sha256(b"wikitext2validation162342").hexdigest()
    result = hashing_mod._compute_actual_window_hashes(report)
    assert result["preview"] == f"sha256:{digest[:32]}"
    assert result["final"].startswith("sha256:")
    assert result["total_tokens"] == (2 * 16) + (3 * 16)


def test_compute_actual_window_hashes_from_sequences():
    report = {
        "evaluation_windows": {
            "preview": {"input_ids": [[1, 2], [3, 4, 5]]},
            "final": {"input_ids": [[6], [7, 8]]},
        }
    }
    result = hashing_mod._compute_actual_window_hashes(report)
    assert result["preview"].startswith("sha256:")
    assert result["final_tokens"] == 3
    assert result["total_tokens"] == 8


def test_extract_dataset_info_prefers_actual_hash(monkeypatch):
    fake_hash = {
        "preview": "sha256:preview",
        "final": "sha256:final",
        "preview_tokens": 10,
        "final_tokens": 20,
        "total_tokens": 30,
    }
    monkeypatch.setattr(
        hashing_mod, "_compute_actual_window_hashes", lambda report: fake_hash
    )
    report = {
        "data": {"dataset": "demo", "split": "test", "seq_len": 32, "stride": 16},
        "evaluation_windows": {},
    }
    info = hashing_mod._extract_dataset_info(report)
    assert info["hash"] == fake_hash
    assert info["provider"] == "demo"
    assert info["windows"]["seed"] is None


def test_compute_actual_window_hashes_handles_non_dict_windows():
    report = {
        "evaluation_windows": [],
        "data": {
            "dataset": "demo",
            "split": "test",
            "seq_len": 8,
            "preview_n": 1,
            "final_n": 1,
        },
    }
    result = hashing_mod._compute_actual_window_hashes(report)
    assert result["preview"].startswith("sha256:")


def test_compute_actual_window_hashes_skips_bad_sequences():
    class Bad:
        def __repr__(self):
            raise RuntimeError("boom")

    report = {
        "evaluation_windows": {
            "preview": {"input_ids": [[Bad()]]},
            "final": {"input_ids": [[1, 2]]},
        }
    }
    result = hashing_mod._compute_actual_window_hashes(report)
    assert result["preview"].startswith("sha256:")
    assert result["final_tokens"] == 2


def test_compute_actual_window_hashes_error_returns_empty():
    class BadReport(dict):
        def get(self, *_args, **_kwargs):
            raise RuntimeError("nope")

    assert hashing_mod._compute_actual_window_hashes(BadReport()) == {}


def test_extract_dataset_info_uses_config_fallback(monkeypatch):
    monkeypatch.setattr(hashing_mod, "_compute_actual_window_hashes", lambda report: {})
    report = {
        "data": {
            "dataset": "demo",
            "split": "train",
            "seq_len": 4,
            "preview_n": 2,
            "final_n": 3,
            "tokenizer_hash": "tok-hash",
        },
        "meta": {"seed": 7},
    }
    info = hashing_mod._extract_dataset_info(report)
    assert info["hash"]["dataset"] == "tok-hash"
    assert info["hash"]["total_tokens"] == (2 * 4) + (3 * 4)
