from __future__ import annotations

import hashlib
import importlib.util
import sys
import typing
from pathlib import Path

import invarlock.eval.data as eval_data_mod
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
    assert result["source"] == "explicit_preview_final_hashes"


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
    assert result["source"] == "explicit_preview_final_hashes"


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
    assert result["source"] == "config_fallback"


def test_compute_actual_window_hashes_from_sequences():
    report = {
        "data": {"dataset_hash": "source-dataset-digest"},
        "evaluation_windows": {
            "preview": {"input_ids": [[1, 2], [3, 4, 5]]},
            "final": {"input_ids": [[6], [7, 8]]},
        },
    }
    result = hashing_mod._compute_actual_window_hashes(report)
    assert result["preview"].startswith("sha256:")
    assert result["final_tokens"] == 3
    assert result["dataset"] == "source-dataset-digest"
    assert result["total_tokens"] == 8
    assert result["source"] == "explicit_token_ids"


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


def test_extract_dataset_info_surfaces_hosted_dataset_identity(monkeypatch):
    revision = "a" * 40
    monkeypatch.setattr(
        hashing_mod,
        "_compute_actual_window_hashes",
        lambda _report: {"preview": "p", "final": "f", "total_tokens": 2},
    )
    report = {
        "data": {
            "dataset": "hf_text",
            "provider": "hf_text",
            "dataset_name": "Salesforce/wikitext",
            "config_name": "wikitext-2-raw-v1",
            "revision": revision,
            "split": "validation",
            "seq_len": 8,
        }
    }

    identity = hashing_mod._extract_dataset_info(report)

    assert identity["provider"] == "hf_text"
    assert identity["dataset_name"] == "Salesforce/wikitext"
    assert identity["config_name"] == "wikitext-2-raw-v1"
    assert identity["revision"] == revision


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


def test_compute_actual_window_hashes_handles_non_mapping_data():
    report = {"data": None, "evaluation_windows": None}
    assert hashing_mod._compute_actual_window_hashes(report) == {}


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
    assert info["hash"]["source"] == "config_fallback"


def test_extract_dataset_info_handles_non_mapping_data(monkeypatch):
    monkeypatch.setattr(hashing_mod, "_compute_actual_window_hashes", lambda report: {})
    report = {
        "data": None,
        "evaluation_windows": {
            "preview": {"window_ids": [1, 2]},
            "final": {"window_ids": [3]},
        },
    }

    info = hashing_mod._extract_dataset_info(report)
    assert info["provider"] == "unknown"
    assert info["windows"]["preview"] == 2
    assert info["windows"]["final"] == 1


def test_extract_dataset_info_handles_scalar_data_payload(monkeypatch):
    monkeypatch.setattr(hashing_mod, "_compute_actual_window_hashes", lambda report: {})
    report = {
        "data": "invalid",
        "evaluation_windows": {
            "preview": {"window_ids": [1]},
            "final": {"window_ids": [2, 3]},
        },
    }

    info = hashing_mod._extract_dataset_info(report)
    assert info["provider"] == "unknown"
    assert info["windows"]["preview"] == 1
    assert info["windows"]["final"] == 2


def test_compute_window_hashes_aggregates_preview_and_final_tokens(monkeypatch):
    class _Window:
        def __init__(self, *input_ids):
            self.input_ids = [list(ids) for ids in input_ids]

    monkeypatch.setattr(
        hashing_mod,
        "compute_window_hash",
        lambda window, *, include_data: (
            "preview-digest" if len(window.input_ids) == 1 else "final-digest"
        ),
    )

    result = hashing_mod.compute_window_hashes(
        _Window([1, 2, 3]),
        _Window([4], [5, 6]),
    )
    assert result == {
        "preview": "sha256:preview-digest",
        "final": "sha256:final-digest",
        "source": "explicit_token_ids",
        "total_tokens": 6,
    }


def test_compute_window_hash_lazy_wrapper_imports_eval_module(monkeypatch):
    calls: list[tuple[object, bool]] = []

    def fake_compute(window, *, include_data):
        calls.append((window, include_data))
        return "wrapped-digest"

    monkeypatch.setattr(eval_data_mod, "compute_window_hash", fake_compute)

    class _Window:
        input_ids = [[1, 2]]

    digest = hashing_mod.compute_window_hash(_Window(), include_data=False)
    assert digest == "wrapped-digest"
    assert calls and calls[0][1] is False


def test_dataset_hashing_type_check_import_branch(monkeypatch, tmp_path: Path):
    module_path = Path(hashing_mod.__file__).resolve()
    module_name = "invarlock.reporting._dataset_hashing_typecheck_test"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setattr(typing, "TYPE_CHECKING", True)
    sys.modules.pop(module_name, None)
    try:
        spec.loader.exec_module(module)
        assert getattr(module, "__name__", "") == module_name
    finally:
        sys.modules.pop(module_name, None)
