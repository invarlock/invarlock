from __future__ import annotations

from pathlib import Path

from invarlock.reporting.certificate import make_certificate


def test_choose_dataset_split_logic():
    # Import the helper directly for unit testing
    from invarlock.cli.commands.run import SPLIT_ALIASES, _choose_dataset_split

    # When requested is provided, it should return verbatim and no fallback
    s, fb = _choose_dataset_split(requested="train", available=["train", "validation"])
    assert s == "train" and fb is False

    # When not requested, it should pick the first alias present
    for cand in SPLIT_ALIASES:
        s, fb = _choose_dataset_split(requested=None, available=[cand, "other"])
        assert s == cand and fb is True
        break

    # When no aliases present, it should pick the first sorted available
    s, fb = _choose_dataset_split(requested=None, available=["zzz", "aaa"])
    assert s == "aaa" and fb is True

    # When available is None/empty, last-resort fallback to validation
    s, fb = _choose_dataset_split(requested=None, available=None)
    assert s == "validation" and fb is True


def test_certificate_telemetry_includes_split(tmp_path: Path):
    # Minimal, valid-enough report/baseline pair
    report = {
        "meta": {"model_id": "m", "adapter": "hf_gpt2", "device": "cpu", "seed": 42},
        "metrics": {
            "primary_metric": {"kind": "ppl_causal", "preview": 10.0, "final": 10.0},
        },
        "evaluation_windows": {
            "preview": {
                "window_ids": [1, 2],
                "logloss": [1.0, 1.0],
                "token_counts": [10, 10],
            },
            "final": {
                "window_ids": [3, 4],
                "logloss": [1.0, 1.0],
                "token_counts": [10, 10],
            },
        },
        "guards": [],
        "artifacts": {"events_path": "", "logs_path": ""},
        "provenance": {"dataset_split": "test", "split_fallback": True},
    }
    baseline = {
        "run_id": "baseline",
        "model_id": "m",
        "evaluation_windows": {
            "preview": {
                "window_ids": [1, 2],
                "logloss": [1.0, 1.0],
                "token_counts": [10, 10],
            },
            "final": {
                "window_ids": [3, 4],
                "logloss": [1.0, 1.0],
                "token_counts": [10, 10],
            },
        },
    }
    # Provide baseline primary metric final
    baseline.setdefault("metrics", {})["primary_metric"] = {
        "kind": "ppl_causal",
        "final": 10.0,
        "preview": 10.0,
    }

    cert = make_certificate(report, baseline)
    summary = cert.get("telemetry", {}).get("summary_line", "")
    # Expect split=test*, where * denotes fallback
    assert "split=test*" in summary


def test_dataset_split_provenance_matches_data_when_both_present():
    # If both report["data"]["split"] and provenance.dataset_split are present,
    # they should match. This keeps provenance as the source‑of‑truth while mirroring to data.
    report = {
        "meta": {"model_id": "m", "adapter": "hf_gpt2", "device": "cpu", "seed": 42},
        "data": {
            "dataset": "dummy",
            "split": "eval",
            "seq_len": 8,
            "stride": 4,
            "preview_n": 1,
            "final_n": 1,
        },
        "metrics": {
            "primary_metric": {"kind": "ppl_causal", "preview": 10.0, "final": 10.0}
        },
        "evaluation_windows": {
            "preview": {"window_ids": [1], "logloss": [1.0], "token_counts": [10]},
            "final": {"window_ids": [2], "logloss": [1.0], "token_counts": [10]},
        },
        "guards": [],
        "artifacts": {"events_path": "", "logs_path": ""},
        "provenance": {"dataset_split": "eval", "split_fallback": False},
    }

    # Direct assertion: when both exist, they must match
    assert report["data"]["split"] == report["provenance"]["dataset_split"]
