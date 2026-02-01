from __future__ import annotations

from invarlock.reporting.guards_analysis import _extract_invariants


def test_invariants_baseline_compare_tokenizer_mismatch_fails() -> None:
    baseline_checks = {
        "parameter_count": 100,
        "layer_norm_paths": ("ln",),
        "embedding_vocab_sizes": {"embed": 10},
        "structure_hash": "deadbeef",
        "weight_tying": True,
    }
    current_checks = {
        **baseline_checks,
        "embedding_vocab_sizes": {"embed": 11},
    }

    baseline_report = {
        "guards": [
            {
                "name": "invariants",
                "metrics": {
                    "checks_performed": 5,
                    "violations_found": 0,
                    "fatal_violations": 0,
                    "warning_violations": 0,
                },
                "violations": [],
                "details": {
                    "baseline_checks": baseline_checks,
                    "current_checks": baseline_checks,
                },
            }
        ],
        "metrics": {"invariants": {}},
    }

    report = {
        "guards": [
            {
                "name": "invariants",
                "metrics": {
                    "checks_performed": 5,
                    "violations_found": 0,
                    "fatal_violations": 0,
                    "warning_violations": 0,
                },
                "violations": [],
                "details": {
                    "baseline_checks": current_checks,
                    "current_checks": current_checks,
                },
            }
        ],
        "metrics": {"invariants": {}},
    }

    out = _extract_invariants(report, baseline=baseline_report)
    assert out["status"] == "fail"
    assert any(f.get("type") == "tokenizer_mismatch" for f in out["failures"])


def test_invariants_baseline_compare_invariant_violation_warns() -> None:
    baseline_checks = {
        "parameter_count": 100,
        "layer_norm_paths": ("ln",),
        "embedding_vocab_sizes": {"embed": 10},
        "structure_hash": "deadbeef",
        "weight_tying": True,
    }
    current_checks = {
        **baseline_checks,
        "parameter_count": 101,
    }

    baseline_report = {
        "guards": [
            {
                "name": "invariants",
                "metrics": {
                    "checks_performed": 5,
                    "violations_found": 0,
                    "fatal_violations": 0,
                    "warning_violations": 0,
                },
                "violations": [],
                "details": {
                    "baseline_checks": baseline_checks,
                    "current_checks": baseline_checks,
                },
            }
        ],
        "metrics": {"invariants": {}},
    }

    report = {
        "guards": [
            {
                "name": "invariants",
                "metrics": {
                    "checks_performed": 5,
                    "violations_found": 0,
                    "fatal_violations": 0,
                    "warning_violations": 0,
                },
                "violations": [],
                "details": {
                    "baseline_checks": current_checks,
                    "current_checks": current_checks,
                },
            }
        ],
        "metrics": {"invariants": {}},
    }

    out = _extract_invariants(report, baseline=baseline_report)
    assert out["status"] == "warn"
    assert any(f.get("type") == "invariant_violation" for f in out["failures"])
