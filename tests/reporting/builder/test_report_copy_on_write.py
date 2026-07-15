from __future__ import annotations

import copy

from invarlock.reporting.report_make import make_report
from invarlock.reporting.report_normalization import (
    normalize_and_validate_run_report,
)
from tests.reporting.builder._support_report_acceptance import (
    mock_baseline,
    mock_report_with_windows,
)


def _large_report_pair() -> tuple[dict[str, object], dict[str, object]]:
    subject = mock_report_with_windows()
    baseline = mock_baseline(subject)
    records = [
        {
            "record_id": f"record-{index:04d}",
            "tokens": list(range(128)),
            "attributes": {"partition": index % 7, "accepted": True},
        }
        for index in range(256)
    ]
    subject["context"]["assembly_fixture"] = records
    subject["meta"].update(
        {
            "env_flags": {"runtime": {"deterministic": True}},
            "determinism": {"algorithms": ["reference"]},
            "model_profile": {"selectors": ["attention"]},
            "cuda_flags": {"devices": [0]},
            "plugins": {"adapter": {"versions": ["1"]}},
        }
    )
    subject["metrics"]["invariants"] = {
        "tokenizer": {"passed": True, "observed": {"vocab_sizes": [1024]}}
    }
    baseline["context"]["assembly_fixture"] = copy.deepcopy(records)
    return subject, baseline


def test_make_report_preserves_large_inputs_and_detaches_output() -> None:
    subject, baseline = _large_report_pair()
    subject_before = copy.deepcopy(subject)
    baseline_before = copy.deepcopy(baseline)

    assembled = make_report(subject, baseline)

    assert subject == subject_before
    assert baseline == baseline_before

    assembled["context"]["assembly_fixture"][0]["tokens"][0] = -1
    assembled["guards"].append({"name": "output-only"})
    assembled["meta"]["env_flags"]["runtime"]["deterministic"] = False
    assembled["meta"]["determinism"]["algorithms"].append("changed")
    assembled["meta"]["model_profile"]["selectors"].append("changed")
    assembled["meta"]["cuda_flags"]["devices"].append(1)
    assembled["plugins"]["adapter"]["versions"].append("2")
    assembled["invariants"]["details"]["tokenizer"]["observed"]["vocab_sizes"].append(
        2048
    )
    assert subject == subject_before
    assert baseline == baseline_before


def test_public_normalization_still_returns_a_defensive_copy() -> None:
    subject = mock_report_with_windows()
    normalized = normalize_and_validate_run_report(subject)

    normalized["context"]["profile"] = "changed"

    assert subject["context"]["profile"] == "dev"
