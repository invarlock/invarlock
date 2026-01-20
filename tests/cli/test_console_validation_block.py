import json
import math
from pathlib import Path
from typing import Any

import pytest

from invarlock.reporting.certificate import make_certificate
from invarlock.reporting.render import compute_console_validation_block


def _mock_report_with_windows() -> dict[str, Any]:
    # Deterministic synthetic windows for ppl_causal
    preview = {
        "window_ids": [1, 2],
        "logloss": [1.00, 1.06],
        "token_counts": [100, 200],
    }
    final = {
        "window_ids": [3, 4],
        "logloss": [1.05, 1.15],
        "token_counts": [100, 200],
    }
    ppl_prev = math.exp((1.00 * 100 + 1.06 * 200) / 300)
    ppl_fin_subj = math.exp((1.05 * 100 + 1.15 * 200) / 300)
    report = {
        "meta": {
            "model_id": "stub",
            "adapter": "hf_causal",
            "device": "cpu",
            "seed": 7,
            "seeds": {"python": 7, "numpy": 7, "torch": 7},
        },
        "metrics": {
            "primary_metric": {
                "kind": "ppl_causal",
                "preview": ppl_prev,
                "final": ppl_fin_subj,
            },
            "bootstrap": {"replicates": 200, "alpha": 0.05, "method": "percentile"},
        },
        "evaluation_windows": {"preview": preview, "final": final},
        "edit": {"name": "structured"},
        "artifacts": {"events_path": "", "logs_path": ""},
        "guards": [],
    }
    return report


def _mock_baseline(report: dict[str, Any]) -> dict[str, Any]:
    prev = report["evaluation_windows"]["preview"]
    fin = report["evaluation_windows"]["final"]
    ppl_fin_base = math.exp((1.00 * 100 + 1.10 * 200) / 300)
    return {
        "run_id": "baseline",
        "model_id": report["meta"]["model_id"],
        "evaluation_windows": {"preview": prev, "final": fin},
        "metrics": {
            "bootstrap": {"replicates": 200, "alpha": 0.05, "method": "percentile"},
            "primary_metric": {
                "kind": "ppl_causal",
                "final": ppl_fin_base,
                "preview": ppl_fin_base,
            },
        },
    }


def _labels_from_block(cert: dict[str, Any]) -> list[str]:
    block = compute_console_validation_block(cert)
    labels = [row["label"] for row in block.get("rows", [])]
    return labels


def test_labels_subset_of_allow_list_and_ordered(tmp_path: Path) -> None:
    report = _mock_report_with_windows()
    baseline = _mock_baseline(report)
    cert = make_certificate(report, baseline)

    observed = _labels_from_block(cert)
    contracts_path = Path.cwd() / "contracts" / "console_labels.json"
    if not contracts_path.exists():
        pytest.skip("console_labels.json contracts file not available")
    allow = json.loads(contracts_path.read_text("utf-8"))
    # Guard Overhead may be omitted when not evaluated; others remain
    assert all(label in allow for label in observed)
    # Preserve allow-list ordering for present labels
    order_index = {label: i for i, label in enumerate(allow)}
    assert observed == sorted(observed, key=lambda x: order_index.get(x, 1_000))


def test_overall_status_policy_from_canonical_rows_only(tmp_path: Path) -> None:
    report = _mock_report_with_windows()
    baseline = _mock_baseline(report)
    cert = make_certificate(report, baseline)

    # Ensure overall status reflects only canonical rows
    block = compute_console_validation_block(cert)
    overall_before = bool(block.get("overall_pass"))

    # Add an extra non-canonical key in-place; block computation must ignore it
    cert.setdefault("validation", {})["non_canonical_key"] = True
    block2 = compute_console_validation_block(cert)
    overall_after = bool(block2.get("overall_pass"))
    assert overall_before == overall_after


def test_guard_overhead_row_omitted_when_not_evaluated(tmp_path: Path) -> None:
    report = _mock_report_with_windows()
    baseline = _mock_baseline(report)
    # No guard_overhead context in report → not evaluated
    cert = make_certificate(report, baseline)
    block = compute_console_validation_block(cert)
    labels = [row["label"] for row in block.get("rows", [])]
    assert "Guard Overhead Acceptable" not in labels
