import math
from pathlib import Path
from typing import Any

from invarlock.public_contracts import load_json_contract
from invarlock.reporting.report_make import make_report
from invarlock.reporting.report_summary import compute_console_validation_block
from tests.cli._support_runtime_policy import bind_runtime_policy


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
        "data": {
            "dataset": "synthetic",
            "split": "validation",
            "seq_len": 8,
            "stride": 4,
            "preview_n": 2,
            "final_n": 2,
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
        "edit": {"name": "structured", "plan_digest": "fixture", "deltas": {}},
        "artifacts": {
            "events_path": "",
            "logs_path": "",
            "checkpoint_path": None,
        },
        "flags": {"guard_recovered": False, "rollback_reason": None},
        "guards": [],
    }
    return bind_runtime_policy(report)


def _mock_baseline(report: dict[str, Any]) -> dict[str, Any]:
    prev = report["evaluation_windows"]["preview"]
    fin = {
        "window_ids": [3, 4],
        "logloss": [1.00, 1.10],
        "token_counts": [100, 200],
    }
    ppl_prev_base = math.exp((1.00 * 100 + 1.06 * 200) / 300)
    ppl_fin_base = math.exp((1.00 * 100 + 1.10 * 200) / 300)
    return {
        "meta": {"auto": {"tier": "balanced"}},
        "run_id": "baseline",
        "model_id": report["meta"]["model_id"],
        "evaluation_windows": {"preview": prev, "final": fin},
        "ppl_final": ppl_fin_base,
        "ppl_preview": ppl_prev_base,
        "primary_metric": {
            "kind": "ppl_causal",
            "final": ppl_fin_base,
            "preview": ppl_prev_base,
        },
    }


def _labels_from_block(cert: dict[str, Any]) -> list[str]:
    block = compute_console_validation_block(cert)
    labels = [row["label"] for row in block.get("rows", [])]
    return labels


def test_labels_subset_of_allow_list_and_ordered(tmp_path: Path) -> None:
    report = _mock_report_with_windows()
    baseline = _mock_baseline(report)
    cert = make_report(report, baseline)

    observed = _labels_from_block(cert)
    allow = load_json_contract("console_labels.json")
    # Guard Metric Impact may be omitted when not evaluated; others remain
    assert all(label in allow for label in observed)
    # Preserve allow-list ordering for present labels
    order_index = {label: i for i, label in enumerate(allow)}
    assert observed == sorted(observed, key=lambda x: order_index.get(x, 1_000))


def test_overall_status_policy_from_canonical_rows_only(tmp_path: Path) -> None:
    report = _mock_report_with_windows()
    baseline = _mock_baseline(report)
    cert = make_report(report, baseline)

    # Ensure overall status reflects only canonical rows
    block = compute_console_validation_block(cert)
    overall_before = bool(block.get("overall_pass"))

    # Add an extra non-canonical key in-place; block computation must ignore it
    cert.setdefault("validation", {})["non_canonical_key"] = True
    block2 = compute_console_validation_block(cert)
    overall_after = bool(block2.get("overall_pass"))
    assert overall_before == overall_after


def test_guard_metric_impact_row_omitted_when_not_evaluated(tmp_path: Path) -> None:
    report = _mock_report_with_windows()
    baseline = _mock_baseline(report)
    # No guard_metric_impact context in report → not evaluated
    cert = make_report(report, baseline)
    block = compute_console_validation_block(cert)
    labels = [row["label"] for row in block.get("rows", [])]
    assert "Guard Metric Impact Acceptable" not in labels
