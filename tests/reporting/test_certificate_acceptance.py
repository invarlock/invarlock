import json
import math
from pathlib import Path
from typing import Any

import pytest

from invarlock.reporting.certificate import CERTIFICATE_SCHEMA_VERSION, make_certificate


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
            "adapter": "hf_gpt2",
            "device": "cpu",
            "seed": 7,
            "seeds": {"python": 7, "numpy": 7, "torch": 7},
        },
        "metrics": {
            "primary_metric": {
                "kind": "ppl_causal",
                "preview": ppl_prev,
                "final": ppl_fin_subj,
                "ratio_vs_baseline": 1.0,
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
    # Baseline ppl computed from preview to keep deterministic pairing
    prev = report["evaluation_windows"]["preview"]
    fin = report["evaluation_windows"]["final"]
    ppl_fin_base = math.exp(
        (1.00 * 100 + 1.10 * 200) / 300
    )  # slightly better than subject
    return {
        "run_id": "baseline",
        "model_id": report["meta"]["model_id"],
        "metrics": {
            "primary_metric": {"kind": "ppl_causal", "final": ppl_fin_base},
            "bootstrap": {"replicates": 200, "alpha": 0.05, "method": "percentile"},
        },
        "evaluation_windows": {"preview": prev, "final": fin},
    }


def test_v1_required_keys_and_shapes(tmp_path: Path) -> None:
    report = _mock_report_with_windows()
    baseline = _mock_baseline(report)
    cert = make_certificate(report, baseline)

    # schema version present and correct
    assert cert.get("schema_version") == CERTIFICATE_SCHEMA_VERSION == "v1"

    # primary metric present with required shape
    pm = cert.get("primary_metric", {})
    assert isinstance(pm, dict) and pm
    # display_ci must be a 2-length array of numbers
    dci = pm.get("display_ci")
    assert isinstance(dci, list | tuple) and len(dci) == 2
    assert all(isinstance(x, int | float) for x in dci)
    # kind must be in the allow-list when contracts are available
    kinds_path = Path.cwd() / "contracts" / "metric_kinds.json"
    if not kinds_path.exists():
        pytest.skip("metric_kinds.json contracts file not available")
    kinds = json.loads(kinds_path.read_text("utf-8"))
    assert str(pm.get("kind", "")).lower() in set(kinds)
    # ratio_vs_baseline must be finite when baseline present
    rvb = pm.get("ratio_vs_baseline")
    assert isinstance(rvb, int | float) and math.isfinite(float(rvb))


def test_validation_keys_subset_only() -> None:
    report = _mock_report_with_windows()
    baseline = _mock_baseline(report)
    cert = make_certificate(report, baseline)
    vkeys_path = Path.cwd() / "contracts" / "validation_keys.json"
    if not vkeys_path.exists():
        pytest.skip("validation_keys.json contracts file not available")
    allowed = set(json.loads(vkeys_path.read_text("utf-8")))
    observed = set((cert.get("validation") or {}).keys())
    # All emitted validation keys must be within the allow-list
    assert observed.issubset(allowed), (
        f"Unexpected validation keys: {sorted(observed - allowed)}"
    )


def test_no_top_level_ppl_keys() -> None:
    report = _mock_report_with_windows()
    baseline = _mock_baseline(report)
    cert = make_certificate(report, baseline)
    offenders = [
        k for k in cert.keys() if isinstance(k, str) and k.lower().startswith("ppl")
    ]
    assert not offenders, f"Certificate contains legacy top-level ppl keys: {offenders}"
