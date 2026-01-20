import math

from invarlock.reporting.certificate import make_certificate, validate_certificate
from invarlock.reporting.policy_utils import _compute_variance_policy_digest
from invarlock.reporting.render import render_certificate_markdown
from invarlock.reporting.utils import (
    _coerce_interval,
    _pair_logloss_windows,
    _sanitize_seed_bundle,
)


def test_certificate_interval_and_seed_sanitizers():
    # _coerce_interval string and bad values
    assert _coerce_interval("(0.1, 0.2)") == (0.1, 0.2)
    bad_lo, bad_hi = _coerce_interval("not-a-tuple")
    assert math.isnan(bad_lo) and math.isnan(bad_hi)

    # _sanitize_seed_bundle respects explicit None and fallback
    bundle = _sanitize_seed_bundle({"python": 7, "torch": None}, fallback=3)
    assert bundle["python"] == 7 and bundle["numpy"] is None and bundle["torch"] is None


def test_variance_policy_digest_and_pairing():
    digest = _compute_variance_policy_digest(
        {
            "deadband": 0.1,
            "min_abs_adjust": 0.01,
            "unrelated": "ignore",
        }
    )
    assert isinstance(digest, str) and len(digest) == 16

    # _pair_logloss_windows pairs by IDs and filters mismatches
    run = {"window_ids": [10, 20, 30], "logloss": [1.0, 2.0, 3.0]}
    base = {"window_ids": [30, 10], "logloss": [2.5, 0.7]}
    paired = _pair_logloss_windows(run, base)
    assert paired is not None
    r, b = paired
    # Expect order aligned to run IDs where baseline exists (10 and 30)
    assert r == [1.0, 3.0] and b == [0.7, 2.5]


def test_make_certificate_uses_paired_baseline_ratio_ci():
    report = {
        "meta": {
            "model_id": "gpt2",
            "adapter": "hf_causal",
            "device": "cpu",
            "ts": "2025-01-01T00:00:00",
            "commit": "deadbeef",
            "seed": 42,
            "auto": {"tier": "balanced", "probes": 0, "target_pm_ratio": None},
        },
        "data": {
            "dataset": "dummy",
            "split": "validation",
            "seq_len": 8,
            "stride": 4,
            "preview_n": 2,
            "final_n": 2,
        },
        "edit": {
            "name": "structured",
            "plan_digest": "abcd",
            "deltas": {
                "params_changed": 0,
                "heads_pruned": 0,
                "neurons_pruned": 0,
                "layers_modified": 0,
                "sparsity": None,
            },
        },
        "guards": [],
        "metrics": {
            "primary_metric": {
                "kind": "ppl_causal",
                "preview": 10.0,
                "final": 10.1,
                "ratio_vs_baseline": 1.01,
                "display_ci": (0.95, 1.05),
            },
            "bootstrap": {"method": "bca", "replicates": 16, "alpha": 0.1, "seed": 0},
        },
        "artifacts": {"events_path": "", "logs_path": "", "checkpoint_path": None},
        "flags": {"guard_recovered": False, "rollback_reason": None},
        "evaluation_windows": {
            "final": {"window_ids": [1, 2], "logloss": [1.0, 2.0]},
        },
    }

    baseline = {
        "run_id": "baseline-1",
        "model_id": "gpt2",
        "evaluation_windows": {
            "final": {"window_ids": [2, 1], "logloss": [2.5, 0.7]},
        },
    }

    cert = make_certificate(report, baseline)
    stats = cert.get("dataset", {}).get("windows", {}).get("stats", {})
    pm = cert.get("primary_metric", {})
    assert (
        isinstance(pm.get("display_ci"), tuple | list)
        and stats.get("pairing") == "paired_baseline"
    )
    # Ensure display_ci approximately equals exp(delta_ci)
    _lo, _hi = pm.get("analysis_point_preview"), pm.get("analysis_point_final")
    # When analysis points are present, Δlog bounds were in report, display_ci is exp(Δlog bounds)
    dci = pm.get("display_ci")
    assert isinstance(dci, tuple | list) and len(dci) == 2

    # Validate and render markdown to cover certificate branches
    assert validate_certificate(cert) is True
    md = render_certificate_markdown(cert)
    assert isinstance(md, str) and "InvarLock Safety Certificate" in md

    # Negative schema version path
    cert_bad = dict(cert)
    cert_bad["schema_version"] = "wrong"
    assert validate_certificate(cert_bad) is False
