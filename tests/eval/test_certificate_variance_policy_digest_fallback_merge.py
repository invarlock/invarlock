from unittest.mock import patch

from invarlock.reporting.certificate import (
    _compute_variance_policy_digest,
    make_certificate,
)


def test_variance_policy_digest_fallback_merges_guard_policy_keys():
    # Build a report where effective policies likely omit variance canonical knobs,
    # but the guard entry contains a full variance policy. Certificate should
    # compute digest from the guard policy and merge missing keys back into policies.variance.
    variance_policy = {
        "deadband": 0.02,
        "min_abs_adjust": 0.012,
        "max_scale_step": 0.03,
        "min_effect_lognll": 0.0009,
        "predictive_one_sided": True,
        "topk_backstop": 1,
        "max_adjusted_modules": 1,
    }
    report = {
        "meta": {"model_id": "m", "seed": 7, "auto": {"tier": "balanced"}},
        "metrics": {"ppl_preview": 10.0, "ppl_final": 10.0},
        "data": {
            "dataset": "dummy",
            "split": "val",
            "seq_len": 8,
            "stride": 1,
            "preview_n": 1,
            "final_n": 1,
        },
        "guards": [
            {
                "name": "variance",
                "policy": dict(variance_policy),
                "metrics": {"ve_enabled": True},
            },
        ],
        "edit": {
            "name": "structured",
            "deltas": {
                "params_changed": 0,
                "heads_pruned": 0,
                "neurons_pruned": 0,
                "layers_modified": 0,
            },
        },
        "evaluation_windows": {"final": {"window_ids": [1], "logloss": [0.1]}},
        "plugins": {"adapter": {}, "edit": {}, "guards": []},
    }
    baseline = {
        "run_id": "b",
        "model_id": "m",
        "ppl_final": 10.0,
        "evaluation_windows": {"final": {"window_ids": [1], "logloss": [0.1]}},
    }

    with patch("invarlock.reporting.certificate.validate_report", return_value=True):
        cert = make_certificate(report, baseline)

    expected = _compute_variance_policy_digest(variance_policy)
    # Digest should be present in auto.policy_digest and policies.variance.policy_digest
    assert cert["auto"]["policy_digest"] == expected
    assert cert["policies"]["variance"]["policy_digest"] == expected
    # And missing keys should have been merged back into policies.variance
    for k, v in variance_policy.items():
        assert cert["policies"]["variance"].get(k) == v
