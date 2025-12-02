from __future__ import annotations

import math

from invarlock.reporting.certificate import (
    TIER_RATIO_LIMITS,
    _compute_confidence_label,
    _compute_edit_digest,
    _is_ppl_kind,
)
from invarlock.reporting.policy_utils import (
    _compute_thresholds_payload,
    _compute_variance_policy_digest,
)
from invarlock.reporting.utils import (
    _coerce_int,
    _coerce_interval,
    _infer_scope_from_modules,
    _pair_logloss_windows,
    _sanitize_seed_bundle,
    _weighted_mean,
)


def test_ppl_helpers_and_edit_digest() -> None:
    assert (
        _is_ppl_kind("ppl")
        and _is_ppl_kind("ppl_causal")
        and _is_ppl_kind("ppl_seq2seq")
    )
    assert not _is_ppl_kind("accuracy")
    # Legacy _get_ppl_final removed; rely on primary_metric parsing in certificates.
    # Edit digest
    ed = _compute_edit_digest({"edit": {"name": "quant_rtn", "config": {"k": 1}}})
    assert ed["family"] == "quantization" and ed["impl_hash"]
    ed2 = _compute_edit_digest({})
    assert ed2["family"] == "cert_only"


def test_confidence_label_and_thresholds() -> None:
    cert = {
        "validation": {"primary_metric_acceptable": True},
        "primary_metric": {"kind": "ppl_causal", "display_ci": (0.97, 1.00)},
        "resolved_policy": {"confidence": {"ppl_ratio_width_max": 0.05}},
    }
    out = _compute_confidence_label(cert)
    assert out["label"] in {"High", "Medium"}
    # Accuracy basis uses pp threshold
    cert_acc = {
        "validation": {"primary_metric_acceptable": True},
        "primary_metric": {"kind": "accuracy", "display_ci": (0.80, 0.805)},
        "resolved_policy": {"confidence": {"accuracy_delta_pp_width_max": 1.0}},
    }
    out_acc = _compute_confidence_label(cert_acc)
    assert out_acc["basis"] == "accuracy"

    # Unstable → Medium at best
    cert_unstable = {
        "validation": {"primary_metric_acceptable": True},
        "primary_metric": {
            "kind": "ppl_causal",
            "display_ci": (0.97, 1.00),
            "unstable": True,
        },
    }
    assert _compute_confidence_label(cert_unstable)["label"] in {"Medium", "Low"}


def test_coercions_and_scope_inference() -> None:
    assert _coerce_int(3.0) == 3
    assert _coerce_int(3.2) is None
    assert _coerce_int(float("inf")) is None
    assert _coerce_int(True) == 1
    sb = _sanitize_seed_bundle({"python": 1, "numpy": None, "torch": 3.0}, fallback=7)
    assert sb == {"python": 1, "numpy": None, "torch": 3}
    assert _infer_scope_from_modules(["layer.attn.q_proj", "mlp.fc1"]) in {
        "attn+ffn",
        "ffn+attn",
    }

    assert _coerce_interval("(1.0, 2.0)") == (1.0, 2.0)
    c = _coerce_interval(["x", 1])
    assert (
        isinstance(c, tuple) and len(c) == 2 and math.isnan(c[0]) and math.isnan(c[1])
    )


def test_weighted_mean_and_pair_logloss_windows() -> None:
    assert math.isclose(_weighted_mean([1, 2, 3], [1, 0, 1]), 2.0)
    run_w = {"window_ids": [1, 2, 3], "logloss": [0.1, 0.2, 0.3]}
    base_w = {"window_ids": [2, 3, 4], "logloss": [0.25, 0.35, 0.45]}
    paired = _pair_logloss_windows(run_w, base_w)
    assert paired and len(paired[0]) == len(paired[1]) == 2


def test_policy_digests_and_threshold_payload() -> None:
    pol = {"deadband": 0.0, "min_abs_adjust": 0.1, "max_scale_step": 0.5}
    d = _compute_variance_policy_digest(pol)
    assert isinstance(d, str)
    thr = _compute_thresholds_payload(
        "balanced", {"variance": {"min_effect_lognll": 0.5}}
    )
    assert thr["tier"] == "balanced"
    assert thr["pm_ratio"]["ratio_limit_base"] == TIER_RATIO_LIMITS["balanced"]
