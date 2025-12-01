from __future__ import annotations

from invarlock.reporting import certificate as C


def _base_cert() -> dict:
    return {
        "schema_version": C.CERTIFICATE_SCHEMA_VERSION,
        "run_id": "r",
        "artifacts": {"generated_at": "t"},
        "meta": {},
        "dataset": {
            "provider": "p",
            "seq_len": 8,
            "windows": {"preview": 1, "final": 1},
        },
        "validation": {"primary_metric_acceptable": True},
        "primary_metric": {"kind": "ppl_causal", "display_ci": [0.99, 1.00]},
    }


def test_confidence_label_high_medium_low_ppl() -> None:
    cert = _base_cert()
    # High: narrow width, acceptable, not unstable
    c1 = dict(cert)
    c1["primary_metric"] = {"kind": "ppl_causal", "display_ci": [0.99, 1.00]}
    out1 = C._compute_confidence_label(c1)
    assert out1["label"] in {"High", "Medium", "Low"}

    # Medium: borderline width and unstable
    c2 = _base_cert()
    c2["primary_metric"] = {
        "kind": "ppl_causal",
        "display_ci": [0.99, 1.02],
        "unstable": True,
    }
    out2 = C._compute_confidence_label(c2)
    assert out2["label"] in {"High", "Medium", "Low"}

    # Low: very wide interval
    c3 = _base_cert()
    c3["primary_metric"] = {"kind": "ppl_causal", "display_ci": [0.90, 1.10]}
    out3 = C._compute_confidence_label(c3)
    assert out3["label"] in {"High", "Medium", "Low"}


def test_confidence_label_accuracy_basis() -> None:
    cert = _base_cert()
    cert["primary_metric"] = {"kind": "accuracy", "display_ci": [0.70, 0.71]}
    out = C._compute_confidence_label(cert)
    assert out["basis"] in {"accuracy", "vqa_accuracy", "ppl_ratio"}
