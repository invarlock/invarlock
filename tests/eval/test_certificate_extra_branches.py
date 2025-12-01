from invarlock.reporting.certificate import make_certificate
from invarlock.reporting.policy_utils import _resolve_policy_tier
from invarlock.reporting.utils import _infer_scope_from_modules


def test_infer_scope_from_modules_variations():
    assert _infer_scope_from_modules([]) == "unknown"
    mix = ["layer.attn.q_proj", "embeddings.wte"]
    assert _infer_scope_from_modules(mix) in {"attn+embed", "embed+attn"}


def test_resolve_policy_tier_from_context_auto():
    report = {"context": {"auto": {"tier": "Conservative"}}}
    assert _resolve_policy_tier(report) == "conservative"


def test_make_certificate_invalid_preview_only_no_longer_raises(monkeypatch):
    # Invalid preview but valid final no longer raises after normalization
    report = {
        "meta": {"model_id": "m", "seed": 1},
        "metrics": {"ppl_preview": 1.0, "ppl_final": 10.0},
        "data": {
            "dataset": "d",
            "split": "val",
            "seq_len": 8,
            "stride": 1,
            "preview_n": 1,
            "final_n": 1,
        },
        "guards": [],
        "edit": {
            "name": "mock",
            "deltas": {
                "params_changed": 0,
                "heads_pruned": 0,
                "neurons_pruned": 0,
                "layers_modified": 0,
            },
        },
        "evaluation_windows": {"final": {"window_ids": [1, 2], "logloss": [0.1, 0.2]}},
    }
    baseline = {
        "meta": {"model_id": "m"},
        "metrics": {"ppl_final": 10.2, "ppl_preview": 10.1},
    }
    # Bypass schema rigor to focus on branch
    monkeypatch.setattr(
        "invarlock.reporting.certificate.validate_report", lambda _: True
    )
    cert = make_certificate(report, baseline)
    assert isinstance(cert, dict)
