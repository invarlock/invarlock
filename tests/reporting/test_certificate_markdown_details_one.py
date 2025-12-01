from __future__ import annotations

from invarlock.reporting.certificate import make_certificate
from invarlock.reporting.render import render_certificate_markdown


def _mk_base_report(commit: str | None) -> dict:
    meta = {"model_id": "m", "adapter": "hf", "device": "cpu", "seed": 1}
    if commit is not None:
        meta["commit"] = commit
    return {
        "meta": meta,
        "data": {
            "dataset": "ds",
            "split": "val",
            "seq_len": 8,
            "stride": 8,
            "preview_n": 1,
            "final_n": 1,
        },
        "edit": {
            "name": "noop",
            "plan_digest": "d",
            "deltas": {"params_changed": 0, "layers_modified": 0},
        },
        "guards": [],
        "metrics": {
            "primary_metric": {
                "kind": "ppl_causal",
                "preview": 10.0,
                "final": 10.0,
                "ratio_vs_baseline": 1.0,
                "display_ci": [1.0, 1.0],
            },
            # Secondary metrics without CI to trigger dash rendering
            "secondary": [
                {
                    "kind": "latency_ms_p50",
                    "preview": 1.2,
                    "final": 1.1,
                    "ratio_vs_baseline": 0.92,
                }
            ],
        },
        "evaluation_windows": {
            "preview": {
                "window_ids": [1],
                "logloss": [2.302585093],
                "token_counts": [1],
            },
            "final": {"window_ids": [2], "logloss": [2.302585093], "token_counts": [1]},
        },
        "artifacts": {"events_path": "", "logs_path": ""},
    }


def _mk_baseline() -> dict:
    return {
        "meta": {"auto": {"tier": "balanced"}},
        "metrics": {"primary_metric": {"kind": "ppl_causal", "final": 10.0}},
    }


def test_markdown_commit_present_and_absent() -> None:
    rep1 = _mk_base_report(commit="abcdef0123456789deadbeef")
    base = _mk_baseline()
    cert1 = make_certificate(rep1, base)
    md1 = render_certificate_markdown(cert1)
    assert "Commit:" in md1

    # Still renders Model Information section when commit absent
    rep2 = _mk_base_report(commit=None)
    cert2 = make_certificate(rep2, base)
    md2 = render_certificate_markdown(cert2)
    assert "Model Information" in md2


def test_markdown_tokenizer_add_prefix_space_and_secondary_metrics_dash() -> None:
    rep = _mk_base_report(commit="a1b2c3d4")
    base = _mk_baseline()
    cert = make_certificate(rep, base)

    # Inject tokenizer with add_prefix_space True and no PAD
    cert.setdefault("dataset", {}).setdefault("tokenizer", {})
    cert["dataset"]["tokenizer"].update(
        {
            "name": "tok",
            "hash": "h",
            "vocab_size": 10,
            "bos_token": "^",
            "eos_token": "$",
            "add_prefix_space": True,
        }
    )
    # Inject secondary_metrics in certificate surface for markdown path
    cert["secondary_metrics"] = [
        {"kind": "accuracy", "preview": 0.7, "final": 0.71, "ratio_vs_baseline": +0.01}
    ]  # no display_ci → dash branch

    md = render_certificate_markdown(cert)
    assert "Tokenizer" in md and "add_prefix_space" in md
    assert (
        "Secondary Metrics" in md and "| Δ vs Baseline |" not in md
    )  # non-accuracy table alt; ensure section exists
