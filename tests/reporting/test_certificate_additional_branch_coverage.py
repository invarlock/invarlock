from __future__ import annotations

from invarlock.reporting.certificate import (
    _is_ppl_kind,
    make_certificate,
    validate_certificate,
)
from invarlock.reporting.report_types import RunReport, create_empty_report


def _mk_report(
    *, replicates: int = 50, bootstrap_method: str = "percentile"
) -> RunReport:
    r = create_empty_report()
    r["meta"]["model_id"] = "m"
    r["meta"]["adapter"] = "hf"
    r["meta"]["device"] = "cpu"
    r["meta"]["auto"] = {"tier": "balanced", "probes_used": 0, "target_pm_ratio": None}  # type: ignore[assignment]
    r["data"]["dataset"] = "unit"
    r["data"]["split"] = "validation"
    r["data"]["seq_len"] = 8
    r["data"]["stride"] = 8
    r["data"]["preview_n"] = 2
    r["data"]["final_n"] = 2
    r["metrics"]["primary_metric"] = {
        "kind": "ppl_causal",
        "preview": 10.0,
        "final": 10.0,
    }
    r["metrics"]["bootstrap"] = {
        "method": bootstrap_method,
        "replicates": replicates,
        "alpha": 0.05,
        "seed": 0,
        "coverage": {"preview": {"used": 2}, "final": {"used": 2}},
    }
    r["metrics"]["paired_delta_summary"] = {"mean": 0.0}
    r["metrics"]["preview_total_tokens"] = 50
    r["metrics"]["final_total_tokens"] = 50
    r["metrics"]["logloss_delta"] = 0.0
    r["metrics"]["logloss_delta_ci"] = (-0.01, 0.01)
    r["evaluation_windows"] = {
        "final": {
            "window_ids": [1, 2],
            "logloss": [2.30, 2.31],
            "token_counts": [100, 100],
        }
    }
    return r


def _mk_baseline() -> dict:
    return {
        "run_id": "base",
        "model_id": "m",
        "meta": {"seed": 0, "model_id": "m"},
        "evaluation_windows": {
            "final": {
                "window_ids": [1, 2],
                "logloss": [2.30, 2.30],
                "token_counts": [100, 100],
            }
        },
        "data": {
            "seq_len": 8,
            "preview_n": 2,
            "final_n": 2,
            "dataset": "unit",
            "split": "validation",
            "stride": 8,
        },
        "edit": {
            "name": "none",
            "plan_digest": "0",
            "deltas": {
                "params_changed": 0,
                "layers_modified": 0,
                "sparsity": None,
                "bitwidth_map": None,
            },
        },
        "guards": [],
        "metrics": {"primary_metric": {"kind": "ppl_causal", "final": 10.0}},
        "artifacts": {"events_path": "", "logs_path": "", "checkpoint_path": None},
        "flags": {"guard_recovered": False, "rollback_reason": None},
    }


def test_is_ppl_kind_accepts_aliases_and_handles_bad_str() -> None:
    assert _is_ppl_kind("ppl")
    assert _is_ppl_kind("perplexity")
    assert _is_ppl_kind("ppl_mlm")
    assert not _is_ppl_kind("accuracy")

    class _Bad:
        def __str__(self) -> str:
            raise RuntimeError("boom")

    assert not _is_ppl_kind(_Bad())


def test_make_certificate_replicates_zero_keeps_run_metrics_pairing_and_fills_counts(
    monkeypatch,
) -> None:
    report = _mk_report(replicates=0)
    baseline = _mk_baseline()

    # Force _as_count float/negative branches; should fall back to data.preview_n/final_n.
    report["metrics"]["stats"] = {"requested_preview": 1.2, "requested_final": -1}

    monkeypatch.setattr(
        "invarlock.reporting.certificate.compute_paired_delta_log_ci",
        lambda *_a, **_k: (_a, _k),  # should not be called when replicates=0
    )
    cert = make_certificate(report, baseline)
    assert validate_certificate(cert)
    stats = cert["dataset"]["windows"]["stats"]
    assert stats["pairing"] == "run_metrics"
    assert stats["requested_preview"] == 2
    assert stats["requested_final"] == 2


def test_make_certificate_uses_bca_when_method_explicit(monkeypatch) -> None:
    report = _mk_report(replicates=10, bootstrap_method="bca")
    baseline = _mk_baseline()

    seen: dict[str, object] = {}

    def _fake_ci(*_a, method: str, **_k):  # noqa: ANN001
        seen["method"] = method
        return (-0.01, 0.01)

    monkeypatch.setattr(
        "invarlock.reporting.certificate.compute_paired_delta_log_ci", _fake_ci
    )
    cert = make_certificate(report, baseline)
    assert validate_certificate(cert)
    assert seen.get("method") == "bca"


def test_make_certificate_env_bca_flag_ignored_when_windows_small(monkeypatch) -> None:
    report = _mk_report(replicates=10, bootstrap_method="percentile")
    baseline = _mk_baseline()
    monkeypatch.setenv("INVARLOCK_BOOTSTRAP_BCA", "1")

    seen: dict[str, object] = {}

    def _fake_ci(*_a, method: str, **_k):  # noqa: ANN001
        seen["method"] = method
        return (-0.01, 0.01)

    monkeypatch.setattr(
        "invarlock.reporting.certificate.compute_paired_delta_log_ci", _fake_ci
    )
    cert = make_certificate(report, baseline)
    assert validate_certificate(cert)
    assert seen.get("method") == "percentile"


def test_make_certificate_marks_unstable_when_token_floor_violated(monkeypatch) -> None:
    report = _mk_report(replicates=200)
    baseline = _mk_baseline()
    report["metrics"]["preview_total_tokens"] = 10
    report["metrics"]["final_total_tokens"] = 10

    monkeypatch.setattr(
        "invarlock.reporting.certificate.get_tier_policies",
        lambda: {"balanced": {"metrics": {"pm_ratio": {"min_tokens": 100}}}},
    )
    monkeypatch.setattr(
        "invarlock.reporting.certificate.compute_paired_delta_log_ci",
        lambda *_a, **_k: (-0.01, 0.01),
    )
    cert = make_certificate(report, baseline)
    assert validate_certificate(cert)
    assert bool(cert["primary_metric"]["unstable"]) is True
