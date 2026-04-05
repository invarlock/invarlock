from __future__ import annotations

import math
from copy import deepcopy

import pytest

import invarlock.eval.primary_metric as primary_metric_mod
from invarlock.core.exceptions import MetricsError, ValidationError
from invarlock.reporting import (
    dataset_hashing,
    guards_invariants,
    guards_rmt,
    guards_spectral,
    guards_variance,
    policy_utils,
    primary_metric_utils,
    report_edit_summary,
    report_normalization,
)
from invarlock.reporting import (
    report_make as cert,
)
from invarlock.reporting import (
    report_primary_metric_analysis as pm_analysis_mod,
)
from invarlock.reporting import utils as report_utils
from invarlock.reporting.report_make import make_report


def _base_report() -> dict:
    return {
        "run_id": "run-1",
        "meta": {
            "model_id": "demo-model",
            "adapter": "hf",
            "device": "cpu",
            "seed": 1,
            "auto": {"tier": "balanced", "probes_used": 0, "target_pm_ratio": None},
        },
        "data": {
            "dataset": "demo-ds",
            "split": "eval",
            "seq_len": 8,
            "stride": 4,
            "preview_n": 2,
            "final_n": 2,
            "windows": {"preview": 2, "final": 2},
        },
        "artifacts": {"events_path": "", "logs_path": "", "generated_at": ""},
        "guards": [],
        "guard_overhead": {},
        "edit": {
            "name": "baseline",
            "plan_digest": "baseline_noop",
            "deltas": {"params_changed": 0},
        },
        "metrics": {
            "primary_metric": {
                "kind": "ppl_causal",
                "preview": 10.0,
                "final": 10.0,
                "ratio_vs_baseline": 1.0,
                "display_ci": [10.0, 10.0],
            },
            "paired_delta_summary": {"mean": 0.0, "degenerate": False},
            "logloss_delta_ci": (0.0, 0.0),
            "bootstrap": {"replicates": 400, "coverage": {"preview": {"used": 0}}},
            "window_plan": {"profile": "dev"},
            "spectral": {"caps_applied": 0, "max_caps": 5, "summary": {}},
            "rmt": {"stable": True},
            "variance": {"enabled": False},
            "window_overlap_fraction": 0.0,
            "window_match_fraction": 1.0,
        },
        "evaluation_windows": {
            "preview": {
                "window_ids": [1, 2],
                "logloss": [0.1, 0.2],
                "token_counts": [10, 12],
            },
            "final": {
                "window_ids": [1, 2],
                "logloss": [0.15, 0.25],
                "token_counts": [10, 12],
            },
        },
    }


def _base_baseline() -> dict:
    base = _base_report()
    base["run_id"] = "base-1"
    base["metrics"]["primary_metric"]["final"] = 10.0
    return base


def _patch_common(monkeypatch, report, baseline):
    monkeypatch.setattr(
        report_normalization,
        "normalize_and_validate_run_report",
        lambda _r: report,
        raising=False,
    )
    monkeypatch.setattr(
        report_normalization, "normalize_baseline", lambda _b: baseline, raising=False
    )
    monkeypatch.setattr(
        primary_metric_utils,
        "attach_primary_metric",
        lambda *args, **kwargs: None,
        raising=False,
    )
    monkeypatch.setattr(
        primary_metric_mod,
        "compute_primary_metric_from_report",
        lambda *args, **kwargs: {},
    )


def test_make_report_wraps_baseline_normalization_errors(monkeypatch) -> None:
    report = _base_report()

    monkeypatch.setattr(
        report_normalization,
        "normalize_and_validate_run_report",
        lambda _r: report,
        raising=False,
    )
    monkeypatch.setattr(
        report_normalization,
        "normalize_baseline",
        lambda _b: (_ for _ in ()).throw(
            ValueError("invalid baseline metrics.ppl_final")
        ),
        raising=False,
    )

    with pytest.raises(ValidationError, match="Baseline normalization failed"):
        make_report(report, _base_baseline())


def _stub_evaluation_report_extractors(
    monkeypatch,
    *,
    dataset_info=None,
    invariants=None,
    spectral=None,
    rmt=None,
    variance=None,
    structure=None,
    policies_payload=None,
    resolved_policy=None,
):
    dataset_info = dataset_info or {"hash": {}, "windows": {}}
    invariants = invariants or {"status": "ok"}
    spectral = spectral or {"caps_applied": 0}
    rmt = rmt or {"stable": True}
    variance = variance or {"enabled": False}
    structure = structure or {
        "compression_diagnostics": {"execution_status": "successful"}
    }
    policies_payload = policies_payload or {}
    resolved_policy = resolved_policy or {"spectral": {}, "variance": {}}

    monkeypatch.setattr(
        dataset_hashing, "_extract_dataset_info", lambda *_: deepcopy(dataset_info)
    )
    monkeypatch.setattr(
        guards_invariants,
        "_extract_invariants",
        lambda *args, **kwargs: invariants,
        raising=False,
    )
    monkeypatch.setattr(
        guards_spectral,
        "_extract_spectral_analysis",
        lambda *_: spectral,
        raising=False,
    )
    monkeypatch.setattr(
        guards_rmt, "_extract_rmt_analysis", lambda *_: rmt, raising=False
    )
    monkeypatch.setattr(
        guards_variance,
        "_extract_variance_analysis",
        lambda *_: variance,
        raising=False,
    )
    monkeypatch.setattr(
        report_edit_summary,
        "extract_structural_deltas",
        lambda *_: deepcopy(structure),
        raising=False,
    )
    monkeypatch.setattr(
        policy_utils,
        "_extract_effective_policies",
        lambda *_: deepcopy(policies_payload),
        raising=False,
    )
    monkeypatch.setattr(
        policy_utils, "_extract_policy_overrides", lambda *_: ["manual"], raising=False
    )
    monkeypatch.setattr(
        policy_utils,
        "_build_resolved_policies",
        lambda *args, **kwargs: deepcopy(resolved_policy),
        raising=False,
    )
    monkeypatch.setattr(
        policy_utils,
        "_compute_policy_digest",
        lambda *_: "resolved-digest",
        raising=False,
    )


def test_make_evaluation_report_raises_on_drift_identity(monkeypatch):
    report = _base_report()
    baseline = _base_baseline()
    report["metrics"]["primary_metric"].update(
        {"preview": 10.0, "final": 11.0, "ratio_vs_baseline": 1.1}
    )
    report["metrics"]["paired_delta_summary"]["mean"] = math.log(1.6)
    report["metrics"]["window_plan"]["profile"] = "ci"

    _patch_common(monkeypatch, report, baseline)

    def fake_pair(_run, _base):
        return ([0.1, 0.2], [0.0, 0.0])

    monkeypatch.setattr(report_utils, "_pair_logloss_windows", fake_pair)
    monkeypatch.setattr(
        "invarlock.core.bootstrap.compute_paired_delta_log_ci",
        lambda *args, **kwargs: (0.0, 0.0),
    )
    monkeypatch.setattr(
        pm_analysis_mod,
        "enforce_drift_ratio_identity",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            ValueError("Paired ΔlogNLL mean is inconsistent with reported drift ratio")
        ),
        raising=False,
    )

    with pytest.raises(ValueError, match="drift ratio"):
        make_report({}, {})


def test_make_evaluation_report_raises_on_ratio_ci_mismatch(monkeypatch):
    report = _base_report()
    baseline = _base_baseline()
    report["metrics"]["window_plan"]["profile"] = "ci"
    report["metrics"]["paired_delta_summary"]["mean"] = 0.0

    _patch_common(monkeypatch, report, baseline)

    def fake_pair(_run, _base):
        return ([0.1, 0.2], [0.1, 0.2])

    monkeypatch.setattr(report_utils, "_pair_logloss_windows", fake_pair)
    monkeypatch.setattr(
        "invarlock.core.bootstrap.compute_paired_delta_log_ci",
        lambda *args, **kwargs: (0.0, 0.0),
    )
    monkeypatch.setattr(
        pm_analysis_mod,
        "enforce_ratio_ci_alignment",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("CI mismatch")),
        raising=False,
    )

    with pytest.raises(ValueError, match="CI mismatch"):
        make_report({}, {})


def test_make_evaluation_report_uses_coverage_fallback(monkeypatch):
    report = _base_report()
    baseline = _base_baseline()
    report["metrics"]["bootstrap"]["coverage"] = {"preview": {"used": 5}}
    report["evaluation_windows"] = {}

    _patch_common(monkeypatch, report, baseline)

    evaluation_report = make_report(report, baseline)
    stats = evaluation_report["dataset"]["windows"]["stats"]
    assert stats["paired_windows"] == 5


def test_make_report_surfaces_baseline_failures_explicitly(monkeypatch) -> None:
    class _BrokenMeta(dict):
        def get(self, key, default=None):  # noqa: ANN001
            if key in {"env_flags", "determinism"}:
                raise RuntimeError(f"{key}-bad")
            return super().get(key, default)

    class _BrokenContext(dict):
        def get(self, key, default=None):  # noqa: ANN001
            if key == "profile":
                raise RuntimeError("profile-bad")
            return super().get(key, default)

    class _BrokenWindows(dict):
        def setdefault(self, key, default=None):  # noqa: ANN001
            raise RuntimeError(f"{key}-bad")

    report = _base_report()
    baseline = _base_baseline()
    baseline["metrics"].pop("primary_metric", None)
    report["meta"] = _BrokenMeta(report["meta"])
    report["context"] = _BrokenContext({"profile": "ci"})
    _patch_common(monkeypatch, report, baseline)
    _stub_evaluation_report_extractors(
        monkeypatch,
        dataset_info={"hash": {}, "windows": _BrokenWindows()},
    )
    monkeypatch.setattr(
        cert,
        "_extract_dataset_info",
        lambda *_args, **_kwargs: {"hash": {}, "windows": _BrokenWindows()},
        raising=False,
    )
    monkeypatch.setattr(
        cert.report_primary_metric_analysis_mod,
        "build_primary_metric_analysis",
        lambda *_args, **_kwargs: ({}, "dev"),
        raising=False,
    )

    monkeypatch.setattr(
        cert.report_normalization_mod,
        "normalize_and_validate_run_report",
        lambda payload: (_ for _ in ()).throw(RuntimeError("baseline-bad"))
        if payload is baseline
        else report,
        raising=False,
    )
    monkeypatch.setattr(
        cert,
        "_extract_report_meta",
        lambda _report, *_args, **_kwargs: {
            "model_id": "demo-model",
            "adapter": "hf",
            "device": "cpu",
        },
        raising=False,
    )
    monkeypatch.setattr(
        cert,
        "compute_primary_metric_from_report",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("pm-bad")),
        raising=False,
    )

    with pytest.raises(ValidationError, match="requires a valid baseline report"):
        make_report(report, baseline)

    monkeypatch.setattr(
        cert.report_normalization_mod,
        "normalize_and_validate_run_report",
        lambda payload: report,
        raising=False,
    )
    with pytest.raises(MetricsError, match="requires a concrete baseline metric"):
        make_report(report, baseline)


def test_make_report_rejects_non_finite_baseline_primary_metric(monkeypatch) -> None:
    report = _base_report()
    baseline = _base_baseline()
    baseline["metrics"]["primary_metric"] = {"kind": "ppl_causal", "final": "bad"}

    _patch_common(monkeypatch, report, baseline)

    with pytest.raises(MetricsError, match="concrete finite `final` value"):
        make_report(report, baseline)


def test_make_evaluation_report_populates_optional_sections(monkeypatch):
    report = _base_report()
    baseline = _base_baseline()

    report["metrics"]["secondary_metrics"] = [
        {
            "kind": "accuracy",
            "preview": 0.8,
            "final": 0.82,
            "ratio_vs_baseline": 0.02,
            "display_ci": [0.8, 0.85],
            "ci": [0.8, 0.86],
            "unit": "pp",
        },
        {"kind": None},
    ]
    report["metrics"]["classification"] = {
        "subgroups": {
            "preview": {
                "group_counts": {"alpha": 10},
                "correct_counts": {"alpha": 8},
            },
            "final": {"group_counts": {"alpha": 10}, "correct_counts": {"alpha": 9}},
        }
    }
    report["metrics"]["window_capacity"] = {"total_tokens": 200, "available_unique": 6}
    report["metrics"]["moe"] = {
        "top_k": 2,
        "capacity_factor": 1.5,
        "utilization": [0.7, 0.9],
    }
    baseline["metrics"]["moe"] = {
        "utilization": [0.6, 0.6],
        "load_balance_loss": 0.2,
        "router_entropy": 0.3,
    }
    report["metrics"]["latency_ms_per_tok"] = 4.0
    report["metrics"]["throughput_tok_per_s"] = 90.0
    baseline["metrics"]["latency_ms_per_tok"] = 5.0
    baseline["metrics"]["throughput_tok_per_s"] = 80.0
    report["metrics"]["edge_device"] = {"provider": "demo-edge"}
    report["metrics"]["masked_tokens_total"] = 100
    report["metrics"]["masked_tokens_preview"] = 40
    report["metrics"]["masked_tokens_final"] = 60
    report["guard_overhead"] = {
        "bare_ppl": 10.0,
        "guarded_ppl": 10.5,
        "warnings": ["slow"],
        "messages": ["note"],
        "checks": {"ratio": True},
    }
    report["provenance"] = {"dataset_split": "eval", "split_fallback": True}
    report["artifacts"]["masks_path"] = "/tmp/masks.bin"
    baseline["artifacts"]["report_path"] = "/tmp/base.json"
    report["meta"]["plugins"] = {"edit": {"module": "demo", "version": "1"}}
    report["guards"] = [
        {"name": "variance", "policy": {"min_effect_lognll": 0.2, "topk_backstop": 3}}
    ]

    _patch_common(monkeypatch, report, baseline)

    def _attach_pm_stub(evaluation_report: dict, *_args, **_kwargs) -> None:
        evaluation_report["primary_metric"] = {
            "kind": "ppl_causal",
            "ratio_vs_baseline": 1.02,
            "display_ci": (1.0, 1.04),
        }

    dataset_info = {
        "hash": {"total_tokens": 130},
        "windows": {"preview": 2, "final": 2, "stats": {}},
    }
    invariants = {"status": "ok"}
    spectral = {"caps_applied": 1}
    rmt = {"stable": True}
    variance = {"enabled": True}
    structure = {"compression_diagnostics": {"execution_status": "successful"}}
    policies_payload = {"variance": {"enabled": True}}
    resolved_policy = {
        "spectral": {"sigma_quantile": 2.0},
        "variance": {"min_effect_lognll": 0.1},
    }

    monkeypatch.setattr(
        primary_metric_utils, "attach_primary_metric", _attach_pm_stub, raising=False
    )
    monkeypatch.setattr(
        dataset_hashing, "_extract_dataset_info", lambda *_: deepcopy(dataset_info)
    )
    monkeypatch.setattr(
        guards_invariants,
        "_extract_invariants",
        lambda *args, **kwargs: invariants,
        raising=False,
    )
    monkeypatch.setattr(
        guards_spectral,
        "_extract_spectral_analysis",
        lambda *_: spectral,
        raising=False,
    )
    monkeypatch.setattr(
        guards_rmt, "_extract_rmt_analysis", lambda *_: rmt, raising=False
    )
    monkeypatch.setattr(
        guards_variance,
        "_extract_variance_analysis",
        lambda *_: variance,
        raising=False,
    )
    monkeypatch.setattr(
        report_edit_summary,
        "extract_structural_deltas",
        lambda *_: deepcopy(structure),
        raising=False,
    )
    monkeypatch.setattr(
        policy_utils,
        "_extract_effective_policies",
        lambda *_: deepcopy(policies_payload),
        raising=False,
    )
    monkeypatch.setattr(
        policy_utils,
        "_extract_policy_overrides",
        lambda *_: ["manual-limit"],
        raising=False,
    )
    monkeypatch.setattr(
        policy_utils,
        "_build_resolved_policies",
        lambda *args, **kwargs: resolved_policy,
        raising=False,
    )
    monkeypatch.setattr(
        policy_utils,
        "_compute_policy_digest",
        lambda *_: "resolved-digest",
        raising=False,
    )

    evaluation_report = make_report(report, baseline)

    assert evaluation_report["secondary_metrics"][0]["kind"] == "accuracy"
    subgroup = evaluation_report["classification"]["subgroups"]["alpha"]
    assert subgroup["delta_pp"] == pytest.approx(10.0)
    assert evaluation_report["guard_overhead"]["evaluated"] is True
    assert evaluation_report["guard_overhead"]["passed"] is False
    assert evaluation_report["system_overhead"]["latency_ms_p50"][
        "ratio"
    ] == pytest.approx(0.8)
    assert evaluation_report["policy_digest"]["policy_version"] == cert.POLICY_VERSION
    assert evaluation_report["telemetry"]["summary_line"].startswith(
        "INVARLOCK_TELEMETRY"
    )


def test_make_evaluation_report_populates_dataset_stats_when_absent(monkeypatch):
    report = _base_report()
    baseline = _base_baseline()
    _patch_common(monkeypatch, report, baseline)

    dataset_stub = {"hash": {}, "windows": {}}
    _stub_evaluation_report_extractors(
        monkeypatch,
        dataset_info=dataset_stub,
        resolved_policy={"spectral": {}, "variance": {}},
    )

    evaluation_report = make_report(report, baseline)
    stats = evaluation_report["dataset"]["windows"]["stats"]
    assert "pairing" in stats
    assert stats["paired_windows"] >= 1


def test_make_evaluation_report_policy_digest_marks_tier_change(monkeypatch):
    report = _base_report()
    baseline = _base_baseline()
    report["meta"]["auto"]["tier"] = "balanced"
    baseline["meta"]["auto"] = {"tier": "conservative"}

    _patch_common(monkeypatch, report, baseline)
    _stub_evaluation_report_extractors(
        monkeypatch,
        dataset_info={"hash": {}, "windows": {"stats": {}}},
        resolved_policy={"spectral": {}, "variance": {}},
    )

    evaluation_report = make_report(report, baseline)
    assert evaluation_report["policy_digest"]["changed"] is True


def test_make_evaluation_report_policy_digest_handles_missing_baseline_tier(
    monkeypatch,
):
    report = _base_report()
    baseline = _base_baseline()
    report["meta"]["auto"]["tier"] = "balanced"
    baseline["meta"].pop("auto", None)

    _patch_common(monkeypatch, report, baseline)
    _stub_evaluation_report_extractors(
        monkeypatch,
        dataset_info={"hash": {}, "windows": {"stats": {}}},
        resolved_policy={"spectral": {}, "variance": {}},
    )

    evaluation_report = make_report(report, baseline)
    assert evaluation_report["policy_digest"]["changed"] is False


def test_make_evaluation_report_policy_digest_detects_threshold_hash_change(
    monkeypatch,
):
    report = _base_report()
    baseline = _base_baseline()
    _patch_common(monkeypatch, report, baseline)
    _stub_evaluation_report_extractors(
        monkeypatch,
        dataset_info={"hash": {}, "windows": {"stats": {}}},
        resolved_policy={"spectral": {}, "variance": {}},
    )

    call_state = {"count": 0}

    def fake_payload(tier, resolved):
        call_state["count"] += 1
        return {"tier": tier, "call": call_state["count"]}

    def fake_hash(payload):
        return f"{payload['tier']}-{payload['call']}"

    monkeypatch.setattr(
        policy_utils, "_compute_thresholds_payload", fake_payload, raising=False
    )
    monkeypatch.setattr(
        policy_utils, "_compute_thresholds_hash", fake_hash, raising=False
    )

    evaluation_report = make_report(report, baseline)
    assert evaluation_report["policy_digest"]["changed"] is True


def test_make_evaluation_report_copies_meta_environment_flags(monkeypatch):
    report = _base_report()
    baseline = _base_baseline()
    report["meta"]["env_flags"] = {"tf32": False}
    report["data"]["tokenizer_hash"] = "tok-data"
    report["meta"]["model_profile"] = {"arch": "gpt"}
    report["meta"]["cuda_flags"] = {"tf32": True}

    monkeypatch.setattr(
        report_normalization,
        "normalize_and_validate_run_report",
        lambda value: value,
        raising=False,
    )
    monkeypatch.setattr(
        report_normalization, "normalize_baseline", lambda value: value, raising=False
    )

    evaluation_report = make_report(report, baseline)
    meta = evaluation_report["meta"]
    assert meta["env_flags"]["tf32"] is False
    assert meta["tokenizer_hash"] == "tok-data"
    assert meta["model_profile"]["arch"] == "gpt"
    assert meta["cuda_flags"]["tf32"] is True


def test_make_evaluation_report_uses_meta_tokenizer_hash(monkeypatch):
    report = _base_report()
    baseline = _base_baseline()
    report["meta"]["tokenizer_hash"] = "tok-meta"

    monkeypatch.setattr(
        report_normalization,
        "normalize_and_validate_run_report",
        lambda value: value,
        raising=False,
    )
    monkeypatch.setattr(
        report_normalization, "normalize_baseline", lambda value: value, raising=False
    )

    evaluation_report = make_report(report, baseline)
    assert evaluation_report["meta"]["tokenizer_hash"] == "tok-meta"


def test_make_evaluation_report_handles_missing_dataset_section(monkeypatch):
    report = _base_report()
    baseline = _base_baseline()
    report["meta"].pop("tokenizer_hash", None)
    report["data"] = None

    monkeypatch.setattr(
        report_normalization,
        "normalize_and_validate_run_report",
        lambda value: value,
        raising=False,
    )
    monkeypatch.setattr(
        report_normalization, "normalize_baseline", lambda value: value, raising=False
    )
    monkeypatch.setattr(
        dataset_hashing,
        "_extract_dataset_info",
        lambda *_: {"hash": {}, "windows": {}},
    )

    evaluation_report = make_report(report, baseline)
    assert "tokenizer_hash" not in evaluation_report["meta"]


def test_make_evaluation_report_preserves_nullable_provenance(monkeypatch):
    report = _base_report()
    baseline = _base_baseline()
    report["meta"]["model_id"] = None
    report["meta"]["adapter"] = ""
    report["meta"]["device"] = None
    report["edit"]["name"] = None
    baseline["meta"].pop("model_id", None)
    baseline.pop("model_id", None)
    baseline.pop("run_id", None)

    _patch_common(monkeypatch, report, baseline)
    _stub_evaluation_report_extractors(
        monkeypatch,
        dataset_info={"hash": {}, "windows": {"stats": {}}},
        resolved_policy={"spectral": {}, "variance": {}},
    )

    evaluation_report = make_report(report, baseline)

    assert evaluation_report["meta"]["model_id"] is None
    assert evaluation_report["meta"]["adapter"] is None
    assert evaluation_report["meta"]["device"] is None
    assert evaluation_report["edit_name"] is None
    assert evaluation_report["edit"]["name"] is None
    assert evaluation_report["baseline_ref"]["model_id"] is None
    assert evaluation_report["baseline_ref"]["run_id"] is None
    diagnostics = evaluation_report["meta"].get("build_diagnostics", [])
    codes = {entry["code"] for entry in diagnostics}
    assert {
        "meta.model_id_unavailable",
        "meta.adapter_unavailable",
        "meta.device_unavailable",
    }.issubset(codes)
