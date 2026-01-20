from __future__ import annotations

from copy import deepcopy

import pytest

from invarlock.reporting.certificate import make_certificate
from invarlock.reporting.report_types import create_empty_report


def _write_tiers_yaml(root, *, ratio_limit_base: float) -> None:
    runtime = root / "runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    tiers_path = runtime / "tiers.yaml"
    tiers_path.write_text(
        f"""
balanced:
  metrics:
    pm_ratio:
      ratio_limit_base: {ratio_limit_base}
      hysteresis_ratio: 0.0
      min_tokens: 0
      min_token_fraction: 0.0
    accuracy:
      delta_min_pp: -1.0
      hysteresis_delta_pp: 0.0
      min_examples: 0
      min_examples_fraction: 0.0
  spectral_guard:
    sigma_quantile: 0.95
    deadband: 0.10
    scope: all
    max_caps: 5
    max_spectral_norm: null
    family_caps:
      ffn: 3.0
      attn: 3.0
      embed: 3.0
      other: 3.0
    multiple_testing:
      method: bh
      alpha: 0.05
      m: 4
  rmt_guard:
    deadband: 0.10
    margin: 1.5
    epsilon_default: 0.10
    epsilon_by_family:
      ffn: 0.10
      attn: 0.08
      embed: 0.12
      other: 0.12
  variance_guard:
    deadband: 0.02
    min_abs_adjust: 0.012
    max_scale_step: 0.03
    min_effect_lognll: 0.0009
    predictive_one_sided: true
    topk_backstop: 1
    max_adjusted_modules: 1
    predictive_gate: true
""".lstrip(),
        encoding="utf-8",
    )


def _make_min_report(*, tier: str, ratio_vs_baseline: float) -> dict:
    report = create_empty_report()
    report["meta"]["model_id"] = "m"
    report["meta"]["adapter"] = "hf_causal"
    report["meta"]["commit"] = "deadbeef"
    report["meta"]["seed"] = 1
    report["meta"]["auto"] = {"enabled": True, "tier": tier}
    report["data"].update(
        {"dataset": "d", "split": "validation", "seq_len": 8, "stride": 8}
    )
    report["metrics"]["primary_metric"] = {
        "kind": "ppl_causal",
        "preview": 40.0,
        "final": 40.0 * float(ratio_vs_baseline),
        "ratio_vs_baseline": float(ratio_vs_baseline),
        "display_ci": [float(ratio_vs_baseline), float(ratio_vs_baseline)],
    }
    report["evaluation_windows"] = {"final": {"window_ids": [1], "logloss": [0.1]}}
    report["guards"] = []
    return report


def test_tiers_yaml_changes_gate_resolved_policy_and_digest(
    tmp_path, monkeypatch
) -> None:
    case_loose = tmp_path / "loose"
    case_tight = tmp_path / "tight"
    _write_tiers_yaml(case_loose, ratio_limit_base=1.15)
    _write_tiers_yaml(case_tight, ratio_limit_base=1.10)

    report = _make_min_report(tier="balanced", ratio_vs_baseline=1.12)
    baseline = {
        "run_id": "b",
        "model_id": "m",
        "evaluation_windows": {"final": {"window_ids": [1], "logloss": [0.1]}},
    }

    monkeypatch.setenv("INVARLOCK_CONFIG_ROOT", str(case_loose))
    cert_loose = make_certificate(deepcopy(report), deepcopy(baseline))

    monkeypatch.setenv("INVARLOCK_CONFIG_ROOT", str(case_tight))
    cert_tight = make_certificate(deepcopy(report), deepcopy(baseline))

    assert cert_loose["validation"]["primary_metric_acceptable"] is True
    assert cert_tight["validation"]["primary_metric_acceptable"] is False

    # Resolved policy should reflect tiers.yaml (no hidden fallback).
    resolved_loose = cert_loose.get("resolved_policy", {})
    resolved_tight = cert_tight.get("resolved_policy", {})
    assert (resolved_loose.get("metrics") or {}).get("pm_ratio", {}).get(
        "ratio_limit_base"
    ) == pytest.approx(1.15)
    assert (resolved_tight.get("metrics") or {}).get("pm_ratio", {}).get(
        "ratio_limit_base"
    ) == pytest.approx(1.10)

    # Policy provenance digest should move when tiers.yaml moves.
    digest_loose = (cert_loose.get("policy_provenance") or {}).get("policy_digest")
    digest_tight = (cert_tight.get("policy_provenance") or {}).get("policy_digest")
    assert isinstance(digest_loose, str) and digest_loose
    assert isinstance(digest_tight, str) and digest_tight
    assert digest_loose != digest_tight
