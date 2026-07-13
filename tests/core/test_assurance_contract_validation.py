from __future__ import annotations

import copy
import math
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import invarlock.core.evaluate_plan as evaluate_plan_mod
from invarlock.core.api import RunReport
from invarlock.core.assurance_contract import (
    CANONICAL_GUARD_CHAIN,
    build_assurance_section,
    strict_report_policy_errors,
)
from invarlock.core.evaluate_plan import (
    build_evaluate_command_plan,
    build_subject_edit_run_config,
    load_evaluate_preset_data,
    resolve_evaluate_tmp_dir,
)
from invarlock.core.runner_runtime.guards import (
    _load_external_guard_baseline,
    guard_phase,
)
from tests.core._support_assurance_contract import (
    _plugin_metadata,
    bind_noop_variance_evidence,
)
from tests.core._support_assurance_contract import (
    strict_report as _strict_report,
)


def _sync_variance_guard_metrics(report: dict[str, Any]) -> None:
    variance = report["variance"]
    variance_guard = next(
        entry for entry in report["guards"] if entry["name"] == "variance"
    )
    metrics = {
        "ve_enabled": variance["enabled"],
        "monitor_only": variance["monitor_only"],
        "predictive_gate": dict(variance["predictive_gate"]),
    }
    for top_key, raw_key in (
        ("ve_enabled_during_validation", "ve_enabled_during_validation"),
        ("subject_restored_after_ab", "subject_restored_after_ab"),
        ("met_threshold", "met_threshold"),
        ("gain", "ab_gain"),
        ("ppl_no_ve", "ppl_no_ve"),
        ("ppl_with_ve", "ppl_with_ve"),
        ("ratio_ci", "ratio_ci"),
        ("proposed_scales", "proposed_scales"),
        ("target_modules", "target_modules"),
        ("target_module_names", "target_module_names"),
        ("proposed_scales_pre_edit", "proposed_scales_pre_edit"),
        ("proposed_scales_post_edit", "proposed_scales_post_edit"),
        ("raw_scales_pre_edit", "raw_scales_pre_edit"),
        ("raw_scales_post_edit", "raw_scales_post_edit"),
    ):
        if top_key in variance:
            metrics[raw_key] = copy.deepcopy(variance[top_key])
    if "calibration" in variance:
        metrics["calibration"] = dict(variance["calibration"])
    ab_test = variance.get("ab_test")
    if isinstance(ab_test, dict):
        metrics["ab_seed_used"] = ab_test.get("seed")
        metrics["ab_windows_used"] = ab_test.get("windows_used")
        raw_provenance = copy.deepcopy(ab_test.get("provenance"))
        if isinstance(raw_provenance, dict):
            raw_provenance.pop("window_ids", None)
        metrics["ab_provenance"] = raw_provenance
        metrics["ab_point_estimates"] = copy.deepcopy(ab_test.get("point_estimates"))
        metrics["ab_measurements"] = copy.deepcopy(ab_test.get("measurements"))
    variance_guard["metrics"] = metrics
    if "policy" in variance:
        variance_guard["policy"] = copy.deepcopy(variance["policy"])
        metrics["mode"] = variance["policy"].get("mode")
        report.setdefault("resolved_policy", {})["variance"] = copy.deepcopy(
            variance["policy"]
        )
    if "subject_restored_after_ab" in variance:
        condition_a = _mapping_for_test(metrics.get("ab_provenance"), "condition_a")
        variance_guard["details"] = {
            "ve_tested": variance["ve_enabled_during_validation"],
            "ve_applied": variance["enabled"],
            "subject_restored_after_ab": variance["subject_restored_after_ab"],
            "policy": copy.deepcopy(variance.get("policy")),
            "proposed_scales": copy.deepcopy(
                metrics.get("proposed_scales_post_edit", {})
            ),
            "stats": {
                "ab_provenance": copy.deepcopy(metrics.get("ab_provenance")),
                "ab_point_estimates": copy.deepcopy(metrics.get("ab_point_estimates")),
                "ab_measurements": copy.deepcopy(metrics.get("ab_measurements")),
                "predictive_gate": copy.deepcopy(metrics.get("predictive_gate")),
                "calibration": {
                    "window_ids": copy.deepcopy(condition_a.get("window_ids"))
                },
                "target_fingerprint": condition_a.get("target_fingerprint"),
                "pairing_reference": {"digest": condition_a.get("pairing_digest")},
                "dataset_meta": {
                    "dataset_hash": condition_a.get("dataset_hash"),
                    "tokenizer_hash": condition_a.get("tokenizer_hash"),
                },
                "target_module_names": copy.deepcopy(
                    metrics.get("target_module_names")
                ),
                "proposed_scales_pre_edit": copy.deepcopy(
                    metrics.get("proposed_scales_pre_edit")
                ),
                "proposed_scales_post_edit": copy.deepcopy(
                    metrics.get("proposed_scales_post_edit")
                ),
                "raw_scales_pre_edit_normalized": copy.deepcopy(
                    metrics.get("raw_scales_pre_edit")
                ),
                "raw_scales_post_edit_normalized": copy.deepcopy(
                    metrics.get("raw_scales_post_edit")
                ),
            },
        }


def _mapping_for_test(value: object, key: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    nested = value.get(key)
    return nested if isinstance(nested, dict) else {}


def test_strict_report_rejects_non_array_blockers_and_cross_section_drift() -> None:
    report = _strict_report()
    report["assurance"] = build_assurance_section(report)
    report["assurance"]["blocking_reasons"] = "not-an-array"
    report["context"]["profile"] = "release"
    report["meta"] = {"profile": "release"}
    report["auto"]["tier"] = "conservative"

    errors = strict_report_policy_errors(report, require_strict=True)

    assert "strict assurance.blocking_reasons must be an array." in errors
    assert "strict assurance.profile must match context.profile exactly." in errors
    assert "strict assurance.profile must match meta.profile exactly." in errors
    assert "strict assurance.tier must match auto.tier exactly." in errors


def _strict_no_adjustment_report() -> dict[str, Any]:
    report = _strict_report()
    report["context"]["profile"] = "release"
    report["edit"] = {"name": "noop"}
    report["structure"] = {"params_changed": 0, "layers_modified": 0}
    report["variance"]["enabled"] = False
    report["variance"]["monitor_only"] = False
    report["variance"]["calibration"] = {
        "status": "no_scaling_required",
        "coverage": 8,
        "min_coverage": 6,
    }
    report["variance"]["predictive_gate"] = {
        "evaluated": True,
        "passed": True,
        "reason": "no_adjustment_required",
        "delta_ci": [None, None],
        "gain_ci": [None, None],
        "mean_delta": None,
    }
    bind_noop_variance_evidence(report)
    report["assurance"] = build_assurance_section(report)
    return report


def test_strict_report_accepts_fully_bound_no_adjustment_exception() -> None:
    report = _strict_no_adjustment_report()

    assert (
        strict_report_policy_errors(
            report,
            require_strict=True,
            verifier_profile="release",
        )
        == []
    )


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        (
            lambda report: report["edit"].update(name="structured"),
            "requires edit.name=noop",
        ),
        (
            lambda report: report["structure"].update(params_changed=123),
            "requires structure.params_changed=0",
        ),
        (
            lambda report: report["structure"].update(params_changed=False),
            "requires structure.params_changed=0",
        ),
        (
            lambda report: report["variance"].pop("calibration"),
            "requires variance.calibration evidence",
        ),
        (
            lambda report: report["variance"]["calibration"].update(coverage=5),
            "requires adequate variance calibration coverage",
        ),
        (
            lambda report: report["variance"]["predictive_gate"].update(
                reason="no_scales"
            ),
            "strict no-op variance requires predictive_gate.reason",
        ),
    ],
)
def test_strict_report_rejects_unbound_no_adjustment_exception(
    mutation: Callable[[dict[str, Any]], object],
    expected: str,
) -> None:
    report = _strict_no_adjustment_report()
    mutation(report)
    report["assurance"] = build_assurance_section(report)

    errors = strict_report_policy_errors(
        report,
        require_strict=True,
        verifier_profile="release",
    )

    assert any(expected in error for error in errors)


def test_strict_report_rejects_non_noop_failure_reason_claimed_as_pass() -> None:
    report = _strict_report()
    report["edit"] = {"name": "quant_rtn"}
    report["plugins"]["edit"] = report["plugins"]["edit"] | {
        "name": "quant_rtn",
        "module": "invarlock.edits.quant_rtn",
    }
    report["structure"]["params_changed"] = 123
    report["variance"] = {
        "enabled": False,
        "monitor_only": False,
        "supported": True,
        "passed": True,
        "decision": "allow",
        "violations": [],
        "predictive_gate": {
            "evaluated": True,
            "passed": True,
            "reason": "no_scales",
            "delta_ci": [None, None],
            "gain_ci": [None, None],
            "mean_delta": None,
        },
    }
    _sync_variance_guard_metrics(report)
    report["assurance"] = build_assurance_section(report)

    errors = strict_report_policy_errors(report, require_strict=True)

    assert any(
        "reason=no_scales cannot be a passing result" in error for error in errors
    )


def test_strict_report_rejects_variance_summary_raw_guard_disagreement() -> None:
    report = _strict_no_adjustment_report()
    variance_guard = next(
        entry for entry in report["guards"] if entry["name"] == "variance"
    )
    variance_guard["metrics"] = {
        "ve_enabled": True,
        "monitor_only": True,
        "predictive_gate": {
            "evaluated": True,
            "passed": False,
            "reason": "no_scales",
            "delta_ci": [None, None],
            "gain_ci": [None, None],
            "mean_delta": None,
        },
        "calibration": {
            "status": "insufficient_coverage",
            "coverage": 1,
            "min_coverage": 6,
        },
    }

    errors = strict_report_policy_errors(report, require_strict=True)

    assert any("guards[3].metrics.monitor_only must match" in error for error in errors)
    assert any(
        "guards[3].metrics.predictive_gate.passed must match" in error
        for error in errors
    )
    assert any(
        "guards[3].metrics.calibration.status must match" in error for error in errors
    )


def _strict_variance_gain_report() -> dict[str, Any]:
    report = _strict_report()
    report["edit"] = {"name": "quant_rtn"}
    report["plugins"]["edit"] = _plugin_metadata("edits", "quant_rtn")
    report["structure"]["params_changed"] = 123
    variance_guard = next(
        entry for entry in report["guards"] if entry["name"] == "variance"
    )
    existing_provenance = variance_guard["metrics"]["ab_provenance"]
    window_ids = list(existing_provenance["condition_a"]["window_ids"])
    delta_log = math.log(98.0) - math.log(100.0)
    measurements = {
        "window_ids": list(window_ids),
        "condition_a": {
            "ppl": [100.0] * 8,
            "log_loss": [math.log(100.0)] * 8,
            "token_counts": [16] * 8,
        },
        "condition_b": {
            "ppl": [98.0] * 8,
            "log_loss": [math.log(98.0)] * 8,
            "token_counts": [16] * 8,
        },
        "ratio_bootstrap": {
            "method": "percentile_mean_ppl_ratio",
            "replicates": 500,
            "alpha": 0.05,
            "seed": 123,
        },
        "delta_log_bootstrap": {
            "method": "bca_paired_delta_log",
            "replicates": 500,
            "alpha": 0.05,
            "seed": 334,
            "weights": "condition_a_token_counts",
        },
        "ratio_ci": [0.98, 0.98],
        "delta_log_ci": [delta_log, delta_log],
    }
    condition_common = {
        "tag": "post_edit",
        "window_ids": window_ids,
        "window_count": 8,
        "target_fingerprint": "strict-target",
        "pairing_digest": existing_provenance["condition_a"]["pairing_digest"],
        "consumed_pairing_digest": existing_provenance["condition_a"][
            "consumed_pairing_digest"
        ],
        "dataset_hash": "strict-dataset",
        "tokenizer_hash": "strict-tokenizer",
        "model_id": "strict-model",
        "run_seed": 123,
        "status": "evaluated",
    }
    report["variance"] = {
        "enabled": False,
        "ve_enabled_during_validation": True,
        "subject_restored_after_ab": True,
        "met_threshold": True,
        "monitor_only": False,
        "supported": True,
        "passed": True,
        "decision": "allow",
        "violations": [],
        "policy": {
            "min_effect_lognll": 0.005,
            "predictive_one_sided": True,
            "predictive_gate": True,
            "alpha": 0.05,
            "min_gain": 0.0,
            "tie_breaker_deadband": 0.005,
            "min_rel_gain": 0.001,
            "seed": 123,
            "mode": "ci",
            "absolute_floor_ppl": 0.05,
            "clamp": [0.5, 2.0],
            "deadband": 0.02,
            "min_abs_adjust": 0.012,
            "max_scale_step": 0.02,
            "topk_backstop": 1,
            "max_adjusted_modules": 0,
            "calibration": {"windows": 8, "min_coverage": 6, "seed": 123},
        },
        "gain": 0.02,
        "ppl_no_ve": 100.0,
        "ppl_with_ve": 98.0,
        "ratio_ci": [0.98, 0.98],
        "proposed_scales": 1,
        "target_modules": 1,
        "target_module_names": ["transformer.h.0.mlp.c_proj"],
        "proposed_scales_pre_edit": {"transformer.h.0.mlp.c_proj": 1.02},
        "proposed_scales_post_edit": {"transformer.h.0.mlp.c_proj": 1.02},
        "raw_scales_pre_edit": {"transformer.h.0.mlp.c_proj": 1.1},
        "raw_scales_post_edit": {"transformer.h.0.mlp.c_proj": 1.1},
        "predictive_gate": {
            "evaluated": True,
            "passed": True,
            "reason": "ci_gain_met",
            "delta_ci": [delta_log, delta_log],
            "gain_ci": [-delta_log, -delta_log],
            "mean_delta": delta_log,
        },
        "calibration": {
            "status": "complete",
            "requested": 8,
            "coverage": 8,
            "min_coverage": 6,
            "seed": 123,
        },
        "ab_test": {
            "seed": 123,
            "windows_used": 8,
            "provenance": {
                "condition_a": {**condition_common, "mode": "edited_no_ve"},
                "condition_b": {**condition_common, "mode": "virtual_ve"},
                "window_ids": window_ids,
            },
            "point_estimates": {
                "tag": "post_edit",
                "ppl_no_ve": 100.0,
                "ppl_with_ve": 98.0,
                "coverage": 8,
            },
            "measurements": measurements,
        },
    }
    _sync_variance_guard_metrics(report)
    report["assurance"] = build_assurance_section(report)
    return report


def test_strict_report_accepts_complete_non_noop_variance_gain() -> None:
    report = _strict_variance_gain_report()

    assert strict_report_policy_errors(report, require_strict=True) == []


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        (
            lambda variance: variance.update(enabled=True),
            "requires final variance.enabled=false",
        ),
        (
            lambda variance: variance["predictive_gate"].update(delta_ci=[None, None]),
            "requires a finite two-value delta_ci",
        ),
        (
            lambda variance: variance["predictive_gate"].update(gain_ci=[0.0, 0.0]),
            "gain_ci must be the exact inverse",
        ),
        (
            lambda variance: variance["predictive_gate"].update(mean_delta=0.0),
            "requires a negative mean_delta",
        ),
        (
            lambda variance: variance["calibration"].update(coverage=1),
            "requires adequate variance calibration coverage",
        ),
        (
            lambda variance: variance["policy"].update(min_effect_lognll=0.021),
            "delta_ci does not meet policy.min_effect_lognll",
        ),
    ],
)
def test_strict_report_rejects_forged_variance_gain_evidence(
    mutation: Callable[[dict[str, Any]], object],
    expected: str,
) -> None:
    report = _strict_variance_gain_report()
    mutation(report["variance"])
    _sync_variance_guard_metrics(report)
    report["assurance"] = build_assurance_section(report)

    errors = strict_report_policy_errors(report, require_strict=True)

    assert any(expected in error for error in errors)


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        (
            lambda metrics, guard: metrics.update(ve_enabled_during_validation=False),
            "ve_enabled_during_validation=true",
        ),
        (
            lambda metrics, guard: metrics.update(subject_restored_after_ab=False),
            "subject_restored_after_ab=true",
        ),
        (
            lambda metrics, guard: metrics.update(met_threshold=False),
            "met_threshold=true",
        ),
        (
            lambda metrics, guard: metrics.update(ab_gain=-0.9),
            "ab_gain must match the measured PPL improvement",
        ),
        (
            lambda metrics, guard: metrics.update(ppl_with_ve=120.0),
            "ppl_with_ve must improve on ppl_no_ve",
        ),
        (
            lambda metrics, guard: metrics.update(ratio_ci=[1.5, 2.0]),
            "ratio_ci does not meet the policy threshold",
        ),
        (
            lambda metrics, guard: guard["policy"].update(min_effect_lognll=0.5),
            "guards[3].policy must match variance.policy exactly",
        ),
        (
            lambda metrics, guard: metrics["calibration"].update(requested=1),
            "calibration.coverage cannot exceed calibration.requested",
        ),
    ],
)
def test_strict_report_rejects_contradictory_raw_variance_success_facts(
    mutation: Callable[[dict[str, Any], dict[str, Any]], object],
    expected: str,
) -> None:
    report = _strict_variance_gain_report()
    variance_guard = next(
        entry for entry in report["guards"] if entry["name"] == "variance"
    )
    mutation(variance_guard["metrics"], variance_guard)
    report["assurance"] = build_assurance_section(report)

    errors = strict_report_policy_errors(report, require_strict=True)

    assert any(expected in error for error in errors)


def test_strict_report_rejects_missing_plugin_provenance() -> None:
    report = _strict_report()
    report["plugins"].pop("adapter")
    report["assurance"] = build_assurance_section(report)

    errors = strict_report_policy_errors(report, require_strict=True)

    assert any("plugins.adapter" in error for error in errors)


def test_strict_report_rejects_non_strict_and_forged_plugin_metadata() -> None:
    report = _strict_report()
    report["plugins"]["adapter"] = {
        "name": "evil_adapter",
        "type": "adapters",
        "module": "evil.adapter",
        "package": "evil",
        "available": True,
        "support_tier": "third_party",
        "strict_assurance_allowed": False,
    }
    report["plugins"]["edit"]["module"] = "evil.noop"
    report["plugins"]["guards"][2]["strict_assurance_allowed"] = False
    report["assurance"] = build_assurance_section(report)

    errors = strict_report_policy_errors(report, require_strict=True)

    assert any(
        "evil_adapter" in error and "shipped plugin" in error for error in errors
    )
    assert any("plugins.edit.module" in error for error in errors)
    assert any(
        "plugins.guards[2].strict_assurance_allowed" in error for error in errors
    )


def test_strict_report_rejects_compressed_tensors_until_packed_storage_is_verified() -> (
    None
):
    report = _strict_report()
    report["meta"]["adapter"] = "hf_ct"
    report["plugins"]["adapter"] = _plugin_metadata("adapters", "hf_ct")
    report["assurance"] = build_assurance_section(report)

    errors = strict_report_policy_errors(report, require_strict=True)

    assert any(
        "plugins.adapter is not eligible for strict assurance" in error
        for error in errors
    )


def test_strict_report_rejects_plugin_name_drift_from_report() -> None:
    report = _strict_report()
    report["meta"]["adapter"] = "hf_mlm"
    report["edit"]["name"] = "quant_rtn"
    report["assurance"] = build_assurance_section(report)

    errors = strict_report_policy_errors(report, require_strict=True)

    assert any(
        "plugins.adapter.name must match meta.adapter" in error for error in errors
    )
    assert any("plugins.edit.name must match edit.name" in error for error in errors)


def test_missing_default_preset_uses_builtin_data(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    missing = tmp_path / "missing-default.yaml"
    monkeypatch.setattr(
        evaluate_plan_mod,
        "default_evaluate_preset_path",
        lambda _adapter: missing,
    )
    path, payload = load_evaluate_preset_data(
        adapter_name="hf_causal",
        preset=None,
        load_yaml_fn=lambda _path: pytest.fail("missing preset must not be loaded"),
    )

    assert path == missing
    assert payload["dataset"]["provider"] == "wikitext2"
    assert payload["dataset"]["seq_len"] == 512


def test_explicit_missing_preset_fails_before_yaml_load(tmp_path: Path) -> None:
    missing = tmp_path / "missing-explicit.yaml"
    with pytest.raises(FileNotFoundError, match="missing-explicit.yaml"):
        load_evaluate_preset_data(
            adapter_name="hf_causal",
            preset=str(missing),
            load_yaml_fn=lambda _path: pytest.fail("missing preset must not be loaded"),
        )


def test_evaluate_tmp_dir_allocates_under_requested_scratch_root(
    tmp_path: Path,
) -> None:
    scratch = tmp_path / "scratch"
    allocated = resolve_evaluate_tmp_dir(None, scratch_root=scratch)

    assert allocated.is_dir()
    assert allocated.parent == scratch.resolve()
    assert allocated.name.startswith("run-")


def test_subject_config_preserves_explicit_adapter_and_canonical_strict_chain() -> None:
    config = build_subject_edit_run_config(
        {},
        {
            "model": {"id": "hf:org/subject", "adapter": "custom_adapter"},
            "guards": {"order": list(CANONICAL_GUARD_CHAIN)},
        },
        subject_model_id="fallback/subject",
        adapter_name="hf_causal",
        output_dir="runs/subject",
        profile="ci",
        tier="balanced",
        guards_order=list(CANONICAL_GUARD_CHAIN),
        assurance_mode="strict",
        execution_mode="container",
    )

    assert config["model"] == {
        "id": "org/subject",
        "adapter": "custom_adapter",
    }
    assert config["guards"]["order"] == list(CANONICAL_GUARD_CHAIN)


def test_evaluate_plan_preserves_explicit_side_adapters(tmp_path: Path) -> None:
    preset = tmp_path / "preset.yaml"
    preset.write_text("guards: {}\n", encoding="utf-8")
    resolved: list[str] = []
    plan = build_evaluate_command_plan(
        baseline_model_id="source",
        subject_model_id="subject",
        profile="dev",
        tier="balanced",
        preset=str(preset),
        out=str(tmp_path / "runs"),
        edit_config=None,
        edit_label=None,
        resolve_auto_adapter_fn=lambda model: resolved.append(model) or "auto",
        load_yaml_fn=lambda _path: {"guards": {}},
        baseline_adapter="baseline_adapter",
        subject_adapter="subject_adapter",
        assurance_mode="off",
    )

    assert plan.baseline_adapter_name == "baseline_adapter"
    assert plan.subject_adapter_name == "subject_adapter"
    assert plan.baseline_adapter_auto is False
    assert plan.subject_adapter_auto is False
    assert resolved == []


class _EventRunner:
    def __init__(self) -> None:
        self.events: list[tuple[str, str, object, dict[str, object]]] = []

    def _log_event(
        self, category: str, event: str, level: object, details: dict[str, object]
    ) -> None:
        self.events.append((category, event, level, details))


def test_external_baseline_loader_fails_closed_for_incompatible_guards() -> None:
    runner = _EventRunner()
    report = RunReport(context={"baseline_guard_evidence_required": True})

    _load_external_guard_baseline(runner, SimpleNamespace(name="variance"), report)
    assert report.meta.get("baseline_guard_evidence") is None
    with pytest.raises(RuntimeError, match="cannot consume required baseline evidence"):
        _load_external_guard_baseline(
            runner,
            SimpleNamespace(name="spectral"),
            report,
        )
    with pytest.raises(TypeError, match="invalid baseline evidence outcome"):
        _load_external_guard_baseline(
            runner,
            SimpleNamespace(
                name="rmt", load_external_baseline_evidence=lambda: ["invalid"]
            ),
            report,
        )
    assert report.meta.get("baseline_guard_evidence") is None


@pytest.mark.parametrize(
    ("result_keys", "result_stages", "match"),
    [
        (["one", "two"], None, "result_keys must align"),
        (None, ["pre", "post"], "result_stages must align"),
    ],
)
def test_guard_phase_rejects_misaligned_result_metadata_without_mutation(
    result_keys: list[str] | None,
    result_stages: list[str] | None,
    match: str,
) -> None:
    runner = _EventRunner()
    report = RunReport(guards={"existing": {"passed": True}})
    guard = SimpleNamespace(name="invariants")
    with pytest.raises(ValueError, match=match):
        guard_phase(
            runner,
            model=object(),
            adapter=object(),
            guards=[guard],
            report=report,
            result_keys=result_keys,
            result_stages=result_stages,
        )
    assert report.guards == {"existing": {"passed": True}}
