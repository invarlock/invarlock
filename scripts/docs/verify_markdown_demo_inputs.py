"""Seeded demo inputs for markdown bash block replay."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEMO_EVALUATION_REPORT_FIXTURE = (
    ROOT / "tests" / "artifacts" / "golden_runs" / "gpt2" / "evaluation.report.json"
)
DEMO_RUNTIME_MANIFEST_FIXTURE = (
    ROOT / "tests" / "fixtures" / "runtime_provenance" / "runtime.manifest.json"
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_runtime_manifest_for_report(report_path: Path) -> None:
    if not DEMO_RUNTIME_MANIFEST_FIXTURE.is_file() or not report_path.is_file():
        return
    manifest = json.loads(DEMO_RUNTIME_MANIFEST_FIXTURE.read_text(encoding="utf-8"))
    manifest["report"] = {
        "filename": report_path.name,
        "path": str(report_path),
        "sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
    }
    _write_json(report_path.parent / "runtime.manifest.json", manifest)


def _build_demo_evaluation_report(
    run_report: dict[str, object],
    baseline_report: dict[str, object],
) -> dict[str, object] | None:
    def _fallback_demo_report() -> dict[str, object]:
        return {
            "schema_version": "v1",
            "run_id": "docs-demo",
            "artifacts": {"generated_at": "2026-04-16T00:00:00+00:00"},
            "plugins": {},
            "meta": {"seed": 42},
            "primary_metric": {
                "kind": "ppl_causal",
                "preview": math.exp(2.30),
                "final": math.exp(2.30),
                "ratio_vs_baseline": 1.0,
                "display_ci": [1.0, 1.01],
            },
            "dataset": {
                "provider": "unit",
                "seq_len": 8,
                "windows": {
                    "preview": 1,
                    "final": 1,
                    "stats": {
                        "window_match_fraction": 1.0,
                        "window_overlap_fraction": 0.0,
                        "coverage": {"preview": {"used": 1}, "final": {"used": 1}},
                        "paired_windows": 1,
                    },
                },
            },
            "baseline_ref": {"primary_metric": {"final": math.exp(2.30)}},
            "validation": {"primary_metric_acceptable": True},
            "resolved_policy": {
                "metrics": {
                    "pm_ratio": {
                        "ratio_limit_base": 1.1,
                        "min_tokens": 1,
                        "min_token_fraction": 0.0,
                        "hysteresis_ratio": 0.0,
                    }
                }
            },
            "policy_digest": {
                "policy_version": "policy-v1",
                "tier_policy_name": "balanced",
                "thresholds_hash": "docs-demo-policy",
                "changed": False,
            },
            "provenance": {"provider_digest": {"ids_sha256": "docs-demo-provider"}},
        }

    src_root = ROOT / "src"
    if not (src_root / "invarlock").is_dir():
        return _fallback_demo_report()
    src_path = str(src_root)
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    try:
        from invarlock.reporting.report_make import make_report
    except Exception:
        return _fallback_demo_report()

    try:
        evaluation_report = make_report(run_report, baseline_report)
    except Exception:
        return _fallback_demo_report()
    validation = evaluation_report.get("validation")
    if isinstance(validation, dict):
        validation["primary_metric_acceptable"] = True
    evaluation_report["resolved_policy"] = {
        "metrics": {
            "pm_ratio": {
                "ratio_limit_base": 1.1,
                "min_tokens": 1,
                "min_token_fraction": 0.0,
                "hysteresis_ratio": 0.0,
            }
        }
    }
    return evaluation_report


def _prepare_demo_evaluation_report_for_replay(
    evaluation_report: dict[str, object],
) -> dict[str, object]:
    canonical_guards = ["invariants", "spectral", "rmt", "variance", "invariants"]
    context = evaluation_report.get("context")
    if not isinstance(context, dict):
        context = {}
    context.update(
        {
            "profile": "ci",
            "tier": "balanced",
            "assurance": {"mode": "strict"},
            "runtime": {"execution_mode": "container"},
            "guard_chain_observed": canonical_guards,
        }
    )
    evaluation_report["context"] = context
    evaluation_report["guards"] = [{"name": name} for name in canonical_guards]
    evaluation_report["invariants"] = {"supported": True, "passed": True}
    evaluation_report["spectral"] = {"supported": True, "passed": True}
    evaluation_report["rmt"] = {"supported": True, "passed": True}
    evaluation_report["variance"] = {
        "enabled": False,
        "supported": True,
        "status": "disabled",
    }
    evaluation_report["report_build"] = {
        "synthesized_fields": [],
        "repaired_fields": [],
        "fallback_fields": [],
    }
    provenance = evaluation_report.get("provenance")
    if not isinstance(provenance, dict):
        provenance = {}
    provenance["edited"] = {"report_path": "../../runs/subject/report.json"}
    provenance["baseline"] = {"report_path": "../../runs/baseline/report.json"}
    evaluation_report["provenance"] = provenance
    evaluation_report["assurance"] = {
        "claim_set": "invarlock-weight-edit-regression-v1",
        "mode": "strict",
        "verdict": "pending_verifier",
        "report_local_verdict": "pass",
        "verified_assurance_verdict": "pending",
        "canonical_guard_chain_enforced": True,
        "guard_chain_observed": canonical_guards,
        "fallback_fields_used": False,
        "runtime_provenance_declared": "container",
        "runtime_provenance_verified": False,
        "runtime_provenance_verification_status": "pending",
        "blocking_reasons": [],
    }
    return evaluation_report


def _demo_window_summary(section: dict[str, object]) -> tuple[float, float, int] | None:
    loglosses = section.get("logloss")
    token_counts = section.get("token_counts")
    if not isinstance(loglosses, list) or not loglosses:
        return None
    if not isinstance(token_counts, list) or len(token_counts) != len(loglosses):
        token_counts = [1] * len(loglosses)

    weighted_total = 0.0
    total_tokens = 0
    for logloss, token_count in zip(loglosses, token_counts, strict=False):
        try:
            logloss_f = float(logloss)
            token_count_i = int(token_count)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(logloss_f) or token_count_i <= 0:
            return None
        weighted_total += logloss_f * token_count_i
        total_tokens += token_count_i
    if total_tokens <= 0:
        return None
    mean_logloss = weighted_total / total_tokens
    return mean_logloss, math.exp(mean_logloss), total_tokens


def _seed_demo_inputs(workspace: Path) -> None:
    evaluation_targets = (
        workspace / "reports" / "eval" / "evaluation.report.json",
        workspace / "reports" / "quant8_demo" / "evaluation.report.json",
        workspace / "report_bundle" / "evaluation.report.json",
        workspace / "reports" / "baseline_calib" / "evaluation.report.json",
        workspace / "reports" / "baseline_cpu" / "evaluation.report.json",
        workspace / "reports" / "baseline_mps" / "evaluation.report.json",
    )
    manifest_targets = (
        workspace / "reports" / "eval" / "runtime.manifest.json",
        workspace / "reports" / "quant8_demo" / "runtime.manifest.json",
        workspace / "report_bundle" / "runtime.manifest.json",
    )

    if DEMO_RUNTIME_MANIFEST_FIXTURE.is_file():
        for target in manifest_targets:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(DEMO_RUNTIME_MANIFEST_FIXTURE, target)

    run_report = {
        "meta": {
            "model_id": "docs-demo-model",
            "adapter": "hf_causal",
            "commit": "docs-demo",
            "seed": 42,
            "device": "cpu",
            "ts": "2026-04-03T00:00:00+00:00",
            "auto": {
                "enabled": False,
                "tier": "balanced",
                "probes_used": 0,
                "target_pm_ratio": None,
            },
        },
        "data": {
            "dataset": "unit",
            "split": "validation",
            "seq_len": 8,
            "stride": 8,
            "preview_n": 2,
            "final_n": 2,
        },
        "edit": {
            "name": "noop",
            "plan_digest": "docs-demo",
            "deltas": {
                "params_changed": 0,
                "sparsity": None,
                "bitwidth_map": None,
                "layers_modified": 0,
            },
        },
        "guards": [],
        "metrics": {
            "primary_metric": {
                "kind": "ppl_causal",
                "preview": 10.0,
                "final": 10.0,
            },
            "bootstrap": {
                "method": "percentile",
                "replicates": 50,
                "alpha": 0.05,
                "seed": 0,
                "coverage": {
                    "preview": {"used": 2},
                    "final": {"used": 2},
                },
            },
            "paired_delta_summary": {"mean": 0.0},
            "preview_total_tokens": 50000,
            "final_total_tokens": 50000,
            "logloss_delta": 0.0,
            "logloss_delta_ci": [-0.01, 0.01],
        },
        "artifacts": {
            "events_path": "",
            "logs_path": "",
            "checkpoint_path": None,
        },
        "flags": {
            "guard_recovered": False,
            "rollback_reason": None,
        },
        "evaluation_windows": {
            "final": {
                "window_ids": [1, 2],
                "logloss": [2.30, 2.30],
                "token_counts": [100, 100],
            }
        },
    }
    baseline_report = {
        "run_id": "docs-demo-base",
        "model_id": "docs-demo-model",
        "meta": {"seed": 0, "model_id": "docs-demo-model"},
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
        "artifacts": {
            "events_path": "",
            "logs_path": "",
            "checkpoint_path": None,
        },
        "flags": {
            "guard_recovered": False,
            "rollback_reason": None,
        },
    }
    baseline_final_summary = _demo_window_summary(
        baseline_report["evaluation_windows"]["final"]
    )
    subject_final_summary = _demo_window_summary(
        run_report["evaluation_windows"]["final"]
    )
    if baseline_final_summary is not None:
        baseline_mean_logloss, baseline_ppl, baseline_tokens = baseline_final_summary
        run_report["metrics"]["primary_metric"]["preview"] = baseline_ppl
        run_report["metrics"]["preview_total_tokens"] = baseline_tokens
        baseline_report["metrics"]["primary_metric"]["final"] = baseline_ppl
    else:
        baseline_mean_logloss = None
    if subject_final_summary is not None:
        subject_mean_logloss, subject_ppl, subject_tokens = subject_final_summary
        run_report["metrics"]["primary_metric"]["final"] = subject_ppl
        run_report["metrics"]["final_total_tokens"] = subject_tokens
    else:
        subject_mean_logloss = None
    if baseline_mean_logloss is not None and subject_mean_logloss is not None:
        delta_mean_logloss = subject_mean_logloss - baseline_mean_logloss
        run_report["metrics"]["paired_delta_summary"]["mean"] = delta_mean_logloss
        run_report["metrics"]["logloss_delta"] = delta_mean_logloss
    for target in (
        workspace / "runs" / "baseline" / "report.json",
        workspace / "runs" / "source" / "report.json",
        workspace / "runs" / "baseline_calib" / "source" / "demo" / "report.json",
    ):
        _write_json(target, baseline_report)
    _write_json(workspace / "runs" / "subject" / "report.json", run_report)

    evaluation_report = _build_demo_evaluation_report(run_report, baseline_report)
    if evaluation_report is not None:
        evaluation_report = _prepare_demo_evaluation_report_for_replay(
            evaluation_report
        )
        target_payloads = {
            workspace / "reports" / "baseline_cpu" / "evaluation.report.json": {
                **evaluation_report,
                "meta": {**evaluation_report.get("meta", {}), "device": "cpu"},
            },
            workspace / "reports" / "baseline_mps" / "evaluation.report.json": {
                **evaluation_report,
                "meta": {**evaluation_report.get("meta", {}), "device": "mps"},
            },
        }
        for target in evaluation_targets:
            _write_json(target, target_payloads.get(target, evaluation_report))
            _write_runtime_manifest_for_report(target)
    elif DEMO_EVALUATION_REPORT_FIXTURE.is_file():
        for target in evaluation_targets:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(DEMO_EVALUATION_REPORT_FIXTURE, target)
            _write_runtime_manifest_for_report(target)

    _write_json(
        workspace / "resolved_policy.json",
        {"metrics": {"pm_ratio": {"ratio_limit_base": 1.1}}},
    )
    _write_json(
        workspace / "overrides.json",
        [{"path": "metrics.pm_ratio.ratio_limit_base", "value": 1.1}],
    )
    _write_json(
        workspace / "compatibility.json",
        {"support_tiers": ["published_basis"]},
    )


__all__ = [
    "DEMO_EVALUATION_REPORT_FIXTURE",
    "DEMO_RUNTIME_MANIFEST_FIXTURE",
    "_build_demo_evaluation_report",
    "_demo_window_summary",
    "_prepare_demo_evaluation_report_for_replay",
    "_seed_demo_inputs",
    "_write_json",
    "_write_runtime_manifest_for_report",
]
