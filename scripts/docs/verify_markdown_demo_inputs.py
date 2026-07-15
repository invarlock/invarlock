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


class DemoInputBuildError(RuntimeError):
    """Raised when live demo inputs cannot be built honestly."""


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
) -> dict[str, object]:
    src_root = ROOT / "src"
    if not (src_root / "invarlock").is_dir():
        raise DemoInputBuildError(
            "Cannot build markdown demo inputs: src/invarlock is unavailable."
        )
    src_path = str(src_root)
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    try:
        from invarlock.reporting.report_make import make_report
    except Exception as exc:
        raise DemoInputBuildError(
            "Cannot build markdown demo inputs: report builder import failed."
        ) from exc

    try:
        evaluation_report = make_report(run_report, baseline_report)
    except Exception as exc:
        raise DemoInputBuildError(
            "Cannot build markdown demo inputs: report builder failed."
        ) from exc
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


def _bind_demo_runtime_policy_receipt(report: dict[str, object]) -> None:
    """Attach the same effective-policy receipt required from a real run."""

    src_path = str(ROOT / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    from invarlock.core.auto_tuning import resolve_tier_policies
    from invarlock.reporting.runtime_policy_receipt import (
        build_runtime_policy_receipt,
    )

    meta = report["meta"]
    edit = report["edit"]
    guards = report["guards"]
    assert isinstance(meta, dict)
    assert isinstance(edit, dict)
    assert isinstance(guards, list)
    auto = meta.get("auto")
    auto = auto if isinstance(auto, dict) else {}
    tier = str(auto.get("tier") or "balanced")
    edit_name = str(edit["name"])
    profile = "dev"
    policies = resolve_tier_policies(tier, edit_name, profile=profile)
    resolved, receipt = build_runtime_policy_receipt(
        policies,
        guards,
        tier=tier,
        profile=profile,
        edit_name=edit_name,
    )
    report["resolved_policy"] = resolved
    report["policy_resolution"] = receipt


def _seed_demo_inputs(workspace: Path, *, fixture_mode: bool = False) -> None:
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
            "preview_final_slice_delta_summary": {
                "mean": 0.0,
                "ci": [0.0, 0.0],
                "basis": "independent_disjoint_slices",
                "paired": False,
                "ci_method": "none",
                "ci_reason": "constant_demo_slices",
                "preview_windows": 2,
                "final_windows": 2,
                "degenerate": True,
                "degenerate_reason": "constant_demo_slices",
            },
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
            "preview": {
                "window_ids": [0, 1],
                "logloss": [2.30, 2.30],
                "token_counts": [100, 100],
            },
            "final": {
                "window_ids": [2, 3],
                "logloss": [2.30, 2.30],
                "token_counts": [100, 100],
            },
        },
    }
    baseline_report = {
        "run_id": "docs-demo-base",
        "model_id": "docs-demo-model",
        "meta": {
            "seed": 0,
            "model_id": "docs-demo-model",
            "adapter": "hf_causal",
            "auto": {
                "enabled": False,
                "tier": "balanced",
                "probes_used": 0,
                "target_pm_ratio": None,
            },
        },
        "context": {"profile": "dev"},
        "evaluation_windows": {
            "preview": {
                "window_ids": [0, 1],
                "logloss": [2.30, 2.30],
                "token_counts": [100, 100],
            },
            "final": {
                "window_ids": [2, 3],
                "logloss": [2.30, 2.30],
                "token_counts": [100, 100],
            },
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
            "name": "noop",
            "plan_digest": "0",
            "deltas": {
                "params_changed": 0,
                "layers_modified": 0,
                "sparsity": None,
                "bitwidth_map": None,
            },
        },
        "guards": [],
        "metrics": {
            "primary_metric": {
                "kind": "ppl_causal",
                "preview": 10.0,
                "final": 10.0,
            }
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
    }
    baseline_final_summary = _demo_window_summary(
        baseline_report["evaluation_windows"]["final"]
    )
    subject_final_summary = _demo_window_summary(
        run_report["evaluation_windows"]["final"]
    )
    subject_preview_summary = _demo_window_summary(
        run_report["evaluation_windows"]["preview"]
    )
    if baseline_final_summary is not None:
        baseline_mean_logloss, baseline_ppl, _ = baseline_final_summary
        baseline_report["metrics"]["primary_metric"]["preview"] = baseline_ppl
        baseline_report["metrics"]["primary_metric"]["final"] = baseline_ppl
    else:
        baseline_mean_logloss = None
    if subject_final_summary is not None:
        subject_mean_logloss, subject_ppl, subject_tokens = subject_final_summary
        run_report["metrics"]["primary_metric"]["final"] = subject_ppl
        run_report["metrics"]["final_total_tokens"] = subject_tokens
    else:
        subject_mean_logloss = None
    if subject_preview_summary is not None:
        _, subject_preview_ppl, subject_preview_tokens = subject_preview_summary
        run_report["metrics"]["primary_metric"]["preview"] = subject_preview_ppl
        run_report["metrics"]["preview_total_tokens"] = subject_preview_tokens
    if baseline_mean_logloss is not None and subject_mean_logloss is not None:
        delta_mean_logloss = subject_mean_logloss - baseline_mean_logloss
        run_report["metrics"]["preview_final_slice_delta_summary"]["mean"] = (
            delta_mean_logloss
        )
        run_report["metrics"]["logloss_delta"] = delta_mean_logloss
    _bind_demo_runtime_policy_receipt(run_report)
    _bind_demo_runtime_policy_receipt(baseline_report)
    for target in (
        workspace / "runs" / "baseline" / "report.json",
        workspace / "runs" / "source" / "report.json",
        workspace / "runs" / "baseline_calib" / "source" / "demo" / "report.json",
    ):
        _write_json(target, baseline_report)
    _write_json(workspace / "runs" / "subject" / "report.json", run_report)

    if fixture_mode:
        if not DEMO_EVALUATION_REPORT_FIXTURE.is_file():
            raise DemoInputBuildError(
                "Explicit fixture mode requested, but the demo report fixture is missing."
            )
        fixture_payload = json.loads(
            DEMO_EVALUATION_REPORT_FIXTURE.read_text(encoding="utf-8")
        )
        if not isinstance(fixture_payload, dict):
            raise DemoInputBuildError("The explicit demo report fixture is invalid.")
        fixture_meta = fixture_payload.get("meta")
        if not isinstance(fixture_meta, dict):
            fixture_meta = {}
        fixture_meta["demo_input_mode"] = "explicit_fixture"
        fixture_payload["meta"] = fixture_meta
        for target in evaluation_targets:
            _write_json(target, fixture_payload)
            _write_runtime_manifest_for_report(target)
        _write_json(
            workspace / "demo_input_mode.json",
            {
                "mode": "explicit_fixture",
                "source": "tests/artifacts/golden_runs/gpt2/evaluation.report.json",
            },
        )
    else:
        evaluation_report = _build_demo_evaluation_report(run_report, baseline_report)
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
        {"support_tiers": ["maintained_catalog"]},
    )


__all__ = [
    "DEMO_EVALUATION_REPORT_FIXTURE",
    "DEMO_RUNTIME_MANIFEST_FIXTURE",
    "DemoInputBuildError",
    "_build_demo_evaluation_report",
    "_demo_window_summary",
    "_seed_demo_inputs",
    "_write_json",
    "_write_runtime_manifest_for_report",
]
