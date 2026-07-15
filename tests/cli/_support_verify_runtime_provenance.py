from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

from invarlock.core.assurance_contract import ASSURANCE_CLAIM_SET, CANONICAL_GUARD_CHAIN
from invarlock.core.dataset_identity import dataset_identity_from_report
from invarlock.policy_pack import build_policy_pack
from invarlock.reporting import verify_contract as verify_mod
from invarlock.reporting.report_provenance import compute_report_digest
from invarlock.reporting.runtime_policy_receipt import build_runtime_policy_receipt
from invarlock.runtime_security import (
    RUNTIME_MANIFEST_FILENAME,
    RUNTIME_MANIFEST_VERSION,
    RUNTIME_VERIFIER_CONTRACT_VERSION,
)
from invarlock.utils import hash_json
from tests.core._support_assurance_contract import (
    _plugin_metadata,
    bind_noop_variance_evidence,
    bind_raw_guard_evidence,
)

_VALID_TEST_IMAGE_DIGEST = "sha256:" + ("a" * 64)


def bind_runtime_policy_receipt(payload: dict) -> dict:
    meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
    context = payload.get("context") if isinstance(payload.get("context"), dict) else {}
    run_context = context.get("run") if isinstance(context.get("run"), dict) else {}
    meta_auto = meta.get("auto") if isinstance(meta.get("auto"), dict) else {}
    context_auto = context.get("auto") if isinstance(context.get("auto"), dict) else {}
    report_auto = payload.get("auto") if isinstance(payload.get("auto"), dict) else {}
    assurance = (
        payload.get("assurance") if isinstance(payload.get("assurance"), dict) else {}
    )
    tier = (
        str(
            meta_auto.get("tier")
            or context_auto.get("tier")
            or run_context.get("tier")
            or context.get("tier")
            or report_auto.get("tier")
            or assurance.get("tier")
            or "balanced"
        )
        .strip()
        .lower()
    )
    profile = (
        str(
            context.get("profile")
            or run_context.get("profile")
            or assurance.get("profile")
            or "ci"
        )
        .strip()
        .lower()
    )
    resolved, receipt = build_runtime_policy_receipt(
        payload["resolved_policy"],
        payload["guards"],
        tier=tier,
        profile=profile,
        edit_name=(
            str(payload.get("edit", {}).get("name"))
            if payload.get("edit", {}).get("name") is not None
            else None
        ),
    )
    payload["resolved_policy"] = resolved
    payload["policy_resolution"] = receipt
    return payload


def _matching_strict_policy_pack(payload: dict | None = None) -> dict:
    subject = payload if payload is not None else _strict_provenance_gate_cert()
    assurance = subject.get("assurance", {})
    auto = subject.get("auto", {})
    tier = str(assurance.get("tier") or auto.get("tier") or "balanced")
    resolved_policy = subject.get("resolved_policy")
    assert isinstance(resolved_policy, dict) and resolved_policy
    return build_policy_pack(
        tier=tier,
        resolved_policy=copy.deepcopy(resolved_policy),
        compatibility={
            "support_tiers": ["published_basis"],
            "dataset_identity": dataset_identity_from_report(subject),
        },
    )


def _write_matching_strict_policy_pack(path: Path, payload: dict) -> Path:
    policy_path = path.with_name("trusted-policy-pack.json")
    policy_path.parent.mkdir(parents=True, exist_ok=True)
    policy_path.write_text(
        json.dumps(_matching_strict_policy_pack(payload), sort_keys=True),
        encoding="utf-8",
    )
    return policy_path


def _final_window_schedule_digest(window_ids: list[int]) -> str:
    digest = hashlib.blake2s(digest_size=16)
    for window_id in window_ids:
        digest.update(window_id.to_bytes(8, "little", signed=True))
    return digest.hexdigest()


def _provenance_gate_cert() -> dict:
    schedule_digest = _final_window_schedule_digest([1])
    return {
        "schema_version": "v1",
        "run_id": "runtime-provenance-gate",
        "artifacts": {"generated_at": "2024-01-01T00:00:00Z"},
        "plugins": {},
        "meta": {},
        "provenance": {
            "provider_digest": {"ids_sha256": "subject-ids"},
            "window_ids_digest": schedule_digest,
            "window_plan_digest": schedule_digest,
        },
        "dataset": {
            "provider": "test-fixture",
            "seq_len": 1,
            "windows": {
                "preview": 1,
                "final": 1,
                "stats": {
                    "coverage": {"preview": {"used": 1}, "final": {"used": 1}},
                    "actual_preview": 1,
                    "actual_final": 1,
                    "paired_windows": 1,
                    "window_match_fraction": 1.0,
                    "window_overlap_fraction": 0.0,
                },
            },
        },
        "primary_metric": {
            "kind": "ppl_causal",
            "preview": 9.0,
            "final": 9.0,
            "ratio_vs_baseline": 1.0,
            "display_ci": [1.0, 1.0],
            "analysis_basis": "mean_logloss",
            "analysis_point_preview": math.log(9.0),
            "analysis_point_final": math.log(9.0),
        },
        "evaluation_windows": {
            "preview": {
                "window_ids": [0],
                "logloss": [math.log(9.0)],
                "token_counts": [1],
            },
            "final": {
                "window_ids": [1],
                "logloss": [math.log(9.0)],
                "token_counts": [1],
            },
        },
        "baseline_ref": {"primary_metric": {"kind": "ppl_causal", "final": 9.0}},
        "guard_metric_impact": {
            "evaluated": True,
            "metric_kind": "ppl_causal",
            "direction": "lower",
            "bare_value": 9.0,
            "guarded_value": 9.0,
            "degradation_basis": "relative_increase",
            "degradation": 0.0,
            "degradation_limit": 0.01,
            "display_value": 0.0,
            "display_unit": "percent",
            "bare_facts": {
                "weighted_logloss_sum": math.log(9.0),
                "token_count": 1,
                "example_ids_digest": hashlib.sha256(b"[1]").hexdigest(),
            },
            "guarded_facts": {
                "weighted_logloss_sum": math.log(9.0),
                "token_count": 1,
                "example_ids_digest": hashlib.sha256(b"[1]").hexdigest(),
            },
            "bare_report": {
                "primary_metric": {"kind": "ppl_causal", "final": 9.0},
                "final": {
                    "window_ids": [1],
                    "logloss": [math.log(9.0)],
                    "token_counts": [1],
                },
                "status": "success",
            },
            "checks": {
                "metric_kind_matches": True,
                "measurements_valid": True,
                "guard_metric_impact": True,
                "arm_facts_replay": True,
            },
            "diagnostics": [],
            "source": "strict_fixture",
            "passed": True,
            "schedule_digest": schedule_digest,
        },
        "validation": {
            "primary_metric_acceptable": True,
            "preview_final_drift_acceptable": True,
            "invariants_pass": True,
            "spectral_stable": True,
            "rmt_stable": True,
        },
    }


def _strict_provenance_gate_cert() -> dict:
    payload = _provenance_gate_cert()
    window_count = 180
    preview_window_ids = list(range(window_count))
    final_window_ids = list(range(window_count, window_count * 2))
    schedule_digest = _final_window_schedule_digest(final_window_ids)
    payload["provenance"]["window_ids_digest"] = schedule_digest
    payload["provenance"]["window_plan_digest"] = schedule_digest
    payload["provenance"]["provider_digest"] = {
        "ids_sha256": "strict-schedule-ids",
        "tokenizer_sha256": "strict-tokenizer",
    }
    payload["guard_metric_impact"]["schedule_digest"] = schedule_digest
    payload["guard_metric_impact"].update(
        {
            "degradation": 0.0,
            "display_value": 0.0,
            "degradation_limit": 0.01,
        }
    )
    payload["dataset"]["windows"] = {
        "preview": window_count,
        "final": window_count,
        "stats": {
            "coverage": {
                "tier": "balanced",
                "preview": {
                    "used": window_count,
                    "required": window_count,
                    "ok": True,
                },
                "final": {
                    "used": window_count,
                    "required": window_count,
                    "ok": True,
                },
                "replicates": {"used": 1200, "required": 1200, "ok": True},
            },
            "actual_preview": window_count,
            "actual_final": window_count,
            "paired_windows": window_count,
            "window_match_fraction": 1.0,
            "window_overlap_fraction": 0.0,
            "window_pairing_reason": None,
            "preview_final_slice_delta_summary": {
                "mean": 0.0,
                "ci": [0.0, 0.0],
                "basis": "independent_disjoint_slices",
                "paired": False,
                "ci_method": "independent_percentile_delta_log",
                "ci_reason": None,
                "preview_windows": window_count,
                "final_windows": window_count,
                "degenerate": True,
                "degenerate_reason": "constant_bootstrap_distribution",
            },
            "bootstrap": {
                "enabled": True,
                "method": "bca_paired_delta_log",
                "alpha": 0.05,
                "replicates": 1200,
                "seed": 43,
                "preview_final_delta_basis": "independent_disjoint_slices",
                "preview_final_delta_method": ("independent_percentile_delta_log"),
                "preview_final_delta_seed": 140,
            },
        },
    }
    payload["evaluation_windows"] = {
        "preview": {
            "window_ids": preview_window_ids,
            "logloss": [math.log(9.0)] * window_count,
            "token_counts": [1] * window_count,
        },
        "final": {
            "window_ids": final_window_ids,
            "logloss": [math.log(9.0)] * window_count,
            "token_counts": [1] * window_count,
        },
    }
    payload["plugins"] = {
        "adapter": _plugin_metadata("adapters", "hf_causal"),
        "edit": _plugin_metadata("edits", "noop"),
        "guards": [_plugin_metadata("guards", name) for name in CANONICAL_GUARD_CHAIN],
    }
    payload["guards"] = []
    for index, name in enumerate(CANONICAL_GUARD_CHAIN):
        entry = {
            "name": name,
            "supported": True,
            "passed": True,
            "decision": "allow",
            "violations": [],
            "diagnostics": [],
        }
        if name == "invariants" and index == 0:
            entry["stage"] = "pre"
        if name == "invariants" and index == len(CANONICAL_GUARD_CHAIN) - 1:
            entry["stage"] = "post"
        payload["guards"].append(entry)
    payload["context"] = {
        "profile": "ci",
        "runtime": {"execution_mode": "container"},
        "guard_chain_observed": list(CANONICAL_GUARD_CHAIN),
    }
    payload["auto"] = {"tier": "balanced"}
    payload["meta"] = {
        "profile": "ci",
        "model_id": "strict-test-model",
        "adapter": "hf_causal",
        "tokenizer_hash": "strict-tokenizer",
        "model_identity": {"kind": "remote_revision", "revision": "a" * 40},
    }
    payload["edit"] = {"name": "noop"}
    payload["structure"] = {"params_changed": 0, "layers_modified": 0}
    payload["subject_ref"] = {
        "model_id": "strict-test-model",
        "adapter": "hf_causal",
        "model_identity": {"kind": "remote_revision", "revision": "a" * 40},
    }
    payload["dataset"].update(
        {
            "split": "validation",
            "hash": {
                "preview": "strict-preview-dataset",
                "final": "strict-final-dataset",
                "dataset": "strict-dataset",
            },
            "tokenizer": {"hash": "strict-tokenizer"},
        }
    )
    payload["baseline_ref"]["tokenizer_hash"] = "strict-tokenizer"
    spectral_contract = {"kind": "spectral_norm_power_iter", "version": 1}
    rmt_contract = {"kind": "activation_edge_risk", "version": 1}
    payload["spectral"] = {
        "supported": True,
        "passed": True,
        "decision": "allow",
        "violations": [],
        "status": "pass",
        "evaluated": True,
        "caps_applied": 0,
        "max_caps": 5,
        "caps_exceeded": False,
        "summary": {
            "status": "stable",
            "modules_checked": 2,
            "max_caps": 5,
            "caps_exceeded": False,
        },
        "measurement_contract": spectral_contract,
        "measurement_contract_hash": verify_mod._measurement_contract_digest(
            spectral_contract
        ),
        "measurement_contract_match": True,
    }
    payload["rmt"] = {
        "supported": True,
        "passed": True,
        "decision": "allow",
        "violations": [],
        "status": "stable",
        "evaluated": True,
        "stable": True,
        "epsilon_default": 0.01,
        "epsilon_by_family": {"ffn": 0.01},
        "epsilon_violations": [],
        "edge_risk_by_family_base": {"ffn": 1.0},
        "edge_risk_by_family": {"ffn": 1.0},
        "families": {
            "ffn": {
                "edge_base": 1.0,
                "edge_cur": 1.0,
                "epsilon": 0.01,
                "allowed": 1.01,
                "ratio": 1.0,
                "delta": 0.0,
            }
        },
        "measurement_contract": rmt_contract,
        "measurement_contract_hash": verify_mod._measurement_contract_digest(
            rmt_contract
        ),
        "measurement_contract_match": True,
    }
    payload["resolved_policy"] = {
        "spectral": {"measurement_contract": spectral_contract},
        "rmt": {"measurement_contract": rmt_contract},
        "metrics": {
            "accuracy": {
                "delta_min_pp": -1.0,
                "min_examples": 200,
                "min_examples_fraction": 0.01,
            }
        },
    }
    payload["policy_provenance"] = {"source": "runtime"}
    payload["variance"] = {
        "supported": True,
        "passed": True,
        "decision": "allow",
        "violations": [],
        "status": "pass",
        "enabled": False,
        "monitor_only": False,
        "predictive_gate": {
            "evaluated": True,
            "passed": True,
            "reason": "no_adjustment_required",
        },
        "calibration": {
            "status": "no_scaling_required",
            "coverage": 8,
            "min_coverage": 6,
        },
    }
    bind_raw_guard_evidence(payload)
    bind_noop_variance_evidence(payload)
    payload["invariants"] = {
        "supported": True,
        "passed": True,
        "decision": "allow",
        "violations": [],
        "status": "pass",
        "pre": "pass",
        "post": "pass",
        "summary": {
            "checks_performed": 2,
            "violations_found": 0,
            "fatal_violations": 0,
            "warning_violations": 0,
        },
        "failures": [],
    }
    payload["validation"]["guard_metric_impact_acceptable"] = True
    payload["primary_metric"]["ci"] = [0.0, 0.0]
    payload["assurance"] = {
        "mode": "strict",
        "profile": "ci",
        "tier": "balanced",
        "claim_set": ASSURANCE_CLAIM_SET,
        "canonical_guard_chain": list(CANONICAL_GUARD_CHAIN),
        "guard_chain_observed": list(CANONICAL_GUARD_CHAIN),
        "canonical_guard_chain_enforced": True,
        "fallback_fields_used": False,
        "runtime_provenance_verified": False,
        "runtime_provenance_declared": "container",
        "runtime_provenance_verification_status": "pending",
        "verdict": "pending_verifier",
        "report_local_verdict": "pass",
        "verified_assurance_verdict": "pending",
        "blocking_reasons": [],
    }
    bind_runtime_policy_receipt(payload)
    _bind_strict_baseline(payload, _ppl_baseline_for_subject(payload))
    return payload


def _matching_strict_ppl_baseline(payload: dict | None = None) -> dict:
    subject = payload if payload is not None else _strict_provenance_gate_cert()
    baseline = _ppl_baseline_for_subject(subject)
    _bind_strict_baseline(subject, baseline)
    return baseline


def _ppl_baseline_for_subject(subject: dict) -> dict:
    baseline_ref = subject.get("baseline_ref", {}).get("primary_metric", {})
    baseline_final = float(baseline_ref.get("final", 9.0))
    subject_final = subject.get("evaluation_windows", {}).get("final", {})
    window_ids = list(subject_final.get("window_ids", [2, 3]))
    token_counts = list(subject_final.get("token_counts", [1, 1]))
    baseline = _strict_baseline_run_core(subject)
    baseline["metrics"] = {
        "primary_metric": {
            "kind": str(baseline_ref.get("kind", "ppl_causal")),
            "preview": baseline_final,
            "final": baseline_final,
            "ratio_vs_baseline": 1.0,
            "ci": [0.0, 0.0],
            "display_ci": [1.0, 1.0],
        },
        "bootstrap": copy.deepcopy(
            subject.get("dataset", {})
            .get("windows", {})
            .get("stats", {})
            .get("bootstrap", {})
        ),
    }
    baseline["evaluation_windows"]["final"] = {
        "window_ids": window_ids,
        "logloss": [math.log(baseline_final)] * len(window_ids),
        "token_counts": token_counts,
    }
    return baseline


def _matching_strict_accuracy_baseline(payload: dict | None = None) -> dict:
    subject = payload if payload is not None else _strict_accuracy_cert()
    baseline = _strict_baseline_run_core(subject)
    baseline["metrics"] = copy.deepcopy(subject["metrics"])
    baseline["metrics"]["primary_metric"] = copy.deepcopy(subject["primary_metric"])
    baseline["metrics"]["primary_metric"]["ratio_vs_baseline"] = 0.0
    baseline["metrics"]["primary_metric"]["delta_vs_baseline_pp"] = 0.0
    _bind_strict_baseline(subject, baseline)
    return baseline


def _strict_baseline_run_core(subject: dict) -> dict:
    dataset = subject.get("dataset", {})
    return {
        "meta": {
            "run_id": "strict-baseline-run",
            "model_id": "strict-test-model",
            "adapter": "hf_causal",
            "tokenizer_hash": "strict-tokenizer",
            "commit": "strict-baseline-commit",
            "ts": "2026-07-09T00:00:00Z",
            "model_identity": {
                "kind": "remote_revision",
                "revision": "b" * 40,
            },
        },
        "context": {
            "profile": "ci",
            "auto": {"tier": "balanced"},
            "assurance": {"mode": "strict"},
        },
        "edit": {"name": "noop"},
        "data": {
            "dataset": dataset.get("provider"),
            "split": dataset.get("split"),
            "seq_len": dataset.get("seq_len"),
            "preview_n": dataset.get("windows", {}).get("preview"),
            "final_n": dataset.get("windows", {}).get("final"),
            "dataset_hash": dataset.get("hash", {}).get("dataset"),
            "tokenizer_hash": "strict-tokenizer",
        },
        "metrics": {},
        "evaluation_windows": copy.deepcopy(subject.get("evaluation_windows", {})),
        "provenance": {
            "provider_digest": copy.deepcopy(
                subject.get("provenance", {}).get("provider_digest", {})
            )
        },
        "artifacts": {},
        "guards": [],
    }


def _bind_strict_baseline(subject: dict, baseline: dict) -> None:
    report_hash = compute_report_digest(baseline)
    assert report_hash is not None
    provider_digest = copy.deepcopy(baseline["provenance"]["provider_digest"])
    subject["baseline_ref"].update(
        {
            "run_id": "strict-baseline-run",
            "model_id": "strict-test-model",
            "adapter": "hf_causal",
            "tokenizer_hash": "strict-tokenizer",
            "report_hash": report_hash,
            "provider_digest": provider_digest,
            "model_identity": copy.deepcopy(baseline["meta"]["model_identity"]),
        }
    )
    subject["provenance"]["baseline"] = {
        "run_id": "strict-baseline-run",
        "report_hash": report_hash,
    }


def _strict_accuracy_cert() -> dict:
    payload = _strict_provenance_gate_cert()
    correctness = [1] * 160 + [0] * 40
    preview_ids = [str(index) for index in range(200)]
    final_ids = [str(index) for index in range(200, 400)]
    preview_records = [
        {"id": example_id, "correct": bool(value)}
        for example_id, value in zip(preview_ids, correctness, strict=True)
    ]
    final_records = [
        {"id": example_id, "correct": bool(value)}
        for example_id, value in zip(final_ids, correctness, strict=True)
    ]
    payload["primary_metric"] = {
        "kind": "accuracy",
        "preview": 0.8,
        "final": 0.8,
        "delta_vs_baseline_pp": 0.0,
        "ci": [0.74, 0.85],
        "display_ci": [0.74, 0.85],
        "n_preview": 200,
        "n_final": 200,
        "counts_source": "measured",
        "estimated": False,
    }
    payload["metrics"] = {
        "classification": {
            "n_correct": 160,
            "n_total": 200,
            "counts_source": "measured",
            "estimated": False,
            "preview": {
                "correct_total": 160,
                "total": 200,
                "example_correct": correctness,
            },
            "final": {
                "correct_total": 160,
                "total": 200,
                "example_correct": correctness,
            },
        }
    }
    payload["evaluation_windows"] = {
        "preview": {
            "records": preview_records,
            "input_records": [{"id": value} for value in preview_ids],
            "example_ids": preview_ids,
        },
        "final": {
            "records": final_records,
            "input_records": [{"id": value} for value in final_ids],
            "example_ids": final_ids,
        },
    }
    arm_digest = hashlib.sha256(
        json.dumps(final_ids, separators=(",", ":")).encode()
    ).hexdigest()
    payload["guard_metric_impact"] = {
        "metric_kind": "accuracy",
        "direction": "higher",
        "degradation_basis": "absolute_drop",
        "bare_value": 0.8,
        "guarded_value": 0.8,
        "bare_facts": {
            "correct": 160,
            "total": 200,
            "example_ids_digest": arm_digest,
        },
        "guarded_facts": {
            "correct": 160,
            "total": 200,
            "example_ids_digest": arm_digest,
        },
        "bare_report": {
            "primary_metric": {"kind": "accuracy", "final": 0.8},
            "final": {
                "correct_total": 160,
                "total": 200,
                "example_ids": final_ids,
            },
            "status": "success",
        },
        "degradation": 0.0,
        "degradation_limit": 0.01,
        "display_value": 0.0,
        "display_unit": "percentage_points",
        "evaluated": True,
        "passed": True,
        "checks": {
            "metric_kind_matches": True,
            "measurements_valid": True,
            "guard_metric_impact": True,
            "arm_facts_replay": True,
        },
        "diagnostics": [],
        "source": "strict_fixture",
        "schedule_digest": _final_window_schedule_digest(list(range(200, 400))),
    }
    payload["provenance"]["provider_digest"]["ids_sha256"] = hash_json(list(range(400)))
    payload["dataset"]["windows"] = {
        "preview": 200,
        "final": 200,
        "stats": {
            "coverage": {
                "preview": {"used": 200, "required": 200, "ok": True},
                "final": {"used": 200, "required": 200, "ok": True},
            },
            "actual_preview": 200,
            "actual_final": 200,
            "paired_windows": 200,
            "window_match_fraction": 1.0,
            "window_overlap_fraction": 0.0,
        },
    }
    payload["baseline_ref"]["primary_metric"] = {
        "kind": "accuracy",
        "final": 0.8,
    }
    bind_noop_variance_evidence(payload)
    bind_runtime_policy_receipt(payload)
    _matching_strict_accuracy_baseline(payload)
    return payload


def _write_runtime_manifest(
    report_path: Path, *, execution_mode: str = "container"
) -> Path:
    payload = {
        "manifest_version": RUNTIME_MANIFEST_VERSION,
        "generated_at_utc": "2026-05-24T00:00:00+00:00",
        "verifier_contract_version": RUNTIME_VERIFIER_CONTRACT_VERSION,
        "report": {
            "path": str(report_path.resolve()),
            "filename": report_path.name,
            "sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
        },
        "config": {"path": None, "sha256": None, "source": "missing"},
        "execution_mode": execution_mode,
        "runtime": {
            "image_ref": "ghcr.io/invarlock/invarlock-runtime:test",
            "image_digest": _VALID_TEST_IMAGE_DIGEST,
            "container_execution": execution_mode == "container",
            "allow_network": False,
            "allow_remote_code": False,
            "allow_third_party_plugins": False,
        },
    }
    path = report_path.parent / RUNTIME_MANIFEST_FILENAME
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path
