from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

COLUMNS = ["journey", "expectation", "status", "verify", "metric", "artifact", "note"]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_result_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _clean_table_cell(value: object) -> str:
    text = str(value or "").replace("|", "\\|")
    return text if text else "-"


def metric_summary(args: argparse.Namespace) -> int:
    report = _read_json(args.report)
    pm = report.get("primary_metric", {})
    ratio = pm.get("ratio_vs_baseline")
    ci = pm.get("display_ci") or pm.get("ci")
    kind = pm.get("kind", "metric")
    if isinstance(ratio, (int, float)) and isinstance(ci, list) and len(ci) == 2:
        print(f"{kind} ratio={ratio:.3f} ci={ci[0]:.3f}-{ci[1]:.3f}")
    elif isinstance(ratio, (int, float)):
        print(f"{kind} ratio={ratio:.3f}")
    else:
        print(f"{kind} metric=n/a")
    return 0


def verify_reason(args: argparse.Namespace) -> int:
    payload = _read_json(args.verify_json)
    print(payload.get("summary", {}).get("reason", "unknown"))
    return 0


def print_results_table(args: argparse.Namespace) -> int:
    rows = _read_result_rows(args.results_tsv)
    print("")
    print("GPT-2 User Journey Smoke Results")
    print("")
    print("| " + " | ".join(COLUMNS) + " |")
    print("| " + " | ".join("---" for _ in COLUMNS) + " |")
    for row in rows:
        print("| " + " | ".join(_clean_table_cell(row.get(column, "")) for column in COLUMNS) + " |")
    print("")
    passed = sum(row.get("status") == "PASS" for row in rows)
    skipped = sum(row.get("status") == "SKIP" for row in rows)
    failed = sum(row.get("status") not in {"PASS", "SKIP"} for row in rows)
    print(f"Summary: {passed} passed, {skipped} skipped, {failed} failed.")
    return 0


def write_final_verdict(args: argparse.Namespace) -> int:
    rows = _read_result_rows(args.results_tsv)
    payload = {
        "verdict": args.verdict,
        "note": "gpt2 user journey smoke",
        "summary": {
            "total": len(rows),
            "passed": sum(row.get("status") == "PASS" for row in rows),
            "skipped": sum(row.get("status") == "SKIP" for row in rows),
            "failed": sum(row.get("status") not in {"PASS", "SKIP"} for row in rows),
        },
        "journeys": rows,
    }
    _write_json(args.output, payload)
    return 0


def mutate_negative_report(args: argparse.Namespace) -> int:
    report = _read_json(args.source_report)
    report.setdefault("primary_metric", {})["display_ci"] = [1.20, 1.30]
    report.setdefault("meta", {})["failure_smoke_mutation"] = (
        "display_ci intentionally diverges from exp(ci)"
    )
    args.target_report.parent.mkdir(parents=True, exist_ok=True)
    _write_json(args.target_report, report)
    return 0


def write_strict_bundle_fixture(args: argparse.Namespace) -> int:
    from invarlock.core.assurance_contract import (
        ASSURANCE_CLAIM_SET,
        CANONICAL_GUARD_CHAIN,
    )
    from invarlock.reporting import verify_contract as verify_mod
    from invarlock.runtime_security import RUNTIME_VERIFIER_CONTRACT_VERSION

    report_path = args.report
    report_path.parent.mkdir(parents=True, exist_ok=True)

    spectral_contract = {"estimator": {"type": "power_iter", "iters": 4, "init": "ones"}}
    rmt_contract = {
        "estimator": {"type": "power_iter", "iters": 3, "init": "ones"},
        "activation_sampling": {
            "windows": {"count": 8, "indices_policy": "evenly_spaced"}
        },
    }
    guard_chain = list(CANONICAL_GUARD_CHAIN)
    report = {
        "schema_version": "v1",
        "run_id": "evidence-pack-wheel-smoke",
        "artifacts": {"generated_at": "2024-01-01T00:00:00"},
        "plugins": {"guards": guard_chain},
        "guards": [{"name": name} for name in guard_chain],
        "meta": {"profile": "ci"},
        "context": {
            "profile": "ci",
            "runtime": {"execution_mode": "container"},
        },
        "auto": {"tier": "balanced"},
        "dataset": {
            "provider": "unit",
            "seq_len": 8,
            "windows": {
                "preview": 2,
                "final": 2,
                "stats": {
                    "window_match_fraction": 1.0,
                    "window_overlap_fraction": 0.0,
                    "coverage": {"preview": {"used": 2}, "final": {"used": 2}},
                    "paired_windows": 2,
                },
            },
        },
        "validation": {
            "primary_metric_acceptable": True,
            "preview_final_drift_acceptable": True,
            "invariants_pass": True,
            "spectral_stable": True,
            "rmt_stable": True,
        },
        "baseline_ref": {
            "run_id": "baseline-run",
            "model_id": "model",
            "primary_metric": {"kind": "ppl_causal", "final": 10.0},
        },
        "provenance": {"provider_digest": {"ids_sha256": "subject-ids"}},
        "artifacts_extra": {},
        "report_build": {
            "synthesized_fields": [],
            "repaired_fields": [],
            "fallback_fields": [],
        },
        "primary_metric": {
            "kind": "ppl_causal",
            "final": 10.0,
            "preview": 10.0,
            "ratio_vs_baseline": 1.0,
            "ci": [0.0, 0.0],
            "display_ci": [1.0, 1.0],
        },
        "spectral": {
            "evaluated": True,
            "supported": True,
            "status": "pass",
            "measurement_contract": spectral_contract,
            "measurement_contract_hash": verify_mod._measurement_contract_digest(
                spectral_contract
            ),
            "measurement_contract_match": True,
        },
        "rmt": {
            "evaluated": True,
            "supported": True,
            "status": "pass",
            "measurement_contract": rmt_contract,
            "measurement_contract_hash": verify_mod._measurement_contract_digest(
                rmt_contract
            ),
            "measurement_contract_match": True,
        },
        "variance": {"supported": True, "status": "pass"},
        "invariants": {"supported": True, "status": "pass"},
        "resolved_policy": {
            "spectral": {"measurement_contract": spectral_contract},
            "rmt": {"measurement_contract": rmt_contract},
        },
        "evaluation_windows": {
            "final": {
                "logloss": [math.log(10.0)],
                "token_counts": [1],
            }
        },
        "assurance": {
            "mode": "strict",
            "profile": "ci",
            "tier": "balanced",
            "claim_set": ASSURANCE_CLAIM_SET,
            "canonical_guard_chain": guard_chain,
            "guard_chain_observed": guard_chain,
            "canonical_guard_chain_enforced": True,
            "fallback_fields_used": False,
            "runtime_provenance_verified": False,
            "runtime_provenance_declared": "container",
            "runtime_provenance_verification_status": "pending",
            "verdict": "pending_verifier",
            "report_local_verdict": "pass",
            "verified_assurance_verdict": "pending",
            "blocking_reasons": [],
        },
    }

    _write_json(report_path, report)
    report_sha = hashlib.sha256(report_path.read_bytes()).hexdigest()
    manifest = {
        "manifest_version": 1,
        "generated_at_utc": "2026-05-25T00:00:00+00:00",
        "verifier_contract_version": RUNTIME_VERIFIER_CONTRACT_VERSION,
        "execution_mode": "container",
        "report": {
            "filename": report_path.name,
            "path": report_path.as_posix(),
            "sha256": report_sha,
        },
        "config": {
            "path": None,
            "sha256": None,
            "source": "missing",
        },
        "runtime": {
            "container_execution": True,
            "image_digest": "sha256:" + ("a" * 64),
            "image_ref": "invarlock-runtime:local",
            "allow_network": False,
            "allow_remote_code": False,
            "allow_third_party_plugins": False,
        },
    }
    _write_json(report_path.parent / "runtime.manifest.json", manifest)
    return 0


def append_child_results(args: argparse.Namespace) -> int:
    payload = _read_json(args.final_verdict)
    rows = payload.get("journeys", [])
    failed = int(payload.get("summary", {}).get("failed", 0) or 0)
    with args.results_tsv.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=COLUMNS, delimiter="\t", lineterminator="\n"
        )
        for row in rows:
            if not isinstance(row, dict):
                continue
            out = {column: str(row.get(column, "") or "") for column in COLUMNS}
            out["journey"] = f"{args.suite}/{out['journey']}" if out["journey"] else args.suite
            writer.writerow(out)
    if payload.get("verdict") != "PASS" or failed:
        return 1
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GPT-2 smoke helper utilities.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    metric_parser = subparsers.add_parser("metric-summary")
    metric_parser.add_argument("report", type=Path)
    metric_parser.set_defaults(func=metric_summary)

    reason_parser = subparsers.add_parser("verify-reason")
    reason_parser.add_argument("verify_json", type=Path)
    reason_parser.set_defaults(func=verify_reason)

    table_parser = subparsers.add_parser("print-results-table")
    table_parser.add_argument("results_tsv", type=Path)
    table_parser.set_defaults(func=print_results_table)

    verdict_parser = subparsers.add_parser("write-final-verdict")
    verdict_parser.add_argument("results_tsv", type=Path)
    verdict_parser.add_argument("output", type=Path)
    verdict_parser.add_argument("verdict", choices=("PASS", "FAIL"))
    verdict_parser.set_defaults(func=write_final_verdict)

    mutate_parser = subparsers.add_parser("mutate-negative-report")
    mutate_parser.add_argument("source_report", type=Path)
    mutate_parser.add_argument("target_report", type=Path)
    mutate_parser.set_defaults(func=mutate_negative_report)

    strict_parser = subparsers.add_parser("write-strict-bundle-fixture")
    strict_parser.add_argument("report", type=Path)
    strict_parser.set_defaults(func=write_strict_bundle_fixture)

    append_parser = subparsers.add_parser("append-child-results")
    append_parser.add_argument("results_tsv", type=Path)
    append_parser.add_argument("final_verdict", type=Path)
    append_parser.add_argument("suite")
    append_parser.set_defaults(func=append_child_results)
    return parser


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
