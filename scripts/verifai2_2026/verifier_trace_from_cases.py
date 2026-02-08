#!/usr/bin/env python3
"""
verifier_trace_from_cases.py
============================

Build a `verifier_trace.v1` JSON record from:
- a prompt_set JSON (as produced by make_prompt_set.py)
- a cases JSONL file (per-id verdicts, optionally per-attempt)

This is the "exporter" glue for S1/F4: it converts harness outputs into a
machine-checkable trace record that can be embedded into the
verifier-carrying artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

VERDICT_ENUM = {"pass", "fail", "error", "timeout", "skipped"}


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_json_bytes(obj: Any) -> bytes:
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{i}: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"Expected JSON object at {path}:{i}")
            out.append(obj)
    return out


def _compute_prompt_set_digest(prompt_set: dict[str, Any]) -> str:
    dataset_raw = prompt_set.get("dataset", {})
    dataset = {}
    if isinstance(dataset_raw, dict):
        for k in ("name", "config", "split", "revision"):
            if k in dataset_raw:
                dataset[k] = dataset_raw[k]
    items_raw = prompt_set.get("items", [])
    items = []
    if isinstance(items_raw, list):
        for it in items_raw:
            if not isinstance(it, dict):
                continue
            items.append({"id": it.get("id"), "sha256": it.get("sha256")})
    payload = {"dataset": dataset, "items": items}
    return _sha256_hex(_canonical_json_bytes(payload))


def _normalize_case_record(rec: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}

    rid = rec.get("id")
    if not isinstance(rid, str) or not rid:
        raise ValueError("case record missing id")
    out["id"] = rid

    verdict = rec.get("verdict")
    if not isinstance(verdict, str) or verdict not in VERDICT_ENUM:
        raise ValueError(f"case record id={rid} has invalid verdict={verdict!r}")
    out["verdict"] = verdict

    if rec.get("error_type") is not None:
        out["error_type"] = str(rec.get("error_type"))

    if rec.get("wall_time_s") is not None:
        try:
            out["wall_time_s"] = float(rec.get("wall_time_s"))
        except Exception:
            pass

    # Optional: output/stderr raw strings -> sha256
    if rec.get("output_sha256") is not None:
        out["output_sha256"] = str(rec.get("output_sha256"))
    elif isinstance(rec.get("output"), str):
        out["output_sha256"] = _sha256_hex(rec["output"].encode("utf-8"))

    if rec.get("stderr_sha256") is not None:
        out["stderr_sha256"] = str(rec.get("stderr_sha256"))
    elif isinstance(rec.get("stderr"), str):
        out["stderr_sha256"] = _sha256_hex(rec["stderr"].encode("utf-8"))

    # Optional: counterexample slice
    failing = rec.get("failing_test_ids")
    msg = rec.get("message_excerpt")
    if failing is not None or msg is not None:
        ce: dict[str, Any] = {}
        if isinstance(failing, list) and all(isinstance(x, str) for x in failing):
            ce["failing_test_ids"] = failing
        if isinstance(msg, str) and msg:
            ce["message_excerpt"] = msg
            ce["message_sha256"] = _sha256_hex(msg.encode("utf-8"))
        out["counterexample"] = ce

    return out


def _aggregate_verdict(attempt_verdicts: list[str]) -> str:
    if any(v == "pass" for v in attempt_verdicts):
        return "pass"
    # Prefer explicit functional failure over infrastructure errors/timeouts.
    if any(v == "fail" for v in attempt_verdicts):
        return "fail"
    if any(v == "timeout" for v in attempt_verdicts):
        return "timeout"
    if any(v == "error" for v in attempt_verdicts):
        return "error"
    return "skipped"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--prompt-set", type=Path, required=True)
    p.add_argument("--cases", type=Path, required=True, help="Cases JSONL.")
    p.add_argument(
        "--out", type=Path, required=True, help="Output verifier_trace.v1 JSON."
    )

    # Verifier identity
    p.add_argument("--verifier-name", type=str, required=True)
    p.add_argument(
        "--verifier-kind",
        type=str,
        required=True,
        choices=["code_execution", "proof_checker", "smt_solver", "static_analyzer"],
    )
    p.add_argument("--harness-name", type=str, required=True)
    p.add_argument("--harness-version", type=str, default="")
    p.add_argument("--harness-git-commit", type=str, default="")
    p.add_argument("--harness-container-image", type=str, default="")
    p.add_argument(
        "--harness-config",
        type=Path,
        help="Optional JSON/YAML-ish config file whose bytes are hashed as harness.config_digest_sha256.",
    )

    # Sandbox (required for code_execution)
    p.add_argument("--sandbox-timeout-s", type=float, default=10.0)
    p.add_argument("--sandbox-cpu-limit", type=int, default=2)
    p.add_argument("--sandbox-mem-limit-mb", type=int, default=2048)
    p.add_argument("--sandbox-wall-limit-s", type=float, default=10.0)

    # Trace contract: model/tokenizer/decoding
    p.add_argument("--model-id", type=str, required=True)
    p.add_argument("--model-revision", type=str, required=True)
    p.add_argument("--tokenizer-id", type=str, required=True)
    p.add_argument("--tokenizer-revision", type=str, required=True)

    p.add_argument(
        "--decoding-method",
        type=str,
        choices=["greedy", "sample", "beam"],
        required=True,
    )
    p.add_argument("--temperature", type=float, required=True)
    p.add_argument("--top-p", type=float, required=True)
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--max-new-tokens", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument(
        "--num-samples",
        type=int,
        default=0,
        help="Optional generations per prompt (k).",
    )

    p.add_argument("--metric-name", type=str, default="pass@1")
    p.add_argument(
        "--k", type=int, default=0, help="Optional pass@k value for metric_name/pass@k."
    )
    args = p.parse_args(argv)

    prompt_set_obj = _read_json(args.prompt_set)
    if not isinstance(prompt_set_obj, dict):
        print("prompt_set must be a JSON object", file=sys.stderr)
        return 2

    recorded = prompt_set_obj.get("digest_sha256")
    expected = _compute_prompt_set_digest(prompt_set_obj)
    if isinstance(recorded, str) and recorded != expected:
        print(
            f"prompt_set.digest_sha256 mismatch: recorded={recorded} expected={expected}",
            file=sys.stderr,
        )
        return 2

    items = prompt_set_obj.get("items")
    if not isinstance(items, list) or not items:
        print("prompt_set.items must be a non-empty array", file=sys.stderr)
        return 2
    item_ids: list[str] = []
    for it in items:
        if not isinstance(it, dict) or not isinstance(it.get("id"), str):
            print(
                "prompt_set.items entries must be objects with string id",
                file=sys.stderr,
            )
            return 2
        item_ids.append(it["id"])

    case_rows = _read_jsonl(args.cases)
    # Map id -> list of attempt records (attempt_id optional)
    by_id: dict[str, list[dict[str, Any]]] = {}
    for row in case_rows:
        norm = _normalize_case_record(row)
        rid = norm["id"]
        attempt_id = row.get("attempt_id")
        if attempt_id is not None:
            try:
                norm["attempt_id"] = int(attempt_id)
            except Exception:
                pass
        by_id.setdefault(rid, []).append(norm)

    cases_out: list[dict[str, Any]] = []
    for rid in item_ids:
        attempts = by_id.get(rid, [])
        if not attempts:
            cases_out.append(
                {"id": rid, "verdict": "error", "error_type": "missing_result"}
            )
            continue

        # If attempt_id is provided, preserve that ordering; else preserve input order.
        has_attempt_ids = any("attempt_id" in a for a in attempts)
        if has_attempt_ids:
            attempts = sorted(attempts, key=lambda a: int(a.get("attempt_id", 0)))

        if len(attempts) == 1 and "attempt_id" not in attempts[0]:
            # pass@1 style
            case = dict(attempts[0])
            case.pop("attempt_id", None)
            cases_out.append(case)
            continue

        attempt_verdicts = [str(a.get("verdict")) for a in attempts]
        case_verdict = _aggregate_verdict(attempt_verdicts)
        case: dict[str, Any] = {"id": rid, "verdict": case_verdict}
        case["attempts"] = [
            {k: v for k, v in a.items() if k != "id"}
            | {"attempt_id": int(a.get("attempt_id", i))}
            for i, a in enumerate(attempts)
        ]
        cases_out.append(case)

    n_total = len(cases_out)
    n_pass = sum(1 for c in cases_out if c.get("verdict") == "pass")
    pass_rate = (n_pass / n_total) if n_total else 0.0

    k = int(args.k) if args.k and args.k > 0 else None
    num_samples = (
        int(args.num_samples) if args.num_samples and args.num_samples > 0 else None
    )

    summary: dict[str, Any] = {
        "n_total": n_total,
        "n_pass": n_pass,
        "pass_rate": pass_rate,
        "metric_name": str(args.metric_name),
    }
    if k is not None:
        summary["k"] = k
    if num_samples is not None:
        summary["n_samples_per_case"] = num_samples

    harness: dict[str, Any] = {"name": str(args.harness_name)}
    if args.harness_version:
        harness["version"] = str(args.harness_version)
    if args.harness_git_commit:
        harness["git_commit"] = str(args.harness_git_commit)
    if args.harness_container_image:
        harness["container_image"] = str(args.harness_container_image)
    if args.harness_config is not None:
        harness["config_digest_sha256"] = _sha256_hex(args.harness_config.read_bytes())
    if len(harness.keys()) == 1:
        print(
            "Harness identity incomplete: set --harness-version or --harness-git-commit or --harness-container-image",
            file=sys.stderr,
        )
        return 2

    verifier: dict[str, Any] = {
        "name": str(args.verifier_name),
        "kind": str(args.verifier_kind),
        "harness": harness,
    }
    if args.verifier_kind == "code_execution":
        verifier["sandbox"] = {
            "network_enabled": False,
            "timeout_s": float(args.sandbox_timeout_s),
            "cpu_limit": int(args.sandbox_cpu_limit),
            "mem_limit_mb": int(args.sandbox_mem_limit_mb),
            "wall_limit_s": float(args.sandbox_wall_limit_s),
        }

    trace_contract: dict[str, Any] = {
        "prompt_set": prompt_set_obj,
        "model": {"id": str(args.model_id), "revision": str(args.model_revision)},
        "tokenizer": {
            "id": str(args.tokenizer_id),
            "revision": str(args.tokenizer_revision),
        },
        "decoding": {
            "method": str(args.decoding_method),
            "temperature": float(args.temperature),
            "top_p": float(args.top_p),
            "top_k": int(args.top_k),
            "max_new_tokens": int(args.max_new_tokens),
            "seed": int(args.seed),
        },
    }
    if num_samples is not None:
        trace_contract["decoding"]["num_samples"] = num_samples

    trace: dict[str, Any] = {
        "schema_version": "verifier_trace.v1",
        "verifier": verifier,
        "trace_contract": trace_contract,
        "results": {"summary": summary, "cases": cases_out},
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(trace, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
