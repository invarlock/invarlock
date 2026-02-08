#!/usr/bin/env python3
"""
schema_verify.py
================

Validate a verifier-carrying artifact against the v1 JSON Schemas and apply
lightweight cross-block consistency checks.

This is intentionally a small, standalone tool to support the S1 paper:
- Schema conformance (JSON Schema draft 2020-12)
- Required verifier trace contract fields
- Optional file hash checks for referenced artifacts
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import jsonschema
from referencing import Registry, Resource
from referencing.jsonschema import DRAFT202012

_VERIFIER_TRACE_SCHEMA_URI = (
    "https://invarlock.dev/schemas/verifier_trace.v1.schema.json"
)


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical_json_bytes(obj: Any) -> bytes:
    # Canonicalization for hashing: sorted keys, no whitespace, UTF-8.
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _compute_prompt_set_digest(prompt_set: dict[str, Any]) -> str:
    dataset_raw = prompt_set.get("dataset", {})
    items_raw = prompt_set.get("items", [])

    # Contract: digest is independent of embedded prompt text.
    dataset = {}
    if isinstance(dataset_raw, dict):
        for k in ("name", "config", "split", "revision"):
            if k in dataset_raw:
                dataset[k] = dataset_raw[k]

    items: list[dict[str, Any]] = []
    if isinstance(items_raw, list):
        for it in items_raw:
            if not isinstance(it, dict):
                continue
            items.append({"id": it.get("id"), "sha256": it.get("sha256")})

    payload = {"dataset": dataset, "items": items}
    return _sha256_hex(_canonical_json_bytes(payload))


def _validate_prompt_set(trace: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    tc = trace.get("trace_contract") or {}
    prompt_set = tc.get("prompt_set") or {}
    if not isinstance(prompt_set, dict):
        return errors

    mode = prompt_set.get("mode")
    items = prompt_set.get("items")
    if mode == "embedded":
        if isinstance(items, list):
            missing = [
                it.get("id")
                for it in items
                if isinstance(it, dict) and "text" not in it
            ]
            if missing:
                errors.append(
                    "prompt_set.mode=embedded but some items are missing text: "
                    + ", ".join(str(x) for x in missing[:5])
                )
            for it in items:
                if not isinstance(it, dict):
                    continue
                text = it.get("text")
                sha = it.get("sha256")
                if isinstance(text, str) and isinstance(sha, str):
                    actual = _sha256_hex(text.encode("utf-8"))
                    if sha != actual:
                        errors.append(
                            f"prompt_set item sha256 mismatch for id={it.get('id')}: "
                            f"recorded={sha} expected={actual}"
                        )

    digest = prompt_set.get("digest_sha256")
    if isinstance(digest, str):
        expected = _compute_prompt_set_digest(prompt_set)
        if digest != expected:
            errors.append(
                f"prompt_set.digest_sha256 mismatch: recorded={digest} expected={expected}"
            )
    return errors


def _validate_results_consistency(trace: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    results = trace.get("results") or {}
    summary = results.get("summary") if isinstance(results, dict) else None
    cases = results.get("cases") if isinstance(results, dict) else None
    if not (isinstance(summary, dict) and isinstance(cases, list)):
        return errors

    try:
        n_total = int(summary.get("n_total"))
        n_pass = int(summary.get("n_pass"))
        pass_rate = float(summary.get("pass_rate"))
    except Exception:
        return errors

    observed_total = len(cases)
    if n_total != observed_total:
        errors.append(
            f"results.summary.n_total={n_total} but cases has {observed_total}"
        )

    observed_pass = sum(
        1 for c in cases if isinstance(c, dict) and str(c.get("verdict")) == "pass"
    )
    if n_pass != observed_pass:
        errors.append(
            f"results.summary.n_pass={n_pass} but cases has {observed_pass} pass"
        )

    expected = (observed_pass / observed_total) if observed_total else 0.0
    if abs(pass_rate - expected) > 1e-9:
        errors.append(
            f"results.summary.pass_rate={pass_rate} but expected {expected} from cases"
        )
    return errors


def _validate_case_ids_match_prompt_set(trace: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    tc = trace.get("trace_contract") or {}
    prompt_set = tc.get("prompt_set") if isinstance(tc, dict) else None
    if not isinstance(prompt_set, dict):
        return errors

    items = prompt_set.get("items")
    if not isinstance(items, list):
        return errors
    expected_ids = []
    for it in items:
        if isinstance(it, dict) and isinstance(it.get("id"), str):
            expected_ids.append(it["id"])

    results = trace.get("results") or {}
    cases = results.get("cases") if isinstance(results, dict) else None
    if not isinstance(cases, list):
        return errors
    observed_ids = []
    for c in cases:
        if isinstance(c, dict) and isinstance(c.get("id"), str):
            observed_ids.append(c["id"])

    if expected_ids and observed_ids and expected_ids != observed_ids:
        errors.append(
            "results.cases ids do not exactly match trace_contract.prompt_set.items ids"
        )

    if len(set(observed_ids)) != len(observed_ids):
        errors.append("results.cases contains duplicate ids")

    return errors


def _validate_attempts_consistency(trace: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    results = trace.get("results") or {}
    summary = results.get("summary") if isinstance(results, dict) else None
    cases = results.get("cases") if isinstance(results, dict) else None
    if not (isinstance(summary, dict) and isinstance(cases, list)):
        return errors

    n_samples_per_case = summary.get("n_samples_per_case")
    if n_samples_per_case is not None:
        try:
            n_samples_per_case = int(n_samples_per_case)
        except Exception:
            n_samples_per_case = None

    tc = trace.get("trace_contract") or {}
    decoding = tc.get("decoding") if isinstance(tc, dict) else None
    num_samples = None
    if isinstance(decoding, dict) and decoding.get("num_samples") is not None:
        try:
            num_samples = int(decoding.get("num_samples"))
        except Exception:
            num_samples = None

    k = summary.get("k")
    if k is not None:
        try:
            k = int(k)
        except Exception:
            k = None

    if k is not None and num_samples is not None and num_samples < k:
        errors.append(
            f"results.summary.k={k} but trace_contract.decoding.num_samples={num_samples}"
        )

    if (
        n_samples_per_case is not None
        and num_samples is not None
        and n_samples_per_case != num_samples
    ):
        errors.append(
            "results.summary.n_samples_per_case does not match trace_contract.decoding.num_samples"
        )

    for c in cases:
        if not isinstance(c, dict):
            continue
        attempts = c.get("attempts")
        if attempts is None:
            continue
        if not isinstance(attempts, list) or not attempts:
            errors.append(
                f"case id={c.get('id')}: attempts must be a non-empty array when present"
            )
            continue

        if n_samples_per_case is not None and len(attempts) != n_samples_per_case:
            errors.append(
                f"case id={c.get('id')}: attempts has {len(attempts)} entries but n_samples_per_case={n_samples_per_case}"
            )

        attempt_ids: list[int] = []
        any_pass = False
        for a in attempts:
            if not isinstance(a, dict):
                continue
            try:
                attempt_ids.append(int(a.get("attempt_id")))
            except Exception:
                pass
            if str(a.get("verdict")) == "pass":
                any_pass = True

        if len(set(attempt_ids)) != len(attempt_ids):
            errors.append(
                f"case id={c.get('id')}: attempts contains duplicate attempt_id values"
            )

        case_verdict = str(c.get("verdict"))
        if case_verdict == "pass" and not any_pass:
            errors.append(f"case id={c.get('id')}: verdict=pass but no attempt passed")
        if case_verdict != "pass" and any_pass:
            errors.append(
                f"case id={c.get('id')}: verdict={case_verdict} but at least one attempt passed"
            )

    return errors


def _check_file_sha256(path: Path, expected: str) -> str | None:
    if not path.exists():
        return f"referenced file does not exist: {path}"
    actual = _sha256_hex(path.read_bytes())
    if actual != expected:
        return f"sha256 mismatch for {path}: recorded={expected} actual={actual}"
    return None


def _validate_file_refs(artifact: dict[str, Any], *, check_files: bool) -> list[str]:
    errors: list[str] = []
    ge = artifact.get("guard_evidence") if isinstance(artifact, dict) else None
    inv = ge.get("invarlock") if isinstance(ge, dict) else None
    eref = inv.get("evaluation_report") if isinstance(inv, dict) else None
    if isinstance(eref, dict):
        sha = eref.get("sha256")
        path = eref.get("path")
        if check_files and isinstance(path, str) and isinstance(sha, str):
            err = _check_file_sha256(Path(path), sha)
            if err:
                errors.append(err)
    return errors


def _load_schemas(schema_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    wrapper = _read_json(schema_root / "verifier_carrying_artifact.v1.schema.json")
    trace = _read_json(schema_root / "verifier_trace.v1.schema.json")
    return wrapper, trace


def validate_artifact(path: Path, *, schema_root: Path, check_files: bool) -> list[str]:
    artifact = _read_json(path)
    wrapper_schema, trace_schema = _load_schemas(schema_root)

    # Wrapper schema refs the trace schema by URL; register it explicitly so we
    # never depend on network retrieval.
    registry = Registry().with_resource(
        _VERIFIER_TRACE_SCHEMA_URI,
        Resource.from_contents(trace_schema, default_specification=DRAFT202012),
    )

    validator = jsonschema.Draft202012Validator(wrapper_schema, registry=registry)

    errors: list[str] = []
    for err in sorted(validator.iter_errors(artifact), key=str):
        errors.append(f"schema: {err.message}")

    # Cross-block checks (only run if schema-level is OK enough to parse).
    if isinstance(artifact, dict):
        errors.extend(_validate_file_refs(artifact, check_files=check_files))

        traces = artifact.get("verifier_traces")
        if isinstance(traces, list):
            for i, t in enumerate(traces):
                if not isinstance(t, dict):
                    continue
                for e in _validate_prompt_set(t):
                    errors.append(f"trace[{i}]: {e}")
                for e in _validate_results_consistency(t):
                    errors.append(f"trace[{i}]: {e}")
                for e in _validate_case_ids_match_prompt_set(t):
                    errors.append(f"trace[{i}]: {e}")
                for e in _validate_attempts_consistency(t):
                    errors.append(f"trace[{i}]: {e}")

    return errors


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "artifact", type=Path, help="Path to verifier-carrying artifact JSON."
    )
    p.add_argument(
        "--schema-root",
        type=Path,
        default=Path("research/verifai2_2026/specs"),
        help="Directory containing v1 schema JSON files.",
    )
    p.add_argument(
        "--check-files",
        action="store_true",
        help="If set, verify sha256 for any referenced local files with path+sha256.",
    )
    args = p.parse_args(argv)

    errors = validate_artifact(
        args.artifact, schema_root=args.schema_root, check_files=args.check_files
    )
    if errors:
        for e in errors:
            print(e, file=sys.stderr)
        return 2

    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
