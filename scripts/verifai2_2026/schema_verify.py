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


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical_json_bytes(obj: Any) -> bytes:
    # Canonicalization for hashing: sorted keys, no whitespace, UTF-8.
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    )


def _compute_prompt_set_digest(prompt_set: dict[str, Any]) -> str:
    dataset = prompt_set.get("dataset", {})
    items = prompt_set.get("items", [])
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
            missing = [it.get("id") for it in items if isinstance(it, dict) and "text" not in it]
            if missing:
                errors.append(
                    "prompt_set.mode=embedded but some items are missing text: "
                    + ", ".join(str(x) for x in missing[:5])
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
        pass_at_1 = float(summary.get("pass_at_1"))
    except Exception:
        return errors

    observed_total = len(cases)
    if n_total != observed_total:
        errors.append(f"results.summary.n_total={n_total} but cases has {observed_total}")

    observed_pass = sum(
        1 for c in cases if isinstance(c, dict) and str(c.get("verdict")) == "pass"
    )
    if n_pass != observed_pass:
        errors.append(f"results.summary.n_pass={n_pass} but cases has {observed_pass} pass")

    expected = (observed_pass / observed_total) if observed_total else 0.0
    if abs(pass_at_1 - expected) > 1e-9:
        errors.append(
            f"results.summary.pass_at_1={pass_at_1} but expected {expected} from cases"
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

    store = {}
    trace_id = trace_schema.get("$id")
    if isinstance(trace_id, str) and trace_id:
        store[trace_id] = trace_schema

    resolver = jsonschema.RefResolver.from_schema(wrapper_schema, store=store)
    validator = jsonschema.Draft202012Validator(wrapper_schema, resolver=resolver)

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

    return errors


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("artifact", type=Path, help="Path to verifier-carrying artifact JSON.")
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

