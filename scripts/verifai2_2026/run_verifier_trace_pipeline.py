#!/usr/bin/env python3
"""
run_verifier_trace_pipeline.py
==============================

Deterministic-ish wrapper to produce a verifier_trace.v1 record from:
  - prompts/tasks (prompt canonicalization -> prompt_set)
  - verifier outcomes (either by running a verifier or ingesting harness outputs)

This script exists to make the F4/S1 workflow less error-prone:
  - builds a prompt_set with stable hashing rules
  - produces normalized cases.jsonl
  - exports verifier_trace.v1 with pinned contract fields

It intentionally does *not* download datasets or models. In workshop-scale runs,
fetch data/models in the surrounding pipeline (respecting repo network policy).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from scripts.verifai2_2026 import (
    cases_from_harness,
    make_prompt_set,
    run_code_tests_verifier,
    verifier_trace_from_cases,
)


def _read_jsonl_ids(path: Path, *, id_field: str) -> list[str]:
    ids: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict) and isinstance(obj.get(id_field), str):
                ids.append(obj[id_field])
    return ids


def _validate_decoding(
    *,
    method: str,
    temperature: float,
    top_p: float,
    top_k: int,
    num_samples: int,
    k: int,
) -> list[str]:
    errors: list[str] = []
    if method == "greedy":
        if temperature != 0.0:
            errors.append("decoding.method=greedy requires temperature=0.0")
        if top_p != 1.0:
            errors.append("decoding.method=greedy requires top_p=1.0")
        if top_k != 0:
            errors.append("decoding.method=greedy requires top_k=0")
        if num_samples not in {0, 1}:
            errors.append("decoding.method=greedy requires num_samples unset/1")
    if k > 0 and num_samples > 0 and num_samples < k:
        errors.append("num_samples must be >= k for pass@k")
    return errors


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--backend",
        type=str,
        choices=["code_tests", "harness_jsonl", "cases_jsonl"],
        default="code_tests",
        help="How to obtain per-case verdicts.",
    )

    # Inputs (prompts/tasks/completions)
    p.add_argument(
        "--prompts",
        type=Path,
        help="JSONL containing prompts used for generation (id + prompt text).",
    )
    p.add_argument(
        "--tasks",
        type=Path,
        help="JSONL tasks (id + prompt + tests) for code_tests backend.",
    )
    p.add_argument(
        "--completions",
        type=Path,
        help="JSONL completions (id + completion[, attempt_id]) for code_tests backend.",
    )
    p.add_argument(
        "--harness-results",
        type=Path,
        help="Harness output JSONL for harness_jsonl backend.",
    )
    p.add_argument(
        "--cases", type=Path, help="Precomputed cases JSONL (cases_jsonl backend)."
    )

    # Outputs
    p.add_argument("--prompt-set-out", type=Path, required=True)
    p.add_argument("--cases-out", type=Path, required=True)
    p.add_argument("--trace-out", type=Path, required=True)

    # Prompt-set options
    p.add_argument("--prompt-id-field", type=str, default="id")
    p.add_argument("--prompt-text-field", type=str, default="prompt")
    p.add_argument("--prompt-template", type=str, default="{text}")
    p.add_argument(
        "--prompt-mode",
        type=str,
        choices=["hash_only", "embedded"],
        default="hash_only",
    )
    p.add_argument("--dataset-name", type=str, default="local")
    p.add_argument("--dataset-config", type=str, default="")
    p.add_argument("--dataset-split", type=str, default="test")
    p.add_argument("--dataset-revision", type=str, default="")
    p.add_argument("--dataset-manifest-sha256", type=str, default="")
    p.add_argument("--selection-script", type=Path)
    p.add_argument("--limit", type=int, default=0)

    # Harness ingestion field mapping (harness_jsonl backend)
    p.add_argument("--harness-id-field", type=str, default="id")
    p.add_argument("--harness-attempt-field", type=str, default="attempt_id")
    p.add_argument("--harness-completion-field", type=str, default="completion")
    p.add_argument("--harness-stderr-field", type=str, default="stderr")
    p.add_argument("--harness-verdict-field", type=str, default="verdict")
    p.add_argument("--harness-passed-field", type=str, default="passed")
    p.add_argument("--harness-status-field", type=str, default="status")
    p.add_argument("--harness-wall-time-field", type=str, default="wall_time_s")
    p.add_argument("--harness-error-type-field", type=str, default="error_type")
    p.add_argument(
        "--harness-failing-tests-field", type=str, default="failing_test_ids"
    )
    p.add_argument("--harness-message-field", type=str, default="message_excerpt")
    p.add_argument("--harness-non-strict", action="store_true")

    # Verifier identity (exporter args)
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
    p.add_argument("--harness-config", type=Path)

    # Sandbox (required for code_execution)
    p.add_argument("--sandbox-timeout-s", type=float, default=10.0)
    p.add_argument("--sandbox-cpu-limit", type=int, default=2)
    p.add_argument("--sandbox-mem-limit-mb", type=int, default=2048)
    p.add_argument("--sandbox-wall-limit-s", type=float, default=10.0)

    # Trace contract
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
    p.add_argument("--num-samples", type=int, default=0)
    p.add_argument("--metric-name", type=str, default="pass@1")
    p.add_argument("--k", type=int, default=0)

    # code_tests backend options
    p.add_argument("--timeout-s", type=float, default=10.0)
    p.add_argument("--cpu-limit-s", type=int, default=10)
    p.add_argument("--mem-limit-mb", type=int, default=2048)
    p.add_argument("--python", type=str, default=sys.executable)

    args = p.parse_args(argv)

    prompts_path = args.prompts or args.tasks
    if prompts_path is None:
        print("--prompts or --tasks is required", file=sys.stderr)
        return 2

    decoding_errors = _validate_decoding(
        method=str(args.decoding_method),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        top_k=int(args.top_k),
        num_samples=int(args.num_samples),
        k=int(args.k),
    )
    if decoding_errors:
        for e in decoding_errors:
            print(e, file=sys.stderr)
        return 2

    # 1) prompt_set
    rc = make_prompt_set.main(
        [
            "--in",
            str(prompts_path),
            "--id-field",
            str(args.prompt_id_field),
            "--text-field",
            str(args.prompt_text_field),
            "--template",
            str(args.prompt_template),
            "--mode",
            str(args.prompt_mode),
            "--dataset-name",
            str(args.dataset_name),
            "--dataset-config",
            str(args.dataset_config),
            "--dataset-split",
            str(args.dataset_split),
            "--dataset-revision",
            str(args.dataset_revision),
            "--dataset-manifest-sha256",
            str(args.dataset_manifest_sha256),
            "--out",
            str(args.prompt_set_out),
            *(
                ["--selection-script", str(args.selection_script)]
                if args.selection_script is not None
                else []
            ),
            *(
                ["--limit", str(int(args.limit))]
                if args.limit is not None and int(args.limit) > 0
                else []
            ),
        ]
    )
    if rc != 0:
        return rc

    # 2) cases
    args.cases_out.parent.mkdir(parents=True, exist_ok=True)
    if args.backend == "code_tests":
        if args.tasks is None or args.completions is None:
            print(
                "--tasks and --completions are required for backend=code_tests",
                file=sys.stderr,
            )
            return 2
        rc = run_code_tests_verifier.main(
            [
                "--tasks",
                str(args.tasks),
                "--completions",
                str(args.completions),
                "--out-cases",
                str(args.cases_out),
                "--python",
                str(args.python),
                "--timeout-s",
                str(float(args.timeout_s)),
                "--cpu-limit-s",
                str(int(args.cpu_limit_s)),
                "--mem-limit-mb",
                str(int(args.mem_limit_mb)),
            ]
        )
        if rc != 0:
            return rc
    elif args.backend == "harness_jsonl":
        if args.harness_results is None:
            print(
                "--harness-results is required for backend=harness_jsonl",
                file=sys.stderr,
            )
            return 2
        rc = cases_from_harness.main(
            [
                "--in",
                str(args.harness_results),
                "--out",
                str(args.cases_out),
                "--id-field",
                str(args.harness_id_field),
                "--attempt-field",
                str(args.harness_attempt_field),
                "--completion-field",
                str(args.harness_completion_field),
                "--stderr-field",
                str(args.harness_stderr_field),
                "--verdict-field",
                str(args.harness_verdict_field),
                "--passed-field",
                str(args.harness_passed_field),
                "--status-field",
                str(args.harness_status_field),
                "--wall-time-field",
                str(args.harness_wall_time_field),
                "--error-type-field",
                str(args.harness_error_type_field),
                "--failing-tests-field",
                str(args.harness_failing_tests_field),
                "--message-field",
                str(args.harness_message_field),
                *(["--non-strict"] if bool(args.harness_non_strict) else []),
            ]
        )
        if rc != 0:
            return rc
    else:
        if args.cases is None:
            print("--cases is required for backend=cases_jsonl", file=sys.stderr)
            return 2
        if args.cases.resolve() != args.cases_out.resolve():
            shutil.copyfile(args.cases, args.cases_out)

    # Sanity: prompt ids should exist. (Strict id matching is enforced later.)
    if not _read_jsonl_ids(args.cases_out, id_field="id"):
        print("cases-out produced no ids; check backend inputs.", file=sys.stderr)
        return 2

    # 3) trace
    rc = verifier_trace_from_cases.main(
        [
            "--prompt-set",
            str(args.prompt_set_out),
            "--cases",
            str(args.cases_out),
            "--out",
            str(args.trace_out),
            "--verifier-name",
            str(args.verifier_name),
            "--verifier-kind",
            str(args.verifier_kind),
            "--harness-name",
            str(args.harness_name),
            "--harness-version",
            str(args.harness_version),
            "--harness-git-commit",
            str(args.harness_git_commit),
            "--harness-container-image",
            str(args.harness_container_image),
            *(
                ["--harness-config", str(args.harness_config)]
                if args.harness_config is not None
                else []
            ),
            "--sandbox-timeout-s",
            str(float(args.sandbox_timeout_s)),
            "--sandbox-cpu-limit",
            str(int(args.sandbox_cpu_limit)),
            "--sandbox-mem-limit-mb",
            str(int(args.sandbox_mem_limit_mb)),
            "--sandbox-wall-limit-s",
            str(float(args.sandbox_wall_limit_s)),
            "--model-id",
            str(args.model_id),
            "--model-revision",
            str(args.model_revision),
            "--tokenizer-id",
            str(args.tokenizer_id),
            "--tokenizer-revision",
            str(args.tokenizer_revision),
            "--decoding-method",
            str(args.decoding_method),
            "--temperature",
            str(float(args.temperature)),
            "--top-p",
            str(float(args.top_p)),
            "--top-k",
            str(int(args.top_k)),
            "--max-new-tokens",
            str(int(args.max_new_tokens)),
            "--seed",
            str(int(args.seed)),
            "--num-samples",
            str(int(args.num_samples)),
            "--metric-name",
            str(args.metric_name),
            "--k",
            str(int(args.k)),
        ]
    )
    return int(rc)


if __name__ == "__main__":
    raise SystemExit(main())
