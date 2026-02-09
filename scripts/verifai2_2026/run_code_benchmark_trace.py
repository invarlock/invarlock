#!/usr/bin/env python3
"""
run_code_benchmark_trace.py
===========================

Convenience wrapper to run an end-to-end code-verifier trace:
  tasks -> completions -> cases -> verifier_trace.v1

This exists to make the verifier-trace contract hard to "almost follow": the
wrapper pins decoding/sandbox parameters at invocation time and produces all
intermediate artifacts in one place.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from scripts.verifai2_2026 import generate_completions, run_verifier_trace_pipeline


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--verifier-name", type=str, required=True)

    # Model/tokenizer identity for loading and trace contract.
    p.add_argument("--model-id", type=str, required=True)
    p.add_argument("--model-revision", type=str, required=True)
    p.add_argument("--model", type=str, required=True, help="HF id or local dir.")
    p.add_argument("--model-load-revision", type=str, default="")
    p.add_argument("--tokenizer-id", type=str, required=True)
    p.add_argument("--tokenizer-revision", type=str, required=True)
    p.add_argument("--trust-remote-code", action="store_true")

    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--dtype", type=str, default="fp16")
    p.add_argument("--batch-size", type=int, default=8)

    # Decoding parameters (mirrors verifier-trace contract fields)
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
    p.add_argument("--num-beams", type=int, default=0)

    # Verifier identity + sandbox limits
    p.add_argument("--harness-name", type=str, required=True)
    p.add_argument("--harness-version", type=str, default="")
    p.add_argument("--harness-git-commit", type=str, default="")
    p.add_argument("--harness-container-image", type=str, default="")
    p.add_argument("--sandbox-timeout-s", type=float, default=10.0)
    p.add_argument("--sandbox-cpu-limit", type=int, default=2)
    p.add_argument("--sandbox-mem-limit-mb", type=int, default=2048)
    p.add_argument("--sandbox-wall-limit-s", type=float, default=10.0)

    p.add_argument("--metric-name", type=str, default="pass@1")
    p.add_argument("--k", type=int, default=0)

    p.add_argument(
        "--skip-generation",
        action="store_true",
        help="If set, reuse existing completions.jsonl in out-dir.",
    )
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args(argv)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    completions = out_dir / "completions.jsonl"
    prompt_set = out_dir / "prompt_set.json"
    cases = out_dir / "cases.jsonl"
    trace = out_dir / "verifier_trace.v1.json"

    if not bool(args.skip_generation):
        rc = int(
            generate_completions.main(
                [
                    "--tasks",
                    str(args.tasks),
                    "--out",
                    str(completions),
                    "--model",
                    str(args.model),
                    "--revision",
                    str(args.model_load_revision),
                    "--tokenizer",
                    str(args.tokenizer_id),
                    "--tokenizer-revision",
                    str(args.tokenizer_revision),
                    *(["--trust-remote-code"] if bool(args.trust_remote_code) else []),
                    "--device",
                    str(args.device),
                    "--dtype",
                    str(args.dtype),
                    "--batch-size",
                    str(int(args.batch_size)),
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
                    "--num-beams",
                    str(int(args.num_beams)),
                    *(["--limit", str(int(args.limit))] if int(args.limit) > 0 else []),
                ]
            )
        )
        if rc != 0:
            return rc
    elif not completions.exists():
        print(
            f"--skip-generation set but missing completions file: {completions}",
            file=sys.stderr,
        )
        return 2

    rc = int(
        run_verifier_trace_pipeline.main(
            [
                "--backend",
                "code_tests",
                "--tasks",
                str(args.tasks),
                "--completions",
                str(completions),
                "--prompt-set-out",
                str(prompt_set),
                "--cases-out",
                str(cases),
                "--trace-out",
                str(trace),
                "--verifier-name",
                str(args.verifier_name),
                "--verifier-kind",
                "code_execution",
                "--harness-name",
                str(args.harness_name),
                "--harness-version",
                str(args.harness_version),
                "--harness-git-commit",
                str(args.harness_git_commit),
                "--harness-container-image",
                str(args.harness_container_image),
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
                "--prompt-id-field",
                "id",
                "--prompt-text-field",
                "prompt",
                "--prompt-mode",
                "hash_only",
                "--dataset-name",
                str(args.verifier_name),
                "--dataset-split",
                "test",
                "--dataset-revision",
                "unknown",
                *(["--limit", str(int(args.limit))] if int(args.limit) > 0 else []),
            ]
        )
    )
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
