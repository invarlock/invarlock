#!/usr/bin/env python3
"""Benchmark reproducible InvarLock command hot paths without publishing results."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import platform
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, NamedTuple

import psutil

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

BENCHMARK_NAMES = (
    "schema-validation",
    "bootstrap",
    "report-assembly",
    "evidence-snapshot-verify",
)
SCHEMA = "invarlock.command_hotpaths_benchmark.v1"


class Workload(NamedTuple):
    input_payload: object
    run: Callable[[], object]
    close: Callable[[], None] = lambda: None


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


class _RssMonitor:
    """Sample this process and transient child processes during one benchmark."""

    def __init__(self) -> None:
        self._process = psutil.Process()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self.peak_bytes = 0

    def _sample(self) -> None:
        processes = [self._process]
        try:
            processes.extend(self._process.children(recursive=True))
        except (psutil.NoSuchProcess, psutil.AccessDenied, PermissionError):
            pass
        total = 0
        for process in processes:
            try:
                total += int(process.memory_info().rss)
            except (psutil.NoSuchProcess, psutil.AccessDenied, PermissionError):
                continue
        self.peak_bytes = max(self.peak_bytes, total)

    def _poll(self) -> None:
        while not self._stop.wait(0.001):
            self._sample()

    def start(self) -> None:
        self._sample()
        self._thread.start()

    def stop(self) -> None:
        self._sample()
        self._stop.set()
        self._thread.join()


def _commit_identity() -> dict[str, object]:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        sha = "unknown"
        dirty = False
    return {"sha": sha, "dirty": dirty}


def _platform_identity() -> dict[str, str]:
    return {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
    }


def _cuda_start(enabled: bool) -> tuple[object | None, dict[str, object] | None]:
    if not enabled:
        return None, None
    try:
        import torch
    except ImportError:
        return None, {"available": False, "reason": "torch unavailable"}
    if not torch.cuda.is_available():
        return torch, {"available": False, "reason": "CUDA unavailable"}
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    return torch, None


def _cuda_finish(
    torch_module: object | None, unavailable: dict[str, object] | None
) -> dict[str, object] | None:
    if unavailable is not None:
        return unavailable
    if torch_module is None:
        return None
    cuda = torch_module.cuda  # type: ignore[attr-defined]
    cuda.synchronize()
    index = int(cuda.current_device())
    return {
        "available": True,
        "device_index": index,
        "device_name": str(cuda.get_device_name(index)),
        "peak_allocated_bytes": int(cuda.max_memory_allocated(index)),
        "peak_reserved_bytes": int(cuda.max_memory_reserved(index)),
    }


def _schema_validation_workload() -> Workload:
    from invarlock.reporting.report_schema import validate_report

    report_path = REPO_ROOT / "tests/artifacts/golden_runs/gpt2/evaluation.report.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))

    def run() -> object:
        return {"valid": bool(validate_report(payload))}

    return Workload({"report": payload}, run)


def _bootstrap_workload() -> Workload:
    from invarlock.core.bootstrap import compute_paired_delta_log_ci
    from invarlock.reporting.verify_bootstrap_math import replay_paired_delta_log_ci

    count = 64
    final = [1.0 + 0.002 * index + 0.01 * math.sin(index) for index in range(count)]
    baseline = [1.0 + 0.002 * index for index in range(count)]
    weights = [float(32 + (index % 11)) for index in range(count)]
    options = {"method": "bca", "replicates": 400, "alpha": 0.05, "seed": 17}

    def run() -> object:
        producer = compute_paired_delta_log_ci(final, baseline, weights, **options)
        verifier = replay_paired_delta_log_ci(final, baseline, weights, **options)
        return {"producer": producer, "verifier": verifier}

    return Workload(
        {"final": final, "baseline": baseline, "weights": weights, **options}, run
    )


def _merge(target: dict[str, Any], source: Mapping[str, Any]) -> None:
    for key, value in source.items():
        current = target.get(key)
        if isinstance(current, dict) and isinstance(value, Mapping):
            _merge(current, value)
        else:
            target[key] = copy.deepcopy(value)


def _canonical_run(source: Mapping[str, Any]) -> dict[str, Any]:
    from invarlock.core.auto_tuning import resolve_tier_policies
    from invarlock.reporting.report_types import create_empty_report
    from invarlock.reporting.runtime_policy_receipt import build_runtime_policy_receipt

    report = create_empty_report()
    _merge(report, source)
    tier = str(report["meta"]["auto"]["tier"])
    profile = str(report["context"]["profile"])
    edit_name = str(report["edit"]["name"])
    policies = resolve_tier_policies(tier, edit_name, profile=profile)
    resolved, receipt = build_runtime_policy_receipt(
        policies,
        report["guards"],
        tier=tier,
        profile=profile,
        edit_name=edit_name,
    )
    report["resolved_policy"] = resolved
    report["policy_resolution"] = receipt
    return report


def _report_assembly_inputs() -> tuple[dict[str, Any], dict[str, Any]]:
    preview_logloss = [1.0, 1.1]
    final_logloss = [1.04, 1.14]
    weights = [100, 200]
    preview_ppl = math.exp(
        sum(a * b for a, b in zip(preview_logloss, weights, strict=True)) / 300
    )
    final_ppl = math.exp(
        sum(a * b for a, b in zip(final_logloss, weights, strict=True)) / 300
    )
    common = {
        "meta": {
            "model_id": "benchmark/model",
            "adapter": "hf_causal",
            "commit": "benchmark",
            "device": "cpu",
            "seed": 17,
            "ts": "2026-01-01T00:00:00+00:00",
            "auto": {"tier": "balanced"},
        },
        "context": {"profile": "dev"},
        "data": {
            "dataset": "benchmark",
            "split": "validation",
            "seq_len": 8,
            "stride": 4,
            "preview_n": 2,
            "final_n": 2,
        },
        "guards": [],
        "artifacts": {"events_path": "", "logs_path": ""},
    }
    subject = _canonical_run(
        {
            **common,
            "edit": {"name": "structured"},
            "metrics": {
                "primary_metric": {
                    "kind": "ppl_causal",
                    "preview": preview_ppl,
                    "final": final_ppl,
                    "ratio_vs_baseline": final_ppl / preview_ppl,
                },
                "bootstrap": {
                    "replicates": 150,
                    "alpha": 0.05,
                    "method": "percentile",
                },
            },
            "evaluation_windows": {
                "preview": {
                    "window_ids": [1, 2],
                    "logloss": preview_logloss,
                    "token_counts": weights,
                },
                "final": {
                    "window_ids": [3, 4],
                    "logloss": final_logloss,
                    "token_counts": weights,
                },
            },
        }
    )
    baseline = _canonical_run(
        {
            **common,
            "run_id": "benchmark-baseline",
            "edit": {"name": "noop"},
            "metrics": {
                "primary_metric": {
                    "kind": "ppl_causal",
                    "preview": preview_ppl,
                    "final": preview_ppl,
                },
                "bootstrap": {
                    "replicates": 150,
                    "alpha": 0.05,
                    "method": "percentile",
                },
            },
            "evaluation_windows": {
                "preview": {
                    "window_ids": [1, 2],
                    "logloss": preview_logloss,
                    "token_counts": weights,
                },
                "final": {
                    "window_ids": [3, 4],
                    "logloss": preview_logloss,
                    "token_counts": weights,
                },
            },
        }
    )
    return subject, baseline


def _report_assembly_workload() -> Workload:
    from invarlock.reporting.report_make import make_report

    subject, baseline = _report_assembly_inputs()

    def run() -> object:
        assembled = make_report(
            copy.deepcopy(subject),
            copy.deepcopy(baseline),
            provenance_env_flags={"benchmark": "command-hotpath-v1"},
        )
        # Assembly timestamps are deliberately generated at runtime. Normalize
        # only those documented volatile fields so the digest still detects any
        # behavioral output change across repetitions and comparison runs.
        assembled["artifacts"]["generated_at"] = "<runtime>"
        assembled["policy_provenance"]["resolved_at"] = "<runtime>"
        assembled["provenance"]["policy"]["resolved_at"] = "<runtime>"
        return assembled

    return Workload({"subject": subject, "baseline": baseline}, run)


def _evidence_snapshot_workload() -> Workload:
    from invarlock.evidence_pack_snapshot import PackSnapshot

    temporary = tempfile.TemporaryDirectory(prefix="invarlock-hotpath-")
    root = Path(temporary.name)
    (root / "checksums.sha256").write_text("benchmark\n", encoding="utf-8")
    (root / "evaluation.report.json").write_text(
        json.dumps({"schema_version": "benchmark"}), encoding="utf-8"
    )
    nested = root / "materials"
    nested.mkdir()
    (nested / "receipt.json").write_text(
        json.dumps({"status": "complete"}), encoding="utf-8"
    )
    large_material = nested / "large-evidence.bin"
    with large_material.open("wb") as handle:
        handle.truncate(16 * 1024 * 1024)

    def run() -> object:
        snapshot, errors = PackSnapshot.capture(root)
        if errors or snapshot is None:
            raise RuntimeError(f"snapshot capture failed: {errors}")
        verification_errors = snapshot.stability_errors()
        with snapshot.files.materialized() as materialized:
            verification_errors.extend(
                snapshot.files.materialized_stability_errors(materialized)
            )
        return {
            "inventory": sorted(snapshot.files.inventory),
            "digests": [entry.sha256 for entry in snapshot.files.entries],
            "verification_errors": verification_errors,
        }

    inputs: dict[str, object] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        with path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        inputs[path.relative_to(root).as_posix()] = {
            "sha256": digest,
            "size_bytes": path.stat().st_size,
        }
    return Workload(inputs, run, temporary.cleanup)


WORKLOAD_FACTORIES: dict[str, Callable[[], Workload]] = {
    "schema-validation": _schema_validation_workload,
    "bootstrap": _bootstrap_workload,
    "report-assembly": _report_assembly_workload,
    "evidence-snapshot-verify": _evidence_snapshot_workload,
}


def _measure(name: str, *, repeat: int, cuda: bool) -> dict[str, object]:
    workload = WORKLOAD_FACTORIES[name]()
    timings: list[float] = []
    output_digests: set[str] = set()
    torch_module, cuda_unavailable = _cuda_start(cuda)
    rss = _RssMonitor()
    rss.start()
    try:
        for _ in range(repeat):
            started = time.perf_counter()
            output = workload.run()
            timings.append(time.perf_counter() - started)
            output_digests.add(_digest(output))
    finally:
        rss.stop()
        workload.close()
    if len(output_digests) != 1:
        raise RuntimeError(f"{name} output changed between benchmark repetitions")
    ordered = sorted(timings)
    p95_index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "repeat": repeat,
        "median_walltime_seconds": statistics.median(timings),
        "p95_walltime_seconds": ordered[p95_index],
        "peak_rss_bytes": rss.peak_bytes,
        "input_sha256": _digest(workload.input_payload),
        "output_sha256": next(iter(output_digests)),
        "cuda": _cuda_finish(torch_module, cuda_unavailable),
    }


def compare_benchmark_results(
    current: Mapping[str, Any],
    baseline: Mapping[str, Any],
    *,
    targets: set[str],
    minimum_target_improvement: float,
    maximum_untargeted_regression: float,
) -> dict[str, object]:
    current_items = current.get("benchmarks")
    baseline_items = baseline.get("benchmarks")
    if not isinstance(current_items, Mapping) or not isinstance(
        baseline_items, Mapping
    ):
        raise ValueError("current and baseline payloads must contain benchmarks")
    comparisons: dict[str, object] = {}
    for name in sorted(set(current_items) | set(baseline_items)):
        candidate = current_items.get(name)
        reference = baseline_items.get(name)
        if not isinstance(candidate, Mapping) or not isinstance(reference, Mapping):
            comparisons[name] = {"passed": False, "reason": "benchmark missing"}
            continue
        if candidate.get("input_sha256") != reference.get("input_sha256"):
            comparisons[name] = {"passed": False, "reason": "input digest mismatch"}
            continue
        if candidate.get("output_sha256") != reference.get("output_sha256"):
            comparisons[name] = {"passed": False, "reason": "output digest mismatch"}
            continue
        baseline_seconds = float(reference["median_walltime_seconds"])
        current_seconds = float(candidate["median_walltime_seconds"])
        if baseline_seconds <= 0.0 or current_seconds <= 0.0:
            comparisons[name] = {
                "passed": False,
                "reason": "walltime must be positive",
            }
            continue
        improvement = (baseline_seconds - current_seconds) / baseline_seconds
        regression = (current_seconds - baseline_seconds) / baseline_seconds
        if name in targets:
            passed = improvement >= minimum_target_improvement
            reason = "accepted" if passed else "target improvement below threshold"
        else:
            passed = regression <= maximum_untargeted_regression
            reason = "accepted" if passed else "untargeted regression above threshold"
        comparisons[name] = {
            "passed": passed,
            "reason": reason,
            "baseline_median_walltime_seconds": baseline_seconds,
            "current_median_walltime_seconds": current_seconds,
            "improvement_fraction": improvement,
            "regression_fraction": regression,
        }
    unknown_targets = targets - set(comparisons)
    for name in sorted(unknown_targets):
        comparisons[name] = {"passed": False, "reason": "target benchmark missing"}
    return {
        "passed": bool(comparisons)
        and all(bool(item["passed"]) for item in comparisons.values()),
        "minimum_target_improvement": minimum_target_improvement,
        "maximum_untargeted_regression": maximum_untargeted_regression,
        "targets": sorted(targets),
        "benchmarks": comparisons,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark",
        action="append",
        choices=BENCHMARK_NAMES,
        help="benchmark to run; repeat to select several (default: all)",
    )
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--cuda", action="store_true", help="capture CUDA peak memory")
    parser.add_argument("--baseline-json", type=Path)
    parser.add_argument("--target", action="append", choices=BENCHMARK_NAMES)
    parser.add_argument("--min-target-improvement", type=float, default=0.10)
    parser.add_argument("--max-untargeted-regression", type=float, default=0.0)
    parser.add_argument("--json", action="store_true")
    return parser


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be at least 1")
    if not 0.0 <= args.min_target_improvement < 1.0:
        parser.error("--min-target-improvement must be in [0, 1)")
    if args.max_untargeted_regression < 0.0:
        parser.error("--max-untargeted-regression must be non-negative")
    if bool(args.baseline_json) != bool(args.target):
        parser.error("--baseline-json and at least one --target must be used together")

    selected = tuple(dict.fromkeys(args.benchmark or BENCHMARK_NAMES))
    payload: dict[str, object] = {
        "schema": SCHEMA,
        "publication": "local-only",
        "results_persisted": False,
        "commit": _commit_identity(),
        "platform": _platform_identity(),
        "repeat": args.repeat,
        "benchmarks": {
            name: _measure(name, repeat=args.repeat, cuda=args.cuda)
            for name in selected
        },
    }
    exit_code = 0
    if args.baseline_json:
        baseline = json.loads(args.baseline_json.read_text(encoding="utf-8"))
        acceptance = compare_benchmark_results(
            payload,
            baseline,
            targets=set(args.target),
            minimum_target_improvement=args.min_target_improvement,
            maximum_untargeted_regression=args.max_untargeted_regression,
        )
        payload["acceptance"] = acceptance
        exit_code = 0 if acceptance["passed"] else 1

    if args.json:
        print(json.dumps(payload, allow_nan=False, sort_keys=True))
    else:
        for name, result in payload["benchmarks"].items():
            print(
                f"{name}: median={result['median_walltime_seconds']:.6f}s "
                f"p95={result['p95_walltime_seconds']:.6f}s"
            )
        if "acceptance" in payload:
            print(
                f"acceptance: {'pass' if payload['acceptance']['passed'] else 'fail'}"
            )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
