#!/usr/bin/env python3
"""Validate and render the separate five-candidate Linux measurement-v2 comparison."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import statistics
from pathlib import Path, PurePosixPath

CANDIDATES = (
    "reference",
    "rust-standalone-before",
    "rust-ffi-before",
    "rust-ffi",
    "rust-standalone",
)
LABELS = (
    "Python",
    "Before Rust standalone",
    "Before Rust FFI",
    "After Rust FFI",
    "After Rust standalone",
)
WORKLOADS = (
    "captured-nll-1000",
    "captured-nll-5000",
    "captured-nll-12000",
    "captured-nll-50000",
    "captured-regression",
    "native-pack",
    "judge-pack",
    "evidence-set",
)
WORKLOAD_LABELS = (
    "NLL 1,000",
    "NLL 5,000",
    "NLL 12,000",
    "NLL 50,000",
    "Captured\nregression",
    "Native\npack",
    "Judge\npack",
    "Evidence\nset",
)
BEFORE_COMMIT = "673d492dde9d7b320c6f082a5ca2a8191ab37f66"
BASELINE_COMMIT = "ce04f20ed7ccefc1bc826098f1b0aa4102b63ef3"
METRICS = (
    "wall_seconds",
    "user_cpu_seconds",
    "system_cpu_seconds",
    "max_rss_bytes",
    "operation_wall_seconds",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()


def read(path):
    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate JSON object key")
            result[key] = value
        return result

    value = json.loads(Path(path).read_bytes(), object_pairs_hook=unique_pairs)
    canonical(value)
    return value


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def hash_value(value):
    return value.removeprefix("sha256:")


def materialize(value, output):
    if isinstance(value, str):
        return value.replace("$OUTPUT", output)
    if isinstance(value, list):
        return [materialize(item, output) for item in value]
    if isinstance(value, dict):
        return {key: materialize(item, output) for key, item in value.items()}
    return value


def comparison(row):
    result = copy.deepcopy(row["payload"]["result"])
    require(
        isinstance(result["outputs"], list) and isinstance(result["files"], dict),
        "malformed transaction result",
    )
    output = PurePosixPath(row["output_directory"])
    for item in result["outputs"]:
        require(isinstance(item.get("payload"), dict), "malformed operation payload")
        report = item["payload"].get("report")
        if isinstance(report, dict):
            for field in ("requested_outputs", "written_outputs"):
                for name, path in report.get(field, {}).items():
                    selected = PurePosixPath(path)
                    require(
                        selected.parent == output,
                        "report destination escaped the recorded output directory",
                    )
                    report[field][name] = "$OUTPUT/" + selected.name
    return canonical(result)


def order(round_index, workload_index):
    base = list(CANDIDATES)
    random.Random(4127 + workload_index).shuffle(base)
    shift = round_index % len(CANDIDATES)
    return base[shift:] + base[:shift]


def median_comparison(candidate_median, python_median):
    """Describe observed medians without imposing a significance threshold."""
    require(
        all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            and value >= 0
            for value in (candidate_median, python_median)
        ),
        "invalid median for comparison",
    )
    status = (
        "win"
        if candidate_median < python_median
        else "loss"
        if candidate_median > python_median
        else "tie"
    )
    ratio = candidate_median / python_median if python_median else None
    reason = "zero_python_median" if python_median == 0 else None
    if ratio is not None and not math.isfinite(ratio):
        ratio, reason = None, "ratio_not_finite"
    return {
        "candidate_median": candidate_median,
        "python_median": python_median,
        "candidate_over_python_median_ratio": ratio,
        "ratio_unavailable_reason": reason,
        "observed_median_status": status,
        "observed_median_win": status == "win",
    }


def median_change(after, before):
    """Compare unrounded descriptive medians; zero denominators stay unavailable."""
    result = median_comparison(after, before)
    return {
        "after_median": after,
        "before_median": before,
        "after_over_before_ratio": result["candidate_over_python_median_ratio"],
        "ratio_unavailable_reason": "zero_before_median"
        if before == 0
        else result["ratio_unavailable_reason"],
        "observed_median_status": result["observed_median_status"],
    }


def derived_metrics(samples, retained_summary):
    """Sum CPU within each attempt before computing medians and ranges."""
    cells = {(r["workload"], r["candidate"]): r for r in retained_summary}
    require(len(cells) == len(retained_summary), "duplicate retained summary cell")
    groups, totals, seen = {}, [], set()
    for sample in samples:
        if sample["phase"] != "measurement":
            continue
        key = (sample["workload"], sample["candidate"])
        require(key in cells, "sample has no summary cell")
        identity = (*key, sample["round"])
        require(identity not in seen, "duplicate timed sample round")
        seen.add(identity)
        components = [
            sample["metrics"][k] for k in ("user_cpu_seconds", "system_cpu_seconds")
        ]
        require(
            all(
                type(v) in (int, float) and math.isfinite(v) and v >= 0
                for v in components
            ),
            "invalid per-attempt CPU components",
        )
        total = sum(components)
        require(math.isfinite(total), "non-finite per-attempt CPU total")
        item = {
            "attempt": sample["attempt"],
            "workload": key[0],
            "candidate": key[1],
            "round": sample["round"],
            "total_cpu_seconds": total,
        }
        totals.append(item)
        groups.setdefault(key, []).append(item)
    require(set(groups) == set(cells), "missing timed summary cell")
    cpu_summary, cpu_cells = [], {}
    for key, rows in sorted(groups.items()):
        require(
            len(rows) == 5 and {r["round"] for r in rows} == set(range(5)),
            "five distinct timed rounds required per cell",
        )
        values = [r["total_cpu_seconds"] for r in rows]
        stats = {
            "median": statistics.median(values),
            "minimum": min(values),
            "maximum": max(values),
        }
        cpu_cells[key] = stats
        cpu_summary.append(
            {
                "workload": key[0],
                "candidate": key[1],
                "n": 5,
                "total_cpu_seconds": stats,
            }
        )
    versus_python, before_after = [], []

    def stats(key, metric):
        return cpu_cells[key] if metric == "total_cpu_seconds" else cells[key][metric]

    for workload in sorted({key[0] for key in cells}):
        require(
            {c for w, c in cells if w == workload} == set(CANDIDATES),
            "five candidate cells required",
        )
        for candidate in CANDIDATES[1:]:
            metrics = {
                metric: median_comparison(
                    stats((workload, candidate), metric)["median"],
                    stats((workload, "reference"), metric)["median"],
                )
                for metric in ("wall_seconds", "total_cpu_seconds", "max_rss_bytes")
            }
            versus_python.append(
                {
                    "workload": workload,
                    "candidate": candidate,
                    "reference": "reference",
                    "metrics": metrics,
                    "all_three_observed_medians_lower": all(
                        v["observed_median_win"] for v in metrics.values()
                    ),
                }
            )
        for mode, before, after in (
            ("standalone", "rust-standalone-before", "rust-standalone"),
            ("ffi", "rust-ffi-before", "rust-ffi"),
        ):
            before_after.append(
                {
                    "workload": workload,
                    "mode": mode,
                    "before": before,
                    "after": after,
                    "metrics": {
                        metric: median_change(
                            stats((workload, after), metric)["median"],
                            stats((workload, before), metric)["median"],
                        )
                        for metric in (
                            "wall_seconds",
                            "total_cpu_seconds",
                            "max_rss_bytes",
                        )
                    },
                }
            )
    return {
        "total_cpu_definition": "Sum user and system CPU for each measured attempt, then compute median/minimum/maximum; never sum component medians.",
        "comparison_definition": "Unrounded observed medians only; lower is better. Ratios with zero denominators are unavailable. No statistical-significance or confidence claim.",
        "qualification_samples_included": False,
        "total_cpu_samples": totals,
        "total_cpu_summary": cpu_summary,
        "versus_python": versus_python,
        "before_after": before_after,
    }


def expected_operations(workload):
    if workload.startswith("captured-"):
        return ["verify-captured", "verify-captured-receipt", "report"]
    return {
        "native-pack": ["verify-native", "verify-native-receipt", "report"],
        "judge-pack": [
            "verify-judge",
            "write-judge-receipt",
            "verify-judge-receipt",
            "report",
        ],
        "evidence-set": ["verify-evidence-set", "verify-stored-evidence-set", "report"],
    }[workload]


def domain_statuses(workload, outputs):
    expected = (
        [7, 0, 0]
        if workload == "captured-regression"
        else [0] * len(expected_operations(workload))
    )
    actual = [o["status"] for o in outputs]
    require(
        all(type(v) is int for v in actual) and actual == expected,
        "domain statuses differ from the independently audited fixture contract",
    )
    return actual


def validate_process_metrics(row, raw, supervisor):
    fields = {
        "format",
        "exit_code",
        "wait_status",
        "timed_out",
        "exec_errno",
        "wall_seconds",
        "user_cpu_seconds",
        "system_cpu_seconds",
        "max_rss_raw",
        "max_rss_raw_unit",
        "max_rss_bytes",
    }
    require(set(raw) == fields, "unexpected process metrics fields")
    require(
        raw["format"] == "invarlock/linux-process-measurement-v2",
        "unexpected process measurement format",
    )
    require(
        all(row.get(k) == v and type(row.get(k)) is type(v) for k, v in raw.items()),
        "sample differs from raw process metrics",
    )
    require(row.get("attempt_dispatched") is True, "sample dispatch was not recorded")
    require(
        type(raw["exit_code"]) is int
        and raw["exit_code"] == 0
        and raw["timed_out"] is False,
        "worker exit or timeout failed",
    )
    require(
        row.get("measurement_supervisor") == supervisor,
        "sample supervisor identity differs",
    )
    require(
        type(raw["exec_errno"]) is int
        and raw["exec_errno"] == 0
        and type(raw["wait_status"]) is int
        and raw["wait_status"] == 0,
        "worker launch or wait status failed",
    )
    require(
        type(raw["max_rss_raw"]) is int and type(raw["max_rss_bytes"]) is int,
        "RSS must be exact integers",
    )


def validate_sources(metadata, expected_python, expected_before, expected_after):
    for label, expected, actual in (
        ("Python baseline", expected_python, metadata["source_commit"]),
        ("before", expected_before, metadata["before_source"]["commit"]),
        ("final candidate", expected_after, metadata["after_source"]["commit"]),
    ):
        require(
            len(expected) == 40 and all(c in "0123456789abcdef" for c in expected),
            "expected source revisions must be full lowercase Git commit IDs",
        )
        require(actual == expected, f"unexpected {label} source commit")


def validate(
    run,
    metadata_path,
    environment_path,
    *,
    expected_python=BASELINE_COMMIT,
    expected_before=BEFORE_COMMIT,
    expected_after,
):
    metadata = read(metadata_path)
    environment = read(environment_path)
    manifest = read(run / "manifest.json")
    summary = read(run / "summary.json")
    require(
        summary.get("complete") is True and summary.get("comparison_complete") is True,
        "run is incomplete; retain failures and do not publish a completed comparison",
    )
    require(
        summary.get("stop_reason") is None and summary.get("validation_error") is None,
        "run retained a stop or validation failure",
    )
    for key, value in (
        ("qualification_planned", 40),
        ("qualification_attempted", 40),
        ("qualification_valid", 40),
        ("timing_planned", 200),
        ("timing_attempted", 200),
        ("timing_valid", 200),
        ("retries", 0),
        ("exclusions", 0),
    ):
        require(summary.get(key) == value, f"unexpected {key}")
    require(
        manifest.get("format") == "invarlock/five-candidate-comparison-v2"
        and manifest.get("qualification_only") is False
        and summary.get("qualification_complete") is True
        and summary.get("qualification_only") is False
        and metadata == manifest["metadata"]
        and sha(metadata_path) == manifest["metadata_sha256"],
        "run metadata identity differs",
    )
    validate_sources(metadata, expected_python, expected_before, expected_after)
    require(
        environment["machine"] == "aarch64"
        and environment["platform"].startswith("Linux"),
        "expected Linux ARM64 environment",
    )
    require(
        environment["work_filesystem"] in ("ext2/ext3", "ext2", "ext3", "ext4"),
        "expected retained Linux ext storage",
    )
    original_root = PurePosixPath(metadata["environment"]["path"]).parent
    local_root = metadata_path.resolve().parent

    def relocate(path):
        relative = PurePosixPath(path).relative_to(original_root)
        selected = local_root.joinpath(*relative.parts)
        require(
            selected.resolve().is_relative_to(local_root),
            "relocated artifact escaped the export directory",
        )
        return selected

    artifacts = {}
    for key, item in metadata.items():
        if isinstance(item, dict) and set(item) >= {"path", "sha256"}:
            selected = relocate(item["path"])
            require(
                sha(selected) == hash_value(item["sha256"]),
                f"artifact identity differs: {key}",
            )
            artifacts[key] = {"filename": selected.name, "sha256": sha(selected)}
    require(
        sha(environment_path) == hash_value(metadata["environment"]["sha256"]),
        "supplied environment differs from the frozen environment",
    )
    for path, expected in metadata["harness_sources"].items():
        require(
            sha(relocate(path)) == hash_value(expected),
            "retained harness source changed",
        )
    bundle_path = relocate(metadata["input_bundle"]["path"])
    bundle = read(bundle_path)
    require(
        bundle["before"] == metadata["before_source"]
        and bundle["after"] == metadata["after_source"],
        "source archive provenance differs",
    )
    for relative, expected in bundle["files"].items():
        selected = bundle_path.parent / relative
        require(
            selected.resolve().is_relative_to(bundle_path.parent.resolve()),
            "source bundle path escapes its root",
        )
        require(
            sha(selected) == hash_value(expected),
            f"source bundle identity differs: {relative}",
        )
    matrix_path = relocate(metadata["matrix"]["path"])
    matrix = read(matrix_path)
    require(
        matrix == manifest["matrix"] and sha(matrix_path) == manifest["matrix_sha256"],
        "matrix identity differs",
    )
    require(
        matrix
        == {
            "candidates": list(CANDIDATES),
            "rounds": 5,
            "seed": 4127,
            "timeout_seconds": 300,
            "total_budget_seconds": 7200,
            "workloads": list(WORKLOADS),
        },
        "unexpected comparison protocol",
    )
    fixtures_path = relocate(metadata["fixtures_manifest"]["path"])
    require(
        sha(fixtures_path) == manifest["fixtures_sha256"],
        "fixture manifest identity differs",
    )
    fixtures = read(fixtures_path)
    require(
        [item["name"] for item in fixtures["workloads"]] == list(WORKLOADS),
        "unexpected workload set",
    )
    for fixture in fixtures["workloads"]:
        for path, expected in fixture["files"].items():
            require(
                sha(relocate(path)) == hash_value(expected),
                "fixture input bytes changed",
            )
    before_metadata = read(relocate(metadata["before_runtime_metadata"]["path"]))
    require(
        before_metadata["root_wheel"] == metadata["root_wheel"]
        and before_metadata["source_commit"] == expected_python,
        "before Python baseline differs",
    )
    require(
        before_metadata["native_wheel"]["sha256"] != metadata["native_wheel"]["sha256"],
        "before and after native wheels are not independently bound",
    )
    require(
        before_metadata["standalone"] == metadata["standalone_before"],
        "before executable binding differs",
    )
    require(
        sha(relocate(before_metadata["native_wheel"]["path"]))
        == hash_value(before_metadata["native_wheel"]["sha256"]),
        "before native wheel bytes differ",
    )
    artifacts["native_wheel_before"] = {
        "filename": PurePosixPath(before_metadata["native_wheel"]["path"]).name,
        "sha256": hash_value(before_metadata["native_wheel"]["sha256"]),
    }
    require(
        set(metadata["runtimes"]) == {"reference", "rust-ffi-before", "rust-ffi"},
        "runtime candidate set differs",
    )
    samples, groups, names = [], {}, set()
    identities = {}
    for phase in ("qualification", "measurement"):
        for workload_index, fixture in enumerate(fixtures["workloads"]):
            require(
                [s["operation"] for s in fixture["request"]["steps"]]
                == expected_operations(fixture["name"]),
                "fixture operation contract differs",
            )
            for round_index in range(1 if phase == "qualification" else 5):
                group = []
                for position, candidate in enumerate(
                    order(round_index, workload_index)
                ):
                    name = f"{phase}-{fixture['name']}-{round_index}-{position}-{candidate}"
                    names.update((name, name + "-inputs"))
                    directory = run / name
                    row = read(directory / "sample.json")
                    require(
                        (
                            row["phase"],
                            row["workload"],
                            row["round"],
                            row["position"],
                            row["candidate"],
                        )
                        == (phase, fixture["name"], round_index, position, candidate),
                        "sample scheduling identity differs",
                    )
                    require(
                        row["status"] == "valid"
                        and row["exit_code"] == 0
                        and row["timed_out"] is False
                        and row["output_comparison"]
                        == "exact_except_declared_report_destinations",
                        "sample was not an exact valid outcome",
                    )
                    require(
                        read(directory / "stdout.txt") == row["payload"],
                        "retained stdout differs from the sample payload",
                    )
                    validate_process_metrics(
                        row,
                        read(directory / "process-metrics.json"),
                        metadata["measurement_supervisor"],
                    )
                    inputs = run / (name + "-inputs")
                    require(
                        relocate(row["output_directory"]).resolve()
                        == (inputs / "outputs").resolve(),
                        "recorded output directory differs from retained location",
                    )
                    request_path = inputs / "request.json"
                    request = read(request_path)
                    require(
                        sha(request_path) == row["request_sha256"],
                        "retained request identity differs",
                    )
                    require(
                        request
                        == materialize(fixture["request"], row["output_directory"]),
                        "request differs from the frozen fixture",
                    )
                    output_files = {}
                    for label, path in request["files"].items():
                        selected = PurePosixPath(path)
                        require(
                            selected.parent == PurePosixPath(row["output_directory"]),
                            "request output path escapes its directory",
                        )
                        output_files[label] = "sha256:" + sha(
                            inputs / "outputs" / selected.name
                        )
                    require(
                        output_files
                        == row["payload"]["result"]["files"]
                        == row["observed_output_files"],
                        "retained output file hashes differ",
                    )
                    require(
                        len(row["payload"]["result"]["outputs"])
                        == len(request["steps"]),
                        "operation output count differs",
                    )
                    require(
                        row["payload"].get("ok") is True,
                        "worker did not return a successful transaction",
                    )
                    require(
                        type(row["payload"]["native_transactions"]) is int
                        and row["payload"]["native_transactions"]
                        == (0 if candidate == "reference" else len(request["steps"])),
                        "native operation count differs",
                    )
                    statuses = domain_statuses(
                        fixture["name"], row["payload"]["result"]["outputs"]
                    )
                    if candidate in ("reference", "rust-ffi-before", "rust-ffi"):
                        runtime = metadata["runtimes"][candidate]
                        runtime_metadata = (
                            before_metadata
                            if candidate == "rust-ffi-before"
                            else metadata
                        )
                        expected_meta_path = (
                            metadata["before_runtime_metadata"]["path"]
                            if candidate == "rust-ffi-before"
                            else str(original_root / metadata_path.name)
                        )
                        require(
                            runtime["metadata"] == expected_meta_path,
                            "runtime metadata path differs",
                        )
                        expected_command = [
                            runtime["python"],
                            "-I",
                            "-c",
                            "import sys;sys.path.insert(0,sys.argv.pop(1));from invarlock_rust_experiment.bounded_worker import main;raise SystemExit(main())",
                            str(original_root / "inputs/harness"),
                            "--candidate",
                            "rust-ffi" if candidate == "rust-ffi-before" else candidate,
                            "--request",
                            str(
                                PurePosixPath(row["output_directory"]).parent
                                / "request.json"
                            ),
                            "--metadata",
                            runtime["metadata"],
                        ]
                        require(
                            row["command"] == expected_command,
                            "worker command differs from bound invocation",
                        )
                        current = {
                            key: value
                            for key, value in row["payload"]["identity"].items()
                            if key != "check_wall_seconds"
                        }
                        require(
                            current["executable"]
                            == row["command"][0]
                            == runtime["python"],
                            "interpreter identity differs",
                        )
                        for key in ("root_wheel", "native_wheel"):
                            require(
                                hash_value(current["wheel_bindings"][key]["sha256"])
                                == hash_value(runtime_metadata[key]["sha256"])
                                and type(
                                    current["wheel_bindings"][key][
                                        "installed_files_checked"
                                    ]
                                )
                                is int
                                and current["wheel_bindings"][key][
                                    "installed_files_checked"
                                ]
                                > 0,
                                "installed wheel identity differs",
                            )
                        require(
                            candidate not in identities
                            or current == identities[candidate],
                            "installed runtime identity changed",
                        )
                        identities[candidate] = current
                    else:
                        key = (
                            "standalone_before"
                            if candidate.endswith("-before")
                            else "standalone"
                        )
                        require(
                            row["command"]
                            == [
                                metadata[key]["path"],
                                str(
                                    PurePosixPath(row["output_directory"]).parent
                                    / "request.json"
                                ),
                            ]
                            and row["executable_artifact"] == metadata[key],
                            "standalone executable identity differs",
                        )
                    metrics = {
                        key: row["payload"][key]
                        if key == "operation_wall_seconds"
                        else row[key]
                        for key in METRICS
                    }
                    require(
                        all(
                            isinstance(value, (int, float))
                            and not isinstance(value, bool)
                            and math.isfinite(value)
                            and value >= 0
                            for value in metrics.values()
                        ),
                        "invalid numeric measurement",
                    )
                    require(
                        metrics["wall_seconds"] > 0 and metrics["max_rss_bytes"] > 0,
                        "nonpositive wall time or RSS",
                    )
                    require(
                        row["platform"] == "linux"
                        and row["max_rss_raw_unit"] == "KiB"
                        and row["max_rss_bytes"] == row["max_rss_raw"] * 1024,
                        "Linux RSS unit conversion differs",
                    )
                    samples.append(
                        {
                            "attempt": name,
                            "phase": phase,
                            "workload": fixture["name"],
                            "round": round_index,
                            "position": position,
                            "candidate": candidate,
                            "metrics": metrics,
                            "sha256": {
                                "sample.json": sha(directory / "sample.json"),
                                "stdout.txt": sha(directory / "stdout.txt"),
                                "stderr.txt": sha(directory / "stderr.txt"),
                                "request.json": sha(request_path),
                                "process-metrics.json": sha(
                                    directory / "process-metrics.json"
                                ),
                            },
                            "output_file_sha256": output_files,
                            "domain_statuses": statuses,
                            "exact_output_qualified": True,
                        }
                    )
                    group.append(row)
                    if phase == "measurement":
                        groups.setdefault((fixture["name"], candidate), []).append(
                            metrics
                        )
                expected = comparison(
                    next(row for row in group if row["candidate"] == "reference")
                )
                require(
                    all(comparison(row) == expected for row in group),
                    "exact outputs differ within a balanced round",
                )
    require(
        {path.name for path in run.iterdir() if path.is_file()}
        == {"manifest.json", "summary.json"},
        "unexpected retained run files or failure record",
    )
    observed = {path.name for path in run.iterdir() if path.is_dir()}
    require(observed == names, "unexpected, missing or orphaned attempt directories")
    require(len(samples) == 240 and len(groups) == 40, "sample denominator differs")
    computed = []
    for (workload, candidate), values in sorted(groups.items()):
        require(len(values) == 5, "cell does not have five observations")
        item = {
            "workload": workload,
            "candidate": candidate,
            "attempted": 5,
            "valid": 5,
            "failures": [],
        }
        for key in METRICS:
            numbers = [value[key] for value in values]
            item[key] = {
                "median": statistics.median(numbers),
                "minimum": min(numbers),
                "maximum": max(numbers),
            }
        computed.append(item)
    require(
        computed == summary["summary"],
        "retained summary differs from recomputed sample statistics",
    )
    require(
        {candidate: identities.get(candidate) for candidate in CANDIDATES}
        == summary["python_identities"],
        "retained runtime identity summary differs",
    )
    require(
        identities["reference"] == identities["rust-ffi"],
        "reference and after FFI environments differ",
    )
    derived = derived_metrics(samples, computed)
    for item, cpu in zip(computed, derived["total_cpu_summary"], strict=True):
        require(
            (item["workload"], item["candidate"])
            == (cpu["workload"], cpu["candidate"]),
            "CPU summary ordering differs",
        )
        item["total_cpu_seconds"] = cpu["total_cpu_seconds"]
    return {
        "format": "invarlock/linux-five-candidate-results-v2",
        "scope": "Linux ARM64 Docker VM; observed fixture journeys only. No production x86 or bare-metal performance claim.",
        "identity_check_scope": "Python and FFI process metrics include runtime identity and streamed verification of both installed wheels; Python also checks the unused native wheel. Standalone executable hashing happens before measurement. Cross-language process ratios therefore include asymmetric harness work. Before/after comparisons within each path share the boundary. Identity-check wall time and operation wall time are diagnostics, never subtracted from process CPU or peak RSS.",
        "comparison_complete": True,
        "qualification_attempted": 40,
        "qualification_valid": 40,
        "timing_attempted": 200,
        "timing_valid": 200,
        "retries": 0,
        "exclusions": 0,
        "candidate_labels": dict(zip(CANDIDATES, LABELS, strict=True)),
        "sources": {
            "python_baseline": expected_python,
            "rust_before": metadata["before_source"],
            "rust_after": metadata["after_source"],
            "source_archives": {
                name: {
                    "filename": name + ".tar",
                    "sha256": bundle["files"][name + ".tar"],
                }
                for name in ("before", "after")
            },
        },
        "artifacts": artifacts,
        "environment": {
            "platform": environment["platform"],
            "machine": environment["machine"],
            "work_filesystem": environment["work_filesystem"],
            "runtime_identities": identities,
            "supervisor_compiler": metadata["supervisor_compiler"],
            "environment_sha256": sha(environment_path),
            "boundary": "Linux guest ext filesystem; inputs and outputs staged before timing; VM and host storage remain part of the environment.",
        },
        "protocol": matrix,
        "measurement_notes": [
            "Fresh processes; filesystem warmed by qualification.",
            "Whiskers show observed minimum and maximum of five timings; these are descriptive ranges, not confidence intervals.",
            "Elapsed time is monotonic fork-to-wait4 for the actual worker; supervisor startup and retained-output validation are excluded.",
            "CPU and RSS are unadjusted direct-child wait4 measurements. RSS includes the small supervisor pre-exec image; it is not aggregate concurrent descendant RSS. Charts convert bytes to MiB.",
            "The 7200-second budget controls round admission and is not a hard elapsed-time cap.",
            "Historical macOS and optimized-Python measurements are not pooled.",
        ],
        "evidence_sha256": {
            "manifest.json": sha(run / "manifest.json"),
            "summary.json": sha(run / "summary.json"),
            "metadata.json": sha(metadata_path),
        },
        "summary": computed,
        "samples": samples,
        "derived": derived,
        "domain_contract": {
            w: (
                [7, 0, 0]
                if w == "captured-regression"
                else [0] * len(expected_operations(w))
            )
            for w in WORKLOADS
        },
        "exact_output_qualification": "All 40 qualification and 200 timed outcomes agree with Python, except declared report destinations; independent domain statuses also match.",
    }


def charts(report, output, *, include_total_cpu=True):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cells = {
        (item["workload"], item["candidate"]): dict(item) for item in report["summary"]
    }
    chart_specs = [
        ("wall_seconds", "wall-time", "Fresh-process wall time", "Seconds", 1),
        ("max_rss_bytes", "peak-rss", "Peak process memory", "MiB", 1024 * 1024),
    ]
    if include_total_cpu:
        derived = report.get("derived")
        if derived is None:
            derived = derived_metrics(report["samples"], report["summary"])
        for item in derived["total_cpu_summary"]:
            cells[item["workload"], item["candidate"]]["total_cpu_seconds"] = item[
                "total_cpu_seconds"
            ]
        chart_specs.append(
            (
                "total_cpu_seconds",
                "total-cpu",
                "Total process CPU time (user + system)",
                "CPU seconds",
                1,
            )
        )
    colors = ("#334155", "#D97706", "#9333EA", "#2563EB", "#0D9488")
    for metric, filename, title, unit, divisor in chart_specs:
        figure, axes = plt.subplots(1, 2, figsize=(13.5, 5.6))
        for panel, indexes in enumerate((range(4), range(4, 8))):
            axis = axes[panel]
            indexes = list(indexes)
            for candidate_index, (candidate, label, color) in enumerate(
                zip(CANDIDATES, LABELS, colors, strict=True)
            ):
                rows = [cells[WORKLOADS[index], candidate][metric] for index in indexes]
                medians = [row["median"] / divisor for row in rows]
                errors = [
                    [(row["median"] - row["minimum"]) / divisor for row in rows],
                    [(row["maximum"] - row["median"]) / divisor for row in rows],
                ]
                positions = [index + (candidate_index - 2) * 0.16 for index in range(4)]
                axis.bar(
                    positions,
                    medians,
                    width=0.15,
                    label=label,
                    color=color,
                    yerr=errors,
                    capsize=2,
                    error_kw={"elinewidth": 1},
                )
            logarithmic = (
                metric in ("wall_seconds", "total_cpu_seconds")
                and panel == 0
                and all(
                    cells[WORKLOADS[index], candidate][metric]["minimum"] > 0
                    for index in indexes
                    for candidate in CANDIDATES
                )
            )
            if logarithmic:
                axis.set_yscale("log")
            else:
                axis.set_ylim(bottom=0)
            axis.set_ylabel(
                unit + (" · log scale" if logarithmic else " · linear scale")
            )
            axis.set_title(
                "Captured NLL workloads" if panel == 0 else "Small fixture journeys"
            )
            axis.set_xticks(range(4), [WORKLOAD_LABELS[index] for index in indexes])
            axis.grid(axis="y", alpha=0.2)
            axis.set_axisbelow(True)
            axis.spines[["top", "right"]].set_visible(False)
        figure.suptitle(title + " — Linux ARM64 Docker VM", fontsize=15, y=0.98)
        handles, labels = axes[0].get_legend_handles_labels()
        figure.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.92),
            ncol=5,
            frameon=False,
        )
        figure.text(
            0.5,
            0.065,
            "Python/FFI include wheel identity checks; standalone executable hashing occurs before measurement.",
            ha="center",
            fontsize=9,
        )
        figure.text(
            0.5,
            0.025,
            "Median with observed min–max whiskers; n=5 per cell. Descriptive ranges, not confidence intervals.",
            ha="center",
            fontsize=10,
        )
        figure.subplots_adjust(
            top=0.77, bottom=0.18, wspace=0.25, left=0.07, right=0.98
        )
        for extension in ("png", "svg"):
            destination = output / (filename + "." + extension)
            figure.savefig(destination, dpi=180, facecolor="white")
            if extension == "svg":
                destination.write_text(
                    "\n".join(
                        line.rstrip() for line in destination.read_text().splitlines()
                    )
                    + "\n"
                )
        plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "metadata", "environment", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--expected-python", default=BASELINE_COMMIT)
    parser.add_argument("--expected-before", default=BEFORE_COMMIT)
    parser.add_argument("--expected-after", required=True)
    args = parser.parse_args()
    report = validate(
        args.run.resolve(),
        args.metadata.resolve(),
        args.environment.resolve(),
        expected_python=args.expected_python,
        expected_before=args.expected_before,
        expected_after=args.expected_after,
    )
    args.output.mkdir(parents=True, exist_ok=False)
    charts(report, args.output)
    report["charts"] = {
        path.name: sha(path)
        for path in sorted(args.output.iterdir())
        if path.suffix in (".png", ".svg")
    }
    (args.output / "results-linux-arm64.json").write_bytes(canonical(report) + b"\n")
    chart_count = 3
    print(
        f"Validated 40 qualifications and 200 timings; wrote results and {chart_count} PNG/SVG charts to {args.output}"
    )


if __name__ == "__main__":
    main()
