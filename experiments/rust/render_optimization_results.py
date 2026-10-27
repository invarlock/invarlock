#!/usr/bin/env python3
"""Validate a relocated complete Linux run and render descriptive public results."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import statistics
from pathlib import Path, PurePosixPath

CANDIDATES = ("reference", "rust-standalone-before", "rust-ffi", "rust-standalone")
LABELS = ("Python", "Old Rust standalone", "New Rust FFI", "New Rust standalone")
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
AFTER_COMMIT = "5e4c0c53ac2fe593880fda713b3c57f9d3d12478"
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
    value = json.loads(Path(path).read_bytes())
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
    shift = round_index % 4
    return base[shift:] + base[:shift]


def validate(run, metadata_path, environment_path):
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
        ("qualification_attempted", 32),
        ("qualification_valid", 32),
        ("timing_attempted", 128),
        ("timing_valid", 128),
        ("retries", 0),
        ("exclusions", 0),
    ):
        require(summary.get(key) == value, f"unexpected {key}")
    require(
        metadata == manifest["metadata"]
        and sha(metadata_path) == manifest["metadata_sha256"],
        "run metadata identity differs",
    )
    require(
        metadata["source_commit"] == BASELINE_COMMIT,
        "unexpected Python baseline commit",
    )
    require(
        metadata["before_source"]["commit"].startswith("09e23090"),
        "unexpected before source commit",
    )
    require(
        metadata["after_source"]["commit"] == AFTER_COMMIT,
        "unexpected final candidate source commit",
    )
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
            "rounds": 4,
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
    samples, groups, names = [], {}, set()
    identity = None
    for phase in ("qualification", "measurement"):
        for workload_index, fixture in enumerate(fixtures["workloads"]):
            for round_index in range(1 if phase == "qualification" else 4):
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
                    inputs = run / (name + "-inputs")
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
                        row["payload"]["native_transactions"]
                        == (0 if candidate == "reference" else len(request["steps"])),
                        "native operation count differs",
                    )
                    if candidate in ("reference", "rust-ffi"):
                        current = {
                            key: value
                            for key, value in row["payload"]["identity"].items()
                            if key != "check_wall_seconds"
                        }
                        require(
                            current["executable"]
                            == row["command"][0]
                            == manifest["python"],
                            "interpreter identity differs",
                        )
                        for key in ("root_wheel", "native_wheel"):
                            require(
                                hash_value(current["wheel_bindings"][key]["sha256"])
                                == hash_value(metadata[key]["sha256"]),
                                "installed wheel identity differs",
                            )
                        require(
                            identity is None or current == identity,
                            "installed runtime identity changed",
                        )
                        identity = current
                    else:
                        key = (
                            "standalone_before"
                            if candidate.endswith("-before")
                            else "standalone"
                        )
                        require(
                            row["command"][0] == metadata[key]["path"]
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
                            },
                            "output_file_sha256": output_files,
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
    observed = {path.name for path in run.iterdir() if path.is_dir()}
    require(observed == names, "unexpected, missing or orphaned attempt directories")
    require(len(samples) == 160 and len(groups) == 32, "sample denominator differs")
    computed = []
    for (workload, candidate), values in sorted(groups.items()):
        require(len(values) == 4, "cell does not have four observations")
        item = {
            "workload": workload,
            "candidate": candidate,
            "attempted": 4,
            "valid": 4,
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
    return {
        "format": "invarlock/linux-optimization-public-results-v1",
        "scope": "Linux ARM64 Docker VM; observed fixture journeys only. No production x86 or bare-metal performance claim.",
        "comparison_complete": True,
        "qualification_attempted": 32,
        "qualification_valid": 32,
        "timing_attempted": 128,
        "timing_valid": 128,
        "retries": 0,
        "exclusions": 0,
        "candidate_labels": dict(zip(CANDIDATES, LABELS, strict=True)),
        "sources": {
            "python_baseline": BASELINE_COMMIT,
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
            "python_version": identity["python"].split()[0],
            "native_version": identity["native_version"],
            "environment_sha256": sha(environment_path),
            "boundary": "Linux guest ext filesystem; inputs and outputs staged before timing; VM and host storage remain part of the environment.",
        },
        "protocol": matrix,
        "measurement_notes": [
            "Fresh processes; filesystem warmed by qualification.",
            "Whiskers show observed minimum and maximum of four timings; these are descriptive ranges, not confidence intervals.",
            "Peak RSS is for the entire process, in bytes; charts convert to MiB.",
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
    }


def charts(report, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cells = {(item["workload"], item["candidate"]): item for item in report["summary"]}
    colors = ("#334155", "#D97706", "#2563EB", "#0D9488")
    for metric, filename, title, unit, divisor in (
        ("wall_seconds", "wall-time", "Fresh-process wall time", "Seconds", 1),
        ("max_rss_bytes", "peak-rss", "Peak process memory", "MiB", 1024 * 1024),
    ):
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
                positions = [
                    index + (candidate_index - 1.5) * 0.19 for index in range(4)
                ]
                axis.bar(
                    positions,
                    medians,
                    width=0.18,
                    label=label,
                    color=color,
                    yerr=errors,
                    capsize=2,
                    error_kw={"elinewidth": 1},
                )
            logarithmic = metric == "wall_seconds" and panel == 0
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
            ncol=4,
            frameon=False,
        )
        figure.text(
            0.5,
            0.025,
            "Median with observed min–max whiskers; n=4 per cell. Descriptive ranges, not confidence intervals.",
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
    args = parser.parse_args()
    report = validate(
        args.run.resolve(), args.metadata.resolve(), args.environment.resolve()
    )
    args.output.mkdir(parents=True, exist_ok=False)
    charts(report, args.output)
    report["charts"] = {
        path.name: sha(path)
        for path in sorted(args.output.iterdir())
        if path.suffix in (".png", ".svg")
    }
    (args.output / "results-linux-arm64.json").write_bytes(canonical(report) + b"\n")
    print(
        f"Validated 32 qualifications and 128 timings; wrote results and two PNG/SVG charts to {args.output}"
    )


if __name__ == "__main__":
    main()
