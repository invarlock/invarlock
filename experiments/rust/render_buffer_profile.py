#!/usr/bin/env python3
"""Validate and render the separately measured FFI byte-buffer diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def summarize(profile, metadata):
    plan = json.loads((profile / "plan.json").read_bytes())
    rows = json.loads((profile / "results.json").read_bytes())
    source = json.loads(metadata.read_bytes())
    if plan["metadata_sha256"] != sha(metadata):
        raise ValueError("profile metadata differs")
    sizes = [64, 1024 * 1024, 16 * 1024 * 1024, 64 * 1024 * 1024]
    if (
        plan["format"] != "invarlock/ffi-buffer-profile-v1"
        or plan["sizes"] != sizes
        or plan["rounds"] != 3
    ):
        raise ValueError("unexpected profile protocol")
    keys = {
        (size, kind, candidate, index)
        for size in sizes
        for kind in ("immutable", "mutable")
        for candidate in ("before", "after")
        for index in range(3)
    }
    observed = [(r["size"], r["kind"], r["candidate"], r["round"]) for r in rows]
    if len(rows) != 48 or len(set(observed)) != 48 or set(observed) != keys:
        raise ValueError("profile attempts differ from the declared matrix")
    expected = {}
    for size in sizes:
        with (profile / f"input-{size}.json").open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256")
        digest.update(b"\n")
        expected[size] = digest.hexdigest()
    for row in rows:
        metrics, payload = row["metrics"], row["payload"]
        numbers = [
            metrics[k]
            for k in (
                "wall_seconds",
                "user_cpu_seconds",
                "system_cpu_seconds",
                "max_rss_bytes",
            )
        ]
        numbers.append(payload["operation_wall_seconds"])
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
            for value in numbers
        ):
            raise ValueError("invalid profile resource measurement")
        if (
            metrics["exit_code"] != 0
            or metrics["timed_out"]
            or payload["output_sha256"] != expected[row["size"]]
            or payload["native_modules"] != plan["native_modules"][row["candidate"]]
            or payload["iterations"] != (1000 if row["size"] == 64 else 1)
            or payload["output_bytes"] != row["size"] + len(b'{"blob":""}\n')
        ):
            raise ValueError("profile outcome or identity differs")
    groups = []
    for size in sizes:
        for kind in ("immutable", "mutable"):
            for candidate in ("before", "after"):
                selected = [
                    r
                    for r in rows
                    if (r["size"], r["kind"], r["candidate"]) == (size, kind, candidate)
                ]
                values = {
                    "peak_rss_mib": [
                        r["metrics"]["max_rss_bytes"] / 2**20 for r in selected
                    ],
                    "wall_seconds": [r["metrics"]["wall_seconds"] for r in selected],
                    "cpu_seconds": [
                        r["metrics"]["user_cpu_seconds"]
                        + r["metrics"]["system_cpu_seconds"]
                        for r in selected
                    ],
                    "operation_seconds_per_call": [
                        r["payload"]["operation_wall_seconds"]
                        / r["payload"]["iterations"]
                        for r in selected
                    ],
                }
                groups.append(
                    {
                        "size": size,
                        "kind": kind,
                        "candidate": candidate,
                        "process_observations": 3,
                        **{
                            key: {
                                "median": statistics.median(v),
                                "minimum": min(v),
                                "maximum": max(v),
                            }
                            for key, v in values.items()
                        },
                    }
                )
    return {
        "format": "invarlock/ffi-buffer-results-v1",
        "python_reference": source["source_commit"],
        "native_before": source["before_source"],
        "native_after": source["after_source"],
        "attempts": 48,
        "exact_outputs": 48,
        "retries": 0,
        "exclusions": 0,
        "plan_sha256": sha(profile / "plan.json"),
        "raw_results_sha256": sha(profile / "results.json"),
        "script_sha256": plan["script_sha256"],
        "metadata_sha256": plan["metadata_sha256"],
        "scope": plan["scope"],
        "limitations": [
            "Three fresh processes per configuration; descriptive ranges, not population intervals.",
            "The 64-byte case makes 1000 calls per process; these are three independent observations, not 3000.",
            "RSS includes interpreter startup, input reads and output/module hashing; operation time covers native calls only.",
            "Direct canonicalization diagnostics are separate from full-workflow measurements.",
        ],
        "summary": groups,
    }


def charts(report, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sizes = [64, 2**20, 16 * 2**20, 64 * 2**20]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    for axis, kind, title in zip(
        axes,
        ("immutable", "mutable"),
        ("Immutable bytes: input copy removed", "Mutable buffers: snapshot retained"),
        strict=True,
    ):
        for offset, candidate, color, label in [
            (-0.18, "before", "#8b929a", "Before"),
            (0.18, "after", "#2779a5", "After"),
        ]:
            values = [
                next(
                    r["peak_rss_mib"]
                    for r in report["summary"]
                    if r["size"] == size
                    and r["kind"] == kind
                    and r["candidate"] == candidate
                )
                for size in sizes
            ]
            medians = [v["median"] for v in values]
            axis.bar(
                [i + offset for i in range(4)],
                medians,
                width=0.34,
                color=color,
                label=label,
                yerr=[
                    [v["median"] - v["minimum"] for v in values],
                    [v["maximum"] - v["median"] for v in values],
                ],
                capsize=3,
            )
        axis.set_title(title, fontsize=11)
        axis.set_xticks(range(4), ["64 B", "1 MiB", "16 MiB", "64 MiB"])
        axis.set_xlabel("JSON string payload size")
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Fresh-process peak RSS (MiB)")
    axes[1].legend(frameon=False)
    fig.suptitle("Rust FFI buffer memory on Linux ARM64", fontsize=15)
    fig.text(
        0.5,
        0.025,
        "Medians and observed ranges of 3 processes. All 48 output hashes match.\nIncludes Python startup, input reads and hashing; wheel checks run outside the measured child.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.11, 1, 0.95))
    for suffix in ("png", "svg"):
        fig.savefig(
            output / f"buffer-memory.{suffix}",
            dpi=180,
            metadata={"Date": None} if suffix == "svg" else None,
        )
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("profile", "metadata", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.profile, args.metadata)
    args.output.mkdir(parents=True, exist_ok=False)
    charts(report, args.output)
    report["charts"] = {p.name: sha(p) for p in args.output.iterdir()}
    (args.output / "buffer-results.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print("Validated 48 buffer observations and wrote the summary and chart.")


if __name__ == "__main__":
    main()
