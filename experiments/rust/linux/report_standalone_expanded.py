"""Render retained expanded Linux comparison summaries; never runs workloads."""

from __future__ import annotations

import argparse
import csv
import json
import textwrap
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "invarlock-standalone-expanded-v1"


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--preview", type=Path, required=True)
    args = parser.parse_args()
    results = args.repo / "experiments/rust/results"
    assets = args.repo / "docs/assets"
    args.preview.mkdir(parents=True, exist_ok=True)
    metrics = ["elapsed_seconds", "cpu_seconds", "peak_rss_mib"]
    labels = ["Elapsed-time ratio", "CPU-time ratio", "Peak-memory ratio"]
    colors = {"rust": "#14806d", "rust-ffi": "#4e63c4"}
    parts = [
        "# Expanded standalone Rust comparison\n",
        "The completed offline implementation is compared with corrected Python "
        "0.17.3 at `staging/next` revision "
        "`1dc9cb644684dc915810da77326f6da8764c58c1`. "
        "Standalone Rust and Python with Rust are measured separately. "
        "The latter retains Python startup, import adapters and facade work. "
        "It invokes the Python facade directly; the separate control-process "
        "compatibility launcher is functionally tested but not timed here.\n",
        "## Completed scope\n",
        "Native offline evaluation supports all eight export adapter entry points, "
        "nineteen evaluator profiles, retained runtime imports and retained judge "
        "imports. Verification, receipts, reports and handoff packages cover all "
        "four evidence families. Setup, strict JSON/YAML, environment defaults, "
        "plain terminal output and Linux binary bundles are implemented. "
        "Model SDK execution, live judge collection and custom Python plugins "
        "use an explicitly selected compatibility worker with bounded I/O, "
        "a deadline and process-group cancellation.\n",
        "## Comparison protocol\n",
        "Each Linux architecture ran 68 qualification executions followed by 340 "
        "timed executions: 24 workloads, five balanced rounds, seed 4127. "
        "Twenty workloads have all three candidates; four handoff-package "
        "workloads have Python and standalone Rust because the Python facade "
        "does not expose a Rust package backend. Every final execution matched. "
        "There were no retries or exclusions within either final run.\n",
        "Results are medians of five fresh processes. Elapsed time includes "
        "startup and complete command execution; CPU is user plus system time; "
        "memory is process peak RSS. Filesystem caches are not flushed. "
        "Rust may use bounded parallel work within the same CPU quota, so CPU "
        "time is reported separately from elapsed time. Identity checks happen outside measured "
        "workers. Both containers have four CPU equivalents and a 12 GiB limit. "
        "ARM64 runs in a local Docker VM; x86_64 runs in a remote "
        "KVM environment reporting AMD EPYC 9655 processors. Ratios compare "
        "implementations within each machine, not the machines against each "
        "other. Five samples establish observed medians, not universal "
        "performance guarantees. No inference, GPU or live provider work is timed.\n",
        "Machine JSON is compared with numeric types preserved. Deterministic "
        "files are compared by bytes, with output locations normalized. Package "
        "generator identities correctly differ; ZIP container metadata and "
        "compression differ, but every uncompressed member is checked against "
        "the delivered file. Standalone commands run with no discoverable Python "
        "interpreter. Default Rich terminal layout is outside byte parity; "
        "native reports print Markdown.\n",
        "## Observed median wins\n",
        "A win means a strictly lower measured median. Small differences are "
        "not claims of statistical significance.\n",
        "| Linux target | Candidate | Lower elapsed | Lower CPU | Lower peak RSS |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    datasets = {}
    csv_rows = []
    for arch in ["aarch64", "x86_64"]:
        data = json.loads((results / f"standalone-expanded-{arch}.json").read_text())
        assert data["all_matched"] and data["outcomes"] == 408
        assert data["qualified"] == 68 and data["timed"] == 340
        datasets[arch] = data
        for candidate, name in [
            ("rust", "Standalone Rust"),
            ("rust-ffi", "Python + Rust"),
        ]:
            rows = [r for r in data["rows"] if candidate in r]
            wins = [
                sum(r[candidate][m] < r["python"][m] for r in rows) for m in metrics
            ]
            parts.append(
                f"| {arch} | {name} | "
                + " | ".join(f"{n}/{len(rows)}" for n in wins)
                + " |"
            )
        for row in data["rows"]:
            for candidate in ["python", "rust-ffi", "rust"]:
                if candidate in row:
                    csv_rows.append(
                        {
                            "architecture": arch,
                            "workload": row["workload"],
                            "candidate": candidate,
                            **row[candidate],
                        }
                    )
    native_rows = [row for data in datasets.values() for row in data["rows"]]
    if all(row["rust"][m] < row["python"][m] for row in native_rows for m in metrics):
        ratios = [
            row["python"]["elapsed_seconds"] / row["rust"]["elapsed_seconds"]
            for row in native_rows
        ]
        savings = [
            100 * (1 - row["rust"]["peak_rss_mib"] / row["python"]["peak_rss_mib"])
            for row in native_rows
        ]
        parts.insert(
            1,
            f"Standalone Rust has lower median elapsed time, CPU time and peak RSS in all 24 measured workflows on both Linux targets. Observed elapsed speedups range from {min(ratios):.2f}× to {max(ratios):.2f}×, with peak RSS reductions from {min(savings):.1f}% to {max(savings):.1f}%. This supports the standalone offline architecture within the measured scope; it does not imply that every Python/Rust FFI path wins.\n",
        )
    parts.append("")
    for arch, data in datasets.items():
        rows = [row for row in data["rows"] if "rust-ffi" in row]
        memory_regressions = sum(
            row["rust-ffi"]["peak_rss_mib"] > row["python"]["peak_rss_mib"]
            for row in rows
        )
        regressions = []
        for row in rows:
            worse = [
                f"{label} +{100 * (row['rust-ffi'][metric] / row['python'][metric] - 1):.1f}%"
                for metric, label in [
                    ("elapsed_seconds", "elapsed"),
                    ("cpu_seconds", "CPU"),
                ]
                if row["rust-ffi"][metric] >= row["python"][metric]
            ]
            if worse:
                regressions.append(
                    "`" + row["workload"] + "` (" + ", ".join(worse) + ")"
                )
        parts.append(
            f"On {arch}, Python with Rust used more peak memory in {memory_regressions} of {len(rows)} workloads. Its elapsed/CPU median regressions were: "
            + ("; ".join(regressions) if regressions else "none")
            + ". The complete values below retain these outcomes; small timing differences remain subject to run variation.\n"
        )
    parts.append(
        "The results support using standalone Rust for offline work and retaining Python for compatibility integrations. FFI memory overhead is not solved: the Python facade still holds imported values and crosses conversion boundaries that the native offline path avoids. The separate buffer diagnostic below measures one boundary, not the whole facade.\n"
    )
    for arch, data in datasets.items():
        parts += [
            f"## Linux {arch}\n",
            f"![Ratios for Linux {arch}](../assets/rust-standalone-expanded-{arch}.svg)\n",
            "Seconds and MiB are absolute medians. A dash means no FFI route.\n",
            "| Workload | Python s | FFI s | Rust s | Python MiB | FFI MiB | Rust MiB |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in data["rows"]:
            values = []
            for metric in ["elapsed_seconds", "peak_rss_mib"]:
                for candidate in ["python", "rust-ffi", "rust"]:
                    values.append(
                        f"{row[candidate][metric]:.4f}"
                        if metric == "elapsed_seconds" and candidate in row
                        else f"{row[candidate][metric]:.2f}"
                        if candidate in row
                        else "—"
                    )
            parts.append(f"| `{row['workload']}` | " + " | ".join(values) + " |")
        parts.append("")
        fig, axes = plt.subplots(1, 3, figsize=(16, 10.5), sharey=True)
        fig.patch.set_facecolor("#fafbf9")
        rows = data["rows"]
        for axis, metric, label in zip(axes, metrics, labels, strict=True):
            axis.set_facecolor("#fafbf9")
            for candidate, offset, title in [
                ("rust", -0.13, "Standalone Rust"),
                ("rust-ffi", 0.13, "Python + Rust"),
            ]:
                xs, ys = [], []
                for i, row in enumerate(rows):
                    if candidate in row:
                        xs.append(row["python"][metric] / row[candidate][metric])
                        ys.append(i + offset)
                axis.scatter(
                    xs, ys, s=35, color=colors[candidate], label=title, zorder=3
                )
            axis.axvline(1, color="#b34a42", linewidth=1.1, linestyle="--")
            axis.set_xscale("log")
            if metric == "peak_rss_mib":
                axis.set_xticks([0.5, 1, 2, 4])
                axis.set_xlim(0.45, 6)
            else:
                axis.set_xticks([1, 2, 5, 10, 20, 40])
                axis.set_xlim(0.85, 46)
            axis.xaxis.set_major_formatter(
                FuncFormatter(lambda value, _: f"{value:g}×")
            )
            axis.set_xlabel(
                label + "\nPython / candidate · higher is better", fontsize=10
            )
            axis.grid(axis="x", alpha=0.17)
            axis.spines[["top", "right", "left"]].set_visible(False)
            axis.tick_params(axis="y", length=0)
        axes[0].set_yticks(range(len(rows)), [r["workload"] for r in rows], fontsize=9)
        axes[0].invert_yaxis()
        fig.legend(
            *axes[0].get_legend_handles_labels(),
            loc="upper right",
            bbox_to_anchor=(0.98, 0.965),
            frameon=False,
            fontsize=10,
            ncol=2,
        )
        fig.suptitle(
            f"Offline InvarLock · Linux {arch}",
            fontsize=20,
            fontweight="bold",
            x=0.03,
            ha="left",
        )
        fig.text(
            0.03,
            0.935,
            "24 workflows · five fresh-process samples · dashed line = equal to Python",
            fontsize=12,
            color="#47544f",
        )
        fig.subplots_adjust(left=0.20, right=0.98, top=0.90, bottom=0.10, wspace=0.16)
        fig.savefig(
            assets / f"rust-standalone-expanded-{arch}.svg", metadata={"Date": None}
        )
        svg_path = assets / f"rust-standalone-expanded-{arch}.svg"
        svg_path.write_text(
            "\n".join(line.rstrip() for line in svg_path.read_text().splitlines())
            + "\n"
        )
        fig.savefig(args.preview / f"rust-standalone-expanded-{arch}.png", dpi=130)
        plt.close(fig)
    parts += [
        "## Separate FFI buffer diagnostic\n",
        "This synthetic canonical-JSON diagnostic isolates immutable `bytes` "
        "and mutable `bytearray` input handling. Immutable input is borrowed; "
        "mutable input keeps an owned snapshot. Each size has three fresh "
        "processes per candidate. The 64-byte blob uses 1,000 calls per "
        "process; larger blobs use one. The inner clock excludes imports, "
        "input reads and the final hash; fresh-process measurements include "
        "them. All 36 executions per architecture matched output hashes. "
        "This is not a model of every FFI transaction.\n",
        "| Target | Blob | Candidate | Fresh s | Inner s | Peak MiB |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for arch in datasets:
        data = json.loads((results / f"ffi-boundary-{arch}.json").read_text())
        assert data["all_matched"] and data["outcomes"] == 36
        for row in data["rows"]:
            size = (
                "64 B × 1,000" if row["size"] == 64 else f"{row['size'] // 1024**2} MiB"
            )
            for candidate in ["python", "immutable", "mutable"]:
                m = row[candidate]
                parts.append(
                    f"| {arch} | {size} | {candidate} | {m['elapsed_seconds']:.5f} | {m['operation_seconds']:.5f} | {m['peak_rss_mib']:.2f} |"
                )
    parts.append("")
    parts += [
        "## Correctness evidence and limits\n",
        "Each final Linux artifact passed 583 differential tests and 16,448 "
        "subtests. The one Linux skip was a separate sdist archive input; the "
        "source archive's exact-byte test passed independently on the host. "
        "The tests exercise native contracts, replay, signing, filesystem "
        "mutation checks, public offline commands and worker cancellation. "
        "They do not establish equivalence for every possible input.\n",
        "A retained external-import corpus matched 275 accepted exports and "
        "191 rejection outcomes, with numeric types checked for accepted "
        "values. One test-only altered Langfuse record limit is retained as "
        "an alternate configuration. The built-in provider identity suite "
        "covers 521 configuration cases across four providers. These are "
        "configuration cases, not 521 model executions. The comparison also "
        "exposed and repaired a native TensorRT artifact-format binding typo.\n",
        "The corrected Python implementation is a reference, not an infallible "
        "oracle. It includes the merged paired-overflow error-normalization "
        "fix from PR #301. The [independent reference audit](rust-reference-audit.md) "
        "retains separate arithmetic checks and their numeric-domain limits. "
        "Independent native parsing, mutation/rejection tests and "
        "contract checks reduce shared blind spots; matching implementations "
        "alone cannot prove the underlying statistical or product design "
        "universally correct. Exact malformed-input wording, simultaneous "
        "failure ordering and rare invalid-signature behavior are not "
        "exhaustively equivalent. Native parser/report work limits remain "
        "documented in the workspace README.\n",
        "Rust workspace checks and Clippy passed. Python test, lint and type "
        "gates passed; documentation and examples gates passed after targeted "
        "repairs. The initially failed aggregate gate was repaired by rerunning "
        "its affected targets. Optional live evaluator SDK tests remain "
        "skipped, and live model/provider calls, GPU execution, real OCI "
        "engines, Windows and older Linux C library compatibility were not "
        "qualified. Package timings use small retained evidence fixtures; large "
        "real handoff packs require separate performance evidence. No paid or "
        "live model calls were made.\n",
        "## Retained failed setup attempts\n",
        "The final comparison began only after qualification. Earlier attempts "
        "remain retained: missing binary environment variables in a host test "
        "invocation; missing HTTP test dependencies and process utilities on "
        "Linux; oversized parametrized test names leaking into child "
        "environments; and a custom-plugin child missing its test dependency "
        "path. Test environment and fixture repairs were checked on both "
        "architectures. An unchanged recovery archive was copied after a "
        "failed edit command but was not used for qualification.\n",
        "The added-workload setup first pointed FFI at the baseline Python "
        "installation, which has no Rust-backend option. A separate "
        "experimental-facade installation repaired that boundary without "
        "changing the baseline. The next setup retained 15 matched executions "
        "before Python rejected a synthetic DeepEval metric in the wrong "
        "shape. Correcting that fixture produced 24 matched setup executions. "
        "These attempts are separate from the final 816 workflow executions "
        "and 72 buffer-diagnostic executions; none was discarded or counted "
        "as a successful final comparison.\n",
        "## Reproduction and artifacts\n",
        "The measured native source archive SHA-256 is "
        "`80385c6dab7bf8adb0410847c2527eb5021d844f4a944fe2d1f636b712477496`. "
        "Fixture and harness repairs followed the binary source freeze and "
        "have separate identities in each summary. The candidate Python wheel, "
        "native extension, installed baseline files, supervisor, fixture "
        "manifest and native executable are also individually identified.\n",
        "Compact machine-readable summaries and the median CSV are in "
        "`experiments/rust/results/`. Full attempts, output files, fixtures, "
        "binaries, wheels, setup failures and logs are retained on TB4; archive "
        "identities are recorded in `standalone-expanded-artifacts.json`. "
        "Fixture signing keys are synthetic and remain in the local retained "
        "evidence, not the public report. The "
        "[installation guide](rust-native-installation.md) describes the "
        "Linux bundles and explicit compatibility worker.\n",
    ]
    document = "\n".join(parts)
    document = (
        "\n".join(
            line
            if line.startswith(("#", "|", "!["))
            else textwrap.fill(
                line, width=96, break_long_words=False, break_on_hyphens=False
            )
            for line in document.splitlines()
        )
        + "\n"
    )
    (args.repo / "docs/reference/rust-standalone-expanded-results.md").write_text(
        document
    )
    with (results / "standalone-expanded-medians.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(csv_rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(csv_rows)


if __name__ == "__main__":
    main()
