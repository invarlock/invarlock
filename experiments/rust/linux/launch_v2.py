#!/usr/bin/env python3
"""Separate export/build/run path for the five-candidate measurement revision."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import launch as legacy

NEW_HARNESS = ("comparison_v2.py", "bounded_worker.py")
SUPPORT = ("measure_v2.c", "qualify_measurement_v2.py", "launch.py", "launch_v2.py")


PRIOR_MEASURED_COMMIT = "e3caec51941990a2d620c00984c8b08e662e8b76"


def harness_revision(repo, name, old):
    relative = "experiments/rust/python/invarlock_rust_experiment/" + name
    source = repo / relative
    observed = legacy.digest(source)
    reference = None
    if name == "optimization_runner.py" and old is None:
        # This module postdates the original expanded metadata. Bind its exact
        # prior measured revision instead of inventing an original identity.
        raw = subprocess.check_output(
            ["git", "-C", str(repo), "show", PRIOR_MEASURED_COMMIT + ":" + relative]
        )
        if hashlib.sha256(raw).hexdigest() != observed:
            raise ValueError(
                "later-added optimization runner differs from frozen prior revision"
            )
        reference = {"commit": PRIOR_MEASURED_COMMIT, "path": relative}
        change = "added after original baseline; bound to prior measured revision"
    elif name in NEW_HARNESS:
        change = "new measurement revision"
    elif old != observed:
        raise ValueError(
            f"historical harness source changed: {name}; keep legacy bytes intact"
        )
    else:
        change = "unchanged"
    return {
        "prior_sha256": old,
        "current_sha256": observed,
        "change": change,
        "source_reference": reference,
    }


def export(args):
    repo, output = args.repository.resolve(), args.output.resolve()
    metadata = json.loads(args.metadata.read_text())
    manifest = json.loads(args.fixtures.read_text())
    if legacy.digest(args.fixtures) != metadata["fixtures_manifest"][
        "sha256"
    ].removeprefix("sha256:"):
        raise ValueError("fixture manifest differs from supplied frozen metadata")
    paths = sorted({path for row in manifest["workloads"] for path in row["files"]})
    fixture_root = Path(os.path.commonpath(paths))
    output.mkdir(parents=True, exist_ok=False)
    sources = {}
    for name, ref in (("before", args.before), ("after", args.after)):
        commit = legacy.capture(
            ["git", "-C", str(repo), "rev-parse", "--verify", ref + "^{commit}"]
        )
        tree = legacy.capture(["git", "-C", str(repo), "rev-parse", commit + ":native"])
        sources[name] = {"commit": commit, "native_tree": tree}
        with (output / (name + ".tar")).open("xb") as stream:
            subprocess.run(
                ["git", "-C", str(repo), "archive", "--format=tar", commit, "native"],
                stdout=stream,
                check=True,
            )
    for name in ("root_wheel", "dependency_lock"):
        item = metadata[name]
        target = output / "artifacts" / Path(item["path"]).name
        legacy.copy_verified(item["path"], item["sha256"], target)
        sources[name] = str(target.relative_to(output))
    for workload in manifest["workloads"]:
        for source, expected in workload["files"].items():
            legacy.copy_verified(
                source,
                expected,
                output / "fixtures" / Path(source).relative_to(fixture_root),
            )
    historical = {
        Path(path).name: value for path, value in metadata["harness_sources"].items()
    }
    harness = output / "harness/invarlock_rust_experiment"
    harness.mkdir(parents=True)
    revisions = {}
    for name in (*legacy.HARNESS, *NEW_HARNESS):
        source = repo / "experiments/rust/python/invarlock_rust_experiment" / name
        revision = harness_revision(repo, name, historical.get(name))
        legacy.copy_verified(source, revision["current_sha256"], harness / name)
        revisions[name] = revision
    for name in SUPPORT:
        shutil.copyfile(repo / "experiments/rust/linux" / name, output / name)
    shutil.copyfile(args.fixtures, output / "fixtures-original.json")
    shutil.copyfile(
        repo / "experiments/rust/matrix-comparison-v2.json", output / "matrix.json"
    )
    legacy.write(
        output / "harness-revision.json",
        {
            "format": "invarlock/harness-revision-v2",
            "prior_metadata": legacy.descriptor(args.metadata),
            "files": revisions,
            "reason": "Fresh-exec child measurement, event-driven timeout, bounded installed-wheel verification, five independently bound candidates.",
            "measurement_scope": "Monotonic fork-to-wait4 interval of actual worker; excludes supervisor startup and retained-output validation. Unadjusted direct-child wait4 CPU/RSS; RSS includes small supervisor pre-exec image, not outer Python memory.",
        },
    )
    sources.update(
        format="invarlock/linux-comparison-inputs-v2",
        created_at=datetime.now(UTC).isoformat(),
        baseline_commit=metadata["source_commit"],
        baseline_tree=metadata["source_tree_sha256"],
        original_fixture_root=str(fixture_root),
        original_metadata_sha256=legacy.digest(args.metadata),
        fixture_manifest_sha256=legacy.digest(args.fixtures),
        harness_scope="Historical source bytes retained; explicitly versioned additional measurement/worker modules.",
    )
    sources["files"] = {
        str(path.relative_to(output)): legacy.digest(path)
        for path in sorted(output.rglob("*"))
        if path.is_file()
    }
    legacy.write(output / "bundle.json", sources)
    print(output)


def prepare(args):
    # The prior build workflow remains byte-for-byte reproducible. Reuse it to
    # build both standalone revisions and the after wheel in this new work tree.
    legacy.prepare(args)
    work = args.work.resolve()
    metadata = json.loads((work / "metadata.json").read_text())
    shutil.copyfile(work / "metadata.json", work / "metadata-pre-v2-bindings.json")
    target = next(
        line.removeprefix("host: ")
        for line in legacy.capture(["rustc", "-vV"]).splitlines()
        if line.startswith("host: ")
    )
    wheels = work / "before-wheels"
    wheels.mkdir()
    env = os.environ.copy()
    for key in (
        "CARGO_BUILD_TARGET",
        "RUSTFLAGS",
        "CARGO_ENCODED_RUSTFLAGS",
        "RUSTC_WRAPPER",
        "RUSTC_WORKSPACE_WRAPPER",
    ):
        env.pop(key, None)
    env.update(
        CARGO_NET_OFFLINE="true",
        CARGO_BUILD_JOBS="2",
        CARGO_TARGET_DIR=str(work / "before-target"),
    )
    legacy.logged(
        [
            sys.executable,
            "-m",
            "maturin",
            "build",
            "--release",
            "--locked",
            "--offline",
            "--target",
            target,
            "--compatibility",
            "linux",
            "--interpreter",
            sys.executable,
            "--out",
            str(wheels),
        ],
        work / "logs/before-wheel.log",
        cwd=work / "before/native",
        env=env,
    )
    native_wheels = list(wheels.glob("invarlock_native_experiment-*.whl"))
    if len(native_wheels) != 1:
        raise ValueError("expected one independently built previous native wheel")
    legacy.logged(
        [sys.executable, "-m", "venv", str(work / "before-venv")],
        work / "logs/before-install.log",
    )
    python = work / "before-venv/bin/python"
    legacy.logged(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--no-index",
            "--find-links",
            str(work / "dependency-wheels"),
            "--require-hashes",
            "-r",
            metadata["dependency_lock"]["path"],
        ],
        work / "logs/before-install.log",
    )
    legacy.logged(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--no-index",
            "--no-deps",
            metadata["root_wheel"]["path"],
            str(native_wheels[0]),
        ],
        work / "logs/before-install.log",
    )
    legacy.logged([str(python), "-m", "pip", "check"], work / "logs/before-install.log")
    source = work / "inputs/measure_v2.c"
    supervisor = work / "measure_v2"
    legacy.logged(
        [
            "cc",
            "-std=c11",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-o",
            str(supervisor),
            str(source),
        ],
        work / "logs/supervisor-build.log",
    )
    before = dict(metadata)
    before.update(
        native_wheel=legacy.descriptor(native_wheels[0]),
        native_cargo_lock=metadata["before_cargo_lock"],
        standalone=metadata["standalone_before"],
    )
    legacy.write(work / "before-runtime-metadata.json", before)
    metadata.update(
        before_runtime_metadata=legacy.descriptor(
            work / "before-runtime-metadata.json"
        ),
        measurement_supervisor=legacy.descriptor(supervisor),
        measurement_source=legacy.descriptor(source),
        harness_revision=legacy.descriptor(work / "inputs/harness-revision.json"),
        supervisor_compiler=legacy.capture(["cc", "--version"]),
        runtimes={
            name: {
                "python": str(
                    work
                    / ("before-venv" if name == "rust-ffi-before" else "venv")
                    / "bin/python"
                ),
                "metadata": str(
                    work
                    / (
                        "before-runtime-metadata.json"
                        if name == "rust-ffi-before"
                        else "metadata.json"
                    )
                ),
            }
            for name in ("reference", "rust-ffi-before", "rust-ffi")
        },
    )
    legacy.write(work / "metadata.json", metadata)
    print(f"Five candidate artifacts prepared; qualification required: {work}")


def run(args):
    work = args.work.resolve()
    legacy.require_linux_storage(work)
    legacy.verify_bundle(work / "inputs")
    if Path(args.name).name != args.name or args.name in {".", ".."}:
        raise ValueError("run name must be one new directory component")
    output, log = work / "runs" / args.name, work / "logs" / (args.name + "-run.log")
    if output.exists() or log.exists():
        raise ValueError("run name already exists; no automatic retry or continuation")
    python = work / "venv/bin/python"
    command = [
        str(python),
        "-I",
        "-c",
        "import sys;sys.path.insert(0,sys.argv.pop(1));from invarlock_rust_experiment.comparison_v2 import main;raise SystemExit(main())",
        str(work / "inputs/harness"),
        "--python",
        str(python),
        "--metadata",
        str(work / "metadata.json"),
        "--fixtures",
        str(work / "fixtures.json"),
        "--matrix",
        str(work / "inputs/matrix.json"),
        "--matrix-sha256",
        legacy.digest(work / "inputs/matrix.json"),
        "--output",
        str(output),
    ]
    if args.qualification_only:
        command.append("--qualification-only")
    facts = {
        "as_of_utc": datetime.now(UTC).isoformat(),
        "prepared_environment": legacy.descriptor(work / "environment.json"),
        "measurement_supervisor": legacy.descriptor(work / "measure_v2"),
    }
    for name in ("cpu.max", "memory.max", "cpuset.cpus.effective"):
        path = Path("/sys/fs/cgroup") / name
        facts[name] = path.read_text().strip() if path.exists() else "unavailable"
    legacy.write(work / "logs" / (args.name + "-environment.json"), facts)
    legacy.logged(command, log)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    cmd = commands.add_parser("export")
    for name in ("repository", "metadata", "fixtures", "output"):
        cmd.add_argument("--" + name, type=Path, required=True)
    cmd.add_argument("--before", default="673d492dde9d7b320c6f082a5ca2a8191ab37f66")
    cmd.add_argument("--after", required=True)
    cmd = commands.add_parser("prepare")
    for name in ("inputs", "work", "wheelhouse"):
        cmd.add_argument("--" + name, type=Path, required=True)
    for name in ("host-description", "vm-description", "storage-description"):
        cmd.add_argument("--" + name, required=True)
    cmd = commands.add_parser("run")
    cmd.add_argument("--work", type=Path, required=True)
    cmd.add_argument("--name", required=True)
    cmd.add_argument("--qualification-only", action="store_true")
    args = parser.parse_args()
    {"export": export, "prepare": prepare, "run": run}[args.command](args)


if __name__ == "__main__":
    main()
