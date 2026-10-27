#!/usr/bin/env python3
"""Export explicit inputs, prepare native Linux artifacts, and run the fixed matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
from datetime import UTC, datetime
from pathlib import Path

HARNESS = (
    "__init__.py",
    "adapters.py",
    "fixtures.py",
    "full_fixtures.py",
    "full_runner.py",
    "full_worker.py",
    "runner.py",
    "smoke.py",
    "worker.py",
    "optimization_runner.py",
)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def capture(command, **kwargs):
    return subprocess.check_output(command, text=True, **kwargs).strip()


def descriptor(path):
    return {"path": str(Path(path).resolve()), "sha256": digest(path)}


def copy_verified(source, expected, destination):
    source = Path(source)
    if source.is_symlink() or digest(source) != expected.removeprefix("sha256:"):
        raise ValueError(f"input is changed or a symlink: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    if digest(destination) != expected.removeprefix("sha256:"):
        raise ValueError(f"copied input changed: {destination}")


def export(args):
    repo = args.repository.resolve()
    output = args.output.resolve()
    metadata = json.loads(args.metadata.read_text())
    manifest = json.loads(args.fixtures.read_text())
    if digest(args.fixtures) != metadata["fixtures_manifest"]["sha256"].removeprefix(
        "sha256:"
    ):
        raise ValueError("fixture manifest differs from the frozen metadata")
    paths = sorted({path for row in manifest["workloads"] for path in row["files"]})
    fixture_root = Path(os.path.commonpath(paths))
    output.mkdir(parents=True, exist_ok=False)
    sources = {}
    for name, ref in (("before", "09e23090"), ("after", args.after)):
        commit = capture(
            ["git", "-C", str(repo), "rev-parse", "--verify", ref + "^{commit}"]
        )
        tree = capture(["git", "-C", str(repo), "rev-parse", commit + ":native"])
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
        copy_verified(item["path"], item["sha256"], target)
        sources[name] = str(target.relative_to(output))
    for workload in manifest["workloads"]:
        for source, expected in workload["files"].items():
            target = output / "fixtures" / Path(source).relative_to(fixture_root)
            copy_verified(source, expected, target)
    harness = output / "harness" / "invarlock_rust_experiment"
    harness.mkdir(parents=True)
    for name in HARNESS:
        source = repo / "experiments/rust/python/invarlock_rust_experiment" / name
        expected = metadata["harness_sources"].get(str(source))
        if name != "optimization_runner.py" and expected is None:
            raise ValueError(f"original harness lacks a frozen identity: {name}")
        copy_verified(source, expected or digest(source), harness / name)
    shutil.copyfile(args.fixtures, output / "fixtures-original.json")
    shutil.copyfile(
        repo / "experiments/rust/matrix-optimization.json", output / "matrix.json"
    )
    shutil.copyfile(__file__, output / "launch.py")
    sources.update(
        format="invarlock/linux-optimization-inputs-v1",
        created_at=datetime.now(UTC).isoformat(),
        baseline_commit=metadata["source_commit"],
        baseline_tree=metadata["source_tree_sha256"],
        original_fixture_root=str(fixture_root),
        original_metadata_sha256=digest(args.metadata),
        fixture_manifest_sha256=digest(args.fixtures),
        harness_scope="Separate frozen harness bytes; after archive contains committed native sources only.",
    )
    sources["files"] = {
        str(path.relative_to(output)): digest(path)
        for path in sorted(output.rglob("*"))
        if path.is_file()
    }
    write(output / "bundle.json", sources)
    print(output)


def verify_bundle(root):
    bundle = json.loads((root / "bundle.json").read_text())
    for relative, expected in bundle["files"].items():
        path = root / relative
        if (
            not path.resolve().is_relative_to(root.resolve())
            or path.is_symlink()
            or digest(path) != expected
        ):
            raise ValueError(f"bundle identity changed: {relative}")
    return bundle


def require_linux_storage(path):
    if sys.platform != "linux":
        raise RuntimeError("native Linux preparation and measurement require Linux")
    kind = capture(["stat", "-f", "-c", "%T", str(path)])
    if kind not in {"ext2/ext3", "ext2", "ext3", "ext4"}:
        raise RuntimeError(
            f"work directory must use Linux ext storage, observed {kind}"
        )
    return kind


def rebind(value, old_root, new_root):
    if isinstance(value, str):
        if value == old_root or value.startswith(old_root + "/"):
            return new_root + value[len(old_root) :]
        return value
    if isinstance(value, list):
        return [rebind(item, old_root, new_root) for item in value]
    if isinstance(value, dict):
        return {
            rebind(key, old_root, new_root): rebind(item, old_root, new_root)
            for key, item in value.items()
        }
    return value


def extract_native(archive, output):
    output.mkdir()
    with tarfile.open(archive) as source:
        for member in source.getmembers():
            if not (member.isfile() or member.isdir()) or not (
                output / member.name
            ).resolve().is_relative_to(output.resolve()):
                raise ValueError(
                    "native source archive contains a link or escaping path"
                )
        source.extractall(output, filter="data")


def logged(command, log, **kwargs):
    with log.open("ab") as stream:
        stream.write(
            ("COMMAND " + json.dumps([str(item) for item in command]) + "\n").encode()
        )
        stream.flush()
        subprocess.run(
            command, stdout=stream, stderr=subprocess.STDOUT, check=True, **kwargs
        )


def prepare(args):
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("preparation requires CPython 3.12")
    args.work = args.work.resolve()
    require_linux_storage(args.work.parent)
    bundle = verify_bundle(args.inputs.resolve())
    args.work.mkdir(exist_ok=False)
    work = args.work
    shutil.copytree(args.inputs, work / "inputs")
    inputs = work / "inputs"
    verify_bundle(inputs)
    logs = work / "logs"
    logs.mkdir()
    rust = capture(["rustc", "--version"])
    if not rust.startswith("rustc 1.94.0 "):
        raise RuntimeError(f"requires native Rust 1.94.0, observed {rust}")
    maturin = capture([sys.executable, "-m", "maturin", "--version"])
    if maturin != "maturin 1.12.0":
        raise RuntimeError(f"requires Maturin 1.12.0, observed {maturin}")
    rust_identity = capture(["rustc", "-vV"])
    target = next(
        line.removeprefix("host: ")
        for line in rust_identity.splitlines()
        if line.startswith("host: ")
    )
    if target.split("-")[0] != platform.machine():
        raise RuntimeError("Rust host architecture differs from the running Linux host")
    wheels = work / "wheels"
    wheels.mkdir()
    environment = os.environ.copy()
    for key in (
        "CARGO_BUILD_TARGET",
        "RUSTFLAGS",
        "CARGO_ENCODED_RUSTFLAGS",
        "RUSTC_WRAPPER",
        "RUSTC_WORKSPACE_WRAPPER",
    ):
        environment.pop(key, None)
    environment.update(CARGO_NET_OFFLINE="true", CARGO_BUILD_JOBS="2")
    for name in ("before", "after"):
        extract_native(inputs / (name + ".tar"), work / name)
        native = work / name / "native"
        environment["CARGO_TARGET_DIR"] = str(work / (name + "-target"))
        logged(
            [
                "cargo",
                "build",
                "--release",
                "--locked",
                "--offline",
                "--target",
                target,
                "--workspace",
                "--bins",
                "--examples",
            ],
            logs / (name + "-build.log"),
            cwd=native,
            env=environment,
        )
        if name == "after":
            logged(
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
                logs / "after-wheel.log",
                cwd=native,
                env=environment,
            )
    native_wheels = list(wheels.glob("invarlock_native_experiment-*.whl"))
    if len(native_wheels) != 1:
        raise RuntimeError("expected exactly one freshly built native wheel")
    # Copy explicitly supplied Linux wheel files; no Cargo credentials/config or host homes.
    wheelhouse = work / "dependency-wheels"
    wheelhouse.mkdir()
    for path in sorted(args.wheelhouse.glob("*.whl")):
        if path.is_symlink():
            raise ValueError("dependency wheel symlinks are not accepted")
        shutil.copyfile(path, wheelhouse / path.name)
    logged([sys.executable, "-m", "venv", str(work / "venv")], logs / "install.log")
    python = work / "venv/bin/python"
    lock = inputs / bundle["dependency_lock"]
    root_wheel = inputs / bundle["root_wheel"]
    logged(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--no-index",
            "--find-links",
            str(wheelhouse),
            "--require-hashes",
            "-r",
            str(lock),
        ],
        logs / "install.log",
    )
    logged(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--no-index",
            "--no-deps",
            str(root_wheel),
            str(native_wheels[0]),
        ],
        logs / "install.log",
    )
    logged([str(python), "-m", "pip", "check"], logs / "install.log")
    original = json.loads((inputs / "fixtures-original.json").read_text())
    # Rebind only the manifest. Signed fixture file contents are never rewritten.
    manifest = rebind(
        original, bundle["original_fixture_root"], str(inputs / "fixtures")
    )
    for row in manifest["workloads"]:
        for path, expected in row["files"].items():
            if digest(path) != expected:
                raise ValueError(f"rebinding changed fixture bytes: {path}")
        for step in row["request"]["steps"]:
            for value in step["config"].values():
                if (
                    isinstance(value, str)
                    and value.startswith("/")
                    and not Path(value).is_relative_to(inputs / "fixtures")
                ):
                    raise ValueError(f"unbound absolute request path: {value}")
    write(work / "fixtures.json", manifest)
    facts = {
        "as_of_utc": datetime.now(UTC).isoformat(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version,
        "rustc": rust_identity,
        "cargo": capture(["cargo", "-vV"]),
        "maturin": maturin,
        "host_description": args.host_description,
        "vm_description": args.vm_description,
        "storage_description": args.storage_description,
        "work_filesystem": require_linux_storage(work),
        "cpuinfo": Path("/proc/cpuinfo").read_text(),
        "meminfo": Path("/proc/meminfo").read_text(),
        "mountinfo": Path("/proc/self/mountinfo").read_text(),
        "dependency_wheels": {
            path.name: digest(path) for path in wheelhouse.glob("*.whl")
        },
        "scope": "This Linux architecture and host/VM only; no production x86 performance claim.",
    }
    for name in ("cpu.max", "memory.max", "cpuset.cpus.effective"):
        path = Path("/sys/fs/cgroup") / name
        facts[name] = path.read_text().strip() if path.exists() else "unavailable"
    write(work / "environment.json", facts)
    metadata = {
        "source_commit": bundle["baseline_commit"],
        "source_tree_sha256": bundle["baseline_tree"],
        "before_source": bundle["before"],
        "after_source": bundle["after"],
        "input_bundle": descriptor(inputs / "bundle.json"),
        "environment": descriptor(work / "environment.json"),
        "root_wheel": descriptor(root_wheel),
        "native_wheel": descriptor(native_wheels[0]),
        "dependency_lock": descriptor(lock),
        "native_cargo_lock": descriptor(work / "after/native/Cargo.lock"),
        "before_cargo_lock": descriptor(work / "before/native/Cargo.lock"),
        "standalone_before": descriptor(
            work / "before-target" / target / "release/examples/transaction_benchmark"
        ),
        "standalone": descriptor(
            work / "after-target" / target / "release/examples/transaction_benchmark"
        ),
        "offline_cli": descriptor(
            work / "after-target" / target / "release/invarlock-native-verify"
        ),
        "control_cli": descriptor(
            work / "after-target" / target / "release/invarlock-native"
        ),
        "fixtures_manifest": descriptor(work / "fixtures.json"),
        "matrix": descriptor(inputs / "matrix.json"),
        "harness_sources": {
            str(path.resolve()): digest(path)
            for path in (inputs / "harness/invarlock_rust_experiment").glob("*.py")
        },
    }
    write(work / "metadata.json", metadata)
    print(f"Artifacts prepared, not yet qualified: {work}")


def run(args):
    work = args.work.resolve()
    require_linux_storage(work)
    verify_bundle(work / "inputs")
    output = work / "runs" / args.name
    if Path(args.name).name != args.name or args.name in {".", ".."}:
        raise ValueError("run name must be one new directory component")
    python = work / "venv/bin/python"
    command = [
        str(python),
        "-I",
        "-c",
        "import sys;sys.path.insert(0,sys.argv.pop(1));from invarlock_rust_experiment.optimization_runner import main;raise SystemExit(main())",
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
        digest(work / "inputs/matrix.json"),
        "--output",
        str(output),
    ]
    if args.qualification_only:
        command.append("--qualification-only")
    log = work / "logs" / (args.name + "-run.log")
    if log.exists() or output.exists():
        raise ValueError("run name already exists; no automatic retry or continuation")
    facts = {
        "as_of_utc": datetime.now(UTC).isoformat(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "work_filesystem": require_linux_storage(work),
        "prepared_environment": descriptor(work / "environment.json"),
        "mountinfo": Path("/proc/self/mountinfo").read_text(),
        "meminfo": Path("/proc/meminfo").read_text(),
    }
    for name in ("cpu.max", "memory.max", "cpuset.cpus.effective"):
        path = Path("/sys/fs/cgroup") / name
        facts[name] = path.read_text().strip() if path.exists() else "unavailable"
    write(work / "logs" / (args.name + "-environment.json"), facts)
    logged(command, log)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser(
        "export", help="copy an explicit, hashed source/fixture bundle; no builds"
    )
    for name in ("repository", "metadata", "fixtures", "output"):
        command.add_argument("--" + name, type=Path, required=True)
    command.add_argument(
        "--after",
        required=True,
        help="committed candidate revision; dirty edits are excluded",
    )
    command = commands.add_parser(
        "prepare",
        help="build/install on Linux using pre-provisioned offline tools/caches",
    )
    for name in ("inputs", "work", "wheelhouse"):
        command.add_argument("--" + name, type=Path, required=True)
    for name in ("host-description", "vm-description", "storage-description"):
        command.add_argument("--" + name, required=True)
    command = commands.add_parser(
        "run", help="32 qualifications then 128 measurements, or qualification only"
    )
    command.add_argument("--work", type=Path, required=True)
    command.add_argument("--name", required=True)
    command.add_argument("--qualification-only", action="store_true")
    args = parser.parse_args()
    {"export": export, "prepare": prepare, "run": run}[args.command](args)


if __name__ == "__main__":
    main()
