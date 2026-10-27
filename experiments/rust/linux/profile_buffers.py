#!/usr/bin/env python3
"""Bounded Linux diagnostic for immutable input borrowing, separate from workflows."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def worker(args):
    import invarlock_native as native

    raw = Path(args.input).read_bytes()
    if args.kind == "mutable":
        raw = bytearray(raw)
    started = time.perf_counter()
    for _ in range(args.iterations):
        result = native.canonicalize_json(raw)
    elapsed = time.perf_counter() - started
    print(
        json.dumps(
            {
                "ok": True,
                "output_sha256": hashlib.sha256(result).hexdigest(),
                "output_bytes": len(result),
                "operation_wall_seconds": elapsed,
                "iterations": args.iterations,
                "native_modules": {
                    p.name: digest(p) for p in Path(native.__file__).parent.glob("*.so")
                },
            }
        )
    )


def run(args):
    work, out = args.work.resolve(), args.output.resolve()
    if sys.platform != "linux":
        raise ValueError("this measurement requires Linux")
    out.mkdir(parents=True, exist_ok=False)
    metadata = json.loads((work / "metadata.json").read_bytes())
    supervisor = metadata["measurement_supervisor"]
    if digest(supervisor["path"]) != supervisor["sha256"].removeprefix("sha256:"):
        raise ValueError("measurement supervisor changed")
    bindings = {}
    for name, candidate in (("before", "rust-ffi-before"), ("after", "rust-ffi")):
        runtime = metadata["runtimes"][candidate]
        bindings[name] = json.loads(
            subprocess.check_output(
                [
                    runtime["python"],
                    "-I",
                    "-c",
                    "import sys,json;sys.path.insert(0,sys.argv[1]);from invarlock_rust_experiment.bounded_worker import check_wheel_binding;print(json.dumps(check_wheel_binding(sys.argv[2])))",
                    str(work / "inputs/harness"),
                    runtime["metadata"],
                ]
            )
        )
    plan = {
        "format": "invarlock/ffi-buffer-profile-v1",
        "sizes": [64, 1024 * 1024, 16 * 1024 * 1024, 64 * 1024 * 1024],
        "rounds": 3,
        "kinds": ["immutable", "mutable"],
        "iterations": {"64": 1000, "other": 1},
        "retry_rule": "No retries or exclusions; stop on first failure and retain all attempts.",
        "scope": "Fresh-process input read, native canonicalization, output hash, and native module hash; no wheel checks inside the measured process. Distinct from the full-workflow measurement boundary.",
        "bindings": bindings,
        "metadata_sha256": digest(work / "metadata.json"),
        "script_sha256": digest(__file__),
        "measurement_supervisor": supervisor,
    }
    (out / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    rows = []
    for size in plan["sizes"]:
        source = out / f"input-{size}.json"
        source.write_bytes(b'{"blob":"' + b"x" * size + b'"}')
        with source.open("rb") as stream:
            expected = hashlib.file_digest(stream, "sha256")
        expected.update(b"\n")
        for round_index in range(plan["rounds"]):
            for kind in plan["kinds"]:
                for name in (
                    ("before", "after") if round_index % 2 == 0 else ("after", "before")
                ):
                    directory = out / f"{size}-{round_index}-{kind}-{name}"
                    directory.mkdir()
                    candidate = "rust-ffi-before" if name == "before" else "rust-ffi"
                    runtime = metadata["runtimes"][candidate]
                    metrics = directory / "metrics.json"
                    command = [
                        supervisor["path"],
                        str(metrics),
                        "60",
                        "--",
                        runtime["python"],
                        "-I",
                        str(Path(__file__).resolve()),
                        "worker",
                        "--input",
                        str(source),
                        "--kind",
                        kind,
                        "--iterations",
                        str(1000 if size == 64 else 1),
                    ]
                    (directory / "command.json").write_text(json.dumps(command) + "\n")
                    with (
                        (directory / "stdout.json").open("wb") as stdout,
                        (directory / "stderr.txt").open("wb") as stderr,
                    ):
                        result = subprocess.run(
                            command, stdout=stdout, stderr=stderr, check=False
                        )
                    record = json.loads(metrics.read_bytes())
                    payload = json.loads((directory / "stdout.json").read_bytes())
                    row = {
                        "size": size,
                        "round": round_index,
                        "kind": kind,
                        "candidate": name,
                        "metrics": record,
                        "payload": payload,
                    }
                    rows.append(row)
                    (out / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
                    if (
                        result.returncode
                        or record["exit_code"]
                        or record["timed_out"]
                        or payload["output_sha256"] != expected.hexdigest()
                        or payload["output_bytes"] != source.stat().st_size + 1
                        or not payload["native_modules"]
                    ):
                        raise ValueError(f"buffer diagnostic failed: {directory.name}")
    print(
        json.dumps(
            {"attempts": len(rows), "exact_outputs": len(rows), "output": str(out)}
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    child = modes.add_parser("worker")
    child.add_argument("--input", required=True)
    child.add_argument("--kind", choices=("mutable", "immutable"), required=True)
    child.add_argument("--iterations", type=int, required=True)
    parent = modes.add_parser("run")
    parent.add_argument("--work", type=Path, required=True)
    parent.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    {"worker": worker, "run": run}[args.mode](args)


if __name__ == "__main__":
    main()
