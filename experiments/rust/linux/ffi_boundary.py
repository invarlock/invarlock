"""Separate Linux FFI diagnostic: canonical JSON, immutable and mutable buffers.

Identity checks are outside measured workers. Each fresh child includes imports,
input read, repeated canonicalization and one output hash. The inner operation
clock excludes imports, input loading and the final hash. No retry or exclusions.
This diagnostic is not a substitute for complete evaluation/verification timing.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(os.environ.get("INVARLOCK_COMPARISON_ROOT", "/work/standalone-v4"))
PYTHON = Path("/work/optimization-v5-candidate/venv/bin/python")
SUPERVISOR = Path("/work/optimization-v5-candidate/measure_v2")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def worker(kind, path, iterations):
    if kind != "python":
        import invarlock_native
    raw = Path(path).read_bytes()
    if kind == "mutable":
        raw = bytearray(raw)
    started = time.perf_counter()
    for _ in range(int(iterations)):
        if kind == "python":
            result = (
                json.dumps(
                    json.loads(raw),
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            ).encode()
        else:
            result = invarlock_native.canonicalize_json(raw)
    elapsed = time.perf_counter() - started
    print(
        json.dumps(
            {
                "output_sha256": hashlib.sha256(result).hexdigest(),
                "output_bytes": len(result),
                "operation_seconds": elapsed,
                "iterations": int(iterations),
            }
        )
    )


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "worker":
        worker(*sys.argv[2:])
        return
    if sys.platform != "linux":
        raise RuntimeError("Linux supervisor required")
    output = ROOT / "ffi-boundary"
    output.mkdir()
    identity = json.loads((ROOT / "identity.json").read_bytes())
    if digest(SUPERVISOR) != identity["supervisor_sha256"]:
        raise RuntimeError("measurement supervisor changed")
    modules = json.loads(
        subprocess.check_output(
            [
                str(PYTHON),
                "-c",
                'import invarlock_native,pathlib,hashlib,json;print(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in pathlib.Path(invarlock_native.__file__).parent.glob("*.so")}))',
            ]
        )
    )
    if modules != identity["native_extension_files"]:
        raise RuntimeError("native extension changed")
    plan = {
        "format": "invarlock/ffi-boundary-v2",
        "sizes": [64, 1024**2, 16 * 1024**2, 64 * 1024**2],
        "rounds": 3,
        "candidates": ["python", "immutable", "mutable"],
        "tiny_iterations": 1000,
        "other_iterations": 1,
        "script_sha256": digest(__file__),
        "native_extension_files": modules,
        "scope": __doc__,
        "retries": 0,
        "exclusions": 0,
    }
    (output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    records = []
    randomizer = random.Random(713)
    for size in plan["sizes"]:
        raw = b'{"blob":"' + b"x" * size + b'"}'
        path = output / f"input-{size}.json"
        path.write_bytes(raw)
        expected = hashlib.sha256(raw + b"\n").hexdigest()
        del raw
        for round_index in range(plan["rounds"]):
            candidates = plan["candidates"].copy()
            randomizer.shuffle(candidates)
            for kind in candidates:
                attempt = output / f"{size}-{round_index}-{kind}"
                attempt.mkdir()
                command = [
                    str(SUPERVISOR),
                    str(attempt / "metrics.json"),
                    "120",
                    "--",
                    str(PYTHON),
                    "-I",
                    str(Path(__file__).resolve()),
                    "worker",
                    kind,
                    str(path),
                    str(1000 if size == 64 else 1),
                ]
                (attempt / "command.json").write_text(json.dumps(command) + "\n")
                with (
                    (attempt / "stdout").open("wb") as stdout,
                    (attempt / "stderr").open("wb") as stderr,
                ):
                    run = subprocess.run(command, stdout=stdout, stderr=stderr)
                metrics = json.loads((attempt / "metrics.json").read_bytes())
                if (
                    run.returncode
                    or metrics["timed_out"]
                    or metrics["exec_errno"]
                    or metrics["exit_code"]
                ):
                    raise RuntimeError(f"failed boundary attempt {attempt}")
                payload = json.loads((attempt / "stdout").read_bytes())
                record = {
                    "size": size,
                    "round": round_index,
                    "candidate": kind,
                    "metrics": metrics,
                    "payload": payload,
                }
                records.append(record)
                (output / "records.json").write_text(
                    json.dumps(records, indent=2) + "\n"
                )
                if (
                    payload["output_sha256"] != expected
                    or payload["output_bytes"] != path.stat().st_size + 1
                ):
                    raise RuntimeError(f"boundary parity mismatch {attempt}")
                print(
                    json.dumps(
                        {
                            "size": size,
                            "round": round_index,
                            "candidate": kind,
                            "matched": True,
                        }
                    ),
                    flush=True,
                )
    rows = []
    for size in plan["sizes"]:
        row = {"size": size}
        for candidate in plan["candidates"]:
            selected = [
                r for r in records if r["size"] == size and r["candidate"] == candidate
            ]
            row[candidate] = {
                "elapsed_seconds": statistics.median(
                    r["metrics"]["wall_seconds"] for r in selected
                ),
                "cpu_seconds": statistics.median(
                    r["metrics"]["user_cpu_seconds"]
                    + r["metrics"]["system_cpu_seconds"]
                    for r in selected
                ),
                "peak_rss_mib": statistics.median(
                    r["metrics"]["max_rss_bytes"] / 1024**2 for r in selected
                ),
                "operation_seconds": statistics.median(
                    r["payload"]["operation_seconds"] for r in selected
                ),
                "samples": len(selected),
            }
        rows.append(row)
    (output / "summary.json").write_text(
        json.dumps(
            {"plan": plan, "all_matched": True, "outcomes": len(records), "rows": rows},
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
