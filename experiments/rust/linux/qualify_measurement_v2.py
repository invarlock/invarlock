#!/usr/bin/env python3
"""Bounded real Linux controls; each attempt retained without timing comparisons."""

import argparse
import hashlib
import json
import os
import platform
import signal
import subprocess
import sys
import time
from pathlib import Path


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--supervisor", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frozen-work", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []

    def measure(name, command, timeout=5):
        directory = args.output / name
        directory.mkdir()
        with (
            (directory / "stdout.txt").open("wb") as out,
            (directory / "stderr.txt").open("wb") as err,
        ):
            run = subprocess.run(
                [
                    str(args.supervisor),
                    str(directory / "metrics.json"),
                    str(timeout),
                    "--",
                    *command,
                ],
                stdout=out,
                stderr=err,
                start_new_session=True,
            )
        assert run.returncode == 0, (name, run.returncode)
        row = json.loads((directory / "metrics.json").read_text())
        row.update(
            name=name,
            command=command,
            stdout_sha256=digest(directory / "stdout.txt"),
            stderr_sha256=digest(directory / "stderr.txt"),
        )
        rows.append(row)
        return row

    for mb in (0, 128):
        allocation = bytearray(mb * 1024 * 1024)
        for repeat in range(3):
            old_start = time.monotonic()
            child = subprocess.Popen(["/bin/true"], start_new_session=True)
            while True:
                pid, status, usage = os.wait4(child.pid, os.WNOHANG)
                if pid:
                    break
                time.sleep(0.01)
            child.returncode = os.waitstatus_to_exitcode(status)
            rows.append(
                {
                    "name": f"historical-true-{mb}-{repeat}",
                    "parent_allocation_mib": mb,
                    "max_rss_raw": usage.ru_maxrss,
                    "wall_seconds": time.monotonic() - old_start,
                }
            )
            row = measure(f"supervised-true-{mb}-{repeat}", ["/bin/true"])
            assert row["exit_code"] == 0 and not row["timed_out"]
        del allocation
    old_low = [
        r["max_rss_raw"] for r in rows if r["name"].startswith("historical-true-0-")
    ]
    old_high = [
        r["max_rss_raw"] for r in rows if r["name"].startswith("historical-true-128-")
    ]
    new_low = [
        r["max_rss_raw"] for r in rows if r["name"].startswith("supervised-true-0-")
    ]
    new_high = [
        r["max_rss_raw"] for r in rows if r["name"].startswith("supervised-true-128-")
    ]
    assert min(old_high) - max(old_low) > 100 * 1024
    assert max(new_high) - min(new_low) < 16 * 1024
    row = measure("nonzero", ["/bin/sh", "-c", "printf out; printf err >&2; exit 23"])
    assert row["exit_code"] == 23 and not row["timed_out"]
    assert (args.output / "nonzero/stdout.txt").read_bytes() == b"out"
    assert (args.output / "nonzero/stderr.txt").read_bytes() == b"err"
    assert (
        measure("signal", ["/bin/sh", "-c", "kill -TERM $$"])["exit_code"]
        == -signal.SIGTERM
    )
    assert measure("exec-error", ["/no/such/command"])["exec_errno"] == 2
    pidfile = args.output / "descendant.pid"
    code = (
        "import subprocess,time,pathlib; p=subprocess.Popen(['/bin/sleep','60']); pathlib.Path("
        + repr(str(pidfile))
        + ").write_text(str(p.pid)); time.sleep(60)"
    )
    row = measure("timeout-descendants", [sys.executable, "-c", code], 0.2)
    assert row["timed_out"] and row["exit_code"] == -signal.SIGKILL
    assert not Path("/proc/" + pidfile.read_text()).exists()
    row = measure(
        "larger-memory",
        [sys.executable, "-c", "x=bytearray(64*1024*1024); print(len(x))"],
    )
    assert row["max_rss_raw"] > 60 * 1024
    row = measure(
        "large-output",
        [
            sys.executable,
            "-c",
            "import sys;sys.stdout.buffer.write(b'x'*(8*1024*1024));sys.stderr.buffer.write(b'y'*(2*1024*1024))",
        ],
    )
    assert row["exit_code"] == 0
    assert (args.output / "large-output/stdout.txt").stat().st_size == 8 * 1024 * 1024
    if args.frozen_work:
        work = args.frozen_work
        metadata = json.loads((work / "metadata.json").read_text())
        sys.path.insert(0, str(work / "inputs/harness"))
        from invarlock_rust_experiment.full_runner import materialize

        fixture = json.loads((work / "fixtures.json").read_text())["workloads"][0]
        for mb in (0, 128):
            allocation = bytearray(mb * 1024 * 1024)
            output = args.output / f"frozen-artifact-{mb}-files"
            output.mkdir()
            request = args.output / f"frozen-artifact-{mb}-request.json"
            request.write_text(json.dumps(materialize(fixture["request"], output)))
            artifact = metadata["standalone"]
            assert digest(artifact["path"]) == artifact["sha256"]
            row = measure(f"frozen-artifact-{mb}", [artifact["path"], str(request)])
            row["artifact"] = artifact
            assert row["exit_code"] == 0
            del allocation
    result = {
        "format": "invarlock/measurement-controls-v2",
        "complete": True,
        "supervisor": {"path": str(args.supervisor), "sha256": digest(args.supervisor)},
        "source": {"path": str(args.source), "sha256": digest(args.source)},
        "platform": platform.platform(),
        "python": sys.version,
        "rows": rows,
        "scope": "Boundary controls only; no candidate performance comparison. RSS is unadjusted wait4 of the direct child; it is not aggregate concurrent descendant RSS.",
    }
    (args.output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
