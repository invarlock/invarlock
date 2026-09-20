"""Run an admitted local command under an external process-group deadline."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import time
from pathlib import Path

import common


def run(command, seconds, output):
    if not command or type(seconds) is not int or not 1 <= seconds <= 86400:
        raise ValueError("declare a command and a 1 to 86400 second wall-clock cap")
    output = Path(output)
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    common.write(
        output / "admission.json", {"command": command, "max_seconds": seconds}
    )
    started = time.monotonic()
    timed_out = False
    with (output / "output.log").open("xb") as log:
        process = subprocess.Popen(
            command, stdout=log, stderr=log, start_new_session=True
        )
        try:
            try:
                code = process.wait(timeout=max(0.01, seconds - min(5, seconds / 2)))
            except subprocess.TimeoutExpired:
                timed_out = True
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(
                        timeout=max(0.01, seconds - (time.monotonic() - started))
                    )
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                code = 124
        finally:
            # A command exiting does not authorize leaving its descendants running.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
    result = {
        "exit_code": code,
        "timed_out": timed_out,
        "elapsed_seconds": time.monotonic() - started,
    }
    common.write(output / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    raise SystemExit(run(command, args.seconds, args.output)["exit_code"])


if __name__ == "__main__":
    main()
