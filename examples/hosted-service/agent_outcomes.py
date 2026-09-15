"""Synthetic outcome-import fixture; no model or general agent runner is used.

Only the fixed sources in this module are executed. Imported captures are
validated and projected without executing their supplied files. Capture
validation authenticates observations, not the claimed execution itself.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from invarlock.engine import capture_evaluator_run, digest

FORMAT = "invarlock/agent-outcome-fixture-v1"
TASK = "Fix count_through(n) to count integers from zero through n, inclusive."
FINAL_MESSAGE = "Implemented the boundary fix and verified the result."
STARTING_FILES = {
    "solution.py": "def count_through(n):\n    return len(range(n))\n",
}
CANDIDATES = {
    "correct": {"solution.py": "def count_through(n):\n    return len(range(n + 1))\n"},
    "incorrect": {
        "solution.py": "def count_through(n):\n    return len(range(n)) + 2\n"
    },
}
CHECK_SOURCE = """import runpy
import unittest
from pathlib import Path

count_through = runpy.run_path(str(Path.cwd() / "solution.py"))["count_through"]

class BoundaryTests(unittest.TestCase):
    def test_inclusive_boundary(self):
        self.assertEqual(count_through(0), 1)
        self.assertEqual(count_through(3), 4)

unittest.main()
"""
MAX_OUTPUT_BYTES = 16 * 1024
TIMEOUT_SECONDS = 5


def harness_digest():
    return "sha256:" + hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _capture_variant(variant):
    _require(variant in CANDIDATES, "unknown fixed candidate")
    files = CANDIDATES[variant]
    command = [sys.executable, "-I", "check.py"]
    with tempfile.TemporaryDirectory(prefix="invarlock-outcome-") as directory:
        root = Path(directory)
        for name, contents in STARTING_FILES.items():
            (root / name).write_text(contents, encoding="utf-8")
        starting_files = {
            name: (root / name).read_text(encoding="utf-8") for name in STARTING_FILES
        }
        _require(
            digest(starting_files) == digest(STARTING_FILES), "starting files differ"
        )
        for name, contents in files.items():
            (root / name).write_text(contents, encoding="utf-8")
        (root / "check.py").write_text(CHECK_SOURCE, encoding="utf-8")
        with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
            timed_out = False
            try:
                process = subprocess.run(
                    command,
                    cwd=root,
                    stdin=subprocess.DEVNULL,
                    stdout=stdout,
                    stderr=stderr,
                    timeout=TIMEOUT_SECONDS,
                    check=False,
                    env={},
                )
                returncode = process.returncode
            except subprocess.TimeoutExpired:
                timed_out = True
                returncode = None
            output = {}
            for name, stream in (("stdout", stdout), ("stderr", stderr)):
                stream.seek(0)
                raw = stream.read(MAX_OUTPUT_BYTES + 1)
                _require(
                    len(raw) <= MAX_OUTPUT_BYTES, "fixed check output exceeds bound"
                )
                output[name] = raw.decode("utf-8", errors="strict")
        _require(
            all(
                (root / name).read_text(encoding="utf-8") == code
                for name, code in files.items()
            )
            and (root / "check.py").read_text(encoding="utf-8") == CHECK_SOURCE,
            "fixed check changed retained files",
        )
    return {
        "variant": variant,
        "candidate_files": dict(files),
        "candidate_digest": digest(files),
        "final_message": FINAL_MESSAGE,
        "command": command,
        "timeout_seconds": TIMEOUT_SECONDS,
        "output_byte_limit": MAX_OUTPUT_BYTES,
        **output,
        "returncode": returncode,
        "timed_out": timed_out,
        "outcome": "pass" if returncode == 0 and not timed_out else "fail",
    }


def capture_fixture():
    """Execute the two fixed candidate files in separate temporary directories."""
    return {
        "format": FORMAT,
        "qualification": "synthetic_integration_fixture",
        "task": TASK,
        "harness_digest": harness_digest(),
        "starting_files": dict(STARTING_FILES),
        "starting_digest": digest(STARTING_FILES),
        "test_files": {"check.py": CHECK_SOURCE},
        "test_digest": digest({"check.py": CHECK_SOURCE}),
        "baseline": _capture_variant("correct"),
        "subject": _capture_variant("incorrect"),
    }


def project_capture(capture, *, expected_capture_digest, expected_harness_digest):
    """Project independently pinned observations; never execute imported code."""
    _require(digest(capture) == expected_capture_digest, "capture digest differs")
    _require(
        capture.get("harness_digest") == expected_harness_digest == harness_digest(),
        "harness digest differs",
    )
    _require(
        set(capture)
        == {
            "format",
            "qualification",
            "task",
            "harness_digest",
            "starting_files",
            "starting_digest",
            "test_files",
            "test_digest",
            "baseline",
            "subject",
        }
        and capture["format"] == FORMAT
        and capture["qualification"] == "synthetic_integration_fixture"
        and capture["task"] == TASK,
        "invalid fixture identity",
    )
    _require(
        capture["starting_files"] == STARTING_FILES
        and capture["starting_digest"] == digest(STARTING_FILES)
        and capture["test_files"] == {"check.py": CHECK_SOURCE}
        and capture["test_digest"] == digest({"check.py": CHECK_SOURCE}),
        "starting or test binding differs",
    )
    runs = {}
    for side, variant in (("baseline", "correct"), ("subject", "incorrect")):
        observation = capture[side]
        _require(
            set(observation)
            == {
                "variant",
                "candidate_files",
                "candidate_digest",
                "final_message",
                "command",
                "timeout_seconds",
                "output_byte_limit",
                "stdout",
                "stderr",
                "returncode",
                "timed_out",
                "outcome",
            }
            and observation["variant"] == variant
            and observation["candidate_files"] == CANDIDATES[variant]
            and observation["candidate_digest"] == digest(CANDIDATES[variant]),
            "candidate binding differs",
        )
        command = observation["command"]
        _require(
            isinstance(command, list)
            and len(command) == 3
            and isinstance(command[0], str)
            and bool(command[0])
            and command[1:] == ["-I", "check.py"]
            and observation["timeout_seconds"] == TIMEOUT_SECONDS
            and observation["output_byte_limit"] == MAX_OUTPUT_BYTES
            and observation["final_message"] == FINAL_MESSAGE
            and all(
                isinstance(observation[key], str)
                and len(observation[key].encode("utf-8")) <= MAX_OUTPUT_BYTES
                for key in ("stdout", "stderr")
            ),
            "invalid fixed check observation",
        )
        code, timed_out = observation["returncode"], observation["timed_out"]
        _require(
            type(timed_out) is bool
            and ((timed_out and code is None) or (not timed_out and type(code) is int))
            and observation["outcome"]
            == ("pass" if code == 0 and not timed_out else "fail"),
            "outcome disagrees with check exit",
        )
        runs[side] = capture_evaluator_run(
            [
                {
                    "id": "inclusive-boundary",
                    "input": TASK,
                    "expected": "pass",
                    "output": observation["outcome"],
                    "context": {
                        "qualification": "synthetic_integration_fixture",
                        "final_message": observation["final_message"],
                        "outcome_evidence": observation,
                        "starting_digest": capture["starting_digest"],
                        "test_digest": capture["test_digest"],
                    },
                }
            ],
            source={"name": "fixed-agent-outcome-fixture", "version": "1"},
            run_id=side,
            artifact_digest=observation["candidate_digest"],
            source_digest=expected_capture_digest,
        )
    return runs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=False)
    capture = capture_fixture()
    capture_pin = digest(capture)
    runs = project_capture(
        capture,
        expected_capture_digest=capture_pin,
        expected_harness_digest=harness_digest(),
    )
    for name, value in (
        ("capture.json", capture),
        ("baseline.json", runs["baseline"]),
        ("subject.json", runs["subject"]),
    ):
        (args.output / name).write_text(
            json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n",
            encoding="utf-8",
        )
    print(
        json.dumps(
            {
                "capture_digest": capture_pin,
                "baseline": capture["baseline"]["outcome"],
                "subject": capture["subject"]["outcome"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
