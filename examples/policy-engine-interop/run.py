#!/usr/bin/env python3
"""Run the maintained OPA and CUE policy-engine conformance matrix."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FIXTURES = ROOT / "fixtures"


def version(executable: str) -> str:
    completed = subprocess.run(
        [executable, "version"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout + completed.stderr


def run_opa(executable: str, fixture: Path) -> dict[str, object]:
    completed = subprocess.run(
        [
            executable,
            "eval",
            "--format",
            "raw",
            "--data",
            str(ROOT / "policy" / "acceptance.rego"),
            "--input",
            str(fixture),
            "data.invarlock.acceptance.decision",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    value = json.loads(completed.stdout)
    if not isinstance(value, dict):
        raise ValueError("OPA decision must be an object")
    return value


def cue_accepts(executable: str, fixture: Path) -> bool:
    completed = subprocess.run(
        [
            executable,
            "vet",
            str(ROOT / "policy" / "acceptance.cue"),
            str(fixture),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.returncode == 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--opa", default="opa")
    parser.add_argument("--cue", default="cue")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    expected_versions = json.loads((ROOT / "tool-versions.json").read_bytes())
    observed_opa = version(args.opa)
    observed_cue = version(args.cue)
    if expected_versions["opa"].removeprefix("v") not in observed_opa:
        raise RuntimeError(f"expected OPA {expected_versions['opa']}")
    if expected_versions["cue"].removeprefix("v") not in observed_cue:
        raise RuntimeError(f"expected CUE {expected_versions['cue']}")

    expectations = json.loads((FIXTURES / "expectations.json").read_bytes())
    for name, expectation in expectations["scenarios"].items():
        fixture = FIXTURES / f"{name}.json"
        opa = run_opa(args.opa, fixture)
        cue = cue_accepts(args.cue, fixture)
        expected = expectation["allow"]
        if opa["allow"] is not expected:
            raise AssertionError(f"{name}: OPA returned {opa}")
        if opa["reasons"] != expectation["reasons"]:
            raise AssertionError(f"{name}: OPA returned {opa}")
        if cue is not expected:
            raise AssertionError(f"{name}: CUE acceptance was {cue}")
        print(f"{name}: opa={str(opa['allow']).lower()} cue={str(cue).lower()}")


if __name__ == "__main__":
    main()
