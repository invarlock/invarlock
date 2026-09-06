"""Execute a current scalar profile and independently qualify a fresh output."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

import matrix  # noqa: E402
from maintained.scalar_semantics import CONFIGURATIONS  # noqa: E402

from invarlock.filesystem import publish_directory_no_replace  # noqa: E402


def execute(
    *, provider: str, cases: Path, schedule: Path, output: Path
) -> dict[str, Any]:
    if output.exists() or output.is_symlink():
        raise ValueError("fresh scalar qualification output already exists")
    definition = next(
        item
        for item in matrix.load(ROOT / "scalar-profiles.json")["profiles"]
        if item["historical_profile"] == provider
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT.parent)
    environment["UV_PYTHON"] = sys.executable
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".scalar-qualification-", dir=output.parent
    ) as temporary:
        artifacts = Path(temporary)
        profile_path = matrix.write_profile(definition, artifacts=artifacts)
        artifact = profile_path.parent
        captured_cases = artifact / "cases.json"
        captured_schedule = artifact / "schedule.json"
        shutil.copyfile(cases, captured_cases)
        shutil.copyfile(schedule, captured_schedule)
        subprocess.run(
            matrix.runner_command(
                definition,
                profile_path,
                cases=captured_cases,
                schedule=captured_schedule,
            ),
            check=True,
            env=environment,
        )
        result = matrix.qualify(
            definition, artifacts=artifacts, schedule=captured_schedule
        ).as_dict()
        (artifact / "qualification-result.json").write_bytes(
            matrix.canonical_json_bytes(result)
        )
        publish_directory_no_replace(artifact, output)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=CONFIGURATIONS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--cases", type=Path, default=ROOT.parent / "authoritative/cases.json"
    )
    parser.add_argument(
        "--schedule", type=Path, default=ROOT.parent / "authoritative/schedule.json"
    )
    args = parser.parse_args(argv)
    try:
        result = execute(
            provider=args.provider,
            cases=args.cases,
            schedule=args.schedule,
            output=args.output,
        )
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        print(f"Scalar qualification failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
