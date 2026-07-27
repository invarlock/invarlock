"""Execute Garak's offline repeat generator and retain detector observations."""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from runner_support import arguments, finish_observation


def main() -> None:
    args = arguments()
    with tempfile.TemporaryDirectory(prefix="invarlock-garak-") as temporary:
        root = Path(temporary)
        prefix = root / "qualification"
        environment = os.environ.copy()
        environment["XDG_CACHE_HOME"] = str(root / "cache")
        environment["XDG_DATA_HOME"] = str(root / "data")
        command = [
            sys.executable,
            "-m",
            "garak",
            "-m",
            "test.Repeat",
            "-n",
            "offline-repeat",
            "-p",
            "dan.Dan_11_0",
            "-g",
            "1",
            "--report_prefix",
            str(prefix),
            "--narrow_output",
            "--seed",
            "1",
            "--confidence_interval_method",
            "none",
        ]
        subprocess.run(command, env=environment, check=True)
        entries = [
            json.loads(line)
            for line in (prefix.with_suffix(".report.jsonl")).read_text().splitlines()
        ]
    evaluations = [
        {
            "detector": entry["detector"],
            "fails": entry["fails"],
            "passed": entry["passed"],
            "probe": entry["probe"],
            "total_evaluated": entry["total_evaluated"],
        }
        for entry in entries
        if entry.get("entry_type") == "eval"
    ]
    finish_observation(
        args=args,
        entrypoint="python -m garak",
        summary_kind="unsupported_semantics",
        summary_data={
            "evaluations": sorted(
                evaluations,
                key=lambda value: (value["probe"], value["detector"]),
            )
        },
    )


if __name__ == "__main__":
    main()
