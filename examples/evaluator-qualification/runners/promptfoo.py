"""Execute the pinned Promptfoo CLI and retain a normalized upstream result."""

import json
import os
import subprocess
import tempfile
from pathlib import Path

from runner_support import arguments, finish_deterministic, load_inputs


def main() -> None:
    args = arguments()
    profile, _, cases = load_inputs(args)
    version = profile["upstream"]["package"]["version"]
    lock = dict(
        line.split("=", 1)
        for line in args.dependency_lock.read_text(encoding="utf-8").splitlines()
    )
    expected_spec = f"promptfoo@{version}"
    if lock.get("package") != expected_spec:
        raise ValueError("Promptfoo dependency declaration does not match the profile")
    metadata = json.loads(
        subprocess.run(
            [
                "npm",
                "view",
                expected_spec,
                "dist.integrity",
                "dist.shasum",
                "--json",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    if metadata.get("dist.integrity") != lock.get("integrity") or metadata.get(
        "dist.shasum"
    ) != lock.get("shasum"):
        raise ValueError("Promptfoo registry integrity does not match the declaration")
    environment = os.environ.copy()
    environment["PROMPTFOO_DISABLE_TELEMETRY"] = "1"
    with tempfile.TemporaryDirectory(prefix="invarlock-promptfoo-") as temporary:
        config = Path(temporary) / "promptfoo.json"
        config.write_text(
            json.dumps(
                {
                    "description": "InvarLock evaluator qualification",
                    "prompts": ["{{output}}"],
                    "providers": ["echo"],
                    "tests": [
                        {
                            "assert": [
                                {
                                    "type": "equals",
                                    "value": case["reference"],
                                }
                            ],
                            "description": case["record_id"],
                            "vars": {"output": case["output"]},
                        }
                        for case in cases
                    ],
                },
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        native = Path(temporary) / "result.json"
        command = [
            "npx",
            "--yes",
            f"promptfoo@{version}",
            "eval",
            "--config",
            str(config),
            "--output",
            str(native),
            "--no-cache",
            "--no-progress-bar",
            "--no-share",
            "--no-table",
        ]
        completed = subprocess.run(command, env=environment, check=False)
        if completed.returncode not in (0, 100):
            raise RuntimeError(f"Promptfoo exited with {completed.returncode}")
        document = json.loads(native.read_bytes())
    native_results = document["results"]["results"]
    by_id = {result["testCase"]["description"]: result for result in native_results}
    scores: list[float] = []
    details = []
    for case in cases:
        result = by_id[case["record_id"]]
        output = result["response"]["output"]
        if output != case["output"]:
            raise ValueError(f"Promptfoo output mismatch for {case['record_id']}")
        score = float(result["score"])
        scores.append(score)
        details.append(
            {
                "output": output,
                "score": score,
                "success": bool(result["success"]),
            }
        )
    finish_deterministic(
        args=args,
        entrypoint="promptfoo eval",
        scores=scores,
        details=details,
        environment=[
            {
                "integrity": metadata["dist.integrity"],
                "name": "promptfoo",
                "shasum": metadata["dist.shasum"],
                "version": version,
            }
        ],
    )


if __name__ == "__main__":
    main()
