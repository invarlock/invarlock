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
    config = Path(__file__).resolve().parent.parent / "promptfoo.yaml"
    environment = os.environ.copy()
    environment["PROMPTFOO_DISABLE_TELEMETRY"] = "1"
    with tempfile.TemporaryDirectory(prefix="invarlock-promptfoo-") as temporary:
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
        environment=[{"name": "promptfoo", "version": version}],
    )


if __name__ == "__main__":
    main()
