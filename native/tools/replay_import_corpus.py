"""Compare retained Python import outcomes with a standalone native binary."""

import argparse
import json
import os
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus", type=Path)
    parser.add_argument("binary", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Refusing to overwrite retained replay")
    failures = []
    counts = {
        "attempted": 0,
        "accepted_matched": 0,
        "rejected_matched": 0,
        "alternate_configuration": 0,
    }
    for index, line in enumerate(args.corpus.open()):
        item = json.loads(line)
        if any(
            value != {"MAX_RECORDS": 50_000, "MAX_INPUT_BYTES": 128 * 1024 * 1024}[key]
            for limits in item.get("reference_limits", {}).values()
            for key, value in limits.items()
        ):
            counts["alternate_configuration"] += 1
            continue
        process = subprocess.run(
            [str(args.binary), "import-evaluation-run", "--request", "-"],
            input=json.dumps(item["request"]),
            text=True,
            capture_output=True,
            env={**os.environ, "PATH": "/nonexistent"},
            timeout=30,
        )
        counts["attempted"] += 1
        try:
            actual = json.loads(process.stdout)
        except ValueError:
            actual = {"invalid_stdout": process.stdout, "stderr": process.stderr}
        expected = item["expected"]
        if (
            "ok" in expected
            and process.returncode == 0
            and json.dumps(actual, sort_keys=True, ensure_ascii=True)
            == json.dumps(expected["ok"], sort_keys=True, ensure_ascii=True)
        ):
            counts["accepted_matched"] += 1
        elif "error" in expected and process.returncode != 0 and "error" in actual:
            counts["rejected_matched"] += 1
        else:
            failures.append(
                {
                    "index": index,
                    "returncode": process.returncode,
                    "adapter": item["request"]["options"].get("adapter", "invarlock"),
                    "expected": expected,
                    "actual": actual,
                }
            )
    report = {**counts, "failed": len(failures), "failures": failures}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "failures"}))
    raise SystemExit(bool(failures))


if __name__ == "__main__":
    main()
