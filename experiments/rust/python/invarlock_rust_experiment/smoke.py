"""Seven-record installed-wheel qualification, with no timing conclusions."""

from __future__ import annotations

import argparse
from pathlib import Path

from .fixtures import write_json
from .runner import CANDIDATES, command, run_process, verify_metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    metadata = verify_metadata(args.metadata)
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / "metadata.json", metadata)
    source = Path(__file__).resolve().parents[1]
    records = []
    for scenario in ("pass", "regression"):
        fixture = root / f"fixture7-{scenario}"
        built = run_process(
            command(
                args.python.absolute(),
                source,
                "fixture",
                fixture,
                root / f"fixture-artifacts-{scenario}",
                count=7,
                metadata=args.metadata.resolve(),
                scenario=scenario,
            ),
            root / f"fixture-process-{scenario}",
            30,
        )
        if built["status"] != "valid":
            write_json(
                root / "qualification.json",
                {"ok": False, "scenario": scenario, "fixture": built},
            )
            return 1
        stages = (
            ("kernel", "verify", "reject", "zero")
            if scenario == "pass"
            else ("kernel", "verify")
        )
        for stage in stages:
            expected = None
            for candidate in CANDIDATES:
                sample_root = root / f"{scenario}-{candidate}-{stage}"
                record = run_process(
                    command(
                        args.python.absolute(),
                        source,
                        stage,
                        fixture,
                        sample_root / "artifacts",
                        candidate,
                        metadata=args.metadata.resolve(),
                    ),
                    sample_root,
                    30,
                )
                record.update(candidate=candidate, stage=stage, scenario=scenario)
                if record["status"] == "valid":
                    digest = record["payload"]["result_sha256"]
                    if expected is None:
                        expected = digest
                    elif digest != expected:
                        record["status"] = "wrong_output"
                record["output_comparison"] = (
                    "matched" if record["status"] == "valid" else "failed"
                )
                write_json(sample_root / "assessment.json", record)
                records.append(record)
                if record["status"] != "valid":
                    write_json(
                        root / "qualification.json", {"ok": False, "samples": records}
                    )
                    return 1
    write_json(
        root / "qualification.json",
        {
            "ok": True,
            "record_count": 7,
            "claim": "installed signed pass and regression replay, untrusted-signer rejection and zero-baseline rejection on these fixtures only",
            "timing_conclusion": None,
            "samples": records,
        },
    )
    print(f"Seven-record signed path passed for all candidates; retained at {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
