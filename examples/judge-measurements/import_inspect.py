"""Import a supported expanded Inspect judge export without provider calls."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


def _object(path: Path, *, maximum: int) -> dict[str, Any]:
    from invarlock.evidence_pack_json import parse_json_bytes, read_regular_file_bytes

    value = parse_json_bytes(
        read_regular_file_bytes(path, label=str(path), max_bytes=maximum),
        label=str(path),
    )
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--plan", default="plan.json")
    parser.add_argument("--collection", default="collection.json")
    parser.add_argument("--baseline-run", default="baseline_run.json")
    parser.add_argument("--subject-run", default="subject_run.json")
    parser.add_argument("--export", required=True, help="supported expanded-event JSON")
    parser.add_argument(
        "--output", required=True, help="new measurements file within root"
    )
    args = parser.parse_args()
    try:
        from invarlock.evaluation_record_contracts.contracts import MAX_INPUT_BYTES
        from invarlock.evidence_pack_json import read_regular_file_bytes
        from invarlock.filesystem.atomic_file import write_file_no_replace
        from invarlock.filesystem.paths import pinned_directory
        from invarlock.judge_measurements.contracts import (
            MEASUREMENTS_MAX_BYTES,
            canonical_payload,
            load_measurement_plan,
        )

        root = args.root.resolve(strict=True)
        if not root.is_dir():
            raise ValueError("import root must be an existing directory")
        output_argument = Path(args.output)
        if output_argument.is_absolute() or ".." in output_argument.parts:
            raise ValueError(
                "measurement output must be a relative path within the root"
            )
        output = root / output_argument
        if output.exists() or output.is_symlink():
            raise ValueError("measurement output must be a new file")
        with pinned_directory(output.parent):
            pass
        from invarlock_addins.inspect_judge import CollectionOptions, import_export

        measurements = import_export(
            read_regular_file_bytes(
                root / args.export,
                label="Inspect judge export",
                max_bytes=MEASUREMENTS_MAX_BYTES,
            ),
            plan=load_measurement_plan(root / args.plan),
            options=CollectionOptions.from_mapping(
                _object(root / args.collection, maximum=1024 * 1024)
            ),
            baseline_run=_object(root / args.baseline_run, maximum=MAX_INPUT_BYTES),
            subject_run=_object(root / args.subject_run, maximum=MAX_INPUT_BYTES),
        )
        write_file_no_replace(
            output, canonical_payload(measurements), create_parents=False
        )
    except ImportError:
        parser.exit(
            2, "Judge import requires matching core and inspect_judge packages.\n"
        )
    except (OSError, ValueError) as exc:
        parser.exit(2, f"Judge import failed: {exc}\n")
    completed = measurements["completeness"]["completed_trials"]
    expected = measurements["completeness"]["expected_trials"]
    print(f"Imported {completed}/{expected} completed trials into {output}")
    if completed != expected:
        print(
            "Evidence is incomplete; missing or failed trials remain recorded as such."
        )


if __name__ == "__main__":
    main()
