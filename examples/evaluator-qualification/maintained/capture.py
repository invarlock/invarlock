"""Capture evaluator case facts without transferring upstream scoring authority."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

from invarlock.evaluator_capture import (
    capture_evaluator_run,
    evaluator_input_capabilities,
)
from invarlock.evaluator_qualification import qualify_evaluator_export
from invarlock.evidence_pack_contract import canonical_json_bytes
from invarlock.evidence_pack_json import parse_json_bytes, read_regular_file_bytes
from invarlock.filesystem.atomic_file import write_file_no_replace

ROOT = Path(__file__).resolve().parents[1]


def digest(body: bytes) -> str:
    return "sha256:" + hashlib.sha256(body).hexdigest()


def load(path: Path) -> tuple[Any, bytes]:
    body = read_regular_file_bytes(
        path, label="capture input", max_bytes=64 * 1024 * 1024
    )
    return parse_json_bytes(body, label="capture input"), body


def profiles() -> dict[str, dict[str, Any]]:
    return {
        item["profile_id"]: item for item in load(ROOT / "matrix.json")[0]["profiles"]
    }


def capture_qualification(
    *,
    ecosystem: str,
    cases_path: Path,
    schedule_path: Path,
    profile_path: Path,
    export_path: Path,
    raw_output_path: Path,
    run_id: str,
    artifact_digest: str,
) -> dict[str, Any]:
    """Join supplied original text to independently bound per-case exports.

    Historical qualifications retain their original profile identity. This creates
    a new capture only; it neither reruns upstream nor extends qualification to a
    different scorer. Inputs are frozen before checking and joining their bytes.
    """
    definition = profiles()[ecosystem]
    sources = {
        "cases": cases_path,
        "schedule": schedule_path,
        "profile": profile_path,
        "export": export_path,
        "raw": raw_output_path,
    }
    documents, bodies = {}, {}
    for name, path in sources.items():
        documents[name], bodies[name] = load(path)
        if not isinstance(documents[name], dict):
            raise ValueError("qualification capture inputs must be JSON objects")
    with tempfile.TemporaryDirectory(prefix="evaluator-capture-") as temporary:
        frozen = {}
        for name, body in bodies.items():
            frozen[name] = Path(temporary) / name
            frozen[name].write_bytes(body)
        result = qualify_evaluator_export(
            profile_path=frozen["profile"],
            schedule_path=frozen["schedule"],
            export_path=frozen["export"],
            raw_output_path=frozen["raw"],
        )
    if result.authority != "verdict_authority":
        raise ValueError(
            "aggregate or detector summaries cannot manufacture per-case capture rows"
        )
    profile, raw = (documents[key] for key in ("profile", "raw"))
    package = profile["upstream"]["package"]
    if package != definition["upstream"]:
        raise ValueError("source package differs from the selected ecosystem")
    if raw.get("upstream") != package or raw.get("profile_id") != profile["profile_id"]:
        raise ValueError("raw source identity differs from its profile")
    cases, scheduled, rows = (
        documents[key].get("records") for key in ("cases", "schedule", "export")
    )
    native = raw.get("records")
    if (
        not isinstance(cases, list)
        or not cases
        or not isinstance(native, list)
        or any(not isinstance(row, dict) for row in native)
    ):
        raise ValueError("original cases and raw per-case records are required")
    if len(cases) != len(scheduled) or len(native) != len(rows):
        raise ValueError("all sources must contain the same ordered cases")
    records, seen = [], set()
    for case, scheduled_row, row, native_row in zip(
        cases, scheduled, rows, native, strict=True
    ):
        if not isinstance(case, dict) or any(
            not isinstance(case.get(key), str)
            for key in ("record_id", "input", "reference", "output")
        ):
            raise ValueError("original case fields must be strings")
        identifier = case["record_id"]
        if not identifier or identifier in seen:
            raise ValueError("case IDs must be nonempty and unique")
        seen.add(identifier)
        if any(
            value.get("record_id") != identifier
            for value in (scheduled_row, row, native_row)
        ):
            raise ValueError("all case identities and order must agree")
        input_digest = digest(case["input"].encode())
        if (
            input_digest != scheduled_row["input_sha256"]
            or case.get("input_sha256", input_digest) != input_digest
        ):
            raise ValueError("original input text differs from its schedule digest")
        if (
            digest(case["reference"].encode())
            != scheduled_row["reference_output_sha256"]
        ):
            raise ValueError("original reference text differs from its schedule digest")
        if case["output"] != row["output_text"]:
            raise ValueError("original output text differs from its exported output")
        if (
            type(native_row.get("score")) not in (int, float)
            or native_row["score"] != row["reported_score"]
        ):
            raise ValueError("raw and exported scores disagree")
        records.append(
            {
                "id": identifier,
                "input": case["input"],
                "expected": case["reference"],
                "output": case["output"],
                "context": {
                    "evaluator_capture": {
                        "capture_kind": "qualification-case-capture",
                        "ecosystem": ecosystem,
                        "original_profile_id": profile["profile_id"],
                        **{
                            name + "_sha256": digest(body)
                            for name, body in bodies.items()
                        },
                    }
                },
            }
        )
    return capture_evaluator_run(
        records,
        source={"name": package["name"], "version": package["version"]},
        run_id=run_id,
        artifact_digest=artifact_digest,
        source_digest=digest(bodies["export"]),
    )


def capture_records(
    *,
    ecosystem: str,
    records_path: Path,
    run_id: str,
    artifact_digest: str,
    source_version: str,
    input_projection: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Capture explicit canonical per-case records from any shortlisted workflow.

    The records file must be a JSON array of SDK records, with original id/input/
    output and optional expected/context/likelihood. MLflow callers retain their
    original prediction table; Garak callers review actual per-attempt records and
    preserve attempt/output identities. A detector report is not a records file.
    """
    definition = profiles()[ecosystem]
    records, body = load(records_path)
    if (
        not isinstance(records, list)
        or not records
        or any(not isinstance(row, dict) for row in records)
    ):
        raise ValueError(
            "capture requires a nonempty array of explicit per-case records"
        )
    return capture_evaluator_run(
        records,
        source={"name": definition["upstream"]["name"], "version": source_version},
        run_id=run_id,
        artifact_digest=artifact_digest,
        source_digest=digest(body),
        input_projection=input_projection,
    )


def integration_matrix() -> list[dict[str, str]]:
    """Describe retained facts separately from available capture interfaces."""
    return [
        {
            "ecosystem": name,
            "source_package": item["upstream"]["name"],
            "retained": "per-case export and original cases"
            if item["authority"]["mode"] == "deterministic_per_record"
            else "observation summary only",
            "capture": "qualification or canonical records"
            if item["authority"]["mode"] == "deterministic_per_record"
            else "explicit canonical per-case records required",
            "exact_match": "requires original output and string reference",
            "judge": "requires original input/output, configured rubric and judge execution",
            "normalized_nll_per_utf8_byte": "requires typed likelihood facts; retained exports have none",
            "scoring_owner": "InvarLock",
        }
        for name, item in profiles().items()
    ]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("matrix")
    for command in ("qualification", "records"):
        child = commands.add_parser(command)
        child.add_argument("--ecosystem", required=True, choices=sorted(profiles()))
        child.add_argument("--run-id", required=True)
        child.add_argument("--artifact-digest", required=True)
        child.add_argument("--output", type=Path, required=True)
        if command == "qualification":
            for name in ("cases", "schedule", "profile", "export", "raw-output"):
                child.add_argument("--" + name, type=Path, required=True)
        else:
            child.add_argument("--records", type=Path, required=True)
            child.add_argument("--source-version", required=True)
            child.add_argument("--input-pointer")
    args = parser.parse_args(argv)
    if args.command == "matrix":
        print(json.dumps(integration_matrix(), indent=2))
        return
    common = {
        "ecosystem": args.ecosystem,
        "run_id": args.run_id,
        "artifact_digest": args.artifact_digest,
    }
    if args.command == "qualification":
        run = capture_qualification(
            **common,
            **{
                name + "_path": getattr(args, name)
                for name in ("cases", "schedule", "profile", "export", "raw_output")
            },
        )
    else:
        run = capture_records(
            **common,
            records_path=args.records,
            source_version=args.source_version,
            input_projection={"kind": "json-pointer", "pointer": args.input_pointer}
            if args.input_pointer
            else None,
        )
    write_file_no_replace(args.output, canonical_json_bytes(run))
    print(json.dumps(evaluator_input_capabilities(run), indent=2))


if __name__ == "__main__":
    main()
