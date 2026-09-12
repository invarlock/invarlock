"""Prepare judge collection from complete frozen answer capture; makes no calls."""

from __future__ import annotations

import argparse
import copy
import hashlib
from pathlib import Path

from examples.answer_capture import check_result, digest, exact, read
from invarlock.evaluation_record_contracts.contracts import validate
from invarlock.evaluation_records.cases import case_set_digest, validate_run_case_set
from invarlock.evaluation_records.io import run_digest
from invarlock.evidence_pack_contract import canonical_json_bytes
from invarlock.filesystem.atomic_file import write_file_no_replace
from invarlock.judge_measurements.analysis import decode_analysis_policy
from invarlock.judge_measurements.contracts import (
    measurement_plan_digest,
    render_judge_request,
    validate_measurement_plan,
)


def prepare(
    capture: Path,
    template: dict,
    policy: dict,
    units: dict,
    collection: dict,
    output: Path,
) -> dict:
    manifest = read(capture / "manifest.json")
    manifest_digest = digest(manifest)
    runs = {
        side: read(capture / f"{side}_run.json") for side in ("baseline", "subject")
    }
    case_digest = case_set_digest(manifest["case_set"])
    rows = {}
    for side, run in runs.items():
        validate(run, "run")
        validate_run_case_set(run, case_digest)
        if (
            run["source_digest"] != manifest_digest
            or run["artifact_digest"] != manifest["config"][side]["artifact_digest"]
        ):
            raise ValueError("captured run does not bind its retained manifest")
        rows[side] = {row["id"]: row for row in run["records"]}
    for index, job in enumerate(manifest["jobs"]):
        if read(capture / f"{index:06}.attempt.json") != {
            "manifest_sha256": manifest_digest,
            "job": index,
        }:
            raise ValueError("capture attempt identity mismatch")
        result = read(capture / f"{index:06}.result.json")
        exact(result, {"manifest_sha256", "job", "result"}, "retained result")
        if (
            result["manifest_sha256"] != manifest_digest
            or type(result["job"]) is not int
            or result["job"] != index
        ):
            raise ValueError("capture result identity mismatch")
        check_result(result["result"], job, manifest["config"]["limits"])
        row = rows[job["request"]["side"]][job["request"]["case_id"]]
        if row["error"] is not None or row["output"] != result["result"]["output"]:
            raise ValueError("captured answer differs from retained first result")
    exact(units, set(rows["baseline"]), "case unit mapping")
    if not all(isinstance(unit, str) and unit for unit in units.values()):
        raise ValueError("each case needs an explicit independent unit identity")
    plan = copy.deepcopy(template)
    validate_measurement_plan(plan)
    plan.update(
        case_set_sha256=case_digest,
        baseline_run_sha256=run_digest(runs["baseline"]),
        subject_run_sha256=run_digest(runs["subject"]),
    )
    plan["sampling"]["case_units"] = [
        {"case_id": key, "unit_id": units[key]} for key in sorted(units)
    ]
    plan["schedule"]["expected_trials"] = (
        2 * len(units) * plan["schedule"]["repetitions"]
    )
    bindings = []
    for case_id in sorted(units):
        binding = {"case_id": case_id}
        for side in ("baseline", "subject"):
            row = rows[side][case_id]
            binding[f"{side}_answer_sha256"] = hashlib.sha256(
                row["output"].encode()
            ).hexdigest()
            binding[f"{side}_request_sha256"] = hashlib.sha256(
                render_judge_request(
                    plan, input_text=row["input"], answer_text=row["output"]
                )
            ).hexdigest()
        bindings.append(binding)
    plan["answer_bindings"] = bindings
    validate_measurement_plan(plan)
    policy = {**copy.deepcopy(policy), "plan_sha256": measurement_plan_digest(plan)}
    decode_analysis_policy(policy, plan=plan)
    request = {
        "format_version": "invarlock/evaluation-request-v3",
        "execution": {
            "mode": "judge_collect",
            "collection": {
                "integration": "inspect-judge",
                "configuration": "collection.json",
            },
        },
        "comparison": {
            "baseline_run": "baseline_run.json",
            "subject_run": "subject_run.json",
            "plan": "plan.json",
            "measurements": None,
            "policy": "analysis_policy.json",
        },
        "output": {"evidence": "evidence", "signer_identity": "answer-capture-judge"},
    }
    # Only a new directory can be published. Preparation never calls a judge.
    output.mkdir(mode=0o700, parents=False, exist_ok=False)
    for name, value in {
        "baseline_run.json": runs["baseline"],
        "subject_run.json": runs["subject"],
        "plan.json": plan,
        "analysis_policy.json": policy,
        "collection.json": collection,
        "request.json": request,
    }.items():
        write_file_no_replace(
            output / name, canonical_json_bytes(value), create_parents=False
        )
    return {
        "plan_sha256": measurement_plan_digest(plan),
        "expected_trials": plan["schedule"]["expected_trials"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "capture",
        "plan-template",
        "policy",
        "units",
        "collection",
        "directory",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = prepare(
            args.capture,
            read(args.plan_template),
            read(args.policy),
            read(args.units),
            read(args.collection),
            args.directory,
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f"Judge preparation stopped: {exc}\n")
    print(f"Judge inputs ready; no calls made: {result}")
    print(f"Next: invarlock evaluate {args.directory / 'request.json'} --preflight")


if __name__ == "__main__":
    main()
