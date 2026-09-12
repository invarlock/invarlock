"""Freeze one reviewer's K2 labels and gate final plans without model calls."""

from __future__ import annotations

import argparse
import copy
import io
import os
import re
import tempfile
import zipfile
from collections import Counter
from pathlib import Path

try:
    from examples import judge_measurements_reference as ref
except ModuleNotFoundError:  # Direct execution from the source checkout.
    import judge_measurements_reference as ref

from invarlock.evidence_pack_json import parse_json_bytes, read_regular_file_bytes
from invarlock.judge_measurements.contracts import (
    MEASUREMENTS_MAX_BYTES,
    validate_measurements,
)

MAX_NOTES = 4096
PROTOCOL = {
    "id": "single-reviewer-exact-label-v1",
    "unit": "frozen answer",
    "aggregation": "compare every scheduled judge repetition to the one human label",
    "missing": "retain incomplete trials in coverage; exclude from agreement denominator",
    "statistics": "exact label matches and confusion counts only; no pass threshold or confidence interval",
    "scope": "descriptive agreement for this reviewed subset; not inter-rater reliability or population accuracy",
}


def pinned_files(bundle: Path, expected_sha256: str) -> dict[str, bytes]:
    """Snapshot inputs before validation so later reads cannot change their meaning."""
    if not isinstance(expected_sha256, str) or not re.fullmatch(
        r"[0-9a-f]{64}", expected_sha256
    ):
        raise ValueError("an independent reference manifest SHA-256 pin is required")
    with tempfile.TemporaryDirectory(prefix="invarlock-review-reference-") as directory:
        root = Path(directory)
        if bundle.is_dir():
            return ref.authenticated_bundle_files(bundle, expected_sha256)
        payload = ref.read(bundle, ref.MAX_ARCHIVE_BYTES)
        archive = root / "reference.zip"
        archive.write_bytes(payload)
        ref.validate_archive(archive, expected_sha256=expected_sha256)
        with zipfile.ZipFile(io.BytesIO(payload)) as packed:
            return {name: packed.read(name) for name in packed.namelist()}


def decode(files: dict[str, bytes], name: str):
    return ref.parse_json_bytes(files[name], label=name)


def measurements_object(path: Path) -> dict:
    payload = read_regular_file_bytes(
        path, label="judge measurements", max_bytes=MEASUREMENTS_MAX_BYTES
    )
    value = parse_json_bytes(payload, label="judge measurements")
    if not isinstance(value, dict):
        raise ValueError("judge measurements must contain a JSON object")
    return value


def validate_review(sheet: dict, template: dict) -> None:
    """Only ratings and bounded optional notes may differ from the frozen sheet."""
    if not isinstance(sheet, dict) or set(sheet) != set(template):
        raise ValueError("review fields differ from the frozen template")
    if not isinstance(sheet["cases"], list) or len(sheet["cases"]) != len(
        template["cases"]
    ):
        raise ValueError("review must cover every frozen case exactly once")
    normalized = copy.deepcopy(sheet)
    labels = {rating["label"] for rating in template["scale"]["ratings"]}
    for row, frozen in zip(normalized["cases"], template["cases"], strict=True):
        if not isinstance(row, dict) or set(row) != set(frozen):
            raise ValueError("review case fields differ from the frozen template")
        for field in ("response_1_rating", "response_2_rating"):
            if not isinstance(row[field], str) or row[field] not in labels:
                raise ValueError("every response requires one allowed rating label")
            row[field] = None
        notes = row["review_notes"]
        if notes is not None and (
            not isinstance(notes, str) or len(notes.encode("utf-8")) > MAX_NOTES
        ):
            raise ValueError("review notes must be null or at most 4096 UTF-8 bytes")
        row["review_notes"] = None
    if ref.canonical_payload(normalized) != ref.canonical_payload(template):
        raise ValueError(
            "review changed frozen IDs, ordering, input, answers, rubric or scale"
        )


def reconcile(sheet: dict, selection: dict) -> list[dict]:
    workflow, stage = sheet["workflow"], sheet["stage"]
    ids = selection["pilot" if stage == "rubric_development" else "human_review"]
    mapping = {
        "review-" + ref.rank(workflow, stage + "_id", case_id): case_id
        for case_id in ids
    }
    result = []
    for row in sheet["cases"]:
        case_id = mapping[row["review_id"]]
        sides = ["baseline", "subject"]
        if int(ref.rank(workflow, stage + "_orientation", case_id), 16) % 2:
            sides.reverse()
        result.extend(
            {
                "review_id": row["review_id"],
                "case_id": case_id,
                "side": side,
                "rating": row[f"response_{index}_rating"],
            }
            for index, side in enumerate(sides, 1)
        )
    return result


def agreement(labels: list[dict], measurements: dict) -> dict:
    human = {(row["case_id"], row["side"]): row["rating"] for row in labels}
    confusion: Counter = Counter()
    scheduled = incomplete = 0
    for trial in measurements["trials"]:
        key = (trial["case_id"], trial["side"])
        if key not in human:
            continue
        scheduled += 1
        if trial["status"] != "complete":
            incomplete += 1
        else:
            confusion[human[key], trial["parse"]["rating"]] += 1
    compared = sum(confusion.values())
    matches = sum(count for (a, b), count in confusion.items() if a == b)
    return {
        "protocol": PROTOCOL,
        "reviewed_answers": len(human),
        "scheduled_trials": scheduled,
        "incomplete_trials": incomplete,
        "compared_trials": compared,
        "exact_matches": matches,
        "exact_agreement": {"numerator": matches, "denominator": compared},
        "confusion": [
            {"human": a, "judge": b, "count": count}
            for (a, b), count in sorted(confusion.items())
        ],
    }


def write_new(path: Path, value: object) -> str:
    data = ref.canonical_payload(value)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(data)
    return ref.sha(data)


def complete_review(
    *,
    bundle: Path,
    expected_sha256: str,
    completed: Path,
    reviewer: str,
    outcome: str,
    output: Path,
    measurements: Path | None = None,
    activate: bool = False,
) -> dict:
    if not isinstance(reviewer, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", reviewer
    ):
        raise ValueError("reviewer must be a bounded pseudonymous identifier")
    if outcome not in {"rubric_confirmed", "revision_required"}:
        raise ValueError("review outcome must confirm the rubric or require revision")
    files = pinned_files(bundle, expected_sha256)
    sheet = ref.obj(completed)
    workflow, stage = sheet["workflow"], sheet["stage"]
    if workflow not in ref.WORKFLOWS or stage not in {
        "rubric_development",
        "final_validation",
    }:
        raise ValueError("unsupported review workflow or stage")
    template = decode(files, f"human_review/{stage}/{workflow}.json")
    validate_review(sheet, template)
    if activate and (stage != "rubric_development" or outcome != "rubric_confirmed"):
        raise ValueError("activation requires a complete rubric-confirmed pilot review")
    split = "pilot" if stage == "rubric_development" else "final"
    candidate = decode(files, f"{workflow}/final/candidate_plan.json")
    plan = (
        decode(files, f"{workflow}/pilot/plan.json")
        if split == "pilot"
        else candidate["plan"]
    )
    measured = measurements_object(measurements) if measurements is not None else None
    if measured is not None:
        validate_measurements(
            measured,
            plan,
            baseline_run=decode(files, f"{workflow}/{split}/baseline_run.json"),
            subject_run=decode(files, f"{workflow}/{split}/subject_run.json"),
        )
    if activate and (
        measured is None or measured["completeness"]["status"] != "complete"
    ):
        raise ValueError("activation requires complete retained pilot measurements")
    # Exclusive publication freezes the complete submitted labels before unblinding.
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    sheet_digest = write_new(output / "completed-review.json", sheet)
    labels = reconcile(sheet, decode(files, f"{workflow}/selection.json"))
    labels_digest = write_new(output / "reconciled-labels.json", labels)
    result = {
        "format": "invarlock/k2-single-reviewer-record-v1",
        "reviewer": reviewer,
        "reference_manifest_sha256": expected_sha256,
        "workflow": workflow,
        "stage": stage,
        "outcome": outcome,
        "completed_review_sha256": sheet_digest,
        "reconciled_labels_sha256": labels_digest,
        "agreement_protocol": PROTOCOL,
        "assurance": "reviewer identity, independence and blinding are operator attestations; one reviewer does not establish inter-rater reliability",
        "activation": "activated" if activate else "not_activated",
        "agreement": agreement(labels, measured) if measured is not None else None,
        "measurements_sha256": write_new(output / "measurements.json", measured)
        if measured is not None
        else None,
    }
    if activate:
        result["activated_files"] = {
            f"{name}.json": write_new(output / f"{name}.json", candidate[name])
            for name in ("plan", "analysis_policy")
        }
    write_new(output / "review-record.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--completed", type=Path, required=True)
    parser.add_argument("--reviewer", required=True)
    parser.add_argument(
        "--outcome", choices=("rubric_confirmed", "revision_required"), required=True
    )
    parser.add_argument("--measurements", type=Path)
    parser.add_argument("--activate", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = complete_review(**vars(args))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f"Review validation failed: {exc}\n")
    print(ref.canonical_payload(result).decode())


if __name__ == "__main__":
    main()
