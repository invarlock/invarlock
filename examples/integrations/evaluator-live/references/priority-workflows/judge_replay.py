"""Replay retained judge receipts and bounded pause/resume observations offline."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import tempfile
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
CATALOG_SHA256 = "acf24e0284ee078d0a4eaabd8fef659b873569b873d7f7a7dc278edccb2ad298"


def require(value: bool, message: str) -> None:
    if not value:
        raise ValueError(message)


def same(left, right) -> bool:
    return json.dumps(
        left, sort_keys=True, separators=(",", ":"), allow_nan=False
    ) == json.dumps(right, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def shared(root: Path):
    spec = importlib.util.spec_from_file_location(
        "priority_shared_reference", root / "mistral-7b-sentinel/replay.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_pinned(reader, path: Path, pin: dict) -> dict[str, bytes]:
    require(
        pin["bytes"] <= 10 * 1024**2
        and pin["expanded_bytes"] <= 128 * 1024**2
        and pin["members"] <= 6000,
        "Archive bounds exceed supported limits.",
    )
    return reader.read_archive(
        path,
        expected_sha256=pin["sha256"].removeprefix("sha256:"),
        maximum_bytes=pin["bytes"],
        maximum_expanded_bytes=pin["expanded_bytes"],
        maximum_members=pin["members"],
    )


def read_reference(directory: Path, root: Path) -> tuple[dict, dict[str, bytes], dict]:
    from invarlock.evidence_pack_json import read_regular_file_bytes

    raw = read_regular_file_bytes(
        directory / "judge-reference.json", max_bytes=100_000, label="judge catalog"
    )
    require(
        hashlib.sha256(raw).hexdigest() == CATALOG_SHA256,
        "Independent catalog digest differs.",
    )
    catalog = json.loads(raw)
    require(
        catalog["format"] == "invarlock/priority-judge-catalog-v1", "Unknown catalog."
    )
    reader = shared(root)
    members = {}
    for pin in catalog["archives"]:
        require(Path(pin["path"]).name == pin["path"], "Invalid archive path.")
        part = read_pinned(reader, directory / pin["path"], pin)
        require(not members.keys() & part.keys(), "Overlapping archive members.")
        members.update(part)
    raw = members["judge-manifest.json"]
    require(digest(raw) == catalog["manifest_sha256"], "Manifest digest differs.")
    manifest = json.loads(raw)["files"]
    require(
        set(members) == set(manifest) | {"judge-manifest.json"},
        "Archive inventory differs.",
    )
    for name, pin in manifest.items():
        require(
            len(members[name]) == pin["bytes"]
            and digest(members[name]) == pin["sha256"],
            "Archive member differs.",
        )
    reference = json.loads(members["judge-reference-map.json"])
    require(reference == catalog["reference"], "Pinned replay map differs.")
    companions = {
        name: read_pinned(reader, root / "priority-workflows" / pin["path"], pin)
        for name, pin in reference["companions"].items()
    }
    for pack in reference["packs"]:
        for link in pack["capture_links"]:
            for name, pin in link["files"].items():
                raw = companions[link["companion"]][link["directory"] + "/" + name]
                require(
                    len(raw) == pin["bytes"] and digest(raw) == pin["sha256"],
                    "Model capture link differs.",
                )
    return catalog, members, companions


def check_lifecycle(entry: Path, pack: dict, measured: dict) -> None:
    """Check retained observation consistency; this does not attest wall-clock timing."""
    from invarlock.judge_measurements.contracts import validate_measurements

    directory = entry / "lifecycle"
    stop = json.loads((directory / "lifecycle-stop-observation.json").read_bytes())
    resume = json.loads((directory / "lifecycle-resume-observation.json").read_bytes())
    before, after = resume["before"], resume["after"]
    require(
        stop["checkpoint"] == before
        and stop["admission_sha256"] == resume["admission_sha256"],
        "Lifecycle snapshot identity differs.",
    )
    require(
        before["admissions"] == before["results"] == 2
        and after["admissions"] == after["results"] == pack["retained_attempts"],
        "Lifecycle counts differ.",
    )
    require(
        stop["result"]["stop_reason"] == "requested"
        and stop["result"]["resumable"] is True,
        "Lifecycle stop was not resumable.",
    )
    for name, pin in before["files"].items():
        require(
            after["files"].get(name) == pin, "Initial checkpoint changed during resume."
        )
    for member in pack["lifecycle"]["checked_initial_members"]:
        path = (
            directory / "collection.json"
            if member.endswith("/collection.json")
            else entry / "checkpoints" / Path(member).name
        )
        raw = path.read_bytes()
        pin = before["files"][path.name]
        require(
            len(raw) == pin["bytes"] and digest(raw) == pin["sha256"],
            "Retained initial checkpoint bytes differ.",
        )
    partial = json.loads((directory / "stopped-measurements.json").read_bytes())
    validate_measurements(
        partial,
        json.loads((entry / "frozen/plan.json").read_bytes()),
        baseline_run=json.loads((entry / "frozen/baseline_run.json").read_bytes()),
        subject_run=json.loads((entry / "frozen/subject_run.json").read_bytes()),
    )
    initial = {
        trial["trial_id"]: trial["attempts"]
        for trial in partial["trials"]
        if trial["attempts"]
    }
    final = {trial["trial_id"]: trial["attempts"] for trial in measured["trials"]}
    require(
        len(initial) == 2
        and all(same(final[key], attempts) for key, attempts in initial.items()),
        "Initial measured attempts changed during resume.",
    )


def verify_pack(entry: Path, pack: dict, report_root: Path | None = None) -> dict:
    from invarlock.judge_measurements.acceptance import (
        replay_signed_judge_verification_receipt,
    )

    for name in (
        "plan.json",
        "analysis_policy.json",
        "baseline_run.json",
        "subject_run.json",
    ):
        require(
            json.loads((entry / "frozen" / name).read_bytes())
            == json.loads((entry / "evidence" / name).read_bytes()),
            "Frozen judge input differs.",
        )
    require(
        digest((entry / "frozen/plan.json").read_bytes()) == pack["plan_sha256"],
        "Frozen plan bytes differ.",
    )
    measured = json.loads((entry / "evidence/measurements.json").read_bytes())
    sources = {source["source_id"]: source for source in measured["sources"]}
    events = {}
    for source in pack["event_sources"]:
        raw = (entry / "sources" / Path(source["member"]).name).read_bytes()
        original = sources[source["source_id"]]
        require(
            raw == original["content"].encode("utf-8")
            and len(raw) == original["byte_size"]
            and digest(raw) == source["sha256"],
            "Original normalized SDK event source differs.",
        )
        for record in json.loads(raw)["records"]:
            for attempt, event in enumerate(record["events"], 1):
                events[(record["trial"]["trial_id"], attempt)] = event
    attempts = [
        (trial["trial_id"], attempt["attempt"])
        for trial in measured["trials"]
        for attempt in trial["attempts"]
    ]
    shards = [
        json.loads(path.read_bytes())
        for path in (entry / "checkpoints").glob("result-*.json")
    ]
    require(
        len(attempts) == pack["retained_attempts"]
        and len(shards) == len(attempts)
        and set(attempts)
        == {(shard["trial_id"], shard["attempt"]) for shard in shards},
        "Durable result inventory differs.",
    )
    require(
        all(
            same(shard["event"], events[(shard["trial_id"], shard["attempt"])])
            for shard in shards
        ),
        "Durable result event differs from retained SDK source.",
    )
    if pack["lifecycle"]:
        check_lifecycle(entry, pack, measured)
    result = replay_signed_judge_verification_receipt(
        entry / "recipient/verification.receipt.json",
        evidence_path=entry / "evidence",
        recipient_policy_path=entry / "recipient/trust.json",
        expected_verifier_identity=pack["verifier"]["identity"],
        expected_verifier_fingerprint=pack["verifier"]["signing_key_fingerprint"],
    ).to_dict()
    require(
        all(result[key] == value for key, value in pack["expected"].items()),
        "Original signed receipt or evidence replay differs: " + pack["id"],
    )
    rendered = None
    if report_root is not None:
        from invarlock.judge_measurements.reporting import render_judge_evidence

        destination = report_root / pack["id"]
        destination.mkdir(parents=True, exist_ok=False)
        report = render_judge_evidence(
            entry / "evidence",
            html_path=destination / "report.html",
            markdown_path=destination / "report.md",
            junit_path=destination / "report.xml",
            explain=True,
        )
        require(not report.errors, "Current report rendering failed: " + pack["id"])
        rendered = destination.relative_to(report_root.parent).as_posix()
    return {"id": pack["id"], "verification": result, "current_report": rendered}


def verify_capture_bindings(
    output: Path, reference: dict, companions: dict, root: Path
) -> None:
    """Re-import complete original captures and compare every frozen judge run."""
    spec = importlib.util.spec_from_file_location(
        "judge_capture_recipient", root.parent / "recipient.py"
    )
    recipient = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(recipient)
    with tempfile.TemporaryDirectory(prefix="judge-capture-replay-") as temporary:
        staging = Path(temporary).resolve()
        for companion, files in companions.items():
            for name, raw in files.items():
                path = staging / companion / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(raw)
        for pack in reference["packs"]:
            entry = output / pack["directory"]
            protocol_path = entry / "frozen/protocol.json"
            protocol = json.loads(protocol_path.read_bytes())
            links = {
                link["role"]: staging / link["companion"] / link["directory"]
                for link in pack["capture_links"]
            }
            _, runs = recipient.prepare(
                protocol_path,
                recipient.common.digest(protocol),
                links["baseline"],
                links["subject"],
                pack["evaluator"],
                pack["route"],
                staging / "rebuilt" / pack["id"],
            )
            for role, run in zip(("baseline", "subject"), runs, strict=True):
                require(
                    recipient.same(
                        run,
                        json.loads(
                            (entry / "frozen" / (role + "_run.json")).read_bytes()
                        ),
                    ),
                    "Original SDK/task source does not reproduce frozen judge run.",
                )


def replay(
    output: Path,
    *,
    directory: Path = HERE,
    reference_root: Path | None = None,
    require_installed: bool = True,
) -> dict:
    """Verify physical archives, original receipts, source events and lifecycle facts."""
    root = reference_root or directory.parent
    reader = shared(root)
    if require_installed:
        reader.require_installed()
    catalog, members, companions = read_reference(directory, root)
    reference = catalog["reference"]
    counts = reference["counts"]
    require(
        len({pack["id"] for pack in reference["packs"]}) == len(reference["packs"])
        and all(
            isinstance(pack["id"], str)
            and re.fullmatch(r"[a-z0-9-]{1,128}", pack["id"])
            and pack["directory"] == "judge/" + pack["id"]
            for pack in reference["packs"]
        ),
        "Duplicate or invalid pack identity.",
    )
    require(
        len(reference["packs"]) == counts["packs"] == 24
        and sum(pack["planned_trials"] for pack in reference["packs"])
        == counts["planned_trials"]
        == 1472
        and sum(pack["retained_attempts"] for pack in reference["packs"])
        == counts["retained_attempts"]
        == 1288
        and counts["unadmitted_trials"] == 184,
        "Campaign counts differ.",
    )
    require(
        sum(bool(pack["lifecycle"]) for pack in reference["packs"])
        == counts["lifecycle_entries"]
        == 4,
        "Lifecycle inventory differs.",
    )
    output.mkdir(parents=True, exist_ok=False)
    output = output.resolve(strict=True)
    for name, raw in members.items():
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    verify_capture_bindings(output, reference, companions, root)
    report_root = output / "current-reports"
    report_root.mkdir()
    results = [
        verify_pack(output / pack["directory"], pack, report_root)
        for pack in reference["packs"]
    ]
    require(
        dict(Counter(row["verification"]["decision"] for row in results))
        == counts["decisions"],
        "Replayed decision counts differ.",
    )
    report = {
        "archive_sha256": [pin["sha256"] for pin in catalog["archives"]],
        "counts": counts,
        "original_receipts_replayed": len(results),
        "results": results,
    }
    (output / "replay-results.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--directory", type=Path, default=HERE)
    parser.add_argument("--reference-root", type=Path)
    args = parser.parse_args(argv)
    from invarlock.security import enforce_network_policy

    enforce_network_policy(False)
    result = replay(
        args.output, directory=args.directory, reference_root=args.reference_root
    )
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "results"}, indent=2
        )
    )


if __name__ == "__main__":
    main()
