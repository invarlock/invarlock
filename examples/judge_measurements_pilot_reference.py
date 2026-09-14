"""Check the retained K2 pilot archive and replay its signed corrected evidence."""

from __future__ import annotations

import argparse
import hashlib
import io
import re
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

from invarlock.evidence_pack_json import parse_json_bytes, read_regular_file_bytes
from invarlock.judge_measurements.acceptance import (
    replay_signed_judge_verification_receipt,
)
from invarlock.judge_measurements.evidence import replay_judge_evidence

WORKFLOWS = ("grounded_qa", "extraction")


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def read_archive(bundle: Path, expected_sha256: str) -> dict[str, bytes]:
    """Verify the independently selected physical archive before extracting it."""
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError("an independent archive SHA-256 pin is required")
    raw = read_regular_file_bytes(
        bundle, label="pilot reference archive", max_bytes=8 * 1024 * 1024
    )
    if sha(raw) != expected_sha256:
        raise ValueError("archive bytes differ from the expected SHA-256 pin")
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        entries = archive.infolist()
        if len(entries) > 128 or sum(x.file_size for x in entries) > 64 * 1024 * 1024:
            raise ValueError("pilot archive exceeds the retained reference limits")
        names = set()
        for entry in entries:
            path = PurePosixPath(entry.filename)
            if (
                not re.fullmatch(r"[A-Za-z0-9_./-]+", entry.filename)
                or entry.filename in names
                or path.is_absolute()
                or ".." in path.parts
                or str(path) != entry.filename
                or entry.is_dir()
                or (entry.external_attr >> 16) != 0o100644
            ):
                raise ValueError("pilot archive contains an unsafe or duplicate entry")
            names.add(entry.filename)
        files = {entry.filename: archive.read(entry) for entry in entries}
    manifest = parse_json_bytes(files["reference.json"], label="pilot reference")
    if manifest["format"] != "invarlock/judge-pilot-reference-v1":
        raise ValueError("unsupported pilot reference")
    if set(manifest["files"]) != set(files) - {"reference.json"}:
        raise ValueError("pilot reference file inventory differs")
    for name, pin in manifest["files"].items():
        if pin != {"sha256": sha(files[name]), "size_bytes": len(files[name])}:
            raise ValueError(f"pilot reference file pin differs: {name}")
    return files


def validate_reference(bundle: Path, expected_sha256: str) -> dict:
    files = read_archive(bundle, expected_sha256)
    manifest = parse_json_bytes(files["reference.json"], label="pilot reference")
    results = {}
    with tempfile.TemporaryDirectory(prefix="judge-pilot-reference-") as directory:
        root = Path(directory).resolve()
        for name, raw in files.items():
            target = root.joinpath(*PurePosixPath(name).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
        for workflow in WORKFLOWS:
            current = root / "corrected" / workflow
            receipt = parse_json_bytes(
                (current / "receipt.json").read_bytes(), label="retained receipt"
            )
            # These example anchors are trusted only through the independent
            # physical archive pin; an arbitrary included key grants no authority.
            verifier = receipt["statement"]["verifier"]
            result = replay_signed_judge_verification_receipt(
                current / "receipt.json",
                evidence_path=current / "evidence",
                recipient_policy_path=current / "recipient.json",
                expected_verifier_identity=verifier["identity"],
                expected_verifier_fingerprint=verifier["signing_key_fingerprint"],
            )
            analysis = replay_judge_evidence(current / "evidence").analysis_result
            if not result.verified or not result.authenticated or not result.replayed:
                raise ValueError(
                    f"{workflow}: signed receipt or evidence replay failed"
                )
            if result.accepted or result.decision != "insufficient_evidence":
                raise ValueError(f"{workflow}: unexpected recipient outcome")
            if analysis.reasons != ("maximum_interval_width_exceeded",):
                raise ValueError(f"{workflow}: unexpected analysis reason")
            results[workflow] = {
                "authenticated": result.authenticated,
                "replayed": result.replayed,
                "verified": result.verified,
                "accepted": result.accepted,
                "decision": result.decision,
                "analysis": analysis.to_dict(),
            }
    return {
        "archive_sha256": expected_sha256,
        "reference_manifest_sha256": sha(files["reference.json"]),
        "workflows": results,
        "historical_replay": "retained separately; not promoted to current evidence",
        "human_review": manifest["human_review"],
        "final_plans": manifest["final_plans"],
        "new_model_calls": 0,
    }


def main() -> None:
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--expected-sha256", required=True)
    args = parser.parse_args()
    result = validate_reference(args.bundle, args.expected_sha256)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
