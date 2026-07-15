from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from scripts.release import runtime_release_asset_handoff as handoff
from scripts.release import runtime_release_evidence as evidence

SOURCE_COMMIT = "a" * 40
RELEASE_COMMIT = "9" * 40
SOURCE_ARCHIVE_SHA256 = "b" * 64
IMAGE_DIGEST = "sha256:" + "c" * 64
RELEASE_TAG = "v0.13.0"
REPOSITORY = "invarlock/invarlock"


def canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def write_json(path: Path, value: object, *, newline: bool = False) -> Path:
    path.write_bytes(canonical(value) + (b"\n" if newline else b""))
    return path


def gguf_summary(path: Path) -> Path:
    return write_json(
        path,
        {
            "evidence_sha256": "d" * 64,
            "fixture_revision": "e" * 40,
            "format_version": evidence.GGUF_FORMAT,
            "image_digest": IMAGE_DIGEST,
            "runs": 2,
            "status": "ok",
        },
        newline=True,
    )


def tensorrt_summary(path: Path, *, marker: str = "1") -> Path:
    return write_json(
        path,
        {
            "candidate_image_digest": IMAGE_DIGEST,
            "engine_bundle_tree_sha256": marker * 64,
            "format_version": evidence.TENSORRT_FORMAT,
            "gpu_count": 2,
            "ok": True,
            "output_sha256": "2" * 64,
            "runtime_provider_receipt_sha256": "3" * 64,
            "tokenizer_sha256": "4" * 64,
        },
        newline=True,
    )


def behavior_receipt(path: Path) -> Path:
    def side(marker: str) -> dict[str, str]:
        return {
            name: hashlib.sha256(f"{marker}-{name}".encode()).hexdigest()
            for name in (
                "artifact_identity_sidecar_sha256",
                "evaluation_report_sha256",
                "provider_receipt_sidecar_sha256",
                "runtime_manifest_sha256",
                "scoring_observation_sidecar_sha256",
            )
        }

    return write_json(
        path,
        {
            "baseline": side("baseline"),
            "baseline_score": 1.0,
            "claim_set": evidence.BEHAVIOR_CLAIM_SET,
            "format_version": evidence.BEHAVIOR_RECEIPT_FORMAT,
            "metric": "exact_match",
            "policy_digest": "sha256:" + "6" * 64,
            "regression": 0.25,
            "schedule_sha256": "7" * 64,
            "subject": side("subject"),
            "subject_score": 0.75,
            "verdict": "pass",
        },
    )


def build_legacy_asset(tmp_path: Path) -> tuple[Path, str]:
    summary = gguf_summary(tmp_path / "gguf.json")
    asset = tmp_path / "runtime-release-evidence.tar.gz"
    evidence.build_asset(
        output=asset,
        source_commit=SOURCE_COMMIT,
        source_archive_sha256=SOURCE_ARCHIVE_SHA256,
        qualification_summaries={"llama_cpp": summary},
        behavioral_receipts=[],
    )
    return asset, hashlib.sha256(asset.read_bytes()).hexdigest()


def stage_legacy_asset(tmp_path: Path) -> tuple[dict[str, object], Path, Path]:
    source, digest = build_legacy_asset(tmp_path)
    output = tmp_path / "handoff"
    output.mkdir()
    result = handoff.stage_handoff(
        source_asset=source,
        output_dir=output,
        release_tag=RELEASE_TAG,
        expected_source_commit=SOURCE_COMMIT,
        expected_source_archive_sha256=SOURCE_ARCHIVE_SHA256,
        expected_asset_sha256=digest,
        expected_providers=frozenset({"llama_cpp"}),
        expected_qualifications=frozenset({"llama_cpp"}),
        require_behavioral_claim=False,
    )
    return (
        result,
        output / str(result["asset_filename"]),
        output / str(result["digest_filename"]),
    )


def completed(
    arguments: list[str], *, stdout: str = "", code: int = 0
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(arguments, code, stdout=stdout, stderr="")
