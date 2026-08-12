#!/usr/bin/env python3
"""Generate, execute, and independently verify the evaluator matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
ARTIFACTS = ROOT / "artifacts"
SUPPORT = ROOT / "runner_support.py"
REPLAYABLE_CORPUS = ROOT / "authoritative"
REPLAYABLE_ARTIFACTS = REPLAYABLE_CORPUS / "artifacts"
SIGNED_TRANSACTIONS = ROOT / "signed-transactions"

sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPLAYABLE_CORPUS))
from replay import replay as replay_import  # noqa: E402

from examples.integrations.evaluator_transaction.build_attestation import (  # noqa: E402
    load_evaluator_build_attestation,
    verify_evaluator_build_attestation,
)
from invarlock.evaluator_qualification import (  # noqa: E402
    qualify_evaluator_export,
)
from invarlock.evidence_pack_contract import canonical_json_bytes  # noqa: E402
from invarlock.evidence_pack_json import (  # noqa: E402
    StrictJsonError,
    parse_json_bytes,
    read_regular_file_bytes,
)
from invarlock.evidence_pack_verification import (  # noqa: E402
    verify_comparison_evidence,
)
from invarlock.evidence_receipt import (  # noqa: E402
    verify_signed_verification_receipt,
)

_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_TRANSACTION_FIELDS = {
    "base_image_id",
    "entrypoint",
    "evaluator_version",
    "executed_on",
    "format",
    "lock_sha256",
    "profile_id",
    "runtime_image_id",
    "source_bundle_sha256",
    "source_commit",
    "verification",
}
_VERIFICATION_FIELDS = {
    "artifact_digests",
    "evidence_signer_fingerprint",
    "runtime_digests",
    "schedule_digest",
    "trust_profile_digest",
    "verifier_fingerprint",
    "verifier_identity",
}


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def digest(payload: bytes) -> str:
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def bundle_digest(profile: dict[str, Any]) -> str:
    paths = [SUPPORT, ROOT / profile["runner"]]
    paths.extend(ROOT / asset for asset in profile["runner_assets"])
    document = bytearray()
    for path in sorted(paths):
        relative = path.relative_to(ROOT).as_posix().encode()
        body = path.read_bytes()
        document.extend(len(relative).to_bytes(8, "big"))
        document.extend(relative)
        document.extend(len(body).to_bytes(8, "big"))
        document.extend(body)
    return digest(bytes(document))


def qualification_profile(profile: dict[str, Any]) -> dict[str, Any]:
    lock = ROOT / profile["lock"]
    return {
        "authority": profile["authority"],
        "execution": {
            "dependency_lock_sha256": digest(lock.read_bytes()),
            "runner_sha256": bundle_digest(profile),
        },
        "format": "invarlock/evaluator-qualification-profile-v1",
        "profile_id": profile["profile_id"],
        "upstream": {
            "package": profile["upstream"],
            "project_url": profile["project_url"],
        },
    }


def matrix_document() -> dict[str, Any]:
    return load(ROOT / "matrix.json")


def profiles() -> list[dict[str, Any]]:
    matrix = matrix_document()
    values = matrix.get("profiles")
    if not isinstance(values, list) or any(
        not isinstance(value, dict) for value in values
    ):
        raise ValueError("matrix profiles must be an array of objects")
    return values


def categories() -> dict[str, dict[str, str]]:
    values = matrix_document().get("categories")
    if not isinstance(values, dict) or any(
        not isinstance(key, str)
        or not isinstance(value, dict)
        or not isinstance(value.get("display_name"), str)
        for key, value in values.items()
    ):
        raise ValueError("matrix categories must map identifiers to display names")
    return values


def selection_policy() -> dict[str, Any]:
    value = matrix_document().get("selection")
    if not isinstance(value, dict):
        raise ValueError("matrix selection must be an object")
    return value


def release_focus() -> list[str]:
    value = matrix_document().get("release_focus")
    profiles_value = value.get("flagship_profiles") if isinstance(value, dict) else None
    if (
        not isinstance(profiles_value, list)
        or not profiles_value
        or any(not isinstance(profile_id, str) for profile_id in profiles_value)
        or len(set(profiles_value)) != len(profiles_value)
    ):
        raise ValueError("matrix release focus must name unique flagship profiles")
    return profiles_value


def demonstration_levels() -> dict[str, dict[str, bool]]:
    values = load(ROOT / "demonstrations.json").get("profiles")
    if not isinstance(values, dict) or any(
        not isinstance(key, str) or not isinstance(value, dict)
        for key, value in values.items()
    ):
        raise ValueError("demonstration levels must be an object of profile objects")
    return values


def replayable_profiles() -> list[dict[str, Any]]:
    return [
        profile
        for profile in profiles()
        if profile["authority"]["mode"] == "deterministic_per_record"
    ]


def _sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ValueError(f"{label} must be a sha256 digest")
    return value


def _side_digests(value: object, *, label: str) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != {"baseline", "subject"}:
        raise ValueError(f"{label} must contain exactly baseline and subject")
    return {
        side: _sha256(value[side], label=f"{label}.{side}")
        for side in ("baseline", "subject")
    }


def load_retained_transaction(path: Path, *, profile_id: str) -> dict[str, Any]:
    """Load strict, bounded metadata that selects retained trust inputs."""

    try:
        payload = read_regular_file_bytes(
            path,
            label=f"{profile_id} retained transaction metadata",
            max_bytes=1024 * 1024,
        )
        value = parse_json_bytes(
            payload, label=f"{profile_id} retained transaction metadata"
        )
    except StrictJsonError as exc:
        raise ValueError(str(exc)) from exc
    if not isinstance(value, dict) or set(value) != _TRANSACTION_FIELDS:
        raise ValueError(f"{profile_id}: retained transaction fields are invalid")
    if (
        value.get("format") != "invarlock/retained-evaluator-transaction-v1"
        or value.get("profile_id") != profile_id
    ):
        raise ValueError(f"{profile_id}: retained transaction metadata is invalid")
    for field in (
        "base_image_id",
        "lock_sha256",
        "runtime_image_id",
        "source_bundle_sha256",
    ):
        _sha256(value.get(field), label=f"{profile_id} {field}")
    if (
        not isinstance(value.get("source_commit"), str)
        or _COMMIT.fullmatch(value["source_commit"]) is None
    ):
        raise ValueError(f"{profile_id}: source commit is invalid")
    if (
        not isinstance(value.get("executed_on"), str)
        or _DATE.fullmatch(value["executed_on"]) is None
    ):
        raise ValueError(f"{profile_id}: execution date is invalid")
    if (
        not isinstance(value.get("evaluator_version"), str)
        or not value["evaluator_version"]
    ):
        raise ValueError(f"{profile_id}: evaluator version is invalid")
    entrypoint = value.get("entrypoint")
    if (
        not isinstance(entrypoint, list)
        or not entrypoint
        or any(
            not isinstance(part, str) or not part or "\x00" in part
            for part in entrypoint
        )
    ):
        raise ValueError(f"{profile_id}: entrypoint is invalid")
    verification = value.get("verification")
    if not isinstance(verification, dict) or set(verification) != _VERIFICATION_FIELDS:
        raise ValueError(f"{profile_id}: retained verification fields are invalid")
    _side_digests(verification.get("artifact_digests"), label="artifact digests")
    _side_digests(verification.get("runtime_digests"), label="runtime digests")
    for field in (
        "evidence_signer_fingerprint",
        "schedule_digest",
        "trust_profile_digest",
        "verifier_fingerprint",
    ):
        _sha256(verification.get(field), label=f"{profile_id} {field}")
    if (
        not isinstance(verification.get("verifier_identity"), str)
        or not verification["verifier_identity"].strip()
    ):
        raise ValueError(f"{profile_id}: verifier identity is invalid")
    return value


def verify_signed_transaction(profile_id: str) -> None:
    root = SIGNED_TRANSACTIONS / profile_id
    transaction = load_retained_transaction(
        root / "transaction.json", profile_id=profile_id
    )
    verification = transaction["verification"]
    evidence = verify_comparison_evidence(
        root / "evidence",
        policy_path=root / "policy.json",
        expected_artifact_digests=verification["artifact_digests"],
        expected_schedule_digest=verification["schedule_digest"],
        expected_runtime_digests=verification["runtime_digests"],
        expected_signer_fingerprint=verification["evidence_signer_fingerprint"],
    )
    receipt = verify_signed_verification_receipt(
        root / "verification.receipt.json",
        root / "evidence",
        policy_path=root / "policy.json",
        expected_artifact_digests=verification["artifact_digests"],
        expected_schedule_digest=verification["schedule_digest"],
        expected_runtime_digests=verification["runtime_digests"],
        expected_pack_signer_fingerprint=verification["evidence_signer_fingerprint"],
        expected_verifier_identity=verification["verifier_identity"],
        expected_verifier_fingerprint=verification["verifier_fingerprint"],
        expected_trust_profile_digest=verification["trust_profile_digest"],
    )
    public_key = serialization.load_pem_public_key(
        read_regular_file_bytes(
            root / "builder.public.pem",
            label=f"{profile_id} builder public key",
            max_bytes=64 * 1024,
        )
    )
    if not isinstance(public_key, ed25519.Ed25519PublicKey):
        raise ValueError(f"{profile_id}: builder public key is not Ed25519")
    build = verify_evaluator_build_attestation(
        load_evaluator_build_attestation(root / "build-attestation.json"),
        builder_public_key=public_key,
        evaluator=profile_id,
        evaluator_version=transaction["evaluator_version"],
        runtime_image_id=transaction["runtime_image_id"],
        base_image_id=transaction["base_image_id"],
        source_commit=transaction["source_commit"],
        source_bundle_sha256=transaction["source_bundle_sha256"],
        lock_sha256=transaction["lock_sha256"],
        entrypoint=transaction["entrypoint"],
    )
    if (
        evidence.payload.get("ok") is not True
        or not receipt.ok
        or build.get("runtime_image_id") != transaction["runtime_image_id"]
    ):
        raise ValueError(f"{profile_id}: retained signed transaction did not pass")
    print(f"retained signed transaction {profile_id}: pass", flush=True)


def write_profile(
    profile: dict[str, Any],
    *,
    artifacts: Path = ARTIFACTS,
) -> Path:
    artifact = artifacts / profile["profile_id"]
    artifact.mkdir(parents=True, exist_ok=True)
    destination = artifact / "profile.json"
    destination.write_bytes(canonical_json_bytes(qualification_profile(profile)))
    return destination


def runner_command(
    profile: dict[str, Any],
    profile_path: Path,
    *,
    cases: Path = ROOT / "cases.json",
    schedule: Path = ROOT / "schedule.json",
) -> list[str]:
    artifact = profile_path.parent
    common = [
        "--cases",
        str(cases),
        "--dependency-lock",
        str(ROOT / profile["lock"]),
        "--export",
        str(artifact / "export.json"),
        "--profile",
        str(profile_path),
        "--raw-output",
        str(artifact / "upstream-output.json"),
        "--schedule",
        str(schedule),
    ]
    runner = str(ROOT / profile["runner"])
    launch = [
        "-c",
        "import runpy,sys;p=sys.argv.pop(1);runpy.run_path(p,run_name='__main__')",
        runner,
    ]
    if profile["upstream"]["ecosystem"] == "npm":
        return [sys.executable, *launch, *common]
    return [
        "uv",
        "run",
        "--no-project",
        "--with-requirements",
        str(ROOT / profile["lock"]),
        "python",
        *launch,
        *common,
    ]


def execute(selected: set[str]) -> None:
    _execute_profiles(
        profiles(),
        selected=selected,
        artifacts=ARTIFACTS,
        cases=ROOT / "cases.json",
        schedule=ROOT / "schedule.json",
        replayable=False,
    )


def _execute_profiles(
    matrix_profiles: list[dict[str, Any]],
    *,
    selected: set[str],
    artifacts: Path,
    cases: Path,
    schedule: Path,
    replayable: bool,
) -> None:
    environment = os.environ.copy()
    environment.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
    environment.setdefault("HF_HOME", "/tmp/invarlock-evaluator-hf-cache")
    environment.setdefault("PROMPTFOO_DISABLE_TELEMETRY", "1")
    existing_python_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        f"{ROOT}{os.pathsep}{existing_python_path}"
        if existing_python_path
        else str(ROOT)
    )
    for profile in matrix_profiles:
        if selected and profile["profile_id"] not in selected:
            continue
        print(f"execute {profile['profile_id']}", flush=True)
        profile_path = write_profile(profile, artifacts=artifacts)
        subprocess.run(
            runner_command(
                profile,
                profile_path,
                cases=cases,
                schedule=schedule,
            ),
            check=True,
            env=environment,
        )
        result = qualify(profile, artifacts=artifacts, schedule=schedule)
        (profile_path.parent / "qualification-result.json").write_bytes(
            canonical_json_bytes(result.as_dict())
        )
        if replayable:
            replay_import(profile["profile_id"], write=True)


def execute_replayable(selected: set[str]) -> None:
    _execute_profiles(
        replayable_profiles(),
        selected=selected,
        artifacts=REPLAYABLE_ARTIFACTS,
        cases=REPLAYABLE_CORPUS / "cases.json",
        schedule=REPLAYABLE_CORPUS / "schedule.json",
        replayable=True,
    )


def qualify(
    profile: dict[str, Any],
    *,
    artifacts: Path = ARTIFACTS,
    schedule: Path = ROOT / "schedule.json",
):
    artifact = artifacts / profile["profile_id"]
    return qualify_evaluator_export(
        profile_path=artifact / "profile.json",
        schedule_path=schedule,
        export_path=artifact / "export.json",
        raw_output_path=artifact / "upstream-output.json",
    )


def verify() -> None:
    matrix_profiles = profiles()
    identifiers = [profile["profile_id"] for profile in matrix_profiles]
    if not identifiers or len(set(identifiers)) != len(identifiers):
        raise ValueError("the matrix must contain unique profile identifiers")
    category_ids = categories()
    for profile in matrix_profiles:
        category = profile.get("category")
        if not isinstance(category, str) or category not in category_ids:
            raise ValueError(f"{profile['profile_id']}: category is invalid")
        if profile.get("support_status") != "maintained_adapter":
            raise ValueError(f"{profile['profile_id']}: support status is invalid")
    selection = selection_policy()
    if (
        not isinstance(selection.get("reviewed_on"), str)
        or not isinstance(selection.get("minimum_activity_window_months"), int)
        or selection["minimum_activity_window_months"] < 1
    ):
        raise ValueError("matrix selection review metadata is invalid")
    authority_modes = {profile["authority"]["mode"] for profile in matrix_profiles}
    if authority_modes != {"deterministic_per_record", "observation_only"}:
        raise ValueError(
            "the catalog must demonstrate replayable and observation-only authority"
        )
    levels = demonstration_levels()
    if set(levels) != set(identifiers):
        raise ValueError("demonstration levels must cover exactly the matrix profiles")
    for profile_id, status in levels.items():
        if set(status) != {"retained_signed_transaction"} or not isinstance(
            status["retained_signed_transaction"], bool
        ):
            raise ValueError(f"{profile_id}: demonstration status is invalid")
    profiles_by_id = {profile["profile_id"]: profile for profile in matrix_profiles}
    for profile_id in release_focus():
        profile = profiles_by_id.get(profile_id)
        if profile is None:
            raise ValueError(f"release-focus profile is missing: {profile_id}")
        if profile["authority"]["mode"] != "deterministic_per_record":
            raise ValueError(f"release-focus profile is not replayable: {profile_id}")
        if not levels[profile_id]["retained_signed_transaction"]:
            raise ValueError(
                f"release-focus profile lacks a retained signed transaction: {profile_id}"
            )
    for profile in matrix_profiles:
        artifact = ARTIFACTS / profile["profile_id"]
        expected_profile = canonical_json_bytes(qualification_profile(profile))
        if (artifact / "profile.json").read_bytes() != expected_profile:
            raise ValueError(f"{profile['profile_id']}: profile is stale")
        result = qualify(profile)
        retained = (artifact / "qualification-result.json").read_bytes()
        if retained != canonical_json_bytes(result.as_dict()):
            raise ValueError(f"{profile['profile_id']}: qualification result is stale")
        raw = load(artifact / "upstream-output.json")
        if raw.get("format") != "invarlock/upstream-evaluator-execution-v1":
            raise ValueError(f"{profile['profile_id']}: raw output format is invalid")
        if raw.get("upstream") != profile["upstream"]:
            raise ValueError(f"{profile['profile_id']}: upstream identity is invalid")
        print(
            f"verified {profile['profile_id']}: {result.outcome}",
            flush=True,
        )
    retained_profiles = {
        profile_id
        for profile_id, status in levels.items()
        if status["retained_signed_transaction"]
    }
    retained_directories = {
        path.name for path in SIGNED_TRANSACTIONS.iterdir() if path.is_dir()
    }
    if retained_directories != retained_profiles:
        raise ValueError(
            "retained signed transaction packages do not match demonstration status"
        )
    for profile_id in sorted(retained_profiles):
        verify_signed_transaction(profile_id)


def verify_replayable() -> None:
    cases = load(REPLAYABLE_CORPUS / "cases.json")
    records = cases.get("records")
    source_evaluation = cases.get("source_evaluation")
    if (
        cases.get("format") != "invarlock/evaluator-authoritative-cases-v1"
        or not isinstance(records, list)
        or len(records) != 102
        or not isinstance(source_evaluation, dict)
        or source_evaluation.get("kind") != "model_execution"
    ):
        raise ValueError("replayable corpus must bind one 102-record model execution")
    matrix_profiles = replayable_profiles()
    if not matrix_profiles:
        raise ValueError("at least one independently replayable profile is required")
    for profile in matrix_profiles:
        artifact = REPLAYABLE_ARTIFACTS / profile["profile_id"]
        expected_profile = canonical_json_bytes(qualification_profile(profile))
        if (artifact / "profile.json").read_bytes() != expected_profile:
            raise ValueError(f"{profile['profile_id']}: replayable profile is stale")
        raw = load(artifact / "upstream-output.json")
        if raw.get("source_evaluation") != source_evaluation:
            raise ValueError(
                f"{profile['profile_id']}: model execution provenance is stale"
            )
        result = qualify(
            profile,
            artifacts=REPLAYABLE_ARTIFACTS,
            schedule=REPLAYABLE_CORPUS / "schedule.json",
        )
        retained = (artifact / "qualification-result.json").read_bytes()
        if retained != canonical_json_bytes(result.as_dict()):
            raise ValueError(
                f"{profile['profile_id']}: replayable qualification is stale"
            )
        replay_import(profile["profile_id"], write=False)
        print(
            f"verified independently replayable import {profile['profile_id']}",
            flush=True,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    execute_parser = subparsers.add_parser("execute")
    execute_parser.add_argument("profiles", nargs="*")
    replayable_parser = subparsers.add_parser("execute-replayable")
    replayable_parser.add_argument("profiles", nargs="*")
    subparsers.add_parser("verify")
    subparsers.add_parser("verify-replayable")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "execute":
        execute(set(args.profiles))
    elif args.command == "execute-replayable":
        execute_replayable(set(args.profiles))
    elif args.command == "verify-replayable":
        verify_replayable()
    else:
        verify()


if __name__ == "__main__":
    main()
