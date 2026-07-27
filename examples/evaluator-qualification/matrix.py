#!/usr/bin/env python3
"""Generate, execute, and independently verify the evaluator matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
ARTIFACTS = ROOT / "artifacts"
SUPPORT = ROOT / "runner_support.py"
AUTHORITATIVE = ROOT / "authoritative"
AUTHORITATIVE_ARTIFACTS = AUTHORITATIVE / "artifacts"

sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(AUTHORITATIVE))
from replay import replay as replay_authoritative_import  # noqa: E402

from invarlock.evaluator_qualification import (  # noqa: E402
    qualify_evaluator_export,
)
from invarlock.evidence_pack_contract import canonical_json_bytes  # noqa: E402


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


def demonstration_levels() -> dict[str, dict[str, bool]]:
    values = load(ROOT / "demonstrations.json").get("profiles")
    if not isinstance(values, dict) or any(
        not isinstance(key, str) or not isinstance(value, dict)
        for key, value in values.items()
    ):
        raise ValueError("demonstration levels must be an object of profile objects")
    return values


def authoritative_profiles() -> list[dict[str, Any]]:
    levels = demonstration_levels()
    return [
        profile
        for profile in profiles()
        if levels[profile["profile_id"]]["authoritative_import"]
    ]


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
        authoritative=False,
    )


def _execute_profiles(
    matrix_profiles: list[dict[str, Any]],
    *,
    selected: set[str],
    artifacts: Path,
    cases: Path,
    schedule: Path,
    authoritative: bool,
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
        if authoritative:
            replay_authoritative_import(profile["profile_id"], write=True)


def execute_authoritative(selected: set[str]) -> None:
    _execute_profiles(
        authoritative_profiles(),
        selected=selected,
        artifacts=AUTHORITATIVE_ARTIFACTS,
        cases=AUTHORITATIVE / "cases.json",
        schedule=AUTHORITATIVE / "schedule.json",
        authoritative=True,
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
    if len(identifiers) < 12 or len(set(identifiers)) != len(identifiers):
        raise ValueError(
            "the representative matrix must contain at least 12 unique "
            "profile identifiers"
        )
    category_ids = categories()
    for profile in matrix_profiles:
        category = profile.get("category")
        if not isinstance(category, str) or category not in category_ids:
            raise ValueError(f"{profile['profile_id']}: category is invalid")
    selection = selection_policy()
    if (
        not isinstance(selection.get("reviewed_on"), str)
        or not isinstance(selection.get("minimum_activity_window_months"), int)
        or selection["minimum_activity_window_months"] < 1
    ):
        raise ValueError("matrix selection review metadata is invalid")
    verdict_count = sum(
        profile["authority"]["mode"] == "deterministic_per_record"
        for profile in matrix_profiles
    )
    if verdict_count < 10:
        raise ValueError("the matrix must retain at least ten per-record profiles")
    levels = demonstration_levels()
    if set(levels) != set(identifiers):
        raise ValueError("demonstration levels must cover exactly the matrix profiles")
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


def verify_authoritative() -> None:
    cases = load(AUTHORITATIVE / "cases.json")
    records = cases.get("records")
    producer = cases.get("producer")
    if (
        cases.get("format") != "invarlock/evaluator-authoritative-cases-v1"
        or not isinstance(records, list)
        or len(records) != 102
        or not isinstance(producer, dict)
        or producer.get("kind") != "model_execution"
    ):
        raise ValueError(
            "authoritative corpus must bind one 102-record model execution"
        )
    matrix_profiles = authoritative_profiles()
    if len(matrix_profiles) < 10:
        raise ValueError("at least ten profiles must demonstrate authoritative import")
    for profile in matrix_profiles:
        artifact = AUTHORITATIVE_ARTIFACTS / profile["profile_id"]
        expected_profile = canonical_json_bytes(qualification_profile(profile))
        if (artifact / "profile.json").read_bytes() != expected_profile:
            raise ValueError(f"{profile['profile_id']}: authoritative profile is stale")
        raw = load(artifact / "upstream-output.json")
        if raw.get("source_evaluation") != producer:
            raise ValueError(
                f"{profile['profile_id']}: model execution provenance is stale"
            )
        result = qualify(
            profile,
            artifacts=AUTHORITATIVE_ARTIFACTS,
            schedule=AUTHORITATIVE / "schedule.json",
        )
        retained = (artifact / "qualification-result.json").read_bytes()
        if retained != canonical_json_bytes(result.as_dict()):
            raise ValueError(
                f"{profile['profile_id']}: authoritative qualification is stale"
            )
        replay_authoritative_import(profile["profile_id"], write=False)
        print(
            f"verified authoritative import {profile['profile_id']}",
            flush=True,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    execute_parser = subparsers.add_parser("execute")
    execute_parser.add_argument("profiles", nargs="*")
    authoritative_parser = subparsers.add_parser("execute-authoritative")
    authoritative_parser.add_argument("profiles", nargs="*")
    subparsers.add_parser("verify")
    subparsers.add_parser("verify-authoritative")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "execute":
        execute(set(args.profiles))
    elif args.command == "execute-authoritative":
        execute_authoritative(set(args.profiles))
    elif args.command == "verify-authoritative":
        verify_authoritative()
    else:
        verify()


if __name__ == "__main__":
    main()
