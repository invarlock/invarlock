from __future__ import annotations

import hashlib
import json
from pathlib import Path

from invarlock.evaluator_qualification import (
    EVALUATOR_EXPORT_FORMAT,
    EVALUATOR_PROFILE_FORMAT,
    EVALUATOR_QUALIFICATION_FORMAT,
    EVALUATOR_SCHEDULE_FORMAT,
    qualify_evaluator_export,
)


def _digest_bytes(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def _digest_text(value: str) -> str:
    return _digest_bytes(value.encode("utf-8"))


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def qualification_fixture(
    root: Path,
    *,
    mode: str = "deterministic_per_record",
) -> tuple[Path, Path, Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    raw = root / "upstream-output.json"
    raw.write_bytes(b'{"native_results":[1,0]}\n')
    profile = root / "profile.json"
    schedule = root / "schedule.json"
    export = root / "export.json"
    profile_payload = {
        "authority": (
            {
                "metric": {"kind": "exact_match"},
                "mode": "deterministic_per_record",
                "reason": None,
            }
            if mode == "deterministic_per_record"
            else {
                "metric": None,
                "mode": "observation_only",
                "reason": "nondeterministic_judge",
            }
        ),
        "execution": {
            "dependency_lock_sha256": _digest_text("custom-evaluator==4.2.0\n"),
            "runner_sha256": _digest_text("custom runner"),
        },
        "format": EVALUATOR_PROFILE_FORMAT,
        "profile_id": "acme-proprietary-evaluator",
        "upstream": {
            "package": {
                "ecosystem": "private",
                "name": "acme-evaluator",
                "version": "4.2.0",
            },
            "project_url": "https://evaluator.example.invalid",
        },
    }
    _write_json(profile, profile_payload)
    schedule_payload = {
        "format": EVALUATOR_SCHEDULE_FORMAT,
        "records": [
            {
                "input_sha256": _digest_text("question one"),
                "record_id": "record-1",
                "reference_output_sha256": _digest_text("expected"),
            },
            {
                "input_sha256": _digest_text("question two"),
                "record_id": "record-2",
                "reference_output_sha256": _digest_text("right"),
            },
        ],
        "schedule_id": "custom-smoke",
    }
    _write_json(schedule, schedule_payload)
    records: list[dict[str, object]] = []
    if mode == "deterministic_per_record":
        records = [
            {
                "input_sha256": _digest_text("question one"),
                "output_sha256": _digest_text("expected"),
                "output_text": "expected",
                "record_id": "record-1",
                "reported_score": 1.0,
                "status": "ok",
            },
            {
                "input_sha256": _digest_text("question two"),
                "output_sha256": _digest_text("wrong"),
                "output_text": "wrong",
                "record_id": "record-2",
                "reported_score": 0.0,
                "status": "ok",
            },
        ]
    export_payload = {
        "bindings": {
            "dependency_lock_sha256": profile_payload["execution"][
                "dependency_lock_sha256"
            ],
            "profile_sha256": _digest_bytes(profile.read_bytes()),
            "raw_output_sha256": _digest_bytes(raw.read_bytes()),
            "runner_sha256": profile_payload["execution"]["runner_sha256"],
            "schedule_sha256": _digest_bytes(schedule.read_bytes()),
        },
        "format": EVALUATOR_EXPORT_FORMAT,
        "profile_id": profile_payload["profile_id"],
        "records": records,
        "summary": (
            None
            if mode == "deterministic_per_record"
            else {
                "kind": "judge_scores",
                "sha256": _digest_bytes(raw.read_bytes()),
            }
        ),
        "upstream": profile_payload["upstream"]["package"],
    }
    _write_json(export, export_payload)
    return profile, schedule, export, raw


def test_custom_profile_qualifies_through_generic_sdk(tmp_path: Path) -> None:
    profile, schedule, export, raw = qualification_fixture(tmp_path)

    result = qualify_evaluator_export(
        profile_path=profile,
        schedule_path=schedule,
        export_path=export,
        raw_output_path=raw,
    )

    assert result.format == EVALUATOR_QUALIFICATION_FORMAT
    assert result.profile_id == "acme-proprietary-evaluator"
    assert result.outcome == "qualified_for_import"
    assert result.authority == "verdict_authority"
    assert result.record_count == 2
    assert result.scores == (1.0, 0.0)
    assert result.mean_score == 0.5
    assert [record.record_id for record in result.runtime_records()] == [
        "record-1",
        "record-2",
    ]


def test_observation_only_profile_never_exposes_runtime_records(tmp_path: Path) -> None:
    profile, schedule, export, raw = qualification_fixture(
        tmp_path,
        mode="observation_only",
    )

    result = qualify_evaluator_export(
        profile_path=profile,
        schedule_path=schedule,
        export_path=export,
        raw_output_path=raw,
    )

    assert result.outcome == "observation_only"
    assert result.authority == "observation_only"
    assert result.reason_codes == ("nondeterministic_judge",)
    assert result.record_count == 0
    assert result.scores == ()
    assert result.mean_score is None
    assert result.runtime_records() == ()


def test_result_serialization_is_canonical_and_digest_bound(tmp_path: Path) -> None:
    profile, schedule, export, raw = qualification_fixture(tmp_path)

    result = qualify_evaluator_export(
        profile_path=profile,
        schedule_path=schedule,
        export_path=export,
        raw_output_path=raw,
    )
    payload = json.loads(result.as_json())

    assert result.as_json().endswith("\n")
    assert payload["format"] == EVALUATOR_QUALIFICATION_FORMAT
    assert payload["bindings"] == {
        "dependency_lock_sha256": _digest_text("custom-evaluator==4.2.0\n"),
        "export_sha256": _digest_bytes(export.read_bytes()),
        "profile_sha256": _digest_bytes(profile.read_bytes()),
        "raw_output_sha256": _digest_bytes(raw.read_bytes()),
        "runner_sha256": _digest_text("custom runner"),
        "schedule_sha256": _digest_bytes(schedule.read_bytes()),
    }
    destination = tmp_path / "qualification.json"
    result.write(destination)
    assert destination.read_text(encoding="utf-8") == result.as_json()
