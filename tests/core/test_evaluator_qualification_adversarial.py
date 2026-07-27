from __future__ import annotations

import json
from pathlib import Path

import pytest

from invarlock.evaluator_qualification import (
    EvaluatorQualificationError,
    qualify_evaluator_export,
)
from tests.core.test_evaluator_qualification import (
    _digest_text,
    _write_json,
    qualification_fixture,
)


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _qualify(paths: tuple[Path, Path, Path, Path]):
    profile, schedule, export, raw = paths
    return qualify_evaluator_export(
        profile_path=profile,
        schedule_path=schedule,
        export_path=export,
        raw_output_path=raw,
    )


def test_raw_upstream_output_tamper_fails_closed(tmp_path: Path) -> None:
    paths = qualification_fixture(tmp_path)
    paths[3].write_bytes(b'{"native_results":[1,1]}\n')

    with pytest.raises(
        EvaluatorQualificationError,
        match="raw upstream output digest does not match",
    ):
        _qualify(paths)


def test_profile_and_package_provenance_mismatch_fails_closed(tmp_path: Path) -> None:
    paths = qualification_fixture(tmp_path)
    export = _load(paths[2])
    upstream = export["upstream"]
    assert isinstance(upstream, dict)
    upstream["version"] = "4.2.1"
    _write_json(paths[2], export)

    with pytest.raises(
        EvaluatorQualificationError,
        match="upstream package identity does not match",
    ):
        _qualify(paths)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("reverse", "record order and input identities"),
        ("duplicate", "non-unique elements"),
        ("output_digest", "output digest is invalid"),
        ("reported_score", "reported score does not match"),
        ("failed_status", "is not successful"),
    ],
)
def test_per_record_tampering_fails_closed(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    paths = qualification_fixture(tmp_path)
    export = _load(paths[2])
    records = export["records"]
    assert isinstance(records, list)
    assert all(isinstance(record, dict) for record in records)
    if mutation == "reverse":
        records.reverse()
    elif mutation == "duplicate":
        records[1] = dict(records[0])
    elif mutation == "output_digest":
        records[0]["output_sha256"] = _digest_text("tampered")
    elif mutation == "reported_score":
        records[0]["reported_score"] = 0.0
    elif mutation == "failed_status":
        records[0]["status"] = "error"
    _write_json(paths[2], export)

    with pytest.raises(EvaluatorQualificationError, match=message):
        _qualify(paths)


def test_aggregate_only_export_cannot_use_verdict_profile(tmp_path: Path) -> None:
    paths = qualification_fixture(tmp_path)
    export = _load(paths[2])
    export["records"] = []
    export["summary"] = {
        "kind": "aggregate_metrics",
        "sha256": export["bindings"]["raw_output_sha256"],
    }
    _write_json(paths[2], export)

    with pytest.raises(
        EvaluatorQualificationError,
        match="must not substitute an aggregate summary",
    ):
        _qualify(paths)


def test_unknown_contract_and_noncanonical_json_fail_closed(tmp_path: Path) -> None:
    paths = qualification_fixture(tmp_path)
    profile = _load(paths[0])
    profile["format"] = "invarlock/evaluator-profile-v999"
    _write_json(paths[0], profile)
    with pytest.raises(EvaluatorQualificationError, match="profile is invalid"):
        _qualify(paths)

    paths = qualification_fixture(tmp_path / "second")
    value = _load(paths[1])
    paths[1].write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(EvaluatorQualificationError, match="canonical JSON"):
        _qualify(paths)


def test_core_qualification_module_has_no_evaluator_name_dispatch() -> None:
    module_path = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "invarlock"
        / "evaluator_qualification.py"
    )
    source = module_path.read_text(encoding="utf-8").lower()

    for evaluator_name in (
        "deepeval",
        "garak",
        "giskard",
        "inspect-ai",
        "lm-evaluation-harness",
        "promptfoo",
        "ragas",
    ):
        assert evaluator_name not in source
