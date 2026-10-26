"""Evaluator qualification closes retained export, identity and scoring paths."""

import json

import pytest

from invarlock.evaluator_qualification import qualify_evaluator_export
from native.tests.test_judge_evidence_differential import native
from tests.core.test_evaluator_qualification import (
    _digest_bytes,
    _digest_text,
    _write_json,
    qualification_fixture,
)


@pytest.mark.parametrize(
    "case",
    [
        "pass",
        "observation",
        "raw",
        "package",
        "profile",
        "reverse",
        "duplicate",
        "output_digest",
        "reported_score",
        "status",
        "aggregate",
        "noncanonical",
        "nonobject",
        "missing_summary",
        "summary_digest",
        "reference",
        "nonexact",
    ],
)
def test_qualification(case, tmp_path):
    paths = qualification_fixture(
        tmp_path.resolve(),
        mode="observation_only"
        if case in {"observation", "missing_summary", "summary_digest"}
        else "deterministic_per_record",
    )
    profile, schedule, export, raw = paths
    p = json.loads(profile.read_bytes())
    s = json.loads(schedule.read_bytes())
    e = json.loads(export.read_bytes())
    if case == "raw":
        raw.write_bytes(b"changed")
    elif case == "package":
        e["upstream"]["version"] = "4.2.1"
    elif case == "profile":
        e["profile_id"] = "another"
    elif case == "reverse":
        e["records"].reverse()
    elif case == "duplicate":
        e["records"][1] = e["records"][0]
    elif case == "output_digest":
        e["records"][0]["output_sha256"] = _digest_text("changed")
    elif case == "reported_score":
        e["records"][0]["reported_score"] = 0.0
    elif case == "status":
        e["records"][0]["status"] = "error"
    elif case == "aggregate":
        e.update(
            records=[],
            summary={
                "kind": "aggregate_metrics",
                "sha256": _digest_bytes(raw.read_bytes()),
            },
        )
    elif case == "missing_summary":
        e["summary"] = None
    elif case == "summary_digest":
        e["summary"]["sha256"] = _digest_text("changed")
    elif case == "reference":
        s["records"][0]["reference_output"] = "wrong"
    elif case == "nonexact":
        p["authority"]["metric"] = {
            "kind": "normalized_exact_match",
            "configuration": {},
        }
        s["records"][0]["reference_output"] = "expected"
        s["records"][1]["reference_output"] = "right"
    _write_json(profile, p)
    _write_json(schedule, s)
    e["bindings"]["profile_sha256"] = _digest_bytes(profile.read_bytes())
    e["bindings"]["schedule_sha256"] = _digest_bytes(schedule.read_bytes())
    _write_json(export, e)
    if case == "noncanonical":
        schedule.write_text(json.dumps(s, indent=2))
    elif case == "nonobject":
        profile.write_bytes(b"[]\n")
    config = dict(
        zip(
            ("profile_path", "schedule_path", "export_path", "raw_output_path"),
            paths,
            strict=True,
        )
    )
    try:
        expected = qualify_evaluator_export(**config)
    except ValueError as failure:
        status, actual = native(
            "qualify-export", {k: str(v) for k, v in config.items()}
        )
        assert status != 0
        assert actual["error"] == str(failure)
    else:
        destination = tmp_path / "native-result.json"
        status, actual = native(
            "qualify-export",
            {k: str(v) for k, v in config.items()} | {"destination": str(destination)},
        )
        assert status == 0, actual
        assert actual["result"] == expected.as_dict()
        assert destination.read_bytes() == expected.as_json().encode()
        assert len(actual["runtime_records"]) == len(expected.runtime_records())


def test_fractional_scores_and_installed_public_api(tmp_path):
    from invarlock.core.scoring import score

    profile, schedule, export, raw = qualification_fixture(tmp_path.resolve())
    p = json.loads(profile.read_bytes())
    s = json.loads(schedule.read_bytes())
    e = json.loads(export.read_bytes())
    configuration = {"fields": [f"/{index}" for index in range(7)]}
    p["authority"]["metric"] = {"kind": "json_fields", "configuration": configuration}
    reference = json.dumps({str(index): 1 for index in range(7)})
    s["records"] = []
    e["records"] = []
    for index in range(301):
        observed = json.dumps(
            {str(field): int(field < index % 8) for field in range(7)}
        )
        input_digest = _digest_text(str(index))
        record_id = f"record-{index}"
        s["records"].append(
            {
                "record_id": record_id,
                "input_sha256": input_digest,
                "reference_output": reference,
                "reference_output_sha256": _digest_text(reference),
            }
        )
        e["records"].append(
            {
                "record_id": record_id,
                "input_sha256": input_digest,
                "output_text": observed,
                "output_sha256": _digest_text(observed),
                "reported_score": score(
                    "json_fields", reference, observed, configuration
                ),
                "status": "ok",
            }
        )
    _write_json(profile, p)
    _write_json(schedule, s)
    e["bindings"]["profile_sha256"] = _digest_bytes(profile.read_bytes())
    e["bindings"]["schedule_sha256"] = _digest_bytes(schedule.read_bytes())
    _write_json(export, e)
    config = dict(
        zip(
            ("profile_path", "schedule_path", "export_path", "raw_output_path"),
            (profile, schedule, export, raw),
            strict=True,
        )
    )
    expected = qualify_evaluator_export(**config)
    actual = qualify_evaluator_export(**config, backend="rust")
    assert actual.as_json() == expected.as_json()
    assert actual.runtime_records() == expected.runtime_records()
    assert actual.mean_score.hex() == expected.mean_score.hex()
