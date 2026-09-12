from __future__ import annotations

import json
from pathlib import Path

import pytest

from examples import judge_measurements_reference as ref
from examples import judge_measurements_review as review

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / "examples/judge-measurements/references/k2-32b"


@pytest.fixture(scope="module")
def frozen():
    pins = json.loads((ARCHIVE / "archive.json").read_text())
    # The pin here tests transport binding; operators obtain it independently.
    pin = pins["reference_manifest_sha256"]
    return pin, review.pinned_files(ARCHIVE / "reference.zip", pin)


def filled(files, stage="rubric_development"):
    sheet = review.decode(files, f"human_review/{stage}/grounded_qa.json")
    labels = [r["label"] for r in sheet["scale"]["ratings"]]
    for row in sheet["cases"]:
        row["response_1_rating"] = labels[0]
        row["response_2_rating"] = labels[-1]
        row["review_notes"] = "Reviewed against the supplied context."
    return sheet


@pytest.mark.parametrize("field", ["input", "response_1", "response_2", "review_id"])
def test_review_rejects_changed_immutable_case(frozen, field):
    _, files = frozen
    template = review.decode(files, "human_review/rubric_development/grounded_qa.json")
    sheet = filled(files)
    sheet["cases"][0][field] += "edited"
    with pytest.raises(ValueError, match="changed frozen"):
        review.validate_review(sheet, template)


@pytest.mark.parametrize(
    "change", ["rubric", "scale", "order", "missing", "label", "notes", "extra"]
)
def test_review_rejects_incomplete_or_modified_sheet(frozen, change):
    _, files = frozen
    template = review.decode(files, "human_review/rubric_development/grounded_qa.json")
    sheet = filled(files)
    if change == "rubric":
        sheet["rubric"]["text"] += "x"
    elif change == "scale":
        sheet["scale"]["ratings"][0]["value"] = "0.1"
    elif change == "order":
        sheet["cases"].reverse()
    elif change == "missing":
        sheet["cases"].pop()
    elif change == "label":
        sheet["cases"][0]["response_1_rating"] = None
    elif change == "notes":
        sheet["cases"][0]["review_notes"] = "é" * 2049
    else:
        sheet["cases"][0]["extra"] = True
    with pytest.raises(ValueError):
        review.validate_review(sheet, template)


def test_reconciliation_preserves_frozen_orientation(frozen):
    _, files = frozen
    sheet = filled(files)
    result = review.reconcile(sheet, review.decode(files, "grounded_qa/selection.json"))
    raw = review.decode(files, "grounded_qa/raw.json")
    answers = {
        side: {r["id"]: r["output"] for r in raw[role]}
        for side, role in (("baseline", "A"), ("subject", "B"))
    }
    by_id = {r["review_id"]: r for r in sheet["cases"]}
    for row in result:
        original = by_id[row["review_id"]]
        position = 1 if row["rating"] == original["response_1_rating"] else 2
        assert answers[row["side"]][row["case_id"]] == original[f"response_{position}"]
    assert len(result) == 80


def kwargs(tmp_path, frozen, **overrides):
    pin, files = frozen
    sheet = tmp_path / "completed.json"
    sheet.write_bytes(ref.canonical_payload(filled(files)))
    return dict(
        bundle=ARCHIVE / "reference.zip",
        expected_sha256=pin,
        completed=sheet,
        reviewer="reviewer-1",
        outcome="rubric_confirmed",
        output=tmp_path / "result",
        **overrides,
    )


def test_complete_review_freezes_record_and_refuses_overwrite(tmp_path, frozen):
    args = kwargs(tmp_path, frozen)
    result = review.complete_review(**args)
    assert result["activation"] == "not_activated"
    assert result["agreement"] is None
    assert result["completed_review_sha256"] == ref.sha(
        ref.read(args["output"] / "completed-review.json")
    )
    assert not (args["output"] / "plan.json").exists()
    with pytest.raises(FileExistsError):
        review.complete_review(**args)


@pytest.mark.parametrize("pin", [None, "", "0" * 64, "../bad"])
def test_independent_pin_required(pin):
    with pytest.raises(ValueError):
        review.pinned_files(ARCHIVE / "reference.zip", pin)


def test_activation_requires_complete_pilot_measurements(tmp_path, frozen):
    args = kwargs(tmp_path, frozen, activate=True)
    with pytest.raises(ValueError, match="complete retained pilot"):
        review.complete_review(**args)
    assert not args["output"].exists()
    args["outcome"] = "revision_required"
    with pytest.raises(ValueError, match="rubric-confirmed"):
        review.complete_review(**args)


def test_unrelated_measurements_cannot_activate(tmp_path, frozen):
    args = kwargs(
        tmp_path,
        frozen,
        activate=True,
        measurements=ROOT / "tests/fixtures/judge_measurements/measurements.json",
    )
    with pytest.raises(ValueError):
        review.complete_review(**args)
    assert not args["output"].exists()


def test_activation_exports_exact_candidates_after_validation(
    tmp_path, frozen, monkeypatch
):
    # Collection replay has separate integration coverage; isolate activation ordering.
    _, files = frozen
    measured = {"completeness": {"status": "complete"}, "trials": []}
    measurement_path = tmp_path / "measurements.json"
    measurement_path.write_bytes(ref.canonical_payload(measured))
    seen = []
    monkeypatch.setattr(
        review, "validate_measurements", lambda *a, **kw: seen.append((a, kw))
    )
    args = kwargs(tmp_path, frozen, activate=True, measurements=measurement_path)
    result = review.complete_review(**args)
    assert seen and result["activation"] == "activated"
    candidate = review.decode(files, "grounded_qa/final/candidate_plan.json")
    for name in ("plan", "analysis_policy"):
        assert ref.obj(args["output"] / f"{name}.json") == candidate[name]


def test_agreement_counts_all_repetitions_and_missing_without_inflating_answers():
    labels = [{"case_id": "c", "side": "baseline", "rating": "good"}]
    measured = {
        "trials": [
            {
                "case_id": "c",
                "side": "baseline",
                "status": "complete",
                "parse": {"rating": label},
            }
            for label in ("good", "good", "bad")
        ]
        + [{"case_id": "c", "side": "baseline", "status": "failed"}]
    }
    result = review.agreement(labels, measured)
    assert result["reviewed_answers"] == 1
    assert result["scheduled_trials"] == 4
    assert result["incomplete_trials"] == 1
    assert result["exact_agreement"] == {"numerator": 2, "denominator": 3}
    assert "not inter-rater reliability" in result["protocol"]["scope"]


def test_final_validation_cannot_activate(tmp_path, frozen):
    args = kwargs(tmp_path, frozen, activate=True)
    args["completed"].write_bytes(
        ref.canonical_payload(filled(frozen[1], "final_validation"))
    )
    with pytest.raises(ValueError, match="pilot review"):
        review.complete_review(**args)


def test_directory_snapshot_validates_every_pin(tmp_path, frozen):
    pin, files = frozen
    bundle = tmp_path / "reference"
    bundle.mkdir()
    for name, payload in files.items():
        path = bundle / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    snap = review.pinned_files(bundle, pin)
    assert snap["grounded_qa/selection.json"] == files["grounded_qa/selection.json"]
    (bundle / "grounded_qa/selection.json").write_text("{}")
    with pytest.raises(ValueError, match="digest or size mismatch"):
        review.pinned_files(bundle, pin)


@pytest.mark.parametrize("extra", ("file", "symlink"))
def test_directory_snapshot_rejects_unlisted_entries(tmp_path, frozen, extra):
    pin, files = frozen
    bundle = tmp_path / "reference"
    bundle.mkdir()
    for name, payload in files.items():
        path = bundle / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    unexpected = bundle / "unlisted"
    if extra == "file":
        unexpected.write_text("not retained", encoding="utf-8")
    else:
        unexpected.symlink_to(bundle / "reference.json")
    with pytest.raises(ValueError, match="unlisted|symlink|unexpected"):
        review.pinned_files(bundle, pin)


def test_measurement_loader_uses_the_full_measurement_contract_limit(
    tmp_path, monkeypatch
):
    seen = {}

    def read(_path, *, label, max_bytes):
        seen.update(label=label, max_bytes=max_bytes)
        return b"{}"

    monkeypatch.setattr(review, "read_regular_file_bytes", read)
    assert review.measurements_object(tmp_path / "measurements.json") == {}
    assert seen == {
        "label": "judge measurements",
        "max_bytes": review.MEASUREMENTS_MAX_BYTES,
    }
    assert review.MEASUREMENTS_MAX_BYTES == 384 * 1024 * 1024


def test_labels_are_written_before_orientation_is_revealed(
    tmp_path, frozen, monkeypatch
):
    args = kwargs(tmp_path, frozen)
    original = review.reconcile

    def inspect_order(sheet, selection):
        frozen_sheet = args["output"] / "completed-review.json"
        assert ref.obj(frozen_sheet) == sheet
        assert frozen_sheet.stat().st_mode & 0o777 == 0o600
        return original(sheet, selection)

    monkeypatch.setattr(review, "reconcile", inspect_order)
    review.complete_review(**args)


def test_no_agreement_denominator_is_reported_as_zero_not_perfect():
    result = review.agreement([], {"trials": []})
    assert result["exact_agreement"] == {"numerator": 0, "denominator": 0}
