from __future__ import annotations

import hashlib
import io
import json
from collections import Counter
from pathlib import Path

import pytest
from PIL import Image

from scripts import prepare_qualification_suites as suites


def _selection_rows() -> list[dict[str, object]]:
    return [
        {
            "id": f"item_{group:02d}_{answer}_{copy}",
            "group": f"group_{group:02d}",
            "answer": answer,
            "semantic_sha256": hashlib.sha256(
                f"semantic:{group}:{answer}:{copy}".encode()
            ).hexdigest(),
        }
        for group in range(14)
        for answer in "ABCDEFGHIJ"
        for copy in range(4)
    ]


def _png_bytes(*, color: tuple[int, int, int]) -> bytes:
    output = io.BytesIO()
    image = Image.new("RGB", (4, 3), color=color)
    image.save(output, format="PNG")
    image.close()
    return output.getvalue()


def test_stratified_selection_is_balanced_unique_and_deterministic() -> None:
    rows = _selection_rows()

    selected = suites.select_stratified(rows, count=400, seed="fixed")

    assert selected == suites.select_stratified(rows, count=400, seed="fixed")
    assert len(selected) == 400
    assert len({row["id"] for row in selected}) == 400
    assert set(Counter(str(row["group"]) for row in selected).values()) == {28, 29}
    assert Counter(str(row["answer"]) for row in selected) == dict.fromkeys(
        "ABCDEFGHIJ", 40
    )


def test_stratified_selection_rejects_duplicate_semantics() -> None:
    rows = _selection_rows()
    rows[1]["semantic_sha256"] = rows[0]["semantic_sha256"]

    with pytest.raises(suites.QualificationSuiteError, match="duplicate semantic"):
        suites.select_stratified(rows, count=400, seed="fixed")


def test_normalize_mmlu_rows_authenticates_answer_index() -> None:
    row = {
        "question_id": 7,
        "question": " Which option is correct? ",
        "options": ["first", "second"],
        "answer": "B",
        "answer_index": 1,
        "category": "logic",
        "src": "fixture",
    }

    normalized = suites.normalize_mmlu_rows([row])

    assert normalized[0]["id"] == "mmlu_pro_00007"
    assert normalized[0]["answer"] == "B"
    assert normalized[0]["question"] == "Which option is correct?"
    row["answer_index"] = 0
    with pytest.raises(suites.QualificationSuiteError, match="answer binding"):
        suites.normalize_mmlu_rows([row])


def test_text_runtime_profile_excludes_only_authenticated_ineligible_record() -> None:
    rows = [
        {"id": "mmlu_pro_00001"},
        {"id": "mmlu_pro_12209"},
    ]

    eligible, exclusions = suites.apply_text_runtime_profile(rows)

    assert eligible == [{"id": "mmlu_pro_00001"}]
    assert exclusions == {"prompt_exceeds_maximum_input_tokens": 1}


def test_text_runtime_profile_rejects_source_revision_drift() -> None:
    with pytest.raises(suites.QualificationSuiteError, match="missing source records"):
        suites.apply_text_runtime_profile([{"id": "mmlu_pro_00001"}])


def test_rendered_text_jsonl_round_trips_through_public_schedule(
    tmp_path: Path,
) -> None:
    rows = suites.normalize_mmlu_rows(
        [
            {
                "question_id": 11,
                "question": "Choose the second option.",
                "options": ["first", "second"],
                "answer": "B",
                "answer_index": 1,
                "category": "logic",
                "src": "fixture",
            }
        ]
    )
    rendered = suites.render_text_records(rows, rendering="raw_causal")
    payload = suites._jsonl_bytes(rendered)
    source = tmp_path / "records.jsonl"

    schedule_bytes, schedule_digest = suites._validated_schedule(
        source_path=source,
        source_bytes=payload,
        name="fixture",
        task="text_causal",
    )

    schedule = json.loads(schedule_bytes)
    assert len(schedule["records"]) == 1
    assert schedule["records"][0]["expected_output"] == " B"
    assert schedule_digest == hashlib.sha256(schedule_bytes).hexdigest()


def test_normalize_mmmu_rows_binds_original_image_bytes() -> None:
    payload = _png_bytes(color=(255, 0, 0))
    rows = suites.normalize_mmmu_rows(
        [
            {
                "id": "test_1",
                "image": {"bytes": payload, "path": "test_1.png"},
                "options": "['red', 'blue']",
                "answer": "A",
                "subject": "Art",
            }
        ]
    )

    assert rows[0]["image_sha256"] == hashlib.sha256(payload).hexdigest()
    assert rows[0]["image_byte_length"] == len(payload)
    assert rows[0]["image_media_type"] == "image/png"
    assert rows[0]["image_width"] == 4
    assert rows[0]["image_height"] == 3


def test_mmmu_duplicate_semantics_are_not_independent_samples() -> None:
    payload = _png_bytes(color=(0, 0, 255))
    source_rows = [
        {
            "id": f"test_{index}",
            "image": {"bytes": payload, "path": f"test_{index}.png"},
            "options": "['red', 'blue']",
            "answer": "B",
            "subject": "Art",
        }
        for index in range(2)
    ]
    rows = suites.normalize_mmmu_rows(source_rows)

    with pytest.raises(suites.QualificationSuiteError, match="duplicate semantic"):
        suites.select_stratified(rows, count=2, seed="fixed")


def test_mmmu_unsupported_media_is_excluded_without_transcoding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _png_bytes(color=(0, 255, 0))
    exclusions: dict[str, int] = {}
    monkeypatch.setattr(
        suites, "_image_metadata", lambda _payload, identifier: ("image/mpo", 4, 3)
    )

    rows = suites.normalize_mmmu_rows(
        [
            {
                "id": "test_unsupported",
                "image": {"bytes": payload, "path": "misnamed.png"},
                "options": "['red', 'green']",
                "answer": "B",
                "subject": "Art",
            }
        ],
        exclusions=exclusions,
    )

    assert rows == []
    assert exclusions == {"unsupported_media_type": 1}


def test_mmmu_more_than_ten_choices_is_excluded_from_a_to_j_profile() -> None:
    payload = _png_bytes(color=(0, 255, 0))
    exclusions: dict[str, int] = {}

    rows = suites.normalize_mmmu_rows(
        [
            {
                "id": "test_twelve_choices",
                "image": {"bytes": payload, "path": "twelve.png"},
                "options": repr([f"option {index}" for index in range(12)]),
                "answer": "F",
                "subject": "Computer_Science",
            }
        ],
        exclusions=exclusions,
    )

    assert rows == []
    assert exclusions == {"option_count_limit": 1}


def test_qualification_count_is_not_operator_downgradable(tmp_path: Path) -> None:
    with pytest.raises(suites.QualificationSuiteError, match="exactly 400"):
        suites.prepare_suites(output=tmp_path / "suite", record_count=399)
