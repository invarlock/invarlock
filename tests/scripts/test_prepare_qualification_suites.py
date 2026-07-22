from __future__ import annotations

import hashlib
import io
import json
import sys
from collections import Counter
from pathlib import Path
from types import ModuleType

import pytest
from PIL import Image

from invarlock.core.runtime_provider import build_runtime_behavioral_schedule
from scripts import prepare_qualification_suites as suites

ROOT = Path(__file__).resolve().parents[2]
PUBLIC_MANIFEST = ROOT / "docs" / "reference" / "qualification-suites.manifest.json"
PUBLIC_EVIDENCE_INDEX = ROOT / "public_evidence" / "evidence_index.json"


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


def _synthetic_mmlu_source() -> list[dict[str, object]]:
    rows = [
        {
            "question_id": 20_000 + (group * 40) + (answer_index * 4) + copy,
            "question": (
                f"Which choice identifies group {group}, answer {answer_index}, "
                f"copy {copy}?"
            ),
            "options": [f"choice {index}" for index in range(10)],
            "answer": chr(ord("A") + answer_index),
            "answer_index": answer_index,
            "category": f"group_{group:02d}",
            "src": "offline qualification fixture",
        }
        for group in range(10)
        for answer_index in range(10)
        for copy in range(4)
    ]
    rows.append(
        {
            "question_id": 12_209,
            "question": "This authenticated source record exceeds the runtime profile.",
            "options": [f"choice {index}" for index in range(10)],
            "answer": "A",
            "answer_index": 0,
            "category": "excluded",
            "src": "offline qualification fixture",
        }
    )
    return rows


def _synthetic_mmmu_source() -> list[dict[str, object]]:
    return [
        {
            "id": f"offline_{group:02d}_{answer_index}_{copy}",
            "image": {
                "bytes": _png_bytes(
                    color=(
                        ((group * 40) + (answer_index * 4) + copy) % 256,
                        (((group * 40) + (answer_index * 4) + copy) // 256) % 256,
                        127,
                    )
                ),
                "path": None,
            },
            "options": repr([f"choice {index}" for index in range(10)]),
            "answer": chr(ord("A") + answer_index),
            "subject": f"group_{group:02d}",
        }
        for group in range(10)
        for answer_index in range(10)
        for copy in range(4)
    ]


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


@pytest.mark.parametrize("value", [None, "", " padded "])
def test_required_text_rejects_missing_empty_or_untrimmed_values(value: object) -> None:
    with pytest.raises(suites.QualificationSuiteError, match="trimmed text"):
        suites._required_text(value, label="fixture")


def test_source_text_rejects_non_text_and_normalizes_outer_whitespace() -> None:
    with pytest.raises(suites.QualificationSuiteError, match="must be text"):
        suites._source_text(7, label="fixture")

    assert suites._source_text("  retained content  ", label="fixture") == (
        "retained content"
    )


def test_revision_and_positive_integer_contracts_fail_closed() -> None:
    with pytest.raises(suites.QualificationSuiteError, match="40-character revision"):
        suites._required_revision("main", label="fixture")
    for value in (True, "1", 0):
        with pytest.raises(suites.QualificationSuiteError, match="positive integer"):
            suites._required_positive_int(value, label="fixture")


@pytest.mark.parametrize(
    ("values", "count"),
    [([], 0), (["A", "B"], 1)],
)
def test_balanced_quotas_require_every_stratum(values: list[str], count: int) -> None:
    with pytest.raises(suites.QualificationSuiteError, match="every stratum"):
        suites._balanced_quotas(values, count)


def test_stratified_selection_rejects_impossible_balanced_assignment() -> None:
    cells = [("group_1", "A"), ("group_1", "A"), ("group_1", "A"), ("group_2", "B")]
    rows = [
        {
            "id": f"item_{index}",
            "group": group,
            "answer": answer,
            "semantic_sha256": hashlib.sha256(f"semantic:{index}".encode()).hexdigest(),
        }
        for index, (group, answer) in enumerate(cells)
    ]

    with pytest.raises(suites.QualificationSuiteError, match="cannot satisfy balanced"):
        suites.select_stratified(rows, count=4, seed="fixed")


def test_stratified_selection_rejects_short_input_and_duplicate_ids() -> None:
    with pytest.raises(suites.QualificationSuiteError, match="1 usable rows"):
        suites.select_stratified(_selection_rows()[:1], count=2, seed="fixed")

    rows = _selection_rows()[:2]
    rows[1]["id"] = rows[0]["id"]
    with pytest.raises(suites.QualificationSuiteError, match="duplicate record id"):
        suites.select_stratified(rows, count=2, seed="fixed")


@pytest.mark.parametrize("options", ["not valid Python", ["only one"]])
def test_option_normalization_rejects_invalid_sources(options: object) -> None:
    with pytest.raises(suites.QualificationSuiteError, match="options"):
        suites._normalize_options(options, label="fixture")


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


def test_normalize_mmlu_rows_rejects_non_integer_question_id() -> None:
    with pytest.raises(suites.QualificationSuiteError, match="question_id is invalid"):
        suites.normalize_mmlu_rows([{"question_id": True}])


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


def test_image_payload_supports_an_explicit_path_and_rejects_bad_bindings(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "fixture.png"
    payload = _png_bytes(color=(12, 34, 56))
    image_path.write_bytes(payload)

    assert (
        suites._image_payload(
            {"bytes": None, "path": str(image_path)}, identifier="fixture"
        )
        == payload
    )
    for binding, message in (
        ({"bytes": payload}, "binding"),
        ({"bytes": None, "path": 7}, "path"),
        ({"bytes": bytearray(payload), "path": None}, "bytes"),
        ({"bytes": b"", "path": None}, "bytes"),
    ):
        with pytest.raises(suites.QualificationSuiteError, match=message):
            suites._image_payload(binding, identifier="fixture")


def test_image_metadata_rejects_undecodable_bytes() -> None:
    with pytest.raises(suites.QualificationSuiteError, match="cannot be decoded"):
        suites._image_metadata(b"not an image", identifier="fixture")


def test_mmmu_rejects_an_answer_outside_the_available_choices() -> None:
    with pytest.raises(suites.QualificationSuiteError, match="answer binding"):
        suites.normalize_mmmu_rows(
            [
                {
                    "id": "test_bad_answer",
                    "image": {
                        "bytes": _png_bytes(color=(1, 2, 3)),
                        "path": None,
                    },
                    "options": ["first", "second"],
                    "answer": "C",
                    "subject": "Art",
                }
            ]
        )


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


@pytest.mark.parametrize(
    ("limit_name", "expected_reason"),
    [
        ("MAX_IMAGE_BYTES", "image_byte_limit"),
        ("MAX_IMAGE_PIXELS", "image_pixel_limit"),
    ],
)
def test_mmmu_excludes_images_outside_per_record_resource_limits(
    monkeypatch: pytest.MonkeyPatch,
    limit_name: str,
    expected_reason: str,
) -> None:
    payload = _png_bytes(color=(4, 5, 6))
    exclusions: dict[str, int] = {}
    monkeypatch.setattr(suites, limit_name, 1)

    rows = suites.normalize_mmmu_rows(
        [
            {
                "id": f"test_{expected_reason}",
                "image": {"bytes": payload, "path": None},
                "options": ["first", "second"],
                "answer": "A",
                "subject": "Art",
            }
        ],
        exclusions=exclusions,
    )

    assert rows == []
    assert exclusions == {expected_reason: 1}


@pytest.mark.parametrize(
    ("rendering", "prompt_marker", "expected"),
    [
        ("raw_causal", "Answer:", " A"),
        ("mistral_instruct", "[INST]", "A"),
        ("qwen_instruct", "<|im_start|>assistant", "A"),
    ],
)
def test_text_rendering_profiles_are_explicit(
    rendering: str, prompt_marker: str, expected: str
) -> None:
    rows = [
        {
            "id": "fixture",
            "question": "Choose one.",
            "options": ["first", "second"],
            "answer": "A",
        }
    ]

    rendered = suites.render_text_records(rows, rendering=rendering)

    assert prompt_marker in rendered[0]["prompt"]
    assert rendered[0]["expected"] == expected


def test_text_rendering_rejects_unknown_profiles() -> None:
    with pytest.raises(suites.QualificationSuiteError, match="unsupported text"):
        suites.render_text_records(
            [
                {
                    "id": "fixture",
                    "question": "Choose one.",
                    "options": ["first", "second"],
                    "answer": "A",
                }
            ],
            rendering="implicit-template",
        )


def test_hosted_loader_pins_both_sources_and_disables_image_decoding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    cast_calls: list[tuple[str, object]] = []
    fake_module = ModuleType("datasets")

    class FakeImage:
        def __init__(self, *, decode: bool) -> None:
            self.decode = decode

    class FakeVisionDataset(list[object]):
        def cast_column(self, name: str, feature: object) -> FakeVisionDataset:
            cast_calls.append((name, feature))
            return self

    text_dataset: list[object] = []
    vision_dataset = FakeVisionDataset()

    def fake_load_dataset(*args: object, **kwargs: object) -> object:
        calls.append((args, kwargs))
        return text_dataset if args[0] == suites.MMLU_DATASET else vision_dataset

    fake_module.Image = FakeImage  # type: ignore[attr-defined]
    fake_module.load_dataset = fake_load_dataset  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "datasets", fake_module)

    loaded_text, loaded_vision = suites._load_hosted_datasets()

    assert loaded_text is text_dataset
    assert loaded_vision is vision_dataset
    assert calls == [
        (
            (suites.MMLU_DATASET,),
            {"revision": suites.MMLU_REVISION, "split": "test"},
        ),
        (
            (suites.MMMU_DATASET, suites.MMMU_CONFIG),
            {"revision": suites.MMMU_REVISION, "split": "test"},
        ),
    ]
    assert len(cast_calls) == 1
    assert cast_calls[0][0] == "image"
    assert isinstance(cast_calls[0][1], FakeImage)
    assert cast_calls[0][1].decode is False


def _normalized_resource_limit_rows() -> tuple[
    list[dict[str, object]], list[dict[str, object]]
]:
    text = _selection_rows()
    multimodal = [
        {
            **row,
            "upstream_id": f"upstream_{index}",
            "image_bytes": b"xx",
            "image_byte_length": 2,
            "image_sha256": hashlib.sha256(f"image:{index}".encode()).hexdigest(),
            "image_media_type": "image/png",
            "image_width": 2,
            "image_height": 2,
        }
        for index, row in enumerate(_selection_rows())
    ]
    return text, multimodal


@pytest.mark.parametrize(
    ("limit_name", "limit", "message"),
    [
        ("MAX_TOTAL_IMAGE_BYTES", 799, "byte limit"),
        ("MAX_TOTAL_IMAGE_PIXELS", 1_599, "pixel limit"),
    ],
)
def test_prepare_suites_rejects_aggregate_vision_resource_overruns(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    limit_name: str,
    limit: int,
    message: str,
) -> None:
    text, multimodal = _normalized_resource_limit_rows()
    monkeypatch.setattr(suites, "_load_hosted_datasets", lambda: ([], []))
    monkeypatch.setattr(suites, "normalize_mmlu_rows", lambda _rows: text)
    monkeypatch.setattr(
        suites,
        "apply_text_runtime_profile",
        lambda rows: (list(rows), {}),
    )
    monkeypatch.setattr(
        suites,
        "normalize_mmmu_rows",
        lambda _rows, *, exclusions: multimodal,
    )
    monkeypatch.setattr(suites, limit_name, limit)
    output = tmp_path / "suite"

    with pytest.raises(suites.QualificationSuiteError, match=message):
        suites.prepare_suites(output=output, record_count=400)

    assert not output.exists()


def test_prepare_suites_emits_a_complete_offline_400_record_transaction(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    text_source = _synthetic_mmlu_source()
    multimodal_source = _synthetic_mmmu_source()
    monkeypatch.setattr(
        suites,
        "_load_hosted_datasets",
        lambda: (text_source, multimodal_source),
    )
    output = tmp_path / "qualification"

    manifest = suites.prepare_suites(output=output, record_count=400)

    manifest_path = output / "qualification-suites.manifest.json"
    stored_manifest = json.loads(manifest_path.read_bytes())
    assert (
        manifest["manifest_sha256"]
        == hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    )
    assert stored_manifest["record_count"] == 400
    assert stored_manifest["sources"]["text"]["source_record_count"] == 401
    assert stored_manifest["sources"]["text"]["eligible_record_count"] == 400
    assert stored_manifest["sources"]["text"]["exclusions"] == {
        "prompt_exceeds_maximum_input_tokens": 1
    }
    assert stored_manifest["sources"]["multimodal"]["source_record_count"] == 400
    assert stored_manifest["sources"]["multimodal"]["eligible_record_count"] == 400
    assert stored_manifest["distributions"]["text_answers"] == dict.fromkeys(
        "ABCDEFGHIJ", 40
    )
    assert stored_manifest["distributions"]["multimodal_answers"] == dict.fromkeys(
        "ABCDEFGHIJ", 40
    )
    assert len(stored_manifest["selected_ids"]["text"]) == 400
    assert len(stored_manifest["selected_ids"]["multimodal"]) == 400

    for rendering in ("raw-causal", "mistral-instruct", "qwen-instruct"):
        records_path = output / "text" / f"{rendering}.jsonl"
        schedule_path = output / "text" / f"{rendering}.schedule.json"
        assert len(records_path.read_bytes().splitlines()) == 400
        assert len(json.loads(schedule_path.read_bytes())["records"]) == 400
    multimodal_path = output / "multimodal" / "mmmu-pro-vision.jsonl"
    multimodal_schedule_path = output / "multimodal" / "mmmu-pro-vision.schedule.json"
    assert len(multimodal_path.read_bytes().splitlines()) == 400
    assert len(json.loads(multimodal_schedule_path.read_bytes())["records"]) == 400
    assert len(list((output / "multimodal" / "content-store").iterdir())) == 400
    assert (
        stored_manifest["artifacts"]["multimodal"]["content_store"]["file_count"] == 400
    )


def test_main_resolves_output_and_prints_canonical_manifest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured: dict[str, object] = {}

    def fake_prepare_suites(*, output: Path, record_count: int) -> dict[str, object]:
        captured.update(output=output, record_count=record_count)
        return {"z": 2, "a": 1}

    relative_output = tmp_path / ".." / tmp_path.name / "suite"
    monkeypatch.setattr(suites, "prepare_suites", fake_prepare_suites)
    monkeypatch.setattr(
        sys,
        "argv",
        ["prepare_qualification_suites.py", "--output", str(relative_output)],
    )

    assert suites.main() == 0
    assert captured == {
        "output": relative_output.absolute(),
        "record_count": suites.DEFAULT_RECORD_COUNT,
    }
    assert capsys.readouterr().out == '{"a":1,"z":2}\n'


def test_qualification_count_is_not_operator_downgradable(tmp_path: Path) -> None:
    with pytest.raises(suites.QualificationSuiteError, match="exactly 400"):
        suites.prepare_suites(output=tmp_path / "suite", record_count=399)


def test_checked_in_qualification_manifest_matches_the_pinned_contract() -> None:
    payload = PUBLIC_MANIFEST.read_bytes()
    manifest = json.loads(payload)

    assert hashlib.sha256(payload).hexdigest() == (
        "1cb979170d16328b02b69b32d0ab9670365064ba3c112eed001515c549334d44"
    )
    assert manifest["format_version"] == suites.FORMAT_VERSION
    assert manifest["selection_algorithm"] == suites.SELECTION_ALGORITHM
    assert manifest["record_count"] == suites.DEFAULT_RECORD_COUNT
    assert manifest["sources"]["text"]["dataset"] == suites.MMLU_DATASET
    assert manifest["sources"]["text"]["revision"] == suites.MMLU_REVISION
    assert manifest["sources"]["multimodal"]["dataset"] == suites.MMMU_DATASET
    assert manifest["sources"]["multimodal"]["revision"] == suites.MMMU_REVISION
    assert set(manifest["distributions"]["text_groups"].values()) == {28, 29}
    assert set(manifest["distributions"]["multimodal_groups"].values()) == {13, 14}
    assert manifest["distributions"]["text_answers"] == dict.fromkeys("ABCDEFGHIJ", 40)
    assert manifest["distributions"]["multimodal_answers"] == dict.fromkeys(
        "ABCDEFGHIJ", 40
    )
    assert len(manifest["selected_ids"]["text"]) == 400
    assert len(manifest["selected_ids"]["multimodal"]) == 400


def test_public_evidence_is_bound_to_a_qualified_400_record_suite() -> None:
    manifest = json.loads(PUBLIC_MANIFEST.read_bytes())
    index = json.loads(PUBLIC_EVIDENCE_INDEX.read_bytes())
    qualified_suites = {
        manifest["artifacts"]["text_raw_causal"]["sha256"]: {
            "record_ids": manifest["selected_ids"]["text"],
            "schedule_sha256": manifest["artifacts"]["text_raw_causal"][
                "schedule_sha256"
            ],
            "task": "text_causal",
        },
        manifest["artifacts"]["multimodal"]["sha256"]: {
            "record_ids": None,
            "schedule_sha256": manifest["artifacts"]["multimodal"]["schedule_sha256"],
            "task": "vision_text_generation",
        },
    }

    assert index["evidence_count"] == len(index["entries"])
    assert index["entries"]
    for entry in index["entries"]:
        pack = ROOT / entry["path"] / "evidence"
        schedule = json.loads(
            (pack / "schedule" / "runtime-behavioral-schedule.json").read_bytes()
        )
        paired_records = json.loads(
            (pack / "records" / "paired-records.json").read_bytes()
        )
        report = json.loads((pack / "reports" / "evaluation.report.json").read_bytes())

        suite = qualified_suites[schedule["dataset_identity"]["revision"]]
        canonical_schedule = build_runtime_behavioral_schedule(schedule)
        assert schedule["task"] == suite["task"]
        assert len(schedule["records"]) == manifest["record_count"]
        assert canonical_schedule.schedule_sha256 == suite["schedule_sha256"]
        assert paired_records["schedule_sha256"] == canonical_schedule.schedule_sha256
        if suite["record_ids"] is not None:
            assert [record["record_id"] for record in schedule["records"]] == suite[
                "record_ids"
            ]

        qualification = report["sample_qualification"]
        assert report["record_count"] == manifest["record_count"]
        assert qualification["passed"] is True
        assert qualification["record_count"] == {
            "minimum": manifest["record_count"],
            "observed": manifest["record_count"],
            "passed": True,
        }
