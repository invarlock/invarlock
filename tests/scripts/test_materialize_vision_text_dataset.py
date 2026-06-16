from __future__ import annotations

import json
from pathlib import Path

from tests.scripts._support_model_evidence_sweep import load_script_module


class _FakeImage:
    def save(self, handle, *, format: str) -> None:
        handle.write(f"fake-image-{format}".encode())


def test_materialize_rows_writes_vision_text_manifest_and_summary(
    tmp_path: Path,
) -> None:
    mod = load_script_module("materialize_vision_text_dataset")
    config = mod.MaterializeConfig(
        dataset="public/vqa",
        split="validation",
        revision="abc123",
        config_name=None,
        image_field="image",
        prompt_field="question",
        answer_field="multiple_choice_answer",
        answers_field="answers",
        id_field="question_id",
        prompt_template="{question}\nAnswer with a short phrase.",
        max_samples=2,
        seed=42,
        shuffle=False,
        image_format="png",
    )
    rows = [
        {
            "question_id": 7,
            "question": " What color is the cup? ",
            "multiple_choice_answer": "red",
            "answers": ["red", {"answer": "Red"}, "scarlet"],
            "image": _FakeImage(),
        },
        {
            "question_id": 8,
            "question": "missing image",
            "multiple_choice_answer": "blue",
        },
    ]

    summary = mod.materialize_rows(rows, output_dir=tmp_path, config=config)

    manifest_path = tmp_path / "manifest.jsonl"
    records = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
    ]
    assert summary["record_count"] == 1
    assert summary["skipped_count"] == 1
    assert records[0]["id"] == "7"
    assert records[0]["prompt"] == (
        "What color is the cup?\nAnswer with a short phrase."
    )
    assert records[0]["answers"] == ["red", "scarlet"]
    assert (tmp_path / records[0]["image_path"]).is_file()

    summary_on_disk = json.loads(
        (tmp_path / "materialization_summary.json").read_text(encoding="utf-8")
    )
    assert summary_on_disk["dataset"] == "public/vqa"
    assert summary_on_disk["manifest"]["sha256"] == summary["manifest"]["sha256"]


def test_select_rows_uses_deterministic_shuffle(tmp_path: Path) -> None:
    del tmp_path
    mod = load_script_module("materialize_vision_text_dataset")
    config = mod.MaterializeConfig(
        dataset="public/vqa",
        split="validation",
        revision=None,
        config_name=None,
        image_field="image",
        prompt_field="question",
        answer_field="answer",
        answers_field=None,
        id_field="id",
        prompt_template="{question}",
        max_samples=3,
        seed=11,
        shuffle=True,
        image_format="png",
    )
    rows = [{"id": i} for i in range(8)]

    first = mod._select_rows(rows, config=config)
    second = mod._select_rows(rows, config=config)

    assert first == second
    assert len(first) == 3
    assert [row["id"] for row in first] != [0, 1, 2]
