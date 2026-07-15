from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from tests.scripts._support_model_evidence import load_script_module


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
    assert summary["selected_count"] == 2
    assert records[0]["id"] == "7"
    assert records[0]["prompt"] == (
        "What color is the cup?\nAnswer with a short phrase."
    )
    assert records[0]["answers"] == ["red", "scarlet"]
    assert (tmp_path / records[0]["image_path"]).is_file()

    summary_on_disk = json.loads(
        (tmp_path / "materialization_summary.json").read_text(encoding="utf-8")
    )
    assert summary_on_disk["schema"] == "dataset_evidence.v1"
    assert summary_on_disk["dataset"] == {
        "config_name": None,
        "id": "public/vqa",
        "revision": "abc123",
        "split": "validation",
    }
    assert summary_on_disk["records"][0]["dataset_record_sha256"].startswith("sha256:")
    assert summary_on_disk["records"][0]["record_sha256"].startswith("sha256:")
    assert (
        json.loads((tmp_path / "dataset_evidence.json").read_text(encoding="utf-8"))[
            "semantic_digest"
        ]
        == summary_on_disk["semantic_digest"]
    )
    assert summary_on_disk["manifest"]["sha256"] == summary["manifest"]["sha256"]
    assert (
        summary_on_disk["prompt_template_sha256"]
        == hashlib.sha256(config.prompt_template.encode("utf-8")).hexdigest()
    )


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


def test_record_ids_are_canonical_and_length_safe() -> None:
    mod = load_script_module("materialize_vision_text_dataset")

    assert mod._canonical_record_id(7, row_index=1) == "7"
    assert mod._canonical_record_id(None, row_index=3) == "row-3"
    long_id = "x" * 5000
    first = mod._canonical_record_id(long_id, row_index=4)
    second = mod._canonical_record_id(long_id, row_index=99)
    assert first == second
    assert first.startswith("source-id-sha256-")
    assert len(first.encode("utf-8")) < 1024


def test_hf_loader_forwards_exact_dataset_revision_and_config(
    monkeypatch,
) -> None:
    mod = load_script_module("materialize_vision_text_dataset")
    revision = "99487d2651df3799002b2fb3e455741744514a02"
    calls: list[dict[str, object]] = []

    def load_dataset(**kwargs):
        calls.append(kwargs)
        return [{"id": 1}]

    monkeypatch.setitem(
        sys.modules,
        "datasets",
        SimpleNamespace(load_dataset=load_dataset),
    )
    args = SimpleNamespace(
        dataset="public/vqa",
        split="validation",
        revision=revision,
        config_name="default",
        cache_dir="/cache",
    )

    assert mod._load_hf_dataset(args) == [{"id": 1}]
    assert calls == [
        {
            "path": "public/vqa",
            "split": "validation",
            "name": "default",
            "revision": revision,
            "cache_dir": "/cache",
        }
    ]
