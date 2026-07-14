from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import invarlock.eval.vision_evidence as vision_evidence_mod
from invarlock.eval.data import (
    VisionTextProvider,
    _normalize_answers,
    _resolve_image_path,
)
from invarlock.eval.vision_evidence import bind_loaded_record
from invarlock.vision_dataset_evidence import (
    canonical_json_bytes,
    dataset_record_digest,
    materialized_record_digest,
)
from tests.scripts._support_model_evidence import load_script_module


def _sha256_hex(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


class _FakeImage:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def save(self, handle, *, format: str) -> None:
        del format
        handle.write(self.payload)


def _materialize(
    tmp_path: Path,
    *,
    count: int = 3,
    dataset: str = "public/vision-test",
) -> tuple[Path, list[dict]]:
    mod = load_script_module("materialize_vision_text_dataset")
    config = mod.MaterializeConfig(
        dataset=dataset,
        split="validation",
        revision="a" * 40,
        config_name=None,
        image_field="image",
        prompt_field="question",
        answer_field="answer",
        answers_field=None,
        id_field="id",
        prompt_template="{question}",
        max_samples=count,
        seed=42,
        shuffle=False,
        image_format="png",
    )
    mod.materialize_rows(
        [
            {
                "id": f"img-{index:03d}",
                "question": f"prompt {index}",
                "answer": f"answer {index}",
                "image": _FakeImage(f"image-{index:03d}-bytes".encode()),
            }
            for index in range(1, count + 1)
        ],
        output_dir=tmp_path,
        config=config,
    )
    manifest = tmp_path / "manifest.jsonl"
    records = [
        json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()
    ]
    return manifest, records


def test_vision_text_provider_digest_and_schedule_stable(tmp_path):
    manifest, records = _materialize(tmp_path)
    pipeline = "resize-224-center-crop-normalize(mean=0.5,std=0.5)"

    p1 = VisionTextProvider(path=str(manifest), transform_pipeline=pipeline, seed=42)
    p2 = VisionTextProvider(path=str(manifest), transform_pipeline=pipeline, seed=42)

    # Pairing schedule must be a stable, sorted list of ids
    sched1 = p1.pairing_schedule()
    sched2 = p2.pairing_schedule()
    assert sched1 == ["img-001", "img-002", "img-003"]
    assert sched1 == sched2

    # Digest must be stable and include ids/image hashes and the transform pipeline
    d1 = p1.digest()
    d2 = p2.digest()
    assert d1 == d2
    assert d1["provider"] == "vision_text"
    assert d1["version"] >= 1
    assert d1["transform_pipeline"] == pipeline
    assert p1.dataset_name == "public/vision-test"
    assert p1.config_name is None
    assert p1.revision == "a" * 40
    # IDs use length-safe canonical JSON rather than ambiguous concatenation.
    assert d1["ids_sha256"] == _sha256_hex(
        canonical_json_bytes(["img-001", "img-002", "img-003"])
    )
    # images hash is sha256 over concatenated per-image hashes in schedule order
    per_img_hashes = b"".join(
        _sha256_hex((tmp_path / record["image_path"]).read_bytes()).encode()
        for record in records
    )
    assert d1["images_sha256"] == _sha256_hex(per_img_hashes)
    assert isinstance(d1["prompts_sha256"], str)
    assert isinstance(d1["answers_sha256"], str)

    # Changing the pipeline should change the digest
    p3 = VisionTextProvider(
        path=str(manifest),
        transform_pipeline=pipeline + "+brightness(0.1)",
        seed=42,
    )
    d3 = p3.digest()
    assert d3["transform_pipeline"] != d1["transform_pipeline"]


def test_vision_text_provider_handles_missing_bytes():
    items = [
        {"id": "img-100", "prompt": "prompt 1", "answer": "answer 1"},
        {
            "id": "img-200",
            "prompt": "prompt 2",
            "answer": "answer 2",
            "image_bytes": b"",
        },
    ]
    provider = VisionTextProvider(items=items)
    digest = provider.digest()
    assert "seed" not in digest
    # When bytes missing, sha256 of empty bytes is used
    empty_hash = _sha256_hex(b"").encode()
    combined = _sha256_hex(empty_hash + empty_hash)
    assert digest["images_sha256"] == combined


def test_vision_text_provider_rejects_mixed_materialization_identities(
    tmp_path: Path,
) -> None:
    first, _ = _materialize(tmp_path / "first", count=1)
    second, _ = _materialize(
        tmp_path / "second",
        count=1,
        dataset="other/vision-test",
    )

    provider = VisionTextProvider(data_files=[str(first), str(second)])

    with pytest.raises(Exception, match="different source dataset coordinates"):
        provider.examples()


def test_vision_text_provider_raises_for_missing_image(tmp_path):
    manifest, records = _materialize(tmp_path, count=1)
    (tmp_path / records[0]["image_path"]).unlink()

    provider = VisionTextProvider(path=str(manifest))

    with pytest.raises(Exception, match="image file is missing"):
        provider.examples()


def test_vision_text_provider_rejects_unbound_absolute_image_reference(
    tmp_path: Path,
) -> None:
    manifest, records = _materialize(tmp_path, count=1)
    image = tmp_path / records[0]["image_path"]
    records[0]["image_path"] = str(image)
    manifest.write_text(json.dumps(records[0]) + "\n", encoding="utf-8")

    with pytest.raises(Exception, match="manifest bytes do not match"):
        VisionTextProvider(path=str(manifest)).examples()


def test_vision_text_provider_batches_and_max_samples(tmp_path):
    manifest, records = _materialize(tmp_path)
    images = [tmp_path / record["image_path"] for record in records]

    provider = VisionTextProvider(path=str(manifest), max_samples=2)

    assert provider.available_splits() == ["validation"]
    assert len(provider.examples()) == 2
    assert provider.examples()[0]["image_path"] == str(images[0])
    assert provider.examples()[0]["image_ref"] == records[0]["image_path"]
    assert provider.examples()[0]["source_file"] == str(manifest)
    assert provider.examples()[0]["source_ref"] == manifest.name

    batches = list(provider.batches(seed=123, batch_size=2))
    assert batches == [
        {
            "records": [
                provider.examples()[0],
                provider.examples()[1],
            ]
        }
    ]


def test_vision_text_provider_items_override_caches_and_normalizes_bytes() -> None:
    provider = VisionTextProvider(
        items=[
            "ignore-me",
            {
                "prompt": "What?",
                "answers": [" cat ", ""],
                "image_bytes": bytearray(b"img-bytes"),
            },
        ]
    )

    examples = provider.examples()
    assert examples == provider.examples()
    assert examples[0]["id"] == "memory:2"
    assert examples[0]["answers"] == ["cat"]
    assert examples[0]["image_sha256"] == _sha256_hex(b"img-bytes")
    assert examples[0]["image_ref"] == f"sha256:{_sha256_hex(b'img-bytes')}"

    batches = list(provider.batches(seed=0, batch_size=0))
    assert batches == [examples[0]]


def test_vision_text_provider_answer_normalization_paths() -> None:
    assert _normalize_answers({"answers": [" cat ", ""]}) == ["cat"]
    assert _normalize_answers({"answers": [], "answer": " dog "}) == ["dog"]

    with pytest.raises(Exception, match="missing answer/answers"):
        _normalize_answers({})


def test_vision_text_provider_raises_for_empty_sources_and_bad_manifest_reads(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    with pytest.raises(Exception, match="produced no samples"):
        VisionTextProvider().examples()

    manifest, _ = _materialize(tmp_path, count=1)

    from invarlock import evidence_pack_json

    original_read = evidence_pack_json.read_regular_file_bytes

    def _broken_read(self, *args, **kwargs):  # noqa: ANN001
        if self == manifest:
            raise OSError("boom")
        return original_read(self, *args, **kwargs)

    monkeypatch.setattr(evidence_pack_json, "read_regular_file_bytes", _broken_read)

    with pytest.raises(Exception, match="vision_text manifest is invalid"):
        VisionTextProvider(path=str(manifest)).examples()


def test_vision_text_provider_manifest_validation_errors(tmp_path: Path) -> None:
    invalid_root = tmp_path / "invalid"
    invalid_root.mkdir()
    invalid_manifest, _ = _materialize(invalid_root, count=1)
    invalid_manifest.write_text("{bad json}\n", encoding="utf-8")
    with pytest.raises(Exception, match="vision_text manifest is invalid"):
        VisionTextProvider(path=str(invalid_manifest)).examples()

    prompt_root = tmp_path / "missing-prompt"
    prompt_root.mkdir()
    missing_prompt, prompt_records = _materialize(prompt_root, count=1)
    del prompt_records[0]["prompt"]
    missing_prompt.write_text(json.dumps(prompt_records[0]) + "\n", encoding="utf-8")
    with pytest.raises(Exception, match="manifest bytes do not match"):
        VisionTextProvider(path=str(missing_prompt)).examples()

    image_root = tmp_path / "missing-image-path"
    image_root.mkdir()
    missing_image_path, image_records = _materialize(image_root, count=1)
    del image_records[0]["image_path"]
    missing_image_path.write_text(json.dumps(image_records[0]) + "\n", encoding="utf-8")
    with pytest.raises(Exception, match="manifest bytes do not match"):
        VisionTextProvider(path=str(missing_image_path)).examples()

    empty_root = tmp_path / "empty"
    empty_root.mkdir()
    empty_manifest, _ = _materialize(empty_root, count=1)
    empty_manifest.write_text('\n"ignore"\n', encoding="utf-8")
    with pytest.raises(Exception, match="vision_text manifest is invalid"):
        VisionTextProvider(path=str(empty_manifest)).examples()


def test_vision_text_provider_rejects_duplicate_evidence_key(tmp_path: Path) -> None:
    manifest, _ = _materialize(tmp_path, count=1)
    evidence_path = tmp_path / "dataset_evidence.json"
    evidence = evidence_path.read_text(encoding="utf-8")
    evidence_path.write_text(
        evidence.replace(
            '"schema": "dataset_evidence.v1",',
            '"schema": "dataset_evidence.v1",\n  "schema": "dataset_evidence.v1",',
            1,
        ),
        encoding="utf-8",
    )

    with pytest.raises(Exception, match="dataset_evidence.json is invalid"):
        VisionTextProvider(path=str(manifest)).examples()


def test_vision_text_provider_rejects_duplicate_manifest_key(tmp_path: Path) -> None:
    manifest, _ = _materialize(tmp_path, count=1)
    line = manifest.read_text(encoding="utf-8")
    manifest.write_text(line.replace("{", '{"id":"forged",', 1), encoding="utf-8")

    with pytest.raises(Exception, match="vision_text manifest is invalid"):
        VisionTextProvider(path=str(manifest)).examples()


def test_vision_text_provider_resolve_path_accepts_absolute_and_items_short_circuit(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "absolute.ppm"
    image_path.write_text("P3\n1 1\n255\n0 0 0\n", encoding="utf-8")

    assert _resolve_image_path(str(image_path), base_dir=tmp_path) == image_path

    provider = VisionTextProvider(items=[{"prompt": "What?", "answer": "cat"}])
    assert provider._resolve_files() == []


def _bound_manifest_record() -> tuple[dict, str, dict[str, dict]]:
    answers = ["cat", "feline"]
    image_sha256 = "a" * 64
    prompt = "What animal is shown?"
    source = {
        "dataset": "public/vision-test",
        "revision": "b" * 40,
        "split": "validation",
        "row_index": 7,
        "question": prompt,
        "answer_sha256": hashlib.sha256(canonical_json_bytes(answers)).hexdigest(),
        "image_sha256": image_sha256,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
    }
    source["dataset_record_sha256"] = dataset_record_digest(
        dataset=source["dataset"],
        revision=source["revision"],
        split=source["split"],
        row_index=source["row_index"],
        record_id="img-007",
        question=source["question"],
        answers=answers,
    )
    record = {
        "id": "img-007",
        "prompt": prompt,
        "answer": answers[0],
        "answers": answers,
        "source": source,
    }
    source["record_sha256"] = materialized_record_digest(record)
    binding = {
        "img-007": {
            "image_sha256": image_sha256,
            "dataset_record_sha256": source["dataset_record_sha256"],
            "record_sha256": source["record_sha256"],
        }
    }
    return record, image_sha256, binding


def test_vision_materialization_binding_accepts_only_cross_bound_records() -> None:
    record, image_sha256, bindings = _bound_manifest_record()

    assert bind_loaded_record(
        record_id="img-007",
        raw_record=record,
        observed_image_sha256=image_sha256,
        materialization_digest="sha256:" + "c" * 64,
        manifest_sha256="sha256:" + "d" * 64,
        bindings=bindings,
    ) == {
        "dataset_record_sha256": bindings["img-007"]["dataset_record_sha256"],
        "materialization_digest": "sha256:" + "c" * 64,
        "manifest_sha256": "sha256:" + "d" * 64,
        "record_sha256": bindings["img-007"]["record_sha256"],
    }
    assert (
        bind_loaded_record(
            record_id="img-007",
            raw_record=record,
            observed_image_sha256=image_sha256,
            materialization_digest=None,
            manifest_sha256="ignored",
            bindings={},
        )
        == {}
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda record, binding: binding.clear(), "absent from dataset evidence"),
        (
            lambda record, binding: record.update(source=None),
            "answer_sha256 is invalid",
        ),
        (lambda record, binding: record.update(answers="cat"), "answers are not"),
        (lambda record, binding: record.update(answer="dog"), "primary answer"),
        (lambda record, binding: record.update(prompt=""), "prompt is not"),
        (
            lambda record, binding: record["source"].update(answer_sha256="bad"),
            "answer_sha256 is invalid",
        ),
        (
            lambda record, binding: record["source"].update(image_sha256="bad"),
            "image_sha256 is invalid",
        ),
        (
            lambda record, binding: record["source"].update(prompt_sha256="bad"),
            "prompt_sha256 is invalid",
        ),
        (
            lambda record, binding: record["source"].update(row_index=True),
            "row_index is not",
        ),
        (
            lambda record, binding: record["source"].update(
                dataset_record_sha256="bad"
            ),
            "dataset record digest is invalid",
        ),
        (
            lambda record, binding: record["source"].update(record_sha256="bad"),
            "materialized record digest is invalid",
        ),
        (
            lambda record, binding: binding["img-007"].update(image_sha256="bad"),
            "image bytes changed",
        ),
        (
            lambda record, binding: binding["img-007"].update(
                dataset_record_sha256="bad"
            ),
            "dataset_record_sha256 is not bound",
        ),
        (
            lambda record, binding: binding["img-007"].update(record_sha256="bad"),
            "record_sha256 is not bound",
        ),
    ],
)
def test_vision_materialization_binding_rejects_tampering(mutation, message) -> None:
    record, image_sha256, bindings = _bound_manifest_record()
    mutation(record, bindings)

    with pytest.raises(ValueError, match=message):
        bind_loaded_record(
            record_id="img-007",
            raw_record=record,
            observed_image_sha256=image_sha256,
            materialization_digest="sha256:" + "c" * 64,
            manifest_sha256="sha256:" + "d" * 64,
            bindings=bindings,
        )


def test_load_materialization_snapshot_rejects_semantic_and_order_mismatches(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manifest_bytes = b'{"id":"one"}\n'
    payload = {
        "dataset": {
            "id": "public/vision-test",
            "config_name": None,
            "revision": "a" * 40,
            "split": "validation",
        },
        "records": [{"id": "one"}],
        "semantic_digest": "sha256:semantic",
    }
    monkeypatch.setattr(
        vision_evidence_mod,
        "read_json_object_snapshot",
        lambda *_args, **_kwargs: (b"evidence", payload),
    )
    monkeypatch.setattr(
        vision_evidence_mod,
        "read_jsonl_snapshot",
        lambda *_args, **_kwargs: (manifest_bytes, [{"id": "one"}]),
    )
    monkeypatch.setattr(
        vision_evidence_mod, "validate_dataset_evidence", lambda *_args, **_kwargs: []
    )

    payload["manifest_sha256"] = vision_evidence_mod.sha256_prefixed(manifest_bytes)
    snapshot = vision_evidence_mod.load_materialization_snapshot(
        tmp_path / "manifest.jsonl"
    )
    assert snapshot.records == ({"id": "one"},)
    assert snapshot.bindings == {"one": {"id": "one"}}
    assert snapshot.dataset == payload["dataset"]

    monkeypatch.setattr(
        vision_evidence_mod,
        "validate_dataset_evidence",
        lambda *_args, **_kwargs: ["semantic digest mismatch"],
    )
    with pytest.raises(ValueError, match="semantic digest mismatch"):
        vision_evidence_mod.load_materialization_snapshot(tmp_path / "manifest.jsonl")

    monkeypatch.setattr(
        vision_evidence_mod, "validate_dataset_evidence", lambda *_args, **_kwargs: []
    )
    monkeypatch.setattr(
        vision_evidence_mod,
        "read_jsonl_snapshot",
        lambda *_args, **_kwargs: (manifest_bytes, ["not-an-object"]),
    )
    with pytest.raises(ValueError, match="records must be JSON objects"):
        vision_evidence_mod.load_materialization_snapshot(tmp_path / "manifest.jsonl")

    monkeypatch.setattr(
        vision_evidence_mod,
        "read_jsonl_snapshot",
        lambda *_args, **_kwargs: (b"changed", [{"id": "one"}]),
    )
    with pytest.raises(ValueError, match="manifest bytes do not match"):
        vision_evidence_mod.load_materialization_snapshot(tmp_path / "manifest.jsonl")

    changed_bytes = b'{"id":"two"}\n'
    payload["manifest_sha256"] = vision_evidence_mod.sha256_prefixed(changed_bytes)
    monkeypatch.setattr(
        vision_evidence_mod,
        "read_jsonl_snapshot",
        lambda *_args, **_kwargs: (changed_bytes, [{"id": "two"}]),
    )
    with pytest.raises(ValueError, match="record order does not match"):
        vision_evidence_mod.load_materialization_snapshot(tmp_path / "manifest.jsonl")
