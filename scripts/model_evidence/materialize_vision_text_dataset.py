#!/usr/bin/env python3
"""Materialize a public HF image-text dataset as a local vision_text manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from typing import Any

from invarlock.evidence_pack_json import read_jsonl_snapshot, sha256_prefixed
from invarlock.vision_dataset_evidence import (
    build_materialization_evidence,
    canonical_json_bytes,
    dataset_record_digest,
    materialized_record_digest,
)


@dataclass(frozen=True)
class MaterializeConfig:
    dataset: str
    split: str
    revision: str | None
    config_name: str | None
    image_field: str
    prompt_field: str
    answer_field: str | None
    answers_field: str | None
    id_field: str | None
    prompt_template: str
    max_samples: int
    seed: int
    shuffle: bool
    image_format: str


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download/materialize a public Hugging Face image-text dataset into "
            "the local JSONL schema consumed by the vision_text provider."
        )
    )
    parser.add_argument("--dataset", required=True, help="Hugging Face dataset id.")
    parser.add_argument("--split", default="validation", help="Dataset split.")
    parser.add_argument("--revision", default=None, help="Pinned dataset revision.")
    parser.add_argument("--config-name", default=None, help="Optional dataset config.")
    parser.add_argument("--output-dir", required=True, help="Destination directory.")
    parser.add_argument("--image-field", default="image")
    parser.add_argument("--prompt-field", default="question")
    parser.add_argument("--answer-field", default="multiple_choice_answer")
    parser.add_argument("--answers-field", default="answers")
    parser.add_argument("--id-field", default="question_id")
    parser.add_argument(
        "--prompt-template",
        default=(
            "{question}\n"
            'Return exactly one JSON object like {{"answer":"short phrase"}}. '
            "Use a short phrase only. Do not explain."
        ),
        help="Template used to build prompts from the prompt field.",
    )
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--image-format", choices=("png", "jpeg"), default="png")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output directory before materializing.",
    )
    return parser.parse_args(argv)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _safe_slug(value: object, *, fallback: str) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9._-]+", "-", text)
    text = text.strip("-._")
    return text[:80] or fallback


def _field_value(row: Mapping[str, Any], field_name: str | None) -> Any:
    if not field_name:
        return None
    current: Any = row
    for part in field_name.split("."):
        if isinstance(current, Mapping) and part in current:
            current = current[part]
        else:
            return None
    return current


def _answer_strings(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        candidates: list[Any] = [value]
    elif isinstance(value, Mapping):
        candidates = [value.get("answer") or value.get("text") or value.get("label")]
    elif isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray)):
        candidates = []
        for item in value:
            if isinstance(item, Mapping):
                candidates.append(
                    item.get("answer") or item.get("text") or item.get("label")
                )
            else:
                candidates.append(item)
    else:
        candidates = [value]

    answers: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        text = " ".join(str(candidate or "").strip().split())
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        answers.append(text)
        seen.add(key)
    return answers


def _format_prompt(template: str, *, question: str) -> str:
    try:
        prompt = template.format(question=question)
    except (KeyError, IndexError, ValueError):
        prompt = question
    return prompt.strip()


def _image_to_bytes(image_value: Any, *, image_format: str) -> bytes:
    format_name = "JPEG" if image_format == "jpeg" else "PNG"
    if isinstance(image_value, (bytes, bytearray)):
        return bytes(image_value)
    if isinstance(image_value, (str, Path)):
        return Path(image_value).expanduser().read_bytes()
    if isinstance(image_value, Mapping):
        raw_bytes = image_value.get("bytes")
        if isinstance(raw_bytes, (bytes, bytearray)):
            return bytes(raw_bytes)
        path = image_value.get("path")
        if isinstance(path, (str, Path)) and str(path):
            return Path(path).expanduser().read_bytes()
    save = getattr(image_value, "save", None)
    if callable(save):
        image = image_value
        if image_format == "jpeg":
            convert = getattr(image, "convert", None)
            if callable(convert):
                image = convert("RGB")
        buffer = BytesIO()
        image.save(buffer, format=format_name)
        return buffer.getvalue()
    raise TypeError(f"Unsupported image value type: {type(image_value).__name__}")


def _write_image(
    image_value: Any,
    *,
    images_dir: Path,
    index: int,
    record_id: str,
    image_format: str,
) -> tuple[Path, str, int]:
    image_bytes = _image_to_bytes(image_value, image_format=image_format)
    extension = "jpg" if image_format == "jpeg" else "png"
    image_name = f"{index:06d}-{_safe_slug(record_id, fallback=str(index))}.{extension}"
    image_path = images_dir / image_name
    image_path.write_bytes(image_bytes)
    return image_path, _sha256_bytes(image_bytes), len(image_bytes)


def _canonical_record_id(raw_id: Any, *, row_index: int) -> str:
    if raw_id is None or raw_id == "":
        return f"row-{row_index}"
    rendered = str(raw_id)
    if (
        rendered
        and rendered == rendered.strip()
        and len(rendered.encode("utf-8")) <= 1024
        and all(ord(character) >= 32 for character in rendered)
    ):
        return rendered
    try:
        identity_bytes = canonical_json_bytes(raw_id)
    except (TypeError, ValueError):
        identity_bytes = canonical_json_bytes(rendered)
    return "source-id-sha256-" + hashlib.sha256(identity_bytes).hexdigest()


def _record_from_row(
    row: Mapping[str, Any],
    *,
    index: int,
    config: MaterializeConfig,
    images_dir: Path,
) -> dict[str, Any] | None:
    question_value = _field_value(row, config.prompt_field)
    question = " ".join(str(question_value or "").strip().split())
    if not question:
        return None

    answers = _answer_strings(_field_value(row, config.answer_field))
    answers.extend(_answer_strings(_field_value(row, config.answers_field)))
    deduped_answers: list[str] = []
    seen_answers: set[str] = set()
    for answer in answers:
        key = answer.lower()
        if key in seen_answers:
            continue
        deduped_answers.append(answer)
        seen_answers.add(key)
    if not deduped_answers:
        return None

    image_value = _field_value(row, config.image_field)
    if image_value is None:
        return None

    raw_id = _field_value(row, config.id_field) if config.id_field else None
    record_id = _canonical_record_id(raw_id, row_index=index)
    image_path, image_sha256, image_bytes = _write_image(
        image_value,
        images_dir=images_dir,
        index=index,
        record_id=record_id,
        image_format=config.image_format,
    )
    prompt = _format_prompt(config.prompt_template, question=question)
    prompt_sha256 = _sha256_bytes(prompt.encode("utf-8"))
    answer_sha256 = _sha256_bytes(
        json.dumps(
            deduped_answers,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    dataset_record_sha256 = dataset_record_digest(
        dataset=config.dataset,
        revision=config.revision,
        split=config.split,
        row_index=index,
        record_id=record_id,
        question=question,
        answers=deduped_answers,
    )
    record: dict[str, Any] = {
        "id": record_id,
        "image_path": image_path.relative_to(images_dir.parent).as_posix(),
        "prompt": prompt,
        "answer": deduped_answers[0],
        "answers": deduped_answers,
        "source": {
            "dataset": config.dataset,
            "split": config.split,
            "revision": config.revision,
            "row_index": index,
            "question": question,
            "image_sha256": image_sha256,
            "image_bytes": image_bytes,
            "prompt_sha256": prompt_sha256,
            "answer_sha256": answer_sha256,
            "dataset_record_sha256": dataset_record_sha256,
        },
    }
    record["source"]["record_sha256"] = materialized_record_digest(record)
    return record


def _select_rows(dataset: Any, *, config: MaterializeConfig) -> list[Mapping[str, Any]]:
    total = len(dataset)
    if total <= 0:
        return []
    indices = list(range(total))
    if config.shuffle:
        import random

        random.Random(config.seed).shuffle(indices)
    if config.max_samples > 0:
        indices = indices[: config.max_samples]
    rows: list[Mapping[str, Any]] = []
    for index in indices:
        row = dataset[index]
        if isinstance(row, Mapping):
            rows.append(row)
    return rows


def materialize_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    output_dir: Path,
    config: MaterializeConfig,
) -> dict[str, Any]:
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.jsonl"
    records: list[dict[str, Any]] = []
    skipped = 0

    with manifest_path.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows):
            record = _record_from_row(
                row,
                index=index,
                config=config,
                images_dir=images_dir,
            )
            if record is None:
                skipped += 1
                continue
            handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")
            records.append(record)

    if not records:
        raise RuntimeError("No usable vision_text records were materialized.")

    manifest_bytes, parsed_records = read_jsonl_snapshot(
        manifest_path, label="vision-text materialized manifest"
    )
    if parsed_records != records:
        raise RuntimeError("materialized manifest bytes do not match selected records")
    manifest_sha256 = sha256_prefixed(manifest_bytes)
    image_hashes = [
        str(record.get("source", {}).get("image_sha256", "")) for record in records
    ]
    prompt_hashes = [
        _sha256_bytes(str(record.get("prompt", "")).encode("utf-8"))
        for record in records
    ]
    answer_hashes = [
        _sha256_bytes(
            json.dumps(record.get("answers", []), ensure_ascii=True).encode("utf-8")
        )
        for record in records
    ]
    evidence = build_materialization_evidence(
        dataset=config.dataset,
        revision=config.revision,
        config_name=config.config_name,
        split=config.split,
        seed=config.seed,
        shuffle=config.shuffle,
        prompt_template_sha256=_sha256_bytes(config.prompt_template.encode("utf-8")),
        manifest_sha256=manifest_sha256,
        records=records,
    )
    summary = {
        **evidence,
        "generated_at": datetime.now(UTC).isoformat(),
        "selected_count": len(rows),
        "record_count": len(records),
        "skipped_count": skipped,
        "max_samples": config.max_samples,
        "seed": config.seed,
        "shuffle": config.shuffle,
        "prompt_template_sha256": _sha256_bytes(config.prompt_template.encode("utf-8")),
        "image_format": config.image_format,
        "manifest": {
            "path": manifest_path.name,
            "sha256": manifest_sha256,
            "bytes": len(manifest_bytes),
        },
        "hashes": {
            "ids_sha256": _sha256_bytes(
                canonical_json_bytes([str(record["id"]) for record in records])
            ),
            "images_sha256": _sha256_bytes("".join(image_hashes).encode("utf-8")),
            "prompts_sha256": _sha256_bytes("".join(prompt_hashes).encode("utf-8")),
            "answers_sha256": _sha256_bytes("".join(answer_hashes).encode("utf-8")),
        },
        "fields": {
            "image": config.image_field,
            "prompt": config.prompt_field,
            "answer": config.answer_field,
            "answers": config.answers_field,
            "id": config.id_field,
        },
        "records": evidence["records"],
    }
    (output_dir / "dataset_evidence.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "materialization_summary.json").write_text(
        json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _load_hf_dataset(args: argparse.Namespace) -> Any:
    try:
        from datasets import load_dataset
    except Exception as exc:  # pragma: no cover - dependency/environment bound.
        raise RuntimeError(
            "datasets is required to materialize a Hugging Face vision_text dataset"
        ) from exc

    kwargs: dict[str, Any] = {
        "path": args.dataset,
        "split": args.split,
    }
    if args.config_name:
        kwargs["name"] = args.config_name
    if args.revision:
        kwargs["revision"] = args.revision
    if args.cache_dir:
        kwargs["cache_dir"] = args.cache_dir
    return load_dataset(**kwargs)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    output_dir = Path(args.output_dir).expanduser()
    if output_dir.exists():
        if not args.overwrite:
            print(
                f"Output directory already exists: {output_dir} (use --overwrite)",
                file=sys.stderr,
            )
            return 2
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = MaterializeConfig(
        dataset=args.dataset,
        split=args.split,
        revision=args.revision,
        config_name=args.config_name,
        image_field=args.image_field,
        prompt_field=args.prompt_field,
        answer_field=args.answer_field or None,
        answers_field=args.answers_field or None,
        id_field=args.id_field or None,
        prompt_template=args.prompt_template,
        max_samples=args.max_samples,
        seed=args.seed,
        shuffle=args.shuffle,
        image_format=args.image_format,
    )

    dataset = _load_hf_dataset(args)
    rows = _select_rows(dataset, config=config)
    summary = materialize_rows(rows, output_dir=output_dir, config=config)
    print(json.dumps(summary, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
