#!/usr/bin/env python3
"""Prepare deterministic 400-record qualification suites from pinned datasets.

The script is maintainer tooling rather than a runtime dependency.  It imports
``datasets`` and Pillow only inside the hosted-data loader, writes byte-stable
JSONL inputs, and validates every emitted input through InvarLock's public local
dataset preparation contract.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import re
from collections import defaultdict, deque
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from invarlock.core.runtime_provider import (
    canonical_runtime_behavioral_schedule_json,
)
from invarlock.core.schedule_preparation import (
    LocalDatasetRequest,
    prepare_local_evaluation_schedule_bytes,
)

FORMAT_VERSION = "invarlock/qualification-suites-v1"
SELECTION_ALGORITHM = "balanced-bipartite-sha256-v1"
NORMALIZATION_ALGORITHM = "outer-whitespace-only-v1"
DEFAULT_RECORD_COUNT = 400
MMLU_DATASET = "TIGER-Lab/MMLU-Pro"
MMLU_REVISION = "b189ec765aa7ed75c8acfea42df31fdae71f97be"
MMMU_DATASET = "MMMU/MMMU_Pro"
MMMU_CONFIG = "vision"
MMMU_REVISION = "563f3e84bb3b90893083a1f039cfa13077f2302b"
MAX_TOTAL_IMAGE_BYTES = 512 * 1024 * 1024
MAX_TOTAL_IMAGE_PIXELS = 2_000_000_000
MAX_IMAGE_BYTES = 64 * 1024 * 1024
MAX_IMAGE_PIXELS = 50_000_000
SUPPORTED_IMAGE_MEDIA_TYPES = frozenset({"image/jpeg", "image/png", "image/webp"})
_REVISION = re.compile(r"^[a-f0-9]{40}$")
_ANSWER = re.compile(r"^[A-J]$")


class QualificationSuiteError(ValueError):
    """Raised when upstream material cannot form a qualifying suite."""


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _required_text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise QualificationSuiteError(f"{label} must be non-empty trimmed text")
    return value


def _source_text(value: object, *, label: str) -> str:
    if not isinstance(value, str):
        raise QualificationSuiteError(f"{label} must be text")
    return _required_text(value.strip(), label=label)


def _required_revision(value: str, *, label: str) -> str:
    if _REVISION.fullmatch(value) is None:
        raise QualificationSuiteError(f"{label} must be an exact 40-character revision")
    return value


def _required_positive_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise QualificationSuiteError(f"{label} must be a positive integer")
    return value


def _balanced_quotas(values: Sequence[str], count: int) -> dict[str, int]:
    unique = sorted(set(values))
    if not unique or count < len(unique):
        raise QualificationSuiteError("selection cannot represent every stratum")
    base, remainder = divmod(count, len(unique))
    return {value: base + (index < remainder) for index, value in enumerate(unique)}


def _cell_allocation(
    rows: Sequence[Mapping[str, object]],
    *,
    count: int,
) -> dict[tuple[str, str], int]:
    """Allocate balanced group and answer quotas within observed capacities."""

    groups = [str(row["group"]) for row in rows]
    answers = [str(row["answer"]) for row in rows]
    group_quotas = _balanced_quotas(groups, count)
    answer_quotas = _balanced_quotas(answers, count)
    capacities: dict[tuple[str, str], int] = defaultdict(int)
    for row in rows:
        capacities[(str(row["group"]), str(row["answer"]))] += 1

    source, sink = "source", "sink"
    residual: dict[str, dict[str, int]] = defaultdict(dict)

    def connect(left: str, right: str, capacity: int) -> None:
        residual[left][right] = capacity
        residual[right].setdefault(left, 0)

    for group, quota in group_quotas.items():
        connect(source, f"g:{group}", quota)
    for (group, answer), capacity in sorted(capacities.items()):
        connect(f"g:{group}", f"a:{answer}", capacity)
    for answer, quota in answer_quotas.items():
        connect(f"a:{answer}", sink, quota)

    flow = 0
    while flow < count:
        parent: dict[str, str | None] = {source: None}
        queue = deque([source])
        while queue and sink not in parent:
            current = queue.popleft()
            for target in sorted(residual[current]):
                if residual[current][target] > 0 and target not in parent:
                    parent[target] = current
                    queue.append(target)
        if sink not in parent:
            raise QualificationSuiteError(
                "dataset cannot satisfy balanced group and answer quotas"
            )
        cursor = sink
        path_capacity = count - flow
        while parent[cursor] is not None:
            prior = parent[cursor]
            assert prior is not None
            path_capacity = min(path_capacity, residual[prior][cursor])
            cursor = prior
        cursor = sink
        while parent[cursor] is not None:
            prior = parent[cursor]
            assert prior is not None
            residual[prior][cursor] -= path_capacity
            residual[cursor][prior] += path_capacity
            cursor = prior
        flow += path_capacity

    allocation: dict[tuple[str, str], int] = {}
    for cell, capacity in capacities.items():
        group, answer = cell
        used = capacity - residual[f"g:{group}"][f"a:{answer}"]
        if used:
            allocation[cell] = used
    if sum(allocation.values()) != count:
        raise AssertionError("balanced allocation count drifted")
    return allocation


def select_stratified(
    rows: Sequence[Mapping[str, object]],
    *,
    count: int,
    seed: str,
) -> list[dict[str, object]]:
    """Select one deterministic, balanced, semantically unique subset."""

    if len(rows) < count:
        raise QualificationSuiteError(
            f"dataset has {len(rows)} usable rows but {count} are required"
        )
    identifiers: set[str] = set()
    semantic_digests: set[str] = set()
    cells: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for raw in rows:
        row = dict(raw)
        identifier = _required_text(row.get("id"), label="record id")
        group = _required_text(row.get("group"), label=f"{identifier} group")
        answer = _required_text(row.get("answer"), label=f"{identifier} answer")
        digest = _required_text(
            row.get("semantic_sha256"), label=f"{identifier} semantic digest"
        )
        if identifier in identifiers:
            raise QualificationSuiteError(f"duplicate record id: {identifier}")
        if digest in semantic_digests:
            raise QualificationSuiteError(f"duplicate semantic record digest: {digest}")
        identifiers.add(identifier)
        semantic_digests.add(digest)
        cells[(group, answer)].append(row)

    allocation = _cell_allocation(rows, count=count)
    selected: list[dict[str, object]] = []
    for cell, quota in sorted(allocation.items()):
        candidates = sorted(
            cells[cell],
            key=lambda row: hashlib.sha256(
                f"{seed}\0{row['id']}\0{row['semantic_sha256']}".encode()
            ).hexdigest(),
        )
        selected.extend(candidates[:quota])
    selected.sort(key=lambda row: str(row["id"]))
    if len(selected) != count:
        raise AssertionError("stratified selection count drifted")
    return selected


def _normalize_options(value: object, *, label: str, max_items: int = 10) -> list[str]:
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (SyntaxError, ValueError) as exc:
            raise QualificationSuiteError(f"{label} options are invalid") from exc
    if not isinstance(value, list) or not 2 <= len(value) <= max_items:
        raise QualificationSuiteError(
            f"{label} options must contain 2 to {max_items} items"
        )
    return [
        _source_text(option, label=f"{label} option {index}")
        for index, option in enumerate(value)
    ]


def normalize_mmlu_rows(
    rows: Iterable[Mapping[str, object]],
) -> list[dict[str, object]]:
    normalized: list[dict[str, object]] = []
    for index, row in enumerate(rows):
        question_id = row.get("question_id")
        if isinstance(question_id, bool) or not isinstance(question_id, int):
            raise QualificationSuiteError(f"MMLU row {index} question_id is invalid")
        identifier = f"mmlu_pro_{question_id:05d}"
        question = _source_text(row.get("question"), label=f"{identifier} question")
        options = _normalize_options(row.get("options"), label=identifier)
        answer = _source_text(row.get("answer"), label=f"{identifier} answer")
        answer_index = row.get("answer_index")
        if (
            _ANSWER.fullmatch(answer) is None
            or isinstance(answer_index, bool)
            or not isinstance(answer_index, int)
            or not 0 <= answer_index < len(options)
            or answer != chr(ord("A") + answer_index)
        ):
            raise QualificationSuiteError(f"{identifier} answer binding is invalid")
        category = _source_text(row.get("category"), label=f"{identifier} category")
        source = _source_text(row.get("src"), label=f"{identifier} source")
        semantic = {
            "question": question,
            "options": options,
            "answer": answer,
        }
        normalized.append(
            {
                "id": identifier,
                "group": category,
                "answer": answer,
                "question": question,
                "options": options,
                "source": source,
                "semantic_sha256": _sha256(_canonical_json_bytes(semantic)),
            }
        )
    return normalized


def _image_payload(value: object, *, identifier: str) -> bytes:
    if not isinstance(value, Mapping) or set(value) != {"bytes", "path"}:
        raise QualificationSuiteError(f"{identifier} image binding is invalid")
    payload = value.get("bytes")
    if payload is None:
        path = value.get("path")
        if not isinstance(path, str):
            raise QualificationSuiteError(f"{identifier} image path is invalid")
        payload = Path(path).read_bytes()
    if not isinstance(payload, bytes) or not payload:
        raise QualificationSuiteError(f"{identifier} image bytes are invalid")
    return payload


def _image_metadata(payload: bytes, *, identifier: str) -> tuple[str, int, int]:
    try:
        from PIL import Image as PILImage
    except ImportError as exc:  # pragma: no cover - exercised by maintainer CLI
        raise QualificationSuiteError(
            "Pillow is required to prepare vision suites"
        ) from exc
    try:
        with PILImage.open(io.BytesIO(payload)) as image:
            image.load()
            media_type = PILImage.MIME.get(image.format or "")
            width, height = image.size
    except (OSError, ValueError) as exc:
        raise QualificationSuiteError(f"{identifier} image cannot be decoded") from exc
    if not isinstance(media_type, str) or not media_type:
        raise QualificationSuiteError(f"{identifier} image media type is unavailable")
    return media_type, width, height


def normalize_mmmu_rows(
    rows: Iterable[Mapping[str, object]],
    *,
    exclusions: dict[str, int] | None = None,
) -> list[dict[str, object]]:
    normalized: list[dict[str, object]] = []
    excluded = exclusions if exclusions is not None else {}
    for index, row in enumerate(rows):
        upstream_id = _source_text(row.get("id"), label=f"MMMU row {index} id")
        identifier = "mmmu_pro_" + _sha256(upstream_id.encode())[:20]
        options = _normalize_options(row.get("options"), label=identifier, max_items=26)
        if len(options) > 10:
            excluded["option_count_limit"] = excluded.get("option_count_limit", 0) + 1
            continue
        answer = _source_text(row.get("answer"), label=f"{identifier} answer")
        if _ANSWER.fullmatch(answer) is None or ord(answer) - ord("A") >= len(options):
            raise QualificationSuiteError(f"{identifier} answer binding is invalid")
        subject = _source_text(row.get("subject"), label=f"{identifier} subject")
        payload = _image_payload(row.get("image"), identifier=identifier)
        media_type, width, height = _image_metadata(payload, identifier=identifier)
        if media_type not in SUPPORTED_IMAGE_MEDIA_TYPES:
            excluded["unsupported_media_type"] = (
                excluded.get("unsupported_media_type", 0) + 1
            )
            continue
        if len(payload) > MAX_IMAGE_BYTES:
            excluded["image_byte_limit"] = excluded.get("image_byte_limit", 0) + 1
            continue
        if width * height > MAX_IMAGE_PIXELS:
            excluded["image_pixel_limit"] = excluded.get("image_pixel_limit", 0) + 1
            continue
        image_sha256 = _sha256(payload)
        semantic = {
            "options": options,
            "answer": answer,
            "image_sha256": image_sha256,
        }
        normalized.append(
            {
                "id": identifier,
                "upstream_id": upstream_id,
                "group": subject,
                "answer": answer,
                "options": options,
                "semantic_sha256": _sha256(_canonical_json_bytes(semantic)),
                "image_bytes": payload,
                "image_sha256": image_sha256,
                "image_byte_length": len(payload),
                "image_media_type": media_type,
                "image_width": width,
                "image_height": height,
            }
        )
    return normalized


def _choice_block(row: Mapping[str, object]) -> str:
    options = row["options"]
    assert isinstance(options, list)
    return "\n".join(
        f"{chr(ord('A') + index)}. {option}" for index, option in enumerate(options)
    )


def _question_body(row: Mapping[str, object]) -> str:
    return f"Question: {row['question']}\nChoices:\n{_choice_block(row)}"


def render_text_records(
    rows: Sequence[Mapping[str, object]], *, rendering: str
) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for row in rows:
        body = _question_body(row)
        answer = str(row["answer"])
        if rendering == "raw_causal":
            prompt = f"{body}\nAnswer:"
            expected = f" {answer}"
        elif rendering == "mistral_instruct":
            prompt = (
                "<s>[SYSTEM_PROMPT]You answer multiple-choice questions. "
                "Follow the requested output format exactly.[/SYSTEM_PROMPT]"
                f"[INST]{body}\nReply with exactly one uppercase option letter "
                "and no other text.[/INST]"
            )
            expected = answer
        elif rendering == "qwen_instruct":
            prompt = (
                "<|im_start|>system\nYou answer multiple-choice questions and "
                "follow the requested output format exactly.<|im_end|>\n"
                f"<|im_start|>user\n{body}\nReply with exactly one uppercase "
                "option letter and no other text.<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
            expected = answer
        else:
            raise QualificationSuiteError(f"unsupported text rendering: {rendering}")
        records.append({"id": str(row["id"]), "prompt": prompt, "expected": expected})
    return records


def _jsonl_bytes(rows: Sequence[Mapping[str, object]]) -> bytes:
    return b"".join(_canonical_json_bytes(row) + b"\n" for row in rows)


def _write_bytes(path: Path, payload: bytes, *, root: Path) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": _sha256(payload),
        "bytes": len(payload),
    }


def _validated_schedule(
    *,
    source_path: Path,
    source_bytes: bytes,
    name: str,
    task: str,
    multimodal: bool = False,
) -> tuple[bytes, str]:
    digest = _sha256(source_bytes)
    arguments: dict[str, Any] = {
        "path": source_path,
        "sha256": digest,
        "name": name,
        "split": "qualification",
        "input_field": "prompt",
        "expected_output_field": "expected",
        "id_field": "id",
    }
    if multimodal:
        arguments.update(
            {
                "content_role": "image",
                "content_id_field": "content_id",
                "content_sha256_field": "content_sha256",
                "content_byte_length_field": "content_bytes",
                "content_media_type_field": "content_media_type",
            }
        )
    schedule = prepare_local_evaluation_schedule_bytes(
        LocalDatasetRequest(**arguments), source_bytes, task=task
    )
    return canonical_runtime_behavioral_schedule_json(
        schedule
    ), schedule.schedule_sha256


def _distribution(rows: Sequence[Mapping[str, object]], field: str) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[str(row[field])] += 1
    return dict(sorted(counts.items()))


def _load_hosted_datasets() -> tuple[
    Sequence[Mapping[str, object]], Sequence[Mapping[str, object]]
]:
    try:
        from datasets import Image, load_dataset  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - exercised by maintainer CLI
        raise QualificationSuiteError(
            "datasets and Pillow are required to download qualification sources"
        ) from exc
    mmlu = load_dataset(MMLU_DATASET, revision=MMLU_REVISION, split="test")
    mmmu = load_dataset(
        MMMU_DATASET,
        MMMU_CONFIG,
        revision=MMMU_REVISION,
        split="test",
    ).cast_column("image", Image(decode=False))
    return mmlu, mmmu


def prepare_suites(*, output: Path, record_count: int) -> dict[str, object]:
    _required_revision(MMLU_REVISION, label="MMLU-Pro revision")
    _required_revision(MMMU_REVISION, label="MMMU-Pro revision")
    if record_count != DEFAULT_RECORD_COUNT:
        raise QualificationSuiteError(
            f"public qualification suites require exactly {DEFAULT_RECORD_COUNT} records"
        )
    mmlu_source, mmmu_source = _load_hosted_datasets()
    normalized_text = normalize_mmlu_rows(mmlu_source)
    multimodal_exclusions: dict[str, int] = {}
    normalized_multimodal = normalize_mmmu_rows(
        mmmu_source, exclusions=multimodal_exclusions
    )
    text = select_stratified(
        normalized_text,
        count=record_count,
        seed=f"{MMLU_REVISION}:{SELECTION_ALGORITHM}",
    )
    multimodal = select_stratified(
        normalized_multimodal,
        count=record_count,
        seed=f"{MMMU_REVISION}:{SELECTION_ALGORITHM}",
    )

    image_bytes = sum(
        _required_positive_int(
            row["image_byte_length"], label="selected image byte length"
        )
        for row in multimodal
    )
    image_pixels = sum(
        _required_positive_int(row["image_width"], label="selected image width")
        * _required_positive_int(row["image_height"], label="selected image height")
        for row in multimodal
    )
    if image_bytes > MAX_TOTAL_IMAGE_BYTES:
        raise QualificationSuiteError("selected vision suite exceeds the byte limit")
    if image_pixels > MAX_TOTAL_IMAGE_PIXELS:
        raise QualificationSuiteError("selected vision suite exceeds the pixel limit")

    output.mkdir(parents=True, exist_ok=False)
    artifacts: dict[str, object] = {}
    semantic_rows = [
        {key: value for key, value in row.items() if key not in {"group"}}
        for row in text
    ]
    artifacts["text_semantic_bank"] = _write_bytes(
        output / "text" / "semantic-bank.jsonl",
        _jsonl_bytes(semantic_rows),
        root=output,
    )
    for rendering in ("raw_causal", "mistral_instruct", "qwen_instruct"):
        records = render_text_records(text, rendering=rendering)
        source_path = output / "text" / f"{rendering.replace('_', '-')}.jsonl"
        source_bytes = _jsonl_bytes(records)
        source_artifact = _write_bytes(source_path, source_bytes, root=output)
        schedule_bytes, schedule_digest = _validated_schedule(
            source_path=source_path,
            source_bytes=source_bytes,
            name=f"mmlu-pro-{rendering.replace('_', '-')}",
            task="text_causal",
        )
        schedule_path = output / "text" / f"{rendering.replace('_', '-')}.schedule.json"
        schedule_artifact = _write_bytes(
            schedule_path, schedule_bytes + b"\n", root=output
        )
        source_artifact["schedule"] = schedule_artifact
        source_artifact["schedule_sha256"] = schedule_digest
        artifacts[f"text_{rendering}"] = source_artifact

    content_store = output / "multimodal" / "content-store"
    content_store.mkdir(parents=True)
    multimodal_rows: list[dict[str, object]] = []
    multimodal_bank: list[dict[str, object]] = []
    prompt = (
        "Read the question and answer choices in the image. Reply with exactly "
        "one uppercase option letter (A-J) and no other text."
    )
    for row in multimodal:
        content_id = "image_" + str(row["image_sha256"])[:24]
        payload = row["image_bytes"]
        assert isinstance(payload, bytes)
        content_store.joinpath(content_id).write_bytes(payload)
        multimodal_rows.append(
            {
                "id": row["id"],
                "prompt": prompt,
                "expected": row["answer"],
                "content_id": content_id,
                "content_sha256": row["image_sha256"],
                "content_bytes": row["image_byte_length"],
                "content_media_type": row["image_media_type"],
            }
        )
        multimodal_bank.append(
            {
                key: value
                for key, value in row.items()
                if key not in {"group", "image_bytes"}
            }
        )
    artifacts["multimodal_semantic_bank"] = _write_bytes(
        output / "multimodal" / "semantic-bank.jsonl",
        _jsonl_bytes(multimodal_bank),
        root=output,
    )
    multimodal_path = output / "multimodal" / "mmmu-pro-vision.jsonl"
    multimodal_bytes = _jsonl_bytes(multimodal_rows)
    multimodal_artifact = _write_bytes(multimodal_path, multimodal_bytes, root=output)
    schedule_bytes, schedule_digest = _validated_schedule(
        source_path=multimodal_path,
        source_bytes=multimodal_bytes,
        name="mmmu-pro-vision",
        task="vision_text_generation",
        multimodal=True,
    )
    schedule_artifact = _write_bytes(
        output / "multimodal" / "mmmu-pro-vision.schedule.json",
        schedule_bytes + b"\n",
        root=output,
    )
    multimodal_artifact["schedule"] = schedule_artifact
    multimodal_artifact["schedule_sha256"] = schedule_digest
    multimodal_artifact["content_store"] = {
        "path": content_store.relative_to(output).as_posix(),
        "file_count": len(multimodal_rows),
        "total_bytes": image_bytes,
        "total_pixels": image_pixels,
    }
    artifacts["multimodal"] = multimodal_artifact

    manifest = {
        "format_version": FORMAT_VERSION,
        "selection_algorithm": SELECTION_ALGORITHM,
        "normalization_algorithm": NORMALIZATION_ALGORITHM,
        "record_count": record_count,
        "sources": {
            "text": {
                "dataset": MMLU_DATASET,
                "revision": MMLU_REVISION,
                "split": "test",
                "license": "MIT",
                "source_record_count": len(mmlu_source),
                "eligible_record_count": len(normalized_text),
            },
            "multimodal": {
                "dataset": MMMU_DATASET,
                "config": MMMU_CONFIG,
                "revision": MMMU_REVISION,
                "split": "test",
                "license": "Apache-2.0",
                "source_record_count": len(mmmu_source),
                "eligible_record_count": len(normalized_multimodal),
                "exclusions": dict(sorted(multimodal_exclusions.items())),
            },
        },
        "distributions": {
            "text_groups": _distribution(text, "group"),
            "text_answers": _distribution(text, "answer"),
            "multimodal_groups": _distribution(multimodal, "group"),
            "multimodal_answers": _distribution(multimodal, "answer"),
        },
        "selected_ids": {
            "text": [row["id"] for row in text],
            "multimodal": [row["upstream_id"] for row in multimodal],
        },
        "artifacts": artifacts,
    }
    manifest_path = output / "qualification-suites.manifest.json"
    manifest_bytes = _canonical_json_bytes(manifest) + b"\n"
    manifest_path.write_bytes(manifest_bytes)
    return {**manifest, "manifest_sha256": _sha256(manifest_bytes)}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--record-count", type=int, default=DEFAULT_RECORD_COUNT)
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    result = prepare_suites(
        output=arguments.output.absolute(), record_count=arguments.record_count
    )
    print(_canonical_json_bytes(result).decode("utf-8"))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
