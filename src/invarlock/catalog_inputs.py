"""Catalog-driven, local materialization of pinned vision-text inputs."""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import tempfile
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import yaml

from invarlock.evidence_catalog import (
    EvidenceCatalogError,
    entry_digest,
    load_evidence_catalog,
    load_resolved_inputs,
)
from invarlock.evidence_catalog_binding import EVALUATION_INPUT_BINDING_FORMAT
from invarlock.evidence_pack_json import (
    StrictJsonError,
    read_json_object_snapshot,
    read_jsonl_snapshot,
    read_regular_file_bytes,
    sha256_prefixed,
)
from invarlock.strict_yaml import StrictYamlError, load_yaml_object
from invarlock.vision_dataset_evidence import (
    build_materialization_evidence,
    canonical_json_bytes,
    dataset_record_digest,
    materialized_record_digest,
    validate_dataset_evidence,
)


@dataclass(frozen=True)
class _VisionInputConfig:
    dataset: str
    revision: str
    config_name: str | None
    split: str
    max_samples: int
    min_usable_samples: int | None
    seed: int
    shuffle: bool
    image_field: str
    prompt_field: str
    answer_field: str | None
    answers_field: str | None
    id_field: str | None
    prompt_template: str
    image_format: str


def _field(row: Mapping[str, Any], name: str | None) -> Any:
    if not name:
        return None
    current: Any = row
    for part in name.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _answers(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        values: Iterable[Any] = (value,)
    elif isinstance(value, Mapping):
        values = (value.get("answer") or value.get("text") or value.get("label"),)
    elif isinstance(value, Iterable) and not isinstance(value, bytes | bytearray):
        values = value
    else:
        values = (value,)
    result: list[str] = []
    seen: set[str] = set()
    for item in values:
        if isinstance(item, Mapping):
            item = item.get("answer") or item.get("text") or item.get("label")
        text = " ".join(str(item or "").strip().split())
        if text and text.lower() not in seen:
            result.append(text)
            seen.add(text.lower())
    return result


def _image_bytes(value: Any, *, image_format: str) -> bytes:
    if isinstance(value, bytes | bytearray):
        return bytes(value)
    if isinstance(value, str | Path):
        raise ValueError("row-provided image paths are not accepted")
    if isinstance(value, Mapping):
        raw = value.get("bytes")
        if isinstance(raw, bytes | bytearray):
            return bytes(raw)
        path = value.get("path")
        if isinstance(path, str | Path) and str(path):
            raise ValueError("row-provided image paths are not accepted")
    save = getattr(value, "save", None)
    if callable(save):
        image = value
        if image_format == "jpeg":
            convert = getattr(image, "convert", None)
            if callable(convert):
                image = convert("RGB")
        buffer = BytesIO()
        image.save(buffer, format="JPEG" if image_format == "jpeg" else "PNG")
        return buffer.getvalue()
    raise ValueError(f"unsupported image value type: {type(value).__name__}")


def _record_id(value: Any, *, row_index: int) -> str:
    if isinstance(value, str) and value.strip() == value and value:
        return value
    if value is None or value == "":
        return f"row-{row_index}"
    return "source-id-sha256-" + hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _config(entry: Mapping[str, object]) -> _VisionInputConfig:
    inputs = entry.get("inputs")
    if not isinstance(inputs, Mapping) or inputs.get("kind") != "vision_text":
        raise EvidenceCatalogError("catalog entry does not define vision_text inputs")
    materialization = inputs.get("materialization")
    if not isinstance(materialization, Mapping):
        raise EvidenceCatalogError("catalog entry has no vision_text materialization")
    source = inputs.get("source")
    if not isinstance(source, Mapping):
        raise EvidenceCatalogError("catalog entry has no input source")
    try:
        return _VisionInputConfig(
            dataset=str(materialization["dataset"]),
            revision=str(materialization["revision"]),
            config_name=(
                str(materialization["config_name"])
                if materialization.get("config_name") is not None
                else None
            ),
            split=str(materialization["split"]),
            max_samples=int(materialization["max_samples"]),
            min_usable_samples=(
                int(materialization["min_usable_samples"])
                if materialization.get("min_usable_samples") is not None
                else None
            ),
            seed=int(materialization.get("seed", 0)),
            shuffle=bool(materialization.get("shuffle", False)),
            image_field=str(materialization["image_field"]),
            prompt_field=str(materialization["prompt_field"]),
            answer_field=(
                str(materialization["answer_field"])
                if materialization.get("answer_field") is not None
                else None
            ),
            answers_field=(
                str(materialization["answers_field"])
                if materialization.get("answers_field") is not None
                else None
            ),
            id_field=(
                str(materialization["id_field"])
                if materialization.get("id_field") is not None
                else None
            ),
            prompt_template=str(materialization.get("prompt_template", "{question}")),
            image_format=str(materialization.get("image_format", "png")),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise EvidenceCatalogError(
            "catalog vision_text materialization is incomplete"
        ) from exc


def _load_rows_from_hub(
    *, dataset: str, revision: str, split: str, config_name: str | None
) -> Iterable[Mapping[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - dependency dependent.
        raise EvidenceCatalogError(
            "datasets is required for catalog materialization"
        ) from exc
    kwargs: dict[str, Any] = {"path": dataset, "revision": revision, "split": split}
    if config_name is not None:
        kwargs["name"] = config_name
    loaded = load_dataset(**kwargs)
    return (row for row in loaded if isinstance(row, Mapping))


def _select_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    max_samples: int,
    shuffle: bool,
    seed: int,
) -> list[tuple[int, Mapping[str, Any]]]:
    """Select rows with memory bounded by max_samples."""

    limit = max(max_samples, 0)
    if limit == 0:
        return []
    selected: list[tuple[int, Mapping[str, Any]]] = []
    generator = random.Random(seed)
    for row_index, row in enumerate(rows):
        if not shuffle:
            selected.append((row_index, row))
            if len(selected) == limit:
                break
            continue
        if len(selected) < limit:
            selected.append((row_index, row))
            continue
        replacement = generator.randrange(row_index + 1)
        if replacement < limit:
            selected[replacement] = (row_index, row)
    if shuffle:
        generator.shuffle(selected)
    return selected


def materialize_catalog_input(
    *,
    catalog_path: Path,
    lane_id: str,
    output_dir: Path,
    load_rows: Callable[..., Iterable[Mapping[str, Any]]] | None = None,
) -> dict[str, object]:
    """Materialize a catalog-pinned vision-text input without run scheduling."""

    catalog = load_evidence_catalog(catalog_path)
    entry = catalog.entries.get(lane_id)
    if entry is None:
        raise EvidenceCatalogError("catalog entry id is not present")
    config = _config(entry)
    if output_dir.exists():
        raise EvidenceCatalogError("materialization output already exists")
    rows_loader = load_rows or _load_rows_from_hub
    selected_rows = _select_rows(
        rows_loader(
            dataset=config.dataset,
            revision=config.revision,
            split=config.split,
            config_name=config.config_name,
        ),
        max_samples=config.max_samples,
        shuffle=config.shuffle,
        seed=config.seed,
    )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    try:
        images_dir = staging_dir / "images"
        images_dir.mkdir()
        records: list[dict[str, Any]] = []
        skipped = 0
        manifest_path = staging_dir / "manifest.jsonl"
        with manifest_path.open("x", encoding="utf-8") as handle:
            for selected_index, (row_index, row) in enumerate(selected_rows):
                question = " ".join(str(_field(row, config.prompt_field) or "").split())
                answers = _answers(_field(row, config.answer_field))
                for answer in _answers(_field(row, config.answers_field)):
                    if answer.lower() not in {existing.lower() for existing in answers}:
                        answers.append(answer)
                image = _field(row, config.image_field)
                if not question or not answers or image is None:
                    skipped += 1
                    continue
                record_id = _record_id(
                    _field(row, config.id_field), row_index=row_index
                )
                image_bytes = _image_bytes(image, image_format=config.image_format)
                image_name = (
                    f"{selected_index:06d}-"
                    f"{hashlib.sha256(record_id.encode()).hexdigest()[:16]}"
                )
                image_name += ".jpg" if config.image_format == "jpeg" else ".png"
                image_path = images_dir / image_name
                image_path.write_bytes(image_bytes)
                try:
                    prompt = config.prompt_template.format(question=question).strip()
                except (KeyError, IndexError, ValueError):
                    prompt = question
                record: dict[str, Any] = {
                    "id": record_id,
                    "image_path": image_path.relative_to(staging_dir).as_posix(),
                    "prompt": prompt,
                    "answer": answers[0],
                    "answers": answers,
                    "source": {
                        "dataset": config.dataset,
                        "revision": config.revision,
                        "split": config.split,
                        "row_index": row_index,
                        "question": question,
                        "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
                        "image_bytes": len(image_bytes),
                        "dataset_record_sha256": dataset_record_digest(
                            dataset=config.dataset,
                            revision=config.revision,
                            split=config.split,
                            row_index=row_index,
                            record_id=record_id,
                            question=question,
                            answers=answers,
                        ),
                    },
                }
                record["source"]["record_sha256"] = materialized_record_digest(record)
                handle.write(
                    json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n"
                )
                records.append(record)
        if not records or (
            config.min_usable_samples is not None
            and len(records) < config.min_usable_samples
        ):
            raise EvidenceCatalogError(
                "materialization did not produce the required usable records"
            )
        manifest_bytes, parsed_records = read_jsonl_snapshot(
            manifest_path, label="materialized vision-text manifest"
        )
        if parsed_records != records:
            raise EvidenceCatalogError(
                "materialized manifest does not match its records"
            )
        evidence = build_materialization_evidence(
            dataset=config.dataset,
            revision=config.revision,
            config_name=config.config_name,
            split=config.split,
            seed=config.seed,
            shuffle=config.shuffle,
            prompt_template_sha256=hashlib.sha256(
                config.prompt_template.encode("utf-8")
            ).hexdigest(),
            manifest_sha256=sha256_prefixed(manifest_bytes),
            records=records,
        )
        summary: dict[str, object] = {
            **evidence,
            "catalog_digest": catalog.digest,
            "catalog_entry_id": lane_id,
            "record_count": len(records),
            "selected_count": len(selected_rows),
            "skipped_count": skipped,
            "manifest": {
                "path": manifest_path.name,
                "sha256": sha256_prefixed(manifest_bytes),
            },
        }
        (staging_dir / "dataset_evidence.json").write_text(
            json.dumps(evidence, sort_keys=True) + "\n", encoding="utf-8"
        )
        (staging_dir / "materialization_summary.json").write_text(
            json.dumps(summary, sort_keys=True) + "\n", encoding="utf-8"
        )
        if output_dir.exists():
            raise EvidenceCatalogError("materialization output already exists")
        staging_dir.replace(output_dir)
        return {"ok": True, **summary}
    except BaseException:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise


def prepare_catalog_preset(
    *,
    catalog_path: Path,
    lane_id: str,
    resolved_inputs_path: Path,
    preset_path: Path,
    output_path: Path,
    materialization_dir: Path | None = None,
) -> dict[str, object]:
    """Write the deterministic evaluator preset for one resolved catalog lane."""

    catalog = load_evidence_catalog(catalog_path)
    entry = catalog.entries.get(lane_id)
    if entry is None:
        raise EvidenceCatalogError("catalog entry id is not present")
    resolved, _resolved_digest = load_resolved_inputs(resolved_inputs_path, entry=entry)
    catalog_preset = entry.get("preset")
    if not isinstance(catalog_preset, Mapping):
        raise EvidenceCatalogError("catalog entry preset is invalid")
    expected = catalog_preset.get("sha256")
    if not isinstance(expected, str):
        raise EvidenceCatalogError("catalog entry preset digest is invalid")
    try:
        preset_bytes = read_regular_file_bytes(preset_path, label="catalog preset")
    except StrictJsonError as exc:
        raise EvidenceCatalogError(f"catalog preset cannot be loaded: {exc}") from exc
    actual = sha256_prefixed(preset_bytes)
    if actual != expected:
        raise EvidenceCatalogError("preset digest does not match catalog entry")
    try:
        preset = load_yaml_object(preset_path, label="evaluation preset")
    except StrictYamlError as exc:
        raise EvidenceCatalogError(f"catalog preset cannot be loaded: {exc}") from exc
    if not isinstance(preset, dict):
        raise EvidenceCatalogError("evaluation preset must be an object")
    resolved_model = resolved.get("model")
    if not isinstance(resolved_model, Mapping):
        raise EvidenceCatalogError("resolved inputs model is invalid")
    model = preset.setdefault("model", {})
    if not isinstance(model, dict):
        raise EvidenceCatalogError("evaluation preset model must be an object")
    model["id"] = resolved_model["id"]
    model["adapter"] = resolved_model["adapter"]
    model["model_identity"] = {
        "kind": "remote_revision",
        "revision": resolved_model["revision"],
    }
    dataset = preset.setdefault("dataset", {})
    if not isinstance(dataset, dict):
        raise EvidenceCatalogError("evaluation preset dataset must be an object")
    resolved_dataset = resolved.get("dataset")
    if not isinstance(resolved_dataset, Mapping):
        raise EvidenceCatalogError("resolved inputs dataset is invalid")
    inputs = entry.get("inputs")
    vision = isinstance(inputs, Mapping) and inputs.get("kind") == "vision_text"
    provider_value = dataset.get("provider")
    provider = (
        dict(provider_value)
        if isinstance(provider_value, Mapping)
        else {"kind": provider_value}
    )
    provider["kind"] = resolved_dataset["provider"]
    if vision:
        if materialization_dir is None:
            raise EvidenceCatalogError(
                "vision preset preparation requires materialization"
            )
        _config(entry)
        summary_path = materialization_dir / "materialization_summary.json"
        evidence_path = materialization_dir / "dataset_evidence.json"
        manifest_path = materialization_dir / "manifest.jsonl"
        try:
            _raw_summary, summary = read_json_object_snapshot(
                summary_path, label="materialization summary"
            )
            _raw_evidence, materialization_evidence = read_json_object_snapshot(
                evidence_path, label="input materialization"
            )
            manifest_bytes = read_regular_file_bytes(
                manifest_path, label="materialized vision-text manifest"
            )
        except StrictJsonError as exc:
            raise EvidenceCatalogError(
                f"materialization cannot be loaded: {exc}"
            ) from exc
        manifest = summary.get("manifest")
        if (
            summary.get("catalog_digest") != catalog.digest
            or summary.get("catalog_entry_id") != lane_id
            or not isinstance(manifest, Mapping)
            or manifest.get("path") != "manifest.jsonl"
            or manifest.get("sha256") != sha256_prefixed(manifest_bytes)
            or validate_dataset_evidence(
                materialization_evidence,
                strict_counts=True,
                require_runtime_identity=False,
            )
            or {
                "id": resolved_dataset.get("id"),
                "revision": resolved_dataset.get("revision"),
                "config_name": resolved_dataset.get("config_name"),
                "split": resolved_dataset.get("split"),
            }
            != materialization_evidence.get("dataset")
        ):
            raise EvidenceCatalogError(
                "materialization is not bound to resolved inputs"
            )
        try:
            portable_manifest_path = manifest_path.relative_to(
                output_path.parent
            ).as_posix()
        except ValueError as exc:
            raise EvidenceCatalogError(
                "prepared preset and materialization must share one portable run root"
            ) from exc
        if any(part in {"", ".", ".."} for part in Path(portable_manifest_path).parts):
            raise EvidenceCatalogError(
                "prepared preset materialization path is not portable"
            )
        provider.pop("dataset_name", None)
        provider.pop("config_name", None)
        provider.pop("revision", None)
        provider["path"] = portable_manifest_path
    else:
        if materialization_dir is not None:
            raise EvidenceCatalogError(
                "non-vision preset preparation rejects materialization"
            )
        provider.pop("path", None)
        provider["dataset_name"] = resolved_dataset["id"]
        provider["revision"] = resolved_dataset["revision"]
        if resolved_dataset.get("config_name") is None:
            provider.pop("config_name", None)
        else:
            provider["config_name"] = resolved_dataset["config_name"]
    dataset["provider"] = provider
    dataset["split"] = resolved_dataset["split"]
    if output_path.exists():
        raise EvidenceCatalogError("prepared preset output already exists")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rendered = yaml.safe_dump(preset, sort_keys=False)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(output_path, flags, 0o600)
    except FileExistsError as exc:
        raise EvidenceCatalogError("prepared preset output already exists") from exc
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        output_path.unlink(missing_ok=True)
        raise
    return {"ok": True, "catalog_digest": catalog.digest, "lane_id": lane_id}


def build_evaluation_input_binding(
    *,
    catalog_path: Path,
    lane_id: str,
    resolved_inputs_path: Path,
    preset_path: Path,
    input_materialization_path: Path | None = None,
) -> dict[str, object]:
    """Build the closed pre-run binding propagated into report provenance."""

    catalog = load_evidence_catalog(catalog_path)
    entry = catalog.entries.get(lane_id)
    if entry is None:
        raise EvidenceCatalogError("catalog entry id is not present")
    resolved, resolved_digest = load_resolved_inputs(resolved_inputs_path, entry=entry)
    try:
        preset_bytes = read_regular_file_bytes(preset_path, label="catalog preset")
    except StrictJsonError as exc:
        raise EvidenceCatalogError(f"catalog preset cannot be loaded: {exc}") from exc
    preset_digest = sha256_prefixed(preset_bytes)
    declared_preset = entry.get("preset")
    if (
        not isinstance(declared_preset, Mapping)
        or declared_preset.get("sha256") != preset_digest
    ):
        raise EvidenceCatalogError("preset digest does not match catalog entry")
    payload: dict[str, object] = {
        "format_version": EVALUATION_INPUT_BINDING_FORMAT,
        "catalog_digest": catalog.digest,
        "catalog_entry_id": lane_id,
        "catalog_entry_digest": entry_digest(entry),
        "resolved_inputs_digest": resolved_digest,
        "preset_digest": preset_digest,
    }
    inputs = entry.get("inputs")
    vision = isinstance(inputs, Mapping) and inputs.get("kind") == "vision_text"
    if vision:
        if input_materialization_path is None:
            raise EvidenceCatalogError(
                "vision catalog binding requires input materialization"
            )
        try:
            _raw, materialization = read_json_object_snapshot(
                input_materialization_path, label="input materialization"
            )
        except StrictJsonError as exc:
            raise EvidenceCatalogError(
                f"input materialization cannot be loaded: {exc}"
            ) from exc
        materialization_errors = validate_dataset_evidence(
            materialization,
            strict_counts=True,
            require_runtime_identity=False,
        )
        if materialization_errors:
            raise EvidenceCatalogError("; ".join(materialization_errors))
        materialized_dataset = materialization.get("dataset")
        resolved_dataset = resolved.get("dataset")
        if not isinstance(materialized_dataset, Mapping) or not isinstance(
            resolved_dataset, Mapping
        ):
            raise EvidenceCatalogError("input materialization dataset is invalid")
        expected_coordinates = {
            "id": resolved_dataset.get("id"),
            "revision": resolved_dataset.get("revision"),
            "config_name": resolved_dataset.get("config_name"),
            "split": resolved_dataset.get("split"),
        }
        if dict(materialized_dataset) != expected_coordinates:
            raise EvidenceCatalogError(
                "input materialization dataset does not match resolved inputs"
            )
        payload["materialization_digest"] = materialization["semantic_digest"]
        payload["materialization_manifest_digest"] = materialization["manifest_sha256"]
    elif input_materialization_path is not None:
        raise EvidenceCatalogError(
            "non-vision catalog binding rejects input materialization"
        )
    return payload


__all__ = [
    "build_evaluation_input_binding",
    "materialize_catalog_input",
    "prepare_catalog_preset",
]
