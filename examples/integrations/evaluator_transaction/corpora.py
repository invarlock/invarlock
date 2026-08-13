"""Closed corpus profiles shared by the flagship evaluator transactions."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import urllib.request
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

_ROOT = Path(__file__).resolve().parent
_QUICK_RECORDS = _ROOT.parent / "lm-evaluation-harness" / "records.json"
_FLAGSHIP_MANIFEST = _ROOT / "flagship_corpus.json"


@dataclass(frozen=True, slots=True)
class CorpusProfile:
    key: str
    profile_id: str
    dataset_name: str
    split: str
    record_count: int
    dataset_sha256: str
    context_length: int
    minimum_side_accuracy: float
    maximum_interval_width_pp: float

    def dataset_descriptor(self, path: str = "inputs/records.jsonl") -> dict[str, Any]:
        return {
            "path": path,
            "sha256": self.dataset_sha256,
            "format": "jsonl",
            "name": self.dataset_name,
            "split": self.split,
            "input_field": "prompt",
            "expected_output_field": "expected",
            "id_field": "id",
        }

    def acceptance_policy(self) -> dict[str, Any]:
        return {
            "resolved_policy": {
                "metrics": {
                    "exact_match": {
                        "delta_min_pp": -20.0,
                        "maximum_interval_width_pp": self.maximum_interval_width_pp,
                        "minimum_record_count": self.record_count,
                        "minimum_side_accuracy": self.minimum_side_accuracy,
                    }
                }
            }
        }


@dataclass(frozen=True, slots=True)
class SelectionStratum:
    minimum_prompt_tokens: int
    maximum_prompt_tokens: int
    sample_count: int


@dataclass(frozen=True, slots=True)
class SelectionCandidate:
    source_index: int
    prompt_token_count: int
    source_line_sha256: str


@dataclass(frozen=True, slots=True)
class FlagshipSelection:
    seed: str
    eligible_record_count: int
    strata: tuple[SelectionStratum, ...]
    indices: tuple[int, ...]
    indices_sha256: str
    selected_source_lines_sha256: str


def _manifest() -> dict[str, Any]:
    value = json.loads(_FLAGSHIP_MANIFEST.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("flagship corpus manifest must contain an object")
    return value


def records_jsonl(records: Iterable[dict[str, str]]) -> bytes:
    return b"".join(
        (json.dumps(record, sort_keys=True) + "\n").encode("utf-8")
        for record in records
    )


def _quick_profile() -> CorpusProfile:
    return CorpusProfile(
        key="quick",
        profile_id="causal-cloze-102-v1",
        dataset_name="qwen3-0.6b-base-to-post-trained",
        split="validation",
        record_count=102,
        dataset_sha256="d80e81ba17fb93b9b8a46f9817f9841f5f9c2858c9d703b3ce28847b2eaeb57c",
        context_length=64,
        minimum_side_accuracy=0.20,
        maximum_interval_width_pp=20.0,
    )


def _flagship_profile() -> CorpusProfile:
    value = _manifest()["derived_dataset"]
    return CorpusProfile(
        key="flagship",
        profile_id=_manifest()["profile_id"],
        dataset_name=value["name"],
        split=value["split"],
        record_count=value["record_count"],
        dataset_sha256=value["sha256"],
        context_length=256,
        minimum_side_accuracy=0.05,
        maximum_interval_width_pp=10.0,
    )


def corpus_profile(key: str) -> CorpusProfile:
    profiles = {"quick": _quick_profile(), "flagship": _flagship_profile()}
    try:
        return profiles[key]
    except KeyError as exc:
        raise ValueError(f"unknown corpus profile: {key}") from exc


def profile_for_dataset(payload: bytes) -> CorpusProfile:
    observed = hashlib.sha256(payload).hexdigest()
    for key in ("quick", "flagship"):
        profile = corpus_profile(key)
        if observed == profile.dataset_sha256:
            validate_dataset_records(payload, profile)
            return profile
    raise ValueError("dataset is not a pinned evaluator corpus")


def profile_for_descriptor(value: object) -> CorpusProfile:
    for key in ("quick", "flagship"):
        profile = corpus_profile(key)
        if value == profile.dataset_descriptor():
            return profile
    raise ValueError("dataset descriptor is not a pinned evaluator corpus")


def validate_dataset_records(payload: bytes, profile: CorpusProfile) -> None:
    try:
        values = [json.loads(line) for line in payload.splitlines()]
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("evaluator corpus is not valid JSONL") from exc
    if len(values) != profile.record_count or any(
        not isinstance(value, dict)
        or set(value) != {"expected", "id", "prompt"}
        or any(not isinstance(value[key], str) or not value[key] for key in value)
        for value in values
    ):
        raise ValueError(
            f"evaluator corpus must contain {profile.record_count} complete records"
        )
    if len({value["id"] for value in values}) != len(values):
        raise ValueError("evaluator corpus IDs are not unique")
    if records_jsonl(values) != payload:
        raise ValueError("evaluator corpus JSONL is not canonical")


def index_digest(indices: Sequence[int]) -> str:
    payload = json.dumps(list(indices), separators=(",", ":")).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def flagship_selection() -> FlagshipSelection:
    value = _manifest()["selection"]
    strata = tuple(
        SelectionStratum(
            item["minimum_prompt_tokens"],
            item["maximum_prompt_tokens"],
            item["sample_count"],
        )
        for item in value["strata"]
    )
    selection = FlagshipSelection(
        seed=value["seed"],
        eligible_record_count=value["eligible_record_count"],
        strata=strata,
        indices=tuple(value["indices"]),
        indices_sha256=value["indices_sha256"],
        selected_source_lines_sha256=value["selected_source_lines_sha256"],
    )
    if index_digest(selection.indices) != selection.indices_sha256:
        raise ValueError("flagship corpus indices do not match their digest")
    return selection


def flagship_source() -> dict[str, Any]:
    value = _manifest()["source"]
    if not isinstance(value, dict):
        raise ValueError("flagship corpus source metadata is invalid")
    return value


def corpus_provenance(profile: CorpusProfile) -> dict[str, Any]:
    value: dict[str, Any] = {
        "profile_id": profile.profile_id,
        "dataset_name": profile.dataset_name,
        "dataset_sha256": profile.dataset_sha256,
        "record_count": profile.record_count,
    }
    if profile.key == "flagship":
        manifest_payload = _FLAGSHIP_MANIFEST.read_bytes()
        manifest = _manifest()
        selection = manifest["selection"]
        value.update(
            {
                "manifest_sha256": hashlib.sha256(manifest_payload).hexdigest(),
                "source": manifest["source"],
                "selection": {
                    key: selection[key]
                    for key in (
                        "criteria",
                        "eligible_record_count",
                        "indices_sha256",
                        "seed",
                        "selected_source_lines_sha256",
                        "strata",
                    )
                },
            }
        )
    return value


def load_flagship_source(path: Path | None = None) -> bytes:
    source = flagship_source()
    expected_length = source["byte_length"]
    if path is not None:
        nofollow = getattr(os, "O_NOFOLLOW", None)
        if not isinstance(nofollow, int):
            raise RuntimeError("secure benchmark source loading is unavailable")
        try:
            descriptor = os.open(path, os.O_RDONLY | nofollow)
        except OSError as exc:
            raise RuntimeError("benchmark source could not be opened safely") from exc
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode) or before.st_size != expected_length:
                raise RuntimeError("benchmark source does not have its pinned size")
            chunks: list[bytes] = []
            remaining = expected_length + 1
            while remaining:
                chunk = os.read(descriptor, min(1024 * 1024, remaining))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            after = os.fstat(descriptor)
            identity = lambda value: (  # noqa: E731 - stable file projection
                value.st_dev,
                value.st_ino,
                value.st_size,
                value.st_mtime_ns,
                value.st_ctime_ns,
            )
            if identity(before) != identity(after):
                raise RuntimeError("benchmark source changed while being read")
            payload = b"".join(chunks)
        finally:
            os.close(descriptor)
    else:
        request = urllib.request.Request(
            source["url"], headers={"User-Agent": "invarlock-evaluator-corpus/1"}
        )
        with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310
            chunks = []
            total = 0
            while total <= expected_length:
                chunk = response.read(min(1024 * 1024, expected_length + 1 - total))
                if not chunk:
                    break
                chunks.append(chunk)
                total += len(chunk)
            payload = b"".join(chunks)
    if (
        len(payload) != expected_length
        or hashlib.sha256(payload).hexdigest() != source["sha256"]
    ):
        raise RuntimeError("benchmark source does not match its pinned identity")
    return payload


def derive_selected_indices(
    candidates: Sequence[SelectionCandidate],
    strata: Sequence[SelectionStratum],
    *,
    seed: str,
) -> tuple[int, ...]:
    for previous, current in zip(strata, strata[1:], strict=False):
        if previous.maximum_prompt_tokens >= current.minimum_prompt_tokens:
            raise ValueError("selection strata overlap")
    selected: list[int] = []
    for stratum in strata:
        eligible = [
            candidate
            for candidate in candidates
            if stratum.minimum_prompt_tokens
            <= candidate.prompt_token_count
            <= stratum.maximum_prompt_tokens
        ]
        if len(eligible) < stratum.sample_count:
            raise ValueError("selection stratum lacks enough eligible records")
        eligible.sort(
            key=lambda candidate: hashlib.sha256(
                (
                    f"{seed}:{candidate.source_index}:{candidate.source_line_sha256}"
                ).encode("ascii")
            ).digest()
        )
        selected.extend(
            candidate.source_index for candidate in eligible[: stratum.sample_count]
        )
    if len(selected) != len(set(selected)):
        raise ValueError("selection strata produced duplicate records")
    return tuple(sorted(selected))


def project_lambada_records(
    source_payload: bytes, indices: Sequence[int]
) -> list[dict[str, str]]:
    lines = source_payload.splitlines()
    records: list[dict[str, str]] = []
    for index in indices:
        if index < 0 or index >= len(lines):
            raise ValueError("flagship source index is outside the pinned source")
        try:
            value = json.loads(lines[index])
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("flagship source row is not valid JSON") from exc
        if (
            not isinstance(value, dict)
            or set(value) != {"text"}
            or not isinstance(value["text"], str)
        ):
            raise ValueError("flagship source row has an invalid text field")
        prompt, separator, final_word = value["text"].rpartition(" ")
        if not prompt or separator != " " or not final_word:
            raise ValueError("flagship source row lacks a final-word boundary")
        records.append(
            {
                "expected": separator + final_word,
                "id": f"lambada-openai-{index:04d}",
                "prompt": prompt,
            }
        )
    return records


def flagship_records(
    source_payload: bytes, tokenizers: Sequence[Any]
) -> list[dict[str, str]]:
    source = flagship_source()
    if (
        len(source_payload) != source["byte_length"]
        or hashlib.sha256(source_payload).hexdigest() != source["sha256"]
    ):
        raise ValueError("flagship source does not match its pinned identity")
    lines = source_payload.splitlines()
    if len(lines) != source["record_count"] or len(tokenizers) != 2:
        raise ValueError("flagship source or tokenizer set is incomplete")
    selection = flagship_selection()
    candidates: list[SelectionCandidate] = []
    for index, raw in enumerate(lines):
        try:
            value = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("flagship source row is not valid JSON") from exc
        if (
            not isinstance(value, dict)
            or set(value) != {"text"}
            or not isinstance(value["text"], str)
        ):
            raise ValueError("flagship source row has an invalid text field")
        prompt, separator, final_word = value["text"].rpartition(" ")
        if not prompt or separator != " " or not final_word:
            continue
        target = separator + final_word
        target_ids = [
            tokenizer(target, add_special_tokens=False)["input_ids"]
            for tokenizer in tokenizers
        ]
        if any(len(token_ids) != 1 for token_ids in target_ids):
            continue
        if any(
            tokenizer.decode(
                token_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            != target
            for tokenizer, token_ids in zip(tokenizers, target_ids, strict=True)
        ):
            continue
        prompt_token_count = max(
            len(tokenizer(prompt, add_special_tokens=True)["input_ids"])
            for tokenizer in tokenizers
        )
        if prompt_token_count <= corpus_profile("flagship").context_length:
            candidates.append(
                SelectionCandidate(
                    source_index=index,
                    prompt_token_count=prompt_token_count,
                    source_line_sha256=hashlib.sha256(raw).hexdigest(),
                )
            )
    indices = derive_selected_indices(candidates, selection.strata, seed=selection.seed)
    if (
        len(candidates) != selection.eligible_record_count
        or indices != selection.indices
    ):
        raise ValueError("flagship corpus selection does not match the frozen manifest")
    selected_line_digest = hashlib.sha256(
        "".join(hashlib.sha256(lines[index]).hexdigest() for index in indices).encode(
            "ascii"
        )
    ).hexdigest()
    if selected_line_digest != selection.selected_source_lines_sha256:
        raise ValueError("flagship selected source lines do not match their digest")
    records = project_lambada_records(source_payload, indices)
    payload = records_jsonl(records)
    profile = corpus_profile("flagship")
    if hashlib.sha256(payload).hexdigest() != profile.dataset_sha256:
        raise ValueError("flagship derived dataset does not match its pinned digest")
    validate_dataset_records(payload, profile)
    return records


def quick_records() -> list[dict[str, str]]:
    values = json.loads(_QUICK_RECORDS.read_text(encoding="utf-8"))
    payload = records_jsonl(values)
    validate_dataset_records(payload, corpus_profile("quick"))
    return cast(list[dict[str, str]], values)


__all__ = [
    "CorpusProfile",
    "FlagshipSelection",
    "SelectionCandidate",
    "SelectionStratum",
    "corpus_profile",
    "corpus_provenance",
    "derive_selected_indices",
    "flagship_selection",
    "flagship_records",
    "flagship_source",
    "index_digest",
    "load_flagship_source",
    "profile_for_dataset",
    "profile_for_descriptor",
    "project_lambada_records",
    "quick_records",
    "records_jsonl",
    "validate_dataset_records",
]
