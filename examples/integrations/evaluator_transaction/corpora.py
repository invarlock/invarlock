"""Closed corpus profiles shared by maintained evaluator transactions."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

_ROOT = Path(__file__).resolve().parent
_REPOSITORY = _ROOT.parents[2]
_QUICK_RECORDS = _ROOT.parent / "lm-evaluation-harness" / "records.json"
_DEPLOYMENT_RECORDS = _ROOT / "lambada_qwen35_deployment_400.jsonl"
_DEPLOYMENT_MANIFEST = _ROOT / "deployment_corpus.json"
_SEMANTIC_RECORDS = _ROOT / "mmlu_pro_semantic_400.jsonl"
_QUALIFICATION_PROFILES = _ROOT / "qualification_profiles.json"
_QUALIFICATION_MANIFEST = (
    _REPOSITORY / "docs/reference/qualification-suites.manifest.json"
)
_DEPLOYMENT_MANIFEST_BYTES = 6_644
_DEPLOYMENT_MANIFEST_SHA256 = (
    "a774b4369658d6f6c4910b03968008c59e55a3cd04b3737c34373074f583df77"
)
PROFILE_KEYS = ("quick", "deployment", "flagship", "portability")
_INDEPENDENT_CANARY_KEY = "independent-canary"
_INDEPENDENT_CANARY_BYTES = 379_746
_INDEPENDENT_CANARY_SHA256 = (
    "c3d83209d6f36023f0a5aef5ee9be895891cc66ecc1b7196e83227558a38fade"
)
_QWEN38_27B_KEY = "qwen38-27b"
_QWEN38_27B_BYTES = 405_346
_QWEN38_27B_SHA256 = "c3f083ae0443648dc749f16df5bb1f5cd4531e0227903d3e86f8b853f54bd6cb"


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
    delta_min_pp: float

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
                        "delta_min_pp": self.delta_min_pp,
                        "maximum_interval_width_pp": self.maximum_interval_width_pp,
                        "minimum_record_count": self.record_count,
                        "minimum_side_accuracy": self.minimum_side_accuracy,
                    }
                }
            }
        }


def _json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is unavailable or invalid") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain an object")
    return value


def _manifest() -> dict[str, Any]:
    return _json(_QUALIFICATION_PROFILES, label="qualification profile manifest")


def _qualification_manifest() -> dict[str, Any]:
    manifest = _manifest()
    payload = _QUALIFICATION_MANIFEST.read_bytes()
    expected = manifest["qualification_suite"]["manifest_sha256"]
    if hashlib.sha256(payload).hexdigest() != expected:
        raise ValueError(
            "qualification suite manifest does not match its pinned digest"
        )
    return _json(_QUALIFICATION_MANIFEST, label="qualification suite manifest")


def records_jsonl(records: Iterable[dict[str, str]], *, compact: bool = False) -> bytes:
    options: dict[str, Any] = {"sort_keys": True}
    if compact:
        options.update(separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return b"".join(
        (json.dumps(record, **options) + "\n").encode("utf-8") for record in records
    )


def _quick_profile() -> CorpusProfile:
    return CorpusProfile(
        key="quick",
        profile_id="causal-cloze-102-v1",
        dataset_name="qwen35-0.8b-base-to-post-trained",
        split="validation",
        record_count=102,
        dataset_sha256="d80e81ba17fb93b9b8a46f9817f9841f5f9c2858c9d703b3ce28847b2eaeb57c",
        context_length=64,
        minimum_side_accuracy=0.20,
        maximum_interval_width_pp=20.0,
        delta_min_pp=-20.0,
    )


def _deployment_profile() -> CorpusProfile:
    return CorpusProfile(
        key="deployment",
        profile_id="lambada-openai-qwen35-0.8b-400-v1",
        dataset_name="lambada-openai-qwen35-0.8b-one-token-400-v1",
        split="test-stratified-400",
        record_count=400,
        dataset_sha256=(
            "e4a0e431b8b64130cbbf6e8fb3ed7b5769744d18ca6499d2088f2e1b3fb36dda"
        ),
        context_length=256,
        minimum_side_accuracy=0.05,
        maximum_interval_width_pp=10.0,
        delta_min_pp=-20.0,
    )


def _qualification_profile(key: str) -> CorpusProfile:
    manifest = _manifest()
    declared = manifest["profiles"][key]
    dataset = declared["derived_dataset"]
    policy = manifest["acceptance_policy"]
    return CorpusProfile(
        key=key,
        profile_id=declared["profile_id"],
        dataset_name=dataset["name"],
        split=dataset["split"],
        record_count=dataset["record_count"],
        dataset_sha256=dataset["sha256"],
        context_length=dataset["maximum_input_tokens"],
        minimum_side_accuracy=policy["minimum_side_accuracy"],
        maximum_interval_width_pp=policy["maximum_interval_width_pp"],
        delta_min_pp=policy["delta_min_pp"],
    )


def corpus_profile(key: str) -> CorpusProfile:
    factories = {
        "quick": _quick_profile,
        "deployment": _deployment_profile,
        "flagship": lambda: _qualification_profile("flagship"),
        "portability": lambda: _qualification_profile("portability"),
    }
    try:
        factory = factories[key]
    except KeyError as exc:
        raise ValueError(f"unknown corpus profile: {key}") from exc
    return factory()


def independent_canary_corpus_profile() -> CorpusProfile:
    """Return the closed Mistral rendering used by the deployment canary.

    The profile intentionally stays outside ``PROFILE_KEYS``: it is a
    deployment-format canary, not another evaluator-qualification matrix lane.
    It reuses the same selected semantic records and frozen acceptance policy as
    the maintained GPU profiles.
    """

    manifest = _manifest()
    qualification = _qualification_manifest()
    policy = manifest["acceptance_policy"]
    return CorpusProfile(
        key=_INDEPENDENT_CANARY_KEY,
        profile_id="mmlu-pro-ministral3-instruct-400-v1",
        dataset_name="TIGER-Lab/MMLU-Pro/ministral3-instruct",
        split="test-balanced-400",
        record_count=qualification["record_count"],
        dataset_sha256=_INDEPENDENT_CANARY_SHA256,
        context_length=1024,
        minimum_side_accuracy=policy["minimum_side_accuracy"],
        maximum_interval_width_pp=policy["maximum_interval_width_pp"],
        delta_min_pp=policy["delta_min_pp"],
    )


def qwen38_27b_corpus_profile() -> CorpusProfile:
    """Return the closed Qwen3.8 rendering used by its deployment profile.

    Qwen3.8 uses the same authenticated no-thinking ChatML rendering as the
    maintained Qwen3.5 profile. It remains deployment-only and is not returned
    by ``PROFILE_KEYS``.
    """

    manifest = _manifest()
    qualification = _qualification_manifest()
    policy = manifest["acceptance_policy"]
    return CorpusProfile(
        key=_QWEN38_27B_KEY,
        profile_id="mmlu-pro-qwen38-no-think-400-v1",
        dataset_name="TIGER-Lab/MMLU-Pro/qwen38-no-think",
        split="test-balanced-400",
        record_count=qualification["record_count"],
        dataset_sha256=_QWEN38_27B_SHA256,
        context_length=1024,
        minimum_side_accuracy=policy["minimum_side_accuracy"],
        maximum_interval_width_pp=policy["maximum_interval_width_pp"],
        delta_min_pp=policy["delta_min_pp"],
    )


def _canonical_payload(values: list[dict[str, str]], profile: CorpusProfile) -> bytes:
    return records_jsonl(
        values,
        compact=profile.key
        in {
            "flagship",
            "portability",
            _INDEPENDENT_CANARY_KEY,
            _QWEN38_27B_KEY,
        },
    )


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
    records = cast(list[dict[str, str]], values)
    if len({value["id"] for value in records}) != len(records):
        raise ValueError("evaluator corpus IDs are not unique")
    if _canonical_payload(records, profile) != payload:
        raise ValueError("evaluator corpus JSONL is not canonical")


def profile_for_dataset(payload: bytes) -> CorpusProfile:
    observed = hashlib.sha256(payload).hexdigest()
    for key in PROFILE_KEYS:
        profile = corpus_profile(key)
        if observed == profile.dataset_sha256:
            validate_dataset_records(payload, profile)
            return profile
    raise ValueError("dataset is not a pinned evaluator corpus")


def profile_for_descriptor(value: object) -> CorpusProfile:
    for key in PROFILE_KEYS:
        profile = corpus_profile(key)
        if value == profile.dataset_descriptor():
            return profile
    raise ValueError("dataset descriptor is not a pinned evaluator corpus")


def corpus_provenance(profile: CorpusProfile) -> dict[str, Any]:
    value: dict[str, Any] = {
        "profile_id": profile.profile_id,
        "dataset_name": profile.dataset_name,
        "dataset_sha256": profile.dataset_sha256,
        "record_count": profile.record_count,
    }
    if profile.key == "deployment":
        manifest = _deployment_manifest()
        records = deployment_records()
        expected_ids = [
            f"lambada-openai-{index:04d}" for index in manifest["selection"]["indices"]
        ]
        if (
            manifest["profile_id"] != profile.profile_id
            or manifest["derived_dataset"]["sha256"] != profile.dataset_sha256
            or expected_ids != [record["id"] for record in records]
        ):
            raise ValueError(
                "deployment corpus disagrees with its pinned selection manifest"
            )
        value.update(
            {
                "source": manifest["source"],
                "selection": manifest["selection"],
                "selection_manifest": {
                    "path": _DEPLOYMENT_MANIFEST.name,
                    "byte_length": _DEPLOYMENT_MANIFEST_BYTES,
                    "sha256": _DEPLOYMENT_MANIFEST_SHA256,
                },
                "model_profile": "qwen35-0.8b-base-to-post-trained-bf16-v1",
            }
        )
    elif profile.key != "quick":
        manifest = _manifest()
        qualification = _qualification_manifest()
        artifact_name = manifest["qualification_suite"]["semantic_artifact"]
        artifact = qualification["artifacts"][artifact_name]
        if profile.key == _INDEPENDENT_CANARY_KEY:
            records = independent_canary_records()
            rendering = {
                "algorithm": "mistral3-instruct-v1",
                "bos": "runtime-added",
                "suffix": "[/INST]",
            }
            model_profile_id = "ministral3-8b-instruct-bf16-to-q5-k-m-v1"
        elif profile.key == _QWEN38_27B_KEY:
            records = qwen38_27b_records()
            rendering = {
                "algorithm": "qwen-chatml-disable-thinking-v1",
                "suffix": "<think>\n\n</think>\n\n",
            }
            model_profile_id = "qwen38-27b-bf16-to-q5-k-m-v1"
        else:
            records = qualification_records(profile)
            declared = manifest["profiles"][profile.key]
            rendering = declared["rendering"]
            model_profile_id = declared["model_profile"]
        if (
            artifact["sha256"] != manifest["qualification_suite"]["semantic_sha256"]
            or qualification["record_count"] != profile.record_count
            or qualification["selection_algorithm"]
            != manifest["qualification_suite"]["selection_algorithm"]
            or qualification["selected_ids"]["text"]
            != [record["id"] for record in records]
        ):
            raise ValueError("flagship corpus disagrees with its qualification suite")
        value.update(
            {
                "source": manifest["source"],
                "qualification_suite": manifest["qualification_suite"],
                "rendering": rendering,
                "model_profile": model_profile_id,
            }
        )
    return value


def _read_regular_file(path: Path, *, expected_bytes: int) -> bytes:
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if not isinstance(nofollow, int):
        raise RuntimeError("secure bundled corpus loading is unavailable")
    try:
        descriptor = os.open(path, os.O_RDONLY | nofollow)
    except OSError as exc:
        raise RuntimeError(
            "bundled evaluator corpus could not be opened safely"
        ) from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size != expected_bytes:
            raise RuntimeError("bundled evaluator corpus does not have its pinned size")
        chunks: list[bytes] = []
        remaining = expected_bytes + 1
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        after = os.fstat(descriptor)
        identity = lambda item: (  # noqa: E731 - stable file projection
            item.st_dev,
            item.st_ino,
            item.st_size,
            item.st_mtime_ns,
            item.st_ctime_ns,
        )
        if identity(before) != identity(after):
            raise RuntimeError("bundled evaluator corpus changed while being read")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _deployment_manifest() -> dict[str, Any]:
    payload = _read_regular_file(
        _DEPLOYMENT_MANIFEST, expected_bytes=_DEPLOYMENT_MANIFEST_BYTES
    )
    if hashlib.sha256(payload).hexdigest() != _DEPLOYMENT_MANIFEST_SHA256:
        raise RuntimeError(
            "deployment corpus manifest does not match its pinned identity"
        )
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("deployment corpus manifest is invalid") from exc
    if not isinstance(value, dict):
        raise RuntimeError("deployment corpus manifest must contain an object")
    return value


def deployment_records() -> list[dict[str, str]]:
    manifest = _deployment_manifest()
    dataset = manifest["derived_dataset"]
    payload = _read_regular_file(
        _DEPLOYMENT_RECORDS, expected_bytes=dataset["byte_length"]
    )
    if hashlib.sha256(payload).hexdigest() != dataset["sha256"]:
        raise RuntimeError(
            "bundled deployment corpus does not match its pinned identity"
        )
    profile = _deployment_profile()
    validate_dataset_records(payload, profile)
    records = cast(
        list[dict[str, str]], [json.loads(line) for line in payload.splitlines()]
    )
    expected_ids = [
        f"lambada-openai-{index:04d}" for index in manifest["selection"]["indices"]
    ]
    if [record["id"] for record in records] != expected_ids:
        raise RuntimeError("deployment corpus does not match its selected source rows")
    return records


def _semantic_records() -> list[dict[str, Any]]:
    suite = _manifest()["qualification_suite"]
    payload = _read_regular_file(
        _SEMANTIC_RECORDS, expected_bytes=suite["semantic_byte_length"]
    )
    if hashlib.sha256(payload).hexdigest() != suite["semantic_sha256"]:
        raise RuntimeError("bundled semantic corpus does not match its pinned identity")
    records = [json.loads(line) for line in payload.splitlines()]
    if len(records) != 400 or any(
        not isinstance(record, dict)
        or set(record)
        != {"answer", "id", "options", "question", "semantic_sha256", "source"}
        for record in records
    ):
        raise RuntimeError("bundled semantic corpus is incomplete")
    return cast(list[dict[str, Any]], records)


def _question_body(record: dict[str, Any]) -> str:
    options = record["options"]
    if not isinstance(options, list) or not 2 <= len(options) <= 10:
        raise RuntimeError("bundled semantic corpus has invalid choices")
    choices = "\n".join(
        f"{chr(ord('A') + index)}. {option}" for index, option in enumerate(options)
    )
    return f"Question: {record['question']}\nChoices:\n{choices}"


def _render_record(record: dict[str, Any], profile: CorpusProfile) -> dict[str, str]:
    body = _question_body(record)
    instruction = "Reply with exactly one uppercase option letter and no other text."
    if profile.key in {"flagship", _QWEN38_27B_KEY}:
        prompt = (
            "<|im_start|>system\nYou answer multiple-choice questions and follow "
            "the requested output format exactly.<|im_end|>\n"
            f"<|im_start|>user\n{body}\n{instruction}<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        )
    elif profile.key == _INDEPENDENT_CANARY_KEY:
        prompt = (
            "[SYSTEM_PROMPT]You answer multiple-choice questions and follow "
            "the requested output format exactly.[/SYSTEM_PROMPT]"
            f"[INST]{body}\n{instruction}[/INST]"
        )
    elif profile.key == "portability":
        prompt = (
            "<bos><|turn>system\nYou answer multiple-choice questions and follow "
            "the requested output format exactly.<turn|>\n"
            f"<|turn>user\n{body}\n{instruction}<turn|>\n"
            "<|turn>model\n<|channel>thought\n<channel|>"
        )
    else:
        raise ValueError(f"profile does not use the semantic corpus: {profile.key}")
    return {
        "id": str(record["id"]),
        "prompt": prompt,
        "expected": str(record["answer"]),
    }


def qualification_records(profile: CorpusProfile) -> list[dict[str, str]]:
    if profile.key == "deployment":
        return deployment_records()
    if profile.key not in {"flagship", "portability"}:
        raise ValueError("qualification records require a maintained GPU profile")
    records = [_render_record(record, profile) for record in _semantic_records()]
    payload = records_jsonl(records, compact=True)
    expected_length = _manifest()["profiles"][profile.key]["derived_dataset"][
        "byte_length"
    ]
    if (
        len(payload) != expected_length
        or hashlib.sha256(payload).hexdigest() != profile.dataset_sha256
    ):
        raise RuntimeError(
            "rendered evaluator corpus does not match its pinned identity"
        )
    validate_dataset_records(payload, profile)
    return records


def independent_canary_records() -> list[dict[str, str]]:
    """Render the shared semantic suite through the closed Mistral canary form."""

    profile = independent_canary_corpus_profile()
    records = [_render_record(record, profile) for record in _semantic_records()]
    payload = records_jsonl(records, compact=True)
    if (
        len(payload) != _INDEPENDENT_CANARY_BYTES
        or hashlib.sha256(payload).hexdigest() != profile.dataset_sha256
    ):
        raise RuntimeError(
            "rendered independent-canary corpus does not match its pinned identity"
        )
    validate_dataset_records(payload, profile)
    return records


def qwen38_27b_records() -> list[dict[str, str]]:
    """Render the shared semantic suite through the closed Qwen3.8 format."""

    profile = qwen38_27b_corpus_profile()
    records = [_render_record(record, profile) for record in _semantic_records()]
    payload = records_jsonl(records, compact=True)
    if (
        len(payload) != _QWEN38_27B_BYTES
        or hashlib.sha256(payload).hexdigest() != profile.dataset_sha256
    ):
        raise RuntimeError("rendered Qwen3.8 27B corpus does not match its identity")
    validate_dataset_records(payload, profile)
    return records


def flagship_records() -> list[dict[str, str]]:
    return qualification_records(corpus_profile("flagship"))


def quick_records() -> list[dict[str, str]]:
    values = json.loads(_QUICK_RECORDS.read_text(encoding="utf-8"))
    payload = records_jsonl(values)
    validate_dataset_records(payload, corpus_profile("quick"))
    return cast(list[dict[str, str]], values)


__all__ = [
    "CorpusProfile",
    "PROFILE_KEYS",
    "corpus_profile",
    "corpus_provenance",
    "deployment_records",
    "flagship_records",
    "independent_canary_corpus_profile",
    "independent_canary_records",
    "qualification_records",
    "profile_for_dataset",
    "profile_for_descriptor",
    "quick_records",
    "qwen38_27b_corpus_profile",
    "qwen38_27b_records",
    "records_jsonl",
    "validate_dataset_records",
]
