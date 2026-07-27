#!/usr/bin/env python3
"""Generate the retained 102-record model-output evaluation corpus."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import tempfile
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from invarlock.core.checkpoint_identity import checkpoint_tree_sha256
from invarlock.core.runtime_provider import (
    canonical_runtime_behavioral_schedule_json,
)
from invarlock.core.schedule_preparation import (
    LocalDatasetRequest,
    prepare_local_evaluation_schedule_bytes,
)
from invarlock.evidence_pack_contract import canonical_json_bytes

ROOT = Path(__file__).resolve().parent
SOURCE_RECORDS = (
    ROOT.parents[1] / "integrations" / "lm-evaluation-harness" / "records.json"
)
MODEL_ID = "Qwen/Qwen3-0.6B"
MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
MODEL_FILES = {
    "config.json": (
        726,
        "660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd",
    ),
    "merges.txt": (
        1_671_853,
        "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
    ),
    "model.safetensors": (
        1_503_300_328,
        "f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b",
    ),
    "tokenizer.json": (
        11_422_654,
        "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
    ),
    "tokenizer_config.json": (
        9_732,
        "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101",
    ),
    "vocab.json": (
        2_776_833,
        "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
    ),
}
GENERATION = {
    "backend": "transformers",
    "do_sample": False,
    "dtype": "float32",
    "max_new_tokens": 1,
    "seed": 0,
}


def _sha256(payload: bytes) -> str:
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _source_records() -> list[dict[str, str]]:
    value = json.loads(SOURCE_RECORDS.read_bytes())
    if (
        not isinstance(value, list)
        or len(value) != 102
        or any(
            not isinstance(record, dict)
            or set(record) != {"expected", "id", "prompt"}
            or any(not isinstance(item, str) or not item for item in record.values())
            for record in value
        )
    ):
        raise RuntimeError("source evaluation records are not the fixed 102-record set")
    if len({record["id"] for record in value}) != len(value):
        raise RuntimeError("source evaluation record IDs are not unique")
    return value


def _curated_snapshot(
    *, local_files_only: bool
) -> tuple[tempfile.TemporaryDirectory, Path]:
    source = Path(
        snapshot_download(
            repo_id=MODEL_ID,
            revision=MODEL_REVISION,
            allow_patterns=sorted(MODEL_FILES),
            local_files_only=local_files_only,
        )
    )
    temporary = tempfile.TemporaryDirectory(
        dir=ROOT,
        prefix=".authoritative-model-",
    )
    destination = Path(temporary.name)
    for name, (expected_size, expected_sha256) in MODEL_FILES.items():
        payload = (source / name).read_bytes()
        if len(payload) != expected_size or hashlib.sha256(payload).hexdigest() != (
            expected_sha256
        ):
            temporary.cleanup()
            raise RuntimeError(
                f"model file {name!r} does not match its pinned identity"
            )
        (destination / name).write_bytes(payload)
    return temporary, destination


def _dataset(records: list[dict[str, str]]) -> bytes:
    return b"".join(canonical_json_bytes(record) for record in records)


def _qualification_schedule(runtime_schedule: Any) -> dict[str, object]:
    return {
        "format": "invarlock/evaluator-qualification-schedule-v1",
        "records": [
            {
                "input_sha256": f"sha256:{record.input_sha256}",
                "record_id": record.record_id,
                "reference_output_sha256": _sha256(
                    record.expected_output.encode("utf-8")
                ),
            }
            for record in runtime_schedule.records
        ],
        "schedule_id": "qwen3-0.6b-102-record-authoritative-import",
    }


def generate(*, local_files_only: bool) -> dict[str, bytes]:
    records = _source_records()
    dataset = _dataset(records)
    request = LocalDatasetRequest(
        path=ROOT / "dataset.jsonl",
        sha256=hashlib.sha256(dataset).hexdigest(),
        format="jsonl",
        name="qwen3-0.6b-authoritative-evaluator-import",
        split="validation",
        input_field="prompt",
        expected_output_field="expected",
        id_field="id",
    )
    runtime_schedule = prepare_local_evaluation_schedule_bytes(request, dataset)
    temporary, snapshot = _curated_snapshot(local_files_only=local_files_only)
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            snapshot,
            local_files_only=True,
            trust_remote_code=False,
        )
        tokenizer.padding_side = "left"
        model = AutoModelForCausalLM.from_pretrained(
            snapshot,
            local_files_only=True,
            trust_remote_code=False,
            dtype=torch.float32,
        )
        model.eval()
        torch.manual_seed(GENERATION["seed"])
        outputs: list[str] = []
        for start in range(0, len(records), 8):
            batch = records[start : start + 8]
            encoded = tokenizer(
                [record["prompt"] for record in batch],
                return_tensors="pt",
                padding=True,
            )
            with torch.inference_mode():
                generated = model.generate(
                    **encoded,
                    do_sample=GENERATION["do_sample"],
                    max_new_tokens=GENERATION["max_new_tokens"],
                    pad_token_id=tokenizer.eos_token_id,
                )
            width = encoded["input_ids"].shape[1]
            outputs.extend(
                tokenizer.decode(
                    row[width:],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                for row in generated
            )
        if len(outputs) != len(records) or any(not output for output in outputs):
            raise RuntimeError("model execution did not produce one output per record")
        cases = {
            "format": "invarlock/evaluator-authoritative-cases-v1",
            "producer": {
                "dataset_sha256": _sha256(dataset),
                "generation": GENERATION,
                "generator_sha256": _sha256(Path(__file__).read_bytes()),
                "kind": "model_execution",
                "model": {
                    "files": [
                        {
                            "byte_length": size,
                            "name": name,
                            "sha256": f"sha256:{sha256}",
                        }
                        for name, (size, sha256) in sorted(MODEL_FILES.items())
                    ],
                    "immutable_revision": MODEL_REVISION,
                    "model_id": MODEL_ID,
                    "snapshot_tree_sha256": checkpoint_tree_sha256(snapshot),
                },
                "runtime": {
                    "torch": importlib.metadata.version("torch"),
                    "transformers": importlib.metadata.version("transformers"),
                },
            },
            "records": [
                {
                    "input": record["prompt"],
                    "input_sha256": f"sha256:{scheduled.input_sha256}",
                    "output": output,
                    "record_id": record["id"],
                    "reference": record["expected"],
                }
                for record, scheduled, output in zip(
                    records,
                    runtime_schedule.records,
                    outputs,
                    strict=True,
                )
            ],
        }
        return {
            "cases.json": canonical_json_bytes(cases),
            "dataset.jsonl": dataset,
            "runtime-schedule.json": canonical_runtime_behavioral_schedule_json(
                runtime_schedule
            ),
            "schedule.json": canonical_json_bytes(
                _qualification_schedule(runtime_schedule)
            ),
        }
    finally:
        temporary.cleanup()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--allow-download", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generated = generate(local_files_only=not args.allow_download)
    if args.check:
        for name, expected in generated.items():
            if (ROOT / name).read_bytes() != expected:
                raise RuntimeError(f"retained authoritative {name} is stale")
        print("verified retained 102-record model execution")
        return
    for name, payload in generated.items():
        (ROOT / name).write_bytes(payload)
    print("generated retained 102-record model execution")


if __name__ == "__main__":
    main()
