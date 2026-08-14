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
MODEL_ID = "Qwen/Qwen3.5-0.8B"
MODEL_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
MODEL_FILES = {
    "chat_template.jinja": (
        7_755,
        "273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80",
    ),
    "config.json": (
        2_907,
        "b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204",
    ),
    "merges.txt": (
        3_353_259,
        "a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d",
    ),
    "model.safetensors.index.json": (
        50_900,
        "d8a08838a613b025eb7952ed9db11696213e57e76a375661ef5c12f9dd5dcf4e",
    ),
    "model.safetensors-00001-of-00001.safetensors": (
        1_746_942_600,
        "04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696",
    ),
    "tokenizer.json": (
        12_807_982,
        "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
    ),
    "tokenizer_config.json": (
        16_709,
        "49e2b6e395f959f077f1e992b338919c0d4a9732fc6e613995e06557f843500c",
    ),
    "vocab.json": (
        6_722_759,
        "ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003",
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
        "schedule_id": "qwen35-0.8b-102-record-authoritative-import",
    }


def generate(*, local_files_only: bool) -> dict[str, bytes]:
    records = _source_records()
    dataset = _dataset(records)
    request = LocalDatasetRequest(
        path=ROOT / "dataset.jsonl",
        sha256=hashlib.sha256(dataset).hexdigest(),
        format="jsonl",
        name="qwen35-0.8b-authoritative-evaluator-import",
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
            "source_evaluation": {
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
