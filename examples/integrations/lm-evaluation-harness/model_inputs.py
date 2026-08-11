#!/usr/bin/env python3
"""Stage byte-pinned Qwen3 snapshots and author the Harness comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import yaml
from transformers import AutoTokenizer

from invarlock.core.checkpoint_identity import checkpoint_tree_sha256
from invarlock.runtime_providers.hf_transformers import hf_tokenizer_contract_sha256

DATASET_NAME = "qwen3-0.6b-base-to-post-trained"
MAX_GENERATION_TOKENS = 1
HARNESS_BATCH_SIZE = 8
RECORDS = Path(__file__).with_name("records.json")


@dataclass(frozen=True, slots=True)
class SnapshotFile:
    name: str
    byte_length: int
    sha256: str


@dataclass(frozen=True, slots=True)
class Snapshot:
    role: str
    repository: str
    revision: str
    files: tuple[SnapshotFile, ...]

    def url(self, filename: str) -> str:
        return (
            f"https://huggingface.co/{self.repository}/resolve/"
            f"{self.revision}/{filename}"
        )


_SHARED = (
    SnapshotFile(
        "merges.txt",
        1_671_853,
        "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
    ),
    SnapshotFile(
        "vocab.json",
        2_776_833,
        "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
    ),
)
SNAPSHOTS = (
    Snapshot(
        "baseline",
        "Qwen/Qwen3-0.6B-Base",
        "da87bfb608c14b7cf20ba1ce41287e8de496c0cd",
        (
            SnapshotFile(
                "config.json",
                727,
                "504a6b58c4271583724e66584b6b7698aea18450209df6b2f7582df0e89cee59",
            ),
            *_SHARED,
            SnapshotFile(
                "model.safetensors",
                1_192_135_096,
                "cd2a512003e2f9f3cd3c32a9c3573f820bb28c940f73c57b1ddaa983d9223eba",
            ),
            SnapshotFile(
                "tokenizer.json",
                7_031_645,
                "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539",
            ),
            SnapshotFile(
                "tokenizer_config.json",
                9_678,
                "3c04ed3ca964ea2f6b2b5faf0dc4d31aec1cb1e8b4bcf63f402d295046b422b5",
            ),
        ),
    ),
    Snapshot(
        "subject",
        "Qwen/Qwen3-0.6B",
        "c1899de289a04d12100db370d81485cdf75e47ca",
        (
            SnapshotFile(
                "config.json",
                726,
                "660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd",
            ),
            *_SHARED,
            SnapshotFile(
                "model.safetensors",
                1_503_300_328,
                "f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b",
            ),
            SnapshotFile(
                "tokenizer.json",
                11_422_654,
                "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
            ),
            SnapshotFile(
                "tokenizer_config.json",
                9_732,
                "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101",
            ),
        ),
    ),
)


def _download(destination: Path, snapshot: Snapshot, item: SnapshotFile) -> None:
    partial = destination.with_suffix(destination.suffix + ".partial")
    digest = hashlib.sha256()
    length = 0
    request = urllib.request.Request(
        snapshot.url(item.name), headers={"User-Agent": "invarlock-harness-example/1"}
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310
            with partial.open("xb") as output:
                while chunk := response.read(1024 * 1024):
                    if not isinstance(chunk, bytes):
                        raise RuntimeError("snapshot download did not return bytes")
                    length += len(chunk)
                    if length > item.byte_length:
                        raise RuntimeError("snapshot download exceeds its pinned size")
                    digest.update(chunk)
                    output.write(chunk)
        if length != item.byte_length or digest.hexdigest() != item.sha256:
            raise RuntimeError(f"downloaded {snapshot.role}/{item.name} is not pinned")
        partial.chmod(0o644)
        partial.replace(destination)
    except Exception:
        partial.unlink(missing_ok=True)
        raise


def stage_snapshot(root: Path, snapshot: Snapshot) -> Path:
    destination = root / snapshot.role
    destination.mkdir(mode=0o755)
    try:
        for item in snapshot.files:
            _download(destination / item.name, snapshot, item)
        config = json.loads((destination / "config.json").read_text(encoding="utf-8"))
        if config.get("model_type") != "qwen3":
            raise RuntimeError(f"{snapshot.role} snapshot is not a Qwen3 checkpoint")
    except Exception:
        shutil.rmtree(destination)
        raise
    return destination


def stage_snapshots(root: Path) -> dict[str, Path]:
    root.mkdir(parents=True)
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="qwen3-snapshot") as pool:
        return dict(
            pool.map(
                lambda snapshot: (snapshot.role, stage_snapshot(root, snapshot)),
                SNAPSHOTS,
            )
        )


def _records() -> list[dict[str, str]]:
    values = json.loads(RECORDS.read_text(encoding="utf-8"))
    if not isinstance(values, list) or len(values) != 102:
        raise RuntimeError("the Qwen3 Harness journey requires exactly 102 records")
    if any(
        not isinstance(value, dict)
        or set(value) != {"expected", "id", "prompt"}
        or any(not isinstance(value[key], str) or not value[key] for key in value)
        for value in values
    ):
        raise RuntimeError("the Qwen3 Harness records are invalid")
    if len({value["id"] for value in values}) != len(values):
        raise RuntimeError("the Qwen3 Harness record IDs are not unique")
    return values


def prepare(root: Path, image_id: str) -> None:
    if root.exists() or root.is_symlink():
        raise RuntimeError("prepared workspace must be new")
    inputs = root / "evaluation/inputs"
    inputs.mkdir(parents=True)
    models = stage_snapshots(root / "evaluation/models")
    records = _records()
    records_path = inputs / "records.jsonl"
    records_path.write_text(
        "".join(json.dumps(value, sort_keys=True) + "\n" for value in records),
        encoding="utf-8",
    )
    dataset_sha256 = hashlib.sha256(records_path.read_bytes()).hexdigest()
    policy = {
        "resolved_policy": {
            "metrics": {
                "exact_match": {
                    "delta_min_pp": -20.0,
                    "maximum_interval_width_pp": 20.0,
                    "minimum_record_count": 102,
                    "minimum_side_accuracy": 0.20,
                }
            }
        }
    }
    (inputs / "acceptance.json").write_text(
        json.dumps(policy, sort_keys=True) + "\n", encoding="utf-8"
    )
    sides: dict[str, object] = {}
    for snapshot in SNAPSHOTS:
        checkpoint = models[snapshot.role]
        tokenizer = AutoTokenizer.from_pretrained(
            checkpoint, local_files_only=True, trust_remote_code=False
        )
        for record in records:
            token_ids = tokenizer(record["expected"], add_special_tokens=False)[
                "input_ids"
            ]
            decoded = tokenizer.decode(
                token_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            if len(token_ids) > MAX_GENERATION_TOKENS or decoded != record["expected"]:
                raise RuntimeError(
                    f"{snapshot.role} target exceeds the lossless generation bound"
                )
        sides[snapshot.role] = {
            "artifact": {
                "path": f"models/{snapshot.role}",
                "model_id": snapshot.repository,
                "locator": f"hf://{snapshot.repository}@{snapshot.revision}",
            },
            "runtime": {
                "provider": "hf_transformers",
                "settings": {
                    "batch_size": HARNESS_BATCH_SIZE,
                    "checkpoint_tree_sha256": checkpoint_tree_sha256(checkpoint),
                    "context_length": 64,
                    "max_output_tokens": MAX_GENERATION_TOKENS,
                    "offline": True,
                    "seed": 20_260_716,
                    "timeout_seconds": 300,
                    "tokenizer_metadata_sha256": hf_tokenizer_contract_sha256(
                        tokenizer
                    ),
                },
            },
        }
    request = {
        "format_version": "invarlock/evaluation-request-v1",
        "comparison": {
            "baseline": sides["baseline"],
            "subject": sides["subject"],
            "dataset": {
                "path": "inputs/records.jsonl",
                "sha256": dataset_sha256,
                "format": "jsonl",
                "name": DATASET_NAME,
                "split": "validation",
                "input_field": "prompt",
                "expected_output_field": "expected",
                "id_field": "id",
            },
            "policy": "inputs/acceptance.json",
            "task": "text_causal",
            "metric": "exact_match",
        },
        "execution": {"mode": "run"},
        "output": {"evidence": "evidence"},
    }
    (root / "evaluation/request.yaml").write_text(
        yaml.safe_dump(request, sort_keys=False), encoding="utf-8"
    )
    (root / "runtime-image-id.txt").write_text(image_id + "\n", encoding="ascii")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--runtime-image", required=True)
    arguments = parser.parse_args()
    prepare(
        Path(os.path.abspath(arguments.workspace.expanduser())),
        arguments.runtime_image,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
