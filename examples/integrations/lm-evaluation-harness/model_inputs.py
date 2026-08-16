#!/usr/bin/env python3
"""Stage byte-pinned model snapshots and author the Harness comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import yaml
from transformers import AutoTokenizer

from invarlock.core.checkpoint_identity import checkpoint_tree_sha256
from invarlock.runtime_providers.hf_transformers import hf_tokenizer_contract_sha256

_REPOSITORY = Path(__file__).resolve().parents[3]
if str(_REPOSITORY) not in sys.path:
    # Resolve the repository's examples package before an unrelated installed
    # package named ``examples`` can occupy the import slot in direct-script use.
    sys.path.insert(0, str(_REPOSITORY))

try:
    from examples.integrations.evaluator_transaction.corpora import (
        PROFILE_KEYS,
        CorpusProfile,
        corpus_profile,
        corpus_provenance,
        qualification_records,
        quick_records,
        records_jsonl,
    )
    from examples.integrations.evaluator_transaction.model_profiles import (
        Snapshot,
        SnapshotFile,
        model_profile,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - direct-script execution
    if not exc.name or not exc.name.startswith("examples"):
        raise
    from examples.integrations.evaluator_transaction.corpora import (  # type: ignore[no-redef]
        PROFILE_KEYS,
        CorpusProfile,
        corpus_profile,
        corpus_provenance,
        qualification_records,
        quick_records,
        records_jsonl,
    )
    from examples.integrations.evaluator_transaction.model_profiles import (  # type: ignore[no-redef]
        Snapshot,
        SnapshotFile,
        model_profile,
    )

DATASET_NAME = corpus_profile("quick").dataset_name
MAX_GENERATION_TOKENS = 1
RECORDS = Path(__file__).with_name("records.json")
SNAPSHOTS = model_profile("quick").snapshots


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
        if config.get("model_type") != snapshot.model_type:
            raise RuntimeError(f"{snapshot.role} snapshot architecture is not pinned")
    except Exception:
        shutil.rmtree(destination)
        raise
    return destination


def stage_snapshots(root: Path, profile_key: str = "quick") -> dict[str, Path]:
    snapshots = model_profile(profile_key).snapshots
    root.mkdir(parents=True)
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="model-snapshot") as pool:
        return dict(
            pool.map(
                lambda snapshot: (snapshot.role, stage_snapshot(root, snapshot)),
                snapshots,
            )
        )


def _records(
    selected: CorpusProfile | None = None,
) -> list[dict[str, str]]:
    profile = selected or corpus_profile("quick")
    if profile.key == "quick":
        return quick_records()
    return qualification_records(profile)


def _validate_record_tokenization(
    records: list[dict[str, str]],
    tokenizer: Any,
    *,
    role: str,
    context_length: int,
) -> None:
    for record in records:
        prompt_ids = tokenizer(record["prompt"], add_special_tokens=True)["input_ids"]
        target_ids = tokenizer(record["expected"], add_special_tokens=False)[
            "input_ids"
        ]
        decoded = tokenizer.decode(
            target_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if len(target_ids) > MAX_GENERATION_TOKENS or decoded != record["expected"]:
            raise RuntimeError(f"{role} target exceeds the lossless generation bound")
        if len(prompt_ids) + len(target_ids) > context_length:
            raise RuntimeError(f"{role} prompt exceeds the pinned context length")


def prepare(
    root: Path,
    image_id: str,
    *,
    corpus_profile_key: str = "quick",
) -> None:
    if root.exists() or root.is_symlink():
        raise RuntimeError("prepared workspace must be new")
    selected = corpus_profile(corpus_profile_key)
    selected_models = model_profile(corpus_profile_key)
    inputs = root / "evaluation/inputs"
    inputs.mkdir(parents=True)
    models = stage_snapshots(root / "evaluation/models", corpus_profile_key)
    tokenizers = {
        role: AutoTokenizer.from_pretrained(
            checkpoint, local_files_only=True, trust_remote_code=False
        )
        for role, checkpoint in models.items()
    }
    records = _records(selected)
    records_path = inputs / "records.jsonl"
    records_path.write_bytes(
        records_jsonl(records, compact=selected.key in {"flagship", "portability"})
    )
    dataset_sha256 = hashlib.sha256(records_path.read_bytes()).hexdigest()
    if dataset_sha256 != selected.dataset_sha256:
        raise RuntimeError("prepared corpus does not match its pinned profile")
    policy = selected.acceptance_policy()
    (inputs / "acceptance.json").write_text(
        json.dumps(policy, sort_keys=True) + "\n", encoding="utf-8"
    )
    (inputs / "corpus-profile.json").write_text(
        json.dumps(corpus_provenance(selected), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    sides: dict[str, object] = {}
    for snapshot in selected_models.snapshots:
        checkpoint = models[snapshot.role]
        tokenizer = tokenizers[snapshot.role]
        _validate_record_tokenization(
            records,
            tokenizer,
            role=snapshot.role,
            context_length=selected.context_length,
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
                    "batch_size": selected_models.batch_size,
                    "checkpoint_tree_sha256": checkpoint_tree_sha256(checkpoint),
                    "context_length": selected.context_length,
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
            "dataset": selected.dataset_descriptor(),
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
    parser.add_argument("--corpus-profile", choices=PROFILE_KEYS, default="quick")
    arguments = parser.parse_args()
    selected = getattr(arguments, "corpus_profile", "quick")
    if selected != "quick":
        prepare(
            Path(os.path.abspath(arguments.workspace.expanduser())),
            arguments.runtime_image,
            corpus_profile_key=selected,
        )
    else:
        prepare(
            Path(os.path.abspath(arguments.workspace.expanduser())),
            arguments.runtime_image,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
