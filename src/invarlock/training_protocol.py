"""Package-owned identifiers and strict inputs for training assurance."""

from __future__ import annotations

from .evidence_pack_json import (
    read_jsonl_snapshot,
)

TRAINING_PROFILES_SCHEMA = "invarlock/evidence-pack-training-profiles-v1"
TRAINING_RECEIPT_SCHEMA = "invarlock/evidence-pack-training-receipt-v1"
TRAINING_EVIDENCE_PROOF_SCHEMA = "invarlock/training-evidence-proof-v1"
TRAINING_ARTIFACT_REPLAY_SCHEMA = "invarlock/training-artifact-replay-v1"
TRAINING_RUNTIME_RELOAD_PROOF_SCHEMA = "invarlock/training-runtime-reload-proof-v1"
LORA_MERGE_PROOF_SCHEMA = "invarlock/training-lora-merge-proof-v1"
PEFT_LORA_FIXTURE_FORMAT = "peft-lora-fixture-v1"
FULL_FINE_TUNE_FIXTURE_FORMAT = "tiny-fine-tune-fixture-v1"
__all__ = [
    "FULL_FINE_TUNE_FIXTURE_FORMAT",
    "LORA_MERGE_PROOF_SCHEMA",
    "PEFT_LORA_FIXTURE_FORMAT",
    "TRAINING_ARTIFACT_REPLAY_SCHEMA",
    "TRAINING_EVIDENCE_PROOF_SCHEMA",
    "TRAINING_PROFILES_SCHEMA",
    "TRAINING_RECEIPT_SCHEMA",
    "TRAINING_RUNTIME_RELOAD_PROOF_SCHEMA",
    "read_jsonl_snapshot",
]
