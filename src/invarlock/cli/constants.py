"""CLI constants shared across commands to keep outputs consistent."""

from __future__ import annotations

# Human-readable, versioned format identifiers for JSON outputs
# Keep in sync with tests under tests/cli/*_json_*.py
DOCTOR_FORMAT_VERSION = "doctor-v1"
PLUGINS_FORMAT_VERSION = "plugins-v1"
VERIFY_FORMAT_VERSION = "verify-v1"

PROVIDER_NOTES: dict[str, str] = {
    # WikiText-2 is loaded via datasets; works offline if cached.
    "wikitext2": "requires network or local cache",
    # Synthetic corpus used for quick smokes and CI; fully offline.
    "synthetic": "offline; deterministic",
    # Hugging Face text datasets (via datasets.load_dataset)
    "hf_text": "requires network",
    # Local providers (offline)
    "local_jsonl": "local files; offline",
    "local_jsonl_pairs": "paired prompts/responses (JSONL); offline",
    # Seq2Seq providers
    "seq2seq": "toy seq2seq dataset; offline",
    "hf_seq2seq": "requires network",
}

# Optional structured metadata for richer CLI tables
PROVIDER_PARAMS: dict[str, str] = {
    "wikitext2": "-",
    "synthetic": "-",
    "hf_text": "dataset_name[, split, text_field]",
    "hf_seq2seq": "dataset_name[, split, input_field, target_field]",
    "local_jsonl": "path[, text_field]",
    "local_jsonl_pairs": "path[, input_field, target_field]",
    "seq2seq": "-",
}

# Stable network classification to avoid tying UI to note strings
PROVIDER_NETWORK: dict[str, str] = {
    # 'no' | 'cache' | 'yes'
    "wikitext2": "cache",
    "synthetic": "no",
    "hf_text": "yes",
    "local_jsonl": "no",
    "local_jsonl_pairs": "no",
    "seq2seq": "no",
    "hf_seq2seq": "yes",
}

# Simple kind classification for presentation
PROVIDER_KIND: dict[str, str] = {
    "wikitext2": "text",
    "synthetic": "text",
    "hf_text": "text",
    "local_jsonl": "text",
    "local_jsonl_pairs": "pairs",
    "seq2seq": "seq2seq",
    "hf_seq2seq": "seq2seq",
}

__all__ = [
    "DOCTOR_FORMAT_VERSION",
    "PLUGINS_FORMAT_VERSION",
    "VERIFY_FORMAT_VERSION",
    "PROVIDER_NOTES",
    "PROVIDER_PARAMS",
    "PROVIDER_NETWORK",
    "PROVIDER_KIND",
]
