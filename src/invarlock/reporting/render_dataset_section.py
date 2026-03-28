"""Dataset and provenance Markdown section rendering."""

from __future__ import annotations

from typing import Any


def _short_digest(value: str) -> str:
    normalized = str(value)
    return (
        normalized
        if len(normalized) <= 16
        else (normalized[:8] + "…" + normalized[-8:])
    )


def _dataset_hash_source_label(source: Any) -> str | None:
    source_map = {
        "explicit_preview_final_hashes": "provider-derived explicit preview/final hashes",
        "explicit_token_ids": "content-derived token IDs",
        "config_fallback": "config-derived fallback",
    }
    try:
        key = str(source or "").strip()
    except Exception:
        return None
    return source_map.get(key)


def append_dataset_and_provenance_section(
    lines: list[str], evaluation_report: dict[str, Any]
) -> None:
    """Append the dataset/provenance Markdown block."""
    dataset = evaluation_report.get("dataset", {}) or {}
    provenance_info = evaluation_report.get("provenance", {}) or {}

    has_dataset = isinstance(dataset, dict) and bool(dataset)
    has_provenance = isinstance(provenance_info, dict) and bool(provenance_info)
    if not (has_dataset or has_provenance):
        return

    lines.append("## Dataset and Provenance")
    lines.append("")

    if has_dataset:
        provider = dataset.get("provider") or "unknown"
        lines.append(f"- **Provider:** {provider}")
        try:
            seq_len_val = (
                int(dataset.get("seq_len"))
                if isinstance(dataset.get("seq_len"), int | float)
                else dataset.get("seq_len")
            )
        except Exception:  # pragma: no cover
            seq_len_val = dataset.get("seq_len")
        if seq_len_val is not None:
            lines.append(f"- **Sequence Length:** {seq_len_val}")
        windows_blk = (
            dataset.get("windows", {})
            if isinstance(dataset.get("windows"), dict)
            else {}
        )
        win_prev = windows_blk.get("preview")
        win_final = windows_blk.get("final")
        if win_prev is not None and win_final is not None:
            lines.append(f"- **Windows:** {win_prev} preview + {win_final} final")
        if windows_blk.get("seed") is not None:
            lines.append(f"- **Seed:** {windows_blk.get('seed')}")
        hash_blk = (
            dataset.get("hash", {}) if isinstance(dataset.get("hash"), dict) else {}
        )
        if hash_blk.get("preview_tokens") is not None:
            lines.append(f"- **Preview Tokens:** {hash_blk.get('preview_tokens'):,}")
        if hash_blk.get("final_tokens") is not None:
            lines.append(f"- **Final Tokens:** {hash_blk.get('final_tokens'):,}")
        if hash_blk.get("total_tokens") is not None:
            lines.append(f"- **Total Tokens:** {hash_blk.get('total_tokens'):,}")
        if hash_blk.get("dataset"):
            lines.append(f"- **Dataset Hash:** {hash_blk.get('dataset')}")
        hash_source = _dataset_hash_source_label(hash_blk.get("source"))
        if hash_source:
            lines.append(f"- **Hash Source:** {hash_source}")
        tokenizer = dataset.get("tokenizer", {})
        if isinstance(tokenizer, dict) and (
            tokenizer.get("name") or tokenizer.get("hash")
        ):
            vocab_size = tokenizer.get("vocab_size")
            vocab_suffix = (
                f" (vocab {vocab_size})" if isinstance(vocab_size, int) else ""
            )
            lines.append(
                f"- **Tokenizer:** {tokenizer.get('name', 'unknown')}{vocab_suffix}"
            )
            if tokenizer.get("hash"):
                lines.append(f"  - Hash: {tokenizer['hash']}")
            lines.append(
                f"  - BOS/EOS: {tokenizer.get('bos_token')} / {tokenizer.get('eos_token')}"
            )
            if tokenizer.get("pad_token") is not None:
                lines.append(f"  - PAD: {tokenizer.get('pad_token')}")
            if tokenizer.get("add_prefix_space") is not None:
                lines.append(
                    f"  - add_prefix_space: {tokenizer.get('add_prefix_space')}"
                )

    if has_provenance:
        baseline_info = provenance_info.get("baseline", {}) or {}
        edited_info = provenance_info.get("edited", {}) or {}

        if baseline_info or edited_info:
            lines.append("")
        if baseline_info:
            lines.append(f"- **Baseline Run ID:** {baseline_info.get('run_id')}")
            if baseline_info.get("report_hash"):
                lines.append(f"  - Report Hash: `{baseline_info.get('report_hash')}`")
            if baseline_info.get("report_path"):
                lines.append(f"  - Report Path: {baseline_info.get('report_path')}")
        if edited_info:
            lines.append(f"- **Edited Run ID:** {edited_info.get('run_id')}")
            if edited_info.get("report_hash"):
                lines.append(f"  - Report Hash: `{edited_info.get('report_hash')}`")
            if edited_info.get("report_path"):
                lines.append(f"  - Report Path: {edited_info.get('report_path')}")

        provider_digest = provenance_info.get("provider_digest")
        if isinstance(provider_digest, dict) and provider_digest:
            ids_d = provider_digest.get("ids_sha256")
            tok_d = provider_digest.get("tokenizer_sha256")
            mask_d = provider_digest.get("masking_sha256")

            lines.append("- **Provider Digest:**")
            if tok_d:
                lines.append(
                    f"  - tokenizer_sha256: `{_short_digest(tok_d)}` (full in JSON)"
                )
            if ids_d:
                lines.append(f"  - ids_sha256: `{_short_digest(ids_d)}` (full in JSON)")
            if mask_d:
                lines.append(
                    f"  - masking_sha256: `{_short_digest(mask_d)}` (full in JSON)"
                )

        try:
            confidence = evaluation_report.get("confidence", {}) or {}
            if isinstance(confidence, dict) and confidence.get("label"):
                lines.append(f"- **Confidence:** {confidence.get('label')}")
        except Exception:
            pass

        try:
            policy_digest = evaluation_report.get("policy_digest", {}) or {}
            if isinstance(policy_digest, dict) and policy_digest:
                policy_version = policy_digest.get("policy_version")
                thresholds_hash = policy_digest.get("thresholds_hash")
                if policy_version:
                    lines.append(f"- **Policy Version:** {policy_version}")
                if isinstance(thresholds_hash, str) and thresholds_hash:
                    lines.append(
                        f"- **Thresholds Digest:** `{_short_digest(thresholds_hash)}` (full in JSON)"
                    )
                if policy_digest.get("changed"):
                    lines.append("- Note: policy changed")
        except Exception:
            pass

    lines.append("")


__all__ = ["append_dataset_and_provenance_section"]
