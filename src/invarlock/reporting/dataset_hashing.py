from __future__ import annotations

from typing import Any

from ..eval.data import EvaluationWindow, compute_window_hash


def compute_window_hashes(
    preview_window: EvaluationWindow, final_window: EvaluationWindow
) -> dict[str, str | int]:
    """Compute SHA256 hashes of evaluation windows for dataset integrity.

    Returns a mapping with preview/final sha256 and total token count.
    """
    preview_hash = compute_window_hash(preview_window, include_data=True)
    final_hash = compute_window_hash(final_window, include_data=True)
    return {
        "preview": f"sha256:{preview_hash}",
        "final": f"sha256:{final_hash}",
        "total_tokens": sum(
            len(ids) for ids in preview_window.input_ids + final_window.input_ids
        ),
    }


def _compute_actual_window_hashes(report: dict[str, Any]) -> dict[str, Any]:
    """Compute actual window hashes from explicit token IDs when available.

    Expects `report["evaluation_windows"]["preview"|"final"]["input_ids"]` to be present.
    Returns dict with per-split sha256 hashes and token counts when computable; otherwise
    returns an empty dict to signal fallback to config-based hashing.
    """
    try:
        # Prefer explicit preview/final hashes included in the report data
        data_config = report.get("data", {}) if isinstance(report, dict) else {}
        if (
            isinstance(data_config, dict)
            and data_config.get("preview_hash")
            and data_config.get("final_hash")
        ):
            preview_tokens_ct = data_config.get("preview_total_tokens")
            final_tokens_ct = data_config.get("final_total_tokens")
            total_tokens = 0
            if isinstance(preview_tokens_ct, int):
                total_tokens += preview_tokens_ct
            if isinstance(final_tokens_ct, int):
                total_tokens += final_tokens_ct
            return {
                "preview": f"blake2s:{data_config.get('preview_hash')}",
                "final": f"blake2s:{data_config.get('final_hash')}",
                "dataset": data_config.get("dataset_hash"),
                "total_tokens": total_tokens,
                "preview_tokens": preview_tokens_ct,
                "final_tokens": final_tokens_ct,
            }

        windows = report.get("evaluation_windows", {})
        if not isinstance(windows, dict):
            windows = {}
        preview_ids = (
            windows.get("preview", {}).get("input_ids")
            if windows.get("preview")
            else None
        )
        final_ids = (
            windows.get("final", {}).get("input_ids") if windows.get("final") else None
        )
        if not (isinstance(preview_ids, list) and isinstance(final_ids, list)):
            # Config-based fallback (stable sha256 of config tuple)
            import hashlib as _hashlib

            data_config = report.get("data", {}) if isinstance(report, dict) else {}
            dataset = str(data_config.get("dataset", "unknown"))
            split = str(
                data_config.get("split", data_config.get("dataset_split", "val"))
            )
            seed = (
                report.get("meta", {}).get("seed", 0)
                if isinstance(report.get("meta"), dict)
                else 0
            )
            preview_n = int(data_config.get("preview_n", 0) or 0)
            final_n = int(data_config.get("final_n", 0) or 0)
            seq_len = int(data_config.get("seq_len", 0) or 0)
            config_str = f"{dataset}{split}{seq_len}{preview_n}{final_n}{seed}"
            digest = _hashlib.sha256(config_str.encode()).hexdigest()
            preview_tokens = preview_n * seq_len if preview_n and seq_len else 0
            final_tokens = final_n * seq_len if final_n and seq_len else 0
            return {
                "preview": f"sha256:{digest[:32]}",
                "final": f"sha256:{digest[32:64] if len(digest) >= 64 else digest[:32]}",
                "dataset": data_config.get("provider_hash")
                or data_config.get("tokenizer_hash"),
                "preview_tokens": preview_tokens,
                "final_tokens": final_tokens,
                "total_tokens": preview_tokens + final_tokens,
            }
        # Compute hashes directly from token ID sequences for robustness
        import hashlib as _hashlib

        def _hash_sequences(seqs: list[list[int]]) -> str:
            h = _hashlib.sha256()
            for seq in seqs:
                try:
                    h.update(str(list(seq)).encode("utf-8"))
                except Exception:
                    continue
            return h.hexdigest()

        preview_hash = _hash_sequences(preview_ids)
        final_hash = _hash_sequences(final_ids)
        preview_tokens = sum(len(s) for s in preview_ids)
        final_tokens = sum(len(s) for s in final_ids)
        return {
            "preview": f"sha256:{preview_hash}",
            "final": f"sha256:{final_hash}",
            "preview_tokens": preview_tokens,
            "final_tokens": final_tokens,
            "dataset": None,
            "total_tokens": preview_tokens + final_tokens,
        }
    except Exception:
        # Signal caller to use config-based fallback
        return {}


def _extract_dataset_info(report: dict[str, Any]) -> dict[str, Any]:
    """Extract dataset configuration and compute window hashes.

    Tolerates missing `data` by falling back to evaluation_windows lengths and
    placeholder values for non-essential fields.
    """
    data_config = report.get("data", {}) if isinstance(report, dict) else {}
    eval_windows = (
        report.get("evaluation_windows", {}) if isinstance(report, dict) else {}
    )
    preview_section = (
        eval_windows.get("preview", {}) if isinstance(eval_windows, dict) else {}
    )
    final_section = (
        eval_windows.get("final", {}) if isinstance(eval_windows, dict) else {}
    )

    def _len_ids(sec: dict[str, Any]) -> int:
        ids = sec.get("window_ids")
        return int(len(ids)) if isinstance(ids, list) else 0

    preview_n = int(data_config.get("preview_n", 0) or 0) or _len_ids(preview_section)
    final_n = int(data_config.get("final_n", 0) or 0) or _len_ids(final_section)
    seq_len = int(data_config.get("seq_len", 0) or 0)
    stride = int(data_config.get("stride", 0) or 0)

    dataset = str(data_config.get("dataset", "unknown"))
    split = str(data_config.get("split", data_config.get("dataset_split", "val")))

    # Prefer actual window hashes when explicit token IDs are present
    actual_hashes = _compute_actual_window_hashes(report)
    if actual_hashes:
        window_hash = actual_hashes
    else:
        # Config-based fallback: produce a stable sha256 of config tuple
        import hashlib as _hashlib

        seed = (
            report.get("meta", {}).get("seed", 0)
            if isinstance(report.get("meta"), dict)
            else 0
        )
        preview_n = int(data_config.get("preview_n", 0) or 0)
        final_n = int(data_config.get("final_n", 0) or 0)
        seq_len = int(data_config.get("seq_len", 0) or 0)
        config_str = f"{dataset}{split}{seq_len}{preview_n}{final_n}{seed}"
        digest = _hashlib.sha256(config_str.encode()).hexdigest()
        preview_tokens = preview_n * seq_len if preview_n and seq_len else 0
        final_tokens = final_n * seq_len if final_n and seq_len else 0
        window_hash = {
            "preview": f"sha256:{digest[:32]}",
            "final": f"sha256:{digest[32:64] if len(digest) >= 64 else digest[:32]}",
            "dataset": data_config.get("provider_hash")
            or data_config.get("tokenizer_hash"),
            "preview_tokens": preview_tokens,
            "final_tokens": final_tokens,
            "total_tokens": preview_tokens + final_tokens,
        }

    tokenizer_info = {
        "name": data_config.get("tokenizer_name", "unknown"),
        "hash": data_config.get("tokenizer_hash"),
        "vocab_size": data_config.get("vocab_size"),
        "bos_token": data_config.get("bos_token"),
        "eos_token": data_config.get("eos_token"),
        "pad_token": data_config.get("pad_token"),
        "add_prefix_space": data_config.get("add_prefix_space"),
    }

    return {
        "provider": dataset,
        "split": split,
        "seq_len": seq_len,
        "stride": stride,
        "windows": {
            "preview": preview_n,
            "final": final_n,
            "seed": data_config.get("seed"),
        },
        "hash": window_hash,
        "tokenizer": tokenizer_info,
    }


__all__ = [
    "compute_window_hashes",
    "_compute_actual_window_hashes",
    "_extract_dataset_info",
]
