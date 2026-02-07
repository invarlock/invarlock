from __future__ import annotations

import os


def _env(name: str) -> str:
    return str(os.environ.get(name, "")).strip()


def main() -> int:
    provider = _env("INVARLOCK_DATASET").lower() or "wikitext2"
    # Proof packs default to WikiText-2. Other providers may be purely local or synthetic.
    if provider != "wikitext2":
        print(f"[DATASET_PREFLIGHT] provider={provider}: skipped")
        return 0

    try:
        from datasets import load_dataset
    except Exception as exc:
        print("[DATASET_PREFLIGHT] ERROR: datasets library is required for provider=wikitext2.")
        print(f"[DATASET_PREFLIGHT] import_error={type(exc).__name__}: {exc}")
        return 1

    offline = _env("HF_DATASETS_OFFLINE")
    hf_home = _env("HF_HOME")
    datasets_cache = _env("HF_DATASETS_CACHE")

    try:
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    except Exception as exc:
        print("[DATASET_PREFLIGHT] ERROR: failed to load wikitext2 validation split.")
        if offline:
            print(f"[DATASET_PREFLIGHT] HF_DATASETS_OFFLINE={offline}")
        if hf_home:
            print(f"[DATASET_PREFLIGHT] HF_HOME={hf_home}")
        if datasets_cache:
            print(f"[DATASET_PREFLIGHT] HF_DATASETS_CACHE={datasets_cache}")
        print(f"[DATASET_PREFLIGHT] exception={type(exc).__name__}: {exc}")
        return 1

    try:
        size = len(ds)
    except Exception:
        size = -1

    print(f"[DATASET_PREFLIGHT] OK: provider=wikitext2 split=validation size={size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

