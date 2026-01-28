from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path
from typing import Any


def _truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def model_supports_flash_attention(model_id: str) -> bool:
    no_fa2_models = [
        "falcon",
        "mpt-",
        "gpt2",
        "bloom",
        "opt-",
        "gpt-j",
        "gpt-neo",
        "codegen",
        "santacoder",
        "stablelm",
    ]
    model_lower = model_id.lower()
    return not any(pattern in model_lower for pattern in no_fa2_models)


def sanitize_generation_config(model_dir: Path) -> None:
    gen_path = model_dir / "generation_config.json"
    if not gen_path.is_file():
        return
    try:
        gen = json.loads(gen_path.read_text())
    except Exception:
        return

    if gen.get("do_sample") is False:
        temp = gen.get("temperature")
        if temp not in (None, 1.0):
            print(
                f"Fixing generation_config.json: clearing temperature={temp} (do_sample=False)"
            )
            gen["temperature"] = None
        top_p = gen.get("top_p")
        if top_p not in (None, 1.0):
            print(
                f"Fixing generation_config.json: clearing top_p={top_p} (do_sample=False)"
            )
            gen["top_p"] = None
        try:
            gen_path.write_text(json.dumps(gen, indent=2) + "\n")
        except Exception:
            pass


def write_model_profile(model_dir: Path, model_id: str, revision: str | None) -> None:
    weights_bytes = 0
    for pat in ("*.safetensors", "*.bin"):
        for fp in model_dir.glob(pat):
            try:
                weights_bytes += fp.stat().st_size
            except OSError:
                pass

    cfg_path = model_dir / "config.json"
    config: dict[str, Any] = {}
    if cfg_path.is_file():
        try:
            config = json.loads(cfg_path.read_text())
        except Exception:
            config = {}

    profile = {
        "model_id": model_id,
        "revision": revision,
        "weights_bytes": weights_bytes,
        "weights_gb": round(weights_bytes / (1024**3), 3),
        "hidden_size": config.get("hidden_size"),
        "num_layers": config.get("num_hidden_layers"),
        "num_heads": config.get("num_attention_heads"),
        "num_kv_heads": config.get("num_key_value_heads")
        or config.get("num_attention_heads"),
        "max_position_embeddings": config.get("max_position_embeddings"),
        "dtype_bytes": 2,
    }
    (model_dir / "model_profile.json").write_text(json.dumps(profile, indent=2) + "\n")


def download_snapshot(
    repo_id: str, model_dir: Path, mode: str, revision: str | None
) -> None:
    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:
        raise RuntimeError(f"huggingface_hub not available: {exc}") from exc

    local_dir_use_symlinks = mode == "snapshot_symlink"
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(model_dir),
        local_dir_use_symlinks=local_dir_use_symlinks,
        cache_dir=os.environ.get("HF_HUB_CACHE"),
        revision=revision,
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Proof-pack baseline downloader.")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--success-marker", default="")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    model_id = str(args.model_id)
    output_dir = Path(args.output_dir)
    success_marker = Path(args.success_marker) if args.success_marker else None

    flash_available = _truthy(os.environ.get("FLASH_ATTENTION_AVAILABLE"))
    revision = os.environ.get("PACK_MODEL_REVISION") or None
    baseline_mode = (
        os.environ.get("PACK_BASELINE_STORAGE_MODE", "snapshot_symlink").strip().lower()
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    rev_label = f"@{revision}" if revision else ""
    print(f"Downloading {model_id}{rev_label} (proof pack optimized)...")
    print(f"Baseline storage mode: {baseline_mode}")
    if os.environ.get("HF_HUB_CACHE"):
        print(f"HF_HUB_CACHE: {os.environ.get('HF_HUB_CACHE')}")
    print(f"Flash Attention 2: {'enabled' if flash_available else 'disabled'}")

    try:
        if baseline_mode in ("snapshot_symlink", "snapshot_copy"):
            try:
                download_snapshot(model_id, output_dir, baseline_mode, revision)
                sanitize_generation_config(output_dir)
                write_model_profile(output_dir, model_id, revision)
                if success_marker is not None:
                    success_marker.parent.mkdir(parents=True, exist_ok=True)
                    success_marker.touch()
                print(f"Saved to {output_dir} (snapshot)")
                return 0
            except Exception as snap_err:
                print(
                    f"WARNING: snapshot_download failed, falling back to save_pretrained: {snap_err}",
                    file=sys.stderr,
                )
                baseline_mode = "save_pretrained"

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        mode = os.environ.get("PACK_DETERMINISM", "throughput").strip().lower()
        if mode == "strict":
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        else:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        cache_dir = os.environ.get("HF_HUB_CACHE")
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=True, cache_dir=cache_dir, revision=revision
        )
        tokenizer.save_pretrained(output_dir)

        use_fa2 = flash_available and model_supports_flash_attention(model_id)
        model_kwargs: dict[str, Any] = {
            "dtype": torch.bfloat16,
            "trust_remote_code": True,
            "device_map": "auto",
            "low_cpu_mem_usage": True,
            "cache_dir": cache_dir,
            "revision": revision,
        }
        if use_fa2:
            model_kwargs["attn_implementation"] = "flash_attention_2"
            print(f"Using Flash Attention 2 for {model_id}")
        else:
            print(
                f"Using eager attention for {model_id} (FA2 not supported or unavailable)"
            )

        try:
            model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
        except Exception as fa2_err:
            if use_fa2 and "flash" in str(fa2_err).lower():
                print(
                    f"Flash Attention 2 failed, falling back to eager attention: {fa2_err}"
                )
                del model_kwargs["attn_implementation"]
                model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
            else:
                raise

        if hasattr(model, "generation_config"):
            gen_config = model.generation_config
            if getattr(gen_config, "do_sample", True) is False:
                if getattr(gen_config, "temperature", 1.0) not in (None, 1.0):
                    print(
                        "Fixing generation_config: clearing temperature="
                        f"{gen_config.temperature} (do_sample=False)"
                    )
                    gen_config.temperature = None
                if getattr(gen_config, "top_p", 1.0) not in (None, 1.0):
                    print(
                        "Fixing generation_config: clearing top_p="
                        f"{gen_config.top_p} (do_sample=False)"
                    )
                    gen_config.top_p = None

        model.save_pretrained(output_dir, safe_serialization=True)

        del model
        gc.collect()
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.memory.empty_cache()

        sanitize_generation_config(output_dir)
        write_model_profile(output_dir, model_id, revision)
        if success_marker is not None:
            success_marker.parent.mkdir(parents=True, exist_ok=True)
            success_marker.touch()
        print(f"Saved to {output_dir} (save_pretrained)")
        return 0

    except Exception as exc:
        print(f"ERROR: Model download failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
