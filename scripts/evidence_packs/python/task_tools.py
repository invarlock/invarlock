from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import os
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    from .editing.implementations import (
        DEPLOYABLE_OPTIMIZED_SUBJECT,
        EDIT_METADATA_SCHEMA,
        FAULT_INJECTION_FIXTURE,
        VALIDATION_SUBJECT_CHECKPOINT,
        read_edit_metadata,
        resolve_edit_spec,
    )
    from .error_model.common import fix_layer_drop_config_json
    from .runtime_tools import require_remote_code_opt_in
except ImportError:  # pragma: no cover - direct script execution
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from editing.implementations import (
        DEPLOYABLE_OPTIMIZED_SUBJECT,
        EDIT_METADATA_SCHEMA,
        FAULT_INJECTION_FIXTURE,
        VALIDATION_SUBJECT_CHECKPOINT,
        read_edit_metadata,
        resolve_edit_spec,
    )
    from error_model.common import fix_layer_drop_config_json
    from runtime_tools import require_remote_code_opt_in

EDIT_ARTIFACT_SUMMARY_SCHEMA = "invarlock/evidence-pack-edit-artifact-summary-v1"
_BIN_IGNORE_PATTERNS = ["*.bin", "*.bin.index.json"]
_SAFETENSORS_IGNORE_PATTERNS = ["*.safetensors", "*.safetensors.index.json"]


def _get_config_value(cfg: dict[str, Any], key: str, *fallbacks: str) -> Any:
    value = cfg.get(key)
    if value is not None:
        return value
    for fallback in fallbacks:
        value = cfg.get(fallback)
        if value is not None:
            return value
    return None


def _layer_count(cfg: dict[str, Any]) -> int | None:
    for key in ("num_hidden_layers", "n_layer", "num_layers"):
        value = cfg.get(key)
        if isinstance(value, int) and value > 0:
            return value
    return None


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
    except (OSError, json.JSONDecodeError):
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
            if gen_path.is_symlink():
                original = gen_path.read_text(encoding="utf-8")
                gen_path.unlink()
                gen_path.write_text(original, encoding="utf-8")
            gen_path.write_text(json.dumps(gen, indent=2) + "\n")
        except OSError:
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
        except (OSError, json.JSONDecodeError):
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


def _select_weight_download_policy(
    repo_id: str, revision: str | None
) -> tuple[str, list[str]]:
    try:
        from huggingface_hub import list_repo_files
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(f"huggingface_hub not available: {exc}") from exc

    repo_files = list_repo_files(repo_id, repo_type="model", revision=revision)
    has_safetensors = any(
        path.endswith(".safetensors") or path.endswith(".safetensors.index.json")
        for path in repo_files
    )
    has_bin = any(
        path.endswith(".bin") or path.endswith(".bin.index.json") for path in repo_files
    )

    if has_safetensors:
        return "safetensors", _BIN_IGNORE_PATTERNS
    if has_bin:
        return "bin", _SAFETENSORS_IGNORE_PATTERNS
    return "unknown", []


def _symlink_snapshot_tree(snapshot_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for source in snapshot_dir.rglob("*"):
        relative = source.relative_to(snapshot_dir)
        destination = output_dir / relative
        if source.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() or destination.is_symlink():
            destination.unlink()
        try:
            target = os.path.relpath(str(source), str(destination.parent))
            destination.symlink_to(target)
        except OSError as exc:
            raise RuntimeError(
                "PACK_BASELINE_STORAGE_MODE=snapshot_symlink requires symlink support. "
                "Use snapshot_copy or save_pretrained if symlink creation is unavailable."
            ) from exc


def download_snapshot(
    repo_id: str, model_dir: Path, mode: str, revision: str | None
) -> str:
    try:
        from huggingface_hub import snapshot_download
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(f"huggingface_hub not available: {exc}") from exc

    weight_format, ignore_patterns = _select_weight_download_policy(repo_id, revision)
    download_kwargs: dict[str, Any] = {
        "repo_id": repo_id,
        "cache_dir": os.environ.get("HF_HUB_CACHE"),
        "revision": revision,
    }
    if ignore_patterns:
        download_kwargs["ignore_patterns"] = ignore_patterns

    if mode == "snapshot_copy":
        snapshot_download(local_dir=str(model_dir), **download_kwargs)
        return weight_format

    if mode == "snapshot_symlink":
        snapshot_path = Path(snapshot_download(**download_kwargs))
        _symlink_snapshot_tree(snapshot_path, model_dir)
        return weight_format

    raise RuntimeError(f"Unsupported snapshot mode: {mode}")


def _download_baseline(args: argparse.Namespace) -> int:
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
    print(f"Downloading {model_id}{rev_label} (evidence pack optimized)...")
    print(f"Baseline storage mode: {baseline_mode}")
    if os.environ.get("HF_HUB_CACHE"):
        print(f"HF_HUB_CACHE: {os.environ.get('HF_HUB_CACHE')}")
    print(f"Flash Attention 2: {'enabled' if flash_available else 'disabled'}")

    try:
        if baseline_mode in ("snapshot_symlink", "snapshot_copy"):
            try:
                weight_format = download_snapshot(
                    model_id, output_dir, baseline_mode, revision
                )
                sanitize_generation_config(output_dir)
                write_model_profile(output_dir, model_id, revision)
                if success_marker is not None:
                    success_marker.parent.mkdir(parents=True, exist_ok=True)
                    success_marker.touch()
                mode_label = (
                    "snapshot cache symlink"
                    if baseline_mode == "snapshot_symlink"
                    else "snapshot copy"
                )
                if weight_format != "unknown":
                    print(f"Weight format: {weight_format} only")
                print(f"Saved to {output_dir} ({mode_label})")
                return 0
            except (
                ImportError,
                ModuleNotFoundError,
                OSError,
                RuntimeError,
                ValueError,
            ) as snap_err:
                message = str(snap_err)
                if baseline_mode == "snapshot_symlink":
                    print(
                        "ERROR: snapshot_symlink requires a cache-backed symlink tree "
                        f"and could not be prepared: {message}",
                        file=sys.stderr,
                    )
                    return 1
                print(
                    "WARNING: snapshot_copy failed, falling back to save_pretrained: "
                    f"{message}",
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
        trust_remote_code = require_remote_code_opt_in(
            "task_tools.py download-baseline"
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
            cache_dir=cache_dir,
            revision=revision,
        )
        tokenizer.save_pretrained(output_dir)

        use_fa2 = flash_available and model_supports_flash_attention(model_id)
        model_kwargs: dict[str, Any] = {
            "dtype": torch.bfloat16,
            "trust_remote_code": trust_remote_code,
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
                f"Using eager attention for {model_id} "
                "(FA2 not supported or unavailable)"
            )

        try:
            model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
        except (OSError, RuntimeError, ValueError) as fa2_err:
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

    except (
        ImportError,
        ModuleNotFoundError,
        OSError,
        RuntimeError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        print(f"ERROR: Model download failed: {exc}", file=sys.stderr)
        return 1


def _load_yaml_module() -> Any:
    try:
        import yaml
    except (
        ModuleNotFoundError
    ) as exc:  # pragma: no cover - evidence-pack preflight enforces PyYAML
        raise SystemExit(
            "PyYAML is required to normalize staged evidence-pack presets"
        ) from exc
    return yaml


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    return {}


def _normalized_preset_payload(
    payload: Any,
    *,
    seq_len: int,
    stride: int,
    preview_n: int,
    final_n: int,
    skip_overhead_check: bool,
) -> dict[str, Any]:
    normalized = _mapping(payload)

    dataset = _mapping(normalized.get("dataset"))
    dataset["seq_len"] = seq_len
    dataset["stride"] = stride
    dataset["preview_n"] = preview_n
    dataset["final_n"] = final_n
    normalized["dataset"] = dataset

    if skip_overhead_check:
        context = _mapping(normalized.get("context"))
        run = _mapping(context.get("run"))
        run["skip_overhead_check"] = True
        context["run"] = run
        normalized["context"] = context

    return normalized


def normalize_staged_preset(
    preset_path: Path,
    *,
    seq_len: int,
    stride: int,
    preview_n: int,
    final_n: int,
    skip_overhead_check: bool,
) -> None:
    yaml = _load_yaml_module()
    raw = preset_path.read_text()
    loaded = yaml.safe_load(raw) if raw.strip() else {}
    normalized = _normalized_preset_payload(
        loaded,
        seq_len=seq_len,
        stride=stride,
        preview_n=preview_n,
        final_n=final_n,
        skip_overhead_check=skip_overhead_check,
    )
    preset_path.write_text(yaml.safe_dump(normalized, sort_keys=False))


def _coerce_required_int(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise SystemExit(
            f"baseline report is missing numeric data.{key} needed to normalize preset"
        )
    return int(value)


def schedule_from_baseline_report(report_path: Path) -> tuple[int, int, int, int]:
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"failed to load baseline report JSON: {report_path}") from exc

    source: dict[str, Any] = {}
    if isinstance(payload, dict):
        data = payload.get("data")
        if isinstance(data, dict):
            source = data
        else:
            dataset = payload.get("dataset")
            if isinstance(dataset, dict):
                source = dataset
    if not source:
        raise SystemExit(
            f"baseline report does not contain data/dataset schedule fields: {report_path}"
        )

    return (
        _coerce_required_int(source, "seq_len"),
        _coerce_required_int(source, "stride"),
        _coerce_required_int(source, "preview_n"),
        _coerce_required_int(source, "final_n"),
    )


def _normalize_staged_preset(args: argparse.Namespace) -> int:
    if args.baseline_report:
        seq_len, stride, preview_n, final_n = schedule_from_baseline_report(
            Path(args.baseline_report)
        )
    else:
        values = (args.seq_len, args.stride, args.preview_n, args.final_n)
        if any(value is None for value in values):
            raise SystemExit(
                "either --baseline-report or all of "
                "--seq-len/--stride/--preview-n/--final-n is required"
            )
        seq_len, stride, preview_n, final_n = (
            int(args.seq_len),
            int(args.stride),
            int(args.preview_n),
            int(args.final_n),
        )
    normalize_staged_preset(
        Path(args.preset),
        seq_len=seq_len,
        stride=stride,
        preview_n=preview_n,
        final_n=final_n,
        skip_overhead_check=bool(args.skip_overhead_check),
    )
    return 0


def _create_error_model(args: argparse.Namespace) -> int:
    try:
        from .error_model.common import (
            _OVERLAY_FALLBACK_ERRORS,
            _collect_block_params,
            _load_error_model,
            _save_error_model,
            _shape_mismatch_overlay_safetensors,
        )
        from .error_model.probe_injections import _apply_error_injection
    except ImportError:  # pragma: no cover - direct script execution
        from error_model.common import (
            _OVERLAY_FALLBACK_ERRORS,
            _collect_block_params,
            _load_error_model,
            _save_error_model,
            _shape_mismatch_overlay_safetensors,
        )
        from error_model.probe_injections import _apply_error_injection

    from transformers import AutoTokenizer

    baseline_path = Path(args.baseline_path)
    output_path = Path(args.output_path)
    error_type = str(args.error_type)

    print(f"Loading baseline from {baseline_path}...")
    trust_remote_code = require_remote_code_opt_in("task_tools.py create-error-model")
    tokenizer = AutoTokenizer.from_pretrained(
        baseline_path, trust_remote_code=trust_remote_code
    )

    if error_type == "shape_mismatch":
        # Large sharded models can be OOM-killed during save_pretrained() shard writes.
        # Prefer an index-based overlay that only rewrites the embedding + lm_head tensors.
        delta = 8
        try:
            error_info = _shape_mismatch_overlay_safetensors(
                baseline_path=baseline_path,
                output_path=output_path,
                tokenizer=tokenizer,
                delta=delta,
            )
        except _OVERLAY_FALLBACK_ERRORS as exc:
            error_info = None
            print(f"WARNING: shape_mismatch overlay failed ({exc}); falling back")

        if error_info is not None:
            output_path.mkdir(parents=True, exist_ok=True)
            (output_path / "error_metadata.json").write_text(
                json.dumps(error_info, indent=2, sort_keys=True) + "\n"
            )
            print(f"Saved error model to {output_path}")
            return 0

    model, use_gpu = _load_error_model(
        baseline_path=baseline_path, trust_remote_code=trust_remote_code
    )
    error_info: dict[str, object] = {"error_type": error_type, "injected": False}
    block_params, num_blocks = _collect_block_params(model)
    print(f"Detected {num_blocks} transformer blocks")

    if error_type == "shape_mismatch":
        try:
            emb = model.get_input_embeddings()
            old_vocab = int(getattr(emb, "num_embeddings", emb.weight.shape[0]))
            delta = 8
            new_vocab = old_vocab + delta
            model.resize_token_embeddings(new_vocab)
            error_info.update(
                {
                    "injected": True,
                    "old_vocab_size": old_vocab,
                    "new_vocab_size": int(new_vocab),
                    "delta": int(delta),
                }
            )
            print(f"Resized token embeddings: {old_vocab} -> {new_vocab}")
        except (RuntimeError, TypeError, ValueError) as exc:
            print(f"WARNING: shape_mismatch not injected ({exc})")
    else:
        _apply_error_injection(
            error_type=error_type,
            model=model,
            baseline_path=baseline_path,
            block_params=block_params,
            error_info=error_info,
        )

    _save_error_model(
        model=model,
        tokenizer=tokenizer,
        output_path=output_path,
        error_info=error_info,
        use_gpu=use_gpu,
    )

    del model
    gc.collect()
    print(f"Saved error model to {output_path}")
    return 0


def _resolve_adapter(args: argparse.Namespace) -> int:
    model_id = str(args.model_id_or_path).strip()
    if not model_id:
        return 0

    from invarlock.adapters.auto import resolve_auto_adapter

    print(resolve_auto_adapter(model_id))
    return 0


def _resolve_edit_params(args: argparse.Namespace) -> int:
    resolved = resolve_edit_spec(
        model_output_dir=Path(args.model_output_dir),
        edit_spec=str(args.edit_spec),
        version_hint=str(args.version_hint or ""),
    )
    print(json.dumps(resolved.to_shell_payload()))
    return 0


def _model_revision(args: argparse.Namespace) -> int:
    path = Path(args.revisions_json)
    model_id = str(args.model_id)

    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return 0

    if not isinstance(data, dict):
        return 0

    revision = ""
    models = data.get("models")
    if isinstance(models, dict):
        entry = models.get(model_id)
        if isinstance(entry, dict):
            revision = str(entry.get("revision") or "")

    if revision:
        print(revision)
    return 0


def _evaluation_report(args: argparse.Namespace) -> int:
    report_path = Path(args.report)
    out_path = Path(args.out)

    try:
        from invarlock.reporting.report_make import make_report
    except (ImportError, ModuleNotFoundError) as exc:
        print(f"Evaluation report generation warning: {exc}", file=sys.stderr)
        return 1

    try:
        report = json.loads(report_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Evaluation report generation warning: {exc}", file=sys.stderr)
        return 1

    try:
        evaluation_report = make_report(report, report)
    except (RuntimeError, TypeError, ValueError, KeyError) as exc:
        print(f"Evaluation report generation warning: {exc}", file=sys.stderr)
        return 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(evaluation_report, indent=2) + "\n")
    return 0


def _validate_baseline_report(args: argparse.Namespace) -> int:
    report_path = Path(args.baseline_report)
    expected_adapter = str(args.expected_adapter)
    expected_profile = str(args.expected_profile)
    expected_tier = str(args.expected_tier)

    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"baseline_report_invalid_json:{exc}", file=sys.stderr)
        return 1

    if not isinstance(payload, dict):
        print("baseline_report_not_object", file=sys.stderr)
        return 1

    edit = payload.get("edit")
    edit_name = edit.get("name") if isinstance(edit, dict) else None
    if edit_name != "noop":
        print(f"baseline_report_edit_not_noop:{edit_name!r}", file=sys.stderr)
        return 1

    meta = payload.get("meta")
    adapter = meta.get("adapter") if isinstance(meta, dict) else None
    if not isinstance(adapter, str) or not adapter:
        print("baseline_report_missing_adapter", file=sys.stderr)
        return 1
    if adapter != expected_adapter:
        print(
            f"baseline_report_adapter_mismatch:{adapter!r}!={expected_adapter!r}",
            file=sys.stderr,
        )
        return 1

    context = payload.get("context")
    if not isinstance(context, dict):
        print("baseline_report_missing_context", file=sys.stderr)
        return 1
    profile = context.get("profile")
    if not isinstance(profile, str) or not profile:
        print("baseline_report_missing_profile", file=sys.stderr)
        return 1
    if profile.strip().lower() != expected_profile.strip().lower():
        print(
            f"baseline_report_profile_mismatch:{profile!r}!={expected_profile!r}",
            file=sys.stderr,
        )
        return 1
    auto = context.get("auto")
    if not isinstance(auto, dict):
        print("baseline_report_missing_auto", file=sys.stderr)
        return 1
    tier = auto.get("tier")
    if not isinstance(tier, str) or not tier:
        print("baseline_report_missing_tier", file=sys.stderr)
        return 1
    if tier != expected_tier:
        print(
            f"baseline_report_tier_mismatch:{tier!r}!={expected_tier!r}",
            file=sys.stderr,
        )
        return 1

    windows = payload.get("evaluation_windows")
    if not isinstance(windows, dict):
        print("baseline_report_missing_evaluation_windows", file=sys.stderr)
        return 1

    for phase_name in ("preview", "final"):
        phase = windows.get(phase_name)
        if not isinstance(phase, dict):
            print(f"baseline_report_missing_phase:{phase_name}", file=sys.stderr)
            return 1
        window_ids = phase.get("window_ids")
        input_ids = phase.get("input_ids")
        if not isinstance(window_ids, list) or not window_ids:
            print(f"baseline_report_missing_window_ids:{phase_name}", file=sys.stderr)
            return 1
        if not isinstance(input_ids, list) or not input_ids:
            print(f"baseline_report_missing_input_ids:{phase_name}", file=sys.stderr)
            return 1
        if len(window_ids) != len(input_ids):
            print(f"baseline_report_mismatched_windows:{phase_name}", file=sys.stderr)
            return 1

    return 0


def _write_model_profile(args: argparse.Namespace) -> int:
    baseline_dir = Path(args.baseline_dir)
    model_id = str(args.model_id)
    profile_path = baseline_dir / "model_profile.json"
    if profile_path.exists():
        return 0

    config_path = baseline_dir / "config.json"
    if not config_path.exists():
        return 0

    try:
        cfg = json.loads(config_path.read_text())
    except (OSError, json.JSONDecodeError):
        return 0

    if not isinstance(cfg, dict):
        return 0

    weights_bytes = 0
    for pattern in ("*.safetensors", "*.bin"):
        for file_path in baseline_dir.glob(pattern):
            try:
                weights_bytes += file_path.stat().st_size
            except OSError:
                pass

    profile = {
        "model_id": model_id,
        "weights_bytes": weights_bytes,
        "weights_gb": round(weights_bytes / (1024**3), 3),
        "hidden_size": _get_config_value(cfg, "hidden_size", "n_embd", "d_model"),
        "num_layers": _get_config_value(cfg, "num_hidden_layers", "n_layer"),
        "num_heads": _get_config_value(cfg, "num_attention_heads", "n_head"),
        "num_kv_heads": _get_config_value(
            cfg, "num_key_value_heads", "num_key_value_groups"
        ),
        "max_position_embeddings": _get_config_value(
            cfg, "max_position_embeddings", "max_seq_len", "seq_length"
        ),
        "dtype_bytes": 2,
    }

    profile_path.write_text(json.dumps(profile, indent=2) + "\n")
    return 0


def _repair_missing_tensors_config(args: argparse.Namespace) -> int:
    baseline_path = Path(args.baseline_config)
    error_path = Path(args.error_config)

    baseline_cfg = json.loads(baseline_path.read_text(encoding="utf-8"))
    error_cfg = json.loads(error_path.read_text(encoding="utf-8"))
    if not isinstance(baseline_cfg, dict) or not isinstance(error_cfg, dict):
        return 2

    total_layers = _layer_count(baseline_cfg)
    kept_layers = _layer_count(error_cfg)
    if total_layers is None or kept_layers is None:
        return 0

    before = json.dumps(error_cfg, sort_keys=True)
    fix_layer_drop_config_json(
        error_cfg,
        total_layers=total_layers,
        kept_layers=kept_layers,
        baseline_config=baseline_cfg,
    )
    after = json.dumps(error_cfg, sort_keys=True)
    if before != after:
        error_path.write_text(json.dumps(error_cfg, indent=2) + "\n", encoding="utf-8")
        print(
            f"Repaired missing_tensors config: total_layers={total_layers} kept_layers={kept_layers}",
            file=sys.stderr,
        )

    return 0


def _load_json_optional(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _scenario_index(scenarios_path: Path) -> dict[str, dict[str, Any]]:
    payload = _load_json_object(scenarios_path)
    scenarios = payload.get("scenarios")
    if not isinstance(scenarios, list):
        return {}
    result: dict[str, dict[str, Any]] = {}
    for item in scenarios:
        if not isinstance(item, dict):
            continue
        scenario_id = item.get("id")
        if isinstance(scenario_id, str) and scenario_id:
            result[scenario_id] = item
    return result


def _scenario_from_report_metadata(pack_dir: Path, metadata_path: Path) -> str | None:
    try:
        rel = metadata_path.relative_to(pack_dir)
    except ValueError:
        return None
    parts = rel.parts
    if len(parts) < 5 or parts[0] != "reports":
        return None
    if parts[2] == "errors":
        return parts[3] if len(parts) > 3 else None
    return parts[2]


def _first_metadata_by_scenario(pack_dir: Path) -> dict[str, dict[str, Any]]:
    observed: dict[str, dict[str, Any]] = {}
    for metadata_path in sorted(pack_dir.glob("reports/**/edit_metadata.json")):
        scenario_id = _scenario_from_report_metadata(pack_dir, metadata_path)
        if not scenario_id or scenario_id in observed:
            continue
        try:
            observed[scenario_id] = read_edit_metadata(metadata_path)
        except (OSError, json.JSONDecodeError, ValueError):
            continue
    return observed


def _first_deployable_validation_by_scenario(
    pack_dir: Path,
) -> dict[str, dict[str, Any]]:
    observed: dict[str, dict[str, Any]] = {}
    for validation_path in sorted(
        pack_dir.glob("reports/**/deployable_artifact_validation.json")
    ):
        scenario_id = _scenario_from_report_metadata(pack_dir, validation_path)
        if not scenario_id or scenario_id in observed:
            continue
        payload = _load_json_object(validation_path)
        if payload:
            observed[scenario_id] = payload
    return observed


def _scenario_artifact_class(spec: dict[str, Any]) -> str:
    artifact_class = spec.get("artifact_class")
    if isinstance(artifact_class, str) and artifact_class:
        return artifact_class
    generation = spec.get("generation")
    kind = generation.get("kind") if isinstance(generation, dict) else ""
    if kind == "error":
        return FAULT_INJECTION_FIXTURE
    if kind == "deployable_edit":
        return DEPLOYABLE_OPTIMIZED_SUBJECT
    if kind == "edit":
        return VALIDATION_SUBJECT_CHECKPOINT
    return "unknown"


def build_edit_artifact_summary(pack_dir: Path, scenarios_path: Path) -> dict[str, Any]:
    scenarios = _scenario_index(scenarios_path)
    observed = _first_metadata_by_scenario(pack_dir)
    deployable_validation = _first_deployable_validation_by_scenario(pack_dir)
    counts: Counter[str] = Counter()
    by_scenario: dict[str, dict[str, Any]] = {}

    for scenario_id, spec in sorted(scenarios.items()):
        artifact_class = _scenario_artifact_class(spec)
        counts[artifact_class] += 1
        metadata = observed.get(scenario_id, {})
        generation = (
            spec.get("generation") if isinstance(spec.get("generation"), dict) else {}
        )
        record: dict[str, Any] = {
            "artifact_class": artifact_class,
            "category": spec.get("category"),
            "failure_class": spec.get("failure_class"),
            "generation_kind": generation.get("kind")
            if isinstance(generation, dict)
            else None,
        }
        for field in (
            "edit_type",
            "optimized_deployment_backend",
            "storage_format",
            "actual_storage_format",
            "packed_quantized_storage",
            "runtime_memory_reduction",
            "backend",
        ):
            if field in metadata:
                record[field] = metadata[field]
        if metadata.get("schema") == EDIT_METADATA_SCHEMA:
            record["metadata_present"] = True
        elif metadata:
            record["metadata_present"] = False
        validation = deployable_validation.get(scenario_id, {})
        if validation:
            record["deployable_validation_ok"] = validation.get("ok")
            record["load_smoke"] = validation.get("load_smoke")
            record["inference_smoke"] = validation.get("inference_smoke")
            if "backend" not in record and validation.get("backend"):
                record["backend"] = validation.get("backend")
        by_scenario[scenario_id] = record

    lanes = {
        "validation_subjects": counts.get(VALIDATION_SUBJECT_CHECKPOINT, 0) > 0,
        "deployable_subjects": counts.get(DEPLOYABLE_OPTIMIZED_SUBJECT, 0) > 0,
        "fault_injection": counts.get(FAULT_INJECTION_FIXTURE, 0) > 0,
    }
    deployable_records = [
        record
        for record in by_scenario.values()
        if record.get("artifact_class") == DEPLOYABLE_OPTIMIZED_SUBJECT
    ]
    return {
        "schema": EDIT_ARTIFACT_SUMMARY_SCHEMA,
        "counts": dict(sorted(counts.items())),
        "evidence_lanes": lanes,
        "deployable_subjects": {
            "count": len(deployable_records),
            "backends": sorted(
                {
                    str(record.get("backend"))
                    for record in deployable_records
                    if record.get("backend")
                }
            ),
            "all_reload_smokes_passed": bool(deployable_records)
            and all(record.get("load_smoke") is True for record in deployable_records),
            "all_inference_smokes_passed": bool(deployable_records)
            and all(
                record.get("inference_smoke") is True for record in deployable_records
            ),
        },
        "by_scenario": by_scenario,
    }


def _edit_artifact_summary(args: argparse.Namespace) -> int:
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_edit_artifact_summary(Path(args.pack_dir), Path(args.scenarios))
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return 0


def _structural_metric_section(
    source_report: dict[str, Any] | None,
) -> dict[str, Any]:
    metrics = {}
    if isinstance(source_report, dict):
        raw_metrics = source_report.get("metrics")
        if isinstance(raw_metrics, dict):
            metrics = raw_metrics
    primary_metric = metrics.get("primary_metric")
    if not isinstance(primary_metric, dict):
        primary_metric = {}

    payload: dict[str, Any] = {
        "kind": primary_metric.get("kind") or "ppl_causal",
        "unit": primary_metric.get("unit") or "ppl",
        "direction": primary_metric.get("direction") or "lower",
        "aggregation_scope": primary_metric.get("aggregation_scope") or "token",
        "paired": bool(primary_metric.get("paired", True)),
        "gating_basis": primary_metric.get("gating_basis") or "upper",
        "supports_bootstrap": bool(primary_metric.get("supports_bootstrap", True)),
        "invalid": True,
        "degraded": True,
    }

    preview = primary_metric.get("preview")
    final = primary_metric.get("final")
    if preview is None:
        preview = metrics.get("ppl_preview")
    if final is None:
        final = metrics.get("ppl_final")
    if preview is not None:
        payload["preview"] = preview
    if final is not None:
        payload["final"] = final

    drift_band = primary_metric.get("drift_band")
    if isinstance(drift_band, dict):
        payload["drift_band"] = drift_band

    return payload


def _build_structural_base_report(
    source_report: dict[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(source_report, dict):
        raise ValueError("source report is required to build a structural failure cert")

    meta = source_report.get("meta")
    if not isinstance(meta, dict):
        meta = {}

    data = source_report.get("data")
    if not isinstance(data, dict):
        data = {}

    edit = source_report.get("edit")
    if not isinstance(edit, dict):
        edit = {}

    artifacts = source_report.get("artifacts")
    if not isinstance(artifacts, dict):
        artifacts = {}

    evaluation_windows = source_report.get("evaluation_windows")
    if not isinstance(evaluation_windows, dict):
        evaluation_windows = {}

    def _non_negative_int(value: Any) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return 0
        return parsed if parsed >= 0 else 0

    preview_windows = _non_negative_int(data.get("preview_n"))
    final_windows = _non_negative_int(data.get("final_n"))

    return {
        "schema_version": "v1",
        "run_id": str(source_report.get("run_id") or "run"),
        "edit_name": str(edit.get("name") or "unknown"),
        "plugins": {
            "adapters": [],
            "edits": [],
            "guards": [],
        },
        "meta": copy.deepcopy(meta),
        "dataset": {
            "provider": str(data.get("dataset") or "unknown"),
            "seq_len": _non_negative_int(data.get("seq_len")),
            "hash": {
                "preview": "",
                "final": "",
                "dataset": None,
                "preview_tokens": None,
                "final_tokens": None,
                "total_tokens": 0,
                "source": "config_fallback",
            },
            "windows": {
                "preview": preview_windows,
                "final": final_windows,
                "seed": meta.get("seed"),
                "stats": {},
            },
        },
        "primary_metric": _structural_metric_section(source_report),
        "artifacts": copy.deepcopy(artifacts),
        "evaluation_windows": copy.deepcopy(evaluation_windows),
        "flags": copy.deepcopy(source_report.get("flags", {})),
    }


def build_structural_failure_report(
    *,
    error_type: str,
    message: str,
    base_report: dict[str, Any],
    source_report: dict[str, Any] | None,
    source_report_path: str | None,
    edited_report_path: str | None,
    edited_events_path: str | None,
) -> dict[str, Any]:
    payload = copy.deepcopy(base_report)
    source_run_id = str(payload.get("run_id") or "run")
    payload["run_id"] = f"{source_run_id}-structural-failure-{error_type}"

    meta = payload.get("meta")
    if not isinstance(meta, dict):
        meta = {}
        payload["meta"] = meta
    meta["structural_failure"] = {
        "error_type": error_type,
        "message": message,
        "source_report": source_report_path,
        "edited_report": edited_report_path,
        "edited_events": edited_events_path,
    }

    validation = payload.get("validation")
    if not isinstance(validation, dict):
        validation = {}
    validation.update(
        {
            "invariants_pass": False,
            "primary_metric_acceptable": False,
            "spectral_stable": False,
            "rmt_stable": False,
            "preview_final_drift_acceptable": False,
            "guard_overhead_acceptable": True,
            "primary_metric_tail_acceptable": False,
        }
    )
    payload["validation"] = validation

    guard_overhead = payload.get("guard_overhead")
    if not isinstance(guard_overhead, dict):
        guard_overhead = {}
    guard_overhead["evaluated"] = bool(guard_overhead.get("evaluated", True))
    payload["guard_overhead"] = guard_overhead

    primary_metric = payload.get("primary_metric")
    if not isinstance(primary_metric, dict):
        primary_metric = {}
    primary_metric.update(_structural_metric_section(source_report))
    primary_metric["invalid"] = True
    primary_metric["degraded"] = True
    primary_metric["degraded_reason"] = "structural_failure"
    primary_metric.pop("ratio_vs_baseline", None)
    payload["primary_metric"] = primary_metric

    payload["_evidence_pack_structural_failure"] = {
        "format": "evidence-pack-structural-failure-report-v1",
        "error_type": error_type,
        "message": message,
        "source_report": source_report_path,
        "edited_report": edited_report_path,
        "edited_events": edited_events_path,
    }

    payload["invariants"] = {
        "status": "fail",
        "failures": [
            {
                "check": "error_injection",
                "type": "evidence_pack_structural_failure",
                "severity": "fatal",
                "detail": {
                    "error_type": error_type,
                    "message": message,
                },
            }
        ],
    }

    spectral = payload.get("spectral")
    if not isinstance(spectral, dict):
        spectral = {}
    spectral["status"] = "structural_failure"
    payload["spectral"] = spectral

    rmt = payload.get("rmt")
    if not isinstance(rmt, dict):
        rmt = {}
    rmt["status"] = "structural_failure"
    payload["rmt"] = rmt

    return payload


def _write_structural_runtime_manifest(
    *,
    out_path: Path,
    source_runtime_manifest: dict[str, Any] | None,
    error_type: str,
    message: str,
) -> None:
    if not isinstance(source_runtime_manifest, dict):
        return

    manifest_payload = copy.deepcopy(source_runtime_manifest)
    manifest_payload["generated_at_utc"] = datetime.now(UTC).isoformat()
    manifest_payload["report"] = {
        "path": str(out_path.resolve()),
        "filename": out_path.name,
        "sha256": hashlib.sha256(out_path.read_bytes()).hexdigest(),
    }
    context = manifest_payload.get("context")
    if not isinstance(context, dict):
        context = {}
    context["evidence_pack_structural_failure"] = {
        "error_type": error_type,
        "message": message,
    }
    manifest_payload["context"] = context
    manifest_path = out_path.parent / "runtime.manifest.json"
    manifest_path.write_text(
        json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _structural_failure_report(args: argparse.Namespace) -> int:
    out_path = Path(args.out)
    source_report_path = Path(args.source_report) if args.source_report else None
    source_report = _load_json_optional(source_report_path)
    source_runtime_manifest_path = (
        Path(args.source_runtime_manifest) if args.source_runtime_manifest else None
    )
    source_runtime_manifest = _load_json_optional(source_runtime_manifest_path)
    base_report = _build_structural_base_report(source_report)

    payload = build_structural_failure_report(
        error_type=str(args.error_type),
        message=str(args.message),
        base_report=base_report,
        source_report=source_report,
        source_report_path=str(source_report_path) if source_report_path else None,
        edited_report_path=str(args.edited_report) if args.edited_report else None,
        edited_events_path=str(args.edited_events) if args.edited_events else None,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    _write_structural_runtime_manifest(
        out_path=out_path,
        source_runtime_manifest=source_runtime_manifest,
        error_type=str(args.error_type),
        message=str(args.message),
    )
    return 0


def _parse_window_candidate(value: str) -> dict[str, int]:
    parts = [segment.strip() for segment in value.split(":")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "candidate must be formatted as seq_len:preview_n:final_n"
        )
    try:
        seq_len, preview_n, final_n = (int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("candidate values must be integers") from exc
    if seq_len <= 0 or preview_n <= 0 or final_n <= 0:
        raise argparse.ArgumentTypeError("candidate values must be positive")
    return {
        "seq_len": seq_len,
        "stride": seq_len,
        "preview_n": preview_n,
        "final_n": final_n,
    }


def _resolve_min_tokens_target(tier: str, profile: str | None) -> int:
    from invarlock.core.auto_tuning import resolve_tier_policies

    resolved = resolve_tier_policies((tier or "balanced").lower(), profile=profile)
    metrics = resolved.get("metrics", {}) if isinstance(resolved, dict) else {}
    pm_ratio = metrics.get("pm_ratio", {}) if isinstance(metrics, dict) else {}
    try:
        return int(pm_ratio.get("min_tokens", 0) or 0)
    except (TypeError, ValueError, OverflowError):
        return 0


def _plan_effective_windows(args: argparse.Namespace) -> int:
    from invarlock.eval.data import get_provider
    from invarlock.eval.window_planning import choose_first_token_sufficient_candidate
    from invarlock.model_profile import detect_model_profile, resolve_tokenizer

    profile = detect_model_profile(args.model_path, adapter="hf_auto")
    tokenizer, _ = resolve_tokenizer(profile)
    provider = get_provider(args.dataset_provider, device_hint="cpu")

    result = choose_first_token_sufficient_candidate(
        data_provider=provider,
        tokenizer=tokenizer,
        split=args.split,
        seed=int(args.seed),
        candidates=args.candidate,
        min_tokens_target=_resolve_min_tokens_target(args.tier, args.profile),
        headroom_ratio=float(args.headroom_ratio),
        profile=args.profile,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evidence-pack task helper tools.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download_parser = subparsers.add_parser(
        "download-baseline",
        help="Download a pinned baseline model with evidence-pack storage policy.",
    )
    download_parser.add_argument("--model-id", required=True)
    download_parser.add_argument("--output-dir", required=True)
    download_parser.add_argument("--success-marker", default="")
    download_parser.set_defaults(func=_download_baseline)

    normalize_parser = subparsers.add_parser(
        "normalize-staged-preset",
        help="Normalize a staged preset for evaluate runtime.",
    )
    normalize_parser.add_argument("--preset", required=True)
    normalize_parser.add_argument("--baseline-report")
    normalize_parser.add_argument("--seq-len", type=int)
    normalize_parser.add_argument("--stride", type=int)
    normalize_parser.add_argument("--preview-n", type=int)
    normalize_parser.add_argument("--final-n", type=int)
    normalize_parser.add_argument("--skip-overhead-check", action="store_true")
    normalize_parser.set_defaults(func=_normalize_staged_preset)

    error_model_parser = subparsers.add_parser(
        "create-error-model",
        help="Create an evidence-pack structural error model.",
    )
    error_model_parser.add_argument("baseline_path")
    error_model_parser.add_argument("output_path")
    error_model_parser.add_argument("error_type")
    error_model_parser.set_defaults(func=_create_error_model)

    adapter_parser = subparsers.add_parser(
        "resolve-adapter",
        help="Resolve the InvarLock adapter for a model id or local path.",
    )
    adapter_parser.add_argument("model_id_or_path")
    adapter_parser.set_defaults(func=_resolve_adapter)

    edit_parser = subparsers.add_parser(
        "resolve-edit-params",
        help="Resolve an edit spec to shell-friendly JSON.",
    )
    edit_parser.add_argument("model_output_dir")
    edit_parser.add_argument("edit_spec")
    edit_parser.add_argument("version_hint", nargs="?", default="")
    edit_parser.set_defaults(func=_resolve_edit_params)

    revision_parser = subparsers.add_parser(
        "model-revision",
        help="Read a model revision from state/model_revisions.json.",
    )
    revision_parser.add_argument("revisions_json")
    revision_parser.add_argument("model_id")
    revision_parser.set_defaults(func=_model_revision)

    report_parser = subparsers.add_parser(
        "evaluation-report",
        help="Generate evaluation.report.json from report.json.",
    )
    report_parser.add_argument("--report", required=True)
    report_parser.add_argument("--out", required=True)
    report_parser.set_defaults(func=_evaluation_report)

    baseline_parser = subparsers.add_parser(
        "validate-baseline-report",
        help="Validate a generated baseline report contract.",
    )
    baseline_parser.add_argument("baseline_report")
    baseline_parser.add_argument("expected_adapter")
    baseline_parser.add_argument("expected_profile")
    baseline_parser.add_argument("expected_tier")
    baseline_parser.set_defaults(func=_validate_baseline_report)

    profile_parser = subparsers.add_parser(
        "write-model-profile",
        help="Write model_profile.json next to a downloaded baseline config.",
    )
    profile_parser.add_argument("baseline_dir")
    profile_parser.add_argument("model_id")
    profile_parser.set_defaults(func=_write_model_profile)

    repair_parser = subparsers.add_parser(
        "repair-missing-tensors-config",
        help="Repair legacy missing_tensors layer-drop config metadata.",
    )
    repair_parser.add_argument("baseline_config")
    repair_parser.add_argument("error_config")
    repair_parser.set_defaults(func=_repair_missing_tensors_config)

    plan_parser = subparsers.add_parser(
        "plan-effective-windows",
        help="Plan CI window schedules using effective post-dedupe token counts.",
    )
    plan_parser.add_argument("--model-path", required=True)
    plan_parser.add_argument("--dataset-provider", default="wikitext2")
    plan_parser.add_argument("--split", default="validation")
    plan_parser.add_argument("--seed", type=int, default=42)
    plan_parser.add_argument("--tier", default="balanced")
    plan_parser.add_argument("--profile", default="ci")
    plan_parser.add_argument("--headroom-ratio", type=float, default=1.05)
    plan_parser.add_argument(
        "--candidate",
        action="append",
        type=_parse_window_candidate,
        default=[],
        help="Candidate schedule as seq_len:preview_n:final_n",
    )
    plan_parser.set_defaults(func=_plan_effective_windows)

    structural_parser = subparsers.add_parser(
        "structural-failure-report",
        help="Emit an evaluation.report.json for structural error evaluation failures.",
    )
    structural_parser.add_argument("--error-type", required=True)
    structural_parser.add_argument("--out", required=True)
    structural_parser.add_argument("--message", required=True)
    structural_parser.add_argument("--source-report")
    structural_parser.add_argument("--source-runtime-manifest")
    structural_parser.add_argument("--edited-report")
    structural_parser.add_argument("--edited-events")
    structural_parser.set_defaults(func=_structural_failure_report)

    summary_parser = subparsers.add_parser(
        "edit-artifact-summary",
        help="Write edit artifact class summary.",
    )
    summary_parser.add_argument("--pack-dir", required=True)
    summary_parser.add_argument("--scenarios", required=True)
    summary_parser.add_argument("--out", required=True)
    summary_parser.set_defaults(func=_edit_artifact_summary)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
