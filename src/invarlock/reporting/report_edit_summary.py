"""Edit-summary owners for evaluation report construction."""

from __future__ import annotations

import copy
from collections.abc import Iterable
from typing import Any

from .report_types import RunReport
from .utils import _coerce_int, _get_mapping, _infer_scope_from_modules

_NON_FATAL_EXCEPTIONS = (AttributeError, TypeError, ValueError)


def analyze_bitwidth_map(bitwidth_map: dict[str, Any]) -> dict[str, Any]:
    """Analyze bitwidth changes for compression diagnostics."""
    if not bitwidth_map:
        return {}

    bitwidths = []
    for module_info in bitwidth_map.values():
        if isinstance(module_info, dict) and "bitwidth" in module_info:
            bitwidths.append(module_info["bitwidth"])

    if not bitwidths:
        return {}

    return {
        "total_modules": len(bitwidths),
        "bitwidths_used": list(set(bitwidths)),
        "avg_bitwidth": sum(bitwidths) / len(bitwidths),
        "min_bitwidth": min(bitwidths),
        "max_bitwidth": max(bitwidths),
    }


def compute_savings_summary(deltas: dict[str, Any]) -> dict[str, Any]:
    """Compute realized vs theoretical savings summary for edits."""
    summary = _get_mapping(deltas, "savings")
    rank_map = _get_mapping(deltas, "rank_map")
    deploy_mode: str | None = summary.get("deploy_mode") if summary else None

    def _accumulate(value: Any) -> int:
        coerced = _coerce_int(value)
        return coerced if coerced is not None else 0

    if rank_map:
        total_realized = 0
        total_theoretical = 0
        for info in rank_map.values():
            total_realized += _accumulate(info.get("realized_params_saved"))
            total_theoretical += _accumulate(info.get("theoretical_params_saved"))
            if deploy_mode is None:
                mode_candidate = info.get("deploy_mode")
                if isinstance(mode_candidate, str):
                    deploy_mode = mode_candidate
    else:
        total_realized = (
            _accumulate(summary.get("total_realized_params_saved")) if summary else 0
        )
        total_theoretical = (
            _accumulate(summary.get("total_theoretical_params_saved")) if summary else 0
        )

    mode = "none"
    if total_realized > 0:
        mode = "realized"
    elif total_theoretical > 0:
        mode = "theoretical"
    elif deploy_mode == "recompose" and any(
        isinstance(info, dict) and not info.get("skipped", False)
        for info in rank_map.values()
    ):
        mode = "theoretical"

    result = {
        "mode": mode,
        "total_realized_params_saved": total_realized,
        "total_theoretical_params_saved": total_theoretical,
    }
    if deploy_mode:
        result["deploy_mode"] = deploy_mode
    return result


def extract_rank_information(
    edit_config: dict[str, Any], deltas: dict[str, Any]
) -> dict[str, Any]:
    """Extract rank information for SVD-based compression."""
    rank_info = {}

    if "frac" in edit_config:
        rank_info["target_fraction"] = edit_config["frac"]
    if "rank_policy" in edit_config:
        rank_info["rank_policy"] = edit_config["rank_policy"]

    rank_map = deltas.get("rank_map")
    if isinstance(rank_map, dict) and rank_map:
        per_module = {}
        skipped = []
        for module_name, info in rank_map.items():
            per_module[module_name] = {
                "rank": info.get("rank"),
                "params_saved": info.get("params_saved"),
                "energy_retained": info.get("energy_retained"),
                "deploy_mode": info.get("deploy_mode"),
                "savings_mode": info.get("savings_mode"),
                "realized_params_saved": info.get("realized_params_saved"),
                "theoretical_params_saved": info.get("theoretical_params_saved"),
                "realized_params": info.get("realized_params"),
                "theoretical_params": info.get("theoretical_params"),
            }
            if info.get("skipped"):
                skipped.append(module_name)

        rank_info["per_module"] = per_module
        if skipped:
            rank_info["skipped_modules"] = skipped
        rank_info["savings_summary"] = compute_savings_summary(deltas)
    else:
        summary = _get_mapping(deltas, "savings")
        if summary:
            rank_info["savings_summary"] = compute_savings_summary(deltas)

    return rank_info


def extract_compression_diagnostics(
    edit_name: str,
    edit_config: dict[str, Any],
    deltas: dict[str, Any],
    structure: dict[str, Any],
    inference_record: dict[str, Any] | None,
) -> dict[str, Any]:
    """Extract comprehensive compression diagnostics."""
    diagnostics: dict[str, Any] = {}
    if not isinstance(inference_record, dict):
        inference_record = {}
    flags = inference_record.setdefault(
        "flags",
        dict.fromkeys(("scope", "seed", "rank_policy", "frac"), False),
    )
    sources = inference_record.setdefault("sources", {})
    log_entries = inference_record.setdefault("log", [])
    if not isinstance(flags, dict):
        flags = {}
        inference_record["flags"] = flags
    if not isinstance(sources, dict):
        sources = {}
        inference_record["sources"] = sources
    if not isinstance(log_entries, list):
        log_entries = []
        inference_record["log"] = log_entries

    def mark(field: str, value: Any, source: str) -> bool:
        if value in (None, "unknown"):
            return False
        current = edit_config.get(field)
        if current not in (None, "unknown"):
            return False
        edit_config[field] = value
        if not bool(flags.get(field)):
            flags[field] = True
            sources[field] = source
            log_entries.append(f"{field} inferred from {source}: {value}")
        return True

    params_changed = _coerce_int(deltas.get("params_changed")) or 0
    diagnostics["execution_status"] = (
        "successful" if params_changed > 0 else "no_modifications"
    )

    bitwidth_map = _get_mapping(deltas, "bitwidth_map")
    num_quantized_modules = len(bitwidth_map) if bitwidth_map else 0

    target_analysis: dict[str, Any] = {
        "modules_found": num_quantized_modules
        if bitwidth_map
        else deltas.get("layers_modified", 0),
        "modules_eligible": num_quantized_modules
        if bitwidth_map
        else deltas.get("layers_modified", 0),
        "modules_modified": num_quantized_modules
        if bitwidth_map
        else deltas.get("layers_modified", 0),
        "scope": edit_config.get("scope", "unknown"),
    }
    diagnostics["target_analysis"] = target_analysis
    existing_scope = edit_config.get("scope")
    if existing_scope not in (None, "unknown"):
        target_analysis["scope"] = existing_scope
    else:
        module_iter: Iterable[str]
        source_label = "modules"
        if isinstance(bitwidth_map, dict) and bitwidth_map:
            module_iter = bitwidth_map.keys()
            source_label = "bitwidth_map"
        else:
            rank_map = _get_mapping(deltas, "rank_map")
            if rank_map:
                module_iter = rank_map.keys()
                source_label = "rank_map"
            else:
                module_iter = []
        inferred_scope = _infer_scope_from_modules(module_iter)
        if inferred_scope != "unknown" and mark("scope", inferred_scope, source_label):
            target_analysis["scope"] = inferred_scope
    target_analysis["scope"] = edit_config.get(
        "scope", target_analysis.get("scope", "unknown")
    )

    param_analysis: dict[str, Any] = {}

    rank_map = _get_mapping(deltas, "rank_map")
    if rank_map:
        modules_modified = [
            name for name, info in rank_map.items() if not info.get("skipped", False)
        ]
        diagnostics["rank_summary"] = {
            "modules": rank_map,
            "modules_modified": len(modules_modified),
            "skipped_modules": [
                name for name, info in rank_map.items() if info.get("skipped", False)
            ],
        }
        target_analysis["modules_modified"] = len(modules_modified)
        if modules_modified:
            diagnostics["execution_status"] = (
                "partial"
                if len(modules_modified) < len(rank_map)
                else diagnostics["execution_status"]
            )

    if "quant" in edit_name.lower():
        actual_bitwidth: Any = "unknown"
        if bitwidth_map:
            first_module: dict[str, Any] = next(iter(bitwidth_map.values()), {})
            actual_bitwidth = first_module.get(
                "bitwidth", edit_config.get("bitwidth", "unknown")
            )
        else:
            actual_bitwidth = edit_config.get("bitwidth", "unknown")

        param_analysis["bitwidth"] = {
            "value": actual_bitwidth,
            "effectiveness": "applied" if params_changed > 0 else "ineffective",
        }

        if bitwidth_map:
            first_module = next(iter(bitwidth_map.values()), {})
            group_size_used = first_module.get("group_size")
            param_analysis["group_size"] = {
                "value": group_size_used,
                "effectiveness": "used" if group_size_used else "per_channel",
            }
        elif edit_config.get("group_size") not in (None, "unknown"):
            group_size_cfg = edit_config["group_size"]
            param_analysis["group_size"] = {
                "value": group_size_cfg,
                "effectiveness": "used" if group_size_cfg else "per_channel",
            }

        if edit_config.get("clamp_ratio") not in (None, "unknown"):
            param_analysis["clamp_ratio"] = {
                "value": edit_config["clamp_ratio"],
                "effectiveness": "applied"
                if edit_config["clamp_ratio"] > 0
                else "disabled",
            }
    elif "svd" in edit_name.lower() or "rank" in edit_name.lower():
        param_analysis["frac"] = {
            "value": edit_config.get("frac", "unknown"),
            "effectiveness": "applied" if params_changed > 0 else "too_conservative",
        }
        param_analysis["rank_policy"] = {
            "value": edit_config.get("rank_policy", "unknown"),
            "effectiveness": "used",
        }

    diagnostics["parameter_analysis"] = param_analysis

    algo_details: dict[str, Any] = {}
    algo_details["scope_targeting"] = edit_config.get("scope", "unknown")
    algo_details["seed"] = edit_config.get("seed", "unknown")

    if "quant" in edit_name.lower() and bitwidth_map:
        algo_details["modules_quantized"] = len(bitwidth_map)
        algo_details["quantization_type"] = (
            "per_channel"
            if not any(m.get("group_size") for m in bitwidth_map.values())
            else "grouped"
        )

        total_quantized_params = sum(m.get("params", 0) for m in bitwidth_map.values())
        algo_details["total_params_quantized"] = total_quantized_params

        memory_saved_bytes = 0
        if isinstance(actual_bitwidth, int) and actual_bitwidth < 32:
            memory_saved_bytes = total_quantized_params * (32 - actual_bitwidth) / 8

        algo_details["estimated_memory_saved_mb"] = round(
            memory_saved_bytes / (1024 * 1024), 2
        )

    diagnostics["algorithm_details"] = algo_details

    warnings: list[str] = []
    if params_changed == 0:
        warnings.append(
            "No parameters were modified - algorithm may be too conservative"
        )
        warnings.append("Check scope configuration and parameter thresholds")

        if edit_config.get("scope") == "ffn":
            warnings.append(
                "FFN scope may not match model architecture - try 'all' scope"
            )

        if "frac" in edit_config and edit_config["frac"] < 0.1:
            warnings.append(
                f"Fraction {edit_config['frac']} may be too small for meaningful compression"
            )

    diagnostics["warnings"] = warnings

    diagnostics["inferred"] = flags
    if sources:
        diagnostics["inference_source"] = sources
    if log_entries:
        diagnostics["inference_log"] = log_entries

    return diagnostics


def extract_structural_deltas(report: RunReport) -> dict[str, Any]:
    """Extract structural parameter changes with compression diagnostics."""
    edit_section = _get_mapping(report, "edit")
    deltas = _get_mapping(edit_section, "deltas")

    primary_config = _get_mapping(edit_section, "plan")
    if not primary_config:
        primary_config = _get_mapping(edit_section, "config")
    edit_config = dict(primary_config) if primary_config else {}

    inference_record: dict[str, Any] = {
        "flags": dict.fromkeys(("scope", "seed", "rank_policy", "frac"), False),
        "sources": {},
        "log": [],
    }
    flags = inference_record["flags"]
    sources = inference_record["sources"]
    log_entries = inference_record["log"]

    def _infer(field: str, value: Any, source: str) -> bool:
        if value in (None, "unknown"):
            return False
        current = edit_config.get(field)
        if current not in (None, "unknown"):
            return False
        edit_config[field] = value
        flags[field] = True
        sources[field] = source
        log_entries.append(f"{field} inferred from {source}: {value}")
        return True

    if isinstance(edit_section, dict):
        for key, value in edit_section.items():
            if key in {"plan", "config", "deltas"}:
                continue
            if value is None or isinstance(value, dict):
                continue
            edit_config.setdefault(key, value)

        plan_digest = str(edit_section.get("plan_digest", "")).lower()
        if "energy" in plan_digest:
            _infer("rank_policy", "energy", "plan_digest")

        if "energy_" in plan_digest and not edit_config.get("frac"):
            try:
                fraction_str = plan_digest.split("energy_")[1].split("_")[0]
                _infer("frac", float(fraction_str), "plan_digest")
            except (IndexError, ValueError):
                pass
        if not edit_config.get("scope"):
            if "ffn" in plan_digest:
                _infer("scope", "ffn", "plan_digest")
            elif "attn" in plan_digest:
                _infer("scope", "attn", "plan_digest")
            elif "embed" in plan_digest or "embedding" in plan_digest:
                _infer("scope", "embed", "plan_digest")
    edit_name = str(edit_section.get("name") or "unknown")

    structure: dict[str, Any] = {
        "params_changed": deltas.get("params_changed", 0),
        "layers_modified": deltas.get("layers_modified", 0),
    }

    if deltas.get("sparsity") is not None:
        structure["sparsity"] = deltas["sparsity"]

    if deltas.get("bitwidth_map"):
        structure["bitwidths"] = deltas["bitwidth_map"]
        structure["bitwidth_analysis"] = analyze_bitwidth_map(deltas["bitwidth_map"])

    if "rank" in edit_name.lower() or "svd" in edit_name.lower():
        structure["ranks"] = extract_rank_information(edit_config, deltas)
        structure["savings"] = compute_savings_summary(deltas)
    else:
        structure["ranks"] = {}

    compression_diag = extract_compression_diagnostics(
        edit_name, edit_config, deltas, structure, inference_record
    )
    structure["compression_diagnostics"] = compression_diag

    target_analysis = _get_mapping(compression_diag, "target_analysis")
    algo_details = _get_mapping(compression_diag, "algorithm_details")
    if "algorithm_details" not in compression_diag:
        compression_diag["algorithm_details"] = algo_details

    fallback_scope = edit_section.get("scope")
    if _infer("scope", fallback_scope, "report.edit.scope"):
        target_analysis["scope"] = fallback_scope
    elif fallback_scope and target_analysis.get("scope") in (None, "unknown"):
        target_analysis["scope"] = fallback_scope

    edit_seed = edit_section.get("seed")
    _infer("seed", edit_seed, "report.edit.seed")

    if not bool(flags.get("seed")):
        meta = _get_mapping(report, "meta")
        meta_seed = None
        seeds_bundle = meta.get("seeds")
        if isinstance(seeds_bundle, dict):
            meta_seed = seeds_bundle.get("python")
        if meta_seed is None:
            meta_seed = meta.get("seed")
        _infer("seed", meta_seed, "report.meta.seeds")

    target_analysis["scope"] = edit_config.get(
        "scope", target_analysis.get("scope", "unknown")
    )
    algo_details["scope_targeting"] = target_analysis.get("scope", "unknown")

    final_seed = edit_config.get("seed", algo_details.get("seed", "unknown"))
    algo_details["seed"] = final_seed

    compression_diag["inferred"] = flags
    if sources:
        compression_diag["inference_source"] = sources
    if log_entries:
        compression_diag["inference_log"] = log_entries

    return structure


def extract_edit_metadata(
    report: RunReport, plugin_provenance: dict[str, Any]
) -> dict[str, Any]:
    """Extract edit-level provenance and configuration metadata."""
    edit_section = _get_mapping(report, "edit")
    if not edit_section:
        return {}

    edit_name = str(edit_section.get("name", "") or "")

    plugin_edit: dict[str, Any] = {}
    if isinstance(plugin_provenance, dict):
        candidate = plugin_provenance.get("edit")
        if isinstance(candidate, dict):
            plugin_edit = candidate

    algorithm = edit_section.get("algorithm")
    if not algorithm:
        algorithm = edit_name or ""
    try:
        alg_lower = str(algorithm).strip().lower()
    except _NON_FATAL_EXCEPTIONS:  # pragma: no cover
        alg_lower = ""
    allowed_algorithms = {
        "fine_tune",
        "fp8_quant",
        "lora_merge",
        "lowrank_svd",
        "magnitude_prune",
        "noop",
        "quant_rtn",
        "custom",
    }
    if alg_lower not in allowed_algorithms:
        algorithm = ""

    algorithm_version = (
        edit_section.get("algorithm_version") or plugin_edit.get("version") or ""
    )

    implementation = (
        edit_section.get("implementation") or plugin_edit.get("module") or ""
    )
    if isinstance(implementation, str) and (
        "structured" in implementation.lower() or "lowrank" in implementation.lower()
    ):
        implementation = ""

    plan_dict: dict[str, Any] = {}
    raw_plan = edit_section.get("plan")
    if isinstance(raw_plan, dict):
        plan_dict = copy.deepcopy(raw_plan)
    else:
        config_section = edit_section.get("config")
        if isinstance(config_section, dict):
            config_plan = config_section.get("plan")
            if isinstance(config_plan, dict):
                plan_dict = copy.deepcopy(config_plan)
            else:
                plan_dict = copy.deepcopy(config_section)

    scope = plan_dict.get("scope") or edit_section.get("scope")
    ranking = plan_dict.get("ranking") or edit_section.get("ranking") or ""
    grouping = plan_dict.get("grouping") or edit_section.get("grouping")

    budgets: dict[str, Any] = {}
    for key in (
        "head_budget",
        "mlp_budget",
        "heads",
        "mlp",
        "neuron_budget",
        "ffn_budget",
    ):
        value = plan_dict.get(key)
        if isinstance(value, dict):
            budgets[key] = copy.deepcopy(value)

    target_sparsity = plan_dict.get("target_sparsity")
    if isinstance(target_sparsity, int | float):
        budgets["target_sparsity"] = float(target_sparsity)

    if not scope:
        if "head_budget" in budgets and "mlp_budget" in budgets:
            scope = "heads+ffn"
        elif "head_budget" in budgets:
            scope = "heads"
        elif "mlp_budget" in budgets:
            scope = "ffn"
        else:
            scope = ""

    if not grouping:
        grouping = "auto" if scope == "heads" else ("none" if scope else "")

    seed_candidate = plan_dict.get("seed", edit_section.get("seed"))
    if seed_candidate is None:
        meta_section = _get_mapping(report, "meta")
        seed_candidate = meta_section.get("seed")
    seed_value = _coerce_int(seed_candidate)

    edit_metadata: dict[str, Any] = {
        "name": edit_name,
        "algorithm": algorithm,
        "algorithm_version": str(algorithm_version),
        "implementation": str(implementation),
        "scope": scope,
        "ranking": ranking,
        "grouping": grouping,
        "budgets": budgets,
        "seed": seed_value,
        "plan_digest": str(edit_section.get("plan_digest") or ""),
        "mask_digest": str(edit_section.get("mask_digest") or ""),
    }

    if plan_dict:
        edit_metadata["plan"] = copy.deepcopy(plan_dict)

    for optional_key in (
        "edit_provenance",
        "edit_impact",
        "edit_topology",
        "delta_privacy",
    ):
        optional_value = edit_section.get(optional_key)
        if not isinstance(optional_value, dict):
            config_section = edit_section.get("config")
            if isinstance(config_section, dict):
                optional_value = config_section.get(optional_key)
        if isinstance(optional_value, dict):
            edit_metadata[optional_key] = copy.deepcopy(optional_value)

    if not budgets:
        edit_metadata.pop("budgets")
    if seed_value is None:
        edit_metadata.pop("seed")
    if not scope:
        edit_metadata.pop("scope")
    if not ranking:
        edit_metadata.pop("ranking")
    if not grouping:
        edit_metadata.pop("grouping")

    return edit_metadata


__all__ = [
    "analyze_bitwidth_map",
    "compute_savings_summary",
    "extract_compression_diagnostics",
    "extract_edit_metadata",
    "extract_rank_information",
    "extract_structural_deltas",
]
