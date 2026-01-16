#!/usr/bin/env bash
# config_generator.sh - InvarLock config generation + certify helpers for proof packs.

# ============ INVARLOCK CONFIG FOR PROOF PACKS ============
generate_invarlock_config() {
    local model_path="$1"
    local output_yaml="$2"
    local edit_name="${3:-noop}"
    local seed="${4:-42}"
    local preview_n="${5:-${INVARLOCK_PREVIEW_WINDOWS}}"
    local final_n="${6:-${INVARLOCK_FINAL_WINDOWS}}"
    local bootstrap_n="${7:-${INVARLOCK_BOOTSTRAP_N:-2000}}"
    local seq_len="${8:-${INVARLOCK_SEQ_LEN}}"
    local stride="${9:-${INVARLOCK_STRIDE}}"
    local eval_batch="${10:-${INVARLOCK_EVAL_BATCH}}"

    # Use auto adapter for generic causal LM support (LLaMA, Mistral, Qwen, MPT, Falcon, etc.)
    local adapter="hf_causal_auto"
    local dataset_provider="${INVARLOCK_DATASET}"

    local attn_impl_yaml=""
    if [[ "${FLASH_ATTENTION_AVAILABLE}" == "true" ]]; then
        attn_impl_yaml='attn_implementation: "flash_attention_2"'
    else
        attn_impl_yaml='# flash_attention_2 not available'
    fi
    local accel_compile="true"
    local accel_tf32="true"
    local accel_benchmark="true"
    if [[ "${PACK_DETERMINISM}" == "strict" ]]; then
        accel_compile="false"
        accel_tf32="false"
        accel_benchmark="false"
    fi

    # Window overlap control (calibration and eval safety)
    local eval_overlap="${INVARLOCK_WINDOW_OVERLAP_FRACTION:-0.0}"

    # Optional: override guard order for the suite (comma-separated list).
    # Default is a lightweight chain to keep calibration tractable on 70B+.
    local guards_order_csv="${PACK_GUARDS_ORDER:-}"
    local -a guards_order=()
    if [[ -n "${guards_order_csv}" ]]; then
        IFS=',' read -ra guards_order <<< "${guards_order_csv}"
    else
        guards_order=("invariants" "spectral" "rmt" "variance" "invariants")
    fi
    local guards_order_yaml=""
    local g
    for g in "${guards_order[@]}"; do
        g="$(echo "${g}" | xargs)"
        [[ -z "${g}" ]] && continue
        guards_order_yaml+=$'    - '"${g}"$'\n'
    done
    if [[ -z "${guards_order_yaml}" ]]; then
        guards_order_yaml=$'    - invariants\n    - spectral\n    - rmt\n    - variance\n    - invariants\n'
    fi

    cat > "${output_yaml}" << YAML_EOF
# Auto-generated InvarLock config for proof packs
# Platform: proof pack runner


model:
  id: "${model_path}"
  adapter: "${adapter}"
  device: "auto"
  device_map: "auto"
  torch_dtype: "bfloat16"
  trust_remote_code: true
  low_cpu_mem_usage: true
  ${attn_impl_yaml}

dataset:
  provider: "${dataset_provider}"
  preview_n: ${preview_n}
  final_n: ${final_n}
  seq_len: ${seq_len}
  stride: ${stride}
  seed: ${seed}
  num_workers: 8
  prefetch_factor: 4
  pin_memory: true

edit:
  name: "${edit_name}"

guards:
  order:
${guards_order_yaml}

eval:
  window_overlap_fraction: ${eval_overlap}
  bootstrap:
    replicates: ${bootstrap_n}
    parallel: true
  max_pm_ratio: 2.0
  batch_size: ${eval_batch}


auto:
  enabled: true
  tier: "${INVARLOCK_TIER}"
  probes: 0

output:
  dir: "."

accelerator:
  compile: ${accel_compile}
  tf32: ${accel_tf32}
  benchmark: ${accel_benchmark}
  memory_efficient_attention: false
  gradient_checkpointing: false

memory:
  target_fraction: 0.92
  preallocate: true
  cache_enabled: true
YAML_EOF
}
export -f generate_invarlock_config

# ============ CALIBRATION RUN ============
run_single_calibration() {
    local model_path="$1"
    local run_dir="$2"
    local seed="$3"
    local preview_n="$4"
    local final_n="$5"
    local bootstrap_n="$6"
    local log_file="$7"
    local gpu_id="${8:-0}"
    local seq_len="${9:-${INVARLOCK_SEQ_LEN}}"
    local stride="${10:-${INVARLOCK_STRIDE}}"
    local eval_batch="${11:-${INVARLOCK_EVAL_BATCH}}"

    mkdir -p "${run_dir}"
    local config_yaml="${run_dir}/calibration_config.yaml"

    generate_invarlock_config \
        "${model_path}" \
        "${config_yaml}" \
        "noop" \
        "${seed}" \
        "${preview_n}" \
        "${final_n}" \
        "${bootstrap_n}" \
        "${seq_len}" \
        "${stride}" \
        "${eval_batch}"

    # Force no-overlap calibration to avoid pairing mismatches
    python3 - "${config_yaml}" <<'PY'
import sys, yaml, pathlib
path = pathlib.Path(sys.argv[1])
cfg = yaml.safe_load(path.read_text())
cfg.setdefault('eval', {})['window_overlap_fraction'] = 0.0
path.write_text(yaml.safe_dump(cfg, sort_keys=False))
PY

    # For large models, skip overhead check to avoid OOM (task-local via env)
    local model_size
    model_size=$(estimate_model_params "${model_path}")

    local cuda_devices="${CUDA_VISIBLE_DEVICES:-${gpu_id}}"

    local exit_code=0
    # Enforce no-overlap windows and skip overhead checks to avoid E001/pairing issues

    if [[ "${model_size}" == "70" || "${model_size}" == "72" || "${model_size}" == "moe" ]]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Large model (${model_size}): using INVARLOCK_SKIP_OVERHEAD_CHECK=1 for calibration" >> "${log_file}"
    fi

    INVARLOCK_WINDOW_OVERLAP_FRACTION=0.0 \
    INVARLOCK_SKIP_OVERHEAD_CHECK=1 \
    CUDA_VISIBLE_DEVICES="${cuda_devices}" invarlock run \
        --config "${config_yaml}" \
        --profile ci \
        --out "${run_dir}" \
        >> "${log_file}" 2>&1 || exit_code=$?

    # Generate certificate from report
    local report_file=$(find "${run_dir}" -name "report*.json" -type f 2>/dev/null | head -1)
    if [[ -n "${report_file}" ]]; then
        cp "${report_file}" "${run_dir}/baseline_report.json" 2>/dev/null || true

        python3 << CERT_EOF >> "${log_file}" 2>&1
import json
from pathlib import Path
try:
    from invarlock.reporting.certificate import make_certificate
    report_path = Path("${report_file}")
    cert_path = Path("${run_dir}") / "evaluation.cert.json"

    report = json.loads(report_path.read_text())
    cert = make_certificate(report, report)
    with open(cert_path, 'w') as f:
        json.dump(cert, f, indent=2)
except Exception as e:
    print(f"Certificate generation warning: {e}")
CERT_EOF
    fi

    return ${exit_code}
}
export -f run_single_calibration

# ============ CALIBRATION ORCHESTRATION ============
run_invarlock_calibration() {
    local model_path="$1"
    local model_name="$2"
    local output_dir="$3"
    local num_runs="$4"
    local preset_output_dir="$5"
    local gpu_id="${6:-0}"

    local model_size=$(estimate_model_params "${model_path}")
    local bootstrap_n="${INVARLOCK_BOOTSTRAP_N:-2000}"

    # Get model-size-aware configuration
    local config=$(get_model_invarlock_config "${model_size}")
    IFS=':' read -r effective_seq_len effective_stride effective_preview_n effective_final_n effective_eval_batch <<< "${config}"
    # Force non-overlapping windows for calibration to avoid pairing mismatches
    effective_stride="${effective_seq_len}"
    export INVARLOCK_WINDOW_OVERLAP_FRACTION=0.0

    # Log calibration start with proper model size label
    if [[ "${model_size}" == "moe" ]]; then
        log "  Calibration: ${num_runs} runs on GPU ${gpu_id} (MoE architecture)"
    else
        log "  Calibration: ${num_runs} runs on GPU ${gpu_id} (${model_size}B params)"
    fi
    log "    Config: seq_len=${effective_seq_len}, stride=${effective_stride}, windows=${effective_preview_n}+${effective_final_n}"

    mkdir -p "${output_dir}" "${preset_output_dir}"

    local calibration_failures=0
    for run in $(seq 1 "${num_runs}"); do
        local seed=$((41 + run))
        local run_dir="${output_dir}/run_${run}"
        local run_log="${OUTPUT_DIR}/logs/calibration_${model_name}_run${run}.log"

        if ! run_single_calibration \
            "${model_path}" \
            "${run_dir}" \
            "${seed}" \
            "${effective_preview_n}" \
            "${effective_final_n}" \
            "${bootstrap_n}" \
            "${run_log}" \
            "${gpu_id}" \
            "${effective_seq_len}" \
            "${effective_stride}" \
            "${effective_eval_batch}"; then
            log "  WARNING: Calibration run ${run} failed for ${model_name}"
            calibration_failures=$((calibration_failures + 1))
        fi
    done

    if [[ ${calibration_failures} -eq ${num_runs} ]]; then
        log "  ERROR: All calibration runs failed for ${model_name}"
        log "         Skipping preset generation (no valid calibration data)"
        return 1
    fi

    # Generate calibrated preset
    python3 << CALIBRATION_SCRIPT
import json
import math
import os
import re
import statistics
from pathlib import Path
from collections import defaultdict

try:
    import yaml
    YAML_AVAILABLE = True
except Exception:
    YAML_AVAILABLE = False

output_dir = Path("${output_dir}")
preset_output_dir = Path("${preset_output_dir}")
model_name = "${model_name}"
model_path = "${model_path}"
tier = "${INVARLOCK_TIER}".strip().lower()
dataset_provider = "${INVARLOCK_DATASET}"
seq_len = int("${effective_seq_len}")
stride = int("${effective_stride}")
preview_n = int("${effective_preview_n}")
final_n = int("${effective_final_n}")

guards_order = None
assurance_cfg = None
if YAML_AVAILABLE:
    cfg_path = None
    for candidate in sorted(output_dir.glob("run_*/calibration_config.yaml")):
        cfg_path = candidate
        break
    if cfg_path is not None:
        try:
            cfg = yaml.safe_load(cfg_path.read_text())
            if isinstance(cfg, dict):
                guards_block = cfg.get("guards") or {}
                if isinstance(guards_block, dict):
                    order = guards_block.get("order")
                    if isinstance(order, list) and order:
                        guards_order = [str(item) for item in order]
                ab = cfg.get("assurance")
                if isinstance(ab, dict) and ab:
                    assurance_cfg = ab
        except Exception:
            guards_order = None

if guards_order is None:
    guards_order = ["invariants", "spectral", "rmt", "variance", "invariants"]

enabled_guards = set(guards_order)

def _safe_float(value):
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None

def _quantile(values, q):
    if not values:
        return None
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * q
    lower = int(math.floor(pos))
    upper = int(math.ceil(pos))
    if lower == upper:
        return values[lower]
    frac = pos - lower
    return values[lower] + (values[upper] - values[lower]) * frac

def _merge_record(cert, report):
    rec = {}
    if isinstance(cert, dict):
        rec = json.loads(json.dumps(cert))
    if not isinstance(report, dict):
        return rec or None

    # Primary metric from report when cert is missing it.
    metrics = report.get("metrics", {}) or {}
    pm = metrics.get("primary_metric", {}) or {}
    if not pm and "ppl_final" in metrics:
        pm = {
            "final": metrics.get("ppl_final"),
            "preview": metrics.get("ppl_preview"),
        }
        try:
            pm["ratio_vs_baseline"] = float(pm["final"]) / max(float(pm["preview"]), 1e-10)
        except Exception:
            pass
    if pm and not rec.get("primary_metric"):
        rec["primary_metric"] = pm

    guards = report.get("guards", []) or []
    for guard in guards:
        if not isinstance(guard, dict):
            continue
        name = str(guard.get("name", "")).lower()
        gmetrics = guard.get("metrics", {}) or {}
        gpolicy = guard.get("policy", {}) or {}

        if name == "spectral":
            spec = rec.get("spectral", {}) if isinstance(rec.get("spectral"), dict) else {}
            if gmetrics.get("family_z_quantiles"):
                spec.setdefault("family_z_quantiles", gmetrics.get("family_z_quantiles"))
            if gmetrics.get("family_z_summary"):
                spec.setdefault("family_z_summary", gmetrics.get("family_z_summary"))
            if gmetrics.get("family_caps"):
                spec.setdefault("family_caps", gmetrics.get("family_caps"))
            if gmetrics.get("sigma_quantile") is not None:
                spec.setdefault("sigma_quantile", gmetrics.get("sigma_quantile"))
            if gmetrics.get("deadband") is not None:
                spec.setdefault("deadband", gmetrics.get("deadband"))
            if gmetrics.get("max_caps") is not None:
                spec.setdefault("max_caps", gmetrics.get("max_caps"))
            if gmetrics.get("families"):
                spec.setdefault("families", gmetrics.get("families"))
            if gmetrics.get("family_stats"):
                spec.setdefault("families", gmetrics.get("family_stats"))
            z_scores = guard.get("final_z_scores") or gmetrics.get("final_z_scores")
            if isinstance(z_scores, dict):
                spec["final_z_scores"] = z_scores
            fam_map = guard.get("module_family_map") or gmetrics.get("module_family_map")
            if isinstance(fam_map, dict):
                spec["module_family_map"] = fam_map
            if gpolicy and not spec.get("policy"):
                spec["policy"] = gpolicy
            rec["spectral"] = spec

        elif name == "rmt":
            rmt = rec.get("rmt", {}) if isinstance(rec.get("rmt"), dict) else {}
            for key in ("outliers_per_family", "baseline_outliers_per_family", "families"):
                val = gmetrics.get(key)
                if isinstance(val, dict) and val:
                    rmt.setdefault(key, val)
            epsilon_by_family = gmetrics.get("epsilon_by_family")
            if epsilon_by_family:
                rmt.setdefault("epsilon_by_family", epsilon_by_family)
            else:
                epsilon = gmetrics.get("epsilon")
                if epsilon is not None:
                    if isinstance(epsilon, dict):
                        rmt.setdefault("epsilon_by_family", epsilon)
                    else:
                        rmt.setdefault("epsilon_default", epsilon)
            if gmetrics.get("epsilon_default") is not None:
                rmt.setdefault("epsilon_default", gmetrics.get("epsilon_default"))
            if gmetrics.get("margin_used") is not None:
                rmt.setdefault("margin", gmetrics.get("margin_used"))
            if gmetrics.get("deadband_used") is not None:
                rmt.setdefault("deadband", gmetrics.get("deadband_used"))
            if gpolicy and not rmt.get("policy"):
                rmt["policy"] = gpolicy
            rec["rmt"] = rmt

        elif name == "variance":
            var = rec.get("variance", {}) if isinstance(rec.get("variance"), dict) else {}
            if gmetrics.get("predictive_gate") is not None:
                var.setdefault("predictive_gate", gmetrics.get("predictive_gate"))
            if gmetrics.get("ab_windows_used") is not None:
                var.setdefault("ab_windows_used", gmetrics.get("ab_windows_used"))
            if gmetrics.get("deadband") is not None:
                var.setdefault("deadband", gmetrics.get("deadband"))
            if gmetrics.get("min_gain") is not None:
                var.setdefault("min_gain", gmetrics.get("min_gain"))
            if gmetrics.get("min_effect_lognll") is not None:
                var.setdefault("min_effect_lognll", gmetrics.get("min_effect_lognll"))
            if gmetrics.get("calibration") is not None:
                var.setdefault("calibration", gmetrics.get("calibration"))
            if gmetrics.get("calibration_stats") is not None:
                var.setdefault("calibration_stats", gmetrics.get("calibration_stats"))
            if gpolicy and not var.get("policy"):
                var["policy"] = gpolicy
            rec["variance"] = var

    return rec or None

def load_records():
    records = []
    for run_dir in sorted(output_dir.glob("run_*")):
        cert = None
        report = None
        cert_path = run_dir / "evaluation.cert.json"
        if cert_path.exists():
            try:
                cert = json.loads(cert_path.read_text())
            except Exception:
                cert = None
        report_path = run_dir / "baseline_report.json"
        if not report_path.exists():
            report_files = list(run_dir.glob("**/report*.json"))
            if report_files:
                report_path = report_files[0]
        if report_path.exists():
            try:
                report = json.loads(report_path.read_text())
            except Exception:
                report = None

        record = _merge_record(cert, report)
        if record:
            records.append(record)
    return records

records = load_records()
if len(records) == 0:
    print("ERROR: No calibration records found - cannot create valid preset")
    import sys
    sys.exit(1)
if len(records) < 2:
    print(f"WARNING: Only {len(records)} calibration record(s) found (expected >= 2)")

def calibrate_drift(recs):
    try:
        ratios = []
        for rec in recs:
            pm = rec.get("primary_metric", {}) or {}
            ratio = pm.get("ratio_vs_baseline") or pm.get("drift")
            if ratio is None:
                preview = pm.get("preview")
                final = pm.get("final")
                if preview is not None and final is not None:
                    try:
                        ratio = float(final) / max(float(preview), 1e-10)
                    except Exception:
                        ratio = None
            if ratio is not None:
                try:
                    ratios.append(float(ratio))
                except Exception:
                    pass

        ratios = [r for r in ratios if math.isfinite(r)]
        if len(ratios) < 2:
            base = ratios[0] if ratios else 1.0
            return {
                "mean": float(base),
                "std": 0.0,
                "min": float(base),
                "max": float(base),
                "suggested_band": [0.95, 1.05],
                "band_compatible": True,
            }

        try:
            mean = sum(ratios) / len(ratios)
        except Exception:
            mean = 1.0
        try:
            var = sum((r - mean) ** 2 for r in ratios) / max(len(ratios), 1)
            std = math.sqrt(var) if math.isfinite(var) else 0.0
        except Exception:
            std = 0.0
        margin = max(2 * std, 0.05)
        band = [round(mean - margin, 3), round(mean + margin, 3)]
        return {
            "mean": round(mean, 4),
            "std": round(std, 4),
            "min": round(min(ratios), 4),
            "max": round(max(ratios), 4),
            "suggested_band": band,
            "band_compatible": 0.95 <= mean <= 1.05,
        }
    except Exception as e:
        print(f"ERROR: failed to compute drift stats: {e}")
        return {
            "mean": 1.0,
            "std": 0.0,
            "min": 1.0,
            "max": 1.0,
            "suggested_band": [0.95, 1.05],
            "band_compatible": True,
        }

def _spectral_margin(tier_name):
    return 0.10 if tier_name == "conservative" else 0.05

def _default_max_caps(tier_name):
    if tier_name == "conservative":
        return 3
    if tier_name == "aggressive":
        return 8
    return 5

def _allocate_budget(counts, budget):
    if not counts or budget <= 0:
        return {fam: 0 for fam in counts}
    total = sum(counts.values())
    if total <= 0:
        return {fam: 0 for fam in counts}
    raw = {fam: budget * count / total for fam, count in counts.items()}
    alloc = {fam: int(round(val)) for fam, val in raw.items()}
    diff = budget - sum(alloc.values())
    if diff > 0:
        for fam in sorted(raw, key=raw.get, reverse=True):
            if diff == 0:
                break
            alloc[fam] += 1
            diff -= 1
    elif diff < 0:
        for fam in sorted(raw, key=raw.get):
            if diff == 0:
                break
            if alloc.get(fam, 0) > 0:
                alloc[fam] -= 1
                diff += 1
    return alloc

def calibrate_spectral(recs):
    per_run_caps = defaultdict(list)
    q99_values = defaultdict(list)
    max_values = defaultdict(list)
    existing_caps = {}
    sigma_quantile = None
    deadband = None
    max_caps = None

    for rec in recs:
        spec = rec.get("spectral", {}) or {}
        if not isinstance(spec, dict):
            continue
        policy = spec.get("policy", {}) if isinstance(spec.get("policy"), dict) else {}

        if sigma_quantile is None:
            sq = (
                policy.get("sigma_quantile")
                or policy.get("contraction")
                or policy.get("kappa")
                or spec.get("sigma_quantile")
                or (spec.get("summary") or {}).get("sigma_quantile")
            )
            sq = _safe_float(sq)
            if sq is not None:
                sigma_quantile = sq

        if deadband is None:
            db = policy.get("deadband") or spec.get("deadband") or (spec.get("summary") or {}).get("deadband")
            db = _safe_float(db)
            if db is not None:
                deadband = db

        if max_caps is None:
            mc = policy.get("max_caps") or spec.get("max_caps") or (spec.get("summary") or {}).get("max_caps")
            try:
                if mc is not None:
                    max_caps = int(mc)
            except Exception:
                pass

        fam_caps = spec.get("family_caps", {})
        if not fam_caps and isinstance(policy.get("family_caps"), dict):
            fam_caps = policy.get("family_caps", {})
        if isinstance(fam_caps, dict):
            for fam, cap in fam_caps.items():
                try:
                    if isinstance(cap, dict):
                        cap = cap.get("kappa")
                    existing_caps[str(fam)] = float(cap)
                except Exception:
                    pass

        z_map = spec.get("final_z_scores")
        fam_map = spec.get("module_family_map")
        if isinstance(z_map, dict) and isinstance(fam_map, dict):
            z_by_family = defaultdict(list)
            for module, z in z_map.items():
                fam = fam_map.get(module)
                if fam is None:
                    continue
                z_val = _safe_float(z)
                if z_val is None:
                    continue
                z_by_family[str(fam)].append(abs(z_val))
            if z_by_family:
                counts = {fam: len(vals) for fam, vals in z_by_family.items() if vals}
                budget = (
                    max_caps
                    if isinstance(max_caps, int) and max_caps >= 0
                    else _default_max_caps(tier)
                )
                alloc = _allocate_budget(counts, budget)
                for fam, values in z_by_family.items():
                    if not values:
                        continue
                    values_sorted = sorted(values, reverse=True)
                    idx = max(0, min(alloc.get(fam, 1) - 1, len(values_sorted) - 1))
                    per_run_caps[fam].append(values_sorted[idx])

        fq = spec.get("family_z_quantiles", {})
        if not fq and isinstance(spec.get("family_z_summary"), dict):
            fq = spec.get("family_z_summary", {})
        if isinstance(fq, dict):
            for fam, stats in fq.items():
                if not isinstance(stats, dict):
                    continue
                val_q99 = _safe_float(stats.get("q99"))
                val_max = _safe_float(stats.get("max"))
                if val_q99 is not None:
                    q99_values[str(fam)].append(val_q99)
                if val_max is not None:
                    max_values[str(fam)].append(val_max)

    summary = {
        "families_seen": sorted(set(per_run_caps) | set(q99_values) | set(existing_caps)),
        "sigma_quantile": sigma_quantile,
        "deadband": deadband,
        "max_caps": max_caps,
    }

    proposed_caps = {}
    margin = _spectral_margin(tier)
    if per_run_caps:
        for fam, candidates in per_run_caps.items():
            if not candidates:
                continue
            base = max(candidates)
            if fam in existing_caps:
                base = max(base, existing_caps[fam])
            proposed_caps[fam] = {"kappa": round(base + margin, 3)}
        for fam in sorted(set(q99_values) | set(max_values)):
            if fam in proposed_caps:
                continue
            observed = q99_values.get(fam, []) + max_values.get(fam, [])
            if not observed:
                continue
            base = max(observed)
            if fam in existing_caps:
                base = max(base, existing_caps[fam])
            proposed_caps[fam] = {"kappa": round(base + margin, 3)}
    elif q99_values or max_values:
        for fam in sorted(set(q99_values) | set(max_values)):
            observed = q99_values.get(fam, []) + max_values.get(fam, [])
            if not observed:
                continue
            base = max(observed)
            if fam in existing_caps:
                base = max(base, existing_caps[fam])
            proposed_caps[fam] = {"kappa": round(base + margin, 3)}
    else:
        for fam, kappa in existing_caps.items():
            proposed_caps[fam] = {"kappa": kappa}

    return summary, proposed_caps

def _rmt_quantile_for_tier(tier_name):
    if tier_name == "conservative":
        return 0.95
    if tier_name == "aggressive":
        return 0.99
    return 0.97

def calibrate_rmt(recs):
    deltas_by_family = defaultdict(list)
    existing_eps = {}
    margin = None
    deadband = None

    for rec in recs:
        rmt = rec.get("rmt", {}) or {}
        if not isinstance(rmt, dict):
            continue
        policy = rmt.get("policy", {}) if isinstance(rmt.get("policy"), dict) else {}

        if margin is None:
            margin = _safe_float(policy.get("margin") or rmt.get("margin") or (rmt.get("summary") or {}).get("margin"))
        if deadband is None:
            deadband = _safe_float(policy.get("deadband") or rmt.get("deadband") or (rmt.get("summary") or {}).get("deadband"))

        eps = (
            rmt.get("epsilon_by_family")
            or rmt.get("epsilon")
            or policy.get("epsilon_by_family")
            or policy.get("epsilon")
        )
        if isinstance(eps, dict):
            for fam, val in eps.items():
                try:
                    existing_eps[str(fam)] = float(val)
                except Exception:
                    pass
        elif isinstance(eps, (int, float)):
            existing_eps["_default"] = float(eps)

        record_has_counts = False
        families = rmt.get("families", {})
        if isinstance(families, dict) and families:
            record_has_counts = True
            for fam, stats in families.items():
                if not isinstance(stats, dict):
                    continue
                bare = stats.get("bare")
                guarded = stats.get("guarded")
                bare_f = _safe_float(bare)
                guarded_f = _safe_float(guarded)
                if bare_f and bare_f > 0:
                    deltas_by_family[str(fam)].append((guarded_f / bare_f) - 1.0)

        outliers = rmt.get("outliers_per_family", {})
        baseline_outliers = rmt.get("baseline_outliers_per_family", {})
        if isinstance(outliers, dict) and isinstance(baseline_outliers, dict) and outliers:
            record_has_counts = True
            for fam in set(outliers) | set(baseline_outliers):
                bare_f = _safe_float(baseline_outliers.get(fam))
                guarded_f = _safe_float(outliers.get(fam))
                if bare_f and bare_f > 0:
                    deltas_by_family[str(fam)].append((guarded_f / bare_f) - 1.0)

        if not record_has_counts:
            for source in ("outliers_by_family", "family_stats"):
                stats_map = rmt.get(source, {})
                if not isinstance(stats_map, dict):
                    continue
                for fam, stats in stats_map.items():
                    if not isinstance(stats, dict):
                        continue
                    for key in ("outlier_fraction", "outlier_rate", "fraction", "rate"):
                        val = _safe_float(stats.get(key))
                        if val is not None:
                            deltas_by_family[str(fam)].append(val)
                            break

    summary = {"families_seen": sorted(deltas_by_family.keys()), "margin": margin, "deadband": deadband}
    quantile_q = _rmt_quantile_for_tier(tier)
    proposed_eps = {}
    if deltas_by_family:
        for fam, deltas in deltas_by_family.items():
            qv = _quantile(deltas, quantile_q)
            if qv is None:
                continue
            qv = max(float(qv), 0.0)
            proposed_eps[fam] = round(qv, 3)

    if not proposed_eps:
        if existing_eps:
            if set(existing_eps.keys()) == {"_default"}:
                default_eps = existing_eps["_default"]
                return summary, {"ffn": default_eps, "attn": default_eps, "embed": default_eps, "other": default_eps}
            return summary, existing_eps
        defaults = {
            "balanced": {"ffn": 0.10, "attn": 0.08, "embed": 0.12, "other": 0.12},
            "conservative": {"ffn": 0.06, "attn": 0.05, "embed": 0.07, "other": 0.07},
        }
        return summary, defaults.get(tier, defaults["balanced"])

    for fam, eps_val in existing_eps.items():
        if fam not in proposed_eps and fam != "_default":
            proposed_eps[fam] = eps_val

    return summary, proposed_eps

def calibrate_variance(recs):
    deadband = None
    min_gain = None
    policy_min_effect = None
    min_effect_samples = []
    variance_changes = []

    for rec in recs:
        var = rec.get("variance", {}) or {}
        if not isinstance(var, dict):
            continue
        policy = var.get("policy", {}) if isinstance(var.get("policy"), dict) else {}

        if deadband is None:
            deadband = _safe_float(policy.get("deadband") or var.get("deadband"))
        if min_gain is None:
            min_gain = _safe_float(policy.get("min_gain") or policy.get("min_rel_gain") or var.get("min_gain"))
        if policy_min_effect is None:
            policy_min_effect = _safe_float(policy.get("min_effect_lognll") or var.get("min_effect_lognll"))

        predictive = var.get("predictive_gate", {}) or {}
        delta_ci = predictive.get("delta_ci")
        if isinstance(delta_ci, (list, tuple)) and len(delta_ci) == 2:
            lo = _safe_float(delta_ci[0])
            hi = _safe_float(delta_ci[1])
            if lo is not None and hi is not None:
                width = abs(hi - lo) / 2.0
                if width > 0:
                    min_effect_samples.append(width)

        calib = var.get("calibration") or var.get("calibration_stats") or {}
        if isinstance(calib, dict):
            vchange = calib.get("variance_change") or calib.get("delta") or calib.get("max_delta")
            vchange = _safe_float(vchange)
            if vchange is not None:
                variance_changes.append(abs(vchange))

    result = {}
    if deadband is None and variance_changes:
        result["deadband"] = round(max(variance_changes) * 1.1 + 0.01, 3)
    elif deadband is not None:
        result["deadband"] = deadband

    if min_effect_samples:
        proposed = _quantile(min_effect_samples, 0.95)
        if proposed is not None:
            result["min_effect_lognll"] = max(round(proposed, 4), 0.0009)
    elif policy_min_effect is not None:
        result["min_effect_lognll"] = policy_min_effect

    if min_gain is not None:
        result["min_gain"] = min_gain

    return result

drift_stats = calibrate_drift(records)
spectral_summary, spectral_caps = calibrate_spectral(records)
rmt_summary, rmt_epsilon = calibrate_rmt(records)
variance_config = calibrate_variance(records)

spectral_max_caps_override = (os.environ.get("PACK_SPECTRAL_MAX_CAPS") or "").strip()
try:
    spectral_max_caps_override = int(spectral_max_caps_override) if spectral_max_caps_override else None
except Exception:
    spectral_max_caps_override = None
if spectral_max_caps_override is None and tier == "balanced":
    spectral_max_caps_override = 15
if spectral_max_caps_override is not None:
    try:
        current_max_caps = spectral_summary.get("max_caps")
    except Exception:
        current_max_caps = None
    try:
        current_int = int(current_max_caps) if current_max_caps is not None else None
    except Exception:
        current_int = None
    if current_int is None or current_int < spectral_max_caps_override:
        spectral_summary["max_caps"] = spectral_max_caps_override

gpu_count = (os.environ.get("PACK_GPU_COUNT") or os.environ.get("NUM_GPUS") or "").strip() or "unknown"
gpu_mem = (os.environ.get("PACK_GPU_MEM_GB") or os.environ.get("GPU_MEMORY_GB") or "").strip()
gpu_name = (os.environ.get("PACK_GPU_NAME") or "GPU").strip() or "GPU"
gpu_mem_label = f"{gpu_mem}GB" if gpu_mem else "unknown"
tag_name = re.sub(r"[^A-Za-z0-9]+", "_", gpu_name).strip("_") or "GPU"
platform_tag = f"{tag_name}_{gpu_mem_label}_x{gpu_count}"

preset = {
    "_calibration_meta": {
        "model_name": model_name,
        "tier": tier,
        "platform": platform_tag,
        "drift_mean": drift_stats.get("mean"),
        "drift_std": drift_stats.get("std"),
        "drift_band_compatible": drift_stats.get("band_compatible"),
        "suggested_drift_band": drift_stats.get("suggested_band"),
    },
    "model": {"id": model_path},
    "dataset": {
        "provider": dataset_provider,
        "split": "validation",
        "seq_len": seq_len,
        "stride": stride,
        "preview_n": preview_n,
        "final_n": final_n,
        "seed": 42,
    },
    "guards": {"order": guards_order},
}

if isinstance(assurance_cfg, dict) and assurance_cfg:
    preset["assurance"] = assurance_cfg

spectral = {}
if spectral_caps:
    spectral["family_caps"] = spectral_caps
if spectral_summary.get("sigma_quantile") is not None:
    spectral["sigma_quantile"] = spectral_summary["sigma_quantile"]
if spectral_summary.get("deadband") is not None:
    spectral["deadband"] = spectral_summary["deadband"]
if spectral_summary.get("max_caps") is not None:
    spectral["max_caps"] = spectral_summary["max_caps"]
if "spectral" in enabled_guards and spectral:
    preset["guards"]["spectral"] = spectral

rmt = {}
if rmt_epsilon:
    rmt["epsilon_by_family"] = rmt_epsilon
if rmt_summary.get("margin") is not None:
    rmt["margin"] = rmt_summary["margin"]
if rmt_summary.get("deadband") is not None:
    rmt["deadband"] = rmt_summary["deadband"]
if "rmt" in enabled_guards and rmt:
    preset["guards"]["rmt"] = rmt

if "variance" in enabled_guards and variance_config:
    preset["guards"]["variance"] = variance_config

stats_path = output_dir / "calibration_stats.json"
with open(stats_path, "w") as f:
    json.dump(
        {
            "guards_order": guards_order,
            "assurance": assurance_cfg,
            "drift": drift_stats,
            "spectral": {**spectral_summary, "family_caps": spectral_caps},
            "rmt": {**rmt_summary, "epsilon_by_family": rmt_epsilon},
            "variance": variance_config,
        },
        f,
        indent=2,
    )

preset_path = preset_output_dir / f"calibrated_preset_{model_name.replace('/', '_')}.yaml"
if YAML_AVAILABLE:
    with open(preset_path, "w") as f:
        yaml.safe_dump(preset, f, sort_keys=False)
else:
    preset_path = preset_path.with_suffix(".json")
    with open(preset_path, "w") as f:
        json.dump(preset, f, indent=2)

print(f"Saved: {stats_path}")
print(f"Saved: {preset_path}")
CALIBRATION_SCRIPT
}

# ============ CERTIFY WITH PROOF PACK SETTINGS ============
run_invarlock_certify() {
    local subject_path="$1"
    local baseline_path="$2"
    local output_dir="$3"
    local run_name="$4"
    local preset_dir="$5"
    local model_name="$6"
    local gpu_id="${7:-0}"

    local run_dir="${output_dir}/${run_name}"
    local cert_dir="${run_dir}/cert"
    mkdir -p "${run_dir}" "${cert_dir}"

    local calibrated_preset=""
    for ext in yaml json; do
        local preset_path="${preset_dir}/calibrated_preset_${model_name}.${ext}"
        if [[ -f "${preset_path}" ]]; then
            calibrated_preset="${preset_path}"
            break
        fi
    done

    local cmd_args=(
        "invarlock" "certify"
        "--source" "${baseline_path}"
        "--edited" "${subject_path}"
        "--profile" "ci"
        "--tier" "${INVARLOCK_TIER}"
        "--out" "${run_dir}"
        "--cert-out" "${cert_dir}"
    )

    if [[ -n "${calibrated_preset}" && -f "${calibrated_preset}" ]]; then
        cmd_args+=("--preset" "${calibrated_preset}")
    fi

    local cuda_devices="${CUDA_VISIBLE_DEVICES:-${gpu_id}}"

    local exit_code=0
    # For large models, skip overhead check to avoid OOM (task-local via env)
    local model_size
    model_size=$(estimate_model_params "${baseline_path}")
    if [[ "${model_size}" == "70" || "${model_size}" == "72" || "${model_size}" == "moe" ]]; then
        INVARLOCK_SKIP_OVERHEAD_CHECK=1 \
        CUDA_VISIBLE_DEVICES="${cuda_devices}" "${cmd_args[@]}" \
            >> "${OUTPUT_DIR}/logs/gpu_${gpu_id}.log" 2>&1 || exit_code=$?
    else
        CUDA_VISIBLE_DEVICES="${cuda_devices}" "${cmd_args[@]}" \
            >> "${OUTPUT_DIR}/logs/gpu_${gpu_id}.log" 2>&1 || exit_code=$?
    fi

    # Copy certificate to standard location (only the canonical cert)
    local cert_file="${cert_dir}/evaluation.cert.json"
    if [[ -f "${cert_file}" ]]; then
        cp "${cert_file}" "${run_dir}/evaluation.cert.json" 2>/dev/null || true
    else
        local alt_cert
        alt_cert=$(find "${cert_dir}" -name "evaluation.cert.json" -type f 2>/dev/null | head -1)
        if [[ -n "${alt_cert}" && -f "${alt_cert}" ]]; then
            cp "${alt_cert}" "${run_dir}/evaluation.cert.json" 2>/dev/null || true
        fi
    fi

    return ${exit_code}
}
export -f run_invarlock_certify
