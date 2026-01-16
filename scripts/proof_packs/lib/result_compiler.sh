#!/usr/bin/env bash
# result_compiler.sh - Analysis and verdict compilation for proof packs.

# ============ COMPILE RESULTS ============
compile_results() {
    log_section "COMPILING RESULTS"

	    python3 <<- EOF
import json
import csv
import math
from pathlib import Path
from collections import defaultdict

output_dir = Path("${OUTPUT_DIR}")
analysis_dir = output_dir / "analysis"
analysis_dir.mkdir(exist_ok=True)

skip_dirs = {
    "logs",
    "analysis",
    "reports",
    "presets",
    "models",
    "queue",
    "workers",
    "state",
    "evals",
    "certificates",
}

def _is_model_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    if path.name in skip_dirs:
        return False
    if not (path / ".baseline_path").exists():
        return False
    return True


def _pick_metric(task_results: dict):
    for key in (
        "acc_norm,none",
        "acc,none",
        "exact_match,none",
        "acc_norm",
        "acc",
        "exact_match",
    ):
        if key in task_results and isinstance(task_results[key], (int, float)):
            return key, float(task_results[key])
    for key, value in task_results.items():
        if "stderr" in key:
            continue
        if isinstance(value, (int, float)):
            return key, float(value)
    return None, None

# Collect eval results
eval_rows = []
for model_dir in output_dir.iterdir():
    if not _is_model_dir(model_dir):
        continue

    evals_dir = model_dir / "evals"
    if not evals_dir.exists():
        continue

    benchmarks = {"mmlu", "hellaswag", "arc", "winogrande"}
    for results_file in evals_dir.glob("*_results.json"):
        stem = results_file.stem
        base = stem[:-len("_results")] if stem.endswith("_results") else stem
        parts = base.split("_")
        if parts and parts[-1] in benchmarks:
            # Split-eval output: {edit}_{benchmark}_results.json
            edit_type = "_".join(parts[:-1])
        else:
            edit_type = base
        try:
            data = json.loads(results_file.read_text())
            for task, task_results in data.get('results', {}).items():
                if not isinstance(task_results, dict):
                    continue
                metric_key, metric_val = _pick_metric(task_results)
                if metric_key is None:
                    continue
                metric_name = metric_key
                metric_type = ""
                if isinstance(metric_key, str) and "," in metric_key:
                    metric_name, metric_type = metric_key.split(",", 1)
                    metric_name = metric_name.strip()
                    metric_type = metric_type.strip()
                eval_rows.append({
                    'model': model_dir.name,
                    'edit_type': edit_type,
                    'task': task,
                    'metric': metric_name,
                    'metric_type': metric_type,
                    'value': metric_val,
                })
        except Exception as e:
            print(f"Error processing {results_file}: {e}")

if eval_rows:
    with open(analysis_dir / "eval_results.csv", 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=eval_rows[0].keys())
        writer.writeheader()
        writer.writerows(eval_rows)
    jsonl_path = analysis_dir / "eval_results.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for row in eval_rows:
            f.write(json.dumps(row, sort_keys=True) + "\\n")
    print(f"Wrote {len(eval_rows)} eval rows")

# Collect InvarLock results
invar_rows = []
for model_dir in output_dir.iterdir():
    if not _is_model_dir(model_dir):
        continue

    certs_dir = model_dir / "certificates"
    if not certs_dir.exists():
        continue

    for cert_file in certs_dir.rglob("evaluation.cert.json"):
        try:
            cert = json.loads(cert_file.read_text())
            rel_path = cert_file.relative_to(certs_dir)
            parts = list(rel_path.parts)

            v = cert.get('validation', {}) or {}
            def as_bool(val):
                if val is None:
                    return False
                if isinstance(val, bool):
                    return val
                if isinstance(val, str):
                    return val.strip().lower() in ('true', '1', 'yes', 'on')
                return bool(val)

            invariants_ok = as_bool(v.get('invariants_pass', False))
            pm_ok = as_bool(v.get('primary_metric_acceptable', False))
            spectral_ok = as_bool(v.get('spectral_stable', False))
            rmt_ok = as_bool(v.get('rmt_stable', False))
            drift_ok = as_bool(v.get('preview_final_drift_acceptable', True))
            hyst_applied = as_bool(v.get('hysteresis_applied', False))

            guard_overhead = cert.get('guard_overhead') or {}
            guard_evaluated = bool(guard_overhead.get('evaluated')) if isinstance(guard_overhead, dict) else False
            overhead_ok = as_bool(v.get('guard_overhead_acceptable', True))

            pm_block = cert.get('primary_metric') or {}
            pm_degraded = as_bool(pm_block.get('degraded')) or as_bool(pm_block.get('invalid'))
            pm_degraded_reason = pm_block.get('degraded_reason')

            # Backwards-compatible "all_pass" used by existing analysis logic.
            all_pass = all([invariants_ok, pm_ok, spectral_ok, rmt_ok]) and not pm_degraded

            # Canonical overall pass aligned with InvarLock console validation block.
            overall_pass = all([invariants_ok, pm_ok, spectral_ok, rmt_ok, drift_ok]) and not pm_degraded
            if guard_evaluated:
                overall_pass = overall_pass and overhead_ok

            conf = cert.get('confidence') or {}
            conf_label = conf.get('label') if isinstance(conf, dict) else None
            conf_label = str(conf_label).strip() if conf_label is not None else ''
            conf_label = conf_label if conf_label else 'Unknown'

            pm = pm_block
            pm_ratio = pm.get('ratio_vs_baseline') if isinstance(pm, dict) else None
            pm_ci_lo = None
            pm_ci_hi = None
            try:
                dci = pm.get('display_ci') if isinstance(pm, dict) else None
                if isinstance(dci, (list, tuple)) and len(dci) == 2:
                    pm_ci_lo = float(dci[0])
                    pm_ci_hi = float(dci[1])
                    if not (math.isfinite(pm_ci_lo) and math.isfinite(pm_ci_hi)):
                        pm_ci_lo = None
                        pm_ci_hi = None
            except Exception:
                pm_ci_lo = None
                pm_ci_hi = None

            # Estimate the effective PM threshold used by the tier policy.
            tier = ''
            try:
                pd_try = cert.get('policy_digest') or {}
                auto_try = cert.get('auto') or {}
                tier = str(pd_try.get('tier_policy_name') or auto_try.get('tier') or '').strip().lower()
            except Exception:
                tier = ''

            pm_threshold = None
            try:
                pol = cert.get('resolved_policy') or {}
                metrics_pol = pol.get('metrics', {}) if isinstance(pol, dict) else {}
                pm_pol = metrics_pol.get('pm_ratio', {}) if isinstance(metrics_pol, dict) else {}
                base = pm_pol.get('ratio_limit_base')
                hyst = pm_pol.get('hysteresis_ratio', 0.0)
                if base is not None:
                    pm_threshold = float(base) + float(hyst or 0.0)
            except Exception:
                pm_threshold = None
            if pm_threshold is None:
                tier_thresholds = {'conservative': 1.05, 'balanced': 1.10, 'aggressive': 1.20}
                base = tier_thresholds.get(tier, 1.10)
                pm_threshold = float(base) + 0.002

            # "Clear" PM failure if CI lower bound is above threshold, or ratio is far above.
            pm_clear_fail = False
            pm_far_fail = False
            pm_far_margin = 0.03  # absolute ratio margin above threshold
            try:
                if pm_ci_lo is not None and pm_threshold is not None:
                    pm_clear_fail = float(pm_ci_lo) > float(pm_threshold)
            except Exception:
                pm_clear_fail = False
            try:
                if isinstance(pm_ratio, (int, float)) and math.isfinite(float(pm_ratio)):
                    pm_far_fail = float(pm_ratio) > (float(pm_threshold) + float(pm_far_margin))
            except Exception:
                pm_far_fail = False

            # Derive degradation if fields are missing but PM is non-finite
            try:
                prev_val = pm.get('preview')
                fin_val = pm.get('final')
                ratio_val = pm_ratio
                def _nonfinite(v):
                    try:
                        return not (isinstance(v, (int, float)) and math.isfinite(float(v)))
                    except Exception:
                        return True
                if not pm_degraded and (_nonfinite(prev_val) or _nonfinite(fin_val) or _nonfinite(ratio_val)):
                    pm_degraded = True
                    pm_degraded_reason = pm_degraded_reason or 'non_finite_pm'
            except Exception:
                pass

            # Triage layer (PASS/REVIEW/FAIL) for shadow-mode style workflows.
            triage_reasons = []
            if pm_degraded:
                triage_reasons.append('primary_metric_degraded')
            if not invariants_ok:
                triage_reasons.append('invariants_fail')
            if not spectral_ok:
                triage_reasons.append('spectral_fail')
            if not rmt_ok:
                triage_reasons.append('rmt_fail')
            if not pm_ok:
                triage_reasons.append('primary_metric_fail')
            if not drift_ok:
                triage_reasons.append('drift_fail')
            if guard_evaluated and not overhead_ok:
                triage_reasons.append('overhead_fail')
            if hyst_applied:
                triage_reasons.append('hysteresis_applied')
            if conf_label != 'High':
                triage_reasons.append(f'confidence_{conf_label.lower()}')

            triage = 'REVIEW'
            if pm_degraded or (not invariants_ok) or (not spectral_ok) or (not rmt_ok):
                triage = 'FAIL'
            elif (not pm_ok) and (pm_clear_fail or pm_far_fail):
                triage = 'FAIL'
                triage_reasons.append('primary_metric_clear' if pm_clear_fail else 'primary_metric_far')
            elif overall_pass and conf_label == 'High' and not hyst_applied:
                triage = 'PASS'
                triage_reasons = []

            triage_reason = 'strict_pass' if triage == 'PASS' else ('|'.join(triage_reasons) if triage_reasons else 'unspecified')

            pd = cert.get('policy_digest') or {}
            meta = cert.get('meta') or {}
            det = meta.get('determinism') or {}

            invar_rows.append({
                'model': model_dir.name,
                'experiment': parts[0] if parts else 'unknown',
                'run': parts[1] if len(parts) > 1 else '',
                'edit_type': parts[0] if parts else 'unknown',
                'pm_ratio': pm_ratio,
                'pm_ci_low': pm_ci_lo,
                'pm_ci_high': pm_ci_hi,
                'pm_threshold': pm_threshold,
                'pm_acceptable': v.get('primary_metric_acceptable'),
                'pm_degraded': pm_degraded,
                'pm_degraded_reason': pm_degraded_reason,
                'preview_final_drift_acceptable': v.get('preview_final_drift_acceptable'),
                'invariants_pass': v.get('invariants_pass'),
                'spectral_stable': v.get('spectral_stable'),
                'rmt_stable': v.get('rmt_stable'),
                'all_pass': all_pass,
                'overall_pass': overall_pass,
                'hysteresis_applied': v.get('hysteresis_applied'),
                'guard_overhead_acceptable': v.get('guard_overhead_acceptable'),
                'confidence_label': conf_label,
                'triage': triage,
                'triage_reason': triage_reason,
                'policy_digest_hash': pd.get('thresholds_hash'),
                'policy_digest_changed': pd.get('changed'),
                'determinism_level': det.get('level'),
                'determinism_profile': det.get('profile'),
                'determinism_requested': det.get('requested'),
            })
        except Exception as e:
            print(f"Error processing {cert_file}: {e}")

if invar_rows:
    with open(analysis_dir / "invarlock_results.csv", 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=invar_rows[0].keys())
        writer.writeheader()
        writer.writerows(invar_rows)
    print(f"Wrote {len(invar_rows)} InvarLock rows")

# Guard sensitivity matrix
guard_matrix = defaultdict(lambda: defaultdict(list))
for row in invar_rows:
    edit_type = row.get('edit_type', 'unknown')
    for guard in ['spectral_stable', 'rmt_stable', 'invariants_pass']:
        val = row.get(guard)
        if val is not None:
            guard_matrix[edit_type][guard].append(1 if str(val).lower() == 'true' else 0)

sensitivity_rows = []
for edit_type, guards in guard_matrix.items():
    row_data = {'edit_type': edit_type}
    for guard, values in guards.items():
        if values:
            row_data[f'{guard}_pass_rate'] = sum(values) / len(values)
    sensitivity_rows.append(row_data)

if sensitivity_rows:
    with open(analysis_dir / "guard_sensitivity_matrix.csv", 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=sensitivity_rows[0].keys())
        writer.writeheader()
        writer.writerows(sensitivity_rows)
    print(f"Wrote guard sensitivity matrix")

# Policy digest summary (per model)
policy_summary: dict[str, dict[str, object]] = {}
for row in invar_rows:
    model = row.get('model', 'unknown')
    digest = row.get('policy_digest_hash')
    changed = str(row.get('policy_digest_changed')).lower() == 'true'
    entry = policy_summary.setdefault(
        model,
        {
            'thresholds_hashes': set(),
            'policy_changed_true': 0,
            'total_certs': 0,
        },
    )
    entry['total_certs'] = int(entry.get('total_certs', 0)) + 1
    if digest:
        hashes = entry.setdefault('thresholds_hashes', set())
        if isinstance(hashes, set):
            hashes.add(str(digest))
    if changed:
        entry['policy_changed_true'] = int(entry.get('policy_changed_true', 0)) + 1

if policy_summary:
    serializable: dict[str, dict[str, object]] = {}
    for model, data in policy_summary.items():
        hashes = data.get('thresholds_hashes') or set()
        if isinstance(hashes, set):
            hash_list = sorted(hashes)
        else:
            hash_list = []
        serializable[model] = {
            'unique_thresholds_hashes': hash_list,
            'unique_hash_count': len(hash_list),
            'policy_changed_true': int(data.get('policy_changed_true', 0)),
            'total_certs': int(data.get('total_certs', 0)),
        }
    with open(analysis_dir / "policy_digest_summary.json", 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"Wrote policy digest summary for {len(serializable)} models")

# Determinism summary (per model + overall)
det_by_model: dict[str, dict[str, int]] = {}
overall = {'strict': 0, 'tolerance': 0, 'off': 0, 'unknown': 0}
for row in invar_rows:
    model = row.get('model', 'unknown')
    level = str(row.get('determinism_level') or '').strip().lower()
    if not level:
        level = 'unknown'
    if level not in overall:
        level = 'unknown'
    model_counts = det_by_model.setdefault(
        model, {k: 0 for k in overall.keys()}
    )
    model_counts[level] = int(model_counts.get(level, 0)) + 1
    overall[level] = int(overall.get(level, 0)) + 1

if det_by_model:
    det_payload = {
        'by_model': det_by_model,
        'overall': overall,
    }
    with open(analysis_dir / "determinism_summary.json", 'w') as f:
        json.dump(det_payload, f, indent=2)
    print(f"Wrote determinism summary for {len(det_by_model)} models")

# Calibration summary
calibration_summary = {}
for model_dir in output_dir.iterdir():
    if not model_dir.is_dir() or model_dir.name in ['logs', 'analysis', 'reports', 'presets', 'models']:
        continue
    cal_stats = model_dir / "certificates" / "calibration" / "calibration_stats.json"
    if cal_stats.exists():
        try:
            calibration_summary[model_dir.name] = json.loads(cal_stats.read_text())
        except Exception as e:
            print(f"Error loading {cal_stats}: {e}")

if calibration_summary:
    with open(analysis_dir / "calibration_summary.json", 'w') as f:
        json.dump(calibration_summary, f, indent=2)
    print(f"Wrote calibration summary for {len(calibration_summary)} models")
EOF
}

# ============ ANALYSIS ============
run_analysis() {
    log_section "CORRELATION ANALYSIS"

	    python3 <<- EOF
	\
	    import json
	    import csv
	    import math
	    from pathlib import Path
	    from collections import defaultdict
	
	    output_dir = Path("${OUTPUT_DIR}")
	    analysis_dir = output_dir / "analysis"
	
	    skip_dirs = {
	        "logs",
	        "analysis",
	        "reports",
	        "presets",
	        "models",
	        "queue",
	        "workers",
	        "state",
	        "evals",
	        "certificates",
	    }
	
	    def _is_model_dir(path: Path) -> bool:
	        if not path.is_dir():
	            return False
	        if path.name in skip_dirs:
	            return False
	        if not (path / ".baseline_path").exists():
	            return False
	        return True
	
	    def as_bool(val):
	        if val is None:
	            return False
	        if isinstance(val, bool):
	            return val
	        if isinstance(val, str):
	            return val.strip().lower() in ("true", "1", "yes", "on")
	        return bool(val)
	
	    def classify_edit(edit_name: str) -> str:
	        edit_lower = (edit_name or "").strip().lower()
	        if edit_lower == "errors" or edit_lower.startswith("error_"):
	            return "error_injection"
	        if "_clean" in edit_lower or edit_lower == "baseline":
	            return "clean"
	        if "_stress" in edit_lower:
	            return "stress"
	        return "stress"
	
	    eval_data = defaultdict(dict)
	    eval_csv = analysis_dir / "eval_results.csv"
	    if eval_csv.exists():
	        with open(eval_csv) as f:
	            for row in csv.DictReader(f):
	                try:
	                    key = (row["model"], row["edit_type"])
	                    val = row.get("value", "")
	                    if val and val.strip():
	                        eval_data[key][row["task"]] = float(val)
	                except Exception:
	                    pass
	
	    invar_data = defaultdict(list)
	    invar_csv = analysis_dir / "invarlock_results.csv"
	    if invar_csv.exists():
	        with open(invar_csv) as f:
	            for row in csv.DictReader(f):
	                invar_data[(row["model"], row["edit_type"])].append(row)
	
	    cal_summary = {}
	    cal_json = analysis_dir / "calibration_summary.json"
	    if cal_json.exists():
	        cal_summary = json.loads(cal_json.read_text())
	
	    print("=== CORRELATION ANALYSIS (Proof Pack) ===")
	    print()
	
	    results = {
	        "models": {},
	        "error_detection": {"detected": [], "missed": []},
	        "calibration": cal_summary,
	        "pm_correlation": {},
	    }
	
	    degraded_edits = 0
	    degraded_runs = []
	    categories = defaultdict(int)
	    pm_points = []
	    triage_counts = defaultdict(int)
	
	    guard_fields = {
	        "spectral": "spectral_stable",
	        "rmt": "rmt_stable",
	        "invariants": "invariants_pass",
	    }
	    guard_counts = defaultdict(lambda: defaultdict(lambda: {"pass": 0, "total": 0}))
	
	    for model_dir in output_dir.iterdir():
	        if not _is_model_dir(model_dir):
	            continue
	
	        model = model_dir.name
	        results["models"][model] = {}
	        print()
	        print(f"### {model} ###")
	
	        baseline_key = (model, "baseline")
	        baseline_evals = eval_data.get(baseline_key, {})
	
	        for edit_type_key, invar_results in invar_data.items():
	            if edit_type_key[0] != model:
	                continue
	            edit_type = edit_type_key[1]
	            edit_category = classify_edit(edit_type)
	
	            if edit_type not in {"errors", "calibration"} and edit_category in {
	                "clean",
	                "stress",
	            }:
	                for row in invar_results:
	                    for guard_name, guard_field in guard_fields.items():
	                        val = row.get(guard_field)
	                        if val is None:
	                            continue
	                        bucket = guard_counts[guard_name][edit_category]
	                        bucket["total"] += 1
	                        if str(val).lower() == "true":
	                            bucket["pass"] += 1
	
	            if edit_type in {"errors", "calibration"}:
	                continue
	
	            edit_evals = eval_data.get((model, edit_type), {})
	
	            n_table = {
	                "mmlu": 14042,
	                "hellaswag": 10042,
	                "arc_challenge": 2590,
	                "winogrande": 1767,
	            }
	
	            has_regression = False
	            deltas = []
	            delta_by_task = {}
	            regression_tasks = []
	            for task, base_val in baseline_evals.items():
	                edit_val = edit_evals.get(task)
	                if edit_val is None:
	                    continue
	                delta = edit_val - base_val
	                deltas.append(delta)
	                delta_by_task[task] = delta
	                task_key = task
	                if task_key.startswith("arc"):
	                    task_key = "arc_challenge"
	                n = n_table.get(task_key, 1000)
	                p = max(min(base_val, 0.999), 0.001)
	                se = math.sqrt(p * (1.0 - p) / n)
	                if delta < -2.0 * se:
	                    has_regression = True
	                    regression_tasks.append(task)
	            mean_delta = sum(deltas) / len(deltas) if deltas else 0.0
	            worst_task = None
	            worst_delta = None
	            if delta_by_task:
	                worst_task, worst_delta = min(delta_by_task.items(), key=lambda item: item[1])
	
	            invar_flagged = any(
	                str(r.get("all_pass", "")).lower() == "false" or r.get("all_pass") is False
	                for r in invar_results
	            )
	            degraded_present = any(as_bool(r.get("pm_degraded")) for r in invar_results)
	            if degraded_present:
	                degraded_edits += 1
	                degraded_runs.extend(
	                    [
	                        f"{model}/{r.get('run', 'unknown')}"
	                        for r in invar_results
	                        if as_bool(r.get("pm_degraded"))
	                    ]
	                )
	            if degraded_present:
	                invar_flagged = True
	
	            pm_vals = []
	            for r in invar_results:
	                try:
	                    v = r.get("pm_ratio")
	                    if v is None or v == "":
	                        continue
	                    pm_vals.append(float(v))
	                except Exception:
	                    continue
	            pm_ratio_mean = sum(pm_vals) / len(pm_vals) if pm_vals else None
	            if pm_ratio_mean is not None and deltas:
	                pm_points.append((mean_delta, math.log(pm_ratio_mean)))
	
	            triage_votes = []
	            for r in invar_results:
	                t = str(r.get("triage", "") or "").strip().upper()
	                if t:
	                    triage_votes.append(t)
	            if any(t == "FAIL" for t in triage_votes):
	                triage = "FAIL"
	            elif triage_votes and all(t == "PASS" for t in triage_votes):
	                triage = "PASS"
	            else:
	                triage = "REVIEW"
	            triage_counts[triage] += 1
	
	            if has_regression and invar_flagged:
	                category = "TRUE_POSITIVE"
	            elif not has_regression and invar_flagged:
	                category = "FALSE_POSITIVE"
	            elif not has_regression and not invar_flagged:
	                category = "TRUE_NEGATIVE"
	            else:
	                category = "FALSE_NEGATIVE"
	
	            categories[category] += 1
	            results["models"][model][edit_type] = {
	                "category": category,
	                "regression": has_regression,
	                "flagged": invar_flagged,
	                "triage": triage,
	                "mean_delta_eval": mean_delta,
	                "delta_by_task": delta_by_task,
	                "regression_tasks": regression_tasks,
	                "worst_delta_task": worst_task,
	                "worst_delta": worst_delta,
	                "mean_pm_ratio": pm_ratio_mean,
	            }
	            print(f"  {edit_type}: {category}")
	
	        for row in invar_data.get((model, "errors"), []):
	            def is_false(val):
	                if val is None:
	                    return True
	                if isinstance(val, bool):
	                    return not val
	                if isinstance(val, str):
	                    return val.lower() in ("false", "0", "")
	                return False
	
	            caught = is_false(row.get("all_pass")) or is_false(row.get("invariants_pass"))
	            if caught:
	                results["error_detection"]["detected"].append(
	                    f"{model}/{row.get('run', 'unknown')}"
	                )
	            else:
	                results["error_detection"]["missed"].append(
	                    f"{model}/{row.get('run', 'unknown')}"
	                )
	
	    if len(pm_points) >= 2:
	        xs = [p[0] for p in pm_points]
	        ys = [p[1] for p in pm_points]
	        mean_x = sum(xs) / len(xs)
	        mean_y = sum(ys) / len(ys)
	        num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
	        den_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
	        den_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
	        corr = num / (den_x * den_y) if den_x > 0 and den_y > 0 else 0.0
	        results["pm_correlation"] = {
	            "pearson_r_delta_vs_log_pm": corr,
	            "num_points": len(pm_points),
	        }
	        print()
	        print(
	            f"PM correlation (Δeval vs log pm_ratio): r = {corr:.3f} over {len(pm_points)} edits"
	        )
	    else:
	        results["pm_correlation"] = {
	            "pearson_r_delta_vs_log_pm": 0.0,
	            "num_points": len(pm_points),
	        }
	
	    print()
	    print("=== SUMMARY ===")
	    tp, tn = categories["TRUE_POSITIVE"], categories["TRUE_NEGATIVE"]
	    fp, fn = categories["FALSE_POSITIVE"], categories["FALSE_NEGATIVE"]
	    total = tp + tn + fp + fn
	
	    accuracy = (tp + tn) / total if total > 0 else 0
	    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
	    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
	    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
	
	    err_detected = len(results["error_detection"]["detected"])
	    err_missed = len(results["error_detection"]["missed"])
	    err_total = err_detected + err_missed
	    err_rate = err_detected / err_total if err_total > 0 else 0
	
	    def wilson_interval(successes, n, z=1.96):
	        if n <= 0:
	            return (0.0, 0.0)
	        p_hat = successes / n
	        denom = 1.0 + z * z / n
	        centre = p_hat + z * z / (2 * n)
	        margin = z * ((p_hat * (1 - p_hat) + z * z / (4 * n)) / n) ** 0.5
	        lower = max(0.0, (centre - margin) / denom)
	        upper = min(1.0, (centre + margin) / denom)
	        return (lower, upper)
	
	    acc_ci = wilson_interval(tp + tn, total) if total > 0 else (0.0, 0.0)
	    err_ci = wilson_interval(err_detected, err_total) if err_total > 0 else (0.0, 0.0)
	
	    import math as _math
	
	    sample_confidence = (
	        min((_math.log1p(total) / _math.log1p(64)) * 25, 25) if total > 0 else 0
	    )
	    error_confidence = (
	        min((_math.log1p(err_total) / _math.log1p(16)) * 25, 25) if err_total > 0 else 0
	    )
	    acc_ci_width = acc_ci[1] - acc_ci[0]
	    accuracy_confidence = max(0.0, (1.0 - min(acc_ci_width, 1.0)) * 25)
	    balance_confidence = f1 * 25
	    confidence_score = sample_confidence + error_confidence + accuracy_confidence + balance_confidence
	
	    if confidence_score >= 85:
	        confidence_level = "HIGH"
	    elif confidence_score >= 70:
	        confidence_level = "MEDIUM"
	    elif confidence_score >= 50:
	        confidence_level = "LOW"
	    else:
	        confidence_level = "VERY_LOW"
	
	    print(f"Accuracy: {accuracy:.0%}")
	    print(f"Precision: {precision:.0%}")
	    print(f"Recall: {recall:.0%}")
	    print(f"F1 Score: {f1:.0%}")
	    print(f"Degraded edits: {degraded_edits}")
	    print(f"Error Detection: {err_detected}/{err_total} ({err_rate:.0%})")
	    print(
	        "Triage (edits): PASS={pass_count} REVIEW={review_count} FAIL={fail_count}".format(
	            pass_count=triage_counts.get("PASS", 0),
	            review_count=triage_counts.get("REVIEW", 0),
	            fail_count=triage_counts.get("FAIL", 0),
	        )
	    )
	    print(f"Confidence Score: {confidence_score:.1f}/100 ({confidence_level})")
	
	    category_results = {
	        "clean": {"expected": "PASS", "pass": 0, "fail": 0, "total": 0},
	        "stress": {"expected": "FAIL", "pass": 0, "fail": 0, "total": 0},
	        "error_injection": {
	            "expected": "DETECTED",
	            "detected": 0,
	            "missed": 0,
	            "total": 0,
	        },
	    }
	    for model, edits in results["models"].items():
	        for edit_name, edit_data in edits.items():
	            cat = classify_edit(edit_name)
	            if cat not in ("clean", "stress"):
	                continue
	            category_results[cat]["total"] += 1
	            if edit_data.get("flagged"):
	                category_results[cat]["fail"] += 1
	            else:
	                category_results[cat]["pass"] += 1
	
	    category_results["error_injection"]["detected"] = err_detected
	    category_results["error_injection"]["missed"] = err_missed
	    category_results["error_injection"]["total"] = err_total
	
	    guard_matrix = {}
	    for guard_name, categories_data in guard_counts.items():
	        guard_matrix[guard_name] = {}
	        for cat_name, counts in categories_data.items():
	            total_count = counts.get("total", 0)
	            if not total_count:
	                continue
	            guard_matrix[guard_name][f"{cat_name}_pass_rate"] = (
	                counts.get("pass", 0) / total_count
	            )
	
	    results["summary"] = {
	        "accuracy": accuracy,
	        "precision": precision,
	        "recall": recall,
	        "f1_score": f1,
	        "error_detection_rate": err_rate,
	        "categories": dict(categories),
	        "confidence_score": confidence_score,
	        "confidence_level": confidence_level,
	        "triage_counts": dict(triage_counts),
	        "degraded_edits": degraded_edits,
	        "degraded_runs": degraded_runs,
	        "total_tests": total,
	        "models_tested": len(results["models"]),
	        "accuracy_ci": acc_ci,
	        "error_rate_ci": err_ci,
	    }
	
	    results["category_results"] = category_results
	    results["guard_matrix"] = guard_matrix
	
	    with open(analysis_dir / "correlation_analysis.json", "w") as f:
	        json.dump(results, f, indent=2)
EOF
}

# ============ GENERATE VERDICT ============
generate_verdict() {
    log_section "GENERATING FINAL VERDICT"

	    python3 <<- EOF
	\
	    import csv
	    import json
	    import os
	    import re
	    from pathlib import Path
	    from datetime import datetime
	
	    output_dir = Path("${OUTPUT_DIR}")
	    analysis_dir = output_dir / "analysis"
	    reports_dir = output_dir / "reports"
	    reports_dir.mkdir(exist_ok=True)
	
	    analysis_file = analysis_dir / "correlation_analysis.json"
	    if not analysis_file.exists():
	        analysis = {
	            "summary": {
	                "accuracy": 0,
	                "precision": 0,
	                "recall": 0,
	                "f1_score": 0,
	                "error_detection_rate": 0,
	                "confidence_score": 0,
	                "confidence_level": "UNKNOWN",
	            },
	            "calibration": {},
	        }
	    else:
	        try:
	            analysis = json.loads(analysis_file.read_text())
	        except Exception:
	            analysis = {"summary": {}, "calibration": {}}
	
	    summary = analysis.get("summary", {})
	    category_results = analysis.get("category_results", {}) or {}
	    guard_matrix = analysis.get("guard_matrix", {}) or {}
	
	    accuracy = summary.get("accuracy", 0)
	    precision = summary.get("precision", 0)
	    recall = summary.get("recall", 0)
	    f1 = summary.get("f1_score", 0)
	    err_rate = summary.get("error_detection_rate", 0)
	    confidence_score = summary.get("confidence_score", 0)
	    confidence_level = summary.get("confidence_level", "UNKNOWN")
	    total_tests = summary.get("total_tests", 0)
	    models_tested = summary.get("models_tested", 0)
	    triage_counts = summary.get("triage_counts", {}) or {}
	    triage_pass = triage_counts.get("PASS", 0)
	    triage_review = triage_counts.get("REVIEW", 0)
	    triage_fail = triage_counts.get("FAIL", 0)
	    degraded = summary.get("degraded_edits", 0) or 0
	    degraded_runs = summary.get("degraded_runs", []) or []
	    models = analysis.get("models", {}) or {}
	
	    determinism_path = analysis_dir / "determinism_repeats.json"
	    determinism_repeats = None
	    if determinism_path.exists():
	        try:
	            determinism_repeats = json.loads(determinism_path.read_text())
	        except Exception:
	            determinism_repeats = None
	
	    gpu_count = (
	        os.environ.get("PACK_GPU_COUNT") or os.environ.get("NUM_GPUS") or ""
	    ).strip() or "unknown"
	    gpu_mem = (
	        os.environ.get("PACK_GPU_MEM_GB") or os.environ.get("GPU_MEMORY_GB") or ""
	    ).strip()
	    gpu_name = (os.environ.get("PACK_GPU_NAME") or "GPU").strip() or "GPU"
	    gpu_mem_label = f"{gpu_mem}GB" if gpu_mem else "unknown"
	    tag_name = re.sub(r"[^A-Za-z0-9]+", "_", gpu_name).strip("_") or "GPU"
	    platform_header = f"{gpu_name} {gpu_mem_label} x {gpu_count} GPU"
	    platform_line = f"{gpu_count}x {gpu_name} {gpu_mem_label}"
	    platform_tag = f"{tag_name}_{gpu_mem_label}_x{gpu_count}"
	
	    def fmt_delta(val):
	        try:
	            return f"{float(val):+0.4f}"
	        except Exception:
	            return "n/a"
	
	    def ordered_tasks(delta_by_task):
	        order = ["mmlu", "hellaswag", "arc_challenge", "winogrande"]
	        tasks = [task for task in order if task in delta_by_task]
	        extra = sorted(
	            task
	            for task in delta_by_task
	            if task not in order and not str(task).startswith("mmlu_")
	        )
	        return tasks + extra
	
	    def format_pct(value):
	        try:
	            return f"{float(value):.0%}"
	        except Exception:
	            return "n/a"
	
	    def format_ratio(numerator, denominator):
	        if denominator <= 0:
	            return "n/a"
	        return f"{numerator / denominator:.0%}"
	
	    def compute_confidence_breakdown(summary, category_results, guard_matrix):
	        base_score = 100
	        deductions = []
	
	        spectral_clean = guard_matrix.get("spectral", {}).get("clean_pass_rate", 0)
	        try:
	            spectral_clean = float(spectral_clean)
	        except Exception:
	            spectral_clean = 0
	        if spectral_clean < 0.8:
	            pts = int(20 * (1 - spectral_clean / 0.8))
	            deductions.append(
	                {
	                    "reason": f"spectral_clean_pass_rate_{spectral_clean:.0%}",
	                    "points": -pts,
	                    "note": (
	                        "Spectral guard: "
	                        f"{spectral_clean:.0%} pass rate on clean (target: >80%)"
	                    ),
	                }
	            )
	
	        clean = category_results.get("clean", {})
	        if clean.get("total", 0) > 0:
	            clean_pass_rate = clean.get("pass", 0) / clean["total"]
	            if clean_pass_rate < 1.0:
	                pts = int(15 * (1 - clean_pass_rate))
	                deductions.append(
	                    {
	                        "reason": f"clean_specificity_{clean_pass_rate:.0%}",
	                        "points": -pts,
	                        "note": (
	                            f"Clean edits: {clean.get('pass', 0)}/{clean['total']} "
	                            "passed (expected 100%)"
	                        ),
	                    }
	                )
	
	        expected_clean_types = 4
	        actual_clean = clean.get("total", 0)
	        if actual_clean < expected_clean_types:
	            missing = expected_clean_types - actual_clean
	            pts = 2 * missing
	            deductions.append(
	                {
	                    "reason": f"missing_clean_types_{missing}of{expected_clean_types}",
	                    "points": -pts,
	                    "note": f"Missing clean edit types: {missing} of {expected_clean_types}",
	                }
	            )
	
	        final_score = base_score + sum(d["points"] for d in deductions)
	        final_score = max(0, min(100, final_score))
	
	        level = (
	            "HIGH"
	            if final_score >= 85
	            else "MEDIUM"
	            if final_score >= 75
	            else "LOW"
	            if final_score >= 50
	            else "VERY_LOW"
	        )
	
	        return {
	            "base_score": base_score,
	            "deductions": deductions,
	            "final_score": final_score,
	            "level": level,
	        }
	
	    def generate_recommendations(results):
	        recs = []
	
	        guard_matrix = results.get("guard_matrix", {})
	        spectral = guard_matrix.get("spectral", {})
	        spectral_clean = spectral.get("clean_pass_rate", 1.0)
	        try:
	            spectral_clean = float(spectral_clean)
	        except Exception:
	            spectral_clean = 1.0
	
	        if spectral_clean < 0.5:
	            recs.append(
	                {
	                    "id": "spectral_calibration",
	                    "severity": "critical",
	                    "title": "Spectral Calibration Needed",
	                    "description": (
	                        "Spectral guard failing clean edits indicates uncalibrated thresholds."
	                    ),
	                    "fix_command": (
	                        "PACK_GUARDS_ORDER=\"invariants,spectral,rmt,variance,invariants\" "
	                        "./scripts/proof_packs/run_pack.sh --suite subset --net 1"
	                    ),
	                }
	            )
	
	        category = results.get("category_results", {})
	        clean = category.get("clean", {})
	        if clean.get("total", 0) < 4:
	            recs.append(
	                {
	                    "id": "missing_clean_types",
	                    "severity": "high",
	                    "title": "Missing Clean Edit Types",
	                    "description": (
	                        f"Only {clean.get('total', 0)} of 4 clean edit types generated."
	                    ),
	                    "fix_command": "Check CALIBRATE_CLEAN task logs for calibration failures.",
	                }
	            )
	
	        return recs
	
	    metric_definitions = "\n".join(
	        [
	            "METRIC DEFINITIONS:",
	            "  * Regression: any lm-eval benchmark drop below -2xSE vs baseline.",
	            "  * Flagged: InvarLock guard failure or primary metric degradation.",
	            "  * Accuracy/Precision/Recall treat regression as positive class.",
	        ]
	    )
	
	    delta_lines = []
	    delta_summary = {}
	    if models:
	        delta_lines.append("LM-EVAL DELTAS (edit - baseline):")
	        for model in sorted(models):
	            edits = models.get(model) or {}
	            delta_summary[model] = {}
	            if not edits:
	                continue
	            delta_lines.append(f"  {model}:")
	            for edit_name in sorted(edits):
	                entry = edits.get(edit_name) or {}
	                delta_by_task = entry.get("delta_by_task") or {}
	                mean_delta = entry.get("mean_delta_eval")
	                worst_task = entry.get("worst_delta_task")
	                worst_delta = entry.get("worst_delta")
	                regression_tasks = entry.get("regression_tasks") or []
	                tasks_ordered = ordered_tasks(delta_by_task)
	                if tasks_ordered:
	                    task_blob = ", ".join(
	                        f"{task}:{fmt_delta(delta_by_task.get(task))}" for task in tasks_ordered
	                    )
	                else:
	                    task_blob = "n/a"
	                worst_blob = (
	                    f"{worst_task}:{fmt_delta(worst_delta)}" if worst_task else "n/a"
	                )
	                regression_blob = "none" if not regression_tasks else ", ".join(regression_tasks)
	                delta_lines.append(
	                    f"    {edit_name}: mean {fmt_delta(mean_delta)} | "
	                    f"worst {worst_blob} | regressions {regression_blob} | [{task_blob}]"
	                )
	                mmlu_subtasks = {
	                    k: v for k, v in delta_by_task.items() if str(k).startswith("mmlu_")
	                }
	                delta_by_task_compact = {
	                    k: v
	                    for k, v in delta_by_task.items()
	                    if not str(k).startswith("mmlu_")
	                }
	                if mmlu_subtasks:
	                    delta_by_task_compact["mmlu_subtasks"] = dict(
	                        sorted(mmlu_subtasks.items())
	                    )
	                delta_summary[model][edit_name] = {
	                    "mean_delta_eval": mean_delta,
	                    "delta_by_task": delta_by_task_compact,
	                    "regression_tasks": regression_tasks,
	                    "worst_delta_task": worst_task,
	                    "worst_delta": worst_delta,
	                }
	        delta_section = "\n".join(delta_lines) + "\n\n"
	    else:
	        delta_section = ""
	
	    guard_matrix_lines = []
	    guard_matrix_csv = analysis_dir / "guard_sensitivity_matrix.csv"
	    if guard_matrix_csv.exists():
	        rows = []
	        with open(guard_matrix_csv) as f:
	            for row in csv.DictReader(f):
	                rows.append(row)
	        if rows:
	            guard_matrix_lines.append("GUARD SENSITIVITY MATRIX (per edit type):")
	            for row in rows:
	                edit_type = row.get("edit_type", "unknown")
	                spectral_rate = format_pct(row.get("spectral_stable_pass_rate"))
	                rmt_rate = format_pct(row.get("rmt_stable_pass_rate"))
	                inv_rate = format_pct(row.get("invariants_pass_rate"))
	                guard_matrix_lines.append(
	                    f"  {edit_type}: spectral {spectral_rate} | rmt {rmt_rate} | "
	                    f"invariants {inv_rate}"
	                )
	            guard_matrix_lines.append("")
	
	    guard_effect_lines = ["GUARD EFFECTIVENESS BY EDIT CATEGORY:"]
	    clean = category_results.get("clean", {})
	    clean_total = clean.get("total", 0)
	    clean_pass = clean.get("pass", 0)
	    guard_effect_lines.append(
	        "  Clean edits (expected PASS): "
	        f"{clean_pass}/{clean_total} passed ({format_ratio(clean_pass, clean_total)})"
	    )
	
	    stress = category_results.get("stress", {})
	    stress_total = stress.get("total", 0)
	    stress_fail = stress.get("fail", 0)
	    guard_effect_lines.append(
	        "  Stress edits (expected FAIL): "
	        f"{stress_fail}/{stress_total} flagged ({format_ratio(stress_fail, stress_total)})"
	    )
	
	    error_injection = category_results.get("error_injection", {})
	    err_total = error_injection.get("total", 0)
	    err_detected = error_injection.get("detected", 0)
	    guard_effect_lines.append(
	        "  Error injections (expected DETECTED): "
	        f"{err_detected}/{err_total} detected ({format_ratio(err_detected, err_total)})"
	    )
	
	    for guard_name in ("spectral", "rmt", "invariants"):
	        clean_rate = guard_matrix.get(guard_name, {}).get("clean_pass_rate")
	        if clean_rate is not None:
	            guard_effect_lines.append(
	                f"  {guard_name} clean pass rate: {format_pct(clean_rate)}"
	            )
	
	    confidence_breakdown = compute_confidence_breakdown(
	        summary, category_results, guard_matrix
	    )
	    confidence_score = confidence_breakdown["final_score"]
	    confidence_level = confidence_breakdown["level"]
	
	    confidence_lines = ["CONFIDENCE SCORE BREAKDOWN:"]
	    confidence_lines.append(f"  Base score: {confidence_breakdown['base_score']}")
	    for deduction in confidence_breakdown["deductions"]:
	        confidence_lines.append(
	            f"  {deduction['points']} {deduction['reason']}: {deduction['note']}"
	        )
	    confidence_lines.append(
	        f"  Final score: {confidence_breakdown['final_score']} ({confidence_breakdown['level']})"
	    )
	
	    recommendations = generate_recommendations(
	        {"guard_matrix": guard_matrix, "category_results": category_results}
	    )
	    if recommendations:
	        rec_lines = ["RECOMMENDATIONS:"]
	        for rec in recommendations:
	            rec_lines.append(
	                f"  [{rec['severity']}] {rec['title']}: {rec['description']}"
	            )
	            if rec.get("fix_command"):
	                rec_lines.append(f"    fix: {rec['fix_command']}")
	    else:
	        rec_lines = ["RECOMMENDATIONS: None."]
	
	    phase0_pass = accuracy >= 0.6 and err_rate >= 0.8
	    if degraded > 0:
	        phase0_pass = False
	
	    if degraded > 0:
	        verdict = "PHASE0_DEGRADED"
	        verdict_confidence = "LOW"
	    elif phase0_pass and accuracy >= 0.8 and confidence_score >= 75:
	        verdict = "PHASE0_VALIDATED"
	        verdict_confidence = "HIGH"
	    elif phase0_pass and confidence_score >= 60:
	        verdict = "PHASE0_VALIDATED"
	        verdict_confidence = "MEDIUM"
	    elif phase0_pass:
	        verdict = "PHASE0_VALIDATED"
	        verdict_confidence = "LOW"
	    else:
	        verdict = "PHASE0_FAILED"
	        verdict_confidence = "HIGH" if confidence_score >= 60 else "LOW"
	
	    report_lines = [
	        "================================================================================",
	        f"     INVARLOCK PHASE 0 VALIDATION - {platform_header}",
	        "================================================================================",
	        f"     Models Tested:     {models_tested}",
	        f"     Total Tests:       {total_tests}",
	        "     Edit Types:        4 x 2 versions = 8 per model",
	        "--------------------------------------------------------------------------------",
	        f"     Accuracy:          {accuracy:.0%}",
	        f"     Precision:         {precision:.0%}",
	        f"     Recall:            {recall:.0%}",
	        f"     F1 Score:          {f1:.0%}",
	        f"     Error Detection:   {err_rate:.0%}",
	        "--------------------------------------------------------------------------------",
	        f"     CONFIDENCE SCORE:  {confidence_score:.1f}/100 ({confidence_level})",
	        f"     TRIAGE (edits):    PASS={triage_pass} REVIEW={triage_review} FAIL={triage_fail}",
	        f"     DEGRADED CERTS:    {degraded}",
	        "--------------------------------------------------------------------------------",
	        f"     VERDICT: {verdict}",
	        f"     VERDICT CONFIDENCE: {verdict_confidence}",
	        "================================================================================",
	        "",
	        metric_definitions,
	    ]
	
	    report_sections = [
	        "\n".join(report_lines),
	        "\n".join(guard_effect_lines),
	        "\n".join(confidence_lines),
	    ]
	    if guard_matrix_lines:
	        report_sections.append("\n".join(guard_matrix_lines))
	    report_sections.append("\n".join(rec_lines))
	    if delta_section:
	        report_sections.append(delta_section.rstrip("\n"))
	
	    report_sections.append(
	        "\n".join(
	            [
	                "EDIT TYPES TESTED:",
	                "  * Quantization RTN (group-wise): 8-bit (clean), 4-bit (stress)",
	                "  * FP8 Quantization (E4M3 clean, E5M2 stress)",
	                "  * Magnitude Pruning: 10% (clean), 50% (stress)",
	                "  * Low-Rank SVD: rank-256 (clean), rank-32 (stress)",
	                "",
	                f"PLATFORM: {platform_line}",
	            ]
	        )
	    )
	
	    report = "\n\n".join(report_sections) + "\n\n"
	
	    if verdict == "PHASE0_VALIDATED":
	        report += "RESULT: InvarLock Phase 0 VALIDATED on proof pack hardware.\n"
	    elif verdict == "PHASE0_DEGRADED":
	        report += (
	            "RESULT: Phase 0 degraded. "
	            f"{degraded} certificate(s) reported degraded primary metrics. "
	            f"See runs: {', '.join(degraded_runs) if degraded_runs else 'n/a'}\n"
	        )
	    else:
	        report += (
	            "RESULT: Phase 0 validation failed. "
	            f"Accuracy: {accuracy:.0%}, Error Detection: {err_rate:.0%}\n"
	        )
	
	    print(report)
	
	    with open(reports_dir / "final_verdict.txt", "w") as f:
	        f.write(report)
	
	    with open(reports_dir / "final_verdict.json", "w") as f:
	        json.dump(
	            {
	                "verdict": verdict,
	                "verdict_confidence": verdict_confidence,
	                "metrics": {
	                    "accuracy": accuracy,
	                    "precision": precision,
	                    "recall": recall,
	                    "f1_score": f1,
	                    "error_detection_rate": err_rate,
	                },
	                "confidence": {"score": confidence_score, "level": confidence_level},
	                "confidence_breakdown": confidence_breakdown,
	                "triage": {"pass": triage_pass, "review": triage_review, "fail": triage_fail},
	                "degraded": {"count": degraded, "runs": degraded_runs},
	                "phase0_pass": phase0_pass,
	                "platform": platform_tag,
	                "platform_name": gpu_name,
	                "suite": os.environ.get("PACK_SUITE"),
	                "network_mode": "online" if os.environ.get("PACK_NET") == "1" else "offline",
	                "determinism_mode": os.environ.get("PACK_DETERMINISM"),
	                "determinism_repeats": determinism_repeats,
	                "models_tested": models_tested,
	                "total_tests": total_tests,
	                "lm_eval_deltas": delta_summary,
	                "guard_matrix": guard_matrix,
	                "category_results": category_results,
	                "recommendations": recommendations,
	                "timestamp": datetime.now().isoformat(),
	            },
	            f,
	            indent=2,
	        )
EOF
}
