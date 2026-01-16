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

eval_data = defaultdict(dict)
eval_csv = analysis_dir / "eval_results.csv"
if eval_csv.exists():
    with open(eval_csv) as f:
        for row in csv.DictReader(f):
            try:
                key = (row['model'], row['edit_type'])
                val = row.get('value', '')
                if val and val.strip():
                    eval_data[key][row['task']] = float(val)
            except: pass

invar_data = defaultdict(list)
invar_csv = analysis_dir / "invarlock_results.csv"
if invar_csv.exists():
    with open(invar_csv) as f:
        for row in csv.DictReader(f):
            invar_data[(row['model'], row['edit_type'])].append(row)

cal_summary = {}
cal_json = analysis_dir / "calibration_summary.json"
if cal_json.exists():
    cal_summary = json.loads(cal_json.read_text())

print("=== CORRELATION ANALYSIS (Proof Pack) ===\n")

results = {
    'models': {},
    'error_detection': {'detected': [], 'missed': []},
    'calibration': cal_summary,
    'pm_correlation': {},
}
def as_bool(val):
    if val is None: return False
    if isinstance(val, bool): return val
    if isinstance(val, str): return val.strip().lower() in ('true', '1', 'yes', 'on')
    return bool(val)

degraded_edits = 0
degraded_runs = []
categories = defaultdict(int)
# Track (delta_eval, pm_ratio) pairs for correlation analysis
pm_points = []
triage_counts = defaultdict(int)

for model_dir in output_dir.iterdir():
    if not _is_model_dir(model_dir):
        continue

    model = model_dir.name
    results['models'][model] = {}
    print(f"\n### {model} ###")

    baseline_key = (model, 'baseline')
    baseline_evals = eval_data.get(baseline_key, {})

    for edit_type_key, invar_results in invar_data.items():
        if edit_type_key[0] != model:
            continue
        edit_type = edit_type_key[1]

        # Skip runs that are not part of the edit-vs-eval correlation study.
        # - errors: handled separately in error_detection
        # - calibration: no lm-eval baseline, would inflate TRUE_NEGATIVE counts
        if edit_type in {"errors", "calibration"}:
            continue

        edit_evals = eval_data.get((model, edit_type), {})

        # Determine if this edit has a statistically meaningful regression vs baseline.
        # Use a simple binomial standard error approximation per benchmark.
        N_TABLE = {
            'mmlu': 14042,
            'hellaswag': 10042,
            'arc_challenge': 2590,
            'winogrande': 1767,
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
            # Map task name back to benchmark key
            task_key = task
            if task_key.startswith('arc'):
                task_key = 'arc_challenge'
            n = N_TABLE.get(task_key, 1000)
            p = max(min(base_val, 0.999), 0.001)
            se = math.sqrt(p * (1.0 - p) / n)
            if delta < -2.0 * se:
                has_regression = True
                regression_tasks.append(task)
                # Keep scanning to accumulate deltas but we already know it's regressed
        mean_delta = sum(deltas) / len(deltas) if deltas else 0.0
        worst_task = None
        worst_delta = None
        if delta_by_task:
            worst_task, worst_delta = min(delta_by_task.items(), key=lambda item: item[1])

        invar_flagged = any(
            str(r.get('all_pass', '')).lower() == 'false' or r.get('all_pass') is False
            for r in invar_results
        )
        degraded_present = any(as_bool(r.get('pm_degraded')) for r in invar_results)
        if degraded_present:
            degraded_edits += 1
            degraded_runs.extend([f"{model}/{r.get('run', 'unknown')}" for r in invar_results if as_bool(r.get('pm_degraded'))])
        if degraded_present:
            invar_flagged = True

        # Aggregate primary-metric ratio for this edit (continuous InvarLock signal)
        pm_vals = []
        for r in invar_results:
            try:
                v = r.get('pm_ratio')
                if v is None or v == '':
                    continue
                pm_vals.append(float(v))
            except Exception:
                continue
	        pm_ratio_mean = sum(pm_vals) / len(pm_vals) if pm_vals else None
	        if pm_ratio_mean is not None and deltas:
	            pm_points.append((mean_delta, math.log(pm_ratio_mean)))

	        # Aggregate triage across replicates: FAIL if any fail, PASS if all pass.
	        triage_votes = []
	        for r in invar_results:
	            t = str(r.get('triage', '') or '').strip().upper()
	            if t:
	                triage_votes.append(t)
	        if any(t == 'FAIL' for t in triage_votes):
	            triage = 'FAIL'
	        elif triage_votes and all(t == 'PASS' for t in triage_votes):
	            triage = 'PASS'
	        else:
	            triage = 'REVIEW'
	        triage_counts[triage] += 1

	        if has_regression and invar_flagged: category = "TRUE_POSITIVE"
	        elif not has_regression and invar_flagged: category = "FALSE_POSITIVE"
	        elif not has_regression and not invar_flagged: category = "TRUE_NEGATIVE"
	        else: category = "FALSE_NEGATIVE"

        categories[category] += 1
	        results['models'][model][edit_type] = {
	            'category': category,
	            'regression': has_regression,
	            'flagged': invar_flagged,
	            'triage': triage,
	            'mean_delta_eval': mean_delta,
	            'delta_by_task': delta_by_task,
	            'regression_tasks': regression_tasks,
	            'worst_delta_task': worst_task,
	            'worst_delta': worst_delta,
	            'mean_pm_ratio': pm_ratio_mean,
	        }
	        print(f"  {edit_type}: {category}")

    for row in invar_data.get((model, 'errors'), []):
        def is_false(val):
            if val is None: return True
            if isinstance(val, bool): return not val
            if isinstance(val, str): return val.lower() in ('false', '0', '')
            return False
        caught = is_false(row.get('all_pass')) or is_false(row.get('invariants_pass'))
        if caught:
            results['error_detection']['detected'].append(f"{model}/{row.get('run', 'unknown')}")
        else:
            results['error_detection']['missed'].append(f"{model}/{row.get('run', 'unknown')}")

# Compute simple Pearson correlation between Δeval and log(pm_ratio)
if len(pm_points) >= 2:
    xs = [p[0] for p in pm_points]
    ys = [p[1] for p in pm_points]
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    corr = num / (den_x * den_y) if den_x > 0 and den_y > 0 else 0.0
    results['pm_correlation'] = {
        'pearson_r_delta_vs_log_pm': corr,
        'num_points': len(pm_points),
    }
    print(f"\nPM correlation (Δeval vs log pm_ratio): r = {corr:.3f} over {len(pm_points)} edits")
else:
    results['pm_correlation'] = {'pearson_r_delta_vs_log_pm': 0.0, 'num_points': len(pm_points)}

print("\n=== SUMMARY ===")
tp, tn = categories['TRUE_POSITIVE'], categories['TRUE_NEGATIVE']
fp, fn = categories['FALSE_POSITIVE'], categories['FALSE_NEGATIVE']
total = tp + tn + fp + fn

accuracy = (tp + tn) / total if total > 0 else 0
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

err_detected = len(results['error_detection']['detected'])
err_missed = len(results['error_detection']['missed'])
err_total = err_detected + err_missed
err_rate = err_detected / err_total if err_total > 0 else 0

# Wilson score intervals for accuracy and error detection rate (95% CI)
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

# Confidence components:
# - sample_confidence: grows with log(#tests) up to 25
# - error_confidence: grows with number of error injections up to 25
# - accuracy_confidence: higher when CI for accuracy is tight
# - balance_confidence: based on F1
import math as _math
sample_confidence = min((_math.log1p(total) / _math.log1p(64)) * 25, 25) if total > 0 else 0
error_confidence = min((_math.log1p(err_total) / _math.log1p(16)) * 25, 25) if err_total > 0 else 0
acc_ci_width = acc_ci[1] - acc_ci[0]
accuracy_confidence = max(0.0, (1.0 - min(acc_ci_width, 1.0)) * 25)
balance_confidence = f1 * 25
confidence_score = sample_confidence + error_confidence + accuracy_confidence + balance_confidence

if confidence_score >= 85: confidence_level = "HIGH"
elif confidence_score >= 70: confidence_level = "MEDIUM"
elif confidence_score >= 50: confidence_level = "LOW"
else: confidence_level = "VERY_LOW"

print(f"Accuracy: {accuracy:.0%}")
print(f"Precision: {precision:.0%}")
print(f"Recall: {recall:.0%}")
print(f"F1 Score: {f1:.0%}")
print(f"Degraded edits: {degraded_edits}")
	print(f"Error Detection: {err_detected}/{err_total} ({err_rate:.0%})")
	print(f"Triage (edits): PASS={triage_counts.get('PASS', 0)} REVIEW={triage_counts.get('REVIEW', 0)} FAIL={triage_counts.get('FAIL', 0)}")
	print(f"Confidence Score: {confidence_score:.1f}/100 ({confidence_level})")

	results['summary'] = {
	    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'f1_score': f1,
    'error_detection_rate': err_rate,
    'categories': dict(categories),
	    'confidence_score': confidence_score,
	    'confidence_level': confidence_level,
	    'triage_counts': dict(triage_counts),
	    'degraded_edits': degraded_edits,
	    'degraded_runs': degraded_runs,
	    'total_tests': total,
	    'models_tested': len(results['models']),
	    'accuracy_ci': acc_ci,
	    'error_rate_ci': err_ci,
	}

with open(analysis_dir / "correlation_analysis.json", 'w') as f:
    json.dump(results, f, indent=2)
EOF
}

# ============ GENERATE VERDICT ============
generate_verdict() {
    log_section "GENERATING FINAL VERDICT"

	    python3 <<- EOF
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
    analysis = {'summary': {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1_score': 0,
                           'error_detection_rate': 0, 'confidence_score': 0, 'confidence_level': 'UNKNOWN'},
                'calibration': {}}
else:
    try:
        analysis = json.loads(analysis_file.read_text())
    except:
        analysis = {'summary': {}, 'calibration': {}}

summary = analysis.get('summary', {})

accuracy = summary.get('accuracy', 0)
precision = summary.get('precision', 0)
recall = summary.get('recall', 0)
f1 = summary.get('f1_score', 0)
err_rate = summary.get('error_detection_rate', 0)
confidence_score = summary.get('confidence_score', 0)
confidence_level = summary.get('confidence_level', 'UNKNOWN')
total_tests = summary.get('total_tests', 0)
	models_tested = summary.get('models_tested', 0)
	triage_counts = summary.get('triage_counts', {}) or {}
	triage_pass = triage_counts.get('PASS', 0)
	triage_review = triage_counts.get('REVIEW', 0)
	triage_fail = triage_counts.get('FAIL', 0)
degraded = summary.get('degraded_edits', 0) or 0
degraded_runs = summary.get('degraded_runs', []) or []
models = analysis.get('models', {}) or {}

determinism_path = analysis_dir / "determinism_repeats.json"
determinism_repeats = None
if determinism_path.exists():
    try:
        determinism_repeats = json.loads(determinism_path.read_text())
    except Exception:
        determinism_repeats = None

gpu_count = (os.environ.get("PACK_GPU_COUNT") or os.environ.get("NUM_GPUS") or "").strip() or "unknown"
gpu_mem = (os.environ.get("PACK_GPU_MEM_GB") or os.environ.get("GPU_MEMORY_GB") or "").strip()
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

metric_definitions = (
    "METRIC DEFINITIONS:\n"
    "  * Regression: any lm-eval benchmark drop below -2xSE vs baseline.\n"
    "  * Flagged: InvarLock guard failure or primary metric degradation.\n"
    "  * Accuracy/Precision/Recall treat regression as positive class.\n"
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
            worst_blob = f"{worst_task}:{fmt_delta(worst_delta)}" if worst_task else "n/a"
            regression_blob = "none" if not regression_tasks else ", ".join(regression_tasks)
            delta_lines.append(
                f"    {edit_name}: mean {fmt_delta(mean_delta)} | worst {worst_blob} | regressions {regression_blob} | [{task_blob}]"
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

report = f'''
================================================================================
     INVARLOCK PHASE 0 VALIDATION - {platform_header}
================================================================================
     Models Tested:     {models_tested}
     Total Tests:       {total_tests}
     Edit Types:        4 x 2 versions = 8 per model
--------------------------------------------------------------------------------
     Accuracy:          {accuracy:.0%}
     Precision:         {precision:.0%}
     Recall:            {recall:.0%}
     F1 Score:          {f1:.0%}
     Error Detection:   {err_rate:.0%}
	--------------------------------------------------------------------------------
	     CONFIDENCE SCORE:  {confidence_score:.1f}/100 ({confidence_level})
	     TRIAGE (edits):    PASS={triage_pass} REVIEW={triage_review} FAIL={triage_fail}
	     DEGRADED CERTS:    {degraded}
	--------------------------------------------------------------------------------
	     VERDICT: {verdict}
	     VERDICT CONFIDENCE: {verdict_confidence}
	================================================================================

{metric_definitions}{delta_section}EDIT TYPES TESTED:
  * Quantization RTN (group-wise): 8-bit (clean), 4-bit (stress)
  * FP8 Quantization (E4M3 clean, E5M2 stress)
  * Magnitude Pruning: 10% (clean), 50% (stress)
  * Low-Rank SVD: rank-256 (clean), rank-32 (stress)

PLATFORM: {platform_line}

'''

if verdict == "PHASE0_VALIDATED":
    report += "RESULT: InvarLock Phase 0 VALIDATED on proof pack hardware.\n"
elif verdict == "PHASE0_DEGRADED":
    report += f"RESULT: Phase 0 degraded. {degraded} certificate(s) reported degraded primary metrics. See runs: {', '.join(degraded_runs) if degraded_runs else 'n/a'}\n"
else:
    report += f"RESULT: Phase 0 validation failed. Accuracy: {accuracy:.0%}, Error Detection: {err_rate:.0%}\n"

print(report)

with open(reports_dir / "final_verdict.txt", 'w') as f:
    f.write(report)

	with open(reports_dir / "final_verdict.json", 'w') as f:
    json.dump({
        'verdict': verdict,
        'verdict_confidence': verdict_confidence,
        'metrics': {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1_score': f1, 'error_detection_rate': err_rate},
        'confidence': {'score': confidence_score, 'level': confidence_level},
        'triage': {'pass': triage_pass, 'review': triage_review, 'fail': triage_fail},
        'degraded': {'count': degraded, 'runs': degraded_runs},
        'phase0_pass': phase0_pass,
        'platform': platform_tag,
        'platform_name': gpu_name,
        'suite': os.environ.get('PACK_SUITE'),
        'network_mode': 'online' if os.environ.get('PACK_NET') == '1' else 'offline',
        'determinism_mode': os.environ.get('PACK_DETERMINISM'),
        'determinism_repeats': determinism_repeats,
        'models_tested': models_tested,
        'total_tests': total_tests,
        'lm_eval_deltas': delta_summary,
        'timestamp': datetime.now().isoformat()
    }, f, indent=2)


EOF
}
