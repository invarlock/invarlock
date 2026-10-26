//! Bounded JSON diagnostic protocol used by cross-runtime differential tests.
use invarlock_native_contracts::{canonical_json_no_newline, parse_json};
use invarlock_native_core::{
    compare_runs, paired_exact_match_statistics, score, AnalysisError, DEFAULT_MAX_BOOTSTRAP_DRAWS,
};
use serde_json::{json, Value};
use std::io::{self, BufRead, Write};
fn evaluate(v: &Value) -> Result<Value, AnalysisError> {
    match v["operation"].as_str().unwrap_or("compare") {
        "collection_prefix" => invarlock_native_core::validate_retained_collection_prefix(
            v["sources"].as_array().unwrap(),
            &v["plan"],
            &v["baseline_run"],
            &v["subject_run"],
        )
        .map(|_| json!({"valid":true})),
        "inspect_budgets" => {
            invarlock_native_core::validate_inspect_collection_budgets(&v["collection"], &v["plan"])
        }
        "compatible_base_url" => {
            invarlock_native_core::normalized_compatible_base_url(&v["base_url"]).map(|s| json!(s))
        }
        "compatible_collection" => invarlock_native_core::validate_openai_compatible_collection(
            &v["collection"],
            &v["plan"],
        ),
        "import_inspect" => invarlock_native_core::import_inspect_export(
            &v["export"],
            &v["plan"],
            &v["options"],
            &v["baseline_run"],
            &v["subject_run"],
        ),
        "import_inspect_sample" => invarlock_native_core::import_inspect_sample(
            &v["sample"],
            &v["plan"],
            &v["options"],
            &v["baseline_run"],
            &v["subject_run"],
        ),
        "validate_measurements" => invarlock_native_core::validate_measurements(
            &v["measurements"],
            &v["plan"],
            &v["baseline_run"],
            &v["subject_run"],
        )
        .map(|()| json!({"valid":true})),
        "judge_precision" => invarlock_native_core::judge_plan_precision(&v["plan"], &v["policy"]),
        "judge_analysis" => invarlock_native_core::analyze_measurements(
            &v["plan"],
            &v["measurements"],
            &v["policy"],
            &v["baseline_run"],
            &v["subject_run"],
        ),
        "judge" => invarlock_native_core::judge_statistics(v),
        "report" => invarlock_native_core::build_comparison_report(
            v["comparison_id"].as_str().unwrap_or("comparison"),
            &v["paired_records"],
            &v["policy"],
            v["policy_digest"].as_str().unwrap_or(""),
            v["report_format"]
                .as_str()
                .unwrap_or("invarlock/comparison-report-v3"),
        ),
        "score" => score(
            v["kind"].as_str().unwrap_or(""),
            &v["expected"],
            &v["output"],
            v.get("configuration").unwrap_or(&json!({})),
        )
        .map(|v| json!(v)),
        "exact_match" => paired_exact_match_statistics(
            &v["baseline"],
            &v["subject"],
            v["method"]
                .as_str()
                .unwrap_or("newcombe_hybrid_score_paired_v2"),
        ),
        _ => {
            let max = match v.get("max_bootstrap_draws") {
                None => Some(DEFAULT_MAX_BOOTSTRAP_DRAWS),
                Some(Value::Null) => None,
                Some(v) => Some(v.as_u64().ok_or_else(|| {
                    AnalysisError::new("max_bootstrap_draws must be a non-negative integer")
                })?),
            };
            compare_runs(&v["baseline"], &v["subject"], &v["policy"], max)
        }
    }
}
fn main() {
    let input = io::stdin();
    let mut output = io::BufWriter::new(io::stdout().lock());
    for line in input.lock().lines() {
        let result = match line {
            Ok(line) => match parse_json(line.as_bytes(), "analysis request") {
                Ok(v) => match evaluate(&v) {
                    Ok(value) => json!({"ok":value}),
                    Err(e) => json!({"error":e.message,"reason":e.reason}),
                },
                Err(e) => json!({"error":e.message,"reason":e.code}),
            },
            Err(e) => json!({"error":e.to_string()}),
        };
        let bytes = canonical_json_no_newline(&result).unwrap();
        output.write_all(&bytes).unwrap();
        output.write_all(b"\n").unwrap();
        output.flush().unwrap();
    }
}
