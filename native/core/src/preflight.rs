//! Structural captured capacity checks without scoring or bootstrap execution.
use crate::{
    analysis::{missing, py_equal},
    check_policy, AnalysisError,
};
use invarlock_native_contracts::{canonical_json, canonical_json_no_newline};
use serde_json::{json, Value};
use std::collections::BTreeMap;
fn error(s: impl Into<String>) -> AnalysisError {
    AnalysisError::new(s)
}
pub fn captured_preflight_capacity(
    baseline: &Value,
    subject: &Value,
    policy: &Value,
    max_bootstrap_draws: Option<u64>,
) -> Result<Value, AnalysisError> {
    check_policy(policy)?;
    let metrics = policy["metrics"].as_array().unwrap();
    for metric in metrics {
        if metric["kind"] == "recorded" {
            for run in [baseline, subject] {
                if run["score_provenance"]
                    .get(metric["score_key"].as_str().unwrap())
                    .unwrap_or(&Value::Null)
                    != &metric["accepted_provenance"]
                {
                    return Err(error(format!(
                        "metric {}: scorer provenance differs from approved policy",
                        metric["name"].as_str().unwrap()
                    )));
                }
            }
        }
    }
    fn rows(run: &Value) -> Result<BTreeMap<&str, &Value>, AnalysisError> {
        Ok(run["records"]
            .as_array()
            .ok_or_else(|| error("run records must be an array"))?
            .iter()
            .map(|r| (r["id"].as_str().unwrap_or(""), r))
            .collect())
    }
    let left = rows(baseline)?;
    let right = rows(subject)?;
    if left.keys().ne(right.keys()) {
        return Err(error(
            "baseline/subject record IDs differ; export the complete paired schedule",
        ));
    }
    let mut pairs = Vec::new();
    for (id, left) in left {
        let right = right[id];
        for key in ["input", "expected", "metadata"] {
            if canonical_json(&left[key]).map_err(|e| error(e.to_string()))?
                != canonical_json(&right[key]).map_err(|e| error(e.to_string()))?
            {
                return Err(error(format!("record {id}: {key} changed between runs")));
            }
        }
        pairs.push((left, right));
    }
    let mut scopes = vec![pairs.clone()];
    for slice in policy["slices"].as_array().unwrap() {
        scopes.push(
            pairs
                .iter()
                .copied()
                .filter(|(left, _)| {
                    slice["where"]
                        .as_object()
                        .unwrap()
                        .iter()
                        .all(|(key, value)| {
                            py_equal(left["metadata"].get(key).unwrap_or(&Value::Null), value)
                        })
                })
                .collect(),
        );
    }
    let scope_count = scopes.iter().map(Vec::len).sum::<usize>();
    let required = scope_count as u64
        * metrics
            .iter()
            .filter(|m| {
                !matches!(
                    m["kind"].as_str(),
                    Some("exact_match" | "normalized_match" | "numeric_tolerance" | "json_exact")
                )
            })
            .count() as u64
        * 2048;
    if max_bootstrap_draws.is_some_and(|max| required > max) {
        return Err(AnalysisError{reason:"local_work_budget_exceeded",message:format!("comparison plans {required} bootstrap draws; local budget is {}. Increase the local budget only on a suitable host; the complete policy has not been evaluated.",max_bootstrap_draws.unwrap())});
    }
    let mut total = 0;
    let mut lengths = BTreeMap::new();
    for selected in &scopes {
        for metric in metrics {
            total += 2;
            let mut first = true;
            for &(left, right) in selected {
                if missing(left, right, metric) {
                    let id = left["id"].as_str().unwrap();
                    let length = match lengths.get(id) {
                        Some(n) => *n,
                        None => {
                            let n = canonical_json_no_newline(&left["id"])
                                .map_err(|e| error(e.to_string()))?
                                .len();
                            lengths.insert(id, n);
                            n
                        }
                    };
                    total += length + usize::from(!first);
                    first = false;
                }
                if total > 128 * 1024 * 1024 {
                    return Err(error(format!("comparison missing-ID arrays already require at least {total} bytes; the byte limit is 134217728. The complete policy has not been evaluated.")));
                }
            }
        }
    }
    Ok(
        json!({"required_bootstrap_draws":required,"missing_id_bytes":total,"scope_count":scopes.len(),"record_count":pairs.len()}),
    )
}
